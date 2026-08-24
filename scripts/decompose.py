#!/usr/bin/env python3
"""Split an instruction into ordered constraints by asking a model, not a regex.

`instruction_plan.parse_instruction` does the same job deterministically, and
its argument for doing so is sound: every training question is available
offline, so a regex can be measured on all thirty for free. The problem is what
it was measured *against*. Its clause openers -- `go to`, `take the path`,
`pass by`, `avoid`, and six more -- are a closed set read off those same thirty
sentences, while the three evaluation scenes are held out and the README
promises only that their questions are "of similar style". A sentence opening
"head toward" or "make your way past" falls through to the single-`GOTO`
fallback.

That fallback used to be free. When the loop drove to one object and stopped,
collapsing a sequence to its first destination lost nothing that was going to
be attempted anyway. With an executor that walks the clauses in order it is no
longer free: README §175 scores instruction-following on whether the trajectory
"follows the path constraints in the command and in the correct order", so a
question that degrades to one `GOTO` forfeits every ordering point it carried.

So this module asks a model to do the split, and returns the same
:class:`~instruction_plan.Clause` objects, which makes it a drop-in for
`parse_instruction`. It keeps that function as the fallback for an unparseable
reply, and `--diff` measures one against the other over the official thirty --
the same free offline measurement, now applied to the replacement.

    export ANTHROPIC_API_KEY=...
    uv run --with anthropic python scripts/decompose.py --diff
    uv run --with anthropic python scripts/decompose.py "go to the chair ..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instruction_plan import (  # noqa: E402
    AVOID,
    GOTO,
    PASS,
    Clause,
    _questions,
    keepouts,
    parse_instruction,
    steps,
)
from vlm_probe import ask_claude, parse  # noqa: E402

# Relative to the working directory, which is fine for a sweep run from the
# repo root and is not fine inside the ai_module container, where `ros2 launch`
# picks the working directory and it need not be writable. See `_save`.
CACHE = Path(os.environ.get("XIAO_HEI_DECOMPOSE_CACHE",
                            "artifacts/decompose_cache.json"))
MODEL = "claude-opus-5"
# Bump whenever PROMPT changes, so a cached reply is never reused across a
# change in what was asked.
PROMPT_VER = "v2-verb-decides"

KINDS = {GOTO, PASS, AVOID}

# The two examples are the README's own illustrations of the task, deliberately
# *not* drawn from questions/questions.json: `--diff` scores this prompt on
# those thirty, and seeding it with two of them would score the prompt on its
# own answer key.
PROMPT = """\
A mobile robot must follow a natural-language navigation instruction. Split the
instruction into the ordered list of constraints the robot has to satisfy.

The robot is scored on the path it actually drives: it must satisfy the
constraints **in the order the instruction gives them**, and it loses points
for reaching them out of order, for missing one, or for driving through a
region the instruction forbids. So the order of your list is the order the
robot will drive, and every constraint the sentence imposes must appear exactly
once.

Three kinds of constraint:

  GOTO   drive to an object and stop near it. This is a destination.
  PASS   drive through a place the instruction names by nearby landmarks --
         "take the path between X and Y", "pass by Z". Not a destination; the
         robot goes through it on the way to the next one.
  AVOID  a place the robot must NOT drive through, named the same way.

Fields per constraint:

  kind      "GOTO", "PASS" or "AVOID"
  text      the phrase in the instruction's OWN words. Do not paraphrase it,
            do not translate it into some other vocabulary, do not drop
            adjectives. A later stage looks the object up from this text.
  relation  PASS and AVOID only: "between" if the robot passes between two
            landmarks, "near" if it passes close by one. null for GOTO.
  anchors   PASS and AVOID only: the landmarks, each in the instruction's own
            words, as a list. Empty for GOTO.

Rules that decide the hard cases:

* **The verb decides the kind, not the preposition.** "near" appears in both
  kinds and does not by itself mean PASS. Going somewhere is a destination even
  when the sentence says "near": "go near the stool", "stop by the sink",
  "stop at the table" are all GOTO. It is PASS only when the sentence says the
  robot travels through or along: "take the path near the stool", "pass by the
  stool", "go between the columns".
* A comparison that identifies WHICH object a destination is stays inside
  `text` and is not an anchor. "go to the picture closest to the TV" is one
  GOTO whose text is "the picture closest to the TV" -- the TV is how the
  picture is picked out, not a place to drive through.
* When the instruction names a pair with a single noun -- "between the two
  columns", "between the two tables" -- emit ONE anchor holding that phrase.
  Do not invent a second landmark name the sentence never used.
* A destination at the end often has no verb of its own ("...and then to the
  flowers", "...to the fridge"). It is still a GOTO.
* A PASS or AVOID is never the last thing the robot does if the sentence names
  somewhere to end up.

Two examples.

Instruction: Take the path near the window to the fridge.
{"clauses": [
  {"kind": "PASS", "text": "near the window", "relation": "near",
   "anchors": ["the window"]},
  {"kind": "GOTO", "text": "the fridge", "relation": null, "anchors": []}
]}

Instruction: Avoid the path between the two tables and go near the blue trash \
can near the window.
{"clauses": [
  {"kind": "AVOID", "text": "between the two tables", "relation": "between",
   "anchors": ["the two tables"]},
  {"kind": "GOTO", "text": "the blue trash can near the window",
   "relation": null, "anchors": []}
]}

Now split this instruction. Reply with the JSON object only, no prose.

Instruction: {sentence}
"""


def _clauses_from(reply: dict) -> list[Clause] | None:
    """The model's JSON as `Clause` objects, or None if it is not usable."""
    raw = reply.get("clauses")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[Clause] = []
    for c in raw:
        if not isinstance(c, dict):
            return None
        kind = str(c.get("kind", "")).upper()
        text = str(c.get("text") or "").strip()
        if kind not in KINDS or not text:
            return None
        rel = c.get("relation")
        rel = str(rel).lower() if rel else None
        anchors = c.get("anchors") or []
        if not isinstance(anchors, list):
            return None
        out.append(Clause(kind, text, rel,
                          tuple(str(a).strip() for a in anchors if str(a).strip())))
    return out


def _save(cache: dict) -> None:
    """Persist the cache, and never let failing to do so end a run.

    This used to write unguarded, immediately after the API call that produced
    the entry. The cache is a development convenience — it saves re-asking for
    a sentence already decomposed — but the write sits on the only code path
    that answers a question, and it targets a relative directory. Inside the
    ai_module container the working directory is chosen by `ros2 launch` and
    need not be writable, so an unwritable `artifacts/` would have thrown at
    step 0, after the call was paid for and before a single waypoint was
    published. On the test scenes the cache can only ever miss anyway.
    """
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    except OSError as e:
        print(f"decompose: not caching to {CACHE} ({e})", file=sys.stderr)


def decompose(sentence: str, *, model: str = MODEL, cache: dict | None = None,
              ask=ask_claude) -> tuple[list[Clause], bool]:
    """`(clauses, from_model)`.

    Falls back to `parse_instruction` when the reply does not parse, so a bad
    call is never worse than not having made it. `from_model` says which path
    produced the result, because a silent fallback in the middle of a run is
    exactly the thing that would make a later measurement meaningless.
    """
    s = " ".join((sentence or "").split())
    if not s:
        return [], False
    key = f"{PROMPT_VER}|{model}|{s}"
    if cache is None:
        cache = {}
    if key not in cache:
        cache[key] = ask(PROMPT.replace("{sentence}", s), [], model)
        _save(cache)
    got = _clauses_from(parse(cache[key]) or {})
    return (got, True) if got else (parse_instruction(s), False)


def _norm(c: Clause) -> tuple:
    """Compare on meaning, not on articles and trailing punctuation."""
    def t(x: str) -> str:
        x = " ".join(x.lower().replace(",", " ").split())
        return x[4:] if x.startswith("the ") else x
    return (c.kind, t(c.text), c.relation or "",
            tuple(sorted(t(a) for a in c.anchors)))


def _diff(model: str, limit: int | None) -> int:
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    rows = _questions()[:limit]
    same_shape = same_full = fell_back = 0
    disagree: list[tuple] = []
    t0 = time.time()

    for scene, s in rows:
        got, from_model = decompose(s, model=model, cache=cache)
        want = parse_instruction(s)
        fell_back += not from_model
        # Compare what the executor consumes: the drive order of destinations
        # and passages, and the keep-outs as a set. Where a keep-out sits in
        # the clause list carries no meaning -- see `instruction_plan.keepouts`.
        shape_ok = ([c.kind for c in steps(got)] == [c.kind for c in steps(want)]
                    and len(keepouts(got)) == len(keepouts(want)))
        full_ok = ([_norm(c) for c in steps(got)] == [_norm(c) for c in steps(want)]
                   and sorted(map(_norm, keepouts(got)))
                   == sorted(map(_norm, keepouts(want))))
        same_shape += shape_ok
        same_full += full_ok
        mark = "==" if full_ok else ("~~" if shape_ok else "!!")
        print(f"{mark} {scene}: {s}")
        if not full_ok:
            disagree.append((scene, s, want, got))
            for c in want:
                print(f"      regex  {c}")
            for c in got:
                print(f"      model  {c}")
        print(flush=True)

    n = len(rows)
    dt = time.time() - t0
    print(f"\n== {n} questions, {model}, {dt:.0f}s ==")
    print(f"  same drive order    {same_shape}/{n}   "
          f"(this is what fixes the execution order)")
    print(f"  identical clauses    {same_full}/{n}")
    print(f"  model reply unusable {fell_back}/{n}   (fell back to the regex)")
    if disagree:
        print(f"\n  {len(disagree)} to adjudicate by hand — printed above with "
              f"`~~` (order agrees) or `!!` (order differs)")
    out = Path("artifacts/decompose_diff.json")
    out.write_text(json.dumps(
        [{"scene": sc, "q": s,
          "regex": [str(c) for c in w], "model": [str(c) for c in g]}
         for sc, s, w, g in disagree], indent=1))
    print(f"  wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="?")
    ap.add_argument("--diff", action="store_true",
                    help="decompose the official 30 and compare to the regex")
    ap.add_argument("--json", action="store_true",
                    help="print the clauses as the executor will consume them")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.diff:
        return _diff(args.model, args.limit)
    if not args.sentence:
        ap.error("give a sentence, or --diff")
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    got, from_model = decompose(args.sentence, model=args.model, cache=cache)
    if args.json:
        print(json.dumps({
            "from_model": from_model,
            "steps": [asdict(c) for c in steps(got)],
            "keepouts": [asdict(c) for c in keepouts(got)],
        }, indent=2))
        return 0
    print(f"  ({'model' if from_model else 'regex fallback'})")
    for i, c in enumerate(steps(got), 1):
        print(f"  step {i}  {c}")
    for c in keepouts(got):
        print(f"  always  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
