#!/usr/bin/env python3
"""Split a numerical question into the parts the counter needs.

"How many red pillows are on the sofa?" has to become: count `pillow`, keep
only the red ones, keep only those on a `sofa`, and there is one sofa the
sentence means. Four decisions, and the regex that can make all four on the
fifteen released questions is not the regex that will survive the three held-out
ones -- the README promises only "similar style".

So a model does the split, exactly as `decompose.py` does it for instructions,
and the regex stays as the fallback for an unparseable reply. `--diff` measures
one against the other over the released fifteen, which is free.

The one field with no obvious answer is `count`. "Count the number of chairs
with pillows on them" counts *chairs*, not pillows, and it is the only released
question shaped that way -- but that shape is common English and a held-out
scene may well use it, so it is asked for rather than assumed.

    uv run --with anthropic python scripts/numerical_plan.py --diff
    uv run --with anthropic python scripts/numerical_plan.py "How many cups ..."
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_probe import ask_claude, parse  # noqa: E402

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
MODEL = os.environ.get("XIAO_HEI_MODEL", "claude-opus-5")

RELATIONS = ("on", "above", "below", "near", "in", "between", "with", "none")
PROMPT = """Split one counting question from a robot navigation benchmark into
the parts a counter needs. Reply with JSON only, no prose, no markdown fence.

Question: "{q}"

{{
  "target": "the head noun being counted, singular, no article — e.g. \\"pillow\\"",
  "attribute": "a colour or material the question restricts the target to, or null.
      Only what the sentence actually says. \\"red\\" for \\"red pillows\\"; null for
      \\"pillows\\".",
  "relation": one of {rels},
      how the target must sit relative to the anchor. \\"none\\" if the question
      names no anchor at all.
  "anchor": "the head noun the relation points at, singular, no article, or null",
  "anchor_determiner": "the" | "a" | "each" | null,
      the article in front of the anchor, verbatim. This decides whether the
      sentence means one particular anchor or any of them, and we handle those
      differently, so report what is written rather than what seems sensible.
  "anchor_qualifier": "the words that pick out WHICH anchor, or null.
      \\"under the pictures\\" in \\"the sofa under the pictures\\";
      \\"closest to the map wall decal\\" in \\"the table closest to ...\\".",
  "count": "targets" | "anchors",
      which noun the integer is about. Almost always \\"targets\\" — the thing
      right after \\"how many\\". It is \\"anchors\\" when the sentence counts the
      containers instead: \\"count the number of chairs with pillows on them\\"
      wants a number of chairs, not of pillows.
  "scope": "anchor" | "scene",
      \\"anchor\\" when the count is of things gathered on one piece of furniture,
      so a robot could answer it standing in front of that furniture.
      \\"scene\\" when the targets are spread through the room and it has to look
      around — \\"how many sofas are below a window\\".
  "restated": "the question as an instruction to someone in the room, one short
      sentence, so a reader can check the split without re-reading the original"
}}"""


def _fallback(q: str) -> dict:
    """A regex split, for when the model's reply cannot be used.

    Deliberately shallow. It exists so that an unparseable reply costs accuracy
    and not the question, and every field it cannot find it leaves null rather
    than guessing -- a wrong anchor sends the robot to the wrong furniture,
    which is worse than no anchor at all.
    """
    s = " ".join(q.strip().split())
    low = s.lower().rstrip("?.")
    m = re.search(r"(?:how many|count the number of)\s+(.*)", low)
    body = m.group(1) if m else low
    rel, anchor, det, qual = "none", None, None, None
    # Earliest in the sentence, not first in this list. "pillows are on the
    # sofa under the pictures" splits at `on`; scanning the list in order split
    # it at `under` and made the target "pillows are on the sofa".
    words = ("on top of", "above", "below", "underneath", "under", "next to",
             "near", "beside", "between", "with", "in", "on")
    hits = [(m.start(), w, m) for w in words
            for m in [re.search(rf"\b{re.escape(w)}\b", body)] if m]
    for _, word, m in sorted(hits, key=lambda h: (h[0], -len(h[1]))):
        if True:
            rel = {"underneath": "below", "under": "below", "on top of": "on",
                   "next to": "near", "beside": "near"}.get(word, word)
            head, tail = body[:m.start()].strip(), body[m.end():].strip()
            body = head
            m2 = re.match(r"(the|a|an|each)\s+(.*)", tail)
            if m2:
                det = "a" if m2.group(1) in ("a", "an") else m2.group(1)
                tail = m2.group(2)
            # The anchor is the first noun phrase; whatever follows a second
            # preposition or a comparative is the qualifier.
            m3 = re.search(r"\s+(under|above|below|with|closest|farthest|"
                           r"nearest|next|near|that|which)\b", tail)
            if m3:
                anchor, qual = tail[:m3.start()].strip(), tail[m3.start():].strip()
            else:
                anchor = tail.strip()
            break
    body = re.sub(r"\s+(are|is|were|was)$", "", body).strip()
    attr = None
    m = re.match(r"(red|black|blue|green|white|yellow|brown|grey|gray|orange|"
                 r"purple|pink)\s+(.*)", body)
    if m:
        attr, body = m.group(1), m.group(2)
    target = re.sub(r"s$", "", body.strip()) or None
    return {"target": target, "attribute": attr, "relation": rel,
            "anchor": (re.sub(r"s$", "", anchor) if anchor else None),
            "anchor_determiner": det, "anchor_qualifier": qual,
            "count": "targets", "scope": "anchor" if anchor else "scene",
            "restated": None, "from_model": False}


def _clean(p: dict) -> dict:
    """Force the reply into the shape the counter can act on."""
    out = dict(p)
    for k in ("target", "attribute", "anchor", "anchor_determiner",
              "anchor_qualifier", "restated"):
        v = out.get(k)
        out[k] = v.strip() if isinstance(v, str) and v.strip() else None
    if out.get("relation") not in RELATIONS:
        out["relation"] = "none" if not out.get("anchor") else "on"
    if out.get("count") not in ("targets", "anchors"):
        out["count"] = "targets"
    if out.get("scope") not in ("anchor", "scene"):
        out["scope"] = "anchor" if out.get("anchor") else "scene"
    # An anchor-scoped plan with no anchor cannot be executed as one.
    if out["scope"] == "anchor" and not out.get("anchor"):
        out["scope"] = "scene"
    return out


def plan(question: str, *, model: str = MODEL) -> dict:
    """The split. Never raises; falls back to the regex."""
    try:
        raw = ask_claude(PROMPT.format(q=question, rels=list(RELATIONS)), [], model)
        got = parse(raw)
        if not isinstance(got, dict) or not got.get("target"):
            raise ValueError(f"no target in reply {str(got)[:120]}")
        return _clean({**got, "from_model": True})
    except Exception as e:                       # noqa: BLE001 -- see docstring
        print(f"numerical_plan: falling back to the regex after {e!r}",
              file=sys.stderr)
        return _clean(_fallback(question))


def phrase_for(p: dict) -> str:
    """The referring expression to hand `run_goto`, so it drives to the anchor.

    The qualifier is what makes it a *referring* expression -- "the sofa" in a
    room with four is not one, and the grounding call is built to refuse an
    ambiguous phrase rather than pick. So the qualifier goes in whenever the
    sentence supplied one.
    """
    if not p.get("anchor"):
        return ""
    bits = ["the", p["anchor"]]
    if p.get("anchor_qualifier"):
        bits.append(p["anchor_qualifier"])
    return " ".join(bits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--diff", action="store_true",
                    help="model against regex over the released fifteen")
    ap.add_argument("--regex", action="store_true", help="regex only, no call")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if args.diff:
        qs = [(e["scene"], e["questions"]["numerical"][0])
              for e in json.loads(
                  (CHALLENGE / "questions/questions.json").read_text())]
        agree = 0
        for scene, q in qs:
            a, b = plan(q, model=args.model), _clean(_fallback(q))
            keys = ("target", "attribute", "relation", "anchor", "count", "scope")
            same = all(a.get(k) == b.get(k) for k in keys)
            agree += same
            print(f"{scene:18s} {'same' if same else 'DIFF'}  {q[:56]}")
            if not same:
                for k in keys:
                    if a.get(k) != b.get(k):
                        print(f"{'':20s}{k:18s} model {a.get(k)!r:28s} "
                              f"regex {b.get(k)!r}")
        print(f"\n{agree}/{len(qs)} agree on the fields that steer the counter")
        return 0

    if not args.question:
        ap.error("a question is required unless --diff")
    p = _clean(_fallback(args.question)) if args.regex \
        else plan(args.question, model=args.model)
    print(json.dumps(p, indent=1))
    print(f"\nanchor phrase: {phrase_for(p)!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
