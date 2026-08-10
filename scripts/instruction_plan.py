#!/usr/bin/env python3
"""Split an instruction into the ordered constraints it actually asks for.

Instruction-following is 12 of the 17 points a scene is worth, and README §172
scores it on "the actual trajectory followed by the robot ... whether it follows
the path constraints in the command and in the correct order", with penalties
for constraints achieved out of order or not at all.

Measured over the 30 provided instruction questions, one of them is a single
destination. The other 29 are sequences:

    21 / 30   need two or more destinations, in order
    14 / 30   contain a passage to drive through
     3 / 30   contain a region to stay out of

`approach_loop` drives to one object and stops, so it can satisfy at most the
first clause of almost every question. This module is the missing structure:
the sentence becomes an ordered list of clauses, and the loop executes them.

Deterministic rather than model-driven, because every question is available
offline: this parser can be measured on all 30 for free and before any driving,
where a decomposition call could only be measured by spending it, and a failure
at step 0 would cost the whole run. Where it cannot parse, it falls back to one
`GOTO` holding the whole sentence -- exactly today's behaviour, so the fallback
is never worse than not having tried.

    uv run python scripts/instruction_plan.py            # parse all 30, annotated
    uv run python scripts/instruction_plan.py "go to X, then take the path ..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"

GOTO, PASS, AVOID = "GOTO", "PASS", "AVOID"

# Clause openers, in priority order within one alternation so that `go between`
# beats `go` and `avoid the path` beats a bare `avoid`. The bare form catches
# "...and then to the flowers", "...and finally, to the door" -- a destination
# with no verb of its own, which two of the thirty questions use.
_START = re.compile(
    r"\b(?P<avoid>avoiding\s+the\s+path|avoid\s+the\s+path|avoid)\b"
    r"|\b(?P<pas>take\s+the\s+path|go\s+between|pass\s+by|pass\s+through)\b"
    r"|\b(?P<goto>go\s+to|go\s+near|stop\s+at|stop\s+by|go)\b"
    r"|\b(?P<bare>(?:then|finally)\s*,?\s+to)\s",
    re.I)

# Leading junk on a clause body: connectives the split leaves behind.
_LEAD = re.compile(r"^(?:\s|,|;|\.|and\b|then\b|finally\b|first\b)+", re.I)
_TRAIL = re.compile(r"(?:\s|,|;|\.|\band\b|\bthen\b|\bfinally\b)+$", re.I)

# " to " that starts a destination rather than sitting inside a relation:
# "take the path between the TV and the bed >to< the picture closest to the TV".
# The exclusions are the relations that legitimately contain "to"; without them
# this would cut "the table closest to the sofa" in half.
_VIA_TO = re.compile(r"\s+to\s+(?!the\s+(?:left|right)\b)", re.I)
_TO_GUARD = re.compile(r"\b(?:next|close|closest|near|nearest|adjacent|due)\s*$",
                       re.I)


@dataclass(frozen=True)
class Clause:
    """One ordered constraint.

    `text` is the phrase as the model will be asked it, so it stays in the
    instruction's own words. `anchors` is filled for PASS and AVOID, where the
    geometry needs the landmarks separately.
    """

    kind: str
    text: str
    relation: str | None = None          # PASS/AVOID: "between" or "near"
    anchors: tuple[str, ...] = field(default=())

    def __str__(self) -> str:
        if self.kind == GOTO:
            return f"{self.kind}  {self.text}"
        via = f"{self.relation} " if self.relation else ""
        return f"{self.kind}  {via}{' + '.join(self.anchors) or self.text}"


def _tidy(s: str) -> str:
    return _TRAIL.sub("", _LEAD.sub("", s)).strip()


def _split_anchors(body: str,
                   default_rel: str | None = None) -> tuple[str | None, tuple[str, ...]]:
    """`between the sofa and the coffee table` -> ("between", (sofa, table)).

    "the two columns" and "the two tables" name a pair with one noun; they come
    back as a single anchor and the caller has to resolve two instances of it.
    Flattening that here would invent a second landmark that the sentence never
    named.
    """
    b = body.strip()
    m = re.match(r"^(between|near|by|through)\b\s*", b, re.I)
    # `go between the bench and the bed` puts "between" in the opener, so the
    # body starts at the first anchor and the relation has to come from there.
    rel = m.group(1).lower() if m else default_rel
    if rel in ("by", "through"):
        rel = "near"
    if m:
        b = b[m.end():]
    b = _tidy(b)
    if rel == "between":
        parts = [p for p in (x.strip() for x in re.split(r"\s+and\s+", b)) if p]
        return rel, tuple(parts)
    return rel, ((b,) if b else ())


def parse_instruction(sentence: str) -> list[Clause]:
    """The sentence as an ordered list of clauses.

    Never returns empty: an unparseable sentence becomes one `GOTO` holding all
    of it, which is what the loop did before this existed.
    """
    s = " ".join((sentence or "").split())
    if not s:
        return []

    marks = list(_START.finditer(s))
    if not marks:
        return [Clause(GOTO, _tidy(s))]

    out: list[Clause] = []
    for i, m in enumerate(marks):
        body = s[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(s)]
        if m.lastgroup == "avoid":
            kind = AVOID
        elif m.lastgroup == "pas":
            kind = PASS
        else:
            kind = GOTO
            # `go between the bench and the bed` opens with `go`, and only the
            # body says it is a passage. Without this the clause reads as a
            # destination called "between the bench and the bed".
            if re.match(r"^\s*between\b", body, re.I):
                kind = PASS

        if kind == GOTO:
            body = _tidy(body)
            if body:
                out.append(Clause(GOTO, body))
            continue

        # A passage clause can carry the next destination with no verb of its
        # own: "take the path near the wardrobe doors to the flowers ...".
        dest = None
        for t in _VIA_TO.finditer(body):
            if _TO_GUARD.search(body[:t.start()]):
                continue                      # "closest to", not a destination
            body, dest = body[:t.start()], _tidy(body[t.end():])
            break

        opener = m.group(0).lower()
        rel, anchors = _split_anchors(
            body, "between" if "between" in opener else None)
        out.append(Clause(kind, _tidy(body), rel, anchors))
        if dest:
            out.append(Clause(GOTO, dest))

    return out or [Clause(GOTO, _tidy(s))]


def destinations(plan: list[Clause]) -> list[Clause]:
    return [c for c in plan if c.kind == GOTO]


def steps(plan: list[Clause]) -> list[Clause]:
    """The clauses the robot executes in order: destinations and passages.

    A keep-out is not one of them -- see `keepouts`.
    """
    return [c for c in plan if c.kind != AVOID]


def keepouts(plan: list[Clause]) -> list[Clause]:
    """The keep-outs, as an unordered set of constraints on the whole run.

    `parse_instruction` emits clauses in the order the words appear, and for
    destinations and passages that is also the order to drive them. A keep-out
    is different: "go to A, then stop at B, avoiding the path between X and Y"
    writes the forbidden region last, but nothing in the sentence says it
    switches on only after A. Nor is it first: "go to the cup and avoid the
    path near the cabinet" has one destination, so there is no "before" to put
    it in.

    Trying to place it in the sequence at all is the mistake -- it was where a
    regex fix and the model's own ordering disagreed, each right on one phrasing
    and wrong on the other. README §175 penalises a trajectory that "passes
    through areas it is forbidden to go through", with no mention of when, so
    the executor holds these active for the entire run and the ordering
    question does not arise.
    """
    return [c for c in plan if c.kind == AVOID]


def _questions() -> list[tuple[str, str]]:
    p = CHALLENGE / "questions/questions.json"
    qs = json.loads(p.read_text())
    qs = qs if isinstance(qs, list) else qs["questions"]
    return [(e["scene"], s) for e in qs
            for s in e["questions"]["instruction_following"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="?",
                    help="parse one sentence; omit to parse the official 30")
    args = ap.parse_args()

    if args.sentence:
        for c in parse_instruction(args.sentence):
            print(f"  {c}")
        return 0

    rows = _questions()
    if not rows:
        print("no questions found", file=sys.stderr)
        return 1
    n_goto = n_pass = n_avoid = 0
    for scene, s in rows:
        plan = parse_instruction(s)
        n_goto += sum(1 for c in plan if c.kind == GOTO)
        n_pass += sum(1 for c in plan if c.kind == PASS)
        n_avoid += sum(1 for c in plan if c.kind == AVOID)
        print(f"\n{scene}: {s}")
        for c in plan:
            print(f"    {c}")

    print(f"\n\n{len(rows)} questions -> {n_goto} destinations, "
          f"{n_pass} passages, {n_avoid} keep-outs")
    multi = sum(1 for _, s in rows if len(destinations(parse_instruction(s))) >= 2)
    print(f"{multi} of {len(rows)} need two or more destinations in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
