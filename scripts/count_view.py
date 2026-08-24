#!/usr/bin/env python3
"""Count one thing from one vantage point, and say whether the view was enough.

The numerical question is scored 0 or 1 on an exact integer, so the expensive
mistake is not a miss -- it is a confident wrong number. Two rules follow, and
they are the whole design of this module.

**The call is never told the running total.** Not the tally so far, not what a
previous view counted, not how many the last look found. This is not
fastidiousness: TASK 55 measured what happens when a model is told the
hypothesis it is checking. The binding verifier, given the belief in its
prompt, said `holds` on 83% of bindings beyond 5 m against 35% within 5 m --
agreeing *more* the less it could see -- and falsely confirmed a wrong binding
5 times in 10. Blinding the looking call took that to 0 in 10. A counting call
told "you saw four last time" will find four. So `count_view` takes no history
argument, and there is nowhere to put one.

**The model counts pixels; the geometry counts objects.** Each instance comes
back as a box, every box is lifted to a map-frame position by the same lidar
cone the approach loop uses, and the answer is the number of *clusters* across
all the views taken -- never the sum of the per-view counts. Two views of a
sofa with four pillows must not answer eight. This is the same contract the
rest of the system runs on: the model proposes semantics, the lidar decides
metres.

`sufficient` may only withhold a commit. It can never raise the count.

    uv run --with anthropic python scripts/count_view.py snaps/start \\
        "How many pillows are on the bed?" --target pillow --anchor bed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from vlm_approach import _lift_xy  # noqa: E402
from vlm_locate import scan_to_camera  # noqa: E402
from vlm_probe import (FACES, NAMES, ask_claude, parse,  # noqa: E402
                       settle_coord_space, to_pixels)

MODEL = os.environ.get("XIAO_HEI_MODEL", "claude-opus-5")

# Two lifted instances closer than this are the same object seen twice. Sized
# from what is being counted: pillows on a sofa sit 0.35-0.50 m apart centre to
# centre, and a box of theirs subtends a cone about 0.14 m across at 2.5 m, so
# there is room between the two. Tighter than JUMP_M (1.0 m) in `approach_loop`
# for the same reason -- that number separates *furniture*, this one separates
# things that sit on furniture.
MERGE_M = 0.30
# An instance whose box the scanner could not measure has no position, so it
# cannot be de-duplicated. Counted, but only from the view that saw the most --
# see `merge_unlifted`.
MAX_LOOKS = int(os.environ.get("XIAO_HEI_COUNT_LOOKS", "4"))

PROMPT = """You are looking at what a robot sees from one spot in a room.

The four images are perspective views from the same position, each 100° wide:

  image 0 — heading   0°  (front)
  image 1 — heading  90°  (right)
  image 2 — heading 180°  (back)
  image 3 — heading 270°  (left)

Adjacent images overlap by about 10°, so an object near an edge can appear in
two of them. **Report such an object once**, in the image where you can see
more of it. Listing it twice is the single most costly error you can make here.

THE QUESTION
  "{question}"

You are counting: {target}
{anchor_line}
Count ONLY the instances that satisfy the question's relation. An instance of
the right type sitting somewhere else is not an answer, and neither is one you
infer must be there.

WHAT YOU ARE NOT BEING ASKED
  Not the total for the room. Only what you can see FROM HERE, in these four
  images. Something you saw earlier, or expect round the corner, is not in
  this answer. A separate step adds the views up; if you add them up too, the
  answer is counted twice.

  Do not settle on a number you are unsure of. Saying the view is not enough is
  a correct, useful answer and costs the robot one short move — guessing costs
  the whole question, because this is scored on an exact match and there is no
  partial credit.

BOX EVERY INSTANCE YOU COUNT. A laser scanner measures the distance through the
centre of each box, which is how two views get reconciled without asking you.
An instance you count but do not box cannot be placed, so box them all, even
the partly hidden ones — and mark those `certain: false`.

IS THIS VIEW ENOUGH? Answer honestly. It is not enough when:
  - something is in front of what you are counting, and more could be behind it
  - {anchor_word} runs past the edge of the frame, so part of it is unseen
  - you are too far away to tell instances apart, or to tell them from
    something similar
  - the question turns on a colour or a marking you cannot actually resolve

If it is not enough, say where to go. A short move that brings the whole thing
into view, a step around it to see the far side, or simply closer.

Each image is {size} x {size} pixels.

Reply with JSON only, no prose, no markdown fence:

{{
  "coord_space": "pixels" or "normalized_1000",
      which convention the boxes below are in. State it explicitly — with a
      {size}-pixel image both readings produce plausible-looking numbers.
  "instances": [
      {{"image_index": 0-3,
        "box_2d": [ymin, xmin, ymax, xmax],
        "note": "which one this is, in a few words",
        "certain": true or false}}
  ],
      one entry per instance VISIBLE FROM HERE that satisfies the question
  "count_here": <integer>,
      how many you just listed. Must equal the length of "instances".
  "sufficient": true or false,
      could a careful person answer the question from this view alone
  "why_not": "occluded" | "anchor_cut_off" | "too_far"
             | "attribute_unreadable" | null,
      why not, when sufficient is false
  "next_view": {{
      "kind": "closer" | "orbit" | "move",
      "heading_deg": 0-359,
          which way to go, in the same convention as the image headings above
      "box_2d": [ymin, xmin, ymax, xmax] or null,
          if the place to go is something you can see — an opening, a gap, the
          far side of the thing — box it. A box beats a heading: the robot can
          drive to a boxed place and can only drive roughly toward a heading.
      "image_index": 0-3 or null,
      "why": "..."
  }} or null,
      null when sufficient is true
  "confidence": 0.0-1.0,
      how sure you are of "count_here" — of the number, not of the question
  "evidence": "what you actually see, in one sentence"
}}"""


def build_prompt(question: str, target: str, anchor: str | None,
                 size: int = G.FACE_SIZE) -> str:
    """The counting prompt. Deliberately takes no history -- see the module docstring."""
    anchor_line = (f"They must be {_relation_hint(question)} the {anchor}.\n"
                   if anchor else "")
    return PROMPT.format(question=question, target=target,
                         anchor_line=anchor_line,
                         anchor_word=("the " + anchor) if anchor else "what you are counting",
                         size=size)


def _relation_hint(question: str) -> str:
    """The preposition the sentence uses, so the prompt echoes its words.

    Only ever a hint inside a prompt -- nothing downstream branches on it, so
    a miss costs a slightly vaguer sentence and never a wrong answer.
    """
    q = question.lower()
    for word in ("on top of", "above", "below", "under", "beneath", "next to",
                 "near", "beside", "between", "in", "on"):
        if f" {word} " in q:
            return word
    return "related to"


def count_view(faces: list[bytes], question: str, target: str,
               anchor: str | None, *, model: str = MODEL) -> dict:
    """One blind look. Returns the parsed reply, or `{"error": ...}`.

    Never raises: a failed call must cost one look, not the question.
    """
    try:
        raw = ask_claude(build_prompt(question, target, anchor), faces, model)
    except Exception as e:                       # noqa: BLE001 -- see docstring
        return {"error": f"call failed: {e!r}", "instances": [], "count_here": 0,
                "sufficient": False}
    reply = parse(raw)
    if not isinstance(reply, dict):
        return {"error": "unparseable reply", "raw": raw[:400],
                "instances": [], "count_here": 0, "sufficient": False}
    reply.setdefault("instances", [])
    if not isinstance(reply["instances"], list):
        reply["instances"] = []
    # `count_here` is the model's own arithmetic over a list it just wrote, and
    # it is the list we can act on. Where they disagree the list wins, and the
    # disagreement is recorded rather than smoothed over.
    n = len(reply["instances"])
    if reply.get("count_here") != n:
        reply["count_mismatch"] = [reply.get("count_here"), n]
    reply["count_here"] = n
    reply["sufficient"] = bool(reply.get("sufficient"))
    return reply


def lift_instances(reply: dict, scan: np.ndarray, pose: dict) -> list[dict]:
    """Each counted instance as a map-frame position, where the scanner reached it.

    An instance the scanner could not measure keeps its entry with `xy: None`.
    It still happened -- the model saw something -- it just cannot take part in
    de-duplication, which `merge` handles separately.
    """
    # `settle_coord_space` overrides a declaration the arithmetic refutes -- a
    # box coordinate larger than the image is not a pixel whatever the reply
    # calls it. Same correction the grounding call gets; a counting call sends
    # more boxes, so it has more chances to be wrong about the convention.
    space = settle_coord_space(dict(reply), "claude",
                              G.FACE_SIZE).get("coord_space")
    scan_cam = scan_to_camera(scan, pose)
    out = []
    for it in reply.get("instances", []):
        if not isinstance(it, dict):
            continue
        xy = None
        if it.get("box_2d") is not None and it.get("image_index") is not None:
            try:
                xy = _lift_xy(to_pixels(it["box_2d"], space, G.FACE_SIZE),
                              int(it["image_index"]), scan_cam, pose)
            except (ValueError, TypeError, IndexError):
                xy = None
        out.append({"xy": None if xy is None else np.asarray(xy, float)[:2],
                    "note": it.get("note") or "",
                    "certain": bool(it.get("certain", True))})
    return out


def merge(seen: list[dict], found: list[dict],
          gap: float = MERGE_M) -> list[dict]:
    """Fold one view's instances into the running set, by position.

    Positional merge only. Matching on the model's `note` was tried on paper
    and rejected: the notes are written per view ("the left one", "nearest the
    lamp") and mean different things from different sides of a sofa, so they
    are kept for the log and never for identity.
    """
    out = list(seen)
    for f in found:
        if f["xy"] is None:
            continue
        near = min((o for o in out if o["xy"] is not None),
                   key=lambda o: float(np.linalg.norm(o["xy"] - f["xy"])),
                   default=None)
        if near is not None and float(np.linalg.norm(near["xy"] - f["xy"])) < gap:
            near["seen"] = near.get("seen", 1) + 1
            # A later view that could see it plainly upgrades an earlier doubt.
            near["certain"] = near.get("certain", True) or f["certain"]
            continue
        out.append({**f, "seen": 1})
    return out


def merge_unlifted(views: list[dict]) -> int:
    """How many counted instances no view could place, at the most.

    These cannot be de-duplicated, so summing them would count one pillow once
    per view that saw it. The maximum any single view reported is the smallest
    number consistent with every view, and it is what the caller adds to the
    cluster count -- an undercount by construction, which is the right
    direction when a wrong integer and a missing one score the same.
    """
    return max((sum(1 for i in v if i["xy"] is None) for v in views), default=0)


def tally(views: list[list[dict]], gap: float = MERGE_M) -> dict:
    """The count over every view taken, and the working behind it."""
    seen: list[dict] = []
    for v in views:
        seen = merge(seen, v, gap)
    unplaced = merge_unlifted(views)
    return {"count": len(seen) + unplaced, "clusters": len(seen),
            "unplaced": unplaced,
            "per_view": [len(v) for v in views],
            "corroborated": sum(1 for o in seen if o.get("seen", 1) > 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", nargs="?",
                    help="a directory holding the four face JPEGs")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--target", required=True)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt-only", action="store_true")
    args = ap.parse_args()
    if args.snapshot and not args.question:
        args.snapshot, args.question = None, args.snapshot

    if args.prompt_only:
        print(build_prompt(args.question, args.target, args.anchor))
        return 0
    if not args.snapshot:
        print("a snapshot directory is required unless --prompt-only",
              file=sys.stderr)
        return 1

    d = Path(args.snapshot)
    faces = []
    for name in FACES:
        f = d / name
        if not f.is_file():
            f = d / name.replace("face_", "step1_face").replace("_front", "") \
                        .replace("_right", "").replace("_back", "").replace("_left", "")
        if not f.is_file():
            print(f"missing face: {d / name}", file=sys.stderr)
            return 1
        faces.append(f.read_bytes())

    r = count_view(faces, args.question, args.target, args.anchor, model=args.model)
    print(json.dumps(r, indent=1))
    if r.get("error"):
        return 1
    print(f"\ncount here {r['count_here']}   sufficient {r['sufficient']}"
          f"   confidence {r.get('confidence')}", file=sys.stderr)
    for i, it in enumerate(r["instances"]):
        print(f"  {i}  image {it.get('image_index')}  "
              f"{'' if it.get('certain', True) else '(unsure) '}{it.get('note')}",
              file=sys.stderr)
    if not r["sufficient"]:
        print(f"  not enough: {r.get('why_not')}  -> {r.get('next_view')}",
              file=sys.stderr)
    print(f"\nfaces in {NAMES} order", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
