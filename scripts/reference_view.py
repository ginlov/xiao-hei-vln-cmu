#!/usr/bin/env python3
"""Re-identify a committed target in one new view, and say how much of it is in frame.

The reference task has two model calls, and only the second one is new.

**Finding the target** is what `vlm_probe`'s grounding prompt already does, and
TASK 26 measured it: 72.2% of named objects localised within 1.0 m from a single
frame at the start pose, median error 0.11 m, bearing error 0.03°. Nothing here
replaces that; `answer_reference` calls it unchanged.

**Re-looking** is different, and confusing the two is the failure this module
exists to prevent. The grounding prompt asks *"find X in these four views"* --
an open search. After the vehicle has driven to the target and started to orbit
it, the question is no longer which object X is; that is settled. The question
is *"box the same object again from here"*. Replaying
`cv_0818_1745_arabic_room_arabicq4_p1` showed what an open search does across a
sequence of views: the loop's grounding drifted mid-leg from the stool onto a
table carrying a coffee pot, five of eight views landed on the table, and since
each view's weight is its inlier count, the wrong object won 511 to 203.

So this prompt is a re-identification prompt. Three consequences:

* It is shown **the previous view's crop** of the target, and asked whether what
  it is boxing is the same physical object. `vlm_approach.crop_face` already
  produces that crop and the grounding loop already carries one.
* It is **not** told where the running 3D estimate is, or how big the box has
  become. TASK 55 measured what naming the hypothesis does: told what it was
  checking, a verifier said `holds` on 83% of bindings beyond 5 m against 35%
  within 5 m -- it agreed more the less it could see -- and confirmed a wrong
  binding 5 times in 10. Blinding took that to 0 in 10 (p ~ 0.03).
* It must be able to answer *not in this view*. A forced box is worse than no
  box here, because a wrong box does not merely fail to help: it arrives with a
  large inlier count and outvotes the right ones.

It also reports one thing nothing else asks for: **whether the box holds the
whole object or a slice of it**. That is the evidence for a choice
`target_box.size_mode` cannot make on its own -- averaging per-view extents is
right when each view sees the whole object and wrong when each sees a different
face of it, and the model can see which case it is looking at.

    uv run --with anthropic python scripts/reference_view.py \\
        runs/xyz --step 4 "the pillow closest to the book on the stool"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "perception", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geometry as G  # noqa: E402
from vlm_probe import ask_claude, parse, settle_coord_space, to_pixels  # noqa: E402

MODEL = os.environ.get("XIAO_HEI_CLAUDE_MODEL", "claude-opus-5")

# How many views one orbit may spend. Not a budget limit -- the arithmetic says
# ~11 fit in the 540 s after driving there -- but a limit on how long we are
# willing to keep asking after the box has stopped moving.
MAX_VIEWS = int(os.environ.get("XIAO_HEI_REF_VIEWS", "5"))

PROMPT = """You are looking at what a robot sees from one spot in a room, and
you have already found the object you are looking for. It has been identified;
your job now is only to find it AGAIN from this new position.

The {which}four images are perspective views from the robot's current position,
each 100 deg wide:

  image 0 - heading   0 deg  (front)
  image 1 - heading  90 deg  (right)
  image 2 - heading 180 deg  (back)
  image 3 - heading 270 deg  (left)

Adjacent images overlap by about 10 deg, so an object near an edge may appear in
two of them.

THE OBJECT
  "{phrase}"
{crop_line}{drift_line}

WHAT COUNTS AS THE SAME OBJECT

The same physical thing. Not another object of the same type. This room very
likely contains several {type_hint}, and picking a different one is the single
most damaging thing you can do here -- more damaging than not answering.
{reference_line}
If you cannot find that same object in these images -- it is behind the robot's
own body, out of frame, or hidden -- answer `found: false`. That is a useful
answer. We will move and ask again.

Do NOT report an object of the right type that you believe is a different
instance. Answer `found: false` and say so in `why_not`.

THE BOX

Box the object's **whole visible silhouette**, as tightly as you can. The box
is used to select laser-scanner returns, so every pixel of background inside it
pulls the measurement off the object.

Then tell us how much of the object that box actually contains. This is the
question we most need answered and the one you are uniquely able to answer:

  "whole"      the box holds the entire object; nothing of it is outside
  "cut_off"    the image edge crosses the object; part of it is out of frame
  "occluded"   something in front hides part of it
  "face_only"  you can see one side or face of it and not its depth -- e.g.
               a picture seen straight on, a monitor from the front, a box
               whose far side you cannot see at all

`face_only` is not a failure. It is the normal case for a flat object and we
handle it differently, so say so rather than guessing at the hidden extent.

A laser scanner measures whatever the centre of your box points at. Three
things defeat it and you can see all three: glass, open structure (a railing,
a leafy plant, a wire shelf), and something standing in front. Report which.

Each image is {size} x {size} pixels.

Reply with JSON only, no prose, no markdown fence:

{{
  "coord_space": "pixels" or "normalized_1000",
      which convention the box below is in. State it explicitly -- with a
      {size}-pixel image both readings produce plausible-looking numbers, and
      a silent mismatch puts the box on a wall.
  "found": true or false,
  "image_index": 0-3 or null,
  "box_2d": [ymin, xmin, ymax, xmax] or null,
      the object's whole visible silhouette, in the coord_space you declared
  "same_object": true or false,
      is this the same physical object as the earlier view, not another of
      the same type. If you are answering from the description alone with no
      earlier view to compare against, say true and lower `confidence`.
  "extent": "whole" | "cut_off" | "occluded" | "face_only",
  "occlusion": "clear" | "behind_glass" | "see_through" | "partly_occluded",
  "why_not": "" or one of
      "not_in_view" | "hidden" | "cannot_tell_which_instance",
      why `found` is false, or why `same_object` is false
  "confidence": 0.0-1.0,
      probability that this is the same object as the earlier view
  "evidence": "what in the surroundings tells you it is the same one"
}}"""


def build_prompt(phrase: str, *, has_crop: bool, size: int = G.FACE_SIZE) -> str:
    """The re-look prompt. Takes no 3D state and has nowhere to put any.

    `has_crop` only changes the wording -- whether the prompt refers to an
    earlier view it can see. The caller decides whether to attach the crop; if
    it does not, the model is told to answer from the description and to lower
    its confidence, which is the honest fallback rather than a silent one.
    """
    crop_line = (
        "\nThe FIRST image is a crop from an earlier view, showing the object we\n"
        "mean. Use it to tell this object from others of the same type. The four\n"
        "views of the room follow it.\n"
        if has_crop else
        "\nYou have no earlier view of it to compare against -- go by the\n"
        "description alone, and say so by lowering your confidence.\n")
    return PROMPT.format(
        phrase=phrase,
        which="last " if has_crop else "",
        crop_line=crop_line,
        drift_line=(
            "\nThe robot has driven since that earlier view and is now looking at\n"
            "the object from a different angle, so it will not look the same. It\n"
            "may be turned, further away, nearer, or partly hidden by something\n"
            "that was not in the way before.\n"
            if has_crop else
            "\nSeveral objects here may match that description. Report one only if\n"
            "the description picks it out; otherwise answer `found: false`.\n"),
        type_hint=_plural_hint(phrase),
        reference_line=(
            "The earlier view is the reference; use its surroundings, not just\n"
            "the object itself, to tell which one it is.\n"
            if has_crop else
            "Use the surroundings the description names -- what it is on, near,\n"
            "or between -- not the object type alone.\n"),
        size=size)


def _plural_hint(phrase: str) -> str:
    """A rough plural of the phrase's head noun, for one sentence of prompt.

    Only ever cosmetic -- nothing branches on it, so a clumsy plural costs a
    slightly odd sentence and never an answer.
    """
    head = phrase.lower().replace("the ", "").strip().split()
    if not head:
        return "similar objects"
    word = head[0]
    for w in head:
        if w not in ("blue", "red", "green", "black", "white", "small", "large"):
            word = w
            break
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def look(faces: list[bytes], phrase: str, *, crop: bytes | None = None,
         model: str = MODEL) -> dict:
    """One re-look. Returns the parsed reply, never raises.

    A failed call must cost one view, not the question -- there are four more
    orbit positions and the box is already an average over whatever arrives.
    """
    images = ([crop] + list(faces)) if crop is not None else list(faces)
    try:
        raw = ask_claude(build_prompt(phrase, has_crop=crop is not None),
                         images, model)
    except Exception as e:                    # noqa: BLE001 -- see docstring
        return {"error": f"call failed: {e!r}", "found": False,
                "why_not": "call_failed"}
    reply = parse(raw)
    if not isinstance(reply, dict):
        return {"error": "unparseable reply", "raw": raw[:400], "found": False,
                "why_not": "unparseable"}
    return normalise(reply, has_crop=crop is not None)


def normalise(reply: dict, *, has_crop: bool) -> dict:
    """Coerce a reply into the shape the caller may rely on.

    Two corrections rather than trust:

    * the coordinate space is settled by arithmetic, not by the declaration.
      A box coordinate larger than the image is not a pixel whatever the reply
      calls it. TASK 26 is why this exists at all: the first grounding run
      parsed pixels as thousandths, measured a 23.7 deg bearing error, and the
      whole approach would have been recorded as a failure -- the correct
      reading gives 0.03 deg.
    * `found` is downgraded to false whenever the box is missing, the model
      says it is looking at a different instance, or the crop was shown and it
      could not tell which instance this is. A view we cannot trust must not
      reach the estimator with a heavy inlier count.
    """
    out = dict(reply)
    out["found"] = bool(out.get("found"))
    out["same_object"] = bool(out.get("same_object", True))
    out["extent"] = (out.get("extent") if out.get("extent") in EXTENTS
                     else "whole")
    out["confidence"] = _clamp(out.get("confidence"), 0.0, 1.0, 0.5)

    box, idx = out.get("box_2d"), out.get("image_index")
    if box is None or idx is None:
        out["found"] = False
        out.setdefault("why_not", "not_in_view")
    if not out["same_object"]:
        out["found"] = False
        out["why_not"] = out.get("why_not") or "cannot_tell_which_instance"

    if out["found"]:
        space = settle_coord_space(dict(out), "claude",
                                   G.FACE_SIZE).get("coord_space")
        try:
            out["box_px"] = to_pixels(out["box_2d"], space, G.FACE_SIZE)
            out["image_index"] = int(idx)
            out["coord_space_used"] = space
        except (ValueError, TypeError, IndexError) as e:
            out["found"] = False
            out["why_not"] = f"unreadable box ({e!r})"
            out["box_px"] = None
    else:
        out["box_px"] = None
    # Recorded so a run can be read back without re-deriving what was shown.
    out["had_crop"] = bool(has_crop)
    return out


EXTENTS = ("whole", "cut_off", "occluded", "face_only")

# What each `extent` says about how the view's box should be believed. The
# weight multiplies the inlier count `TargetBox` would otherwise use, so a view
# that admits it is looking at a slice counts for less when the extents are
# averaged. These are declared, not measured -- the orbit data that would
# settle them does not exist yet, and `--extent-weights off` turns them off so
# the first run measures the unweighted version too.
EXTENT_WEIGHT = {"whole": 1.0, "occluded": 0.6, "cut_off": 0.4,
                 "face_only": 0.5}


def view_weight(reply: dict, n_points: int, *, use_extent: bool = True) -> float:
    """The weight this view's box carries into the average.

    `n_points` is the measured evidence -- how much of the object the scanner
    actually reached -- and is the term `ObjectMap` measured as worth having
    (mIoU 0.203 -> 0.217). The rest are discounts for a view that told us it
    was not seeing the whole thing, and for a model that is unsure.
    """
    w = float(max(n_points, 0))
    if not use_extent:
        return w
    w *= EXTENT_WEIGHT.get(reply.get("extent", "whole"), 1.0)
    # Confidence is a soft factor, floored so a cautious view still counts.
    return w * max(0.4, float(reply.get("confidence", 0.5)))


def _clamp(v, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path, help="a run dir, or a snapshot dir")
    ap.add_argument("phrase")
    ap.add_argument("--step", type=int, default=None,
                    help="step number inside a run dir")
    ap.add_argument("--crop", type=Path, default=None, help="earlier-view crop")
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()

    if a.prompt_only:
        print(build_prompt(a.phrase, has_crop=a.crop is not None))
        return 0

    stem = (a.snapshot / f"step{a.step}_face") if a.step else (a.snapshot / "face")
    faces = [Path(f"{stem}{i}.jpg").read_bytes() for i in range(4)]
    crop = a.crop.read_bytes() if a.crop else None
    got = look(faces, a.phrase, crop=crop, model=a.model)
    print(json.dumps({k: v for k, v in got.items() if k != "box_px"}, indent=1))
    if got.get("box_px"):
        print(f"box_px: {got['box_px']}  image {got['image_index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
