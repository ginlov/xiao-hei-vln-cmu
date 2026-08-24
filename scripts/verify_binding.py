#!/usr/bin/env python3
"""A second opinion on a binding, asked only where geometry has no leverage.

`odometry_contradicts` in `approach_loop` refutes a binding by arithmetic: the
vehicle drove, so the range to a fixed point is predictable, and a lift that
measures something else is evidence no model opinion is involved in. It is the
better test wherever it applies, and it should keep priority.

It applies rarely. Over the finalised corpus the arbitration ladder reaches the
branch that test lives in on **2.2%** of grounding steps, because it needs a
rival *measurement* to compare against. The situation that actually dominates is
the opposite one: the lift is refused step after step -- a blind bearing, a size
the range cannot explain -- and the leg drives on a binding that nothing has
tested since it was made. That is **21.3%** of steps, and no comparison test can
touch any of it, because there is nothing to compare.

This module is for that gap, and only for it. It asks the model a question it is
good at and that needs no second measurement -- *is the thing you are driving at
the thing you were asked for* -- about a specific place in a specific image,
which turns an open nomination into a claim that can be checked. Three rules
keep it honest:

**It is prosecutorial, not generative.** The verifier never proposes a new
target. Its only power is to drop a binding, which returns the leg to grounding
from scratch. A verifier that could nominate would just be a second grounder
with a worse prompt.

**It is allowed to decline, and declining is most of its value.** The place may
be occluded, out of frame, or too far to resolve. `cannot_tell` is a first-class
answer and the default under any doubt, for the same reason `in_blind_cone`
refuses rather than guesses: a confident wrong answer costs more than no answer.

**It is gated.** It runs on the steps where the geometry has already given up,
not on every step. The measured trigger rate is ~12% of grounding steps, so a
leg's wall clock -- which is the real budget -- is not doubled.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import geometry as G  # noqa: E402
from vlm_locate import rot_from_quat  # noqa: E402
from vlm_probe import NAMES, ask_claude, parse  # noqa: E402
from xiao_hei_vln.perception.geometry import (  # noqa: E402
    sensor_to_camera_transform)

USE_VERIFY = os.environ.get("XIAO_HEI_VERIFY", "0") in ("1", "true", "yes")

# Which prosecutor. `anchored` states the belief and asks whether it holds;
# `blind` asks what is there without naming the phrase, then matches the answer
# to the phrase in a second, text-only call.
#
# The first design was `anchored`, and the audit measured it not working: over
# 29 fired steps it said "holds" on 83% of bindings more than 5 m out against
# 35% inside 5 m -- agreeing more the less it could see -- and refuted right
# bindings slightly more often than wrong ones. Naming the hypothesis in the
# prompt is the obvious mechanism, and `blind` is the obvious control for it.
VERIFY_STYLE = os.environ.get("XIAO_HEI_VERIFY_STYLE", "blind")

# How much of the face to keep either side of the projected column. A face is
# 100 degrees across; a third of it is still a room's worth of furniture, but it
# rules out "somewhere in this image" -- which is what the first design was
# effectively asking.
STRIP_FRAC = 0.34

# How many consecutive steps the lift may be refused before the binding is
# worth questioning. One refusal is ordinary -- the blind cone is body-fixed and
# heals by driving. Two in a row means the leg is committed to a position that
# has had no measurement behind it for two moves.
CARRIED_STEPS = 2

# Only a confident refutation acts. The verifier reports its own confidence and
# is told that `cannot_tell` is preferred to a guess, so this threshold is a
# second line rather than the first.
REFUTE_CONF = 0.7

# Below this the object is close enough that the next lift will settle it
# anyway, and a refutation would throw away a binding about to be confirmed.
MIN_RANGE_M = 1.5


def map_dir_to_cam(d_map: np.ndarray, pose: dict) -> np.ndarray:
    """Map-frame unit direction -> camera frame. Inverse of `cam_dir_to_map`.

    `vlm_approach.cam_dir_to_map` is `(d_cam @ R_sc) @ R_body.T` in row-vector
    form. Undoing it is `(d_map @ R_body) @ R_sc.T`, and `test_verify_binding`
    round-trips the pair rather than trusting this comment.
    """
    R_sc, _ = sensor_to_camera_transform()
    d_cam = (np.asarray(d_map, float) @ rot_from_quat(pose["orientation"])) @ R_sc.T
    n = float(np.linalg.norm(d_cam))
    return d_cam / n if n > 1e-9 else d_cam


def where_is(xy_map, pose) -> dict | None:
    """Which face the bound position falls in, and where across it.

    Returns `None` when the position is on top of the vehicle, where a bearing
    is meaningless. The elevation is not modelled: the binding is a map `xy`
    with no height, so this locates the *column* of the image and says so in
    the prompt rather than pretending to a pixel.
    """
    xy = np.asarray(xy_map, float)[:2]
    here = np.asarray(pose["position"], float)[:2]
    d = xy - here
    rng = float(np.linalg.norm(d))
    if rng < 1e-3:
        return None
    d_map = np.array([d[0] / rng, d[1] / rng, 0.0])
    d_cam = map_dir_to_cam(d_map, pose)

    for i in range(len(NAMES)):
        u, _, in_front = G.world_dir_to_face_pixel(d_cam[None, :], i)
        u, in_front = float(u[0]), bool(np.asarray(in_front).ravel()[0])
        if in_front and 0.0 <= u < G.FACE_SIZE:
            # Where across the face, as a fraction, because a pixel column
            # implies a precision the missing elevation does not support.
            return {"face": i, "u": u, "across": u / G.FACE_SIZE,
                    "range_m": rng}
    return None


def _side(across: float) -> str:
    if across < 0.35:
        return "the left third"
    if across > 0.65:
        return "the right third"
    return "the middle"


def build_verify_prompt(phrase: str, at: dict, why: str, *,
                        carried_steps: int = 0) -> str:
    """The prosecutor's brief: one place, one claim, three permitted answers."""
    place = (f"image {at['face']} ({NAMES[at['face']]}), "
             f"{_side(at['across'])} of it, about {at['range_m']:.1f} m away")
    if why == "carried":
        stale = (f"The lidar has failed to measure this target for "
                 f"{carried_steps} steps running, so nothing has tested this "
                 f"belief since it was made. It may well be right. It has "
                 f"simply not been checked.")
    elif why == "rejected":
        stale = ("A new reading just landed more than a metre from this "
                 "position while still calling it the same object, and the "
                 "geometric tests could not decide between them.")
    else:
        stale = ("The robot is about to stop here and call this leg done, on a "
                 "position no measurement has confirmed.")

    return f"""You are checking one claim. You are not being asked to find anything.

The robot is driving to a position it believes is: "{phrase}"

From where it stands now, that position is at {place}.

{stale}

Look at that place, in that image. Answer only about what is actually there.

Reply as JSON and nothing else:

{{"verdict": "holds" | "refuted" | "cannot_tell",
 "what_is_there": "what you can see at that place, in a few words",
 "why": "one sentence",
 "confidence": 0.0 to 1.0}}

Rules, in order of importance:

1. Answer "refuted" only when you can see that place clearly AND what is there
   is not "{phrase}". A different object of the same kind still counts as
   refuted if the phrase names which one.
2. Answer "cannot_tell" whenever the place is occluded, out of frame, too far
   or too small to resolve, or you are simply unsure. This is a normal answer
   and it is strongly preferred to a guess. You are not penalised for it.
3. Answer "holds" only if you can see the thing the phrase names, there.
4. Do not suggest a better target. That is not your job here, and a suggestion
   will be discarded."""


def strip_of(face: bytes, across: float, frac: float = STRIP_FRAC) -> bytes:
    """A full-height vertical band of one face, centred on the column.

    Full height on purpose: the binding is a map `xy` and carries no elevation,
    so the column is known and the row is not. Cropping vertically would be
    inventing the one coordinate we do not have.
    """
    # cv2 rather than PIL. The submission image installs python3-opencv and
    # not pillow, so the PIL version of this raised ModuleNotFoundError the
    # first time anything switched the verifier on inside the container -- a
    # failure that could not happen here, where pillow is a dev dependency.
    im = cv2.imdecode(np.frombuffer(face, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return face
    h, w = im.shape[:2]
    half = max(1, int(w * frac / 2))
    c = int(round(across * w))
    lo, hi = max(0, c - half), min(w, c + half)
    if hi - lo < 2 * half:                     # keep the band a constant width
        lo, hi = max(0, min(lo, w - 2 * half)), min(w, max(hi, 2 * half))
    ok, buf = cv2.imencode(".jpg", im[:, lo:hi],
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else face


DESCRIBE = """This is a vertical slice of one camera image from a robot.

Something about {rng:.1f} metres away, near the centre of this slice, matters.

List what you can see there, at roughly that distance. Name things plainly, in
a few words each. Include anything you are unsure of, marked as uncertain.

Do not skip an object because it seems unimportant, and do not leave out plain
surfaces -- a wall, a doorway, bare floor are all answers, and often the right
one.

Reply as JSON and nothing else:

{{"objects": ["...", "..."], "unsure": true or false}}

If the slice is too dark, too far, or too cluttered to say, reply with an empty
list and "unsure": true. That is a normal answer."""


MATCH = """A robot was asked to drive to: "{phrase}"

At the place it is driving to, a camera reported seeing:

{seen}

Does that report describe "{phrase}"?

Reply as JSON and nothing else:

{{"verdict": "holds" | "refuted" | "cannot_tell",
 "why": "one sentence",
 "confidence": 0.0 to 1.0}}

Rules:

1. "refuted" only if what was reported clearly is not "{phrase}". If the phrase
   names WHICH one ("closest to the door", "furthest from the sofa"), a
   different instance of the same kind of object still counts as refuted only
   when the report makes that clear -- a bare noun match does not settle it.
2. "cannot_tell" whenever the report is empty, hedged, or simply does not
   settle the question. This is a normal answer and is preferred to a guess.
3. "holds" only if the report names the thing the phrase names."""


def blind_verdict(faces: list[bytes], phrase: str, at: dict,
                  model: str) -> tuple[dict, int]:
    """Describe the place without naming the phrase, then match. `(reply, calls)`.

    Two calls, the second text-only and therefore cheap. The split is the whole
    point: the eye that looks never learns what it is supposed to find, so it
    cannot agree with it.
    """
    face = faces[at["face"]]
    try:
        band = strip_of(face, at["across"])
    except Exception:                              # noqa: BLE001
        band = face                                # a whole face still beats none
    seen_raw = ask_claude(DESCRIBE.format(rng=at["range_m"]), [band], model)
    seen = parse(seen_raw)
    objs = (seen or {}).get("objects") or []
    if not objs:
        return {"verdict": "cannot_tell", "confidence": 0.0,
                "what_is_there": None,
                "why": "nothing reportable at that place"}, 1
    listed = "\n".join(f"- {o}" for o in objs[:8])
    if (seen or {}).get("unsure"):
        listed += "\n(the camera reported it was unsure)"
    out = parse(ask_claude(MATCH.format(phrase=phrase, seen=listed), [], model))
    if not isinstance(out, dict):
        return {"verdict": "cannot_tell", "confidence": 0.0,
                "what_is_there": "; ".join(objs[:4]),
                "why": "match reply unparseable"}, 2
    out["what_is_there"] = "; ".join(objs[:4])
    return out, 2


def verify_binding(faces: list[bytes], phrase: str, bound_xy, pose: dict,
                   *, why: str, model: str, carried_steps: int = 0,
                   prev: bytes | None = None) -> dict:
    """Ask, and return a record. `{"acted": bool, ...}`; never raises.

    A verifier that can take a run down is worse than no verifier: this is an
    extra network call on a ten-minute budget, added late, and every failure of
    it must leave the binding exactly as it was.

    `called` says whether the network was actually reached, so the caller can
    bill wall clock honestly: a refused place costs nothing, an unparseable
    reply costs a full call.
    """
    at = where_is(bound_xy, pose)
    if at is None:
        return {"skipped": "no bearing to the binding", "called": False}
    if at["range_m"] < MIN_RANGE_M:
        return {"skipped": f"already {at['range_m']:.2f} m away", "called": False}

    calls = 1
    try:
        if VERIFY_STYLE == "blind":
            reply, calls = blind_verdict(faces, phrase, at, model)
        else:
            reply = parse(ask_claude(
                build_verify_prompt(phrase, at, why,
                                    carried_steps=carried_steps),
                faces, model, previous=prev))
    except Exception as exc:                       # noqa: BLE001
        # A refused or timed-out call still cost wall clock, and the budget
        # is wall clock, so it is counted.
        return {"skipped": f"call failed: {type(exc).__name__}", "called": True}
    if not isinstance(reply, dict):
        return {"skipped": "unparseable reply", "called": True}

    verdict = str(reply.get("verdict", "")).lower()
    try:
        conf = float(reply.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    rec = {"called": True, "calls": calls, "style": VERIFY_STYLE, "why": why,
           "at": {k: at[k] for k in ("face", "across", "range_m")},
           "verdict": verdict, "confidence": conf,
           "what_is_there": reply.get("what_is_there"),
           "reason": reply.get("why"),
           "acted": verdict == "refuted" and conf >= REFUTE_CONF}
    return rec


def should_verify(*, carried_run: int, rejected: bool, arriving: bool,
                  bound: dict | None) -> str | None:
    """Which of the three triggers fired, or `None`.

    Ordered by how blind the geometry is at that moment, not by how often each
    happens. `carried` is the case the geometric test provably cannot reach; the
    other two are cases where it applied and did not settle the question.
    """
    if bound is None:
        return None
    if carried_run >= CARRIED_STEPS:
        return "carried"
    if rejected:
        return "rejected"
    if arriving and not bound.get("verified", True):
        return "arriving"
    return None


if __name__ == "__main__":                          # a bearing, by hand
    pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    for xy in ([3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]):
        print(xy, json.dumps(where_is(xy, pose)))
