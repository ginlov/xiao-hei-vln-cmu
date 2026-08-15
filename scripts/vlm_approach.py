#!/usr/bin/env python3
"""From a grounding box to the next waypoint, without betting on the range.

TASK 26 measured the two channels a grounding call produces and found them
wildly unequal. Bearing is excellent — 0.03° on the case checked by hand, and
the hits localise to a median 0.11 m. Range is where the approach dies: 9 of
the 15 misses are a correct box whose depth lift went wrong, and the model's
own `distance_m` is *worse than a constant* (median error 2.80 m against 2.36 m
for always answering 5.0).

So this module never asks for a range it does not have. It commits to a
destination only when the lift survives a size check, and otherwise takes a
bounded step along the bearing into space the scan says is free, expecting to
re-observe. That converges for a reason worth stating: returns on a target grow
as 1/r², so the regime where depth fails is the regime you are about to leave.

Two views bought that way also give triangulation for free, which is better
than either estimator we have — see `triangulate`.

    uv run --with anthropic python scripts/vlm_approach.py snaps/start \\
        "the folding screen"
    uv run --with anthropic python scripts/vlm_approach.py snaps/start \\
        snaps/step1 "the folding screen"          # two views, triangulated
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from vlm_locate import locate, rot_from_quat, scan_to_camera  # noqa: E402
from vlm_probe import (DEFAULT_PROMPT_VER, NAMES, ask_claude,  # noqa: E402
                       ask_gemini, build_prompt, load_faces, parse, to_pixels)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from xiao_hei_vln.perception.geometry import sensor_to_camera_transform  # noqa: E402
from xiao_hei_vln.perception.size_prior import size_for  # noqa: E402

# The median distance the reference trajectories keep from the objects they
# name (`scripts/traj_tolerance.py`). Stopping here is stopping "at" the object.
STANDOFF_M = 0.6
# One uncommitted leg. Long enough to change the geometry materially, short
# enough that a wrong bearing costs one extra call rather than the question.
MAX_STEP_M = 2.0
MIN_ADVANCE_M = 0.3
# vehicleWidth 0.5 in local_planner.launch, plus margin.
HALF_WIDTH_M = 0.35
# obstacleHeightThre 0.05; the upper bound keeps the ceiling out of it.
OBSTACLE_Z = (0.05, 1.2)
# How far the implied size may stray from the class prior before the range is
# disbelieved. Wide on purpose: this fires on "a screen would have to be 4.5 m
# tall", not on a 30% disagreement.
SIZE_TOL = 2.5
SIZE_BAND = (0.15, 4.0)

# TASK 26 listed "replace the crude global band with per-class priors" as the
# obvious next step. Measured on the same 54 rows, it loses at every tolerance:
#
#   A + global band [0.15, 4.0]     hits 70.4%   FP 11.1%
#   A + per-class tol 2.5           hits 68.5%   FP 11.1%
#   A + per-class tol 1.75          hits 61.1%   FP  9.3%
#
# Two reasons, both structural. 12 of 52 claimed phrases ("folding screen",
# "exit sign", "tea table") have no entry and fall through to the band anyway.
# And every false positive that survives the band is a *wrong instance* — the
# horse figurine's box on the elephant, a chair on the balcony rather than
# indoors — where the object picked is the same size as the object wanted, so
# no size test can separate them. Kept behind a flag rather than deleted,
# because the argument for it is sound and only the data refutes it.
USE_CLASS_PRIOR = False

# The simulator's lidar is tilted forward: how far below horizontal it sees
# depends on bearing. Measured as the 0.5th percentile of return elevation per
# 15° azimuth bin over seven scenes, where it repeats to within 1-3°:
# -32° dead ahead, -15° abeam, and +9° behind — there is a large blind cone
# under and behind the robot.
#
# This is the whole explanation for "no lidar return". All four such cases in
# the sweep point below this floor at their bearing, 4 for 4; none of them is
# a sampling-density problem and no cone width recovers them. The cone is
# body-fixed, so it rotates with the robot: driving one leg toward the target
# puts it near 0° azimuth where the floor is -32°, which is why the step-and-
# re-observe fallback fixes these rather than merely deferring them.
COVERAGE_FLOOR_DEG = [(7, -32.4), (22, -31.3), (37, -28.9), (52, -25.5),
                      (67, -20.6), (82, -15.5), (97, -11.0), (112, -4.7),
                      (127, -0.2), (142, 4.0), (157, 7.6), (172, 9.1)]


@dataclass
class Waypoint:
    """Where to drive next, and whether it is the destination or a step."""

    xy: np.ndarray
    committed: bool
    range_m: float | None
    reason: str


def ray_from_box(box_px, face_idx: int) -> np.ndarray:
    """Unit direction of the box centre, camera frame."""
    ymin, xmin, ymax, xmax = box_px
    d = G.face_pixel_to_world_dir(np.array([(xmin + xmax) / 2]),
                                  np.array([(ymin + ymax) / 2]), face_idx)[0]
    return d / np.linalg.norm(d)


def cam_dir_to_map(d_cam: np.ndarray, pose: dict) -> np.ndarray:
    """Camera-frame direction → unit direction in the map frame."""
    R_sc, _ = sensor_to_camera_transform()
    d_map = (d_cam @ R_sc) @ rot_from_quat(pose["orientation"]).T
    return d_map / np.linalg.norm(d_map)


def box_angular_size(box_px, face_idx: int) -> tuple[float, float]:
    """True (width, height) of a box in degrees.

    The sweep used pixels/FACE_SIZE × FOV, which the gnomonic projection makes
    wrong away from the face centre — a centred box's real subtense is larger
    than the linear reading, so implied sizes came out systematically small.
    Measuring the angle between the actual edge directions costs four LUT
    lookups and removes the bias.
    """
    ymin, xmin, ymax, xmax = box_px
    cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2

    def ang(pa, pb) -> float:
        (xa, ya), (xb, yb) = pa, pb
        a = G.face_pixel_to_world_dir(np.array([xa]), np.array([ya]), face_idx)[0]
        b = G.face_pixel_to_world_dir(np.array([xb]), np.array([yb]), face_idx)[0]
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        return float(np.rad2deg(np.arccos(np.clip(a @ b, -1.0, 1.0))))

    return ang((xmin, cy), (xmax, cy)), ang((cx, ymin), (cx, ymax))


def implied_size(range_m: float, angle_deg: float) -> float:
    """Physical extent an object must have to subtend ``angle_deg`` at ``range_m``."""
    return 2.0 * range_m * float(np.tan(np.deg2rad(angle_deg) / 2.0))


def prior_for_phrase(phrase: str) -> tuple[str | None, tuple[float, float, float] | None]:
    """Longest suffix of the phrase that has a size prior.

    "the folding screen" has no entry but "screen" may; dropping leading words
    walks from the most specific reading to the most general one.
    """
    toks = re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split()
    for i in range(len(toks)):
        key = " ".join(toks[i:])
        p = size_for(key)
        if p is not None:
            return key, p
    return None, None


def size_gate(range_m: float, height_deg: float, phrase: str | None,
              tol: float = SIZE_TOL,
              use_prior: bool = USE_CLASS_PRIOR) -> tuple[bool, str]:
    """Is ``range_m`` consistent with the object being what it claims to be?

    Gated on the vertical extent because it is the one dimension that does not
    depend on which way the object is facing — a cabinet seen end-on is a third
    of its width but exactly its height.
    """
    h = implied_size(range_m, height_deg)
    key, prior = (prior_for_phrase(phrase or "") if use_prior else (None, None))
    if prior is None:
        lo, hi = SIZE_BAND
        return lo <= h <= hi, f"implied height {h:.2f} m in [{lo}, {hi}]"
    lo, hi = prior[2] / tol, prior[2] * tol
    return (lo <= h <= hi,
            f"implied height {h:.2f} m vs {key!r} prior {prior[2]:.2f} m, "
            f"band [{lo:.2f}, {hi:.2f}]")


def sensor_bearing(d_cam: np.ndarray) -> tuple[float, float]:
    """Camera-frame direction → (azimuth, elevation) in the sensor frame, degrees.

    Azimuth is 0 straight ahead and positive to the left; elevation is positive
    up. Both are what `COVERAGE_FLOOR_DEG` is indexed by.
    """
    R_sc, _ = sensor_to_camera_transform()
    d = d_cam @ R_sc
    d = d / np.linalg.norm(d)
    return (float(np.rad2deg(np.arctan2(d[1], d[0]))),
            float(np.rad2deg(np.arcsin(np.clip(d[2], -1.0, 1.0)))))


def lidar_elevation_floor(az_deg: float) -> float:
    """Lowest elevation the scanner returns anything at, for this bearing."""
    a = abs(((az_deg + 180.0) % 360.0) - 180.0)
    xs = [p[0] for p in COVERAGE_FLOOR_DEG]
    ys = [p[1] for p in COVERAGE_FLOOR_DEG]
    return float(np.interp(a, xs, ys))


def in_blind_cone(d_cam: np.ndarray) -> tuple[bool, float, float, float]:
    """Does this ray point into the scanner's blind cone? (blind, az, el, floor)"""
    az, el = sensor_bearing(d_cam)
    floor = lidar_elevation_floor(az)
    return el < floor, az, el, floor


def free_range_along(ray_xy: np.ndarray, origin: np.ndarray, scan_map: np.ndarray,
                     *, half_width: float = HALF_WIDTH_M) -> float:
    """How far along ``ray_xy`` the robot can drive before something is in the way.

    This is the number that replaces a guessed range. It is a measurement we
    already trust — the same returns the local planner refuses to drive into —
    rather than an estimate of where the target is.
    """
    p = scan_map[:, :3]
    p = p[np.linalg.norm(p, axis=1) > 1e-3]
    z = p[:, 2] - origin[2]
    p = p[(z > OBSTACLE_Z[0]) & (z < OBSTACLE_Z[1])]
    if len(p) == 0:
        return float("inf")
    d = p[:, :2] - origin[:2]
    # Inputs are finite (checked) and ray_xy is unit; the warnings numpy raises
    # here come from the vectorised matmul path, not from the data.
    with np.errstate(all="ignore"):
        along = d @ ray_xy
    perp = np.abs(d[:, 0] * ray_xy[1] - d[:, 1] * ray_xy[0])
    hit = (along > MIN_ADVANCE_M) & (perp < half_width)
    return float(along[hit].min()) if hit.any() else float("inf")


# Two lifts further apart than this are looking at different objects. Not a new
# threshold: it is `dominant_cluster`'s `gap_m`, which is already this file's
# definition of when neighbouring returns stop belonging to one thing.
FEATURE_AGREE_M = 0.35


def box_range(box_px, face_idx: int, scan_cam: np.ndarray,
              pose: dict) -> float | None:
    """Lidar range at a box's centre, or None when nothing returns in the cone."""
    w_deg, h_deg = box_angular_size(box_px, face_idx)
    cone = float(np.clip(min(w_deg, h_deg) / 4.0, 1.0, 5.0))
    res = locate(box_px, face_idx, scan_cam, cone_deg=cone, pose=pose)
    return float(res["range"]) if res.get("n") else None


def aim_box(reply: dict, size: int, scan_map: np.ndarray,
            pose: dict) -> tuple[list[float], str | None]:
    """The box to point the ray at, and why the feature box was refused.

    `feature_box_2d` is the distinguishing feature *of* the target — the
    elephant figurine on the tea table — and where it really is one it is the
    better box to lift from: tighter, and its edges do not run off the object
    onto the wall behind. The trouble is that the anchor lands there too. Asked
    for "the water cooler near the window", `runs/o_1_0814_02` put the cooler
    in `box_2d` and the window frame in `feature_box_2d`, 24° away; the leg
    drove at the window, lifted 1.77 m against the cooler's 1.17 m, and failed
    `boxed in` standing in front of the thing it was sent to.
    `_RELATIONAL_BRANCH` already warns that returning the anchor is the
    commonest way to get this wrong, but it only guards the comparative forms
    it names, and "near the window" is not one of them.

    Containment is the obvious test and it is wrong. A feature sitting *on*
    something legitimately pokes out above its box: over 222 recorded steps
    carrying both boxes, 84% put the feature box outside the target box, and
    the ones that do include the figurine on the tea table and the clock on
    the nightstand, which are exactly the cases worth keeping. Angular
    separation does not divide them either — the clock is 28° off its
    nightstand, and the table *under* the plant, which is an anchor, is 38°.

    What does divide them is the measurement we are about to take. A feature on
    the target is at the target's range; an anchor across the room is not. Over
    the 102 steps where both boxes lift, the figurine reads 0.26 m and 0.21 m
    from its table and the clock 0.17 m from its nightstand, while the window
    reads 0.50 m and 0.60 m from the cooler and the sofa 0.47 m from the
    guitar. Testing the range costs one more `locate` on a scan already in
    memory, and it tests the property that actually matters — whether the ray
    lands on the target — rather than a proxy for it.

    Refusing is cheap because the feature box is not the more accurate of the
    two anyway: scored against the model's own independent `distance_m` over
    the steps this refuses, the target box was closer on 40 and the feature box
    on 41. It is worth keeping when it is real, and it is not worth guessing.
    """
    space = reply.get("coord_space")
    target = to_pixels(reply["box_2d"], space, size)
    raw = reply.get("feature_box_2d")
    if not raw:
        return target, None
    feature = to_pixels(raw, space, size)
    face = int(reply["image_index"])
    cam = scan_map_to_cam(scan_map, pose)
    rt, rf = (box_range(target, face, cam, pose),
              box_range(feature, face, cam, pose))
    # Unverifiable is not the same as wrong, but the target box is the one the
    # model was asked to draw round the target, so it is what an unresolved
    # disagreement falls back to. Where only the feature box lifts, taking the
    # target box costs a commit and buys a step — and a step is what we want
    # when the thing we would have committed to might be a window.
    if rt is None or rf is None:
        return target, ("feature box ignored: no lift to check it against "
                        f"(target {rt}, feature {rf})")
    if abs(rt - rf) > FEATURE_AGREE_M:
        return target, (f"feature box ignored: it lifts to {rf:.2f} m against "
                        f"the target box's {rt:.2f} m — different objects")
    return feature, None


def next_waypoint(box_px, face_idx: int, scan_map: np.ndarray, pose: dict, *,
                  phrase: str | None = None, standoff: float = STANDOFF_M,
                  max_step: float = MAX_STEP_M) -> Waypoint:
    """Destination if the lift is believable, otherwise a bounded step toward it."""
    origin = np.asarray(pose["position"], float)
    d_cam = ray_from_box(box_px, face_idx)
    d_map = cam_dir_to_map(d_cam, pose)

    ray_xy = d_map[:2]
    n = float(np.linalg.norm(ray_xy))
    if n < 1e-3:
        # Straight up or down — a ceiling lamp. There is no ground move that
        # closes on it, and the robot is already under it.
        return Waypoint(origin[:2].copy(), True, None, "target is overhead")
    ray_xy = ray_xy / n

    w_deg, h_deg = box_angular_size(box_px, face_idx)
    # A cone inside the box's inner quarter: wide enough to catch returns,
    # narrow enough to stay off the background the box edges include.
    cone = float(np.clip(min(w_deg, h_deg) / 4.0, 1.0, 5.0))
    res = locate(box_px, face_idx, scan_map_to_cam(scan_map, pose),
                 cone_deg=cone, pose=pose)

    blind, az, el, floor = in_blind_cone(d_cam)
    why = None
    if res.get("n") and not blind:
        ok, why = size_gate(res["range"], h_deg, phrase)
        if ok:
            reach = max(res["range"] - standoff, MIN_ADVANCE_M)
            return Waypoint(origin[:2] + ray_xy * reach, True, res["range"],
                            f"lift {res['range']:.2f} m accepted ({why})")
        why = f"lift {res['range']:.2f} m rejected ({why})"
    elif blind:
        # A blind bearing does not merely lose the lift — it can return a
        # confident wrong one. On the first live run a target at -120° (floor
        # -2°) lifted to 4.70 m against a true ~2.9 m, because the cone widened
        # upward until it found the wall above the object, and the implied size
        # was plausible enough for the gate. TASK 27 let blind lifts through on
        # the strength of two replayed cases; driving found the counterexample.
        #
        # Never commit one. Stepping loses nothing, because the step swings the
        # bearing toward 0° where the floor is -32° and the next lift is sound.
        why = (f"bearing {az:+.0f}° at {el:+.0f}° is below the scanner's "
               f"{floor:+.0f}° floor"
               + (f" — lift {res['range']:.2f} m not trusted" if res.get("n")
                  else " — blind")
               + ", turning fixes it")
    else:
        why = f"no returns in a {res['cone']:.1f}° cone"

    # Do not guess the range. Step into space the scan says is free and look
    # again from there; the lift gets easier as 1/r².
    free = free_range_along(ray_xy, origin, scan_map)
    step = min(max_step, max(free - standoff, MIN_ADVANCE_M))
    return Waypoint(origin[:2] + ray_xy * step, False, None,
                    f"{why}; stepping {step:.2f} m (free {free:.2f} m)")


def scan_map_to_cam(scan_map: np.ndarray, pose: dict) -> np.ndarray:
    return scan_to_camera(scan_map, pose)


def _lift_xy(box_px, face_idx: int, scan_cam: np.ndarray,
             pose: dict) -> np.ndarray | None:
    w, h = box_angular_size(box_px, face_idx)
    cone = float(np.clip(min(w, h) / 4.0, 1.0, 5.0))
    res = locate(box_px, face_idx, scan_cam, cone_deg=cone, pose=pose)
    return res["patch_xy"] if res.get("n") else None


_RELATIONAL = re.compile(
    r"\b(?:closest|nearest|farthest|furthest|between|near|next\s+to|beside)\b",
    re.I)


def has_relation(phrase: str) -> bool:
    """Does this phrase name the target by its relation to something else?

    Asked of the phrase, not of the reply, because the model is not consistent
    about reporting one. Given "the guitar near the couch" it returned no
    relation on one call and `closest_to` on the very next; the call that
    forgot was treated as needing no check, and bound the couch.

    Deliberately narrow: only the relations `resolve_relation` can actually
    settle by measuring. "the vases on the cabinet below the TV" describes a
    position rather than a comparison, and demanding an anchor lift for it
    would block a target that grounds perfectly well on its own.
    """
    return bool(_RELATIONAL.search(phrase or ""))


@dataclass
class Resolved:
    """The winner of a comparative relation, and how much of it was compared.

    `box`, `image_index` and `why` are what the caller has always used.
    `complete` and `missed` are the part `japanese_room` proved was needed: a
    comparison run over a subset that excludes the right answer does not fail,
    it returns a confident wrong winner.
    """

    box: list
    image_index: int
    why: str
    complete: bool = True
    # (azimuth, elevation, note) for each candidate the lift could not place.
    missed: list = field(default_factory=list)

    def __iter__(self):
        """So `box, i, why = resolved` keeps working at the older call sites."""
        return iter((self.box, self.image_index, self.why))


def resolve_relation(reply: dict, scan_map: np.ndarray, pose: dict,
                     *, size: int = 640) -> "Resolved | None":
    """Decide a comparative relation by measuring, not by asking.

    v4 asks the model for every candidate and the anchor rather than for the
    winner, because on the first live relational phrase — "the lantern closest
    to the fan decoration" — it answered with the fan decoration. Here we lift
    each box to a map position and do the comparison ourselves.

    Returns a `Resolved`, or None when the measurement cannot be made and the
    caller should fall back to the model's own pick. Falling back is not a
    silent failure: with fewer than two liftable candidates there is nothing to
    compare, and the model's choice is the only answer available.

    One nomination is a special case, not a failed comparison — see below.

    **A candidate that will not lift is dropped from the comparison, and that
    is reported.** `runs/jr_0812_01` step 2, on this same lantern phrase, is
    what the flag is for: the model listed the correct floor lantern and placed
    it within 0.2° of the truth, the anchor within 0.4°, and the lift refused
    it because at azimuth -174.6° it sat 17.7° under the scanner's floor —
    directly behind the robot. Three of four candidates lifted, `len(cb) >= 2`
    held, and the comparison declared a ceiling lantern 3.98 m from the answer
    with a rationale that reads as authoritative. Fewer than two is not the
    only way this measurement can be wrong; fewer than all is the other, and it
    is the dangerous one because it still produces a number.
    """
    rel = reply.get("relation")
    cands = reply.get("candidates") or []
    anchors = reply.get("anchors") or []
    if rel not in ("closest_to", "farthest_from", "between"):
        return None
    if not cands or not anchors:
        return None

    scan_cam = scan_to_camera(scan_map, pose)
    space = reply.get("coord_space")

    def boxes(items):
        """Lifted items, and the ones that would not lift with their bearing.

        The bearing is kept because it is the useful half of a failed lift:
        a candidate the scanner cannot reach is usually behind the robot, and
        that is a direction to turn, not a thing to forget.
        """
        out, lost = [], []
        for it in items:
            if it.get("box_2d") is None or it.get("image_index") is None:
                continue
            px = to_pixels(it["box_2d"], space, size)
            i = int(it["image_index"])
            xy = _lift_xy(px, i, scan_cam, pose)
            if xy is not None:
                out.append((px, i, xy, it))
            else:
                az, el = sensor_bearing(ray_from_box(px, i))
                lost.append((az, el, (it.get("note") or it.get("name") or "?")))
        return out, lost

    (cb, missed), (ab, _) = boxes(cands), boxes(anchors)
    if not ab:
        # Nothing to measure against: the phrase is unchecked, and the caller
        # must treat whatever the model nominated as a guess at the noun.
        return None
    if len(cands) == 1 and len(cb) == 1:
        # The model saw exactly one thing matching the noun, and it lifted.
        # There is no comparison to make, so refusing to resolve here does not
        # protect anything — it discards a measured position. On `studio` it
        # cost the second destination: the model had correctly boxed "guitar
        # leaning against the wall/shelf beyond the sofa's right end", the loop
        # called that unmeasurable, and kept an earlier binding that sat inside
        # the couch.
        c = cb[0]
        return Resolved(c[0], c[1],
                        f"{rel}: one nomination "
                        f"({(c[3].get('note') or '?')[:52]}), "
                        f"{len(ab)} anchor(s) lifted — nothing to compare it "
                        f"against")
    if len(cb) == 1:
        # One liftable candidate out of several is an *undecided* comparison —
        # the ones that would not lift might have won — and that used to return
        # None, which the caller reads as "unverified" and answers by keeping
        # whatever binding it already had.
        #
        # `runs/jr_0812_04` leg 1 is what that costs. At steps 3 and 4 the
        # anchor lifted to 0.11 m and 0.02 m of the true fan decoration, and
        # the single candidate that lifted was the right lantern, 0.57 m and
        # 0.64 m from it. Both calls produced a committed waypoint **0.05 m and
        # 0.04 m from the truth**. Both were discarded in favour of a binding
        # carried from step 2 that sat 3.93 m away, and the leg then reported
        # `arrived, within standoff` because the robot happened to stop 0.25 m
        # from that wrong binding. The two candidates that failed to lift were
        # the tokonoma ledge lanterns, behind the robot in the blind cone — the
        # wrong ones.
        #
        # So: undecided is not unmeasurable. Return the one that lifted, marked
        # incomplete, which binds it while withholding the licence to overrule
        # a later complete comparison. Same treatment as a partial comparison,
        # because it is one — with one survivor instead of several.
        #
        # **Except on `farthest_from`, where the survivor is evidence against
        # itself.** A lift fails because the object is far, or occluded, or
        # under the scanner's floor — so "the only candidate that lifted" is
        # biased toward the near ones, and `farthest_from` is precisely the
        # question whose answer is the far one. `runs/ar_0812_03` leg 1 is that
        # bias costing a leg: *"the potted plant furthest from the hookah"*,
        # one of two candidates lifted, and it was the nearest plant of five —
        # 8.59 m from the answer. It bound, and steps 2 and 3 then produced
        # committed waypoints 0.37 m and 0.55 m from the truth that `JUMP_M`
        # refused because a binding was already held. Returning None here, as
        # the code did before, leaves those two steps free to drive at the
        # right plant. `jr_0812_04`, which this rescue was written for, is
        # `closest_to`, where the same bias points at the answer instead.
        if rel == "farthest_from":
            return None
        c = cb[0]
        why = (f"{rel}: only 1 of {len(cands)} candidate(s) lifted "
               f"({(c[3].get('note') or '?')[:46]}) vs {len(ab)} anchor(s) — "
               f"undecided")
        if missed:
            why += ("; NOT compared: " +
                    ", ".join(f"{n[:34]} (az {a:+.0f}°, el {e:+.0f}°)"
                              for a, e, n in missed))
        return Resolved(c[0], c[1], why, complete=False, missed=missed)
    if len(cb) < 2:
        return None

    def score(xy: np.ndarray) -> float:
        d = [float(np.linalg.norm(xy - a[2])) for a in ab]
        # "between A and B" has no single reference point; the candidate that
        # best sits between them is the one minimising the total distance to
        # both, which for points on the segment is just the segment length.
        return sum(d) if rel == "between" else d[0]

    pick = (max if rel == "farthest_from" else min)(cb, key=lambda c: score(c[2]))
    ranked = sorted((score(c[2]), (c[3].get("note") or "?")) for c in cb)
    why = (f"{rel} over {len(cb)} lifted candidate(s) vs "
           f"{len(ab)} anchor(s): " +
           ", ".join(f"{n}={s:.2f}m" for s, n in ranked[:4]))
    if missed:
        why += ("; NOT compared: " +
                ", ".join(f"{n[:40]} (az {a:+.0f}°, el {e:+.0f}°)"
                          for a, e, n in missed))
    return Resolved(pick[0], pick[1], why, complete=not missed, missed=missed)


def triangulate(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray,
                *, sigma_deg: float = 1.0) -> dict:
    """Range to a target seen from two poses, from bearings alone.

    No lidar, no model estimate — just the two directions and the baseline the
    robot drove, which `/state_estimation` reports well. Error goes as
    r²·σθ/b_perp, so at 6 m over a 2 m baseline a 1° bearing gives ~0.3 m,
    against 3.7 m for the lift that lost the folding screen.

    Degenerate when the baseline is along the line of sight, which is exactly
    what driving straight at the target produces — hence `parallax_deg`, which
    the caller must check before believing `range_m`.
    """
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    w = o1 - o2
    b = float(d1 @ d2)
    den = 1.0 - b * b
    parallax = float(np.rad2deg(np.arccos(np.clip(abs(b), -1.0, 1.0))))
    if den < 1e-9:
        return {"range_m": None, "parallax_deg": parallax,
                "reason": "baseline parallel to line of sight"}

    d, e = float(d1 @ w), float(d2 @ w)
    t1 = (b * e - d) / den
    t2 = (e - b * d) / den
    p1, p2 = o1 + t1 * d1, o2 + t2 * d2

    base = o2 - o1
    b_perp = float(np.linalg.norm(np.cross(base, d1)))
    sigma = (t1 ** 2 * np.deg2rad(sigma_deg) / b_perp) if b_perp > 1e-6 else float("inf")
    return {"range_m": float(t1), "point": (p1 + p2) / 2.0,
            "miss_m": float(np.linalg.norm(p1 - p2)),
            "parallax_deg": parallax, "baseline_perp_m": b_perp,
            "sigma_m": float(sigma), "reason": "ok"}


def load_snapshot(d: Path) -> tuple[dict, np.ndarray]:
    return (json.loads((d / "pose.json").read_text()), np.load(d / "scan.npy"))


def crop_face(face_jpeg: bytes, box_px, pad: float = 0.25) -> bytes:
    """The boxed region, padded, as a JPEG — the "previous view" of the target.

    Padding matters: a tight crop shows the object with none of what surrounds
    it, and the neighbours are most of what makes one instance of a type
    identifiable as the same one after the robot has moved.
    """
    import cv2

    img = cv2.imdecode(np.frombuffer(face_jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    ymin, xmin, ymax, xmax = box_px
    py, px_ = (ymax - ymin) * pad, (xmax - xmin) * pad
    y0, y1 = int(max(0, ymin - py)), int(min(h, ymax + py))
    x0, x1 = int(max(0, xmin - px_)), int(min(w, xmax + px_))
    if y1 - y0 < 8 or x1 - x0 < 8:
        y0, y1, x0, x1 = 0, h, 0, w
    return cv2.imencode(".jpg", img[y0:y1, x0:x1],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def ground(snapshot: Path, phrase: str, backend: str, model: str,
           prev_face: bytes | None = None) -> tuple[dict | None, list[bytes]]:
    fn = ask_claude if backend == "claude" else ask_gemini
    faces = load_faces(snapshot)
    reply = parse(fn(build_prompt(phrase, approach=True), faces, model,
                     previous=prev_face))
    return reply, faces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+",
                    help="one snapshot, or two to triangulate across the move")
    ap.add_argument("phrase")
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sigma-deg", type=float, default=1.0,
                    help="assumed bearing error, for the triangulation budget")
    args = ap.parse_args()

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else "gemini-2.5-flash")
    seen, prev_crop = [], None
    for s in args.snapshots:
        d = Path(s)
        pose, scan = load_snapshot(d)
        reply, faces = ground(d, args.phrase, args.backend, model, prev_crop)
        if reply is None:
            print(f"{s}: unparseable reply", file=sys.stderr)
            return 1
        print(f"\n=== {s} — pose {np.round(pose['position'], 2)}")
        if not reply.get("visible"):
            e = reply.get("explore") or {}
            print(f"  NOT_VISIBLE — explore heading {e.get('heading_deg')}°")
            print(f"  {reply.get('evidence', '')}")
            continue

        i = int(reply["image_index"])
        px = to_pixels(reply.get("feature_box_2d") or reply["box_2d"],
                       reply.get("coord_space"), G.FACE_SIZE)
        prev_crop = crop_face(faces[i], px)
        w, h = box_angular_size(px, i)
        print(f"  image {i} ({NAMES[i]}), conf {reply.get('confidence')}, "
              f"box {w:.1f}° x {h:.1f}°")
        print(f"  state={reply.get('target_state')} "
              f"same_as_prev={reply.get('same_object_as_previous')} "
              f"occlusion={reply.get('occlusion')}")

        wp = next_waypoint(px, i, scan, pose, phrase=args.phrase)
        tag = "DESTINATION" if wp.committed else "step, re-observe"
        print(f"  -> ({wp.xy[0]:+.2f}, {wp.xy[1]:+.2f})  [{tag}]")
        print(f"     {wp.reason}")

        o = np.asarray(pose["position"], float)
        seen.append((o, cam_dir_to_map(ray_from_box(px, i), pose)))

    if len(seen) >= 2:
        (o1, d1), (o2, d2) = seen[0], seen[1]
        t = triangulate(o1, d1, o2, d2, sigma_deg=args.sigma_deg)
        print(f"\n=== triangulated across a "
              f"{np.linalg.norm(o2 - o1):.2f} m baseline")
        if t["range_m"] is None:
            print(f"  degenerate: {t['reason']} (parallax {t['parallax_deg']:.2f}°)")
        else:
            print(f"  range from view 1  {t['range_m']:.2f} m  "
                  f"± {t['sigma_m']:.2f} m at σθ={args.sigma_deg}°")
            print(f"  parallax {t['parallax_deg']:.2f}°, "
                  f"perpendicular baseline {t['baseline_perp_m']:.2f} m, "
                  f"rays miss by {t['miss_m']:.2f} m")
            p = t["point"]
            print(f"  point ({p[0]:+.2f}, {p[1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
