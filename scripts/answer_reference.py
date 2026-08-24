#!/usr/bin/env python3
"""Answer an object-reference question: drive to the object, orbit it, publish a box.

The challenge scores this by 3D IoU against the object's ground-truth box --
2 points at IoU >= 0.5, 1 at >= 0.25, nothing below -- and the objects are
small. Over the 30 released questions the target's longest side has a median of
0.45 m and 17 of 30 are under half a metre, so for a perfectly sized box the
centre error that still earns a point has a median budget of **0.10 m**, and
0.05 m for two points. TASK 26 measured a single grounding call at a median of
0.11 m from the start pose with a p90 of 0.52 m: the median sits exactly on the
one-point line and the p90 earns nothing on any of the thirty. That is the
whole argument for spending the budget on driving there and looking again
rather than answering from where the vehicle happens to start.

The shape, and which part is new:

  1. `run_goto`      drive to the object            -- unchanged, TASK 26's path
  2. orbit           re-look from several angles    -- `reference_view.look`
  3. `TargetBox`     one box from the views         -- the perception estimator
  4. publish         centre + size on the marker    -- the caller's job

Only step 2 is new. Step 1 is the grounding loop as it stands, and step 3 is
`ObjectMap`'s measured estimator with its label-keyed association removed.

Timing is not the constraint. TASK 53 measured 82 legs at zero timeouts and 55%
of the budget on average, which puts one leg at roughly 150 s of the 540 s --
leaving room for about eleven views where this spends five.

    uv run --with anthropic python scripts/answer_reference.py \\
        "Find the pillow closest to the book on the stool."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "perception", REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from answer_numerical import anchor_offset, go, orbit_point  # noqa: E402
from approach_loop import Ctx, Robot, run_goto  # noqa: E402
from faces import faces_of  # noqa: E402
from reference_view import MAX_VIEWS, look as re_look, view_weight  # noqa: E402
from target_box import TargetBox, lift_box, pca_heading  # noqa: E402
from vlm_approach import crop_face  # noqa: E402
from waypoint_converter_model import ConverterModel  # noqa: E402

# Where to stand while looking. Nearer than the numerical task's 2.6 m: there
# the vehicle frames a *set* spread along a sofa, here it frames one object
# whose median longest side is 0.45 m, and the centre budget is what the range
# buys. Returns on a target grow as 1/r^2.
VIEW_M = 1.8
VIEW_MIN_M = 1.2

# How far round the object each orbit step goes. Larger than the numerical
# task's 55 deg because the point here is a genuinely different face of the
# object -- a box seen only from the front has no depth -- and there is no set
# to keep in frame.
ORBIT_DEG = 75.0

# Stop early once the box stops moving. `settled` compares the box against the
# box one view ago; 0.9 is tighter than the estimator's default because a
# reference answer is scored on this box and nothing else.
SETTLE_IOU = 0.9

# Refuse to publish a box larger than this on any axis. Not a real object at
# this task's scale, and a box that big is a mask that ran onto a wall. The
# largest of the 30 released targets is 1.22 m.
MAX_EXTENT_M = 3.0


def phrase_of(question: str) -> str:
    """The referring expression, with the imperative stripped.

    *"Find the pillow closest to the book on the stool."* names its target in
    the same shape the grounding prompt already takes; only the instruction
    wrapper has to come off. Two of the thirty released questions have no
    "Find" at all, so this must not depend on finding one.
    """
    s = question.strip().rstrip(".")
    low = s.lower()
    for lead in ("find ", "locate ", "identify ", "point to ", "point at "):
        if low.startswith(lead):
            return s[len(lead):].strip()
    return s


def _bearing_deg(pose: dict, target_xy: np.ndarray | None) -> float | None:
    """Where round the object this view was taken, in map degrees.

    Recorded per view so an orbit driven at one step size can be replayed at a
    coarser one: a policy that takes every second view is a subset of a denser
    drive, and the bearing is what identifies the subset.
    """
    if target_xy is None:
        return None
    v = np.asarray(pose["position"], float)[:2] - np.asarray(target_xy, float)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _range_m(pose: dict, target_xy: np.ndarray | None) -> float | None:
    if target_xy is None:
        return None
    return float(np.linalg.norm(np.asarray(pose["position"], float)[:2]
                                - np.asarray(target_xy, float)))


def stand_off(ctx: Ctx, target_xy: np.ndarray, *, want: float = VIEW_M) -> None:
    """Back off to `want` metres if the approach parked closer than that.

    Only ever backs off. `run_goto` stops as close as the platform allows, and
    from 0.6 m a 0.45 m object fills the frame and the scanner sees one face of
    it. Being too far is not a failure mode the approach produces.
    """
    eq, scan, terrain, pose = ctx.robot.capture()
    here = np.asarray(pose["position"], float)[:2]
    gap = float(np.linalg.norm(here - target_xy))
    if gap >= VIEW_MIN_M:
        return
    print(f"  {gap:.2f} m from the binding — backing off to {want:.1f} m")
    go(ctx, ConverterModel(terrain), pose,
       anchor_offset(pose, target_xy, want), why="standoff")


def one_view(ctx: Ctx, phrase: str, crop: bytes | None) -> dict:
    """Capture, re-look, lift. Returns everything the record needs.

    The reply and the lift are kept together because neither is worth much
    alone: a confident box that lifted nothing tells us to move, and a box the
    model refused tells us not to trust the returns under it.
    """
    eq, scan, terrain, pose = ctx.robot.capture()
    faces = faces_of(eq)
    ctx.step += 1
    for i, raw in enumerate(faces):
        (ctx.out / f"step{ctx.step}_face{i}.jpg").write_bytes(raw)
    np.save(ctx.out / f"step{ctx.step}_scan.npy", scan)

    reply = re_look(faces, phrase, crop=crop, model=ctx.model)
    if not reply.get("error"):
        ctx.calls += 1

    lift = {"n": 0, "points": None, "xyz": None, "thin": False, "mask_px": 0}
    if reply.get("found") and reply.get("box_px") is not None:
        lift = lift_box(reply["box_px"], int(reply["image_index"]), scan, pose)
        if lift["points"] is not None:
            # The crop for the *next* view comes from this one, so the
            # reference image is always the most recent confirmed sighting
            # rather than the one the approach ended on.
            reply["crop"] = crop_face(faces[int(reply["image_index"])],
                                      reply["box_px"])
    return {"reply": reply, "lift": lift, "pose": pose, "scan": scan,
            "terrain": terrain}


def answer_reference(ctx: Ctx, question: str, *, max_views: int = MAX_VIEWS,
                     size_mode: str = "average",
                     use_extent: bool = True,
                     orbit_deg: float = ORBIT_DEG,
                     settle: bool = True) -> dict:
    """The 3D box, and everything needed to argue with it.

    `settle=False` keeps looking until `max_views` even after the box has
    stopped moving, and is how a **corpus** drive differs from an answering
    one. Driving is the only part of this that cannot be replayed: the views
    are a function of where the policy chose to stand, so a change to the
    standoff or the orbit step needs the simulator again. What *can* be
    replayed from a recorded run is everything downstream of the viewpoint --
    the mask, the lift, the estimator, the stopping rule -- because the faces
    and the scan are on disk and the model's own box is in `steps.jsonl`.

    So the first drives should be supersets: more views than any policy would
    take, at a finer orbit step than any policy would use. A policy that stops
    at three views, or turns 75 deg instead of 40, is then a *subset* of what
    was driven and costs nothing more to evaluate. The extra cost is model
    calls, not simulator time.
    """
    phrase = phrase_of(question)
    print(f"  target {phrase!r}")

    out = run_goto(ctx, phrase, max_steps=6)
    target_xy = None if out.xy is None else np.asarray(out.xy, float)[:2]
    print(f"  approach: {out.why}"
          + ("" if target_xy is None
             else f" at ({target_xy[0]:+.2f}, {target_xy[1]:+.2f})"))

    if target_xy is None:
        # Nothing was bound, so there is no anchor and no standoff to take.
        # The orbit still runs: a re-look from here may find what the approach
        # could not, and an unanchored `TargetBox` falls back to the heaviest
        # cluster. It is the weaker answer and the record says so.
        print("  no binding — looking from here, unanchored")

    box = TargetBox(anchor=target_xy, size_mode=size_mode)
    crop = out.prev_crop
    if target_xy is not None:
        stand_off(ctx, target_xy)

    for i in range(max_views):
        if ctx.out_of_time():
            print("  out of time; committing the box so far")
            break

        v = one_view(ctx, phrase, crop)
        reply, lift, pose = v["reply"], v["lift"], v["pose"]
        n = int(lift["n"])
        took = False
        if reply.get("found") and lift["points"] is not None:
            took = box.add(lift["points"],
                           weight=view_weight(reply, n, use_extent=use_extent),
                           note=f"step{ctx.step}")
            crop = reply.get("crop") or crop

        print(f"  view {i + 1}: "
              + (f"{reply.get('extent')}, {n} returns"
                 if reply.get("found") else
                 f"not found ({reply.get('why_not') or reply.get('error')})")
              + (f"   {'kept' if took else 'no box'}")
              + (f"   thin mask ({lift['mask_px']} px)" if lift.get("thin") else "")
              + f"   {reply.get('evidence') or ''}"[:90])

        # `box_px` and `image_index` are recorded because without them a run
        # cannot be re-lifted offline: the faces and the scan are on disk, so a
        # different mask (a SAM refinement of the same box), a different
        # `min_inliers`, or a different percentile can all be replayed for free
        # -- but only if the box the model actually returned is here. Leaving
        # them out would make every such experiment a fresh drive.
        ctx.record({"step": ctx.step, "kind": "reference", "pose": pose,
                    "question": question, "phrase": phrase,
                    "found": bool(reply.get("found")),
                    "same_object": reply.get("same_object"),
                    "extent": reply.get("extent"),
                    "why_not": reply.get("why_not"),
                    "confidence": reply.get("confidence"),
                    "evidence": reply.get("evidence"),
                    "box_px": reply.get("box_px"),
                    "image_index": reply.get("image_index"),
                    "coord_space_used": reply.get("coord_space_used"),
                    "n_returns": n, "mask_px": lift.get("mask_px"),
                    "thin": lift.get("thin"), "took": took,
                    "weight": (view_weight(reply, n, use_extent=use_extent)
                               if took else 0.0),
                    "orbit_deg": _bearing_deg(pose, target_xy),
                    "range_m": _range_m(pose, target_xy),
                    "error": reply.get("error")})

        if settle and box.n_views and box.settled(iou=SETTLE_IOU):
            print("  box has stopped moving; committing")
            break
        if i == max_views - 1:
            break
        if target_xy is None:
            print("  nowhere to orbit around; committing")
            break
        aim = orbit_point(pose, target_xy, orbit_deg, VIEW_M)
        if not go(ctx, ConverterModel(v["terrain"]), pose, aim, why="orbit"):
            print("  cannot orbit further; committing")
            break

    # The stack holds the last waypoint for ever, so park before publishing:
    # instruction following is not what is scored here, but a vehicle still
    # driving at a stale goal keeps moving while the marker is read.
    if hasattr(ctx.robot, "stop"):
        parked = ctx.robot.stop()
        if not parked.get("ok"):
            print(f"  could not park the vehicle: {parked.get('why')}")

    return commit(box, question, phrase, target_xy, ctx,
                  policy={"standoff_m": VIEW_M, "orbit_deg": orbit_deg,
                          "max_views": max_views, "settle": settle,
                          "settle_iou": SETTLE_IOU if settle else None,
                          "size_mode": size_mode,
                          "extent_weights": use_extent})


def commit(box: TargetBox, question: str, phrase: str,
           target_xy: np.ndarray | None, ctx: Ctx,
           policy: dict | None = None) -> dict:
    """Turn the accumulated views into the answer, or say why there is none."""
    centre, size = box.box()
    keep = box.consensus()
    ok = bool(keep) and float(np.max(size)) > 0.0
    why = ""
    if not keep:
        why = "no view produced a liftable box"
    elif float(np.max(size)) > MAX_EXTENT_M:
        # A box this big is a mask that ran onto a wall, not an object. Better
        # to say so than to publish it: a 3 m box scores zero against a 0.45 m
        # target anyway, and a recorded refusal is something to fix.
        ok, why = False, f"box {float(np.max(size)):.1f} m on an axis — mask spill"

    got = box.as_dict()
    got.update(question=question, phrase=phrase, ok=ok, why=why,
               calls=ctx.calls, steps=ctx.step,
               binding=None if target_xy is None else target_xy.tolist(),
               anchored=target_xy is not None,
               heading=0.0,
               # The policy this run was driven under. A replay can only ever
               # evaluate *subsets* of it -- the viewpoints are the one thing
               # that cannot be recomputed -- so it has to travel with the data.
               policy=policy or {})
    if ok and keep:
        pts = np.array([box.centres[i] for i in keep])
        # Only with three or more agreeing views is a PCA axis a direction
        # rather than a line through two points. Reported, never published
        # without an A/B -- our own scorer builds ground truth with the heading
        # ignored, and the organisers' is not public.
        got["heading"] = float(pca_heading(pts)) if len(pts) >= 3 else 0.0

    if ok:
        print(f"\n  BOX centre=({centre[0]:+.2f}, {centre[1]:+.2f}, {centre[2]:+.2f})"
              f"  size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
              f"  from {len(keep)}/{box.n_views} views")
    else:
        print(f"\n  NO BOX — {why}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--host", default=os.environ.get("XIAO_HEI_SIM_HOST", "xiaohei1"),
                    help="'local' if this machine is the sim host")
    ap.add_argument("--out", default=None)
    ap.add_argument("--views", type=int, default=MAX_VIEWS)
    ap.add_argument("--size-mode", default="average",
                    choices=("average", "max", "union"),
                    help="how per-view extents combine; see target_box")
    ap.add_argument("--no-extent-weights", action="store_true",
                    help="ignore the model's `extent` when weighting a view")
    ap.add_argument("--orbit-deg", type=float, default=ORBIT_DEG,
                    help="degrees round the object per step")
    ap.add_argument("--corpus", action="store_true",
                    help="a superset drive: keep looking to --views even after "
                         "the box settles, so coarser policies replay as "
                         "subsets. Costs model calls, not simulator time")
    ap.add_argument("--model", default=os.environ.get("XIAO_HEI_CLAUDE_MODEL",
                                                      "claude-opus-5"))
    ap.add_argument("--budget", type=float, default=540.0)
    a = ap.parse_args()

    out = Path(a.out or f"runs/ref_{time.strftime('%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    bot = Robot(None if a.host == "local" else a.host)
    bot.push()
    ctx = Ctx(robot=bot, out=out, log=(out / "steps.jsonl").open("w"),
              model=a.model, deadline=time.time() + a.budget)
    ctx.note_settings()
    print(f"question: {a.question}\nrun: {out}")
    try:
        got = answer_reference(ctx, a.question, max_views=a.views,
                               size_mode=a.size_mode,
                               use_extent=not a.no_extent_weights,
                               orbit_deg=a.orbit_deg,
                               settle=not a.corpus)
    finally:
        ctx.close()
    (out / "answer.json").write_text(json.dumps(got, indent=1, default=str))
    print(f"\n{out}/answer.json")
    return 0 if got.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
