#!/usr/bin/env python3
"""Answer a numerical question: find the anchor, look at it, count, commit.

Assembled almost entirely from parts that were already driving. `run_goto`
finds the furniture the question is about, `ConverterModel` decides where the
vehicle may stand, `explore_direction` turns a heading into a drivable one, and
the lidar lift turns each counted box into a position. What is new is the stop
rule -- a counter wants the anchor *framed*, not touched -- and the arithmetic
that turns several views into one integer.

The count is the number of distinct positions, never the sum of the views. Two
looks at a sofa with four pillows must not answer eight, and no amount of
prompting makes a model reliable about that across calls it cannot remember, so
it is not asked to be: each look is blind (see `count_view`), and the merge
happens here, on metres.

    uv run --with anthropic python scripts/answer_numerical.py \\
        "How many pillows are on the bed?" --host xiaohei1
    uv run python scripts/answer_numerical.py "..." --plan-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from approach_loop import (Ctx, MIN_VIEW_MOVE_M, Robot,  # noqa: E402
                           explore_direction, explore_goal, lift_boxed,
                           nearest_allowed_step, run_goto, yaw_of)
from count_view import (MAX_LOOKS, MERGE_M, count_view,  # noqa: E402
                        lift_instances, merge, tally)
from faces import faces_of  # noqa: E402
from numerical_plan import phrase_for, plan as parse_question  # noqa: E402
from waypoint_converter_model import ConverterModel  # noqa: E402

# Where to stand to count. Not a comfort setting: the approach loop parks as
# close as the platform allows, which measured 1.1-1.5 m to an object centre,
# and from there a sofa runs past both edges of a 100 deg face. A metre and a
# half further back puts a 2.2 m sofa inside one face with room to spare, and
# it is still close enough to tell a pillow from a cushion.
VIEW_M = 2.6
# Nearer than this and the anchor is overflowing the frame, so back off before
# spending a call to be told so.
VIEW_MIN_M = 1.9
# Further than this and the model is being asked to guess. `count_audit`
# reports accuracy against range, and this is the first thing to re-derive from
# it.
VIEW_MAX_M = 4.5
# How far round the anchor one "orbit" step goes. Big enough that the far side
# of a sofa comes into view, small enough that what was counted stays in frame
# and can be re-merged rather than re-counted.
ORBIT_DEG = 55.0
# How much of a 100 deg face the counted set should fill once framed. Leaves a
# 15 deg margin either side, which is what stops an instance at the edge of the
# spread landing in the perspective crop's worst distortion -- or in the next
# face, where the same object gets described twice.
FRAME_DEG = 70.0
# Two placed instances are the minimum that defines a direction; one is a point
# and has no axis to stand square to.
FRAME_MIN_N = 2


def anchor_offset(pose: dict, anchor: np.ndarray, want: float) -> np.ndarray:
    """A point `want` metres from the anchor, on the side we already stand."""
    here = np.asarray(pose["position"], float)[:2]
    v = here - np.asarray(anchor, float)[:2]
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        v, d = np.array([1.0, 0.0]), 1.0
    return np.asarray(anchor, float)[:2] + v / d * want


def orbit_point(pose: dict, anchor: np.ndarray, deg: float,
                want: float) -> np.ndarray:
    """The same standoff, rotated `deg` around the anchor."""
    here = np.asarray(pose["position"], float)[:2]
    v = here - np.asarray(anchor, float)[:2]
    d = float(np.linalg.norm(v)) or 1.0
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.asarray(anchor, float)[:2] + (R @ (v / d)) * want


def frame_pose(seen: list[dict], pose: dict, *,
               want_deg: float = FRAME_DEG) -> np.ndarray | None:
    """Where to stand so the whole counted set fits one face at once.

    The standoff that works for a bed does not work for a 2.9 m desk, and the
    difference is not the anchor's size -- it is how far apart the things being
    counted are. After one look we know that exactly: every instance the
    scanner placed is a map-frame point, so the set has a centroid, a long
    axis, and an extent.

    Stand square to the long axis, far enough back that the extent subtends
    `want_deg`. `office_1` is what this is for. Its six monitors run 2.00 m
    along a desk -- 42 deg from 2.6 m, comfortably inside one face -- but the
    approach loop parked at the desk's far corner, 1.7 m off its axis, where
    the near monitors hide the rest. Four looks all reported `too_far` or
    `occluded` and the run answered 3. This rule sends the vehicle to
    (+2.76, -3.73), which is within 6 cm of the pose it had already driven
    through on its way to the corner.

    `None` when there is not enough to compute one -- fewer than two placed
    instances, or a set with no extent -- and the caller keeps its old
    behaviour rather than inventing an axis.
    """
    P = np.array([o["xy"] for o in seen if o.get("xy") is not None], float)
    if len(P) < FRAME_MIN_N:
        return None
    c = P.mean(axis=0)
    # The long axis of the set, by SVD rather than by pairing up extremes: with
    # three points a max-pair axis swings 90 deg on one outlier.
    axis = np.linalg.svd(P - c, full_matrices=False)[2][0]
    proj = (P - c) @ axis
    spread = float(proj.max() - proj.min())
    normal = np.array([-axis[1], axis[0]])
    want = max(VIEW_MIN_M,
               min(VIEW_MAX_M,
                   (spread / 2) / np.tan(np.deg2rad(want_deg / 2))))
    here = np.asarray(pose["position"], float)[:2]
    # Stay on the side the vehicle is already on. The other side is as good
    # geometrically and may be through a wall.
    side = float(np.sign(np.dot(here - c, normal))) or 1.0
    return c + normal * side * want


def go(ctx: Ctx, cm: ConverterModel, pose: dict, aim: np.ndarray,
       *, why: str, min_move: float = MIN_VIEW_MOVE_M) -> dict | None:
    """Drive at `aim` by whatever the converter will actually accept."""
    here = np.asarray(pose["position"], float)[:2]
    best = cm.best_waypoint_toward(np.asarray(aim, float)[:2], here,
                                   min_move=min_move)
    goal = best[0] if best else nearest_allowed_step(cm, here, aim)
    if goal is None:
        print(f"      no legal waypoint toward {why}")
        return None
    print(f"      {why}: ({goal[0]:+.2f}, {goal[1]:+.2f})")
    return ctx.robot.drive_to(float(goal[0]), float(goal[1]),
                              timeout=min(45.0, ctx.left()))


def reposition(ctx: Ctx, cm: ConverterModel, pose: dict, reply: dict,
               anchor: np.ndarray | None, scan: np.ndarray, *,
               seen: list[dict] | None = None, framed: bool = False) -> bool:
    """Take the next view. True if the vehicle moved.

    The model chooses the *kind* of move and the geometry chooses the point,
    which is the same split as everywhere else here. A boxed place is taken
    literally when the scanner can measure it; a bare heading goes through
    `explore_direction` so the terrain keeps its veto.

    One case overrules the model, and only once per run: when it says the view
    is too far or something is in the way, and the instances already placed
    define a framing pose that is a real move from here, we go there instead.
    The model is reporting a symptom it cannot locate -- it has no map and
    cannot know it is standing at the end of the desk rather than in front of
    it -- while `frame_pose` computes exactly that from positions the lidar has
    already measured. Asking for "closer" from the end of a desk gets a refusal
    from the converter and a shuffle; standing square to it gets the answer.
    """
    nv = reply.get("next_view") if isinstance(reply.get("next_view"), dict) else {}
    kind = str(nv.get("kind") or "move")
    here = np.asarray(pose["position"], float)[:2]

    if not framed and reply.get("why_not") in ("too_far", "occluded",
                                               "anchor_cut_off"):
        aim = frame_pose(seen or [], pose)
        if aim is not None and float(np.linalg.norm(aim - here)) > MIN_VIEW_MOVE_M:
            if go(ctx, cm, pose, aim, why=f"framing the set ({reply['why_not']})"):
                return True

    if anchor is not None and kind in ("closer", "orbit"):
        want = VIEW_M
        if kind == "closer":
            gap = float(np.linalg.norm(here - anchor))
            want = max(VIEW_MIN_M, min(gap - 0.8, VIEW_M))
            aim = anchor_offset(pose, anchor, want)
        else:
            aim = orbit_point(pose, anchor, ORBIT_DEG, want)
        return bool(go(ctx, cm, pose, aim, why=kind))

    # A boxed place: the model can see where to go, so go there rather than in
    # its general direction. `lift_boxed` caps the range, so a box on something
    # far away becomes a step along its bearing.
    if nv.get("box_2d") is not None and nv.get("image_index") is not None:
        xy = lift_boxed({"next_view": nv, "coord_space": reply.get("coord_space")},
                        "next_view", scan, pose)
        if xy is not None:
            return bool(go(ctx, cm, pose, xy, why="boxed next view"))

    heading = nv.get("heading_deg")
    if heading is None:
        return False
    u, reach, off = explore_direction(cm, here, float(heading), ctx.spent)
    ctx.spent.append((here, u))
    return bool(go(ctx, cm, pose, here + u * min(reach, 3.0),
                   why=f"heading {float(heading):.0f} deg ({off:+.0f} off, "
                       f"{reach:.1f} m reach)"))


def look(ctx: Ctx, question: str, p: dict) -> tuple[dict, list[dict], dict,
                                                    np.ndarray, np.ndarray]:
    """One capture and one blind counting call."""
    eq, scan, terrain, pose = ctx.robot.capture()
    faces = faces_of(eq)
    ctx.step += 1
    for i, raw in enumerate(faces):
        (ctx.out / f"step{ctx.step}_face{i}.jpg").write_bytes(raw)
    np.save(ctx.out / f"step{ctx.step}_scan.npy", scan)
    reply = count_view(faces, question, p["target"], p.get("anchor"),
                       model=ctx.model)
    if not reply.get("error"):
        ctx.calls += 1
    return reply, lift_instances(reply, scan, pose), pose, scan, terrain


def answer_numerical(ctx: Ctx, question: str, *, p: dict | None = None,
                     max_looks: int = MAX_LOOKS) -> dict:
    """The integer, and everything needed to argue with it."""
    p = p or parse_question(question, model=ctx.model)
    ctx.calls += 1 if p.get("from_model") else 0
    print(f"  counting {p['target']!r}"
          + (f" {p['relation']} {p['anchor']!r}" if p.get("anchor") else "")
          + (f", {p['attribute']} only" if p.get("attribute") else "")
          + f"   scope {p['scope']}")

    anchor_xy = None
    if p["scope"] == "anchor" and p.get("anchor"):
        out = run_goto(ctx, phrase_for(p), max_steps=6)
        anchor_xy = None if out.xy is None else np.asarray(out.xy, float)[:2]
        print(f"  anchor: {out.why}"
              + ("" if anchor_xy is None
                 else f" at ({anchor_xy[0]:+.2f}, {anchor_xy[1]:+.2f})"))

    views: list[list[dict]] = []
    replies: list[dict] = []
    # The running merged set, kept as the looks come in rather than only at the
    # end, because `frame_pose` needs it to decide where to stand next.
    seen: list[dict] = []
    backed_off = False          # the one pre-look standoff, from the binding
    framed_set = False          # the one geometric override, from the instances
    for i in range(max_looks):
        if ctx.out_of_time():
            print("  out of time; committing what has been counted")
            break

        # Frame before the first look rather than paying a call to be told the
        # sofa runs off the edge. Only ever backs *off*: the approach loop has
        # just parked as close as the platform allows, and being too far is not
        # a failure mode it produces.
        if anchor_xy is not None and not backed_off:
            backed_off = True
            eq, scan, terrain, pose = ctx.robot.capture()
            gap = float(np.linalg.norm(
                np.asarray(pose["position"], float)[:2] - anchor_xy))
            if gap < VIEW_MIN_M:
                print(f"  {gap:.2f} m from the anchor — backing off to {VIEW_M:.1f}")
                go(ctx, ConverterModel(terrain), pose,
                   anchor_offset(pose, anchor_xy, VIEW_M), why="framing")

        reply, lifted, pose, scan, terrain = look(ctx, question, p)
        replies.append(reply)
        rec = {"step": ctx.step, "kind": "count", "pose": pose,
               "question": question, "target": p["target"],
               "anchor": p.get("anchor"),
               "count_here": reply.get("count_here"),
               "sufficient": reply.get("sufficient"),
               "why_not": reply.get("why_not"),
               "confidence": reply.get("confidence"),
               "evidence": reply.get("evidence"),
               "placed": sum(1 for x in lifted if x["xy"] is not None),
               "error": reply.get("error")}
        ctx.record(rec)
        print(f"  look {i + 1}: {reply.get('count_here')} here, "
              f"{'enough' if reply.get('sufficient') else 'not enough'}"
              f"{'' if reply.get('sufficient') else ' (' + str(reply.get('why_not')) + ')'}"
              f"   {reply.get('evidence') or reply.get('error') or ''}")
        if not reply.get("error"):
            views.append(lifted)
            seen = merge(seen, lifted)

        if reply.get("sufficient") and not reply.get("error"):
            break
        if i == max_looks - 1:
            print("  out of looks; committing what has been counted")
            break
        was = framed_set
        if not framed_set and reply.get("why_not") in ("too_far", "occluded",
                                                       "anchor_cut_off") \
                and frame_pose(seen, pose) is not None:
            framed_set = True          # spent whether or not the drive lands
        if not reposition(ctx, ConverterModel(terrain), pose, reply,
                          anchor_xy, scan, seen=seen, framed=was):
            print("  nowhere further to look; committing")
            break

    # The stack holds the last waypoint for ever. Ending a question without
    # retracting it leaves the vehicle driving at a goal nobody is waiting for
    # -- and if that goal is unreachable, oscillating in place until the sim is
    # restarted. Park before reporting, not after: the answer is already
    # decided and the vehicle should stop moving while it is being published.
    if hasattr(ctx.robot, "stop"):
        parked = ctx.robot.stop()
        if not parked.get("ok"):
            print(f"  could not park the vehicle: {parked.get('why')}")

    t = tally(views, MERGE_M)
    t.update(question=question, plan=p, looks=len(views),
             sufficient=any(r.get("sufficient") for r in replies),
             calls=ctx.calls)
    print(f"\n  ANSWER {t['count']}   "
          f"({t['clusters']} placed + {t['unplaced']} unplaced, "
          f"per view {t['per_view']}, {t['corroborated']} seen twice)")
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--host", default=os.environ.get("XIAO_HEI_SIM_HOST", "xiaohei1"),
                    help="'local' if this machine is the sim host")
    ap.add_argument("--out", default=None)
    ap.add_argument("--looks", type=int, default=MAX_LOOKS)
    ap.add_argument("--budget", type=float, default=540.0)
    ap.add_argument("--model", default=os.environ.get("XIAO_HEI_MODEL", "claude-opus-5"))
    ap.add_argument("--plan-only", action="store_true",
                    help="split the question and stop; no robot, one call")
    args = ap.parse_args()

    if args.plan_only:
        p = parse_question(args.question, model=args.model)
        print(json.dumps(p, indent=1))
        print(f"\nanchor phrase: {phrase_for(p)!r}", file=sys.stderr)
        return 0

    out = Path(args.out or f"runs/num_{time.strftime('%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    bot = Robot(None if args.host == "local" else args.host)
    bot.push()
    ctx = Ctx(robot=bot, out=out, log=(out / "steps.jsonl").open("w"),
              model=args.model, deadline=time.time() + args.budget)
    ctx.note_settings()
    try:
        t = answer_numerical(ctx, args.question, max_looks=args.looks)
    finally:
        ctx.close()
    (out / "answer.json").write_text(json.dumps(t, indent=1, default=str))
    print(f"\n{out}/answer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
