#!/usr/bin/env python3
"""Drive a whole instruction: decompose it, then walk the clauses in order.

`approach_loop` drives to one object. 27 of the 30 official instruction
questions name two or more destinations that must be reached in order, and
README §175 scores the trajectory on whether it "follows the path constraints
in the command and in the correct order". This is the carrier for that: one
question in, an ordered drive out, with the vehicle never reset between legs.

Three things are deliberately kept apart.

**The model decomposes** (`decompose.py`), once, at step 0. It replaces a regex
whose ten clause openers were read off the same thirty sentences it was then
measured on; over those thirty the two agree 30/30 on drive order, and on a
phrasing neither had seen the regex collapses the sentence to a single
destination while the model gets it right (TASK 32).

**The executor holds the progress cursor.** It only ever moves forward. The
model is shown the whole question and told which step is current, because a leg
often cannot be read alone -- "the picture closest to the TV" needs the TV that
an earlier leg named -- but it is never asked which step to do next. That is
not a squeamish division: `bind_target` documents the model reporting
`same_object_as_previous: true` at higher confidence both when it had refined a
binding by 0.52 m and when it had jumped 4.37 m to a different object. Judging
what it can see is something it does well; adjudicating its own progress is the
one thing it has been measured failing at, and the score's largest penalty is
for getting the order wrong.

**Geometry decides positions.** A passage is driven by publishing a legal point
on the *far* side of the gap and letting `local_planner` thread it. There is
never a legal point inside the gap -- `obstacleDisThre` governs where a
waypoint may be placed, not where the vehicle may drive -- and a waypoint at
the midpoint just parks the vehicle on the near side. Whether the passage was
satisfied is then decided by the recorded track crossing the line between the
two anchors, not by how near the vehicle got.

    export ANTHROPIC_API_KEY=...
    uv run --with anthropic python scripts/execute_plan.py --host xiaohei1 \\
        "First, go to the vase closest to the easel, then, take the path \\
         between the couch and the table and stop at the window closest to \\
         the couch."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from approach_loop import (COST_PER_CALL, CTR, DEFAULT_HOST,  # noqa: E402
                           KEEPOUT_M, MIN_VIEW_MOVE_M, PROGRESS_M,
                           REVISIT_M, USE_KEEPOUT, VISITED_STYLE,
                           VISITED_STYLES,
                           Ctx, Outcome, Robot,
                           bind_constraints, crosses, explore_direction,
                           explore_goal, gates_from, ground, run_goto,
                           same_thing, yaw_of)
from decompose import decompose  # noqa: E402
from instruction_plan import GOTO, PASS, Clause, keepouts, steps  # noqa: E402
from vlm_approach import STANDOFF_M, _lift_xy  # noqa: E402
from vlm_locate import scan_to_camera  # noqa: E402
from vlm_probe import (DEFAULT_GEMINI_MODEL, DEFAULT_PROMPT_VER,  # noqa: E402
                       to_pixels)
from waypoint_converter_model import ConverterModel  # noqa: E402
from faces import faces_of  # noqa: E402

# README §Timing: 10 minutes per question, exploration included, and the clock
# starts at system launch rather than at the first waypoint. The default leaves
# a minute for the drive in flight to finish and for the run to be written out.
BUDGET_S = 540.0
# A passage is satisfied by driving through it, so it needs no standoff and no
# binding — only a point inside the gap and enough steps to find the anchors.
PASS_STEPS = 4
# How close to the gap point counts as having gone through it. `local_planner`
# aims for `waypointXYRadius` (0.3 m) and `vehicleWidth` is 0.5 m, so a metre
# either side of the midpoint is still inside a gap wide enough to drive.
GATE_REACHED_M = 1.0
# How far beyond a two-sided gap to place the waypoint, so that the shortest
# legal way to it is through the gap. Comfortably past `vehicleWidth` plus
# `waypointXYRadius`, and short enough to stay inside a room.
THROUGH_M = 1.2
# Held back for each leg still to come, so an early leg cannot spend the whole
# budget. Two calls and their drives, measured at 33 s per call-and-drive on
# `home_building_1` — enough for a later leg to ground a target it can already
# see, which is the cheap case every leg after the first one is in.
RESERVE_S = 70.0


def xy_of(pose) -> np.ndarray | None:
    """Planar position out of either pose shape `robot_io` returns.

    `capture` answers `{"position": [x, y, z], "orientation": [...]}` and
    `drive` answers a bare `[x, y, z]`. Both arrive here, and indexing the
    wrong one raises rather than misreading, so this is the one place that
    knows about the difference.
    """
    if pose is None:
        return None
    p = pose["position"] if isinstance(pose, dict) else pose
    return np.asarray(p, float)[:2]


def lift_anchors(reply: dict, scan: np.ndarray, pose: dict) -> list[dict]:
    """Map positions of the objects a passage or keep-out is anchored on.

    Read from `gate` and `avoid` together. Which field the model fills is its
    reading of the phrase, but the executor already knows from the plan which
    kind of clause this is, so taking both and letting the caller label them
    costs nothing and survives the model putting a passage under `avoid`.
    """
    scan_cam = scan_to_camera(scan, pose)
    space = reply.get("coord_space")
    out: list[dict] = []
    for it in (reply.get("gate") or []) + (reply.get("avoid") or []):
        if it.get("box_2d") is None or it.get("image_index") is None:
            continue
        xy = _lift_xy(to_pixels(it["box_2d"], space, G.FACE_SIZE),
                      int(it["image_index"]), scan_cam, pose)
        if xy is None:
            continue
        out.append({"xy": np.asarray(xy, float)[:2],
                    "name": it.get("name") or "?"})
    return out


def went_between(track: list, a: np.ndarray, b: np.ndarray) -> bool:
    """Did the driven path actually pass between the two anchors?

    This replaces a proximity test, which was wrong in a way no log showed. On
    `studio` the vehicle stopped 0.93 m from a gate point and the leg reported
    the passage satisfied; the recorded path then went round the west side of
    the table and never touched the couch-table line at all. Being near a gap
    is not going through it, and README §175 scores the trajectory.
    """
    if not track or len(track) < 2:
        return False
    p = np.asarray(track, float)[:, :2]
    return any(crosses(p[i], p[i + 1], a, b) for i in range(len(p) - 1))


def through_point(a: np.ndarray, b: np.ndarray, entry: np.ndarray,
                  reach: float = THROUGH_M) -> np.ndarray:
    """A waypoint on the FAR side of the gap, so driving to it goes through.

    Aiming at the midpoint asks the vehicle to stop in the doorway, and the
    stack obligingly settles on whichever side it approached from -- which is
    how `studio` ended up parked 1.4 m short of the gap with the leg calling
    itself done. A point beyond the gap has no such reading: the shortest legal
    way to it is through.

    `entry` is where the leg *first saw* the gap, not where the vehicle stands
    now, and the difference is the whole point. Recomputed live, "far" flips
    the instant the vehicle is past the midpoint -- so a crossing that
    `went_between` fails to register (the planner rounded the end of the
    segment, or an anchor lifted half a metre out) turns the next step's aim
    back the way it came, and the leg oscillates through the gap until it runs
    out of steps. Frozen with the gap, forward stays forward.
    """
    u = b - a
    n = np.linalg.norm(u)
    if n < 1e-6:
        return (a + b) / 2.0
    mid = (a + b) / 2.0
    perp = np.array([-u[1], u[0]]) / n
    away = perp if float(np.dot(perp, mid - entry)) > 0 else -perp
    return mid + away * reach


def far_side_goal(cm, sides: tuple[np.ndarray, np.ndarray],
                  vehicle: np.ndarray, aim: np.ndarray, *,
                  entry: np.ndarray | None = None,
                  corridor: float = 2.5):
    """The legal point beyond the gap that is nearest the aim.

    `best_waypoint_toward` minimises distance to the aim over every legal
    point, and on `studio` a near-side point was always closer to a far-side
    aim than any far-side point was: 424 legal points sat 1.4 m north of the
    gap and 4 sat 3.3 m south, so the converter picked north every time and the
    vehicle never crossed. Restricting the candidates to the far half-plane is
    what turns "aim through the gap" into "go through the gap".

    Nothing here asks for a legal point *inside* the gap. There is never one:
    `obstacleDisThre` (0.75 m) governs where a waypoint may be placed, not
    where the vehicle may drive, and `local_planner` threads a gap far narrower
    than 1.5 m using its own path library. The waypoint belongs on the far side
    and the threading is the stack's job.

    Two positions, and they are not the same one. `entry` says which half-plane
    counts as "beyond" and is frozen when the gap binds, for the reason
    `through_point`'s is: read from the live pose, "beyond" reverses the moment
    the vehicle is past the line, and the leg starts publishing waypoints back
    where it came from. `vehicle` is where it stands now, and only reaches
    `settle`, which is asking a different question — where this waypoint would
    actually put it, given where it is approaching from. It defaults to
    `vehicle` so a caller with no bound gap behaves as before.
    """
    L = cm.legal_points()
    if not len(L):
        return None
    entry = vehicle if entry is None else entry
    a, b = sides
    u = b - a
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        return None
    u = u / n
    mid = (a + b) / 2.0
    perp = np.array([-u[1], u[0]])
    s_veh = float(np.dot(perp, entry - mid))
    if abs(s_veh) < 1e-3:
        return None                      # already on the line; nothing is "far"
    beyond = ((L - mid) @ perp) * s_veh < 0
    within = np.abs((L - mid) @ u) < corridor
    cand = L[beyond & within]
    if not len(cand):
        return None
    goal = cand[int(np.argmin(np.linalg.norm(cand - aim, axis=1)))]
    return goal, cm.settle(goal, vehicle)


def far_side_stalled(far, here: np.ndarray,
                     reached: list[np.ndarray]) -> str | None:
    """Why this far-side waypoint is not progress toward the gap, or None.

    `far_side_goal` answering at all used to be taken as progress, and it is
    not. On `living_room_1` it answered every step with the same corner
    2 m north-west of the gate; the leg drove there, was handed it again as a
    0.20 m move, fell through to the unconstrained `best_waypoint_toward` --
    which prefers a near-side point, by the argument in `far_side_goal` -- and
    drove 1.5 m back. Then forward, then back. Four steps, never closer than
    1.37 m to a gate 2.14 m wide.

    Both readings mean the same thing: the converter has no *new* far-side
    point to offer, so standing somewhere else is the only way to learn
    anything, and if that has already been tried the gap is not drivable.
    """
    if far is None:
        return "no legal point beyond the gap yet"
    lands = np.asarray(far[1], float)
    if float(np.linalg.norm(lands - here)) < MIN_VIEW_MOVE_M:
        return "the best far-side waypoint is where it stands"
    if any(float(np.linalg.norm(lands - p)) < REVISIT_M for p in reached):
        return "the far-side waypoint is one already driven to"
    return None


def gate_point(anchors: list[dict], relation: str | None, origin: np.ndarray
               ) -> tuple[np.ndarray, str, tuple[np.ndarray, np.ndarray] | None] | None:
    """`(midpoint, why, sides)` for the passage, from the lifted anchors.

    `sides` is the pair the gap is between, and is what the crossing test uses;
    it is None for a one-landmark passage, which has no line to cross.

    Two anchors that lift to nearly the same place are not a gap — that is one
    object reported twice, and a midpoint on top of it would send the robot
    into it rather than past it.
    """
    if not anchors:
        return None
    if relation == "between" and len(anchors) >= 2:
        # The two furthest apart, in case a third thing was reported: a gap is
        # defined by its sides.
        pairs = [(a, b) for i, a in enumerate(anchors) for b in anchors[i + 1:]
                 if not same_thing(a["name"], b["name"])]
        if not pairs:
            return None
        pair = max(pairs, key=lambda p: float(
            np.linalg.norm(p[0]["xy"] - p[1]["xy"])))
        span = float(np.linalg.norm(pair[0]["xy"] - pair[1]["xy"]))
        if span < 0.5:
            return None
        mid = (pair[0]["xy"] + pair[1]["xy"]) / 2.0
        return mid, (f"between {pair[0]['name']!r} and {pair[1]['name']!r}, "
                     f"{span:.2f} m apart"), (pair[0]["xy"], pair[1]["xy"])
    # `near`: one landmark, and the passage runs alongside it. Aim at the
    # landmark and let the converter's obstacle inflation place the vehicle
    # beside it rather than on it — which is exactly the clearance the passage
    # wants, and is not a number this has to invent.
    a = min(anchors, key=lambda x: float(np.linalg.norm(x["xy"] - origin)))
    return a["xy"], f"alongside {a['name']!r}", None


def run_pass(ctx: Ctx, clause: Clause, k: int, *,
             max_steps: int = PASS_STEPS) -> Outcome:
    """Drive through a passage, rather than to an object.

    Unlike a destination this has no standoff and no binding to defend: the
    constraint is satisfied by the trajectory crossing the gap, so once the
    vehicle is within `GATE_REACHED_M` of the gap point the clause is done and
    the next one starts from there.
    """
    phrase = clause.text
    stalled = 0
    # Every pose the vehicle occupies during this leg, so the crossing test
    # sees the arc `local_planner` chose rather than the straight line between
    # the waypoints we published.
    track: list = []
    bound_gap = None
    # Where the vehicle stood when the gap resolved. Frozen alongside it, and
    # for the same reason: it is what "the far side" means for the rest of the
    # leg. See `through_point`.
    entry: np.ndarray | None = None
    # Where the vehicle has actually come to rest in this leg. A far-side
    # waypoint that lands on one of these is the leg being handed the same
    # answer twice; see the `why_not` block.
    reached: list[np.ndarray] = []
    no_far = 0

    def bank(why: str, mid, sides, where) -> None:
        """Hand what this passage achieved to the legs that come after it.

        A satisfied passage is not just a leg that ended: it is a scored
        constraint now standing behind the robot, and the next leg used to
        start with no knowledge of it whatever. Three things go forward. The
        gap itself, so `explore_direction` can recognise a bearing that would
        undo it. One departure, pointing back at the gap from where the robot
        came out, so that the very first exploration step of the next leg —
        the one measured driving 1.7 m back east through the dining-table gap
        in three `home_building_1` runs of four — is discounted. And a sentence
        for the prompt, because none of the geometry reaches the model.
        """
        ctx.done.append(f"drove the passage {phrase!r} — {why}")
        if sides is None or entry is None:
            return
        ctx.crossed.append((np.asarray(sides[0], float),
                            np.asarray(sides[1], float),
                            np.asarray(entry, float)))
        where = np.asarray(where, float)[:2]
        back = np.asarray(mid, float) - where
        n = float(np.linalg.norm(back))
        if n > 1e-6:
            ctx.spent.append((where.copy(), back / n))
    for _ in range(max_steps):
        if ctx.out_of_time():
            return Outcome(False, "out of time")
        ctx.step += 1
        step = ctx.step
        eq, scan, terrain, pose = ctx.robot.capture()
        faces = faces_of(eq)
        for i, f in enumerate(faces):
            (ctx.out / f"step{step}_face{i}.jpg").write_bytes(f)
        np.save(ctx.out / f"step{step}_terrain.npy", terrain)
        np.save(ctx.out / f"step{step}_scan.npy", scan)

        # Same reasoning as `run_goto`: no single call is worth the rest of the
        # question. See there.
        try:
            reply, raw = ground(faces, phrase, ctx.backend, ctx.model, None,
                                ctx.prompt_version, ctx.visited_for(pose),
                                ctx.mission_for(k),
                                visited_kind=ctx.prompt_visited_kind(),
                                ask_here=not ctx.drop_here)
        except Exception as e:
            print(f"[{step}] grounding call failed ({type(e).__name__}: {e}); "
                  f"ending this leg")
            ctx.record({"step": step, "clause": k, "phrase": phrase,
                        "kind": PASS, "pose": pose, "reply": None,
                        "error": f"{type(e).__name__}: {e}"})
            return Outcome(False, f"grounding call failed ({type(e).__name__})")
        ctx.calls += 1
        o = np.asarray(pose["position"], float)
        rec: dict = {"step": step, "clause": k, "phrase": phrase, "kind": PASS,
                     "relation": clause.relation, "anchors": list(clause.anchors),
                     "pose": pose, "reply": reply}
        if reply is None:
            print(f"[{step}] unparseable reply ({len(raw)} chars); stopping")
            rec["raw"] = raw
            ctx.record(rec)
            return Outcome(False, "unparseable reply")

        noted = ctx.note_sightings(reply, k, scan, pose)
        if noted:
            rec["sightings"] = noted
        here_txt = (reply.get("here") or "").strip()
        if here_txt or ctx.visited_style in ("bearing", "xy"):
            ctx.note_visit(o[:2], here_txt)
            if here_txt:
                rec["here"] = here_txt
        ctx.avoid = bind_constraints(reply, scan, pose, ctx.avoid)
        # A corridor the instruction forbids is a gate, not two discs. Discs
        # big enough to close it close the room as well — see `ConverterModel`.
        gates = (gates_from(ctx.avoid)
                 if (ctx.keepout_is_gate and USE_KEEPOUT) else [])
        keepout = ([] if gates or not USE_KEEPOUT else
                   [(a["xy"], KEEPOUT_M) for a in ctx.avoid])

        found = lift_anchors(reply, scan, pose)
        gp = gate_point(found, clause.relation, o[:2])
        print(f"[{step}] at ({o[0]:+.2f}, {o[1]:+.2f})  PASS {phrase!r}  "
              f"{len(found)} anchor(s) lifted")
        rec["lifted"] = [{"name": a["name"], "xy": a["xy"].tolist()}
                         for a in found]
        # Freeze the gap the first time it resolves. Re-deriving it from fresh
        # lifts every step let it wander across three different pairs on one
        # `studio` leg — `couch`+`table` 1.46 m apart, then `couch`+`coffee
        # table` 1.86 m apart somewhere else, then two views of the same couch.
        # The through-point flips with the gap, so the vehicle was sent a new
        # direction each step and crossed nothing. A destination binding is
        # defended for exactly this reason (`bind_target`); a passage needs the
        # same and had none.
        if gp is not None and gp[2] is not None and bound_gap is None:
            bound_gap, entry = gp, o[:2].copy()
            print(f"      gap bound: {gp[1]}")
        if bound_gap is not None:
            gp = bound_gap

        if gp is None:
            # No usable gap yet. The passage's landmarks are furniture-sized,
            # so this is nearly always a matter of standing somewhere they are
            # both in view rather than of them being unfindable.
            h = (reply.get("explore") or {}).get("heading_deg", 0)
            goal = explore_goal(pose, h)
            try:
                cm = ConverterModel(terrain, keepout=keepout, gates=gates)
                want = yaw_of(pose) - np.deg2rad(float(h))
                u, reach, _ = explore_direction(cm, o[:2], want, ctx.spent,
                                                ctx.crossed)
                best = cm.best_waypoint_toward(o[:2] + u * min(reach, 3.0),
                                               o[:2], min_move=MIN_VIEW_MOVE_M)
                if best is not None:
                    goal = best[0]
            except ValueError as e:
                print(f"      converter model unavailable ({e})")
            print(f"      gap not resolvable yet — looking from "
                  f"({goal[0]:+.2f}, {goal[1]:+.2f})")
            rec["action"] = {"kind": "explore", "goal": goal.tolist()}
            step_v = goal - o[:2]
            if float(np.linalg.norm(step_v)) > 1e-6:
                ctx.spent.append((o[:2].copy(),
                                  step_v / float(np.linalg.norm(step_v))))
            if not ctx.dry_run:
                rec["drive"] = ctx.robot.drive_to(goal[0], goal[1],
                                                  min(20.0, ctx.left()))
            ctx.record(rec)
            continue

        mid, why, sides = gp
        dist = float(np.linalg.norm(mid - o[:2]))
        # Aim past the gap, not at it, from where the gap was first seen rather
        # than from here. See `through_point`.
        here = o[:2] if entry is None else entry
        aim = mid if sides is None else through_point(sides[0], sides[1], here)
        print(f"      gate at ({mid[0]:+.2f}, {mid[1]:+.2f}) — {why}, "
              f"{dist:.2f} m away; aiming through to "
              f"({aim[0]:+.2f}, {aim[1]:+.2f})")
        rec["gate"] = {"xy": mid.tolist(), "why": why, "dist_m": dist,
                       "aim": aim.tolist(),
                       "sides": None if sides is None
                       else [sides[0].tolist(), sides[1].tolist()]}

        # A one-landmark passage has no line to cross, so proximity is all
        # there is for it; a two-sided gap is only satisfied by going through.
        if sides is None and dist <= GATE_REACHED_M:
            print(f"      alongside the landmark ({dist:.2f} m) — done")
            rec["passed"] = "alongside"
            ctx.record(rec)
            bank(why, mid, sides, o[:2])
            return Outcome(True, f"passed ({why})", mid)
        if sides is not None and went_between(track, *sides):
            print(f"      the path already crosses between them — done")
            rec["passed"] = "crossed"
            ctx.record(rec)
            bank(why, mid, sides, o[:2])
            return Outcome(True, f"passed ({why})", mid)

        goal = aim
        try:
            cm = ConverterModel(terrain, keepout=keepout, gates=gates)
            best = None
            if sides is not None:
                far = far_side_goal(cm, sides, o[:2], aim, entry=here)
                why_not = far_side_stalled(far, o[:2], reached)
                if why_not is None:
                    no_far = 0
                    g, lands = far
                    best = (g, lands, float(np.linalg.norm(lands - mid)))
                    print(f"      far-side legal point ({g[0]:+.2f}, {g[1]:+.2f})"
                          f" — the stack has to thread the gap to reach it")
                else:
                    no_far += 1
                    print(f"      {why_not} — moving to "
                          f"see more of the far side ({no_far})")
                    # Some of these gaps are not drivable at all, and the time
                    # spent proving it belongs to the destinations after it.
                    # Measured over the seven two-sided passages in the released
                    # questions, the centre-to-centre span runs 1.55-4.34 m and
                    # three are at or below `studio`'s 1.86 m — a net corridor
                    # under 0.8 m, which `local_planner` routes around. On
                    # `studio` that was confirmed twice by hand, including with
                    # the waypoint published from beyond `adjDisThre` so the
                    # converter could not have snapped it: the vehicle went
                    # round the west end of the table every time, while the
                    # organisers' own reference trajectory threads the gap
                    # 0.12 m from its midpoint.
                    if no_far >= 2:
                        print(f"      twice now — this gap is not drivable by "
                              f"the stack; moving on")
                        rec["stopped"] = why_not
                        ctx.record(rec)
                        return Outcome(False, "gap not drivable by the stack",
                                       mid)
            if best is None:
                best = cm.best_waypoint_toward(aim, o[:2], min_move=0.0)
            # A gap between two pieces of furniture is often not reachable on
            # the first look: the local terrain map is 5 m wide and has not yet
            # seen the floor *inside* the gap, so every legal point sits on this
            # side of it. On `studio` the nearest legal point to a correctly
            # computed gate was 1.76 m away and the best move was 0.10 m — which
            # `waypointXYRadius` turns into no motion at all, and the leg then
            # read its own no-op as the passage being unreachable. Driving is
            # what makes the map grow, so a goal that would not move the vehicle
            # is replaced by one that does, in the same direction.
            if best is not None and float(np.linalg.norm(best[1] - o[:2])) \
                    < MIN_VIEW_MOVE_M:
                alt = cm.best_waypoint_toward(aim, o[:2],
                                              min_move=MIN_VIEW_MOVE_M)
                if alt is not None:
                    print(f"      nearest-to-gate goal would not move the "
                          f"vehicle; taking one that does")
                    best = alt
            if best is not None:
                goal, lands, reach = best
                rec["converter"] = {"goal": goal.tolist(),
                                    "settles_at": lands.tolist(),
                                    "settle_to_gate_m": reach}
                print(f"      publish ({goal[0]:+.2f}, {goal[1]:+.2f}) -> settles "
                      f"({lands[0]:+.2f}, {lands[1]:+.2f}), {reach:.2f} m from "
                      f"the gate")
        except ValueError as e:
            print(f"      converter model unavailable ({e})")

        if ctx.dry_run:
            ctx.record(rec)
            return Outcome(False, "dry run", mid)

        d = float(np.linalg.norm(goal - o[:2]))
        res = ctx.robot.drive_to(goal[0], goal[1],
                                 min(max(12.0, d / 0.4 + 8.0), ctx.left()))
        rec["drive"] = res
        ctx.record(rec)
        print(f"      drive: {res.get('why')}  moved {res.get('moved_m')}")

        track.extend(res.get("track") or [])
        now = xy_of(res.get("pose"))
        if now is not None:
            track.append(now.tolist())
            reached.append(now.copy())
        gap = float(np.linalg.norm(mid - (o[:2] if now is None else now)))
        if sides is not None:
            if went_between(track, *sides):
                print(f"      the driven path crosses between them — passed")
                bank(why, mid, sides, o[:2] if now is None else now)
                return Outcome(True, f"passed ({why})", mid)
            print(f"      drove, but the path has not crossed between them yet "
                  f"({gap:.2f} m from the gap)")
        elif gap <= GATE_REACHED_M:
            bank(why, mid, sides, o[:2] if now is None else now)
            return Outcome(True, f"passed ({why})", mid)
        if (res.get("moved_m") or 0.0) < PROGRESS_M:
            stalled += 1
            # One refusal means the map has not grown yet; two in a row from
            # the same place means it is not going to.
            if stalled >= 2:
                return Outcome(False,
                               f"could not reach the passage ({gap:.2f} m)", mid)
        else:
            stalled = 0
    return Outcome(False, "passage not reached")


def execute(ctx: Ctx, question: str, plan: list[Clause], *,
            goto_steps: int) -> list[dict]:
    """Walk the plan. The cursor only moves forward."""
    todo, keep = steps(plan), keepouts(plan)
    ctx.mission = {"question": question,
                   "plan": [str(c) for c in todo],
                   "keepouts": [str(c) for c in keep]}
    # "avoid the path between X and Y" forbids a corridor; "avoid the area near
    # the stool" forbids a place. The plan already knows which, and the shape of
    # the keep-out follows from it — see `ConverterModel`.
    ctx.keepout_is_gate = any(c.relation == "between" for c in keep)
    results: list[dict] = []

    for k, clause in enumerate(todo, 1):
        # Everything still unspent, less a floor held back for each leg after
        # this one. An equal share looks fairer and is not: the first leg is
        # the only one that starts cold and may have to find a *room*, while
        # every later leg starts from a known place with the scene partly
        # mapped. On `home_building_1` an equal third gave the search leg 180 s;
        # it explored a branch, correctly recognised the dead end, turned back,
        # and had found the bedroom by its eighth call — 84 s after the split
        # cut it off. The same leg had succeeded in six calls the run before.
        ctx.leg_deadline = None
        whole = ctx.whole_left()
        if whole != float("inf"):
            ctx.leg_deadline = time.time() + max(
                RESERVE_S, whole - RESERVE_S * (len(todo) - k))
        left = ctx.left()
        print(f"\n{'=' * 72}\nstep {k}/{len(todo)}  {clause}"
              f"   ({left:.0f}s for this leg, {whole:.0f}s for the question)"
              f"\n{'=' * 72}")
        if ctx.out_of_time():
            results.append({"k": k, "clause": str(clause), "ok": False,
                            "why": "out of time"})
            print("      out of time — not attempted")
            continue
        if clause.kind == GOTO:
            # `confirm` only on the final destination: it records whether the
            # model's own fields agree with the geometry that decided arrival,
            # and one call at the end is worth what one call at the end of every
            # leg is not.
            r = run_goto(ctx, clause.text, max_steps=goto_steps, k=k,
                         confirm=(k == len(todo)))
        else:
            r = run_pass(ctx, clause, k)
        results.append({"k": k, "clause": str(clause), "kind": clause.kind,
                        "ok": r.arrived, "why": r.why,
                        "xy": None if r.xy is None else np.asarray(r.xy).tolist()})
        print(f"      step {k} {'OK' if r.arrived else 'FAILED'}: {r.why}")
        # A failed leg does not end the run. The score is per-constraint with
        # partial credit, so the destinations after this one are still worth
        # driving — abandoning them would forfeit points the robot can reach
        # from exactly where it now stands.
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help="ssh target running the sim (default: $XIAO_HEI_SIM_HOST, "
                         f"currently {DEFAULT_HOST or 'unset'}); omit and leave "
                         "the variable unset if this IS the sim host")
    ap.add_argument("--container", default=CTR)
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--goto-steps", type=int, default=20,
                    help="safety cap on grounding calls per destination; the "
                         "real governor is the leg's share of --budget")
    ap.add_argument("--budget", type=float, default=BUDGET_S,
                    help="seconds for the whole question (README allows 600)")
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VER)
    ap.add_argument("--standoff", type=float, default=STANDOFF_M)
    ap.add_argument("--visited", choices=VISITED_STYLES, default=VISITED_STYLE,
                    help="how 'places already searched' is written: prose (the "
                         "original, and the rollback), bearing, xy, or off. "
                         "Env XIAO_HEI_VISITED sets the default.")
    ap.add_argument("--drop-here", action="store_true",
                    help="stop asking the model for 'here' — 19.7%% of the "
                         "reply, and the run's best diagnostic. Only sane with "
                         "--visited bearing/xy/off.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plan-only", action="store_true",
                    help="decompose and print the plan; touch no robot")
    ap.add_argument("--dry-run", action="store_true",
                    help="ground and compute waypoints, publish nothing")
    args = ap.parse_args()

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else DEFAULT_GEMINI_MODEL)
    cache_p = Path("artifacts/decompose_cache.json")
    cache = json.loads(cache_p.read_text()) if cache_p.is_file() else {}
    plan, from_model = decompose(args.question, cache=cache)
    print(f"plan ({'model' if from_model else 'REGEX FALLBACK'}):")
    for i, c in enumerate(steps(plan), 1):
        print(f"  step {i}  {c}")
    for c in keepouts(plan):
        print(f"  always  {c}")
    if args.plan_only:
        return 0

    out = Path(args.out or f"runs/{time.strftime('%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "steps.jsonl").open("w")

    robot = Robot(args.host, args.container)
    robot.push()
    pre = robot.preflight()
    print(f"\npreflight: {json.dumps(pre)}")
    if not pre.get("ok"):
        print(f"  !! {pre.get('why')}", file=sys.stderr)

    ctx = Ctx(robot=robot, out=out, log=log, backend=args.backend, model=model,
              prompt_version=args.prompt_version, standoff=args.standoff,
              visited_style=args.visited, drop_here=args.drop_here,
              dry_run=args.dry_run, deadline=time.time() + args.budget)
    print(f"visited style: {ctx.visited_style}"
          f"{'  (here not requested)' if ctx.drop_here else ''}")
    ctx.note_settings()
    t0 = time.time()
    results = execute(ctx, args.question, plan, goto_steps=args.goto_steps)
    ctx.leg_deadline = None
    ctx.close()

    done = sum(r["ok"] for r in results)
    print(f"\n{'=' * 72}")
    for r in results:
        print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['k']}. {r['clause']}"
              f"   — {r['why']}")
    print(f"\n{done}/{len(results)} constraints satisfied in {ctx.calls} calls "
          f"(${ctx.calls * COST_PER_CALL:.2f}), {time.time() - t0:.0f}s of "
          f"{args.budget:.0f}   log: {out}/steps.jsonl")
    (out / "plan.json").write_text(json.dumps(
        {"question": args.question, "from_model": from_model,
         "plan": [str(c) for c in plan], "results": results}, indent=1))
    return 0 if done == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
