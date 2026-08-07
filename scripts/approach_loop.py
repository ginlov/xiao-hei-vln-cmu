#!/usr/bin/env python3
"""Drive to one named object: ground, step, re-observe, repeat.

The simplest instruction the challenge asks — a single target, no ordering and
no forbidden region — closed end to end:

    capture → VLM → box → bearing → lidar → waypoint → drive → repeat

Everything TASK 26 and 27 measured is here as one policy. Bearing is trusted
because it was measured good (0.48° median on hits). Range is not: the lift is
believed only when it survives a size check, and otherwise the loop takes a
bounded step and looks again rather than guessing a distance the model answers
worse than a constant. Arrival comes from `/state_estimation` alone — our own
distance to the waypoint, plus whether the vehicle has stopped — because
`/way_point_reached` is not on the challenge's allowed-topic list.

Comparative relations ("closest to", "farthest from", "between") are resolved
by measuring the candidates the model reports, never by asking it which wins.

Runs from anywhere; `--host` selects whether the docker commands are local or
tunnelled over ssh. The API key stays wherever this runs, so from a laptop the
sim host never needs one.

    uv run --with anthropic python scripts/approach_loop.py \\
        --host xiaohei1 "the tea table with the elephant figurine on it"

Every step writes its faces, reply, waypoint and drive result under
`--out`, so a run can be read back without re-flying it.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from vlm_approach import (STANDOFF_M, _lift_xy, box_angular_size,  # noqa: E402
                          crop_face, has_relation, in_blind_cone,
                          next_waypoint, ray_from_box, resolve_relation)
from vlm_locate import rot_from_quat, scan_to_camera  # noqa: E402
from waypoint_converter_model import (WAYPOINT_XY_RADIUS,  # noqa: E402
                                      ConverterModel)
from vlm_probe import (DEFAULT_PROMPT_VER, NAMES, ask_claude,  # noqa: E402
                       ask_gemini, build_prompt, parse, to_pixels)
from vlm_sweep import faces_of  # noqa: E402

BRIDGE = Path(__file__).resolve().parent / "robot_io.py"
CTR = "iros2026_system"
ROS_ENV = ("source /opt/ros/jazzy/setup.bash && "
           "source /home/docker/autonomy_stack_mecanum_wheel_platform/install/setup.bash && "
           "export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ")
# One grounding call, measured in TASK 26.
COST_PER_CALL = 0.0265
# How far to drive when the target is not visible and only a heading is known.
# TURN_STEP_M is the fallback when the terrain cannot be read; otherwise the leg
# runs as far as the legal set allows, capped, because a call that buys 0.13 m
# of parallax costs the same as one that buys 3 m.
TURN_STEP_M = 0.8
MIN_EXPLORE_M = 0.5
MAX_EXPLORE_M = 3.0

# `/way_point_reached` answers "did you get to the waypoint", where *the
# waypoint* is what `waypoint_converter` snapped ours to — not what we asked
# for. The two differ by `dist_to_requested_m`, and on the first live run they
# differed by 0.69 m: the stack refuses to place the vehicle within
# `obstacleDisThre` (0.75 m) of an obstacle, so any standoff under that is
# unachievable no matter how many times it is asked for.
#
# So arrival is not "reached". It is either landing near what we asked for, or
# the stack declining to take us any closer — which is the real definition of
# as-close-as-possible and needs no model to decide.
# No arrival tolerance here any more: reaching our own waypoint says nothing
# about reaching the target, now that the waypoint is chosen for where it makes
# the converter *settle*. `robot_io.py` still uses one to end a drive.
PROGRESS_M = 0.25          # moved less than this toward a destination = clamped
# How far a new reading may move the bound target before it stops being a
# refinement and starts being a different object. See `bind_target`.
JUMP_M = 1.0
# Inside this range of the target, "the converter cannot do better" means the
# platform's floor; outside it, it means the terrain map has not seen enough.
# Calibrated on three scenes, where the floor sat at 1.1-1.5 m to the object
# centre — not a measured constant, and the first thing to re-derive if a scene
# stops short for no visible reason.
NEAR_M = 1.5
# The comparisons geometry can settle by measuring, rather than by asking.
RELATIONS = ("closest_to", "farthest_from", "between")
# `scripts/keepout_radius.py` bounds this two-sided: at least 0.86 m to swallow
# our own p90 centre error plus the vehicle, at most 1.98 m or the zone would
# forbid the reference trajectory the organisers shipped as the answer.
KEEPOUT_M = 1.2
# The vehicle considers itself arrived once it is within `waypointXYRadius` of
# its waypoint, so a commanded move shorter than that is not a small move — it
# is no move at all. On `hotel_room_2` the blind-cone branch asked for 0.30 m,
# the platform did not budge, and the loop read its own no-op as being stuck
# and gave up with the target 2.8 m away. Any leg whose *purpose* is a new
# viewpoint has to clear this; an approach does not, because settling close by
# is the correct answer there.
MIN_VIEW_MOVE_M = WAYPOINT_XY_RADIUS + 0.2


class Robot:
    """Capture and drive, over `docker exec` either locally or through ssh."""

    def __init__(self, host: str | None, container: str = CTR) -> None:
        self.host, self.ctr = host, container

    def _run(self, cmd: str, stdin: bytes | None = None,
             binary: bool = False, timeout: float | None = None) -> bytes:
        argv = ["ssh", self.host, cmd] if self.host else ["bash", "-lc", cmd]
        p = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
        if p.returncode != 0 and not binary:
            sys.stderr.write(p.stderr.decode(errors="replace"))
        return p.stdout

    def _bridge(self, args: str, timeout: float) -> dict:
        cmd = (f"docker exec {self.ctr} bash -lc '{ROS_ENV}"
               f"python3 /tmp/robot_io.py {args}'")
        # Hard ceiling above the bridge's own timeout: if rclpy wedges on
        # discovery the subprocess would otherwise hang the whole loop.
        out = self._run(cmd, timeout=timeout + 15)
        # The bridge prints exactly one JSON object; ROS chatter goes to stderr,
        # but a crash inside the container arrives here as an empty stdout.
        line = next((l for l in reversed(out.decode(errors="replace").splitlines())
                     if l.startswith("{")), None)
        if line is None:
            raise SystemExit(f"bridge produced no JSON for {args!r} — "
                             f"is {self.ctr} running?")
        return json.loads(line)

    def push(self) -> None:
        self._run(f"docker exec -i {self.ctr} tee /tmp/robot_io.py >/dev/null",
                  stdin=BRIDGE.read_bytes())

    def preflight(self) -> dict:
        return self._bridge("preflight", 10)

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        r = self._bridge("capture", 20)
        if not r.get("ok"):
            raise SystemExit(f"capture failed, missing {r.get('missing')}")
        jpg = self._run(f"docker exec {self.ctr} cat /tmp/loop_img.jpg",
                        binary=True)
        npy = self._run(f"docker exec {self.ctr} cat /tmp/loop_scan.npy",
                        binary=True)
        ter = self._run(f"docker exec {self.ctr} cat /tmp/loop_terrain.npy",
                        binary=True)
        eq = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return (eq, np.load(io.BytesIO(npy)), np.load(io.BytesIO(ter)),
                r["pose"])

    def drive_to(self, x: float, y: float, timeout: float) -> dict:
        return self._bridge(f"drive {x:.4f} {y:.4f} --timeout {timeout:.1f}",
                            timeout + 10)


def yaw_of(pose: dict) -> float:
    R = rot_from_quat(pose["orientation"])
    return float(np.arctan2(R[1, 0], R[0, 0]))


def explore_goal(pose: dict, heading_deg: float) -> np.ndarray:
    """Where to drive when the target is not in view.

    The prompt numbers the faces clockwise — image 1 is "heading 90°, right" —
    while the map frame measures yaw counter-clockwise, so the two differ by a
    sign. Getting this backwards sends the robot away from the thing it was
    told to look for, and nothing downstream would flag it.
    """
    theta = yaw_of(pose) - np.deg2rad(float(heading_deg))
    o = np.asarray(pose["position"], float)[:2]
    return o + TURN_STEP_M * np.array([np.cos(theta), np.sin(theta)])


def bind_constraints(reply: dict, scan: np.ndarray, pose: dict,
                     avoid: list[dict]) -> list[dict]:
    """Lift the objects a keep-out is anchored on, and remember them.

    Identifying "the cabinet" is the model's half; staying 1.2 m away from it
    is geometry's, and the score is on the trajectory driven, so it has to be
    geometry's. Keep-out anchors are also the easy half of the sentence — they
    are furniture-sized landmarks, where the target can be a 13 cm cup.
    """
    items = reply.get("avoid") or []
    if not items:
        return avoid
    scan_cam = scan_to_camera(scan, pose)
    space = reply.get("coord_space")
    for it in items:
        if it.get("box_2d") is None or it.get("image_index") is None:
            continue
        xy = _lift_xy(to_pixels(it["box_2d"], space, G.FACE_SIZE),
                      int(it["image_index"]), scan_cam, pose)
        if xy is None:
            continue
        xy = np.asarray(xy, float)[:2]
        near = next((a for a in avoid
                     if np.linalg.norm(a["xy"] - xy) < JUMP_M), None)
        if near is None:
            avoid.append({"xy": xy, "name": it.get("name") or "?"})
            print(f"      keep-out bound: {it.get('name')!r} at "
                  f"({xy[0]:+.2f}, {xy[1]:+.2f}), radius {KEEPOUT_M} m")
        else:
            near["xy"] = xy          # same anchor, seen better
    return avoid


def explore_direction(cm: ConverterModel, origin: np.ndarray,
                      want: float) -> tuple[np.ndarray, float, float]:
    """A direction that is both what the model asked for and drivable.

    The model says which way is worth looking; the terrain says which way the
    vehicle can go. Taking the first without the second drove loft into a wall
    three times — heading 270° had 0.00 m of reach while 30° away had 5.16 m,
    and every one of those refusals cost a grounding call.

    Scored as `reach · cos(Δ)`: progress made in the direction actually asked
    for, so a long detour never beats a shorter leg that goes the right way.
    Returns `(unit direction, reach, Δ in degrees)`.
    """
    best = (0.0, np.array([np.cos(want), np.sin(want)]), 0.0, 0.0)
    for delta in range(-90, 91, 15):
        th = want + np.deg2rad(delta)
        u = np.array([np.cos(th), np.sin(th)])
        reach = cm.reach_along(origin, u)
        score = reach * float(np.cos(np.deg2rad(delta)))
        if score > best[0]:
            best = (score, u, reach, float(delta))
    return best[1], best[2], best[3]


def bind_target(wp, origin: np.ndarray, reply: dict, bound: dict | None,
                rec: dict, *, verified: bool = True,
                measured: bool = False) -> tuple[bool, dict | None]:
    """Keep one map position for the target, and defend it from later readings.

    A binding is not a detection. Re-grounding from a new pose is free to
    nominate a different instance, and on `japanese_room` it did: step 1 bound
    the lantern 0.19 m from the truth, step 2 produced one 4.18 m away, and the
    loop drove to it. The model was no help — it reported
    `same_object_as_previous: True` and *higher* confidence in both that case
    and the healthy one.

    What separates them is how far the reading moved. Refining a binding from
    closer up shifts it a little; switching objects teleports it:

        office_1       0.56 m error -> 0.05 m      binding moved 0.52 m
        japanese_room  0.19 m error -> 4.18 m      binding moved 4.37 m

    So the gate is on distance, not identity — which is what makes it safe
    against the obvious objection, that binding early locks in an early
    mistake. It does not block refinement, only teleportation. The failure it
    does accept is a *grossly* wrong first sighting, measured at 4/54 = 7.4 %
    in TASK 26; a jump is then only allowed back if the model itself reports a
    different object with more confidence than the binding was made with.

    `verified` says this reading may be bound at all; `measured` says the
    phrase's relation was settled by lifting an anchor rather than assumed, and
    is what lets a later reading overrule an earlier unchecked one.

    Returns `(committed, bound)`. A binding also rescues an untrusted lift: the
    blind cone costs us the range, not the position we already measured.
    """
    conf = float(reply.get("confidence") or 0.0)
    if not verified:
        # A comparative phrase whose anchor never lifted carries no evidence at
        # all about the comparison. On loft the model nominated a cup 0.06 m
        # from a real cup and 5.68 m from the right one, because the TV remote
        # it was supposed to be near is 5x20x2 cm and was never found. Binding
        # on that is binding on the noun and discarding the phrase.
        if bound is not None:
            print(f"      relation still unmeasurable — keeping the binding at "
                  f"({bound['xy'][0]:+.2f}, {bound['xy'][1]:+.2f})")
            rec["binding"] = {"xy": bound["xy"].tolist(), "conf": bound["conf"],
                              "carried": True}
            return True, bound
        print(f"      relation unmeasurable and nothing bound yet — this is a "
              f"guess at the noun, not the phrase; moving to look, not to stop")
        rec["unverified"] = True
        return False, bound
    if wp.committed and wp.range_m is not None:
        d = wp.xy - origin
        n = float(np.linalg.norm(d))
        seen = origin + (d / n if n > 1e-6 else d) * wp.range_m
        if bound is None:
            print(f"      bound the target at ({seen[0]:+.2f}, {seen[1]:+.2f})")
            bound = {"xy": seen, "conf": conf, "verified": measured}
        else:
            jump = float(np.linalg.norm(seen - bound["xy"]))
            switched = reply.get("same_object_as_previous") is False
            if jump <= JUMP_M:
                print(f"      binding refined {jump:.2f} m -> "
                      f"({seen[0]:+.2f}, {seen[1]:+.2f})")
                bound = {"xy": seen, "conf": conf,
                         "verified": measured or bound.get("verified", True)}
            elif measured and not bound.get("verified", True):
                # The distance gate exists to stop an unverified reading from
                # teleporting the binding. It is not meant to defend a binding
                # that was itself never checked: on `studio` an early call that
                # reported no relation bound the couch, and every later call
                # that measured the phrase properly was then refused for
                # jumping too far. Measurement outranks a guess at any distance.
                print(f"      re-bound {jump:.2f} m away — this reading "
                      f"measured the phrase, the binding it replaces did not")
                bound = {"xy": seen, "conf": conf, "verified": True}
            elif switched and conf > bound["conf"]:
                print(f"      re-bound {jump:.2f} m away — the model reports a "
                      f"different object at higher confidence")
                bound = {"xy": seen, "conf": conf, "verified": measured}
            else:
                print(f"      this lift lands {jump:.2f} m from the binding "
                      f"while still calling it the same object — keeping the "
                      f"binding at ({bound['xy'][0]:+.2f}, {bound['xy'][1]:+.2f})")
                rec["binding_rejected"] = {"seen": seen.tolist(), "jump_m": jump}
        rec["binding"] = {"xy": bound["xy"].tolist(), "conf": bound["conf"]}
        return True, bound
    if bound is not None:
        # The lift was refused — a blind bearing, or a size the range cannot
        # explain. None of that unmakes a position already measured, and
        # `/state_estimation` carries it across the move.
        print(f"      lift not usable, but the target is bound at "
              f"({bound['xy'][0]:+.2f}, {bound['xy'][1]:+.2f}) — driving to it")
        rec["binding"] = {"xy": bound["xy"].tolist(), "conf": bound["conf"],
                          "carried": True}
        return True, bound
    return False, bound


def ground(faces: list[bytes], phrase: str, backend: str, model: str,
           prev: bytes | None, version: str,
           visited: list[str] | None = None) -> tuple[dict | None, str]:
    """The parsed reply and the text it came from.

    The raw text is returned because three runs died on "unparseable reply"
    while discarding the only evidence of why. It was truncation.
    """
    fn = ask_claude if backend == "claude" else ask_gemini
    text = fn(build_prompt(phrase, approach=True, version=version,
                           visited=visited),
              faces, model, previous=prev)
    return parse(text), text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrase")
    ap.add_argument("--host", default=None,
                    help="ssh target running the sim; omit if this IS the sim host")
    ap.add_argument("--container", default=CTR)
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VER,
                    help="v3-occlusion-distance to reproduce TASK 26/28")
    ap.add_argument("--standoff", type=float, default=STANDOFF_M)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="ground and compute waypoints, publish nothing")
    args = ap.parse_args()

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else "gemini-2.5-flash")
    out = Path(args.out or f"runs/{time.strftime('%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "steps.jsonl").open("w")

    robot = Robot(args.host, args.container)
    robot.push()
    pre = robot.preflight()
    print(f"preflight: {json.dumps(pre)}")
    if not pre.get("ok"):
        print(f"  !! {pre.get('why')}", file=sys.stderr)

    print(f"\ntarget: {args.phrase!r}   model={model}   out={out}\n")
    prev_crop, calls, arrived, bound = None, 0, False, None
    misses = stuck_explores = 0
    avoid: list[dict] = []
    visited: list[str] = []

    for step in range(1, args.max_steps + 1):
        eq, scan, terrain, pose = robot.capture()
        faces = faces_of(eq)
        for i, f in enumerate(faces):
            (out / f"step{step}_face{i}.jpg").write_bytes(f)
        # Keep the geometry too, not just the pictures. Without the terrain the
        # converter's choice cannot be re-derived after the fact, which is how a
        # bad waypoint on chinese_room went unexplained.
        np.save(out / f"step{step}_terrain.npy", terrain)
        np.save(out / f"step{step}_scan.npy", scan)
        keepout: list[tuple[np.ndarray, float]] = []

        reply, raw = ground(faces, args.phrase, args.backend, model, prev_crop,
                            args.prompt_version, visited)
        calls += 1
        rec: dict = {"step": step, "pose": pose, "reply": reply}
        if reply is None:
            print(f"[{step}] unparseable reply ({len(raw)} chars); stopping")
            print(f"      tail: ...{raw[-200:]!r}")
            rec["raw"] = raw
            log.write(json.dumps(rec) + "\n")
            break

        o = np.asarray(pose["position"], float)
        print(f"[{step}] at ({o[0]:+.2f}, {o[1]:+.2f})  "
              f"visible={reply.get('visible')}  conf={reply.get('confidence')}  "
              f"state={reply.get('target_state')}  "
              f"same_as_prev={reply.get('same_object_as_previous')}")

        # Before the visibility branch: a keep-out anchor is most likely to be
        # reported on exactly the calls where the *target* is not visible,
        # because that is when the robot is looking around at the furniture.
        here_txt = (reply.get("here") or "").strip()
        if here_txt:
            visited.append(here_txt)
            rec["here"] = here_txt
            print(f'      here: "{here_txt[:96]}"')

        avoid = bind_constraints(reply, scan, pose, avoid)
        keepout = [(a["xy"], KEEPOUT_M) for a in avoid]
        if reply.get("gate"):
            # Parsed and logged; not enforced yet. Ten of the thirty
            # instruction questions need it and it is a different mechanism —
            # a point to pass through, not a region to stay out of.
            rec["gate"] = reply["gate"]
            print(f"      gate reported ({len(reply['gate'])} anchors) — "
                  f"logged, not yet enforced")

        if not reply.get("visible"):
            misses += 1
            # One frame of occlusion is ordinary; two calls in a row that cannot
            # find the target mean the position we measured is not where the
            # thing is. Keeping it would let a wrong binding steer the rest of
            # the run silently, which is what happened on loft.
            if bound is not None and misses >= 2:
                print(f"      not seen for {misses} calls — dropping the binding "
                      f"at ({bound['xy'][0]:+.2f}, {bound['xy'][1]:+.2f})")
                rec["binding_dropped"] = bound["xy"].tolist()
                bound = None
            h = (reply.get("explore") or {}).get("heading_deg", 0)
            goal, reach, delta = explore_goal(pose, h), None, None
            # Exploration used to publish a fixed 0.8 m hop along whatever
            # heading came back, without ever asking whether the vehicle could
            # go that way. Ask the terrain instead, and then go as far as it
            # says — a leg that moves 0.13 m costs the same call as one that
            # moves 3 m and reveals nothing.
            asked = None
            try:
                cm = ConverterModel(terrain, keepout=keepout)
                want = yaw_of(pose) - np.deg2rad(float(h))
                asked = cm.reach_along(
                    o[:2], np.array([np.cos(want), np.sin(want)]))
                u, reach, delta = explore_direction(cm, o[:2], want)
                if reach >= MIN_EXPLORE_M:
                    goal = o[:2] + u * min(reach, MAX_EXPLORE_M)
                    best = cm.best_waypoint_toward(goal, o[:2],
                                                   min_move=MIN_VIEW_MOVE_M)
                    if best is not None:
                        goal = best[0]
            except ValueError as e:
                print(f"      converter model unavailable ({e})")
            if asked is not None:
                print(f"      heading {h}° reaches {asked:.2f} m; best drivable "
                      f"is {delta:+.0f}° off it at {reach:.2f} m")
            print(f"      NOT_VISIBLE — heading {h}°, "
                  f"driving to ({goal[0]:+.2f}, {goal[1]:+.2f})")
            rec["action"] = {"kind": "explore", "heading_deg": h,
                             "goal": goal.tolist(), "reach_m": reach,
                             "delta_deg": delta}
            if not args.dry_run:
                rec["drive"] = robot.drive_to(goal[0], goal[1], 20)
                if (rec["drive"].get("moved_m") or 0.0) < PROGRESS_M:
                    stuck_explores += 1
                else:
                    stuck_explores = 0
            log.write(json.dumps(rec) + "\n")
            log.flush()
            prev_crop = None
            if stuck_explores >= 2:
                print(f"      two exploration legs in a row went nowhere — the "
                      f"heading is not reachable from here; stopping")
                break
            continue
        misses = 0

        i = int(reply["image_index"])
        box = to_pixels(reply.get("feature_box_2d") or reply["box_2d"],
                        reply.get("coord_space"), G.FACE_SIZE)
        # A comparative relation is decided here, by measuring the candidates,
        # not by whichever one the model nominated.
        chosen = resolve_relation(reply, scan, pose, size=G.FACE_SIZE)
        if chosen is not None:
            box, i, rel_why = chosen
            rec["relation"] = rel_why
            print(f"      relation resolved: {rel_why}")
        elif reply.get("relation"):
            print(f"      relation {reply['relation']!r} not measurable "
                  f"({len(reply.get('candidates') or [])} candidates, "
                  f"{len(reply.get('anchors') or [])} anchors) — using the "
                  f"model's own pick")
        # Whether the phrase needs checking is a property of the phrase. Asking
        # the reply instead let `studio` through: the model reported no relation
        # for "the guitar near the couch" on one call and `closest_to` on the
        # next, and the call that forgot was treated as nothing to verify.
        relational = has_relation(args.phrase) or reply.get("relation") in RELATIONS
        verified = (not relational) or chosen is not None
        w, h_deg = box_angular_size(box, i)
        blind, az, el, floor = in_blind_cone(ray_from_box(box, i))
        print(f"      image {i} ({NAMES[i]}), box {w:.1f}x{h_deg:.1f}°, "
              f"bearing {az:+.0f}°/{el:+.0f}° "
              f"({'BLIND' if blind else 'covered'}, floor {floor:+.0f}°)")

        wp = next_waypoint(box, i, scan, pose, phrase=args.phrase,
                           standoff=args.standoff)
        rec["waypoint"] = {"xy": wp.xy.tolist(), "committed": wp.committed,
                           "range_m": wp.range_m, "reason": wp.reason,
                           "blind": blind, "az": az, "el": el}
        print(f"      -> ({wp.xy[0]:+.2f}, {wp.xy[1]:+.2f}) "
              f"[{'DESTINATION' if wp.committed else 'step'}]  {wp.reason}")

        committed, bound = bind_target(wp, o[:2], reply, bound, rec,
                                       verified=verified,
                                       measured=chosen is not None)

        # Already inside the standoff: driving further would push into the
        # object, and the stack would only snap the waypoint back out again.
        if committed and bound is not None:
            here = float(np.linalg.norm(bound["xy"] - o[:2]))
            if here <= args.standoff:
                print(f"      already within {args.standoff} m — arrived")
                rec["arrived"] = "within standoff"
                log.write(json.dumps(rec) + "\n")
                arrived = True
                break

        # What the converter will do with this waypoint, before we spend a
        # drive and another grounding call finding out. TASK 28 read a 1.08 m
        # displacement as the platform clamping an approach; it was the
        # converter discarding the waypoint and re-minimising elsewhere.
        goal, will_move, cm = wp.xy, None, None
        try:
            cm = ConverterModel(terrain, keepout=keepout)
            # Aim at the target itself, not at a standoff from it: the standoff
            # is what the converter's inflation is *for*, and asking for a point
            # inside it gets the waypoint discarded rather than clamped. Once
            # the target is bound, the binding is the better estimate of where
            # it is than any single reading.
            aim = bound["xy"] if bound is not None else wp.xy
            # A step exists to buy a better view, so it has to actually move
            # the vehicle; an approach may legitimately settle where it stands.
            best = cm.best_waypoint_toward(
                aim, o[:2], min_move=0.0 if committed else MIN_VIEW_MOVE_M)
            if best is not None:
                goal, lands, reach = best
                will_move = float(np.linalg.norm(lands - o[:2]))
                rec["converter"] = {
                    "aim": aim.tolist(), "goal": goal.tolist(),
                    "settles_at": lands.tolist(), "settle_to_aim_m": reach,
                    "will_move_m": will_move,
                    "asked_would_settle": cm.settle(wp.xy, o[:2]).tolist(),
                    "legal_points": int(len(cm.legal_points()))}
                # `aim` is the target only when the lift was committed; on a
                # step it is just the next place to look from, and calling that
                # "the target" in the log invites exactly the wrong reading.
                what = "the target" if committed else "the step point"
                print(f"      publish ({goal[0]:+.2f}, {goal[1]:+.2f}) -> settles "
                      f"({lands[0]:+.2f}, {lands[1]:+.2f}), {reach:.2f} m from "
                      f"{what}, {will_move:.2f} m from here")
        except ValueError as e:
            # A terrain frame we cannot read is a reason to fly blind, not to
            # abort a run that would otherwise work.
            print(f"      converter model unavailable ({e})")

        # Stop when driving would not get us *closer*, which is not the same as
        # the converter refusing to move us. The legal points around an object
        # form a ring at roughly equal distance from it, so there is always
        # another one worth 1.4 m of driving and 0.08 m of progress: on
        # chinese_room the loop circled the tea table for six calls that way.
        # What that stop *means* still depends on the waypoint — a committed one
        # is the platform's floor, a step is a robot that can neither improve
        # its view nor move, which is not an arrival however close it stands.
        gain = here = None
        if will_move is not None:
            here = float(np.linalg.norm(aim - o[:2]))
            gain = here - reach
            print(f"      {here:.2f} m from it now, {reach:.2f} m after "
                  f"{will_move:.2f} m of driving — gain {gain:+.2f} m")
        # A small gain far from the target is not the platform's floor, it is a
        # 5 m local terrain map that has not seen the ground near the object
        # yet. Driving anywhere is then still worth it, because it is what makes
        # the map grow — every chinese_room run that reached 0.4 m did so by
        # first driving to a point the model rated poorly. Only near the object
        # does "cannot improve" mean "cannot get closer".
        # With nothing bound and an unverified nomination there is no target
        # position to be as-close-as-possible to, so "gain" is measuring the
        # distance to a guess. The point of moving is to see better, and only
        # the post-drive progress test can end that.
        may_stop = committed or bound is not None
        if gain is not None and gain < PROGRESS_M and (here > NEAR_M or not may_stop):
            print(f"      not close enough to call this the floor "
                  f"({here:.2f} m{'' if may_stop else ', nothing bound yet'}) "
                  f"— driving to look")
        elif gain is not None and gain < PROGRESS_M:
            if committed:
                print(f"      no legal point closer than where we stand — this "
                      f"is as near as the platform allows")
                rec["arrived"] = "no legal point closer (predicted)"
                arrived = True
            else:
                print(f"      stuck: the lift is untrustworthy here and the "
                      f"converter has nowhere legal to move us")
                rec["stopped"] = "stuck (untrusted lift, no legal move)"
            log.write(json.dumps(rec) + "\n")
            break

        # Past the stop tests, so this step is going to drive — and a goal the
        # vehicle settles less than `waypointXYRadius` from is one it considers
        # already reached. Publishing it produces zero motion, which the
        # post-drive test below would read as the stack clamping an approach.
        # On `hotel_room_2` that turned "the map has not seen the floor near the
        # lamp yet" into ARRIVED, 2.24 m short.
        if cm is not None and will_move is not None and will_move < MIN_VIEW_MOVE_M:
            alt = cm.best_waypoint_toward(aim, o[:2], min_move=MIN_VIEW_MOVE_M)
            if alt is None:
                print(f"      nowhere legal to move that the platform would act "
                      f"on — boxed in {here:.2f} m from it")
                rec["stopped"] = "boxed in (no legal move above waypointXYRadius)"
                log.write(json.dumps(rec) + "\n")
                break
            goal, lands, reach = alt
            will_move = float(np.linalg.norm(lands - o[:2]))
            rec["converter"]["requeried_for_motion"] = {
                "goal": goal.tolist(), "settles_at": lands.tolist(),
                "settle_to_aim_m": reach, "will_move_m": will_move}
            print(f"      that goal would not move the vehicle; going to "
                  f"({goal[0]:+.2f}, {goal[1]:+.2f}) instead -> settles "
                  f"{reach:.2f} m from it, {will_move:.2f} m from here")

        prev_crop = crop_face(faces[i], box)
        (out / f"step{step}_target.jpg").write_bytes(prev_crop)

        if args.dry_run:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            break

        dist = float(np.linalg.norm(goal - o[:2]))
        res = robot.drive_to(goal[0], goal[1], max(12.0, dist / 0.4 + 8.0))
        rec["drive"] = res
        print(f"      drive: {res.get('why')}  moved {res.get('moved_m')}  "
              f"final gap to requested point {res.get('dist_to_requested_m')}")
        log.write(json.dumps(rec) + "\n")
        log.flush()

        if res.get("why") == "timeout":
            print("      drive timed out; stopping")
            break
        gap = res.get("dist_to_requested_m")
        moved = res.get("moved_m") or 0.0
        if committed:
            if moved < PROGRESS_M:
                # The stack declining to move is the platform's floor only when
                # we are near the thing. Far away it means something else — a
                # local map that has not seen the ground near the target — and
                # calling that an arrival reports success 2 m short.
                if here is not None and here > NEAR_M:
                    print(f"      asked for {will_move:.2f} m and moved "
                          f"{moved:.2f} m, still {here:.2f} m from it — boxed "
                          f"in, not arrived")
                    rec["stopped"] = "boxed in (stack would not move us)"
                    break
                print(f"      stack will not close the last {gap:.2f} m "
                      f"(moved {moved:.2f} m) — as near as it allows")
                rec["arrived"] = "clamped by obstacle clearance"
                arrived = True
                break
            # Reaching the waypoint is not arriving. Since the waypoint became
            # "the legal point that settles nearest the target" rather than the
            # target itself, it can sit metres away — on chinese_room the loop
            # drove to one 2.61 m from the tea table and declared victory.
            # Getting there is a better vantage point, nothing more; arrival is
            # decided by the converter having nothing closer to offer.
            print(f"      reached it ({gap:.2f} m from the requested point) — "
                  f"re-observing from here")
        elif moved < PROGRESS_M:
            # A step that went nowhere. Re-grounding from an unchanged pose
            # would ask the same question and get the same answer.
            print(f"      step made no progress ({moved:.2f} m); stopping")
            break

    # One last look, purely to record whether the two new fields agree with the
    # geometry that actually decided this. They gate nothing yet.
    if arrived and not args.dry_run:
        eq, scan, terrain, pose = robot.capture()
        faces = faces_of(eq)
        try:
            confirm, _ = ground(faces, args.phrase, args.backend, model,
                                prev_crop, args.prompt_version, visited)
            calls += 1
        except Exception as e:
            # Advisory only — it records whether the new fields agree with the
            # geometry that already decided this. Losing it must not turn a
            # completed run into a failed one.
            print(f"\nconfirm call failed ({type(e).__name__}); arrival stands")
            confirm = None
        if confirm:
            print(f"\nconfirm: visible={confirm.get('visible')} "
                  f"state={confirm.get('target_state')} "
                  f"same_object={confirm.get('same_object_as_previous')} "
                  f"conf={confirm.get('confidence')}")
        log.write(json.dumps({"step": "confirm", "pose": pose,
                              "reply": confirm}) + "\n")

    log.close()
    print(f"\n{'ARRIVED' if arrived else 'did not arrive'} in {calls} calls "
          f"(${calls * COST_PER_CALL:.2f})   log: {out}/steps.jsonl")
    return 0 if arrived else 1


if __name__ == "__main__":
    raise SystemExit(main())
