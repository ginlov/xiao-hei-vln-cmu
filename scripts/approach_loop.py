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
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from instruction_plan import GOTO  # noqa: E402
from vlm_approach import (STANDOFF_M, _lift_xy, box_angular_size,  # noqa: E402
                          crop_face, has_relation, in_blind_cone,
                          next_waypoint, ray_from_box, resolve_relation)
from vlm_locate import rot_from_quat, scan_to_camera  # noqa: E402
from waypoint_converter_model import (WAYPOINT_XY_RADIUS,  # noqa: E402
                                      ConverterModel)
from vlm_probe import (DEFAULT_PROMPT_VER, NAMES, ask_claude,  # noqa: E402
                       ask_gemini, build_prompt, parse, to_pixels)
from faces import faces_of  # noqa: E402

BRIDGE = Path(__file__).resolve().parent / "robot_io.py"
CTR = os.environ.get("XIAO_HEI_SIM_CONTAINER", "iros2026_system")
# The sim host, shared with `sim.sh` and `drive.sh` so that one setting governs
# the whole session. There are two boxes, `xiaohei1` and `xiaohei2`, and having
# `sim.sh` read the variable while the loop needed `--host` meant a scene could
# be restarted on one and driven on the other — both commands succeed, the
# robot is at the origin of a scene nobody is watching, and nothing says so.
#
# `local` means this machine *is* the sim host: no ssh, `docker exec` straight
# into the container. Spelled out rather than left as the empty string, so that
# running on the box is a stated intent and not a variable someone forgot.
LOCAL_HOSTS = {"local", "localhost", "127.0.0.1", ""}
_env_host = os.environ.get("XIAO_HEI_SIM_HOST", "")
DEFAULT_HOST = None if _env_host.lower() in LOCAL_HOSTS else _env_host
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
# Two poses this close are the same place. Sized above `waypointXYRadius`, so
# that landing on a waypoint already visited counts, and well under the 1.4 m
# hops the ring around an object is walked in. See the circling test.
REVISIT_M = 0.5
# Two departures within this angle of each other are the same door. Wide,
# because the model's heading is a guess off a 90° face and the terrain then
# moves it up to another 90°; anything tighter would call the same corridor a
# new one every time.
SPENT_CONE_DEG = 40.0
# What a repeated departure is worth. Not zero — see `explore_direction`.
SPENT_PENALTY = 0.25
# How many times a leg may loop back to a place it has already stood before we
# accept that it cannot find the room. Three is one more than the two doors a
# small hall usually offers, so a leg that is genuinely working its way round
# is not cut off, while one oscillating between two of them is.
MAX_LOOPS = 3
# How far a boxed opening is allowed to be believed. The scanner sees *through*
# a doorway and returns whatever stands behind it, which is what puts the lifted
# point usefully past the threshold at close range and uselessly in the next
# room but one at long range. On `home_building_1` four of seven lifts came back
# beyond 9 m and the first, at 13.69 m, landed at (+8.57, -10.68) — outside the
# whole extent the reference trajectory ever visits — which is what aimed the
# leg east from its first move. Past this, keep the bearing and drop the range.
WAY_MAX_M = 7.0


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


def lift_way(reply: dict, scan: np.ndarray, pose: dict) -> np.ndarray | None:
    """The opening the model boxed, as a point on the floor plan.

    Same lift as a target or a gate anchor, on the one field that says where to
    go rather than what to look at. Returns `None` when nothing was boxed or
    the scanner had no return through it.

    The range is trusted only out to `WAY_MAX_M`. What the model knows is which
    way the opening lies; the distance comes from a ray that went *through* the
    gap and stopped on whatever was behind it, so it is right near to hand and
    meaningless far away. Beyond the cap the bearing is kept and the point is
    pulled back onto it, which makes the robot approach the opening and look
    again rather than commit to a coordinate in another room.
    """
    w = reply.get("way")
    if not isinstance(w, dict) or w.get("box_2d") is None \
            or w.get("image_index") is None:
        return None
    xy = _lift_xy(to_pixels(w["box_2d"], reply.get("coord_space"), G.FACE_SIZE),
                  int(w["image_index"]), scan_to_camera(scan, pose), pose)
    if xy is None:
        return None
    here = np.asarray(pose["position"], float)[:2]
    v = np.asarray(xy, float)[:2] - here
    d = float(np.linalg.norm(v))
    if d <= 1e-6:
        return None
    return here + v / d * min(d, WAY_MAX_M)


def already_tried(origin: np.ndarray, u: np.ndarray,
                  spent: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Has the robot already left roughly here, going roughly this way?

    Same place within `REVISIT_M`, same bearing within `SPENT_CONE_DEG`. Both
    halves are needed: the same heading from a room away is a different move,
    and the same spot by a different door is the whole point of coming back.
    """
    lim = float(np.cos(np.deg2rad(SPENT_CONE_DEG)))
    return any(float(np.linalg.norm(origin - p)) < REVISIT_M
               and float(np.dot(u, v)) > lim for p, v in spent)


def explore_direction(cm: ConverterModel, origin: np.ndarray, want: float,
                      spent: list[tuple[np.ndarray, np.ndarray]] | None = None,
                      ) -> tuple[np.ndarray, float, float]:
    """A direction that is both what the model asked for and drivable.

    The model says which way is worth looking; the terrain says which way the
    vehicle can go. Taking the first without the second drove loft into a wall
    three times — heading 270° had 0.00 m of reach while 30° away had 5.16 m,
    and every one of those refusals cost a grounding call.

    Reach is a **gate, not an objective**. Among the bearings the vehicle can
    actually make `MIN_EXPLORE_M` along, the one nearest what was asked wins;
    how much further any of them run does not enter the score.

    It used to maximise `reach · cos(Δ)`, and that quietly made the loop unable
    to leave a room. A doorway is a *short*-reach bearing — an 0.9 m opening
    onto a small room stops the ray a couple of metres in — while the corridor
    beside it runs five or six. Maximising reach therefore prefers the open
    space to the door, every time, by construction. On `home_building_1` leg 1
    that showed up as: the model named a door it could see on seven of nine
    calls ("the wide framed doorway ... opens into a bedroom-like room with a
    bed and lamp"), five of those nine were swung 15-30° off onto bearings
    reaching 4.4-6.8 m, and the leg walked 20 m for 7 m of net displacement
    without once entering a room.

    `spent` holds departures already made in this leg, and one that repeats is
    scored down by `SPENT_PENALTY` rather than forbidden. The model has no
    record of which door it has already been through, so it keeps nominating
    the one it just came out of; the terrain cannot tell those apart and only
    the history can. Discounting rather than forbidding matters when a room has
    exactly one exit: the way back is then the only legal move, and it must
    stay reachable once everything else has been discounted equally.

    When nothing clears the gate the old rule decides, so a vehicle hemmed in
    on all sides still moves rather than standing still.

    Returns `(unit direction, reach, Δ in degrees)`.
    """
    spent = spent or []
    best = (0.0, np.array([np.cos(want), np.sin(want)]), 0.0, 0.0)
    fallback = (0.0, np.array([np.cos(want), np.sin(want)]), 0.0, 0.0)
    for delta in range(-90, 91, 15):
        th = want + np.deg2rad(delta)
        u = np.array([np.cos(th), np.sin(th)])
        reach = cm.reach_along(origin, u)
        near = float(np.cos(np.deg2rad(delta)))
        if already_tried(origin, u, spent):
            near *= SPENT_PENALTY
        if reach >= MIN_EXPLORE_M and near > best[0]:
            best = (near, u, reach, float(delta))
        far = reach * float(np.cos(np.deg2rad(delta)))
        if far > fallback[0]:
            fallback = (far, u, reach, float(delta))
    win = best if best[0] > 0.0 else fallback
    return win[1], win[2], win[3]


def revisited(here: np.ndarray, stood: list[np.ndarray]) -> bool:
    """Has the vehicle come back to a place it already stood in this leg?

    The immediately previous pose is excluded: a leg that legitimately made a
    short move would otherwise read as a cycle, and a move too short to count
    is already caught by the progress tests. Anything before that is a return.
    """
    return any(float(np.linalg.norm(here - p)) < REVISIT_M for p in stood[:-1])


def corroborated(seen: np.ndarray, pending: list | None) -> bool:
    """Does this reading agree with the last one the binding also refused?

    One reading that disagrees with the binding is what the distance gate
    exists to reject — `japanese_room` produced a lift 4.18 m from a binding
    0.19 m from the truth, and following it drove to the wrong lantern. Two in
    a row that disagree with the binding *and agree with each other* are a
    different thing: the disagreement is now reproducible from two poses, which
    a one-off misread is not.

    On `studio` the two refused readings sat 0.52 m apart and 3.2 m from the
    binding, and both were within 0.82 m of the right window.
    """
    return (pending is not None and len(pending) >= 1
            and float(np.linalg.norm(seen - pending[-1])) <= JUMP_M)


def bind_target(wp, origin: np.ndarray, reply: dict, bound: dict | None,
                rec: dict, *, verified: bool = True,
                measured: bool = False,
                pending: list | None = None) -> tuple[bool, dict | None]:
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
            if pending is not None:
                pending.clear()
        else:
            jump = float(np.linalg.norm(seen - bound["xy"]))
            switched = reply.get("same_object_as_previous") is False
            if jump <= JUMP_M:
                print(f"      binding refined {jump:.2f} m -> "
                      f"({seen[0]:+.2f}, {seen[1]:+.2f})")
                bound = {"xy": seen, "conf": conf,
                         "verified": measured or bound.get("verified", True)}
                # A reading the binding accepted ends any run of ones it did
                # not, so two refusals separated by an agreement never add up.
                if pending is not None:
                    pending.clear()
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
                if pending is not None:
                    pending.clear()
            elif switched and conf >= bound["conf"]:
                # `>` used to be `>=`'s stricter sibling here, and on `studio`
                # that cost the whole leg: the model said `same_object: false`
                # twice, both times at 0.6 against a binding also made at 0.6,
                # and `0.6 > 0.6` kept a binding 3.2 m from the right window
                # against a reading 0.32 m from it. Confidence comes back
                # quantised to a handful of values, so requiring a strict
                # increase is a coin toss dressed as a threshold.
                print(f"      re-bound {jump:.2f} m away — the model reports a "
                      f"different object at no less confidence")
                bound = {"xy": seen, "conf": conf, "verified": measured}
                if pending is not None:
                    pending.clear()
            elif corroborated(seen, pending):
                # Two readings in a row that agree with each other and not with
                # the binding are evidence about the binding, not about
                # themselves. This is the same rule as everywhere else here —
                # measure rather than ask — applied to the binding itself, and
                # it is what makes the arbitration independent of the model's
                # self-report, which `bind_target` already documents as
                # uncalibrated in both directions.
                print(f"      re-bound {jump:.2f} m away — two readings in a "
                      f"row landed within {JUMP_M} m of each other and this far "
                      f"from the binding")
                bound = {"xy": seen, "conf": conf, "verified": measured}
                rec["binding_corroborated"] = True
                pending.clear()
            else:
                print(f"      this lift lands {jump:.2f} m from the binding "
                      f"while still calling it the same object — keeping the "
                      f"binding at ({bound['xy'][0]:+.2f}, {bound['xy'][1]:+.2f})")
                rec["binding_rejected"] = {"seen": seen.tolist(), "jump_m": jump}
                if pending is not None:
                    pending.append(seen)
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
           visited: list[str] | None = None,
           mission: dict | None = None) -> tuple[dict | None, str]:
    """The parsed reply and the text it came from.

    The raw text is returned because three runs died on "unparseable reply"
    while discarding the only evidence of why. It was truncation.
    """
    fn = ask_claude if backend == "claude" else ask_gemini
    text = fn(build_prompt(phrase, approach=True, version=version,
                           visited=visited, mission=mission),
              faces, model, previous=prev)
    return parse(text), text


@dataclass
class Ctx:
    """What outlives one clause of a plan.

    A question is a sequence of legs driven without ever resetting the vehicle,
    so some state belongs to the run and some to the leg, and putting a piece
    on the wrong side is a real bug either way. `visited` and `avoid` are the
    run's: the places the robot has stood, and the regions it must keep out of,
    are facts about the whole trajectory. `bound` and `prev_crop` are the leg's
    and stay local to `run_goto` — carrying a binding into the next clause
    would aim the next leg at the previous leg's object.
    """

    robot: Robot
    out: Path
    log: object
    backend: str = "claude"
    model: str = "claude-opus-5"
    prompt_version: str = DEFAULT_PROMPT_VER
    standoff: float = STANDOFF_M
    dry_run: bool = False
    visited: list[str] = field(default_factory=list)
    avoid: list[dict] = field(default_factory=list)
    calls: int = 0
    step: int = 0
    deadline: float | None = None
    mission: dict | None = None

    # A per-leg deadline, set by the executor from the time left divided among
    # the clauses still to drive. `deadline` is the question's; this one stops
    # an early leg from spending the budget the later ones need.
    leg_deadline: float | None = None

    def out_of_time(self) -> bool:
        return (self.deadline is not None and time.time() >= self.deadline) or \
               (self.leg_deadline is not None and time.time() >= self.leg_deadline)

    def left(self) -> float:
        ends = [d for d in (self.deadline, self.leg_deadline) if d is not None]
        return float("inf") if not ends else min(ends) - time.time()

    def whole_left(self) -> float:
        """Time left for the question, ignoring this leg's share of it."""
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def record(self, rec: dict) -> None:
        self.log.write(json.dumps(rec, default=str) + "\n")
        self.log.flush()

    def mission_for(self, k: int) -> dict | None:
        return None if not self.mission else {**self.mission, "k": k}


@dataclass
class Outcome:
    """How one clause ended, and what it learned that the next leg can use."""

    arrived: bool
    why: str
    xy: np.ndarray | None = None          # where the target was bound
    prev_crop: bytes | None = None        # last view of it, for the confirm call


def run_goto(ctx: Ctx, phrase: str, *, max_steps: int = 6, k: int = 1,
             confirm: bool = False) -> Outcome:
    """Drive to one named object, starting from wherever the vehicle stands.

    Extracted from `main` so a plan can call it once per destination without
    resetting the pose between them. `ctx.step` keeps counting across legs, so
    the artefacts on disk stay in the order they were captured and a step
    number is unique within a run.
    """
    prev_crop, bound, arrived = None, None, False
    misses = stuck_explores = loops = 0
    stood: list[np.ndarray] = []
    # Departures already made in this leg, as (from, unit direction). The model
    # cannot remember which door it has been through; this can.
    spent: list[tuple[np.ndarray, np.ndarray]] = []
    # Readings the binding refused, so that two in a row agreeing with each
    # other can overrule it. Per-leg: the next clause is a different object.
    pending: list[np.ndarray] = []

    for _ in range(max_steps):
        if ctx.out_of_time():
            return Outcome(False, "out of time", None, prev_crop)
        ctx.step += 1
        step = ctx.step
        eq, scan, terrain, pose = ctx.robot.capture()
        faces = faces_of(eq)
        for i, f in enumerate(faces):
            (ctx.out / f"step{step}_face{i}.jpg").write_bytes(f)
        # Keep the geometry too, not just the pictures. Without the terrain the
        # converter's choice cannot be re-derived after the fact, which is how a
        # bad waypoint on chinese_room went unexplained.
        np.save(ctx.out / f"step{step}_terrain.npy", terrain)
        np.save(ctx.out / f"step{step}_scan.npy", scan)

        reply, raw = ground(faces, phrase, ctx.backend, ctx.model, prev_crop,
                            ctx.prompt_version, ctx.visited, ctx.mission_for(k))
        ctx.calls += 1
        rec: dict = {"step": step, "clause": k, "phrase": phrase,
                     "kind": GOTO, "pose": pose, "reply": reply}
        if reply is None:
            print(f"[{step}] unparseable reply ({len(raw)} chars); stopping")
            print(f"      tail: ...{raw[-200:]!r}")
            rec["raw"] = raw
            ctx.record(rec)
            return Outcome(False, "unparseable reply", None, prev_crop)

        o = np.asarray(pose["position"], float)
        print(f"[{step}] at ({o[0]:+.2f}, {o[1]:+.2f})  "
              f"visible={reply.get('visible')}  conf={reply.get('confidence')}  "
              f"state={reply.get('target_state')}  "
              f"same_as_prev={reply.get('same_object_as_previous')}")

        # Circling. The legal points around an object form a ring at roughly
        # equal distance from it, and `NEAR_M` was meant to stop the loop
        # walking that ring forever — but it is a distance, calibrated on
        # floor-standing furniture, and on `studio` the target was a skylight
        # in a sloped attic roof with a bookshelf beneath it. The platform's
        # floor there is 2.1 m, above `NEAR_M`, so every step read as "not close
        # enough to be the floor" and the leg burned all five calls returning to
        # within 0.06 m of where it had stood two steps earlier.
        #
        # Coming back to a place already stood in needs no constant to detect,
        # and means the same thing at any distance: there is nothing further to
        # be gained by moving. Whether that counts as arriving still depends on
        # whether the target was ever bound.
        if revisited(o[:2], stood):
            if bound is not None:
                d = float(np.linalg.norm(bound["xy"] - o[:2]))
                print(f"      back where it already stood — the ring around the "
                      f"target has been walked; {d:.2f} m is the floor here")
                rec["arrived"] = f"circled back ({d:.2f} m)"
                ctx.record(rec)
                return Outcome(True, f"arrived, circled back ({d:.2f} m)",
                               bound["xy"], prev_crop)
            # With nothing bound this used to end the leg, and that was wrong.
            # Backing out of a dead end and returning to the hall to try another
            # door is what searching a building *is*; the constant was written
            # for a leg circling an object it had already found, where coming
            # back means there is nothing more to gain. Here it means the
            # opposite. On `home_building_1` it fired at step 8 of a leg holding
            # 400 s and threw away 217 of them — the corridor to the bedrooms was
            # then spotted five steps later, by a different clause.
            #
            # So: count it, tell the model, discount the ways already taken, and
            # keep going. The leg still ends — on its own time slice, or after
            # `MAX_LOOPS` returns, which is a bound on fruitless oscillation
            # rather than on searching.
            loops += 1
            print(f"      back where it already stood, nothing bound "
                  f"({loops}/{MAX_LOOPS}) — the way taken from here led nowhere "
                  f"new; discounting it and looking again")
            rec["looped"] = loops
            ctx.visited.append(
                "(the robot came back to this spot after leaving it — whatever "
                "route it took from here revealed nothing new, so send it a "
                "different way)")
            if loops >= MAX_LOOPS:
                print(f"      {loops} returns to the same ground with nothing "
                      f"bound — this leg is going in circles")
                rec["stopped"] = "circling with nothing bound"
                ctx.record(rec)
                return Outcome(False, "circling with nothing bound", None,
                               prev_crop)
        stood.append(o[:2].copy())

        # Before the visibility branch: a keep-out anchor is most likely to be
        # reported on exactly the calls where the *target* is not visible,
        # because that is when the robot is looking around at the furniture.
        here_txt = (reply.get("here") or "").strip()
        if here_txt:
            ctx.visited.append(here_txt)
            rec["here"] = here_txt
            print(f'      here: "{here_txt[:96]}"')

        ctx.avoid = bind_constraints(reply, scan, pose, ctx.avoid)
        keepout = [(a["xy"], KEEPOUT_M) for a in ctx.avoid]

        # Resolved before the visibility branch, because whether the phrase's
        # relation could be *measured* now decides whether a sighting counts as
        # having found anything. Whether the phrase needs checking is a property
        # of the phrase: the model reported no relation for "the guitar near the
        # couch" on one call and `closest_to` on the next, and the call that
        # forgot was treated as nothing to verify.
        relational = has_relation(phrase) or reply.get("relation") in RELATIONS
        chosen = (resolve_relation(reply, scan, pose, size=G.FACE_SIZE)
                  if reply.get("visible") and reply.get("image_index") is not None
                  else None)
        # A relational phrase whose anchor never lifted, with nothing bound: the
        # model has found *a* thing of the right type, not the one the phrase
        # names. On `home_building_1` it found a waste bin in the bathroom while
        # the question asked for the one nearest a refrigerator 14.6 m away in
        # another room — and treating that sighting as an arrival target sent
        # the leg to circle the wrong bin until it ran out of calls.
        #
        # The model was not wrong; it was not being listened to. Its `explore`
        # field said, at that exact step, "the kitchen (and therefore the
        # refrigerator and any kitchen trash can) lies past the dining area
        # behind the robot", and the loop discarded it because `visible` was
        # true. So an unverified sighting keeps exploring, along the heading the
        # model gives, instead of driving at the nomination.
        adrift = (reply.get("visible") and relational and chosen is None
                  and bound is None)
        if adrift:
            print(f"      seen, but the relation is unmeasurable and nothing is "
                  f"bound — this is the right kind of object, not the one the "
                  f"phrase names; still searching")
            rec["adrift"] = True

        if not reply.get("visible") or adrift:
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
            asked = u = None
            # An opening the model could actually see beats any bearing: a
            # doorway is a short-reach direction with a long corridor beside it,
            # so "through that door" and "along the wall past it" are only a few
            # degrees apart as headings and a room apart as destinations.
            way = lift_way(reply, scan, pose)
            if way is not None and float(np.linalg.norm(way - o[:2])) \
                    < MIN_VIEW_MOVE_M:
                print(f"      way out boxed but already at it — falling back "
                      f"to the heading")
                way = None
            try:
                cm = ConverterModel(terrain, keepout=keepout)
                want = yaw_of(pose) - np.deg2rad(float(h))
                asked = cm.reach_along(
                    o[:2], np.array([np.cos(want), np.sin(want)]))
                if way is not None:
                    # Aim past the opening, not at it. The scanner returns the
                    # far side of the gap, and a waypoint on the threshold parks
                    # the vehicle in the doorway the way a passage midpoint did.
                    best = cm.best_waypoint_toward(way, o[:2],
                                                   min_move=MIN_VIEW_MOVE_M)
                    goal = way if best is None else best[0]
                else:
                    u, reach, delta = explore_direction(cm, o[:2], want, spent)
                    if reach >= MIN_EXPLORE_M:
                        goal = o[:2] + u * min(reach, MAX_EXPLORE_M)
                        best = cm.best_waypoint_toward(goal, o[:2],
                                                       min_move=MIN_VIEW_MOVE_M)
                        if best is not None:
                            goal = best[0]
            except ValueError as e:
                print(f"      converter model unavailable ({e})")
            if way is not None:
                nm = (reply.get("way") or {}).get("name") or "opening"
                wd = float(np.linalg.norm(way - o[:2]))
                capped = " (range capped; bearing kept)" \
                    if wd >= WAY_MAX_M - 1e-3 else ""
                print(f"      way out: {nm!r} lifted to "
                      f"({way[0]:+.2f}, {way[1]:+.2f}), "
                      f"{wd:.2f} m away{capped}")
            elif asked is not None:
                repeat = u is not None and already_tried(o[:2], u, spent)
                print(f"      heading {h}° reaches {asked:.2f} m; best drivable "
                      f"is {delta:+.0f}° off it at {reach:.2f} m"
                      f"{' (already taken from here)' if repeat else ''}")
            print(f"      NOT_VISIBLE — heading {h}°, "
                  f"driving to ({goal[0]:+.2f}, {goal[1]:+.2f})")
            rec["action"] = {"kind": "explore", "heading_deg": h,
                             "goal": goal.tolist(), "reach_m": reach,
                             "delta_deg": delta,
                             "way": None if way is None else way.tolist()}
            # Remember the departure, not the heading the model asked for: what
            # must not be repeated is the move actually made. Recorded from the
            # goal so the fallback path — no converter, no `u` — is covered too.
            step_v = goal - o[:2]
            if float(np.linalg.norm(step_v)) > 1e-6:
                spent.append((o[:2].copy(),
                              step_v / float(np.linalg.norm(step_v))))
            if not ctx.dry_run:
                rec["drive"] = ctx.robot.drive_to(goal[0], goal[1],
                                                  min(20.0, ctx.left()))
                if (rec["drive"].get("moved_m") or 0.0) < PROGRESS_M:
                    stuck_explores += 1
                else:
                    stuck_explores = 0
            ctx.record(rec)
            prev_crop = None
            if stuck_explores >= 2:
                print(f"      two exploration legs in a row went nowhere — the "
                      f"heading is not reachable from here; stopping")
                return Outcome(False, "heading not reachable", None, prev_crop)
            continue
        misses = 0

        i = int(reply["image_index"])
        box = to_pixels(reply.get("feature_box_2d") or reply["box_2d"],
                        reply.get("coord_space"), G.FACE_SIZE)
        # A comparative relation is decided by measuring the candidates, not by
        # whichever one the model nominated; `chosen` was resolved above.
        if chosen is not None:
            box, i, rel_why = chosen
            rec["relation"] = rel_why
            print(f"      relation resolved: {rel_why}")
        elif reply.get("relation"):
            print(f"      relation {reply['relation']!r} not measurable "
                  f"({len(reply.get('candidates') or [])} candidates, "
                  f"{len(reply.get('anchors') or [])} anchors) — using the "
                  f"model's own pick")
        verified = (not relational) or chosen is not None
        w, h_deg = box_angular_size(box, i)
        blind, az, el, floor = in_blind_cone(ray_from_box(box, i))
        print(f"      image {i} ({NAMES[i]}), box {w:.1f}x{h_deg:.1f}°, "
              f"bearing {az:+.0f}°/{el:+.0f}° "
              f"({'BLIND' if blind else 'covered'}, floor {floor:+.0f}°)")

        wp = next_waypoint(box, i, scan, pose, phrase=phrase,
                           standoff=ctx.standoff)
        rec["waypoint"] = {"xy": wp.xy.tolist(), "committed": wp.committed,
                           "range_m": wp.range_m, "reason": wp.reason,
                           "blind": blind, "az": az, "el": el}
        print(f"      -> ({wp.xy[0]:+.2f}, {wp.xy[1]:+.2f}) "
              f"[{'DESTINATION' if wp.committed else 'step'}]  {wp.reason}")

        committed, bound = bind_target(wp, o[:2], reply, bound, rec,
                                       verified=verified,
                                       measured=chosen is not None,
                                       pending=pending)

        # Already inside the standoff: driving further would push into the
        # object, and the stack would only snap the waypoint back out again.
        if committed and bound is not None:
            here = float(np.linalg.norm(bound["xy"] - o[:2]))
            if here <= ctx.standoff:
                print(f"      already within {ctx.standoff} m — arrived")
                rec["arrived"] = "within standoff"
                ctx.record(rec)
                arrived, bound_xy = True, bound["xy"]
                break

        # What the converter will do with this waypoint, before we spend a
        # drive and another grounding call finding out. TASK 28 read a 1.08 m
        # displacement as the platform clamping an approach; it was the
        # converter discarding the waypoint and re-minimising elsewhere.
        goal, will_move, cm = wp.xy, None, None
        aim = bound["xy"] if bound is not None else wp.xy
        try:
            cm = ConverterModel(terrain, keepout=keepout)
            # Aim at the target itself, not at a standoff from it: the standoff
            # is what the converter's inflation is *for*, and asking for a point
            # inside it gets the waypoint discarded rather than clamped. Once
            # the target is bound, the binding is the better estimate of where
            # it is than any single reading.
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
            ctx.record(rec)
            if committed:
                print(f"      no legal point closer than where we stand — this "
                      f"is as near as the platform allows")
                rec["arrived"] = "no legal point closer (predicted)"
                arrived, bound_xy = True, aim
                break
            print(f"      stuck: the lift is untrustworthy here and the "
                  f"converter has nowhere legal to move us")
            rec["stopped"] = "stuck (untrusted lift, no legal move)"
            return Outcome(False, "stuck (untrusted lift, no legal move)",
                           None, prev_crop)

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
                ctx.record(rec)
                return Outcome(False, "boxed in (no legal move)", None, prev_crop)
            goal, lands, reach = alt
            will_move = float(np.linalg.norm(lands - o[:2]))
            rec["converter"]["requeried_for_motion"] = {
                "goal": goal.tolist(), "settles_at": lands.tolist(),
                "settle_to_aim_m": reach, "will_move_m": will_move}
            print(f"      that goal would not move the vehicle; going to "
                  f"({goal[0]:+.2f}, {goal[1]:+.2f}) instead -> settles "
                  f"{reach:.2f} m from it, {will_move:.2f} m from here")

        prev_crop = crop_face(faces[i], box)
        (ctx.out / f"step{step}_target.jpg").write_bytes(prev_crop)

        if ctx.dry_run:
            ctx.record(rec)
            return Outcome(False, "dry run", aim, prev_crop)

        dist = float(np.linalg.norm(goal - o[:2]))
        res = ctx.robot.drive_to(goal[0], goal[1],
                                 min(max(12.0, dist / 0.4 + 8.0), ctx.left()))
        rec["drive"] = res
        print(f"      drive: {res.get('why')}  moved {res.get('moved_m')}  "
              f"final gap to requested point {res.get('dist_to_requested_m')}")
        ctx.record(rec)

        if res.get("why") == "timeout":
            print("      drive timed out; stopping")
            return Outcome(False, "drive timed out", None, prev_crop)
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
                    return Outcome(False, "boxed in (stack would not move us)",
                                   None, prev_crop)
                print(f"      stack will not close the last {gap:.2f} m "
                      f"(moved {moved:.2f} m) — as near as it allows")
                arrived, bound_xy = True, aim
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
            return Outcome(False, "step made no progress", None, prev_crop)
    else:
        return Outcome(False, f"gave up after {max_steps} steps", None, prev_crop)

    if not arrived:
        return Outcome(False, "did not arrive", None, prev_crop)

    # One last look, purely to record whether the two new fields agree with the
    # geometry that actually decided this. They gate nothing.
    if confirm and not ctx.dry_run:
        eq, scan, terrain, pose = ctx.robot.capture()
        try:
            got, _ = ground(faces_of(eq), phrase, ctx.backend, ctx.model,
                            prev_crop, ctx.prompt_version, ctx.visited,
                            ctx.mission_for(k))
            ctx.calls += 1
        except Exception as e:
            # Advisory only. Losing it must not turn a completed run into a
            # failed one.
            print(f"\nconfirm call failed ({type(e).__name__}); arrival stands")
            got = None
        if got:
            print(f"\nconfirm: visible={got.get('visible')} "
                  f"state={got.get('target_state')} "
                  f"same_object={got.get('same_object_as_previous')} "
                  f"conf={got.get('confidence')}")
        ctx.record({"step": "confirm", "clause": k, "pose": pose, "reply": got})

    return Outcome(True, "arrived", bound_xy, prev_crop)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrase")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help="ssh target running the sim (default: $XIAO_HEI_SIM_HOST, "
                         f"currently {DEFAULT_HOST or 'unset'}); omit and leave "
                         "the variable unset if this IS the sim host")
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
    ctx = Ctx(robot=robot, out=out, log=log, backend=args.backend, model=model,
              prompt_version=args.prompt_version, standoff=args.standoff,
              dry_run=args.dry_run)
    res = run_goto(ctx, args.phrase, max_steps=args.max_steps, confirm=True)
    log.close()
    print(f"\n{'ARRIVED' if res.arrived else 'did not arrive'} ({res.why}) in "
          f"{ctx.calls} calls (${ctx.calls * COST_PER_CALL:.2f})   "
          f"log: {out}/steps.jsonl")
    return 0 if res.arrived else 1



if __name__ == "__main__":
    raise SystemExit(main())
