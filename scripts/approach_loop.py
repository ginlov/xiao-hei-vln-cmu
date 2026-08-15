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
import re
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
from vlm_approach import (STANDOFF_M, _lift_xy, aim_box,  # noqa: E402
                          box_angular_size, crop_face, has_relation,
                          in_blind_cone, next_waypoint, ray_from_box,
                          resolve_relation)
from vlm_locate import rot_from_quat, scan_to_camera  # noqa: E402
from waypoint_converter_model import (WAYPOINT_XY_RADIUS,  # noqa: E402
                                      ConverterModel)
from vlm_probe import (DEFAULT_GEMINI_MODEL, DEFAULT_PROMPT_VER,  # noqa: E402
                       NAMES, ask_claude, ask_gemini, build_prompt, parse,
                       settle_coord_space, to_pixels)
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
# How near the binding must be for "came back to where it stood" to mean the
# ring around the target has been walked, rather than the leg being stuck. The
# platform will not park inside `obstacleDisThre` (0.75 m) of furniture and
# measured floors run 1.1-1.5 m to an object centre, so 2.5 m covers a real
# ring with room for a lift error and excludes what `livingroom_2` reported:
# two shuffles inside half a metre, called arrival with the binding 9.79 m off.
CIRCLE_ARRIVE_M = 2.5
# How far a nominated object may be from the anchor its phrase says it is
# beside before the nomination stops being evidence. Measured over the released
# questions, where both nouns appear in `object_list.txt`: 1.20 m at the median,
# 3.39 m at p95, 4.65 m at the worst honest case (`office_1`, "the bench
# closest to the map wall decal"). Set well past that because the consequence
# is a demotion and not a rejection — a false refusal here costs the whole leg,
# while a false pass costs one more call. The dice ornament `livingroom_2` bound
# and drove to sat 6.09 m from the nearest couch.
RELATION_MAX_M = 6.0
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

# How the "places already searched" block is written. `prose` is the original
# and the exact rollback; `bearing` and `xy` are geometric; `off` removes the
# block. Env so a sim run can be switched without editing anything, overridden
# by `--visited`. See `Ctx.visited_for`.
VISITED_STYLES = ("prose", "bearing", "xy", "off")
VISITED_STYLE = os.environ.get("XIAO_HEI_VISITED", "prose")


def says_far(reply: dict | None) -> bool:
    """Does the model say the target still reads as part of the scene?

    `target_state` is the one distance-ish field that is not a distance. The
    prompt asks for it "judged only from how the target sits in the frame — how
    much of the view it fills, whether the frame cuts it off, whether you can
    make out surface detail", and forbids deriving it from the metre estimate.
    That matters, because the metre estimate is the field that cannot be
    trusted: `distance_m` has a median error of 2.80 m, worse than answering
    with a constant. `far` only needs the model to get "small in the view"
    right, and it is looking at the picture.

    It is also the only signal here that is independent of the scanner, which
    is what makes it worth having. On `runs/o_2_0814_02` all three legs stopped
    at a binding the model put 2.8x, 3.0x and 3.6x further away — but on two of
    them the scanner was right and `distance_m` was wrong (a 10° cone holds no
    returns anywhere near the model's figure), so a rule built on the metres
    would have vetoed two correct arrivals to catch one bad one. `target_state`
    separates them exactly: those two read `approaching`, and the leg that
    stopped 6.5 m short of a door behind glass read `far`.

    Over the 38 recorded arrivals that carry a reply, 8 (21%) were declared on
    a `far`, with the model putting the target 3.6-9.8 m away — including one
    that arrived on a binding 9.79 m from the vehicle.
    """
    return bool(reply) and reply.get("target_state") == "far"


def FACE_OF(heading_deg: float) -> str:
    """Which of the four images a heading falls in, in the model's words."""
    return ("front", "front-right", "right", "back-right", "back",
            "back-left", "left", "front-left")[int(((heading_deg % 360) + 22.5)
                                                   // 45) % 8]
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
# A lift's positional error is a bearing error times a range, so a reading
# taken from half as far away carries about half the uncertainty. Below this
# ratio the newer reading is the better measurement of the two and replaces the
# binding however far it lands from it — `JUMP_M` is a fixed metre and cannot
# tell a disagreement from the leverage of a long lift. On `livingroom_2` the
# binding was made at 11.08 m and the reading that landed 0.12 m from the true
# soccer ball was made at 3.64 m; the gate refused it for jumping 3.52 m, and
# the leg finished 2.77 m short. Half is deliberately conservative: at 0.33 it
# would have fired here with room to spare.
NEARER_RATIO = 0.5
# How far a boxed opening is allowed to be believed. The scanner sees *through*
# a doorway and returns whatever stands behind it, which is what puts the lifted
# point usefully past the threshold at close range and uselessly in the next
# room but one at long range. On `home_building_1` four of seven lifts came back
# beyond 9 m and the first, at 13.69 m, landed at (+8.57, -10.68) — outside the
# whole extent the reference trajectory ever visits — which is what aimed the
# leg east from its first move. Past this, keep the bearing and drop the range.
WAY_MAX_M = 7.0
# How far past each lifted anchor a forbidden gate reaches. The lift lands on
# whichever face the scanner saw, so the segment joining two anchors is short
# of the furniture it names at both ends; half a sofa plus half the vehicle is
# what this has to cover. See `gates_from`.
GATE_PAD_M = 0.6
# How far a single step may go while a keep-out is in force. The constraint is
# checked on the straight line from the vehicle to where the waypoint settles,
# and that line is only a fair model of the driven path over a short hop —
# `local_planner` curves. On `livingroom_2` one 4.83 m drive to a waypoint
# comfortably outside the forbidden region went through the middle of it.
KEEPOUT_STEP_M = 2.0
# How far past the floor the model names to put the waypoint. Enough to clear
# `obstacleDisThre` (0.75 m) from the furniture on either side of it, so that a
# legal point exists there at all; short enough that the step cap above still
# governs how far the vehicle actually goes. See `past`.
DETOUR_BEYOND_M = 1.0
# Whether a keep-out is steered around at all: the gate, the discs, the step
# cap and the model-led detour, together. Off, and the reason is arithmetic.
#
# README §175 penalises a trajectory that "passes through areas it is forbidden
# to go through", and scores 0-6 "with possibility for partial points" — so
# driving through a forbidden region is a deduction, while failing to reach a
# destination forfeits that destination outright. Enforced, `livingroom_2` q5
# reached neither the soccer ball nor partial credit: from the pose the leg
# arrived at, every legal waypoint more than half a metre south lay inside the
# forbidden corridor, and the strip the reference trajectory threads holds no
# legal point at all, being under `obstacleDisThre` from furniture on both
# sides. The robot was correctly refusing the only way there was, and shuffling
# 0.10 m at a time while it did. Unenforced, the same run reaches both
# destinations and loses one penalty.
#
# Three of the thirty released instruction questions carry a keep-out. This
# trades a deduction on those three for the destinations on them, and costs
# the other twenty-seven nothing.
#
# The machinery stays: `XIAO_HEI_KEEPOUT=1` turns all of it back on, and the
# analysis that would justify doing so is in TASK 37.
USE_KEEPOUT = os.environ.get("XIAO_HEI_KEEPOUT", "0") in ("1", "true", "yes")


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
        name = it.get("name") or "?"
        # By name first, and only then by distance. Distance alone made three
        # discs out of one television on `livingroom_2`: the lifts came back
        # 1.3 m and 2.2 m apart as the robot moved, `JUMP_M` called each a new
        # object, and five 1.2 m discs closed every route the leg had. The
        # model names them consistently — "TV", "tv", "tea table" — and that is
        # the more reliable half of the answer.
        near = next((a for a in avoid if same_thing(a["name"], name)), None) \
            or next((a for a in avoid
                     if np.linalg.norm(a["xy"] - xy) < JUMP_M), None)
        if near is None:
            avoid.append({"xy": xy, "name": name})
            print(f"      keep-out bound: {name!r} at "
                  f"({xy[0]:+.2f}, {xy[1]:+.2f}), radius {KEEPOUT_M} m")
        else:
            near["xy"] = xy          # same anchor, seen better
    return avoid


def nearest_allowed_step(cm: ConverterModel, here: np.ndarray,
                         aim: np.ndarray) -> np.ndarray | None:
    """The legal waypoint nearest the aim whose *route from here* is allowed.

    `best_waypoint_toward` scores where the vehicle would settle, and returns
    nothing at all when every candidate is forbidden. That is the right answer
    to the question it was asked and the wrong thing to act on: a leg with a
    keep-out still has to move, and moving toward the aim by whatever the
    constraint permits is what walks the vehicle round the forbidden corridor
    over the next few calls.

    Cheaper than `best_waypoint_toward` on purpose — no settle simulation, just
    the published point — because this runs only when that has already failed.
    """
    L = cm.legal_points()
    if not len(L):
        return None
    ok = np.array([not (cm.gates and cm.crosses_gate(here, p))
                   and not (cm.keepout and cm._crosses_keepout(here, p))
                   for p in L])
    if not ok.any():
        return None
    C = L[ok]
    return C[int(np.argmin(np.linalg.norm(C - np.asarray(aim, float)[:2],
                                          axis=1)))]


def gates_from(avoid: list[dict], pad: float = GATE_PAD_M
               ) -> list[tuple[np.ndarray, np.ndarray]]:
    """The forbidden corridor between two keep-out anchors, as a segment.

    "Avoid the path between the TV and the tea table" forbids a corridor, and a
    corridor is what the two anchors bracket — not two discs centred on them.
    The pair is the two furthest apart, as in `gate_point`, because a gap is
    defined by its sides.

    Both ends are pushed outward by `pad`. The anchors are lifted at the face
    the scanner happened to see, so the segment joining them is shorter than
    the furniture it names at both ends; without the pad, a route that "clears"
    the gate can miss the tea table's centre by 0.03 m, which is to say drive
    through it.
    """
    named = [a for a in avoid if a.get("xy") is not None]
    pairs = [(a, b) for i, a in enumerate(named) for b in named[i + 1:]
             if not same_thing(a["name"], b["name"])]
    if not pairs:
        return []
    a, b = max(pairs, key=lambda p: float(
        np.linalg.norm(p[0]["xy"] - p[1]["xy"])))
    p, q = np.asarray(a["xy"], float), np.asarray(b["xy"], float)
    n = float(np.linalg.norm(q - p))
    if n < 0.5:
        return []                       # one object reported twice, not a gap
    u = (q - p) / n
    return [(p - u * pad, q + u * pad)]


def lift_boxed(reply: dict, field: str, scan: np.ndarray,
               pose: dict) -> np.ndarray | None:
    """A place the model boxed, as a point on the floor plan.

    Same lift as a target or a gate anchor, on the fields that say where to go
    rather than what to look at -- `way` when the target is out of sight,
    `detour` when a keep-out stands between the robot and it. Returns `None`
    when nothing was boxed or the scanner had no return through it.

    The range is trusted only out to `WAY_MAX_M`. What the model knows is which
    way the place lies; the distance comes from a ray that went *through* the
    gap and stopped on whatever was behind it, so it is right near to hand and
    meaningless far away. Beyond the cap the bearing is kept and the point is
    pulled back onto it, which makes the robot approach and look again rather
    than commit to a coordinate in another room.
    """
    w = reply.get(field)
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


def lift_way(reply: dict, scan: np.ndarray, pose: dict) -> np.ndarray | None:
    """The opening onward, when the target is not in sight."""
    return lift_boxed(reply, "way", scan, pose)


def past(here: np.ndarray, there: np.ndarray,
         beyond: float = DETOUR_BEYOND_M) -> np.ndarray:
    """`there`, pushed further along the same bearing.

    A detour names floor the vehicle should *drive over*, and there is often no
    waypoint to be had on it: `obstacleDisThre` (0.75 m) governs where a
    waypoint may be placed, not where the vehicle may drive, and "the clear
    floor between the tea table and the sofa" is by construction within 0.75 m
    of two pieces of furniture. On `livingroom_2` the nearest legal point to
    the detour was 0.96 m from it and moved the vehicle 0.10 m; the leg
    shuffled twice and gave up.

    Aiming past it is the same answer `through_point` gives for a passage: the
    waypoint goes where one can go, and the shortest legal way to it crosses
    the floor that was named.
    """
    v = np.asarray(there, float)[:2] - np.asarray(here, float)[:2]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.asarray(there, float)[:2]
    return np.asarray(there, float)[:2] + v / n * beyond


def lift_detour(reply: dict, scan: np.ndarray, pose: dict) -> np.ndarray | None:
    """Where to aim next to get past a keep-out.

    The model cannot see the path the stack will take and cannot express "round
    the west end of the tea table" as a heading, but it can see the floor and
    point at the piece of it to cross next. That is the half of the problem the
    geometry keeps getting wrong for a different reason: on `livingroom_2` the
    tea table lifted 2.0 m from where it is, so the forbidden corridor was
    drawn across the wrong part of the room and the vehicle drove through the
    real one without ever crossing the computed one. Deciding *which side to
    pass* needs no coordinate at all.
    """
    return lift_boxed(reply, "detour", scan, pose)


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


def same_thing(a: str, b: str) -> bool:
    """Are these two reported names the same object seen twice?

    On `studio` the model returned `couch` and `couch (left view)` as the two
    sides of a gap, 1.35 m apart — wide enough to clear the span guard, and a
    "passage" straight through the middle of one sofa. The parenthetical is the
    model's own note about which image it read, so it is stripped before the
    comparison.

    It lives here rather than in `execute_plan`, where it was written, because
    `bind_constraints` needs it too and `execute_plan` imports this module.
    """
    def norm(s: str) -> str:
        s = re.sub(r"\(.*?\)", " ", (s or "").lower())
        s = " ".join(s.replace("the ", " ").split())
        return s
    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and (x in y or y in x)


def relation_holds(reply: dict, seen: np.ndarray, scan: np.ndarray,
                   pose: dict) -> tuple[bool, str]:
    """Is the nominated object anywhere near the thing it is said to be near?

    The relation is used to *choose* between candidates and never to check the
    one candidate there usually is. On `livingroom_2` the phrase was "the
    soccer ball near the couch" and the loop bound a 0.22 m dice ornament on a
    bookshelf — 6.09 m from the nearest couch, 5.42 m from the ball — then
    drove to it and reported arrival. The model's own `anchors` were in the
    reply the whole time; nothing compared the answer against them.

    Measured over the released questions, an object said to be near another is
    1.20 m from it at the median, 3.39 m at p95, and 4.65 m at the worst
    (`office_1`, "the bench closest to the map wall decal"). `RELATION_MAX_M`
    sits well past that, because the consequence here is not a rejection.

    Returns `(holds, why)`. `holds` is True whenever there is nothing to check:
    no relation named, no anchor that lifted, or a relation whose sense is not
    proximity. Absence of evidence does not fail a binding.
    """
    if reply.get("relation") not in ("closest_to", "near", "between"):
        return True, ""
    anchors = reply.get("anchors") or []
    if not isinstance(anchors, list) or not anchors or not len(scan):
        return True, ""
    try:
        scan_cam = scan_to_camera(scan, pose)
    except (KeyError, TypeError, ValueError):
        return True, ""     # a frame we cannot read is not a failed relation
    space = reply.get("coord_space")
    lifted = []
    for it in anchors:
        if not isinstance(it, dict) or it.get("box_2d") is None \
                or it.get("image_index") is None:
            continue
        xy = _lift_xy(to_pixels(it["box_2d"], space, G.FACE_SIZE),
                      int(it["image_index"]), scan_cam, pose)
        if xy is not None:
            lifted.append((np.asarray(xy, float)[:2], it.get("name") or "?"))
    if not lifted:
        return True, ""
    d, name = min(((float(np.linalg.norm(np.asarray(seen, float)[:2] - xy)), nm)
                   for xy, nm in lifted), key=lambda t: t[0])
    if d <= RELATION_MAX_M:
        return True, ""
    return False, (f"{d:.2f} m from {name!r}, which the phrase says it is "
                   f"beside")


def side_of(a, b, p) -> float:
    """Which side of the line `ab` the point `p` lies on, as -1, 0 or +1."""
    a, b, p = (np.asarray(v, float) for v in (a, b, p))
    return float(np.sign((b[0] - a[0]) * (p[1] - a[1])
                         - (b[1] - a[1]) * (p[0] - a[0])))


def crosses(a, b, c, d) -> bool:
    """Do segments `ab` and `cd` properly intersect?

    Straight orientation test. Both segments are short and the degenerate
    collinear case is not interesting here: a track that runs exactly along the
    line between two anchors has not gone between them either.

    It lives here rather than in `execute_plan`, which is where it was written,
    because the exploration memory below needs it too and `execute_plan`
    imports this module.
    """
    return (side_of(a, b, c) * side_of(a, b, d) < 0
            and side_of(c, d, a) * side_of(c, d, b) < 0)


def recrosses(origin: np.ndarray, u: np.ndarray,
              crossed: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
              reach: float = MAX_EXPLORE_M) -> bool:
    """Would going this way take the robot back through a passage it has used?

    A required passage is scored on the *trajectory*, in order (README §175),
    so driving back out through one already satisfied does not merely waste
    time: it writes a reversal into the thing being marked. On `home_building_1`
    the leg after the dining-table passage did exactly that in three runs of
    four, its first exploration step heading 1.7 m back east straight through
    the gap the robot had just come out of.

    The side test is what keeps this from banning the passage outright. A
    crossing that lands on the far side from where the robot entered is the
    passage being driven, not undone, and on a question whose next destination
    genuinely lies back the way it came the robot must still be able to go.
    """
    tip = origin + u * reach
    for a, b, entry in crossed:
        if crosses(origin, tip, a, b) and side_of(a, b, tip) == side_of(a, b, entry):
            return True
    return False


def explore_direction(cm: ConverterModel, origin: np.ndarray, want: float,
                      spent: list[tuple[np.ndarray, np.ndarray]] | None = None,
                      crossed: list[tuple[np.ndarray, np.ndarray,
                                          np.ndarray]] | None = None,
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

    `crossed` holds passages the robot has already driven through, and a
    bearing that would take it back out through one is discounted the same way
    and for a stronger reason: those are scored constraints, and re-crossing
    writes a reversal into the trajectory being marked. See `recrosses`.

    When nothing clears the gate the old rule decides, so a vehicle hemmed in
    on all sides still moves rather than standing still.

    Returns `(unit direction, reach, Δ in degrees)`.
    """
    spent = spent or []
    crossed = crossed or []
    best = (0.0, np.array([np.cos(want), np.sin(want)]), 0.0, 0.0)
    fallback = (0.0, np.array([np.cos(want), np.sin(want)]), 0.0, 0.0)
    for delta in range(-90, 91, 15):
        th = want + np.deg2rad(delta)
        u = np.array([np.cos(th), np.sin(th)])
        reach = cm.reach_along(origin, u)
        near = float(np.cos(np.deg2rad(delta)))
        if already_tried(origin, u, spent):
            near *= SPENT_PENALTY
        if recrosses(origin, u, crossed, max(reach, MIN_EXPLORE_M)):
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


def closing(gap: float | None, closest: float) -> bool:
    """Is this the nearest the leg has ever been to the target it is holding?

    Passing near a spot already stood in is not a cycle if the vehicle is
    nearer the thing it is driving at than it has ever been — that is a curve,
    which is what an approach round furniture looks like from above. Circling
    is the case where the return buys nothing, and this separates them without
    needing to know the shape of the route.

    `closest` is kept per binding and not per leg: it measures progress toward
    one point, so when the binding moves the record is about a different point
    and comparing across the change is a category error.
    """
    return gap is not None and gap < closest - PROGRESS_M


def nearer_reading(now: float | None, was: float | None) -> bool:
    """Was this reading taken from close enough to outrank the binding's?

    Both ranges are the distance from the vehicle to the object when the lift
    was made, so this compares two measurements and not two opinions. `None`
    means a lift the size check refused or a bearing below the scanner, which
    carries no range and so cannot claim to be the better one.
    """
    return (now is not None and was is not None and was > 0.0
            and float(now) <= NEARER_RATIO * float(was))


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
            bound = {"xy": seen, "conf": conf, "verified": measured,
                     "range_m": wp.range_m}
            if pending is not None:
                pending.clear()
        else:
            jump = float(np.linalg.norm(seen - bound["xy"]))
            switched = reply.get("same_object_as_previous") is False
            if jump <= JUMP_M:
                print(f"      binding refined {jump:.2f} m -> "
                      f"({seen[0]:+.2f}, {seen[1]:+.2f})")
                bound = {"xy": seen, "conf": conf,
                         "verified": measured or bound.get("verified", True),
                         "range_m": wp.range_m}
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
                bound = {"xy": seen, "conf": conf, "verified": True,
                         "range_m": wp.range_m}
                if pending is not None:
                    pending.clear()
            elif nearer_reading(wp.range_m, bound.get("range_m")):
                # The gate is a fixed metre and a lift's error is not: it is a
                # bearing error times a range. A binding made from 11 m and a
                # reading made from 3.6 m disagreeing by 3.5 m is what that
                # leverage looks like, not two different objects — and on
                # `livingroom_2` the reading refused for it was 0.12 m from the
                # true soccer ball while the binding it defended was 3.42 m
                # away. The leg then drove to the binding and reported arrival.
                #
                # This does not widen the gate, which would let any bad reading
                # in. It adds one way past it, and the qualification is a
                # measurement the model has no say in: how far the vehicle was
                # standing when each reading was taken.
                was = float(bound["range_m"])
                print(f"      re-bound {jump:.2f} m away — measured from "
                      f"{wp.range_m:.2f} m where the binding was measured from "
                      f"{was:.2f} m")
                rec["binding_nearer"] = {"was_m": was, "now_m": wp.range_m,
                                         "jump_m": jump}
                bound = {"xy": seen, "conf": conf, "verified": measured,
                         "range_m": wp.range_m}
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
                bound = {"xy": seen, "conf": conf, "verified": measured,
                         "range_m": wp.range_m}
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
                bound = {"xy": seen, "conf": conf, "verified": measured,
                         "range_m": wp.range_m}
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
           mission: dict | None = None, *,
           visited_kind: str = "prose",
           ask_here: bool = True) -> tuple[dict | None, str]:
    """The parsed reply and the text it came from.

    The raw text is returned because three runs died on "unparseable reply"
    while discarding the only evidence of why. It was truncation.
    """
    fn = ask_claude if backend == "claude" else ask_gemini
    text = fn(build_prompt(phrase, approach=True, version=version,
                           visited=visited, visited_kind=visited_kind,
                           ask_here=ask_here, mission=mission),
              faces, model, previous=prev)
    reply = parse(text)
    # Before anything reads a box: the reply's own `coord_space` is not always
    # true, and one wrong word here moves every waypoint. See
    # `settle_coord_space`.
    if reply is not None:
        reply = settle_coord_space(reply, backend, G.FACE_SIZE)
    return reply, text


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

    `spent`, `crossed` and `done` moved to this side after a leg boundary was
    found to erase the vehicle's momentum. `spent` used to be built fresh
    inside `run_goto`, so the direction the robot had just arrived from carried
    no penalty at all in the leg that followed — and that is the one direction
    guaranteed to have open floor, which is exactly what `explore_direction`'s
    reach gate rewards. On `home_building_1` the destination leg after the
    dining-table passage turned round and drove back through the gap on its
    first exploration step in three runs of four, one of them all the way back
    to the pose the passage had started from.
    """

    robot: Robot
    out: Path
    log: object
    backend: str = "claude"
    model: str = "claude-opus-5"
    prompt_version: str = DEFAULT_PROMPT_VER
    standoff: float = STANDOFF_M
    dry_run: bool = False
    # `{"xy": [x, y] | None, "text": str}` per place already searched. Both
    # halves are kept because the style is a run-time choice: `prose` renders
    # `text`, `bearing` and `xy` render the pose, and only one of them is in
    # the prompt on any given run. See `visited_for`.
    visited: list[dict] = field(default_factory=list)
    # "prose" (the original), "bearing", "xy", or "off". See `visited_for`.
    visited_style: str = VISITED_STYLE
    # Stop asking the model for `here` at all. Only sane when nothing renders
    # it: it is 19.7% of the reply's characters and the run's best diagnostic,
    # so the styles that do not feed it back still ask for it by default.
    drop_here: bool = False
    avoid: list[dict] = field(default_factory=list)
    # Departures already made, as (from, unit direction), across every leg.
    spent: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Passages already driven through, as (side a, side b, where it entered
    # from). Only two-sided gaps go in: a one-landmark passage has no line and
    # so no wrong way across it.
    crossed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = \
        field(default_factory=list)
    # Constraints already banked, in the model's language, for the prompt.
    done: list[str] = field(default_factory=list)
    # Objects a later step names, spotted while working on an earlier one:
    # `{step, what, xy}`, `xy` only when the box lifted. The robot arrives at
    # each leg having forgotten the room, and on `home_building_1` it wrote
    # "counter run with window and blue trash can to the right, stainless
    # fridge behind" on step 6 of leg 1 — while leg 3's target is "the trash
    # can closest to the refridgerator". That sentence was thrown away, and
    # leg 3 then bound a different bin on every run.
    sightings: list[dict] = field(default_factory=list)
    # True when the instruction forbids a corridor rather than a place, so the
    # keep-out anchors should be read as the two sides of a gate. Set by the
    # executor from the plan; a single-object run has no keep-out at all.
    keepout_is_gate: bool = False
    calls: int = 0
    step: int = 0
    deadline: float | None = None
    mission: dict | None = None

    # A per-leg deadline, set by the executor from the time left divided among
    # the clauses still to drive. `deadline` is the question's; this one stops
    # an early leg from spending the budget the later ones need.
    leg_deadline: float | None = None
    # The record `record()` is holding, unwritten, so that the verdict set on
    # it after the step decides still reaches the log. See `record`.
    _pending: dict | None = None

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
        """Queue `rec` for the log; it is serialised when the next one arrives.

        How a step ended — `arrived`, `stopped` — is set on `rec` *after* the
        step has decided, which on the write-immediately version was after the
        line had already been serialised. Every terminal verdict was therefore
        missing from `steps.jsonl`, and a failed leg logged exactly like a
        finished one: `runs/o_1_0814_02` holds seven complete-looking steps for
        a leg `plan.json` reports as `boxed in`, with nothing in the log to say
        so. Holding the object and serialising it late captures the mutations
        that say how the step ended, at the cost of one step of durability —
        the window is the microseconds between the verdict and the next call,
        against a step that spends tens of seconds in the model and the drive.

        Re-recording the same object is a no-op rather than a second line, so a
        path that records twice on its way out cannot double-log a step.
        """
        if rec is not self._pending:
            self.flush()
            self._pending = rec

    def flush(self) -> None:
        """Serialise the held record, if any."""
        if self._pending is not None:
            self.log.write(json.dumps(self._pending, default=str) + "\n")
            self.log.flush()
            self._pending = None

    def close(self) -> None:
        self.flush()
        self.log.close()

    def note_settings(self) -> None:
        """First line of the log: what this run was configured with.

        Asked whether `runs/o_1_0814_04` was faster than the runs before it,
        the only way to tell what it had been run with was whether `here`
        appeared in the replies. Wall clock is meaningless without the
        settings that produced it.
        """
        self.record({"step": "settings", "backend": self.backend,
                     "model": self.model, "prompt_version": self.prompt_version,
                     "visited_style": self.visited_style,
                     "drop_here": self.drop_here, "standoff": self.standoff,
                     "dry_run": self.dry_run,
                     "max_tokens": os.environ.get("XIAO_HEI_CLAUDE_MAX_TOKENS"),
                     "started": time.time()})

    def prompt_visited_kind(self) -> str:
        """Which `VISITED_BLOCKS` header frames what `visited_for` returned."""
        return "prose" if self.visited_style == "off" else self.visited_style

    def note_visit(self, xy, text: str) -> None:
        """File a place already searched, with the pose it was written from."""
        self.visited.append(
            {"xy": None if xy is None else list(np.asarray(xy, float)[:2]),
             "text": text})

    def visited_for(self, pose: dict) -> list[str] | None:
        """The "places already searched" list, written the way this run wants it.

        `prose` is what the loop has always sent: the model's own `here`
        clauses. It is the largest single field the model writes back (19.7% of
        the reply) and, once a few steps in, the largest thing in the prompt
        that prompt caching can never touch — it changes every call, so every
        call pays for it twice.

        Measured against a same-prompt control over 41 paired steps, the block
        does reach the model: removing it moves the explore heading further
        than re-rolling the identical prompt does, p = 0.016. What it does not
        do is the thing it asks for. Its advantage on "prefer somewhere it has
        not stood" is 0.06 m, and two samples of the same prompt differ by
        0.06 m.

        `bearing` is the hypothesis that the frame was the problem rather than
        the content. The robot has stood in these places and we know exactly
        where they are; the model cannot use `(2.31, -1.44)` because it has
        never seen that frame, but it can use "4.2 m behind you" because that
        names one of the four images in front of it. Headings therefore invert
        the map yaw the same way `explore_goal` does, so a heading read out of
        this block and handed back in `explore` means the same thing in both
        directions — getting that sign wrong would tell the model to avoid the
        one direction it should go.

        `xy` sends the raw frame, as the control for that hypothesis. `off`
        sends nothing.
        """
        if self.visited_style == "off" or not self.visited:
            return None
        # Tolerate a bare string. `note_visit` is meant to be the only writer,
        # but it was not: `execute_plan.run_pass` kept appending `here` directly
        # and the mismatch surfaced as `TypeError: string indices must be
        # integers` inside a grounding call, which killed a whole question two
        # legs in. A shape guard here is cheaper than that, whatever writes next.
        visits = [v if isinstance(v, dict) else {"xy": None, "text": v}
                  for v in self.visited]
        if self.visited_style == "prose":
            return [v["text"] for v in visits if v["text"]]

        o = np.asarray(pose["position"], float)[:2]
        yaw = yaw_of(pose)
        out: list[str] = []
        for v in visits:
            if v["xy"] is None:
                # A sentinel with no pose — the loop-return note. Keep the
                # words; there is no geometry to replace them with.
                out.append(v["text"])
                continue
            p = np.asarray(v["xy"], float)
            d = float(np.linalg.norm(p - o))
            if self.visited_style == "xy":
                out.append(f"({p[0]:+.2f}, {p[1]:+.2f})")
                continue
            if d < REVISIT_M:
                out.append("the robot is standing on this one now")
                continue
            # Inverse of `explore_goal`: it drives `yaw - heading`, so the
            # heading that points at `p` is `yaw - atan2(dy, dx)`.
            h = np.degrees(yaw - np.arctan2(p[1] - o[1], p[0] - o[0])) % 360.0
            out.append(f"{d:.1f} m away at heading {h:.0f}° ({FACE_OF(h)})")
        # Order-preserving unique, not just consecutive: the geometric
        # renderings collapse to the same string far more often than prose did,
        # and the collisions are not adjacent. Ten steps of `hm1_q2_v6` produce
        # "the robot is standing on this one now" three times, spread through
        # the list. Distinct places along one bearing survive — four entries at
        # 333-336° and 4.5, 8.0, 9.3, 9.5 m are a trail, not a repetition.
        seen: set[str] = set()
        return [x for x in out if not (x in seen or seen.add(x))]

    def mission_for(self, k: int) -> dict | None:
        # Only this step's sightings. A lead for step 5 shown on step 2 is
        # noise, and worse than noise: the prompt's one rule about later steps
        # is not to chase them.
        return None if not self.mission else {
            **self.mission, "k": k, "done": list(self.done),
            "sightings": [s["what"] for s in self.sightings
                          if s.get("step") == k]}

    def note_sightings(self, reply: dict, k: int, scan, pose) -> list[dict]:
        """File what the model saw for a later step, and lift it if it can.

        Kept apart from `visited`, which tells the model where *not* to go. A
        sighting is the opposite instruction and merging them loses the sign.

        Only later steps: a sighting of the current step is just the answer,
        and belongs in `box_2d` where the rest of the loop can see it.
        """
        out: list[dict] = []
        items = reply.get("sightings") or []
        if not isinstance(items, list):
            return out          # the model sometimes answers a field with prose
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                n = int(it.get("step"))
            except (TypeError, ValueError):
                continue
            what = (it.get("what") or "").strip()
            if n <= k or not what:
                continue
            if any(s["step"] == n and same_thing(s["what"], what)
                   for s in self.sightings):
                continue                      # already have this one
            xy = None
            if it.get("box_2d") is not None and it.get("image_index") is not None:
                try:
                    got = _lift_xy(
                        to_pixels(it["box_2d"], reply.get("coord_space"),
                                  G.FACE_SIZE),
                        int(it["image_index"]), scan_to_camera(scan, pose), pose)
                    if got is not None:
                        xy = np.asarray(got, float)[:2].tolist()
                except (KeyError, TypeError, ValueError, IndexError):
                    xy = None
            rec = {"step": n, "what": what, "xy": xy}
            self.sightings.append(rec)
            out.append(rec)
            where = "" if xy is None else f" -> ({xy[0]:+.2f}, {xy[1]:+.2f})"
            print(f"      noted for step {n}: {what[:60]!r}{where}")
        return out


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
    # Nearest the vehicle has been to the binding it currently holds. Reset
    # whenever the binding moves; see `closing`.
    closest = float("inf")
    # Departures already made, as (from, unit direction). The model cannot
    # remember which door it has been through; this can. It lives on `ctx` and
    # not here so that it survives the leg boundary — see the class docstring.
    spent = ctx.spent
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

        # A grounding call is the one place in the loop that reaches outside the
        # process, and until now it was also the one place that could take the
        # whole question down. On `runs/hb_1_0814_01` a `TypeError` raised
        # while *building* the prompt propagated out of `execute`, past the two
        # legs that had already run, and past the third that had not — the
        # process died with the log half written and nothing scored. The score
        # is per-constraint with partial credit, so no single call is worth the
        # rest of the question: one failed leg, and the executor drives on.
        try:
            reply, raw = ground(faces, phrase, ctx.backend, ctx.model,
                                prev_crop, ctx.prompt_version,
                                ctx.visited_for(pose), ctx.mission_for(k),
                                visited_kind=ctx.prompt_visited_kind(),
                                ask_here=not ctx.drop_here)
        except Exception as e:
            print(f"[{step}] grounding call failed ({type(e).__name__}: {e}); "
                  f"ending this leg")
            ctx.record({"step": step, "clause": k, "phrase": phrase,
                        "kind": GOTO, "pose": pose, "reply": None,
                        "error": f"{type(e).__name__}: {e}"})
            return Outcome(False, f"grounding call failed ({type(e).__name__})",
                           None, prev_crop)
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
        #
        # ...unless the return bought something. `chinese_room` lost both of its
        # legs here, each killed on the step where it stood nearer its target
        # than it ever had: leg 1 on step 3, for passing 0.47 m from the pose it
        # had started at, with the potted plant 2.51 m away, bound to within
        # 0.38 m of the truth, and standable floor 0.86 m from it. An approach
        # round furniture is a curve, and a curve crosses its own outbound
        # ground; what distinguishes it from a cycle is not the shape but
        # whether the vehicle is getting closer. See `closing`.
        gap = (float(np.linalg.norm(bound["xy"] - o[:2]))
               if bound is not None else None)
        if revisited(o[:2], stood) and not closing(gap, closest):
            d = gap
            # ...and "the ring around the target" is only a description of the
            # walk if the target is at the middle of it. On `livingroom_2` the
            # leg shuffled twice inside half a metre and returned this with the
            # binding 9.79 m away, which is not a ring and not a floor: it is a
            # leg that never got there. Reported as arrival it is a false
            # positive in the log and in the score, so the distance has to
            # qualify it. `CIRCLE_ARRIVE_M` is the platform's own floor around
            # furniture plus room for a lift error, which is what a genuine
            # ring is made of.
            # The distance qualifies the shape but not the binding, and it
            # cannot tell "walked round the target" from "walked
            # round something the scanner mistook for it": on
            # `runs/o_2_0814_02` leg 3 the binding sat 1.81 m away, well inside
            # `CIRCLE_ARRIVE_M`, and was a glass partition with the door 6.5 m
            # beyond it. The model said `far` on that step and on four of the
            # five before it. Falling through here costs nothing — the next
            # branch counts the return, and `MAX_LOOPS` ends the leg honestly
            # rather than reporting an arrival that did not happen.
            if d is not None and d <= CIRCLE_ARRIVE_M and not says_far(reply):
                print(f"      back where it already stood — the ring around the "
                      f"target has been walked; {d:.2f} m is the floor here")
                rec["arrived"] = f"circled back ({d:.2f} m)"
                ctx.record(rec)
                return Outcome(True, f"arrived, circled back ({d:.2f} m)",
                               bound["xy"], prev_crop)
            if d is not None:
                print(f"      back where it already stood, but the binding is "
                      f"{d:.2f} m away — that is stuck, not arrived")
                rec["stopped"] = f"circling {d:.2f} m short of the binding"
                ctx.record(rec)
                return Outcome(False, f"circling {d:.2f} m short of the binding",
                               None, prev_crop)
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
            # No pose: this note is about the route taken *from* here, not
            # about the spot, so the geometric renderings keep the words.
            ctx.note_visit(None,
                           "(the robot came back to this spot after leaving it "
                           "— whatever route it took from here revealed nothing "
                           "new, so send it a different way)")
            if loops >= MAX_LOOPS:
                print(f"      {loops} returns to the same ground with nothing "
                      f"bound — this leg is going in circles")
                rec["stopped"] = "circling with nothing bound"
                ctx.record(rec)
                return Outcome(False, "circling with nothing bound", None,
                               prev_crop)
        stood.append(o[:2].copy())
        if gap is not None:
            closest = min(closest, gap)

        # Before the visibility branch: a keep-out anchor is most likely to be
        # reported on exactly the calls where the *target* is not visible,
        # because that is when the robot is looking around at the furniture.
        noted = ctx.note_sightings(reply, k, scan, pose)
        if noted:
            rec["sightings"] = noted
        # The pose goes in whether or not the model wrote a clause: under the
        # geometric styles the clause is not asked for, and the place still has
        # to be logged as searched.
        here_txt = (reply.get("here") or "").strip()
        if here_txt or ctx.visited_style in ("bearing", "xy"):
            ctx.note_visit(o[:2], here_txt)
            if here_txt:
                rec["here"] = here_txt
                print(f'      here: "{here_txt[:96]}"')

        ctx.avoid = bind_constraints(reply, scan, pose, ctx.avoid)
        # A corridor the instruction forbids is a gate, not two discs. Discs
        # big enough to close it close the room as well — see `ConverterModel`.
        gates = (gates_from(ctx.avoid)
                 if (ctx.keepout_is_gate and USE_KEEPOUT) else [])
        keepout = ([] if gates or not USE_KEEPOUT else
                   [(a["xy"], KEEPOUT_M) for a in ctx.avoid])
        # One switch for the whole behaviour: no gate, no discs, no step cap,
        # no detour, and so no `diverted` either. See `USE_KEEPOUT`.
        constrained = bool(ctx.avoid) and USE_KEEPOUT

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
        # ...but only while a comparison is actually outstanding. `relational`
        # is our reading of the sentence; whether the model is *mid-comparison*
        # is the model's, and it says so two ways: by naming a relation, or by
        # offering rival candidates. Neither is true of a single unqualified
        # nomination, and treating one as a failure deadlocked an entire leg.
        #
        # On `livingroom_2` q4 the phrase was "the crystal ball decoration on
        # the shelf near the TV". "near" makes `has_relation` true; the model
        # answered `relation: null, candidates: []` on ten consecutive calls —
        # it had resolved the phrase by eye and said so in its evidence ("the
        # same shelf that stands beside the TV unit") — so `chosen` was
        # always None, `adrift` always true, and the leg explored away from a box
        # it was handed every time, confidence climbing 0.40 to 0.82. The
        # fallback below that would have used the model's own pick is gated on
        # `bound is not None`, and `adrift` is what stops a binding ever being
        # made: the two conditions cannot both be satisfied first.
        comparing = (bool(reply.get("relation"))
                     or len(reply.get("candidates") or []) >= 2)
        adrift = (reply.get("visible") and relational and comparing
                  and chosen is None and bound is None)
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
            if way is None:
                # Nothing boxed here, but an earlier leg may have seen this
                # target and said where. A sighting is a *direction*, never a
                # binding: it was lifted from somewhere else, at whatever range
                # the room allowed, and today's measurements put a long lift
                # metres out. It goes in the slot `way` fills — drive that way
                # and look again — and the range cap applies for the same
                # reason it applies there.
                seen = next((s for s in ctx.sightings
                             if s.get("step") == k and s.get("xy")), None)
                if seen is not None:
                    v = np.asarray(seen["xy"], float) - o[:2]
                    d = float(np.linalg.norm(v))
                    if d > MIN_VIEW_MOVE_M:
                        way = o[:2] + v / d * min(d, WAY_MAX_M)
                        print(f"      not in sight, but an earlier leg saw it: "
                              f"{seen['what'][:52]!r} — heading that way")
                        rec["from_sighting"] = seen
            if way is not None and float(np.linalg.norm(way - o[:2])) \
                    < MIN_VIEW_MOVE_M:
                print(f"      way out boxed but already at it — falling back "
                      f"to the heading")
                way = None
            try:
                cm = ConverterModel(terrain, keepout=keepout, gates=gates)
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
                    u, reach, delta = explore_direction(cm, o[:2], want, spent,
                                                        ctx.crossed)
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
                why = []
                if u is not None and already_tried(o[:2], u, spent):
                    why.append("already taken from here")
                if u is not None and recrosses(o[:2], u, ctx.crossed,
                                               max(reach or 0.0, MIN_EXPLORE_M)):
                    why.append("back through a passage already driven")
                print(f"      heading {h}° reaches {asked:.2f} m; best drivable "
                      f"is {delta:+.0f}° off it at {reach:.2f} m"
                      f"{' (' + '; '.join(why) + ')' if why else ''}")
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
        box, feat_why = aim_box(reply, G.FACE_SIZE, scan, pose)
        if feat_why:
            rec["feature_box"] = feat_why
            print(f"      {feat_why}")
        # A comparative relation is decided by measuring the candidates, not by
        # whichever one the model nominated; `chosen` was resolved above.
        if chosen is not None:
            box, i, rel_why = chosen
            rec["relation"] = rel_why
            print(f"      relation resolved: {rel_why}")
            # A winner chosen over some of the candidates is not the winner.
            # On `japanese_room` step 2 the one candidate the lift refused was
            # the answer — the model had placed it within 0.2° of the truth —
            # and the comparison over the other three named a ceiling lantern
            # 3.98 m away with a rationale that reads as authoritative.
            # Demoted, not discarded: the survivors are still the best measured
            # evidence, so the leg drives at one while it keeps looking, and
            # only the licence to overrule a later binding is withdrawn.
            if not chosen.complete:
                rec["relation_partial"] = [
                    {"az": a, "el": e, "what": n} for a, e, n in chosen.missed]
                print(f"      ...over {len(chosen.missed)} fewer candidate(s) "
                      f"than the model reported — treating the winner as "
                      f"unverified")
                # And the bearing is worth keeping: a candidate the scanner
                # could not reach is usually behind the robot, which is a
                # direction to turn rather than a thing to forget.
                for a, e, n in chosen.missed:
                    # A bearing, not a place — no pose to render it from.
                    ctx.note_visit(None,
                                   f"(a possible {n[:60]!r} was seen at bearing "
                                   f"{a:+.0f}° but could not be measured from "
                                   f"here — turning to face it would settle the "
                                   f"comparison)")
        elif reply.get("relation"):
            print(f"      relation {reply['relation']!r} not measurable "
                  f"({len(reply.get('candidates') or [])} candidates, "
                  f"{len(reply.get('anchors') or [])} anchors) — using the "
                  f"model's own pick")
        # Same distinction, and it has to be the same or the fix is half a fix:
        # letting the leg approach a nomination it can never verify only trades
        # exploring-away for driving-at-it-forever. A phrase our parser calls
        # relational, answered with no relation and no rivals, is not an
        # unverifiable comparison — it is an ordinary nomination, and `JUMP_M`,
        # `same_object_as_previous` and `corroborated` defend it exactly as
        # they defend a phrase with no relation in it at all.
        verified = (not relational) or (not comparing) or chosen is not None
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

        # The phrase has to hold of the answer, not only decide between
        # answers. This is where a nomination is checked against the anchor the
        # model itself reported, and it is a demotion rather than a refusal:
        # unverified drives at the thing and keeps looking, where a refusal
        # would throw away the only reading there is.
        if wp.committed and wp.range_m is not None:
            d = wp.xy - o[:2]
            n = float(np.linalg.norm(d))
            where = o[:2] + (d / n if n > 1e-6 else d) * wp.range_m
            ok, why_not = relation_holds(reply, where, scan, pose)
            if not ok:
                print(f"      the phrase does not hold of this: {why_not} — "
                      f"a guess at the noun, not the phrase")
                rec["relation_failed"] = {"seen": where.tolist(), "why": why_not}
                verified = False

        was_bound = None if bound is None else bound["xy"].copy()
        committed, bound = bind_target(wp, o[:2], reply, bound, rec,
                                       verified=verified,
                                       # `measured` is the licence to overrule
                                       # an earlier binding "at any distance",
                                       # and a comparison missing a candidate
                                       # has not earned it: the candidate it
                                       # could not lift is exactly the one that
                                       # might have won. Still bound, so the leg
                                       # drives at the best evidence it has —
                                       # demoted, not discarded.
                                       measured=(chosen is not None
                                                 and chosen.complete),
                                       pending=pending)
        # A new binding is a new destination, and the record of how near the
        # vehicle got to the old one says nothing about it. Leg 2 of
        # `chinese_room` rebound twice while approaching and would have been
        # failed for standing 2.83 m from the painting having once been 1.60 m
        # from a reading it had already discarded.
        #
        # The record restarts at the distance the binding was made from, and not
        # at infinity: "how near was I when I decided this was the thing" is a
        # measurement, where infinity would let any next step count as progress.
        # On `lr_2_0811_06` the binding jumped 5.5 m out on the step before the
        # revisit, and infinity would have excused the 9.79 m that run reported
        # as an arrival.
        moved_binding = (bound is not None and (was_bound is None or float(
            np.linalg.norm(bound["xy"] - was_bound)) > 1e-6))
        if bound is None:
            closest = float("inf")
        elif moved_binding:
            closest = float(np.linalg.norm(bound["xy"] - o[:2]))

        # Already inside the standoff: driving further would push into the
        # object, and the stack would only snap the waypoint back out again.
        if committed and bound is not None:
            here = float(np.linalg.norm(bound["xy"] - o[:2]))
            # Standing inside the standoff of the *binding* is not standing at
            # the target when the binding is a surface in front of it. Both
            # `jr_0812_01` and `jr_0812_04` arrived here on a `far`, with the
            # model putting the object 6.0 m and 7.5 m out.
            if here <= ctx.standoff and not says_far(reply):
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
        # With a keep-out in force, aim at what the model says to cross next
        # rather than at the destination. It cannot see the path the stack will
        # take, but it can see the floor, and "which side to pass" needs no
        # coordinate — which is the half the geometry keeps getting wrong for a
        # different reason: on `livingroom_2` the tea table lifted 2.0 m out and
        # the forbidden corridor was drawn across the wrong part of the room.
        #
        # The destination is not forgotten; it is the next call's problem. A
        # detour is one step of the way round, and the loop asks again from
        # there.
        detour = lift_detour(reply, scan, pose) if constrained else None
        if detour is not None and float(np.linalg.norm(detour - o[:2])) \
                < MIN_VIEW_MOVE_M:
            detour = None                     # already there; nothing to drive
        # `steer` is where this step drives; `aim` stays the target, because
        # the arrival tests below measure against the thing we were asked for
        # and a detour is deliberately not it.
        steer = aim
        # True when this step is not going at the target — a detour, or a
        # capped fraction of the way. Arrival is still measured against `aim`,
        # and a step that is not aimed at it may not settle where it stands:
        # `livingroom_2` spent two calls on 0.10 m moves because a committed
        # approach is allowed to, and those two were going round something.
        diverted = False
        if detour is not None:
            nm = (reply.get("detour") or {}).get("name") or "the way round"
            print(f"      detour: {nm!r} at ({detour[0]:+.2f}, "
                  f"{detour[1]:+.2f}), {float(np.linalg.norm(detour - o[:2])):.2f} m")
            beyond = past(o[:2], detour)
            rec["detour"] = {"xy": detour.tolist(), "name": nm,
                             "aimed_at": beyond.tolist()}
            steer, diverted = beyond, True
        if constrained:
            # Short hops, so that the straight line the constraint is checked on
            # is a fair model of the arc `local_planner` will actually drive.
            # The cap belongs to the constraint and not to which branch chose
            # the aim: leaving the detour uncapped is what let a 2.42 m move
            # end 1.49 m east of where it was planned and take the vehicle
            # through the middle of the forbidden gap. Over the 121 recorded
            # drives, capping at this length takes the worst sideways stray
            # from 2.89 m to 0.95 m.
            v = steer - o[:2]
            d = float(np.linalg.norm(v))
            if d > KEEPOUT_STEP_M:
                steer, diverted = o[:2] + v / d * KEEPOUT_STEP_M, True
                print(f"      keep-out in force — stepping {KEEPOUT_STEP_M} m "
                      f"of the {d:.2f} m, not all of it")
        try:
            cm = ConverterModel(terrain, keepout=keepout, gates=gates)
            # Aim at the target itself, not at a standoff from it: the standoff
            # is what the converter's inflation is *for*, and asking for a point
            # inside it gets the waypoint discarded rather than clamped. Once
            # the target is bound, the binding is the better estimate of where
            # it is than any single reading.
            # A step exists to buy a better view, so it has to actually move
            # the vehicle; an approach may legitimately settle where it stands.
            best = cm.best_waypoint_toward(
                steer, o[:2],
                # A committed approach may settle where it stands — that is
                # arrival. A step taken *round* something may not: its purpose
                # is to get somewhere else, and `livingroom_2` spent two calls
                # on 0.10 m moves because 0.10 m was allowed. Each call costs
                # the same, so a step that buys no parallax buys nothing.
                min_move=(MIN_VIEW_MOVE_M if (diverted or not committed)
                          else 0.0))
            if best is not None:
                goal, lands, reach = best
                will_move = float(np.linalg.norm(lands - o[:2]))
                rec["converter"] = {
                    "aim": steer.tolist(), "goal": goal.tolist(),
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
            elif keepout or gates:
                # The old fallback here was `goal = wp.xy` — publish the raw
                # waypoint, keep-out and all. That is how `livingroom_2` q5 came
                # to drive through the middle of the forbidden gap: five drifted
                # discs left no answer, and the constraint was then dropped in
                # silence rather than the drive being reconsidered. Refusing to
                # answer is information; it means every route from here is
                # forbidden, and the honest move is a step that is not.
                goal = nearest_allowed_step(cm, o[:2], steer)
                rec["constraint_bind"] = {
                    "aim": steer.tolist(),
                    "why": "no legal waypoint toward the aim clears the keep-out",
                    "fallback": None if goal is None else goal.tolist()}
                if goal is None:
                    print(f"      every legal waypoint from here is forbidden, "
                          f"and so is standing still — publishing the raw "
                          f"waypoint and recording the violation")
                    rec["constraint_violated"] = True
                    goal = wp.xy
                else:
                    print(f"      no legal waypoint toward the aim clears the "
                          f"keep-out; stepping to ({goal[0]:+.2f}, "
                          f"{goal[1]:+.2f}) instead")
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
        # A step that is deliberately driving somewhere other than the target
        # cannot report having got as close to it as the platform allows: the
        # gain it made is against wherever it was steered, and the target was
        # never aimed at. Same flag the min_move rule uses, so the two cannot
        # disagree about what this step was for.
        # "The converter has nothing closer" is a fact about the binding, and
        # it is only a fact about the target while the two are the same thing.
        # A scanner that stopped at a glass partition, a chair back or a desk
        # edge produces a binding the vehicle really cannot improve on and
        # really is not at. The model is looking at the picture and saying the
        # thing still reads as part of the scene; that is worth more here than
        # the converter's certainty, because the converter is certain about the
        # wrong point. Driving on costs steps, which `leg_deadline` and
        # `max_steps` already bound, and a leg that runs out having never
        # arrived is a truthful `ok: false` rather than a false `ok: true`.
        far = says_far(reply)
        may_stop = (committed or bound is not None) and not diverted
        if gain is not None and gain < PROGRESS_M and (here > NEAR_M or far
                                                      or not may_stop):
            why = ("the model still reads it as far" if far and here <= NEAR_M
                   else f"{here:.2f} m"
                        f"{'' if may_stop else ', nothing bound yet'}")
            print(f"      not close enough to call this the floor ({why}) "
                  f"— driving to look")
            if far:
                rec["far_veto"] = "target_state=far, not treating this as arrival"
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
            alt = cm.best_waypoint_toward(steer, o[:2], min_move=MIN_VIEW_MOVE_M)
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
                if far:
                    # Same reasoning as the predicted stop above: the stack
                    # refusing to move is the platform's floor around the
                    # binding, and the binding is what is in doubt.
                    print(f"      stack will not move (moved {moved:.2f} m) but "
                          f"the model still reads it as far — not arrived")
                    rec["far_veto"] = ("target_state=far, stack clamp not "
                                       "treated as arrival")
                    return Outcome(False, "stack clamped short of a far target",
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
                            prev_crop, ctx.prompt_version,
                            ctx.visited_for(pose), ctx.mission_for(k),
                            visited_kind=ctx.prompt_visited_kind(),
                            ask_here=not ctx.drop_here)
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
    ap.add_argument("--visited", choices=VISITED_STYLES, default=VISITED_STYLE,
                    help="how 'places already searched' is written: prose (the "
                         "original, and the rollback), bearing, xy, or off. "
                         "Env XIAO_HEI_VISITED sets the default.")
    ap.add_argument("--drop-here", action="store_true",
                    help="stop asking the model for 'here' — 19.7%% of the "
                         "reply, and the run's best diagnostic. Only sane with "
                         "--visited bearing/xy/off.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="ground and compute waypoints, publish nothing")
    args = ap.parse_args()

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else DEFAULT_GEMINI_MODEL)
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
              visited_style=args.visited, drop_here=args.drop_here,
              dry_run=args.dry_run)
    print(f"visited style: {ctx.visited_style}"
          f"{'  (here not requested)' if ctx.drop_here else ''}")
    ctx.note_settings()
    res = run_goto(ctx, args.phrase, max_steps=args.max_steps, confirm=True)
    ctx.close()
    print(f"\n{'ARRIVED' if res.arrived else 'did not arrive'} ({res.why}) in "
          f"{ctx.calls} calls (${ctx.calls * COST_PER_CALL:.2f})   "
          f"log: {out}/steps.jsonl")
    return 0 if res.arrived else 1



if __name__ == "__main__":
    raise SystemExit(main())
