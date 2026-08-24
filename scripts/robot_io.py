#!/usr/bin/env python3
"""ROS side of the approach loop: grab a frame, or drive to a point.

Runs *inside* `iros2026_system`, which is the only place with ROS on its path.
`scripts/approach_loop.py` copies this in and invokes it; nothing else here
knows about the VLM, so this file stays debuggable on its own:

    docker exec iros2026_system bash -lc \\
      'source /opt/ros/jazzy/setup.bash && python3 /tmp/robot_io.py capture'
    docker exec iros2026_system bash -lc \\
      'source /opt/ros/jazzy/setup.bash && python3 /tmp/robot_io.py drive -- -0.01 -1.55'
    docker exec iros2026_system bash -lc \\
      'source /opt/ros/jazzy/setup.bash && python3 /tmp/robot_io.py stop'

Both subcommands print one JSON object on stdout and write nothing else there,
so the caller can parse without stripping ROS log noise.

No numpy-free tricks are needed for the image because `/camera/image/compressed`
is already JPEG on the wire — the scan is the only thing that needs unpacking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import Float32

IMG = "/tmp/loop_img.jpg"
SCAN = "/tmp/loop_scan.npy"
POSE = "/tmp/loop_pose.json"
# `waypointConverter` splits this one cloud into the traversable area it snaps
# to and the obstacles it keeps 0.75 m away from, so it is the only input that
# predicts what the converter will do with a waypoint before we publish it.
TERRAIN = "/tmp/loop_terrain.npy"

# The vehicle runs at 0.875 m/s (waypoint_converter.launch `speed`). If it has
# not covered this much in this long, it is not going to — the waypoint was
# snapped somewhere it cannot reach, or something is in the way.
STALL_M = 0.05
STALL_S = 4.0
# Turning in place is progress and moves the vehicle nowhere. On chinese_room a
# waypoint 138 deg behind the vehicle was declared "settled, moved 0.0008 m"
# while the robot was mid-turn — it had covered 56 deg of it by the time the
# next frame was captured. A position-only stall test reads every large
# reorientation as the platform refusing to move.
STALL_DEG = 3.0
# `waypointXYRadius` is 0.3, so the stack stops that far from its own target;
# anything at or under this counts as having got where we asked.
ARRIVE_TOL_M = 0.35
# How far the vehicle moves between recorded track points.
TRACK_STEP_M = 0.10


class Capture(Node):
    """One frame of everything, with the pose taken *after* the image.

    Each callback used to keep the first message it saw, independently. Pose
    publishes at 100-200 Hz and `/camera/image` at ~9.3 Hz, so the pose landed
    within milliseconds of subscribing and the image up to a full 107 ms camera
    period later: the lift then mapped pixels through a pose from before the
    frame was taken. Every bearing in the loop rides on that pairing.

    PR #28 fixed the same defect on the perception side, where a 2 Hz responder
    running while the vehicle drove measured up to 17° of azimuth error. This
    loop is much less exposed — it captures after a drive returns, not during
    one — and the residual against ground truth over 54 calls whose box is on
    the right object is -0.11° ± 0.33, statistically indistinguishable from
    zero. So this is hardening, not a repair.

    A ring buffer keyed on `header.stamp` is what the perception side needed,
    with several frames in flight. Here one frame is wanted and only the
    ordering is wrong, so the fix is to discard any pose that arrived before
    the image and keep the next one: the wait is then bounded by the *pose*
    period, 5-10 ms, rather than the camera's. The scan is left alone —
    `/registered_scan` arrives already in the map frame, so a stale one is
    stale geometry rather than misregistered geometry, and `noDecayDis` gives
    it a 1.75 m memory anyway.
    """

    def __init__(self) -> None:
        super().__init__("loop_capture")
        self.img = self.scan = self.pose = self.terrain = None
        # Best-effort matches the sim publishers; a reliable subscription would
        # silently never match and just hang.
        self.create_subscription(CompressedImage, "/camera/image/compressed",
                                 self._on_img, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 self._on_scan, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/terrain_map",
                                 self._on_terrain, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation",
                                 self._on_pose, qos_profile_sensor_data)

    def _on_img(self, m) -> None:
        if self.img is None:
            with open(IMG, "wb") as f:
                f.write(bytes(m.data))
            self.img = True
            # Anything already held was sampled before this frame. Drop it and
            # take the next one, which lands within a pose period. See the
            # class docstring.
            self.pose = None

    def _on_scan(self, m) -> None:
        if self.scan is not None:
            return
        # PointXYZI, 16 bytes per point. Assert rather than assume: a layout
        # change would otherwise reshape into plausible-looking garbage.
        assert m.point_step % 4 == 0, m.point_step
        a = np.frombuffer(bytes(m.data), dtype=np.float32)
        np.save(SCAN, a.reshape(-1, m.point_step // 4))
        self.scan = True

    def _on_terrain(self, m) -> None:
        if self.terrain is not None:
            return
        # Do not infer columns from point_step: PCL pads PointXYZI out to 32
        # bytes, so intensity sits at offset 16 with three dead floats after
        # it. Reading the declared offsets is the only way to land on the
        # height field rather than on padding.
        assert m.point_step % 4 == 0, m.point_step
        a = np.frombuffer(bytes(m.data), dtype=np.float32
                          ).reshape(-1, m.point_step // 4)
        col = {f.name: f.offset // 4 for f in m.fields}
        # intensity is height above the local ground — what obstacleHeightThre
        # is compared against in waypointConverter.
        np.save(TERRAIN, a[:, [col["x"], col["y"], col["z"], col["intensity"]]])
        self.terrain = True

    def _on_pose(self, m) -> None:
        if self.pose is not None:
            return
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = {"position": [p.x, p.y, p.z],
                     "orientation": [q.x, q.y, q.z, q.w]}
        with open(POSE, "w") as f:
            json.dump(self.pose, f)

    def done(self) -> bool:
        return bool(self.img and self.scan and self.pose and self.terrain)


class Driver(Node):
    """Publish one waypoint and watch the vehicle until it settles.

    Arrival is decided from `/state_estimation` alone — our own distance to the
    waypoint we asked for, plus whether the vehicle has stopped making progress.
    That restriction is not stylistic. The challenge README lists the five
    topics an AI module may use at test time, and `/way_point_reached` is not
    among them:

        "While more topics may be available from the system, these are the only
         ones allowed to be used during test time."

    `waypoint_converter` snaps our waypoint into the traversable area, so the
    vehicle usually never reaches our literal XY and a pure distance test would
    hang. The stall test is what covers that, and it needs nothing but pose.

    The topic is still subscribed, and reported as `stack_said`, purely so the
    allowed predicate can be checked against the stack's own opinion during
    development. Nothing branches on it.
    """

    def __init__(self, x: float, y: float) -> None:
        super().__init__("loop_driver")
        self.goal = (x, y)
        self.reached: float | None = None
        self.pose: list[float] | None = None
        self.start: list[float] | None = None
        # The path actually driven, not just its endpoints. `local_planner`
        # chooses an arc from its own path library, so the straight line
        # between two waypoints is not where the vehicle went — and README §175
        # scores "the actual trajectory followed by the robot". A passage
        # constraint can only be checked against this.
        self.track: list[list[float]] = []
        self._last_move = time.time()
        self._last_xy: tuple[float, float] | None = None
        self._last_yaw: float | None = None
        self.pub = self.create_publisher(Pose2D, "/way_point_with_heading", 5)
        self.create_subscription(Float32, "/way_point_reached",
                                 self._on_reached, 5)
        self.create_subscription(Odometry, "/state_estimation",
                                 self._on_pose, qos_profile_sensor_data)

    def _on_reached(self, m) -> None:
        if self.reached is None:
            self.reached = float(m.data)

    def _on_pose(self, m) -> None:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.pose = [p.x, p.y, p.z]
        if self.start is None:
            self.start = list(self.pose)
        # Subsampled by distance: /state_estimation runs at 100-200 Hz, and a
        # crossing test needs shape, not sample rate. TRACK_STEP_M is well
        # under the narrowest gap a passage question names.
        if not self.track or np.hypot(p.x - self.track[-1][0],
                                      p.y - self.track[-1][1]) > TRACK_STEP_M:
            self.track.append([round(p.x, 3), round(p.y, 3)])
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        moved = (self._last_xy is None
                 or np.hypot(p.x - self._last_xy[0],
                             p.y - self._last_xy[1]) > STALL_M)
        # A turn counts. The platform points itself at the waypoint before it
        # translates, so a goal behind the vehicle produces seconds of pure
        # rotation that a position-only test cannot distinguish from refusal.
        turned = (self._last_yaw is None
                  or abs(np.rad2deg(np.arctan2(np.sin(yaw - self._last_yaw),
                                               np.cos(yaw - self._last_yaw))))
                  > STALL_DEG)
        if moved or turned:
            self._last_xy = (p.x, p.y)
            self._last_yaw = float(yaw)
            self._last_move = time.time()

    def gap(self) -> float | None:
        """Our own distance to the waypoint we asked for, from pose alone."""
        if self.pose is None:
            return None
        return float(np.hypot(self.pose[0] - self.goal[0],
                              self.pose[1] - self.goal[1]))

    def stalled(self) -> bool:
        return (self._last_xy is not None
                and time.time() - self._last_move > STALL_S)

    def send(self) -> bool:
        """Publish once, after a subscriber exists to receive it."""
        for _ in range(100):                      # ~5 s of discovery
            if self.pub.get_subscription_count() > 0:
                m = Pose2D()
                m.x, m.y, m.theta = float(self.goal[0]), float(self.goal[1]), 0.0
                self.pub.publish(m)
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False


def cmd_capture(timeout: float) -> dict:
    n = Capture()
    deadline = time.time() + timeout
    while time.time() < deadline and not n.done():
        rclpy.spin_once(n, timeout_sec=0.05)
    out = {"ok": n.done(), "image": IMG, "scan": SCAN, "terrain": TERRAIN,
           "pose": n.pose}
    if not n.done():
        out["missing"] = [k for k, v in (("image", n.img), ("scan", n.scan),
                                         ("terrain", n.terrain),
                                         ("pose", n.pose)) if not v]
    n.destroy_node()
    return out


def cmd_stop(timeout: float = 8.0) -> dict:
    """Park the vehicle where it stands.

    `cmd_drive` publishes its waypoint **once** and returns. Nothing retracts
    it, so when the process exits the local planner still holds that goal and
    keeps driving at it -- and when the goal is unreachable, which is exactly
    the case a timeout reports, the vehicle oscillates in place until the stack
    is restarted. Seen on `studio`: the loop gave up 2.58 m short, printed its
    answer, exited, and the robot went on twitching at a waypoint nobody was
    waiting for any more.

    Publishing the current pose as the waypoint is the retraction: it is a goal
    the vehicle is already at, so the planner settles instead of hunting. It
    matters beyond tidiness -- instruction following is scored on the driven
    trajectory, and a vehicle still moving after the run has ended is still
    writing to the thing being marked.
    """
    n = Driver(0.0, 0.0)
    for _ in range(120):                          # wait for /state_estimation
        rclpy.spin_once(n, timeout_sec=0.05)
        if n.pose is not None:
            break
    if n.pose is None:
        n.destroy_node()
        return {"ok": False, "why": "no /state_estimation — nothing to park at"}
    n.goal = np.array(n.pose[:2], float)
    sent = n.send()
    deadline = time.time() + timeout
    while time.time() < deadline and not n.stalled():
        rclpy.spin_once(n, timeout_sec=0.05)
    out = {"ok": bool(sent), "why": "parked" if sent else "nobody subscribed",
           "pose": n.pose, "goal": list(n.goal)}
    n.destroy_node()
    return out


def cmd_drive(x: float, y: float, timeout: float) -> dict:
    n = Driver(x, y)
    # Let /state_estimation arrive first, so `start` is the pose we left from
    # rather than whatever shows up after the vehicle is already moving.
    for _ in range(60):
        rclpy.spin_once(n, timeout_sec=0.05)
        if n.pose is not None:
            break
    if not n.send():
        n.destroy_node()
        return {"ok": False, "why": "no subscriber on /way_point_with_heading — "
                                    "is the autonomy stack up?"}

    deadline = time.time() + timeout
    why = "timeout"
    while time.time() < deadline:
        rclpy.spin_once(n, timeout_sec=0.05)
        if n.gap() is not None and n.gap() <= ARRIVE_TOL_M:
            why = "arrived"
            break
        if n.stalled():
            # Either the stack put us as close as its obstacle clearance
            # allows, or the waypoint is unreachable. `gap` tells them apart
            # downstream; both mean stop driving toward this point.
            why = "settled"
            break

    moved = (float(np.linalg.norm(np.array(n.pose[:2]) - np.array(n.start[:2])))
             if n.pose and n.start else None)
    out = {"ok": why in ("arrived", "settled"), "why": why, "goal": list(n.goal),
           "pose": n.pose, "moved_m": moved, "track": n.track,
           "dist_to_requested_m": n.gap(),
           # dev-only cross-check; never branched on. See Driver.__doc__.
           "stack_said": n.reached}
    n.destroy_node()
    return out


def cmd_preflight() -> dict:
    """Who is publishing what, before the loop spends money finding out.

    `/joy` is the one that bites: without a heartbeat the local planner
    silently discards every waypoint, including clicks in RViz, and the only
    symptom is a robot that never moves. A competing publisher on
    `/way_point_with_heading` is the other — the ai_module's own VLM will fight
    the loop for control if it was left running.
    """
    n = rclpy.create_node("loop_preflight")
    for _ in range(40):                                # let discovery settle
        rclpy.spin_once(n, timeout_sec=0.05)
    want = {"/state_estimation": "pub", "/registered_scan": "pub",
            "/camera/image/compressed": "pub", "/joy": "pub",
            "/way_point_with_heading": "sub", "/way_point_reached": "pub"}
    counts = {}
    for topic, kind in want.items():
        counts[topic] = (n.count_publishers(topic) if kind == "pub"
                         else n.count_subscribers(topic))
    # Name the waypoint publishers rather than counting them. RViz always holds
    # one — that is the manual Waypoint tool, which emits only when a human
    # clicks and so is not a rival. An autonomous node there is.
    rivals = [e.node_name for e in
              n.get_publishers_info_by_topic("/way_point_with_heading")
              if e.node_name != "rviz"]
    n.destroy_node()

    missing = [t for t, c in counts.items() if c == 0]
    why = None
    if "/joy" in missing:
        why = ("nothing is publishing /joy — the local planner will ignore "
               "every waypoint; start the heartbeat first")
    elif missing:
        why = f"no traffic on {missing}"
    elif rivals:
        why = (f"{rivals} also publish(es) /way_point_with_heading and will "
               f"fight the loop — stop xiao_hei_ai_module")
    return {"ok": why is None, "why": why, "counts": counts,
            "rival_waypoint_publishers": rivals}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--timeout", type=float, default=15.0)
    d = sub.add_parser("drive")
    d.add_argument("x", type=float)
    d.add_argument("y", type=float)
    d.add_argument("--timeout", type=float, default=30.0)
    sub.add_parser("preflight")
    st = sub.add_parser("stop", help="park the vehicle where it stands")
    st.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    rclpy.init()
    try:
        if args.cmd == "capture":
            out = cmd_capture(args.timeout)
        elif args.cmd == "drive":
            out = cmd_drive(args.x, args.y, args.timeout)
        elif args.cmd == "stop":
            out = cmd_stop(args.timeout)
        else:
            out = cmd_preflight()
    finally:
        rclpy.shutdown()
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
