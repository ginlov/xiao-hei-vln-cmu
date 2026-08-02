#!/usr/bin/env python3
"""Drive-and-capture harness (Part 1 of the perception benchmark).

The Unity sim has NO teleport: ``vehicleSimulator`` owns the vehicle pose and
only integrates ``/cmd_vel`` (``/unity_sim/set_model_state`` is its *output* to
Unity, not an input). So we move the robot the only way the stack allows —
publish each viewpoint as a ``/way_point_with_heading`` goal and let the base
autonomy (waypointConverter + local planner) drive there, using the stack's own
``/way_point_reached`` distance signal for arrival.

For each viewpoint we: drive to it (pose-based arrival, or a timeout for
unreachable points), wait until the robot is stationary with fresh sensor
frames, then record the 7 legal challenge inputs at the ACHIEVED pose. The
captured dataset is replayed offline (Part 2) through perception + eval.

Runs INSIDE the sim container (needs rclpy + the sim topics), with the
ai_module container STOPPED so nothing else publishes waypoints:

    docker stop xiao_hei_ai_module xiao_hei_perception
    docker cp perception_benchmark/capture_viewpoints.py iros2026_system:/tmp/cap.py
    docker cp perception_benchmark/viewpoints/livingroom_3.json iros2026_system:/tmp/vps.json
    docker exec -it iros2026_system bash -lc \
      'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
       python3 /tmp/cap.py /tmp/vps.json --scene livingroom_3 --out /tmp/captures'
    docker cp iros2026_system:/tmp/captures ./perception_benchmark/captures

Per viewpoint it writes ``captures/<scene>/vp_XXX/``:
    image.npy  registered_scan.npy  sensor_scan.npy  terrain_map.npy
    terrain_map_ext.npy  pose.json  meta.json    (+ scene manifest.json)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32
import sensor_msgs_py.point_cloud2 as pc2

CLOUD_TOPICS = {
    "registered_scan": "/registered_scan",
    "sensor_scan": "/sensor_scan",
    "terrain_map": "/terrain_map",
    "terrain_map_ext": "/terrain_map_ext",
}
PUB_HZ = 5.0


def image_to_array(msg: Image) -> np.ndarray:
    ch = msg.step // msg.width if msg.width else 1
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return arr.reshape(msg.height, msg.width, ch)


def cloud_to_array(msg: PointCloud2) -> np.ndarray:
    """(N,4) float32 — x,y,z,intensity (0 when the cloud has no intensity)."""
    names = [f.name for f in msg.fields]
    pts = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
    xyz = np.stack([pts["x"], pts["y"], pts["z"]], axis=-1).astype(np.float32)
    if "intensity" in names:
        inten = pc2.read_points(msg, field_names=("intensity",), skip_nans=False)
        col = inten["intensity"].astype(np.float32).reshape(-1, 1)
    else:
        col = np.zeros((len(xyz), 1), dtype=np.float32)
    return np.hstack([xyz, col])


class DriveCapture(Node):
    def __init__(self):
        super().__init__("viewpoint_drive_capture")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        nav_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        # Publish goals on /way_point_with_heading so waypointConverter tracks
        # them and emits /way_point_reached (it ignores raw /way_point). This is
        # the official waypoint path (challenge instruction-following topic).
        self.wp_pub = self.create_publisher(Pose2D, "/way_point_with_heading", 5)
        self.latest = {}
        self.seq = {}
        self._sub(Odometry, "/state_estimation", "pose", 10)
        self._sub(Image, "/camera/image", "image", sensor_qos)
        for key, topic in CLOUD_TOPICS.items():
            self._sub(PointCloud2, topic, key, sensor_qos)
        # The autonomy stack's authoritative arrival signal. Its Float32 payload
        # is the stack's CURRENT DISTANCE to the (safe-adjusted) waypoint, not a
        # discrete event — "reached" = value drops below ~0.9 m (see app/main.py,
        # which notes the stack settles at 0.25-0.90 m). MUST be BEST_EFFORT or
        # the subscription silently receives nothing (QoS mismatch).
        self._sub(Float32, "/way_point_reached", "wp_reached", nav_qos)
        self.cur_wp = None                    # (x, y) currently commanded
        self.cur_yaw = 0.0                    # theta for /way_point_with_heading

    def _sub(self, typ, topic, key, qos):
        self.seq[key] = 0
        self.create_subscription(typ, topic, lambda m, k=key: self._on(k, m), qos)

    def _on(self, key, msg):
        self.seq[key] += 1
        self.latest[key] = (self.seq[key], msg)

    # -- pose helpers ------------------------------------------------------------
    def _pose(self):
        return self.latest["pose"][1].pose.pose if "pose" in self.latest else None

    def _speed(self):
        if "pose" not in self.latest:
            return None
        t = self.latest["pose"][1].twist.twist.linear
        return math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)

    def _dist_to(self, xy):
        p = self._pose()
        if p is None:
            return None
        return math.hypot(p.position.x - xy[0], p.position.y - xy[1])

    # -- command + spin ----------------------------------------------------------
    def _publish_wp(self):
        if self.cur_wp is None:
            return
        m = Pose2D()
        m.x = float(self.cur_wp[0]); m.y = float(self.cur_wp[1])
        m.theta = float(self.cur_yaw)
        self.wp_pub.publish(m)

    def _pump(self, secs, *, publish=True):
        """Spin for ``secs``, republishing the current waypoint at PUB_HZ."""
        end = time.monotonic() + secs
        next_pub = 0.0
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if publish and time.monotonic() >= next_pub:
                self._publish_wp()
                next_pub = time.monotonic() + 1.0 / PUB_HZ

    # -- drive + capture ---------------------------------------------------------
    def _wp_reached_val(self):
        m = self.latest.get("wp_reached")
        return float(m[1].data) if m is not None else None

    def drive_to(self, xy, *, timeout_s, reached_thresh, yaw=0.0):
        """Command the waypoint and wait until the autonomy stack's
        ``/way_point_reached`` distance drops below ``reached_thresh`` on a
        FRESH reading. Returns (arrived, dist_to_commanded, reached_val).
        ``dist`` is informational — the stack may settle at a safe nearby point,
        not the literal commanded XY."""
        self.cur_wp = xy
        self.cur_yaw = yaw
        mark = self.seq.get("wp_reached", 0)
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            self._pump(1.0 / PUB_HZ)
            val = self._wp_reached_val()
            # require >=2 fresh readings so we don't trust a stale small value
            # left over from the previous waypoint
            if self.seq.get("wp_reached", 0) - mark >= 2 and val is not None \
                    and val <= reached_thresh:
                return True, self._dist_to(xy), val
        return False, self._dist_to(xy), self._wp_reached_val()

    def settle_and_wait_fresh(self, *, settle_s, timeout_s, speed_max, min_new):
        """Hold position; wait until stationary AND every required topic has
        produced >= min_new fresh messages. Returns True if clean."""
        mark = {k: self.seq[k] for k in self.seq}
        required = ["image", "pose", *CLOUD_TOPICS.keys()]
        self._pump(settle_s)                  # let motion/render settle
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._pump(0.1)
            fresh = all(self.seq[k] - mark.get(k, 0) >= min_new for k in required)
            spd = self._speed()
            if fresh and spd is not None and spd <= speed_max:
                return True
        return False

    def record(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        img = image_to_array(self.latest["image"][1])
        np.save(out_dir / "image.npy", img)
        for key in CLOUD_TOPICS:
            np.save(out_dir / f"{key}.npy", cloud_to_array(self.latest[key][1]))
        p = self._pose()
        json.dump({
            "position": [p.position.x, p.position.y, p.position.z],
            "orientation_xyzw": [p.orientation.x, p.orientation.y,
                                 p.orientation.z, p.orientation.w],
        }, open(out_dir / "pose.json", "w"), indent=2)
        return {"image_shape": list(img.shape),
                "image_encoding": self.latest["image"][1].encoding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("viewpoints", help="viewpoints/<scene>.json from viewgen.py")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/captures"))
    ap.add_argument("--wp-timeout", type=float, default=40.0,
                    help="s to wait for /way_point_reached before giving up on a wp")
    ap.add_argument("--reached-thresh", type=float, default=0.92,
                    help="/way_point_reached distance (m) below which the wp counts as reached")
    ap.add_argument("--settle-s", type=float, default=1.5)
    ap.add_argument("--fresh-timeout", type=float, default=8.0)
    ap.add_argument("--speed-max", type=float, default=0.05, help="m/s to count as stationary")
    ap.add_argument("--limit", type=int, default=0, help="only first N (smoke test)")
    args = ap.parse_args()

    vps = json.load(open(args.viewpoints))["viewpoints"]
    if args.limit:
        vps = vps[: args.limit]
    scene_dir = args.out / args.scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = DriveCapture()
    print("[capture] warming up subscriptions ...", flush=True)
    node._pump(2.0, publish=False)

    manifest = []
    for vp in vps:
        arrived, d, reached_val = node.drive_to(
            (vp["x"], vp["y"]), timeout_s=args.wp_timeout,
            reached_thresh=args.reached_thresh, yaw=vp.get("yaw", 0.0))
        clean = node.settle_and_wait_fresh(
            settle_s=args.settle_s, timeout_s=args.fresh_timeout,
            speed_max=args.speed_max, min_new=2)
        p = node._pose()
        ach = [round(p.position.x, 3), round(p.position.y, 3), round(p.position.z, 3)] \
            if p is not None else None
        vp_dir = scene_dir / f"vp_{vp['id']:03d}"
        meta = {"id": vp["id"], "target": [vp["x"], vp["y"]],
                "achieved": ach, "dist_err": round(d, 3) if d is not None else None,
                "reached_val": round(reached_val, 3) if reached_val is not None else None,
                "arrived": arrived, "clean": clean, "covers": vp.get("covers", [])}
        if "image" in node.latest:
            meta.update(node.record(vp_dir))
            json.dump(meta, open(vp_dir / "meta.json", "w"), indent=2)
        status = "OK  " if (arrived and clean) else "WARN"
        print(f"[capture] {status} vp {vp['id']:3d} target=({vp['x']:.2f},{vp['y']:.2f}) "
              f"achieved={ach} d={meta['dist_err']} reached={meta['reached_val']} "
              f"arrived={arrived} clean={clean}", flush=True)
        manifest.append(meta)

    json.dump({"scene": args.scene, "n": len(manifest), "viewpoints": manifest},
              open(scene_dir / "manifest.json", "w"), indent=2)
    n_ok = sum(m["arrived"] and m["clean"] for m in manifest)
    print(f"[capture] DONE {n_ok}/{len(manifest)} viewpoints captured cleanly -> {scene_dir}",
          flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
