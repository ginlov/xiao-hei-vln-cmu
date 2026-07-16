#!/usr/bin/env python3
"""Branch A capture loop (TASK 10) -- runs INSIDE the sim container (ROS 2 Jazzy).

Drives the robot to a set of map-frame viewpoints via `/way_point`, waits for
arrival (`/way_point_reached` or pose within tolerance), lets it settle, then
snapshots one keyframe per stop:

    frames/NNNNNN/{rgb.npy, semantic.npy, scan.npy, meta.json,
                   AssetList.csv, Categories.csv}

rgb/semantic are bgr8 (matching branchA_sample); scan is (N,4) x,y,z,intensity
in the map frame with (0,0,0)/NaN filler dropped; meta.json carries the pose.
The legend CSVs are copied into every frame dir so host-side `branchA_gt.py`
runs unchanged. GT extraction is NOT done here (no scipy/PIL in the container)
-- copy the output dir to the host and run branchA_gt.py / export_coco.py.

Viewpoints: either an explicit list (CAP_WAYPOINTS) or auto-generated from the
live terrain map by farthest-point sampling traversable cells (CAP_AUTO=1), so
the stops spread across the room instead of clustering.

Config via env vars:
    CAP_OUT        output dir            (default /home/docker/cap_out)
    CAP_AUTO       "1" => auto coverage waypoints from terrain (default 1)
    CAP_NWP        number of waypoints   (default 20)
    CAP_MAXR       max radius from start for auto waypoints, m (default 7)
    CAP_WAYPOINTS  "x,y;x,y;..."  used when CAP_AUTO != 1
    CAP_CAP_START  "1" => also capture at the start pose (default 1)
    CAP_TIMEOUT    per-waypoint seconds  (default 20)
    CAP_REACH_TOL  arrival tolerance m   (default 0.35)
    CAP_LEGEND_DIR env dir with AssetList.csv + Categories.csv
"""
import os
import json
import time
import math
import shutil

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

ENV_DIR_DEFAULT = ("/home/docker/autonomy_stack_mecanum_wheel_platform/src/"
                   "base_autonomy/vehicle_simulator/mesh/unity/environment")


def img_to_bgr8(msg: Image) -> np.ndarray:
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)
    arr = arr[:, : w * 3].reshape(h, w, 3)
    if msg.encoding == "rgb8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def cloud_to_xyzi(msg: PointCloud2) -> np.ndarray:
    """(N,4) x,y,z,intensity with (0,0,0)/NaN filler dropped."""
    try:
        pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"),
                                    skip_nans=True)
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 4)
    except Exception:
        pts = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"),
                              skip_nans=True)
        arr = np.array([[p[0], p[1], p[2], p[3]] for p in pts], dtype=np.float32)
    if len(arr):
        arr = arr[~(np.all(arr[:, :3] == 0, axis=1)
                    | np.isnan(arr[:, :3]).any(axis=1))]
    return arr


class Capturer(Node):
    def __init__(self):
        super().__init__("branchA_capturer")
        self.rgb = self.sem = self.scan = self.terrain = None
        self.pose = None
        self.reached = False
        self.create_subscription(Image, "/camera/image",
                                 self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/semantic_image",
                                 self._sem_cb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/registered_scan",
                                 self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/terrain_map_ext",
                                 self._terrain_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation",
                                 self._odom_cb, 10)
        self.create_subscription(Float32, "/way_point_reached",
                                 self._reached_cb, 10)
        self.wp_pub = self.create_publisher(PointStamped, "/way_point", 10)

    def _rgb_cb(self, m): self.rgb = img_to_bgr8(m)
    def _sem_cb(self, m): self.sem = img_to_bgr8(m)
    def _scan_cb(self, m): self.scan = cloud_to_xyzi(m)
    def _terrain_cb(self, m): self.terrain = cloud_to_xyzi(m)
    def _reached_cb(self, m): self.reached = True

    def _odom_cb(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.pose = {"position": {"x": p.x, "y": p.y, "z": p.z},
                     "orientation": {"w": o.w, "x": o.x, "y": o.y, "z": o.z}}

    @staticmethod
    def _yaw(o):
        return math.atan2(2.0 * (o["w"] * o["z"] + o["x"] * o["y"]),
                          1.0 - 2.0 * (o["y"] ** 2 + o["z"] ** 2))

    def pump(self, secs):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_ready(self, timeout=20.0):
        """Spin until all sensor buffers are populated (fixes lost start frame)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(x is not None for x in (self.rgb, self.sem, self.scan, self.pose)):
                return True
        return False

    def auto_waypoints(self, n, maxr, cost_pct=30.0):
        """Farthest-point sample traversable terrain cells -> spread viewpoints."""
        self.pump(2.0)
        if self.terrain is None or len(self.terrain) == 0:
            return []
        xy = self.terrain[:, :2]
        cost = self.terrain[:, 3]
        org = (np.array([self.pose["position"]["x"], self.pose["position"]["y"]])
               if self.pose else np.zeros(2))
        d = np.linalg.norm(xy - org, axis=1)
        thr = np.percentile(cost, cost_pct)
        m = (cost <= thr) & (d <= maxr) & (d > 0.5)
        cand = xy[m]
        if len(cand) < n:
            m = (d <= maxr) & (d > 0.5)
            cand = xy[m]
        if len(cand) == 0:
            return []
        chosen = []
        mind = np.linalg.norm(cand - org, axis=1)
        for _ in range(min(n, len(cand))):
            i = int(np.argmax(mind))
            chosen.append([float(cand[i][0]), float(cand[i][1])])
            mind = np.minimum(mind, np.linalg.norm(cand - cand[i], axis=1))
        # order the spread points as a nearest-neighbour tour from the start, so
        # the robot traverses the room smoothly instead of thrashing between
        # opposite extremes (raw FPS order alternates far corners).
        remaining = chosen[:]
        cur = list(org)
        tour = []
        while remaining:
            j = min(range(len(remaining)),
                    key=lambda k: (remaining[k][0] - cur[0]) ** 2
                    + (remaining[k][1] - cur[1]) ** 2)
            cur = remaining.pop(j)
            tour.append(cur)
        return tour

    def publish_wp(self, x, y):
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), 0.0
        self.wp_pub.publish(msg)

    def drive_to(self, x, y, timeout, tol):
        self.reached = False
        t0 = time.time()
        last_pub = 0.0
        while time.time() - t0 < timeout:
            if time.time() - last_pub > 0.2:
                self.publish_wp(x, y)
                last_pub = time.time()
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is not None:
                dx = self.pose["position"]["x"] - x
                dy = self.pose["position"]["y"] - y
                if math.hypot(dx, dy) < tol or self.reached:
                    return True, math.hypot(dx, dy)
        d = (math.hypot(self.pose["position"]["x"] - x,
                        self.pose["position"]["y"] - y)
             if self.pose else float("nan"))
        return False, d

    def settle(self, timeout=8.0, pos_tol=0.02, yaw_tol=0.5, window=1.0):
        """Halt the robot and wait until it is truly stationary before a snapshot.

        The local planner keeps executing the last (often unreached) waypoint
        after drive_to() returns, so the robot may still be rotating. A few
        degrees of rotation desyncs the async rgb/semantic streams and floats
        the boxes off the objects in the equirect panorama. Hold position by
        commanding the current pose as the waypoint, and require both position
        and yaw to stay stable over a sliding window."""
        if self.pose is None:
            self.pump(window)
            return False
        yaw_tol = math.radians(yaw_tol)
        hist = []  # (t, x, y, yaw)
        t0 = time.time()
        last_pub = 0.0
        while time.time() - t0 < timeout:
            p = self.pose["position"]
            if time.time() - last_pub > 0.2:
                self.publish_wp(p["x"], p["y"])  # command "stay put" -> brake
                last_pub = time.time()
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.time()
            yaw = self._yaw(self.pose["orientation"])
            hist.append((now, self.pose["position"]["x"],
                         self.pose["position"]["y"], yaw))
            hist = [h for h in hist if now - h[0] <= window]
            if now - hist[0][0] >= window and len(hist) >= 3:
                xs = [h[1] for h in hist]; ys = [h[2] for h in hist]
                yaws = [h[3] for h in hist]
                dpos = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
                dyaw = max(yaws) - min(yaws)
                # unwrap guard for the +/-pi seam
                dyaw = min(dyaw, abs(2 * math.pi - dyaw))
                if dpos <= pos_tol and dyaw <= yaw_tol:
                    return True
        return False

    def fresh_pair(self, timeout=3.0):
        """Drop the buffered images and wait for a freshly arrived rgb+semantic
        pair, so the saved frame is a coherent snapshot (matters most while the
        sim is republishing). Safe because settle() has already stopped motion."""
        self.rgb = self.sem = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rgb is not None and self.sem is not None:
                return True
        return False

    def capture(self, outdir, idx, legend_dir):
        self.fresh_pair()
        self.pump(0.5)
        if self.rgb is None or self.sem is None or self.scan is None:
            self.get_logger().warn(f"frame {idx}: missing data, skipping")
            return False
        fr = os.path.join(outdir, "frames", f"{idx:06d}")
        os.makedirs(fr, exist_ok=True)
        np.save(os.path.join(fr, "rgb.npy"), self.rgb)
        np.save(os.path.join(fr, "semantic.npy"), self.sem)
        np.save(os.path.join(fr, "scan.npy"), self.scan)
        json.dump({"frame": f"{idx:06d}", "pose": self.pose, "stamp": time.time()},
                  open(os.path.join(fr, "meta.json"), "w"), indent=2)
        for csv in ("AssetList.csv", "Categories.csv"):
            src = os.path.join(legend_dir, csv)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(fr, csv))
        return True


def parse_waypoints(s):
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if chunk:
            x, y = chunk.split(",")
            out.append((float(x), float(y)))
    return out


def main():
    outdir = os.environ.get("CAP_OUT", "/home/docker/cap_out")
    auto = os.environ.get("CAP_AUTO", "1") == "1"
    nwp = int(os.environ.get("CAP_NWP", "20"))
    maxr = float(os.environ.get("CAP_MAXR", "7"))
    cap_start = os.environ.get("CAP_CAP_START", "1") == "1"
    timeout = float(os.environ.get("CAP_TIMEOUT", "20"))
    tol = float(os.environ.get("CAP_REACH_TOL", "0.35"))
    legend_dir = os.environ.get("CAP_LEGEND_DIR", ENV_DIR_DEFAULT)

    os.makedirs(os.path.join(outdir, "frames"), exist_ok=True)
    warmup = float(os.environ.get("CAP_WARMUP", "60"))
    rclpy.init()
    node = Capturer()
    if not node.wait_ready(warmup):
        print("WARN: sensors not all ready after warm-up", flush=True)
    else:
        print("sensors ready", flush=True)

    if auto:
        wps = node.auto_waypoints(nwp, maxr)
        print(f"auto-generated {len(wps)} coverage waypoints (FPS over terrain)",
              flush=True)
    else:
        wps = parse_waypoints(os.environ.get("CAP_WAYPOINTS", "1.0,0.0;0.0,1.0"))

    manifest = {"waypoints": wps, "frames": []}
    idx = 0

    if cap_start:
        ok = node.capture(outdir, idx, legend_dir)
        p = node.pose["position"] if node.pose else None
        print(f"[frame {idx:06d}] start pose={p} saved={ok}", flush=True)
        if ok:
            manifest["frames"].append({"idx": idx, "waypoint": None, "pose": p})
            idx += 1

    for (x, y) in wps:
        reached, d = node.drive_to(x, y, timeout, tol)
        stable = node.settle()
        ok = node.capture(outdir, idx, legend_dir)
        p = node.pose["position"] if node.pose else None
        print(f"[frame {idx:06d}] wp=({x:.2f},{y:.2f}) reached={reached} "
              f"dist={d:.2f} stable={stable} pose=({p['x']:.2f},{p['y']:.2f}) "
              f"saved={ok}", flush=True)
        if ok:
            manifest["frames"].append(
                {"idx": idx, "waypoint": [x, y], "reached": reached,
                 "dist": round(d, 3), "pose": p})
            idx += 1

    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
    print(f"DONE  {idx} frames -> {outdir}", flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
