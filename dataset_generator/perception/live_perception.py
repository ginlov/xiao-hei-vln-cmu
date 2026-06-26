#!/usr/bin/env python3
"""Phase 2: live perception worker -- runs INSIDE the sim container.

Single process, two decoupled stages (the producer/consumer split the user
asked for):

  PRODUCER  rclpy node, subscribes /camera/image + /registered_scan +
            /state_estimation, keeps the latest snapshot, and on a timer emits a
            *keyframe* (rgb.npy/scan.npy/meta.json) whenever the robot is
            roughly stationary and has moved enough since the last keyframe.
            Keyframes go onto a bounded queue (newest wins, old dropped) so a
            slow consumer never stalls the ROS callbacks.

  CONSUMER  worker thread, pulls a keyframe and runs the offline pipeline
            unchanged -- detect (YOLO-World + SAM2.1, CUDA) -> lift3d ->
            ObjectMap.add_frame -- maintaining a persistent live object map,
            and prints / writes /tmp/live_map.json every cycle.

Drive the robot manually meanwhile (rviz waypoint clicks). Run:
    /opt/percep/bin/python perception/live_perception.py --names names.json
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import queue
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, String, Empty
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from detect import dets_from_yolo_sam
from lift3d import lift_frame
from objectmap import ObjectMap


def img_to_bgr8(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def cloud_to_xyzi(msg: PointCloud2) -> np.ndarray:
    pts = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"),
                                skip_nans=True)
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 4)
    if len(arr):
        arr = arr[~(np.all(arr[:, :3] == 0, axis=1)
                    | np.isnan(arr[:, :3]).any(axis=1))]
    return arr


def build_markers(objs, node) -> MarkerArray:
    """Merged object boxes -> rviz MarkerArray (map frame). Red=weak->green=strong
    by n_obs; a text marker labels each. Leading DELETEALL clears stale boxes."""
    arr = MarkerArray()
    now = node.get_clock().now().to_msg()
    clear = Marker(); clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    for i, o in enumerate(objs):
        c = o["center_3d"]; s = o["bbox_aabb"]["size"]
        t = min(o["n_obs"], 6) / 6.0
        m = Marker()
        m.header.frame_id = "map"; m.header.stamp = now
        m.ns = o["label"]; m.id = i; m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = c
        m.pose.orientation.w = 1.0
        m.scale.x = max(s[0], 0.05); m.scale.y = max(s[1], 0.05); m.scale.z = max(s[2], 0.05)
        m.color.r, m.color.g, m.color.b, m.color.a = 1 - t, 0.3 + 0.6 * t, 0.1, 0.35
        arr.markers.append(m)
        tx = Marker()
        tx.header.frame_id = "map"; tx.header.stamp = now
        tx.ns = o["label"]; tx.id = 100000 + i; tx.type = Marker.TEXT_VIEW_FACING; tx.action = Marker.ADD
        tx.pose.position.x, tx.pose.position.y = c[0], c[1]
        tx.pose.position.z = c[2] + s[2] / 2 + 0.1
        tx.pose.orientation.w = 1.0
        tx.scale.z = 0.18
        tx.color.r = tx.color.g = tx.color.b = tx.color.a = 1.0
        tx.text = f"{o['label']}({o['n_obs']})"
        arr.markers.append(tx)
    return arr


_CLOUD_FIELDS = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                 for n, o in (("x", 0), ("y", 4), ("z", 8), ("rgb", 12))]


def build_cloud(nodes, node, per_obj_cap=800) -> PointCloud2:
    """Each object's accumulated LiDAR points, one distinct colour per object, as
    a single PointCloud2 -- so every inferred object shows as a solid colour blob
    in rviz (golden-ratio hue spacing keeps neighbours visually distinct)."""
    pts = []
    for i, nd in enumerate(nodes):
        r, g, b = (int(255 * c) for c in colorsys.hsv_to_rgb((0.61803 * i) % 1.0, 0.85, 1.0))
        rgb = struct.unpack("f", struct.pack("I", (r << 16) | (g << 8) | b))[0]
        p = nd.pts
        if len(p) > per_obj_cap:
            p = p[np.random.choice(len(p), per_obj_cap, False)]
        pts.extend([float(p[k, 0]), float(p[k, 1]), float(p[k, 2]), rgb]
                   for k in range(len(p)))
    h = Header(frame_id="map", stamp=node.get_clock().now().to_msg())
    return pc2.create_cloud(h, _CLOUD_FIELDS, pts)


class Producer(Node):
    def __init__(self, q, outdir, period, move_min, still_tol):
        super().__init__("live_perception_producer")
        self.q, self.outdir = q, outdir
        self.period, self.move_min, self.still_tol = period, move_min, still_tol
        self.marker_pub = self.create_publisher(MarkerArray, "/perception/objects", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/perception/object_points", 10)
        self.latest_objs = None         # set by consumer; (re)published at 1 Hz so a
        self.latest_nodes = None        # late-added rviz Display always sees the map
        self.filter = None              # None=all; else set of labels to show
        self.reset_requested = False    # consumer clears the map when set
        self.create_subscription(String, "/perception/filter", self._set_filter, 10)
        self.create_subscription(Empty, "/perception/reset", self._reset, 10)
        self.rgb = self.scan = self.pose = None
        self.hist = []                              # (t, x, y) for stationary test
        self.last_kf_xy = None
        self.idx = 0
        self.create_subscription(Image, "/camera/image", self._rgb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/registered_scan", self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation", self._odom, 10)
        self.create_timer(period, self._tick)
        self.create_timer(1.0, self._republish)

    def _reset(self, _msg: Empty):
        """Clear the accumulated map (consumer rebuilds from empty). Also wipe the
        current display immediately so rviz clears without waiting for a frame."""
        self.reset_requested = True
        self.latest_objs = []; self.latest_nodes = []
        self.publish_map()
        self.get_logger().info("map reset requested")

    def _set_filter(self, msg: String):
        """Live class filter. data='' or 'all' -> show everything; otherwise a
        comma-separated label list, e.g. 'couch' or 'couch,chair'."""
        d = msg.data.strip().lower()
        self.filter = None if d in ("", "all") else {s.strip() for s in d.split(",")}
        self.get_logger().info(f"filter set to: {self.filter or 'ALL'}")
        self.publish_map()

    def publish_map(self):
        """Build + publish markers and cloud from the current map, applying the
        active class filter. Called on each new frame and by the 1 Hz timer."""
        if self.latest_objs is None:
            return
        f = self.filter
        objs = [o for o in self.latest_objs if f is None or o["label"] in f]
        nodes = [n for n in self.latest_nodes if f is None or n.label in f]
        self.marker_pub.publish(build_markers(objs, self))
        self.cloud_pub.publish(build_cloud(nodes, self))

    def _republish(self):
        self.publish_map()

    def _rgb(self, m): self.rgb = img_to_bgr8(m)
    def _scan(self, m): self.scan = cloud_to_xyzi(m)

    def _odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.pose = {"position": {"x": p.x, "y": p.y, "z": p.z},
                     "orientation": {"w": o.w, "x": o.x, "y": o.y, "z": o.z}}
        self.hist.append((time.time(), p.x, p.y))
        self.hist = [h for h in self.hist if h[0] > time.time() - 1.0]

    def _stationary(self) -> bool:
        if len(self.hist) < 3:
            return False
        xs = [h[1] for h in self.hist]; ys = [h[2] for h in self.hist]
        return (max(xs) - min(xs) < self.still_tol
                and max(ys) - min(ys) < self.still_tol)

    def _tick(self):
        if self.rgb is None or self.scan is None or self.pose is None:
            return
        if not self._stationary():
            return
        xy = (self.pose["position"]["x"], self.pose["position"]["y"])
        if self.last_kf_xy is not None:
            if math.dist(xy, self.last_kf_xy) < self.move_min:
                return                              # too close to last keyframe
        fr = os.path.join(self.outdir, f"{self.idx:06d}")
        os.makedirs(fr, exist_ok=True)
        np.save(os.path.join(fr, "rgb.npy"), self.rgb)
        np.save(os.path.join(fr, "scan.npy"), self.scan)
        json.dump({"frame": f"{self.idx:06d}", "pose": self.pose,
                   "stamp": time.time()}, open(os.path.join(fr, "meta.json"), "w"))
        if self.q.full():
            try: self.q.get_nowait()                # drop oldest
            except queue.Empty: pass
        self.q.put(fr)
        self.last_kf_xy = xy
        self.idx += 1
        self.get_logger().info(f"keyframe {self.idx} @ ({xy[0]:.1f},{xy[1]:.1f}) "
                               f"queued ({self.q.qsize()})")


def consumer(q, names, device, out_json, stop, node):
    from pathlib import Path
    omap = ObjectMap()
    n = 0
    while not stop.is_set():
        if node.reset_requested:
            omap = ObjectMap(); node.reset_requested = False
            node.latest_objs = []; node.latest_nodes = []
            n = 0; print("[reset] map cleared", flush=True)
        try:
            fr = Path(q.get(timeout=1.0))
        except queue.Empty:
            continue
        t0 = time.time()
        dets = dets_from_yolo_sam(fr, names, device=device)
        res = lift_frame(fr, dets, keep_pts=True)
        omap.add_frame(res["objects"])
        n += 1
        node.latest_nodes = omap.export_nodes(min_pts=15)
        objs = omap.export(min_pts=15)
        node.latest_objs = objs
        json.dump({"frames": n, "objects": objs}, open(out_json, "w"), indent=2)
        node.publish_map()                                        # filtered -> rviz
        strong = [o for o in objs if o["n_obs"] >= 2]
        print(f"[{n}] {fr.name}: {len(dets)} dets -> {res['n_objects']} 3D | "
              f"map {len(objs)} objs ({len(strong)} strong) | "
              f"{time.time()-t0:.1f}s", flush=True)
        for o in sorted(strong, key=lambda o: -o["n_obs"])[:8]:
            c = o["center_3d"]
            print(f"      {o['label']:18s} obs={o['n_obs']:2d} "
                  f"({c[0]:5.1f},{c[1]:5.1f},{c[2]:4.1f})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", required=True, help="json list of class names")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="/tmp/live_frames")
    ap.add_argument("--out-json", default="/tmp/live_map.json")
    ap.add_argument("--period", type=float, default=1.0, help="keyframe check period s")
    ap.add_argument("--move-min", type=float, default=0.3, help="min move between keyframes m")
    ap.add_argument("--still-tol", type=float, default=0.05, help="stationary window m")
    args = ap.parse_args()

    names = json.load(open(args.names))
    os.makedirs(args.outdir, exist_ok=True)
    q: queue.Queue = queue.Queue(maxsize=3)
    stop = threading.Event()

    rclpy.init()
    prod = Producer(q, args.outdir, args.period, args.move_min, args.still_tol)
    worker = threading.Thread(target=consumer,
                              args=(q, names, args.device, args.out_json, stop, prod),
                              daemon=True)
    worker.start()
    print(f"live perception up: device={args.device}, {len(names)} classes. "
          f"Drive via rviz; map -> {args.out_json}", flush=True)
    try:
        rclpy.spin(prod)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        prod.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
