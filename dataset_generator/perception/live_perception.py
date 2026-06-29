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
            ObjectMap.add_frame -- maintaining a persistent live object map.
            Each cycle it (atomically) writes the comprehensive results the
            downstream VLM stage consumes, under --result-dir:
              scene_objects.json        every predicted object (label, 3D centre,
                                        bbox, score, #observations)
              scene_gt.json             the GT object map (benchmark mode)
              detections_2d/<frame>.json  that keyframe's raw 2D detections
                                        (label, score, equirect bbox, area)

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
from lift3d import dets_from_gt, lift_frame
from objectmap import ObjectMap
from eval_objectmap import evaluate


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


def build_markers(objs, node, rgba=None, with_text=True) -> MarkerArray:
    """Merged object boxes -> rviz MarkerArray (map frame). Default colour is
    red=weak->green=strong by n_obs; pass `rgba` for a fixed colour (e.g. GT in
    green). Leading DELETEALL clears stale boxes."""
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
        if rgba is not None:
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
        else:
            m.color.r, m.color.g, m.color.b, m.color.a = 1 - t, 0.3 + 0.6 * t, 0.1, 0.35
        arr.markers.append(m)
        if not with_text:
            continue
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


def build_scoreboard(metrics, node) -> MarkerArray:
    """A single floating TEXT marker (map frame) with the live benchmark numbers,
    so the score is visible right in the 3D view, updating each cycle."""
    arr = MarkerArray()
    clear = Marker(); clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    if not metrics:
        return arr
    m = Marker()
    m.header.frame_id = "map"; m.header.stamp = node.get_clock().now().to_msg()
    m.ns = "benchmark"; m.id = 0; m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
    m.pose.position.x = 0.0; m.pose.position.y = 0.0; m.pose.position.z = 3.5
    m.pose.orientation.w = 1.0
    m.scale.z = 0.35
    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
    op = metrics["operating_point"].get("dist@1.0m", {})
    m.text = (f"LIVE BENCHMARK  frames={metrics.get('frames', 0)}\n"
              f"GT {metrics['n_gt']}   PRED {metrics['n_pred']}\n"
              f"mAP@1m {metrics['mAP'].get('dist@1.0m')}  "
              f"P {op.get('precision')} R {op.get('recall')} F1 {op.get('f1')}\n"
              f"count MAE {metrics['counting_MAE']}  "
              f"match {int(round(op.get('mean_center_err_m', 0) * 100))}cm")
    arr.markers.append(m)
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


def _atomic_write_json(obj, path):
    """Write JSON via a temp file + os.replace so a reader (the VLM stage) never
    sees a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _dets_to_2d(dets):
    """Compact 2D records from equirect-mask detections, for the per-keyframe 2D
    result file: label, score, crop index, equirect bbox (xyxy) and pixel area
    derived from the mask. Returns (records, (H, W))."""
    out, hw = [], (None, None)
    for d in dets:
        m = d["mask"]
        hw = m.shape
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        out.append({
            "label": d["label"],
            "score": round(float(d["score"]), 4),
            "crop": int(d.get("crop", -1)),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area_px": int(m.sum()),
        })
    return out, hw


class Producer(Node):
    def __init__(self, q, outdir, period, move_min, still_tol,
                 benchmark=False, legend_dir=None):
        super().__init__("live_perception_producer")
        self.q, self.outdir = q, outdir
        self.period, self.move_min, self.still_tol = period, move_min, still_tol
        self.benchmark, self.legend_dir = benchmark, legend_dir
        self.marker_pub = self.create_publisher(MarkerArray, "/perception/objects", 10)
        self.cloud_pub = self.create_publisher(PointCloud2, "/perception/object_points", 10)
        self.latest_objs = None         # set by consumer; (re)published at 1 Hz so a
        self.latest_nodes = None        # late-added rviz Display always sees the map
        self.latest_gt_objs = None      # GT object map (benchmark mode)
        self.latest_metrics = None      # eval_objectmap report (benchmark mode)
        self.filter = None              # None=all; else set of labels to show
        self.reset_requested = False    # consumer clears the map when set
        self.create_subscription(String, "/perception/filter", self._set_filter, 10)
        self.create_subscription(Empty, "/perception/reset", self._reset, 10)
        self.rgb = self.scan = self.pose = self.sem = None
        self.hist = []                              # (t, x, y) for stationary test
        self.last_kf_xy = None
        self.idx = 0
        self.create_subscription(Image, "/camera/image", self._rgb, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/registered_scan", self._scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/state_estimation", self._odom, 10)
        if benchmark:
            # keep a live subscription so the lazy semantic publisher stays active
            self.gt_marker_pub = self.create_publisher(MarkerArray, "/perception/gt_objects", 10)
            self.bench_pub = self.create_publisher(MarkerArray, "/perception/benchmark", 10)
            self.create_subscription(Image, "/camera/semantic_image", self._sem,
                                     qos_profile_sensor_data)
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

    def publish_benchmark(self):
        """Publish GT boxes (green) + the live scoreboard (benchmark mode)."""
        if not self.benchmark:
            return
        if self.latest_gt_objs is not None:
            f = self.filter
            gt = [o for o in self.latest_gt_objs if f is None or o["label"] in f]
            self.gt_marker_pub.publish(
                build_markers(gt, self, rgba=(0.1, 0.9, 0.2, 0.25), with_text=False))
        self.bench_pub.publish(build_scoreboard(self.latest_metrics, self))

    def _republish(self):
        self.publish_map()
        self.publish_benchmark()

    def _rgb(self, m): self.rgb = img_to_bgr8(m)
    def _sem(self, m): self.sem = img_to_bgr8(m)
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
        if self.benchmark and self.sem is None:
            return                                  # need GT semantic to score
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
        if self.benchmark:
            np.save(os.path.join(fr, "semantic.npy"), self.sem)
            if self.legend_dir:                     # GT lift needs the palette legend
                import shutil
                for csv in ("AssetList.csv", "Categories.csv"):
                    src = os.path.join(self.legend_dir, csv)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(fr, csv))
        if self.q.full():
            try: self.q.get_nowait()                # drop oldest
            except queue.Empty: pass
        self.q.put(fr)
        self.last_kf_xy = xy
        self.idx += 1
        self.get_logger().info(f"keyframe {self.idx} @ ({xy[0]:.1f},{xy[1]:.1f}) "
                               f"queued ({self.q.qsize()})")


def consumer(q, names, device, out_json, result_dir, stop, node):
    from pathlib import Path
    # Comprehensive perception results consumed by the downstream VLM stage.
    scene_json = os.path.join(result_dir, "scene_objects.json")   # predicted map
    gt_json = os.path.join(result_dir, "scene_gt.json")           # GT map (benchmark)
    dets2d_dir = os.path.join(result_dir, "detections_2d")        # per-keyframe 2D
    images_dir = os.path.join(result_dir, "images")               # raw keyframe panoramas
    os.makedirs(dets2d_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    omap = ObjectMap()
    gt_omap = ObjectMap()
    n = 0
    while not stop.is_set():
        if node.reset_requested:
            omap = ObjectMap(); gt_omap = ObjectMap(); node.reset_requested = False
            node.latest_objs = []; node.latest_nodes = []
            node.latest_gt_objs = []; node.latest_metrics = None
            n = 0; print("[reset] map cleared", flush=True)
        try:
            fr = Path(q.get(timeout=1.0))
        except queue.Empty:
            continue
        t0 = time.time()
        dets = dets_from_yolo_sam(fr, names, device=device)
        # per-keyframe 2D detection result (consumed by the VLM stage / debugging)
        recs2d, (H2d, W2d) = _dets_to_2d(dets)
        try:
            meta2d = json.load(open(fr / "meta.json"))
        except Exception:
            meta2d = {}
        _atomic_write_json({
            "schema": "detections_2d/v1",
            "frame": fr.name,
            "stamp": time.time(),
            "pose": meta2d.get("pose"),
            "image_hw": [H2d, W2d] if H2d else None,
            "n_dets": len(recs2d),
            "detections": recs2d,
        }, os.path.join(dets2d_dir, f"{fr.name}.json"))
        # raw panorama as a viewable JPG (rgb.npy is BGR -> flip to RGB)
        try:
            from PIL import Image as _PILImage
            _PILImage.fromarray(np.load(fr / "rgb.npy")[:, :, ::-1]).save(
                os.path.join(images_dir, f"{fr.name}.jpg"), quality=90)
        except Exception as e:
            print(f"  [warn] image save failed for {fr.name}: {e}", flush=True)
        res = lift_frame(fr, dets, keep_pts=True)
        omap.add_frame(res["objects"])
        n += 1
        node.latest_nodes = omap.export_nodes(min_pts=15)
        objs = omap.export(min_pts=15)
        node.latest_objs = objs
        json.dump({"frames": n, "objects": objs}, open(out_json, "w"), indent=2)
        _atomic_write_json({                                      # comprehensive pred map
            "schema": "scene_objects/v1",
            "stamp": time.time(),
            "frames": n,
            "source": "yolo-world-v2+sam2.1",
            "n_objects": len(objs),
            "objects": objs,
        }, scene_json)
        node.publish_map()                                        # filtered -> rviz

        bench = ""
        if node.benchmark:
            gt_omap.add_frame(lift_frame(fr, dets_from_gt(fr), keep_pts=True)["objects"])
            gt_objs = gt_omap.export(min_pts=15)
            node.latest_gt_objs = gt_objs
            _atomic_write_json({                                  # GT object map
                "schema": "scene_gt/v1",
                "stamp": time.time(),
                "frames": n,
                "source": "sim-semantic-gt",
                "n_objects": len(gt_objs),
                "objects": gt_objs,
            }, gt_json)
            rep, _ = evaluate(gt_objs, objs, [1.0], [0.25])
            rep["frames"] = n
            node.latest_metrics = rep
            node.publish_benchmark()
            op = rep["operating_point"]["dist@1.0m"]
            bench = (f" | GT {rep['n_gt']} mAP@1m {rep['mAP']['dist@1.0m']} "
                     f"P{op['precision']} R{op['recall']} F1{op['f1']}")

        strong = [o for o in objs if o["n_obs"] >= 2]
        print(f"[{n}] {fr.name}: {len(dets)} dets -> {res['n_objects']} 3D | "
              f"map {len(objs)} objs ({len(strong)} strong) | "
              f"{time.time()-t0:.1f}s{bench}", flush=True)
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
    ap.add_argument("--result-dir", default="/tmp/percep_out",
                    help="dir for the comprehensive results consumed by the VLM "
                         "stage: scene_objects.json (pred), scene_gt.json "
                         "(benchmark GT), detections_2d/<frame>.json (2D)")
    ap.add_argument("--period", type=float, default=1.0, help="keyframe check period s")
    ap.add_argument("--move-min", type=float, default=0.3, help="min move between keyframes m")
    ap.add_argument("--still-tol", type=float, default=0.05, help="stationary window m")
    ap.add_argument("--benchmark", action="store_true",
                    help="also build a live GT map from /camera/semantic_image and "
                         "score the pred map against it (publishes /perception/gt_objects "
                         "+ /perception/benchmark)")
    ap.add_argument("--legend-dir", default=None,
                    help="dir with AssetList.csv + Categories.csv for the GT palette "
                         "(required with --benchmark)")
    args = ap.parse_args()

    names = json.load(open(args.names))
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    q: queue.Queue = queue.Queue(maxsize=3)
    stop = threading.Event()

    rclpy.init()
    prod = Producer(q, args.outdir, args.period, args.move_min, args.still_tol,
                    benchmark=args.benchmark, legend_dir=args.legend_dir)
    worker = threading.Thread(target=consumer,
                              args=(q, names, args.device, args.out_json,
                                    args.result_dir, stop, prod),
                              daemon=True)
    worker.start()
    print(f"live perception up: device={args.device}, {len(names)} classes"
          f"{' | BENCHMARK mode (live GT)' if args.benchmark else ''}. "
          f"Drive via rviz; results -> {args.result_dir}/ "
          f"(scene_objects.json, scene_gt.json, detections_2d/)", flush=True)
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
