#!/usr/bin/env python3
"""Passive recorder — captures the sensor stream during REAL navigation.

The sibling harness (``capture_viewpoints.py``) drives the robot to the
geometrically-chosen viewpoints from ``viewgen.py``. That answers "how good
could perception be, given ideal coverage?" — it does NOT answer "what does
perception actually see on the trajectory the robot really takes." This script
answers the second question.

It reproduces the live ingest cadence exactly. In ``app/main.py`` the node ticks
at ``XIAO_HEI_VLM_TICK_HZ`` (default **2 Hz**, i.e. every 0.5 s) and every
exploration tick calls ``responder.ingest()`` → ``_inject_visible()``, which runs
detect → lift → fuse *unconditionally*. There is no distance, rotation or
motion-blur gate anywhere on that path: a tick is skipped only when the snapshot
is missing an image / pose / scan, and a blurred frame simply yields no
detections and leaves the scene unchanged. So this recorder samples at a fixed
``--rate-hz`` (default 2.0) and applies no quality filter — anything else would
hand the offline replay a cleaner input stream than the live stack ever gets.

There are deliberately no quality, motion or warm-up knobs. Every frame the
sampler can take, it takes — because that is what the live node does. The one
non-system flag is ``--max-seconds``, a disk backstop; when it fires the manifest
records ``stop_reason: max_seconds`` so a truncated dataset can never be mistaken
for a complete one.

It publishes NOTHING. The ai_module container drives (``XIAO_HEI_RESPONDER=dummy``
with exploration enabled); this node only watches. Output layout is identical to
``capture_viewpoints.py``'s, so ``replay_score.py`` / ``dump_debug.py`` /
``viz_app.py`` consume it unchanged (point them at the new root with
``PERCEPTION_CAP_DIR``).

Runs INSIDE the sim container (needs rclpy + the sim topics), alongside a running
ai_module. Normally you want ``run_nav_capture.sh``, which orchestrates the whole
sweep; run it by hand only to debug one scene:

    docker cp perception_benchmark/capture_viewpoints.py iros2026_system:/tmp/capture_viewpoints.py
    docker cp perception_benchmark/record_navigation.py  iros2026_system:/tmp/record_navigation.py
    docker exec -it iros2026_system bash -lc \
      'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
       python3 /tmp/record_navigation.py --scene arabic_room --out /tmp/nav_captures'

DISK: a frame is ~4.2 MB (the 640x1920 uint8 image is 3.7 MB of it). At 2 Hz that
is ~500 MB per minute of exploration, so a 10-minute scene costs ~5 GB and a
15-scene sweep runs into tens of GB. ``--rate-hz`` is the knob; check free space
before a full sweep.

Stops on SIGINT/SIGTERM, on ``--stop-file`` appearing, or on the ``--max-seconds``
backstop — and flushes ``manifest.json`` after EVERY frame,
so an abrupt kill never costs more than the frame in flight.

Per frame it writes ``<out>/<scene>/vp_XXX/``:
    image.npy  registered_scan.npy  sensor_scan.npy  terrain_map.npy
    terrain_map_ext.npy  pose.json  meta.json    (+ scene manifest.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2

# Share the wire-format decoders with the viewpoint harness rather than
# re-deriving them: the intensity-column handling in cloud_to_array is subtle
# and a second copy would drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_viewpoints import CLOUD_TOPICS, cloud_to_array, image_to_array


def yaw_of(q) -> float:
    """Yaw (rad) from a ROS quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavRecorder(Node):
    """Subscribe-only fixed-rate recorder. Never publishes — the explorer drives."""

    def __init__(self, *, out_dir: Path, period_s: float):
        super().__init__("nav_recorder")
        self.out_dir = out_dir
        self.period_s = period_s

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.latest: dict = {}
        self.seq: dict[str, int] = {}
        self._sub(Odometry, "/state_estimation", "pose", 10)
        self._sub(Image, "/camera/image", "image", sensor_qos)
        for key, topic in CLOUD_TOPICS.items():
            self._sub(PointCloud2, topic, key, sensor_qos)

        self.frames: list[dict] = []
        self.last_img_seq = -1
        self.next_due = 0.0
        # Diagnostics, reported in the manifest. `incomplete` is expected to be
        # small but non-zero: it counts the startup ticks before every topic has
        # published, which is exactly what the live node does too (it skips
        # those ticks in _inject_visible rather than waiting for them).
        self.skipped_incomplete = 0    # a topic had not published yet
        self.skipped_stale_image = 0   # camera stalled (it runs ~10 Hz vs a 2 Hz tick)

    def _sub(self, typ, topic, key, qos):
        self.seq[key] = 0
        self.create_subscription(typ, topic, lambda m, k=key: self._on(k, m), qos)

    def _on(self, key, msg):
        self.seq[key] += 1
        self.latest[key] = msg

    # -- state helpers -----------------------------------------------------------
    def _pose(self):
        odom = self.latest.get("pose")
        return odom.pose.pose if odom is not None else None

    def _speed(self) -> float | None:
        odom = self.latest.get("pose")
        if odom is None:
            return None
        t = odom.twist.twist.linear
        return math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)

    # -- sampling decision -------------------------------------------------------
    def should_capture(self, now: float) -> bool:
        """Fixed-rate sampling, mirroring the live node's timer. The only
        rejections are ones the live stack makes too — there is no motion gate,
        no blur gate and no warm-up, because ``_inject_visible`` has none."""
        if now < self.next_due:
            return False
        self.next_due = now + self.period_s

        # Same guard as _inject_visible: no image / pose / scan -> skip the tick.
        if not all(k in self.latest for k in ("pose", "image", *CLOUD_TOPICS)):
            self.skipped_incomplete += 1
            return False
        # /camera/image runs ~10 Hz against a 2 Hz tick, so there is always a
        # fresh frame. If there isn't, the camera has stalled — skip the
        # byte-identical copy and count it, because a non-zero count means the
        # live stack would have re-ingested (and re-fused) that stale image.
        if self.seq["image"] == self.last_img_seq:
            self.skipped_stale_image += 1
            return False

        if self._pose() is None:
            self.skipped_incomplete += 1
            return False
        return True

    # -- write -------------------------------------------------------------------
    def record(self) -> dict:
        idx = len(self.frames)
        vp_dir = self.out_dir / f"vp_{idx:03d}"
        vp_dir.mkdir(parents=True, exist_ok=True)

        img_msg = self.latest["image"]
        img = image_to_array(img_msg)
        np.save(vp_dir / "image.npy", img)
        for key in CLOUD_TOPICS:
            np.save(vp_dir / f"{key}.npy", cloud_to_array(self.latest[key]))
        p = self._pose()
        json.dump({"position": [p.position.x, p.position.y, p.position.z],
                   "orientation_xyzw": [p.orientation.x, p.orientation.y,
                                        p.orientation.z, p.orientation.w]},
                  open(vp_dir / "pose.json", "w"), indent=2)

        spd = self._speed()
        yaw = yaw_of(p.orientation)
        meta = {
            "id": idx,
            "t": self.get_clock().now().nanoseconds / 1e9,
            "achieved": [round(p.position.x, 3), round(p.position.y, 3),
                         round(p.position.z, 3)],
            "yaw": round(yaw, 4),
            # Kept for analysis, NOT used as a filter — the live stack ingests
            # fast/blurred frames too and simply gets no detections from them.
            "speed": round(spd, 3) if spd is not None else None,
            "image_shape": list(img.shape),
            "image_encoding": img_msg.encoding,
            "source": "navigation",
        }
        json.dump(meta, open(vp_dir / "meta.json", "w"), indent=2)

        self.frames.append(meta)
        self.last_img_seq = self.seq["image"]
        return meta

    def flush_manifest(self, scene: str, params: dict,
                       stop_reason: str | None = None) -> None:
        """Rewritten after every frame via a temp file + rename, so a kill
        mid-write can never leave a truncated manifest behind."""
        payload = {"scene": scene, "n": len(self.frames), "source": "navigation",
                   "params": params, "stop_reason": stop_reason,
                   "skipped": {"incomplete": self.skipped_incomplete,
                               "stale_image": self.skipped_stale_image},
                   "viewpoints": self.frames}
        tmp = self.out_dir / "manifest.json.tmp"
        json.dump(payload, open(tmp, "w"), indent=2)
        tmp.replace(self.out_dir / "manifest.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/nav_captures"))
    # Defaults to the live node's own tick rate, so the two stay in step even if
    # XIAO_HEI_VLM_TICK_HZ is overridden.
    ap.add_argument("--rate-hz", type=float,
                    default=float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0")),
                    help="sampling rate; defaults to the live XIAO_HEI_VLM_TICK_HZ (2.0)")
    ap.add_argument("--stop-file", type=Path, default=Path("/tmp/stop_capture"),
                    help="polled; recorder exits cleanly once this path exists")
    ap.add_argument("--max-seconds", type=float, default=1800.0,
                    help="disk backstop, NOT a system behaviour: stop after this long even "
                         "if no stop signal arrives. Recorded as stop_reason. 0 = unlimited")
    args = ap.parse_args()

    # A stale stop-file from a previous scene would end this run instantly.
    if args.stop_file.exists():
        args.stop_file.unlink()

    out_dir = args.out / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    params = {"rate_hz": args.rate_hz}

    rclpy.init()
    node = NavRecorder(out_dir=out_dir, period_s=period_s)

    stopping = {"flag": False}

    def _on_signal(_sig, _frm):
        stopping["flag"] = True                  # let the loop exit and flush

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # No warm-up: the live node starts its timer immediately and simply skips
    # ticks whose snapshot is incomplete. Spinning here instead would suppress
    # ticks the live stack would have attempted.
    node.flush_manifest(args.scene, params)      # exists immediately, even at n=0
    print(f"[record] recording {args.scene} at {args.rate_hz} Hz -> {out_dir}", flush=True)

    start = time.monotonic()
    node.next_due = start
    reason = "stopped"
    while not stopping["flag"]:
        rclpy.spin_once(node, timeout_sec=0.02)
        now = time.monotonic()
        if args.max_seconds and now - start >= args.max_seconds:
            reason = "max_seconds"
            break
        if args.stop_file.exists():
            reason = "stop_file"
            break
        if not node.should_capture(now):
            continue
        meta = node.record()
        node.flush_manifest(args.scene, params)
        if meta["id"] % 20 == 0:                 # 2 Hz is far too chatty to log every frame
            print(f"[record] vp {meta['id']:4d} at {meta['achieved']} "
                  f"yaw={math.degrees(meta['yaw']):7.1f}deg speed={meta['speed']}",
                  flush=True)

    node.flush_manifest(args.scene, params, stop_reason=reason)
    print(f"[record] DONE {len(node.frames)} frames reason={reason} "
          f"(skipped: incomplete={node.skipped_incomplete} "
          f"stale_image={node.skipped_stale_image}) -> {out_dir}", flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
