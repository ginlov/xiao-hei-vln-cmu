"""Run the sidecar detector on every captured viewpoint and save the masks.

This freezes the ONE lifter input that isn't already captured — the detection
masks — so the 3D lifter can be debugged fully offline (mask + registered_scan +
pose), without the sidecar and with deterministic, reproducible inputs.

For each captures/<scene>/vp_XXX/ it writes detections.npz:
    masks   (K, 640, 1920) bool     — per-detection equirect masks (lifter input)
    labels  (K,)           str
    scores  (K,)           float32
    bboxes  (K, 4)         float32   — xyxy (context)
Compressed (masks are sparse) so the footprint stays small.

Needs the sidecar up (localhost:8001).

    uv run --extra perception python perception_benchmark/dump_detections.py --all
    uv run --extra perception python perception_benchmark/dump_detections.py --scene livingroom_3
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from viewgen import load_objects
from replay_score import scoreable
from xiao_hei_vln.perception.client import HTTPPerceptionClient

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))


def dump_scene(scene, *, base_url, score_threshold, keep_arch, request_timeout_s=60.0):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip"); return
    objs = load_objects(scene)
    keep = objs if keep_arch else {i: e for i, e in objs.items() if scoreable(e.label)}
    classes = tuple(sorted({e.label for e in keep.values()}))

    client = HTTPPerceptionClient(base_url=base_url, request_timeout_s=request_timeout_s)
    client.wait_until_ready()
    client.set_classes(classes)

    total = 0
    for vp_dir in vp_dirs:
        vid = os.path.basename(vp_dir)
        img = np.load(Path(vp_dir) / "image.npy")
        dets = client.detect(img, score_threshold=score_threshold)
        if dets:
            masks = np.stack([d.mask for d in dets]).astype(bool)
            labels = np.array([d.label for d in dets], dtype=object)
            scores = np.array([d.score for d in dets], dtype=np.float32)
            bboxes = np.array([d.bbox_xyxy for d in dets], dtype=np.float32)
        else:
            masks = np.zeros((0, img.shape[0], img.shape[1]), dtype=bool)
            labels = np.array([], dtype=object)
            scores = np.zeros((0,), dtype=np.float32)
            bboxes = np.zeros((0, 4), dtype=np.float32)
        np.savez_compressed(Path(vp_dir) / "detections.npz",
                            masks=masks, labels=labels, scores=scores, bboxes=bboxes,
                            score_threshold=score_threshold, classes=np.array(classes, dtype=object))
        total += len(dets)
        sz = (Path(vp_dir) / "detections.npz").stat().st_size / 1024
        print(f"[{scene}] {vid}: {len(dets)} masks saved ({sz:.0f} KB)")
    print(f"[{scene}] done — {total} masks across {len(vp_dirs)} viewpoints")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", "http://localhost:8001"))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--keep-arch", action="store_true",
                    help="detect ALL object classes incl. wall/floor/ceiling (default: scoreable only)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    if args.all:
        scenes = sorted(os.path.basename(os.path.dirname(m))
                        for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
    elif args.scene:
        scenes = [args.scene]
    else:
        ap.error("pass --scene <name> or --all")
    for s in scenes:
        dump_scene(s, base_url=args.base_url, score_threshold=args.score_threshold,
                   keep_arch=args.keep_arch, request_timeout_s=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
