"""Precompute per-detection lifts at a LOW score floor, so the viz app can
rebuild the scene graph at any YOLO threshold on the fly.

Lifting a mask to a 3D cloud depends only on the mask + the registered scan, not
on the detection score — only the *fusion* (ObjectMap) depends on the score cut.
So we lift every detection once here at a low floor (e.g. 0.05) and store each
detection's inlier cloud + score; the viewer then re-fuses instantly for any
threshold >= floor, including per-class thresholds (see viz_app "Live threshold"
tab). This mirrors the offline replay exactly: the novelty gate (position-only)
picks the perceived viewpoints, the ScanAccumulator ingests every tick, and the
same PointLifter runs — we just defer the score cut to fusion time.

Needs the sidecar up (localhost:8001), because the frozen detections.npz is
floored at 0.6 and cannot supply the sub-0.6 tail this tool exists to explore.

    PERCEPTION_CAP_DIR=perception_benchmark/captures_nav \
      uv run --extra perception python perception_benchmark/dump_lifts.py \
      --scene arabic_room --floor 0.05 --novel-viewpoint-m 0.3 \
      --scan-keyframes 2 --image-lag 0.0

Writes perception_benchmark/debug/<scene>/lifts.pkl.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
from pathlib import Path

import numpy as np

from viewgen import load_objects
from replay_score import capture_time, load_capture, scoreable
from dump_debug import DEBUG_DIR, _gt_world_aabb
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.deskew import PoseDeskew
from xiao_hei_vln.perception.lifter import (DEFAULT_CLUSTER_VOXEL_M,
    DEFAULT_MIN_INLIERS, DEFAULT_RANGE_GAP_M, PointLifter)
from xiao_hei_vln.perception.object_map import _is_structure
from xiao_hei_vln.perception.scan_accumulator import (
    DEFAULT_MAX_KEYFRAMES, DEFAULT_VOXEL_M, ScanAccumulator)

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))


def dump_scene(scene, *, base_url, floor, novel_viewpoint_m, scan_keyframes,
               scan_voxel_m, min_inliers, range_gap_m, cluster_voxel_m,
               image_lag_s, sam_thresh, request_timeout_s, out_root):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip"); return
    objs = load_objects(scene)
    # Detection query: ALL classes incl. architecture context (matches the
    # frozen freeze; wall/floor context lifts in-wall objects like doors). GT:
    # scoreable targets only, same as dump_debug.
    classes = tuple(sorted({e.label for e in objs.values()
                            if e.label.lower() != "unknown"}))
    gt = []
    for i, e in objs.items():
        if not scoreable(e.label):
            continue
        bmin, bmax = _gt_world_aabb(e)
        gt.append({"id": i, "label": e.label,
                   "center": [e.center.x, e.center.y, e.center.z],
                   "bmin": bmin, "bmax": bmax})

    client = HTTPPerceptionClient(base_url=base_url,
                                  request_timeout_s=request_timeout_s)
    client.wait_until_ready()
    client.set_classes(classes)
    lifter = PointLifter(min_inliers=min_inliers, range_gap_m=range_gap_m,
                         cluster_voxel_m=cluster_voxel_m)
    accum = ScanAccumulator(max_keyframes=scan_keyframes, voxel_m=scan_voxel_m)
    deskew = PoseDeskew(image_lag_s)

    viewpoints, lifts = [], {}
    kept_xy: list[tuple[float, float]] = []
    n_lift_total = 0
    for vp_dir in vp_dirs:
        vid = os.path.basename(vp_dir)
        img, scan, pos, ori = load_capture(Path(vp_dir))
        cloud = accum.update(scan, pos, ori)             # every tick, like live
        ori_lift = deskew.update(ori, capture_time(Path(vp_dir)))
        yaw = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y),
                         1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
        novel = True
        if novel_viewpoint_m > 0.0 and kept_xy:
            nearest = min((pos.x - kx) ** 2 + (pos.y - ky) ** 2
                          for kx, ky in kept_xy) ** 0.5
            novel = nearest > novel_viewpoint_m
        if novel and novel_viewpoint_m > 0.0:
            kept_xy.append((pos.x, pos.y))
        viewpoints.append({"id": vid, "pose": [pos.x, pos.y, pos.z],
                           "yaw": yaw, "perceived": bool(novel)})
        if not novel:
            continue
        dets = client.detect(img, score_threshold=floor)
        if sam_thresh > 0:
            dets = [d for d in dets if d.sam_score >= sam_thresh]
        vp_lifts = []
        for d in dets:
            res = lifter.lift(d.mask, cloud, pos, ori_lift)
            if res.position is None:
                continue                                 # never cleared min_inliers
            vp_lifts.append({
                "label": d.label, "score": float(d.score),
                "sam": float(d.sam_score),
                "structure": bool(_is_structure(d.label)),
                "pts": np.asarray(res.inlier_points, dtype=np.float32)})
        lifts[vid] = vp_lifts
        n_lift_total += len(vp_lifts)
        print(f"[{scene}] {vid}: {len(dets)} det -> {len(vp_lifts)} lifted "
              f"(floor {floor})")

    out = (out_root or DEBUG_DIR) / scene
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": scene, "floor": floor, "novel_viewpoint_m": novel_viewpoint_m,
        "sam_thresh": sam_thresh, "min_inliers": min_inliers,
        "scan_keyframes": scan_keyframes, "image_lag_s": image_lag_s,
        "classes": list(classes), "gt": gt,
        "viewpoints": viewpoints, "lifts": lifts}
    with open(out / "lifts.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_perc = sum(v["perceived"] for v in viewpoints)
    print(f"[{scene}] wrote {out}/lifts.pkl — {n_perc}/{len(viewpoints)} "
          f"perceived, {n_lift_total} lifted detections >= {floor}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--base-url", default=os.environ.get(
        "XIAO_HEI_PERCEPTION_BASE_URL", "http://localhost:8001"))
    ap.add_argument("--floor", type=float, default=0.05,
                    help="low YOLO score floor to lift at; the viewer re-fuses "
                         "at any threshold >= this")
    ap.add_argument("--novel-viewpoint-m", type=float, default=0.3,
                    help="capture-time novelty gate (0 = perceive every tick)")
    ap.add_argument("--scan-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES)
    ap.add_argument("--scan-voxel", type=float, default=DEFAULT_VOXEL_M)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--range-gap", type=float, default=DEFAULT_RANGE_GAP_M)
    ap.add_argument("--cluster-voxel", type=float, default=DEFAULT_CLUSTER_VOXEL_M)
    ap.add_argument("--image-lag", type=float, default=0.0)
    ap.add_argument("--sam-thresh", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()
    dump_scene(args.scene, base_url=args.base_url, floor=args.floor,
               novel_viewpoint_m=args.novel_viewpoint_m,
               scan_keyframes=args.scan_keyframes, scan_voxel_m=args.scan_voxel,
               min_inliers=args.min_inliers, range_gap_m=args.range_gap,
               cluster_voxel_m=args.cluster_voxel, image_lag_s=args.image_lag,
               sam_thresh=args.sam_thresh, request_timeout_s=args.timeout,
               out_root=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
