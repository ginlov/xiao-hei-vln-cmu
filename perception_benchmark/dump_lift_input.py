"""Precompute the 3D-lift INPUT per viewpoint for the Streamlit viewer.

The lifter's inputs are (mask, registered_scan, pose). This dumps, per viewpoint,
what the lift actually consumes and how a mask carves the scan:

  - scan          : the raw registered-scan cloud (downsampled) — the LiDAR input
  - per detection : the scan points its mask selects (in-mask, PRE z-buffer) and
                    the survivors AFTER the z-buffer occlusion gate (= the inliers
                    the lift medians). Counts too.

Runs sidecar-free: reads the frozen detections.npz masks (from dump_detections.py)
+ registered_scan.npy + pose.json. Writes debug/<scene>/vp_XXX_lift.npz.

    uv run python perception_benchmark/dump_lift_input.py --all
    uv run python perception_benchmark/dump_lift_input.py --scene livingroom_3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.lifter import DEFAULT_CLUSTER_VOXEL_M, PointLifter
from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
DEBUG_DIR = Path("perception_benchmark/debug")


def _pose(vp_dir: Path):
    p = json.load(open(vp_dir / "pose.json"))
    px, py, pz = p["position"]; qx, qy, qz, qw = p["orientation_xyzw"]
    return Vector3(x=px, y=py, z=pz), Quaternion(x=qx, y=qy, z=qz, w=qw)


def dump_scene(scene, *, min_inliers=10, cluster_voxel_m=DEFAULT_CLUSTER_VOXEL_M,
               out_root=None, cap_scan=6000, cap_det=500):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip"); return
    out = (out_root or DEBUG_DIR) / scene; out.mkdir(parents=True, exist_ok=True)
    # Three stages, so the viewer can attribute every dropped point:
    #   raw  = in-mask only          (no z-buffer, no clustering)
    #   zb   = + occlusion gate      (no clustering)
    #   kept = + voxel clustering    (what the lift actually medians)
    # The first two MUST pin cluster_voxel_m=0 or they stop meaning what
    # their labels say once clustering is on by default.
    lift_raw = PointLifter(min_inliers=1, enable_zbuffer=False, cluster_voxel_m=0.0)
    lift_zb = PointLifter(min_inliers=1, enable_zbuffer=True, cluster_voxel_m=0.0)
    lift_kept = PointLifter(min_inliers=1, enable_zbuffer=True,
                            cluster_voxel_m=cluster_voxel_m)
    rng = np.random.default_rng(0)
    # Must mirror replay_score.py: the scored pipeline lifts against the
    # ACCUMULATED cloud, so lifting a single sweep here would make the panel
    # systematically more pessimistic than the run it is meant to explain.
    accum = ScanAccumulator()

    for vp_dir in vp_dirs:
        vd = Path(vp_dir); vid = vd.name
        det_path = vd / "detections.npz"
        if not det_path.exists():
            print(f"[{scene}] {vid}: no detections.npz — run dump_detections.py first"); continue
        z = np.load(det_path, allow_pickle=True)
        masks, labels, scores = z["masks"], z["labels"], z["scores"]
        scan = np.load(vd / "registered_scan.npy")
        pos, ori = _pose(vd)
        cloud = accum.update(scan, pos, ori)

        scan_ds = cloud[:, :3]
        if len(scan_ds) > cap_scan:
            scan_ds = scan_ds[rng.choice(len(scan_ds), cap_scan, False)]

        inmask_pts, inmask_det, front_pts, front_det = [], [], [], []
        kept_pts, kept_det = [], []
        det_rows = []
        for i, m in enumerate(masks):
            raw = lift_raw.lift(m, cloud, pos, ori)       # in-mask (pre z-buffer)
            zb = lift_zb.lift(m, cloud, pos, ori)         # after z-buffer
            kept = lift_kept.lift(m, cloud, pos, ori)     # after clustering (final)
            n_raw = raw.n_inliers; n_front = zb.n_inliers; n_kept = kept.n_inliers
            det_rows.append({"idx": i, "label": str(labels[i]), "score": float(scores[i]),
                             "n_in_mask": int(n_raw), "n_front": int(n_front),
                             "n_kept": int(n_kept),
                             "lifted": bool(n_kept >= min_inliers),
                             "position": [round(float(v), 3) for v in
                                          (kept.position.x, kept.position.y, kept.position.z)]
                                         if kept.position is not None else None})
            for pts, P, D in ((raw.inlier_points, inmask_pts, inmask_det),
                              (zb.inlier_points, front_pts, front_det),
                              (kept.inlier_points, kept_pts, kept_det)):
                if pts is not None and len(pts):
                    p = pts if len(pts) <= cap_det else pts[rng.choice(len(pts), cap_det, False)]
                    P.append(p.astype(np.float32)); D.append(np.full(len(p), i, np.int16))

        np.savez_compressed(
            out / f"{vid}_lift.npz",
            scan=scan_ds.astype(np.float32),
            robot=np.array([pos.x, pos.y, pos.z], np.float32),
            inmask_pts=np.vstack(inmask_pts) if inmask_pts else np.zeros((0, 3), np.float32),
            inmask_det=np.concatenate(inmask_det) if inmask_det else np.zeros((0,), np.int16),
            front_pts=np.vstack(front_pts) if front_pts else np.zeros((0, 3), np.float32),
            front_det=np.concatenate(front_det) if front_det else np.zeros((0,), np.int16),
            kept_pts=np.vstack(kept_pts) if kept_pts else np.zeros((0, 3), np.float32),
            kept_det=np.concatenate(kept_det) if kept_det else np.zeros((0,), np.int16),
            dets=np.array(det_rows, dtype=object), min_inliers=min_inliers,
            cluster_voxel_m=cluster_voxel_m)
        nlift = sum(r["lifted"] for r in det_rows)
        print(f"[{scene}] {vid}: {len(masks)} masks, {nlift} lift, scan {len(scan_ds)} pts")
    print(f"[{scene}] wrote lift-input npz to {out}/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-inliers", type=int, default=10)
    ap.add_argument("--cluster-voxel", type=float, default=DEFAULT_CLUSTER_VOXEL_M)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.all:
        scenes = sorted(os.path.basename(os.path.dirname(m))
                        for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
    elif args.scene:
        scenes = [args.scene]
    else:
        ap.error("pass --scene <name> or --all")
    for s in scenes:
        dump_scene(s, min_inliers=args.min_inliers,
                   cluster_voxel_m=args.cluster_voxel, out_root=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
