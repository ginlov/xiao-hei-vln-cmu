"""Precompute per-scene debug data for the Streamlit viz app.

Runs the perception pipeline over a scene's captures and dumps, per viewpoint:
the mask-overlay image, the detection/lift table, the cumulative ObjectMap nodes
(3D boxes), and the lifted points — plus the scene's GT boxes. The Streamlit app
(viz_app.py) then loads this without needing the sidecar.

Needs the sidecar up (localhost:8001).

    uv run --extra perception python perception_benchmark/dump_debug.py --scene livingroom_3
    uv run --extra perception python perception_benchmark/dump_debug.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viewgen import load_objects
from replay_score import load_capture, scoreable
from debug_viewpoint import _overlay_masks
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.geometry import (EQUIRECT_H, EQUIRECT_W,
    project_camera_points_to_equirect, sensor_to_camera_transform)
from xiao_hei_vln.perception.lifter import (DEFAULT_CLUSTER_VOXEL_M,
    DEFAULT_MIN_INLIERS, DEFAULT_RANGE_GAP_M, PointLifter,
    _rotation_from_quaternion)
from xiao_hei_vln.perception.object_map import NMS_DIST, NMS_GAP, ObjectMap
from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
DEBUG_DIR = Path("perception_benchmark/debug")
_R_SC, _T_SC = sensor_to_camera_transform()


def project_to_equirect(pts_map, pos, ori):
    """Project map-frame points into equirect pixels at this viewpoint's pose
    (same transform the lifter uses). Returns (u, v, in_fov, dist_m)."""
    t_ms = np.array([pos.x, pos.y, pos.z])
    xyz_sensor = (np.asarray(pts_map, float) - t_ms) @ _rotation_from_quaternion(ori)
    xyz_cam = xyz_sensor @ _R_SC.T + _T_SC
    u, v, in_fov = project_camera_points_to_equirect(xyz_cam)
    dist = np.linalg.norm(np.asarray(pts_map, float) - t_ms, axis=1)
    return u, v, in_fov, dist


def dump_scene(scene, *, base_url, score_threshold, min_inliers, accumulate,
               range_gap_m=DEFAULT_RANGE_GAP_M,
               cluster_voxel_m=DEFAULT_CLUSTER_VOXEL_M, out_root=None,
               max_pts=4000, request_timeout_s=60.0):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip"); return
    objs = load_objects(scene)
    keep = {i: e for i, e in objs.items() if scoreable(e.label)}
    classes = tuple(sorted({e.label for e in keep.values()}))
    gt = [{"id": i, "label": e.label,
           "center": [e.center.x, e.center.y, e.center.z],
           "bmin": [e.center.x - e.size.x/2, e.center.y - e.size.y/2, e.center.z - e.size.z/2],
           "bmax": [e.center.x + e.size.x/2, e.center.y + e.size.y/2, e.center.z + e.size.z/2]}
          for i, e in keep.items()]

    client = HTTPPerceptionClient(base_url=base_url, request_timeout_s=request_timeout_s)
    client.wait_until_ready(); client.set_classes(classes)
    lifter = PointLifter(min_inliers=min_inliers, range_gap_m=range_gap_m,
                         cluster_voxel_m=cluster_voxel_m)
    omap = ObjectMap()
    accum = ScanAccumulator() if accumulate else None

    out = (out_root or DEBUG_DIR) / scene
    out.mkdir(parents=True, exist_ok=True)
    vps = []
    for vp_dir in vp_dirs:
        vid = os.path.basename(vp_dir)
        img, scan, pos, ori = load_capture(Path(vp_dir))
        cloud = accum.update(scan, pos, ori) if accum is not None else scan
        dets = client.detect(img, score_threshold=score_threshold)
        recs, flags, pts = [], [], []
        # Second map fed ONLY this viewpoint's lifts, so the viewer can separate
        # what this frame contributed from what it inherited. Its node_ids are
        # local to the viewpoint and do NOT match the cumulative map's ids.
        vpmap = ObjectMap()
        for d in dets:
            res = lifter.lift(d.mask, cloud, pos, ori)
            ok = res.position is not None
            recs.append({"label": d.label, "score": round(float(d.score), 3),
                         "n_inliers": int(res.n_inliers), "lifted": ok,
                         "position": [round(float(v), 3) for v in
                                      (res.position.x, res.position.y, res.position.z)] if ok else None})
            flags.append(ok)
            if ok:
                pts.append(res.inlier_points); omap.add(d.label, d.score, res.inlier_points)
                vpmap.add(d.label, d.score, res.inlier_points)

        # overlay image: MODEL detections (masks + white labels) + GROUND-TRUTH
        # objects projected into the same image (lime diamonds + labels).
        overlay, anchors = _overlay_masks(img[:, :, ::-1], dets, flags)
        fig, ax = plt.subplots(figsize=(19, 6.6)); ax.imshow(overlay)
        ax.set_xlim(0, EQUIRECT_W); ax.set_ylim(EQUIRECT_H, 0); ax.axis("off")
        for u, v, txt, color in anchors:
            ax.text(u, v, txt, fontsize=7, color="white", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc=np.array(color)*0.7, ec="none", alpha=0.85))
        gc = np.array([g["center"] for g in gt])
        gu, gv, gfov, gd = project_to_equirect(gc, pos, ori)
        # observability filter: keep only GT whose location the scan actually
        # reached (nearest scan return within 0.8 m) — this drops behind-wall /
        # out-of-range GT the naive projection would otherwise clutter in.
        cxyz = cloud[:, :3]
        near = (np.array([np.min(np.linalg.norm(cxyz - c, axis=1)) for c in gc])
                if len(cxyz) else np.full(len(gc), 9e9))
        vis = gfov & (gv >= 0) & (gv < EQUIRECT_H) & (gd <= 8.0) & (near < 0.8)
        for j in np.where(vis)[0]:
            ax.scatter([gu[j]], [gv[j]], marker="D", s=80, facecolors="none",
                       edgecolors="lime", linewidths=2.2, zorder=6)
            ax.text(gu[j], gv[j] + 16, gt[j]["label"], fontsize=7, color="lime",
                    ha="center", va="top", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="lime", alpha=0.6))
        ax.set_title("MODEL DETECTIONS = coloured masks + white labels   |   "
                     f"GROUND TRUTH (observable from here) = lime ◇ "
                     f"({int(vis.sum())} of {len(gt)} GT)", fontsize=11)
        fig.savefig(out / f"{vid}.png", dpi=95, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

        # lifted points (downsampled) for the 3D lift view
        allpts = np.vstack(pts) if pts else np.empty((0, 3))
        if len(allpts) > max_pts:
            allpts = allpts[np.random.default_rng(0).choice(len(allpts), max_pts, False)]
        np.save(out / f"{vid}_pts.npy", allpts.astype(np.float32))

        def _summarise(omap_):
            return [{"node_id": n["node_id"], "label": n["label"], "score": n["score"],
                     "n_obs": n["n_obs"], "center": n["center_3d"],
                     "bmin": n["bbox_aabb"]["min"], "bmax": n["bbox_aabb"]["max"]}
                    for n in omap_.export()]

        nodes = _summarise(omap)                 # cumulative through this vp
        nodes_vp = _summarise(vpmap)             # this viewpoint alone
        vps.append({"id": vid, "pose": [pos.x, pos.y, pos.z],
                    "n_det": len(dets), "n_lift": int(sum(flags)),
                    "detections": recs, "nodes": nodes, "nodes_vp": nodes_vp})
        print(f"[{scene}] {vid}: {len(dets)} det, {int(sum(flags))} lift, "
              f"{len(nodes_vp)} nodes this vp, {len(nodes)} cumulative")

    json.dump({"scene": scene, "params": {"score_threshold": score_threshold,
               "min_inliers": min_inliers, "accumulate": accumulate,
               "range_gap_m": range_gap_m,
               "cluster_voxel_m": cluster_voxel_m,
               "nms_dist": NMS_DIST, "nms_gap": NMS_GAP},
               "gt": gt, "viewpoints": vps},
              open(out / "viz.json", "w"))
    print(f"[{scene}] wrote {out}/viz.json (+ {len(vps)} overlays & point files)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", "http://localhost:8001"))
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--no-accumulate", dest="accumulate", action="store_false")
    ap.set_defaults(accumulate=True)
    ap.add_argument("--range-gap", type=float, default=DEFAULT_RANGE_GAP_M,
                    help="range-cluster gap (m) for mask-spill rejection; 0 disables")
    ap.add_argument("--cluster-voxel", type=float, default=DEFAULT_CLUSTER_VOXEL_M,
                    help="voxel size (m) for lift clustering; 0 disables")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"dump root (default {DEBUG_DIR}); use a separate dir to A/B")
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
                   min_inliers=args.min_inliers, accumulate=args.accumulate,
                   range_gap_m=args.range_gap, cluster_voxel_m=args.cluster_voxel,
                   out_root=args.out,
                   request_timeout_s=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
