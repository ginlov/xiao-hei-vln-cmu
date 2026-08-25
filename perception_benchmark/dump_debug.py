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
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viewgen import load_objects
from replay_score import (
    _frozen_detections,
    capture_time,
    load_capture,
    scoreable,
)
from debug_viewpoint import _overlay_masks
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.deskew import PoseDeskew
from xiao_hei_vln.perception.geometry import (EQUIRECT_H, EQUIRECT_W,
    project_camera_points_to_equirect, sensor_to_camera_transform)
from xiao_hei_vln.perception.lifter import (DEFAULT_CLUSTER_VOXEL_M,
    DEFAULT_MIN_INLIERS, DEFAULT_RANGE_GAP_M, PointLifter,
    _rotation_from_quaternion)
from xiao_hei_vln.perception.object_map import NMS_DIST, NMS_GAP, ObjectMap
from xiao_hei_vln.perception.scan_accumulator import (
    DEFAULT_MAX_KEYFRAMES, DEFAULT_VOXEL_M, ScanAccumulator)

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


def _gt_world_aabb(e):
    """World-axis-aligned min/max of a GT object's floor footprint, WITH heading.

    ``e.size`` is expressed in the object's own frame; ``e.heading`` rotates it
    into the world. Ignoring the rotation (center ± size/2) draws a box that is
    mis-oriented whenever heading isn't a multiple of 90° aligned with the size
    axes — e.g. arabic_room's carpets carry a -90° heading, so their long axis
    (size.x) actually runs along world y. We rotate the four footprint corners
    and take their world AABB, which is what the viewer/video consume. z is not
    rotated (floor objects), so it stays center ± size.z/2.
    """
    hx, hy = e.size.x / 2.0, e.size.y / 2.0
    c, s = np.cos(e.heading), np.sin(e.heading)
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    rot = corners @ np.array([[c, s], [-s, c]])          # local -> world (Rz)
    world = rot + [e.center.x, e.center.y]
    lo, hi = world.min(0), world.max(0)
    zc, zh = e.center.z, e.size.z / 2.0
    return [float(lo[0]), float(lo[1]), zc - zh], [float(hi[0]), float(hi[1]), zc + zh]


def dump_scene(scene, *, base_url, score_threshold, min_inliers, accumulate,
               scan_keyframes=DEFAULT_MAX_KEYFRAMES, scan_voxel_m=DEFAULT_VOXEL_M,
               use_frozen=True, image_lag_s=0.0,
               range_cap_m=None, sam_thresh=0.0,
               range_gap_m=DEFAULT_RANGE_GAP_M,
               cluster_voxel_m=DEFAULT_CLUSTER_VOXEL_M, out_root=None,
               max_pts=4000, request_timeout_s=60.0, novel_viewpoint_m=0.0):
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip"); return
    objs = load_objects(scene)
    keep = {i: e for i, e in objs.items() if scoreable(e.label)}
    classes = tuple(sorted({e.label for e in keep.values()}))
    gt = []
    for i, e in keep.items():
        bmin, bmax = _gt_world_aabb(e)
        gt.append({"id": i, "label": e.label,
                   "center": [e.center.x, e.center.y, e.center.z],
                   "bmin": bmin, "bmax": bmax})

    # Skip the sidecar when every frame already has frozen masks — otherwise
    # wait_until_ready() blocks on a service this run never calls.
    all_frozen = use_frozen and all(
        (Path(d) / "detections.npz").is_file() for d in vp_dirs)
    client = None
    if not all_frozen:
        client = HTTPPerceptionClient(base_url=base_url,
                                      request_timeout_s=request_timeout_s)
        client.wait_until_ready()
        client.set_classes(classes)
    lifter = PointLifter(min_inliers=min_inliers, range_gap_m=range_gap_m,
                         cluster_voxel_m=cluster_voxel_m, max_depth_m=range_cap_m)
    omap = ObjectMap()
    # 0 keyframes means no accumulation: a zero-length window is meaningless to
    # the deque, and lifting the raw sweep is the sensible reading.
    if scan_keyframes <= 0:
        accumulate = False
    accum = (ScanAccumulator(max_keyframes=scan_keyframes, voxel_m=scan_voxel_m)
             if accumulate else None)

    # See TASK 27: the image trails the pose, so the lift (not the scan
    # accumulator) needs the pose de-rotated by lag x yaw_rate.
    deskew = PoseDeskew(image_lag_s)

    out = (out_root or DEBUG_DIR) / scene
    out.mkdir(parents=True, exist_ok=True)
    vps = []
    kept_xy: list[tuple[float, float]] = []      # viewpoint-novelty gate (TASK 33)
    for vp_dir in vp_dirs:
        vid = os.path.basename(vp_dir)
        img, scan, pos, ori = load_capture(Path(vp_dir))
        # Accumulator ingests EVERY tick; every tick still renders a frame. The
        # novelty gate only decides which ticks *trigger* perception + fusion —
        # a non-novel tick shows the (unchanged) cumulative map with no new
        # detections, the offline mirror of the live responder gate.
        cloud = accum.update(scan, pos, ori) if accum is not None else scan
        ori_lift = deskew.update(ori, capture_time(Path(vp_dir)))
        novel = True
        if novel_viewpoint_m > 0.0 and kept_xy:
            nearest = min((pos.x - kx) ** 2 + (pos.y - ky) ** 2
                          for kx, ky in kept_xy) ** 0.5
            novel = nearest > novel_viewpoint_m
        if novel and novel_viewpoint_m > 0.0:
            kept_xy.append((pos.x, pos.y))
        dets = (_frozen_detections(Path(vp_dir)) if use_frozen else None) if novel else []
        if dets is None:
            dets = client.detect(img, score_threshold=score_threshold)
        # Raise the YOLO score floor on frozen dets offline (they were dumped
        # at a lower floor) without re-running the sidecar. No-op on the live
        # path, which already applied score_threshold at detection time.
        dets = [d for d in dets if d.score >= score_threshold]
        if sam_thresh > 0:                            # B5 mask-quality gate
            dets = [d for d in dets if d.sam_score >= sam_thresh]
        recs, flags, pts, node_ids = [], [], [], []
        # This-viewpoint node id -> the cumulative ids its detections fused
        # into. The two maps number independently, so a per-viewpoint box can
        # only be labelled with a number the rest of the viewer recognises by
        # going through this.
        vp_to_cum: dict[int, set[int]] = {}
        # Second map fed ONLY this viewpoint's lifts, so the viewer can separate
        # what this frame contributed from what it inherited. Its node_ids are
        # local to the viewpoint and do NOT match the cumulative map's ids.
        vpmap = ObjectMap()
        for d in dets:
            res = lifter.lift(d.mask, cloud, pos, ori_lift)
            ok = res.position is not None
            rec = {"label": d.label, "score": round(float(d.score), 3),
                   "sam": round(float(d.sam_score), 3),
                   "n_inliers": int(res.n_inliers), "lifted": ok,
                   "position": [round(float(v), 3) for v in
                                (res.position.x, res.position.y, res.position.z)] if ok else None}
            flags.append(ok)
            nid = None
            if ok:
                pts.append(res.inlier_points)
                # The cumulative map's id — the number drawn on the overlay and
                # on the 3D box, so the two can be matched by eye.
                nid = omap.add(d.label, d.score, res.inlier_points)
                vid_local = vpmap.add(d.label, d.score, res.inlier_points)
                if vid_local is not None and nid is not None:
                    vp_to_cum.setdefault(vid_local, set()).add(nid)
            rec["node_id"] = nid
            node_ids.append(nid)
            recs.append(rec)

        # overlay image: MODEL detections (masks + white labels) + GROUND-TRUTH
        # objects projected into the same image (lime diamonds + labels).
        overlay, anchors = _overlay_masks(img[:, :, ::-1], dets, flags, node_ids)
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

        def _summarise(omap_, cum_map=None):
            out_ = []
            for n in omap_.export():
                item = {"node_id": n["node_id"], "label": n["label"],
                        "score": n["score"], "n_obs": n["n_obs"],
                        "center": n["center_3d"],
                        "bmin": n["bbox_aabb"]["min"], "bmax": n["bbox_aabb"]["max"]}
                if cum_map is not None:
                    # Ids from the cumulative map, so this box can be matched to
                    # the #N on the 2D overlay. Usually one; more than one means
                    # this frame's blob spans several cumulative nodes, which is
                    # itself worth seeing.
                    item["cum_ids"] = sorted(cum_map.get(n["node_id"], ()))
                out_.append(item)
            return out_

        nodes = _summarise(omap)                             # cumulative
        nodes_vp = _summarise(vpmap, vp_to_cum)              # this viewpoint
        # Heading is recorded alongside the position so the viewer can draw
        # which way the robot faced. It cannot be derived from the path: the
        # robot is stationary for most of a navigation capture, and a
        # zero-length step has no direction.
        yaw = math.atan2(2.0 * (ori.w * ori.z + ori.x * ori.y),
                         1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z))
        vps.append({"id": vid, "pose": [pos.x, pos.y, pos.z], "yaw": yaw,
                    "perceived": bool(novel),
                    "n_det": len(dets), "n_lift": int(sum(flags)),
                    "detections": recs, "nodes": nodes, "nodes_vp": nodes_vp})
        print(f"[{scene}] {vid}: {len(dets)} det, {int(sum(flags))} lift, "
              f"{len(nodes_vp)} nodes this vp, {len(nodes)} cumulative")

    json.dump({"scene": scene, "params": {"score_threshold": score_threshold,
               "min_inliers": min_inliers, "accumulate": accumulate,
               "scan_keyframes": scan_keyframes, "scan_voxel_m": scan_voxel_m,
               "range_gap_m": range_gap_m,
               "cluster_voxel_m": cluster_voxel_m,
               "novel_viewpoint_m": novel_viewpoint_m,
               "nms_dist": NMS_DIST, "nms_gap": NMS_GAP},
               "gt": gt, "viewpoints": vps},
              open(out / "viz.json", "w"))
    print(f"[{scene}] wrote {out}/viz.json (+ {len(vps)} overlays & point files)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", "http://localhost:8001"))
    ap.add_argument("--score-threshold", type=float, default=0.6)
    ap.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    ap.add_argument("--no-accumulate", dest="accumulate", action="store_false")
    ap.set_defaults(accumulate=True)
    ap.add_argument("--scan-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES,
                    help="ScanAccumulator window in ticks; 0 disables accumulation")
    ap.add_argument("--scan-voxel", type=float, default=DEFAULT_VOXEL_M,
                    help="ScanAccumulator downsample voxel (m)")
    ap.add_argument("--no-frozen", dest="use_frozen", action="store_false",
                    help="always call the sidecar, even where dump_detections.py "
                         "has written detections.npz next to the captures")
    ap.set_defaults(use_frozen=True)
    ap.add_argument("--range-gap", type=float, default=DEFAULT_RANGE_GAP_M,
                    help="range-cluster gap (m) for mask-spill rejection; 0 disables")
    ap.add_argument("--cluster-voxel", type=float, default=DEFAULT_CLUSTER_VOXEL_M,
                    help="voxel size (m) for lift clustering; 0 disables")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"dump root (default {DEBUG_DIR}); use a separate dir to A/B")
    ap.add_argument("--image-lag", type=float,
                    default=float(os.environ.get("XIAO_HEI_IMAGE_LAG_S", 0.0)),
                    help="seconds the image trails the pose (TASK 27); the "
                         "lift pose is de-rotated by lag x yaw_rate")
    ap.add_argument("--range-cap", type=float, default=None,
                    help="B4: drop scan returns farther than this (m) before lifting")
    ap.add_argument("--sam-thresh", type=float, default=0.0,
                    help="B5: drop detections with SAM mask-quality below this (0-1)")
    ap.add_argument("--novel-viewpoint-m", type=float, default=0.0,
                    help="TASK 33: viewpoint-novelty gate. Every tick still renders "
                         "a frame and feeds the accumulator, but perception + fusion "
                         "fire only when the pose is farther than this (m) from ALL "
                         "previously perceived viewpoints. 0 = off.")
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
                   scan_keyframes=args.scan_keyframes, scan_voxel_m=args.scan_voxel,
                   use_frozen=args.use_frozen, image_lag_s=args.image_lag,
                   range_cap_m=args.range_cap, sam_thresh=args.sam_thresh,
                   range_gap_m=args.range_gap, cluster_voxel_m=args.cluster_voxel,
                   out_root=args.out, novel_viewpoint_m=args.novel_viewpoint_m,
                   request_timeout_s=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
