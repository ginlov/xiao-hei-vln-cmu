#!/usr/bin/env python3
"""Lift 2D detections to 3D objects using LiDAR + pose (Phase 1 perception worker).

This is the geometry core of the real-robot perception stack (Goal B): given a
captured frame's {scan, pose} and a set of 2D detections expressed as
equirectangular masks, it assigns LiDAR points to each detection and produces a
3D centroid + 3D bounding box per object -- exactly the `reconstruct each
object's 3D point cloud from the predicted masks, LiDAR data, and robot pose`
step in the organizers' SysNav paper.

Detector-agnostic by design: a "detection" is just {label, score, mask(H,W bool)}.
  - feed GT-semantic masks (--source gt)  -> validates the geometry: the 3D
    centres must match branchA_gt's robust_center to the millimetre.
  - feed YOLO-World+SAM masks (later)      -> the real perception path.

Reuses branchA_gt's projection convention (equirect_uv) and the occlusion-aware
nearest-range (z-buffer) gate, so a far background point sharing a bearing with a
near object does NOT get pulled into that object's cloud.

    uv run python perception/lift3d.py --frame captures/run3/frames/000009 --source gt
    uv run python perception/lift3d.py --frame <dir> --source gt --obb --out /tmp/objs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# branchA_gt holds the single source of truth for the projection convention.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branchA_gt import (  # noqa: E402
    BG_CLASSES, EXTERIOR_CLASSES, LIDAR_GATE_M, MIN_CC, MIN_LIDAR_3D, ZBUF_TOL_M,
    clean_mask, equirect_uv, load_legend, robust_center, snap_palette,
    yaw_from_quat,
)


# --------------------------------------------------------------------------- #
# projection: do it once per frame, share across all detections
# --------------------------------------------------------------------------- #
def project_frame(frame_dir: Path, fov_v: float = 120.0):
    """Project the frame's LiDAR into the panorama. Returns a dict with the kept
    map-frame points and, for each point, its pixel (u,v), range, and a `front`
    flag (True iff it is the nearest surface along its bearing, within
    ZBUF_TOL_M). H,W come from the captured image, cam/yaw from the pose."""
    meta = json.load(open(frame_dir / "meta.json"))
    pos = meta["pose"]["position"]
    cam = np.array([pos["x"], pos["y"], pos["z"]])
    yaw = yaw_from_quat(meta["pose"]["orientation"])

    img = frame_dir / ("rgb.npy" if (frame_dir / "rgb.npy").exists() else "semantic.npy")
    H, W = np.load(img, mmap_mode="r").shape[:2]

    scan = np.load(frame_dir / "scan.npy")[:, :3]
    scan = scan[~(np.all(scan == 0, axis=1) | np.isnan(scan).any(axis=1))]
    u, v = equirect_uv(scan - cam, yaw, W, H, fov_v)
    inb = (v >= 0) & (v < H)
    rng = np.linalg.norm(scan - cam, axis=1)

    # z-buffer: per pixel keep the nearest range, then a point is "front" if it
    # is within ZBUF_TOL_M of its pixel's nearest range.
    fidx = np.clip(v, 0, H - 1) * W + u
    nearest = np.full(H * W, np.inf)
    np.minimum.at(nearest, fidx[inb], rng[inb])
    front = inb & (rng <= nearest[fidx] + ZBUF_TOL_M)

    return {"scan": scan, "cam": cam, "yaw": yaw, "u": u, "v": v,
            "rng": rng, "front": front, "H": H, "W": W, "pose": meta["pose"]}


# --------------------------------------------------------------------------- #
# lift one detection (mask -> 3D)
# --------------------------------------------------------------------------- #
def lift_detection(proj: dict, mask: np.ndarray, obb: bool = False,
                   keep_pts: bool = False):
    """Assign the frame's front LiDAR points that fall inside `mask` to this
    detection and return its 3D node, or None if too few inliers.

    mask: (H,W) bool in equirect pixel coords. center_3d uses branchA's
    robust_center (median-gate + mean); the 3D box is min/max (AABB) over the
    same gated inliers, or an oriented box (OBB) if obb=True and open3d is up."""
    H, W = proj["H"], proj["W"]
    v = np.clip(proj["v"], 0, H - 1)
    u = np.clip(proj["u"], 0, W - 1)
    sel = proj["front"] & mask[v, u]
    pts = proj["scan"][sel]

    center, n_inl = robust_center(pts)
    if center is None:
        return None  # < MIN_LIDAR_3D trustworthy points -> 2D-only, no 3D box

    med = np.median(pts, axis=0)
    inl = pts[np.linalg.norm(pts - med, axis=1) <= LIDAR_GATE_M]

    node = {"center_3d": [round(float(c), 4) for c in center],
            "n_lidar_pts": int(sel.sum()), "n_inliers": int(len(inl)),
            "dist_m": round(float(np.linalg.norm(np.array(center) - proj["cam"])), 3)}

    amin, amax = inl.min(0), inl.max(0)
    node["bbox_aabb"] = {"min": [round(float(x), 4) for x in amin],
                         "max": [round(float(x), 4) for x in amax],
                         "size": [round(float(x), 4) for x in (amax - amin)]}
    if obb:
        node["bbox_obb"] = _obb(inl)
    if keep_pts:
        node["pts"] = inl
    return node


def _obb(pts: np.ndarray):
    """Oriented bounding box via open3d (yaw-only would need PCA on xy; here we
    let open3d fit a full 3D OBB). Returns center/extent/R, or None if degenerate."""
    try:
        import open3d as o3d
    except ImportError:
        return None
    if len(pts) < 4:
        return None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    try:
        box = pc.get_oriented_bounding_box()
    except RuntimeError:
        return None  # coplanar / collinear points
    return {"center": [round(float(x), 4) for x in box.center],
            "extent": [round(float(x), 4) for x in box.extent],
            "R": [[round(float(x), 5) for x in row] for row in np.asarray(box.R)]}


# --------------------------------------------------------------------------- #
# detection source: GT semantic palette (for geometry validation)
# --------------------------------------------------------------------------- #
def dets_from_gt(frame_dir: Path, min_cc: int = MIN_CC):
    """Build per-instance equirect masks from the sim semantic GT, so lift3d can
    be validated against branchA_gt with no detector. Mirrors branchA's filtering
    (palette snap, drop background/exterior, connected-component denoise)."""
    S = snap_palette(np.load(frame_dir / "semantic.npy"))  # rgb
    legend = load_legend(frame_dir)
    colours = np.unique(S.reshape(-1, 3), axis=0)
    dets = []
    for col in colours:
        key = tuple(int(c) for c in col)
        name, cleaned, _ = legend.get(key, (None, "", ""))
        if name is None or cleaned in BG_CLASSES or cleaned in EXTERIOR_CLASSES:
            continue
        mask = np.all(S == col, axis=2)
        mask, _ = clean_mask(mask, min_cc)
        if mask.sum() < 40:
            continue
        dets.append({"label": cleaned, "instance": name, "score": 1.0, "mask": mask})
    return dets


# --------------------------------------------------------------------------- #
def lift_frame(frame_dir: Path, dets: list[dict], fov_v: float = 120.0,
               obb: bool = False, keep_pts: bool = False):
    """Lift every detection in `dets` to a 3D object node for one frame."""
    proj = project_frame(frame_dir, fov_v)
    objects = []
    for d in dets:
        node = lift_detection(proj, d["mask"], obb=obb, keep_pts=keep_pts)
        if node is None:
            continue
        node = {"label": d["label"], "score": round(float(d["score"]), 4),
                **({"instance": d["instance"]} if "instance" in d else {}), **node}
        objects.append(node)
    return {"frame": frame_dir.name, "pose": proj["pose"],
            "n_lidar": int(len(proj["scan"])), "n_objects": len(objects),
            "objects": objects}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--source", choices=["gt", "yolo"], default="gt",
                    help="gt = sim semantic palette; yolo = YOLO-World v2 + SAM2.1")
    ap.add_argument("--names", type=Path, default=None,
                    help="COCO annotations.json to read the class vocabulary from "
                         "(default: <frame>/../../coco/annotations.json)")
    ap.add_argument("--no-sam", action="store_true",
                    help="yolo source: skip SAM, use YOLO box rectangles as masks")
    ap.add_argument("--device", default=None, help="torch device (e.g. mps, cuda, cpu)")
    ap.add_argument("--fov-v", type=float, default=120.0)
    ap.add_argument("--obb", action="store_true", help="also fit oriented 3D box")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.source == "gt":
        dets = dets_from_gt(args.frame)
    else:
        import json as _json
        from detect import dets_from_yolo_sam
        names_json = args.names or (args.frame.parent.parent / "coco" / "annotations.json")
        names = [c["name"] for c in _json.load(open(names_json))["categories"]]
        dets = dets_from_yolo_sam(args.frame, names, pano_fov_v=args.fov_v,
                                  use_sam=not args.no_sam, device=args.device)
    res = lift_frame(args.frame, dets, fov_v=args.fov_v, obb=args.obb)

    print(f"{res['frame']}: {res['n_lidar']} lidar pts, "
          f"{len(dets)} dets -> {res['n_objects']} 3D objects")
    print(f"  {'label':22s} {'#pts':>5s} {'dist':>6s}  "
          f"{'center (x,y,z)':>24s}   box size (x,y,z)")
    for o in sorted(res["objects"], key=lambda o: o["dist_m"]):
        c = o["center_3d"]; s = o["bbox_aabb"]["size"]
        print(f"  {o['label']:22s} {o['n_lidar_pts']:5d} {o['dist_m']:6.2f}  "
              f"({c[0]:6.2f},{c[1]:6.2f},{c[2]:6.2f})   "
              f"({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
