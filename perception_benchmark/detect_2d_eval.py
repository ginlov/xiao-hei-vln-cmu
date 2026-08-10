"""Score the 2D detector alone — no lifting, no fusion.

Everything measured so far judged the detector through the 3D scene graph, so
lift and fusion errors were folded in. This projects each scene's GT object
centres into the equirect frame at every captured viewpoint and asks only of
the masks:

  localised   a detection's mask contains at least one visible GT centre
  labelled    ... and its label matches one of the GT objects it contains
  recall      a visible GT object is covered by some detection mask
  recall@lbl  ... by a correctly-labelled one

A GT object counts as visible when its centre projects inside the image and the
LiDAR sweep actually reached it (nearest return within `--near`), which is the
same observability test dump_debug.py uses for its overlays.

    uv run python perception_benchmark/detect_2d_eval.py --all
    uv run python perception_benchmark/detect_2d_eval.py --scene arabic_room
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import numpy as np
from replay_score import scoreable
from viewgen import load_objects

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)
from xiao_hei_vln.perception.lifter import _rotation_from_quaternion

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
_R_SC, _T_SC = sensor_to_camera_transform()


def project(points_map, pos: Vector3, ori: Quaternion):
    t = np.array([pos.x, pos.y, pos.z])
    cam = ((np.asarray(points_map, float) - t) @ _rotation_from_quaternion(ori)) @ _R_SC.T + _T_SC
    u, v, in_fov = project_camera_points_to_equirect(cam)
    return u, v, in_fov, np.linalg.norm(np.asarray(points_map, float) - t, axis=1)


def eval_scene(scene: str, *, near_m: float, max_range_m: float,
               record: list | None = None) -> collections.Counter:
    c = collections.Counter()
    gt = [e for e in load_objects(scene).values() if scoreable(e.label)]
    if not gt:
        return c
    centres = np.array([[e.center.x, e.center.y, e.center.z] for e in gt])
    labels = [e.label for e in gt]

    for vp_dir in sorted(glob.glob(str(CAP_DIR / scene / "vp_*"))):
        vd = Path(vp_dir)
        det_path = vd / "detections.npz"
        if not det_path.is_file():
            continue
        z = np.load(det_path, allow_pickle=True)
        masks, det_labels = z["masks"], [str(x) for x in z["labels"]]
        scores = [float(x) for x in z["scores"]]
        with open(vd / "pose.json") as fh:
            pose = json.load(fh)
        px, py, pz = pose["position"]
        qx, qy, qz, qw = pose["orientation_xyzw"]
        pos = Vector3(x=px, y=py, z=pz)
        ori = Quaternion(x=qx, y=qy, z=qz, w=qw)
        scan = np.load(vd / "registered_scan.npy")[:, :3]

        u, v, in_fov, dist = project(centres, pos, ori)
        # observability: the sweep actually reached the object's location
        near = (np.array([np.min(np.linalg.norm(scan - g, axis=1)) for g in centres])
                if len(scan) else np.full(len(centres), 9e9))
        vis = in_fov & (v >= 0) & (v < EQUIRECT_H) & (dist <= max_range_m) & (near < near_m)
        ui = np.clip(np.rint(u).astype(int) % EQUIRECT_W, 0, EQUIRECT_W - 1)
        vi = np.clip(np.rint(v).astype(int), 0, EQUIRECT_H - 1)

        covered, covered_lbl = set(), set()
        rows = []
        for m, dl, sc in zip(masks, det_labels, scores, strict=True):
            inside = [g for g in np.flatnonzero(vis) if m[vi[g], ui[g]]]
            c["detections"] += 1
            inside_labels = sorted({labels[g] for g in inside})
            correct = bool(inside) and dl in inside_labels
            if inside:
                c["localised"] += 1
                covered.update(inside)
                if correct:
                    c["labelled"] += 1
                    covered_lbl.update(g for g in inside if labels[g] == dl)
            rows.append({"label": dl, "score": round(sc, 3), "hit": bool(inside),
                         "inside": inside_labels, "correct": correct})
        c["gt_visible"] += int(vis.sum())
        c["gt_covered"] += len(covered)
        c["gt_covered_lbl"] += len(covered_lbl)
        if record is not None:
            visible = np.flatnonzero(vis)
            record.append({
                "id": vd.name, "detections": rows,
                "gt_visible": int(vis.sum()),
                "gt_covered": len(covered),
                "gt_covered_lbl": len(covered_lbl),
                "missed": sorted(labels[g] for g in visible if g not in covered),
                "missed_label_only": sorted(labels[g] for g in visible
                                            if g in covered and g not in covered_lbl),
            })
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--near", type=float, default=0.8, help="observability radius (m)")
    ap.add_argument("--max-range", type=float, default=8.0)
    ap.add_argument("--dump", type=Path, default=None,
                    help="also write <dir>/<scene>/detect2d.json for the viewer")
    args = ap.parse_args()

    scenes = (sorted(os.path.basename(os.path.dirname(m))
                     for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
              if args.all else [args.scene])
    if not scenes or scenes == [None]:
        ap.error("pass --scene <name> or --all")

    total = collections.Counter()
    print(f"{'scene':18s}{'dets':>7s}{'loc%':>7s}{'lbl%':>7s}{'gtVis':>7s}{'rec%':>7s}{'recL%':>7s}")
    for s in scenes:
        rec = [] if args.dump else None
        c = eval_scene(s, near_m=args.near, max_range_m=args.max_range, record=rec)
        if args.dump is not None and rec:
            out = args.dump / s
            out.mkdir(parents=True, exist_ok=True)
            with open(out / 'detect2d.json', 'w') as fh:
                json.dump({'scene': s,
                           'params': {'near_m': args.near, 'max_range_m': args.max_range},
                           'viewpoints': rec, 'totals': dict(c)}, fh)
        if not c["detections"]:
            continue
        total.update(c)
        print(f"{s:18s}{c['detections']:7d}"
              f"{100 * c['localised'] / c['detections']:7.1f}"
              f"{100 * c['labelled'] / c['detections']:7.1f}"
              f"{c['gt_visible']:7d}"
              f"{100 * c['gt_covered'] / max(c['gt_visible'], 1):7.1f}"
              f"{100 * c['gt_covered_lbl'] / max(c['gt_visible'], 1):7.1f}")
    if total["detections"]:
        print("-" * 52)
        print(f"{'TOTAL':18s}{total['detections']:7d}"
              f"{100 * total['localised'] / total['detections']:7.1f}"
              f"{100 * total['labelled'] / total['detections']:7.1f}"
              f"{total['gt_visible']:7d}"
              f"{100 * total['gt_covered'] / max(total['gt_visible'], 1):7.1f}"
              f"{100 * total['gt_covered_lbl'] / max(total['gt_visible'], 1):7.1f}")
        loc = total["localised"]
        print(f"\nof detections that hit a GT object, {100 * total['labelled'] / max(loc, 1):.1f}% "
              f"carried the right label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
