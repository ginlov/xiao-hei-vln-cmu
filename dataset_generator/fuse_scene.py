#!/usr/bin/env python3
"""Fuse per-frame detections into a scene-level object inventory (TASK 10).

The same physical object is seen from many viewpoints (and, in the crop
representation, split across tiles). To count / localize it once, detections
must be associated into one scene entity. Two association keys:

  GT  (offline, sim):  the per-instance render COLOUR is a perfect, stable
       object id across all frames -> group by colour. Gives the ground-truth
       scene inventory + exact scene-level counts + a fused 3D position
       (pooled over every view's lidar). Colour is sim-only (test-forbidden).

  TEST (deployed):     no colour. Associate by CLASS + 3D PROXIMITY -- cluster
       per-frame instances of the same class whose map-frame 3D centres fall
       within `--thresh` metres. This is what the real system must do; the
       colour-GT above is its training/eval target.

This module builds both from a run's frames/*/detection_gt.json and reports how
well 3D+class clustering recovers the colour-GT inventory.

Usage:
    uv run --extra viz python dataset_generator/fuse_scene.py \
        --frames dataset_generator/captures/run1/frames --thresh 0.5
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# non-target classes already excluded upstream; keep this in sync with export_coco
from export_coco import DROP


def load_instances(frames_dir: Path):
    """All per-frame instances (one per visible colour per frame)."""
    out = []
    for fr in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        gtp = fr / "detection_gt.json"
        if not gtp.exists():
            continue
        for i in json.load(open(gtp))["instances"]:
            if i["raw_label"] in DROP:
                continue
            out.append({"frame": fr.name, **i})
    return out


def fuse_by_colour(insts):
    """GT fusion: group by instance colour -> scene objects."""
    groups = defaultdict(list)
    for i in insts:
        groups[tuple(i["color_rgb"])].append(i)
    objs = []
    for col, members in groups.items():
        pts = [(np.array(m["center_3d"]), m["n_lidar_pts"])
               for m in members if m["center_3d"]]
        if pts:
            w = np.array([n for _, n in pts], float) + 1e-6
            ctr = (np.stack([p for p, _ in pts]) * w[:, None]).sum(0) / w.sum()
            ctr = ctr.tolist()
        else:
            ctr = None
        objs.append({"color_rgb": list(col),
                     "raw_label": members[0]["raw_label"],
                     "n_views": len(members),
                     "center_3d": ctr,
                     "has_3d": ctr is not None,
                     "frames": sorted({m["frame"] for m in members})})
    return objs


def cluster_by_3d_class(insts, thresh):
    """TEST-style fusion: cluster same-class instances by 3D proximity (no colour)."""
    clusters = []  # each: {raw_label, centroid(np), members[]}
    for i in insts:
        if not i["center_3d"]:
            continue
        c = np.array(i["center_3d"])
        best = None
        bestd = thresh
        for k in clusters:
            if k["raw_label"] != i["raw_label"]:
                continue
            d = np.linalg.norm(k["centroid"] - c)
            if d < bestd:
                bestd, best = d, k
        if best is None:
            clusters.append({"raw_label": i["raw_label"], "centroid": c.copy(),
                             "members": [i]})
        else:
            best["members"].append(i)
            P = np.stack([np.array(m["center_3d"]) for m in best["members"]])
            best["centroid"] = P.mean(0)
    return clusters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="3D association radius (m) for test-style clustering")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    insts = load_instances(args.frames)
    gt = fuse_by_colour(insts)
    gt_3d = [o for o in gt if o["has_3d"]]
    clusters = cluster_by_3d_class(insts, args.thresh)

    print(f"per-frame instances (target vocab): {len(insts)}")
    print(f"GT scene objects (unique colours):  {len(gt)}  "
          f"({len(gt_3d)} with 3D, {len(gt)-len(gt_3d)} without)")
    print(f"3D+class clusters (test-style, thresh={args.thresh}m): {len(clusters)}")
    print(f"  -> vs {len(gt_3d)} colour-GT objects that HAVE 3D "
          f"(ideal: clusters == this)")

    # scene-level counts per class: GT(colour) vs clustered(3D+class)
    gt_cnt = defaultdict(int)
    for o in gt:
        gt_cnt[o["raw_label"]] += 1
    cl_cnt = defaultdict(int)
    for k in clusters:
        cl_cnt[k["raw_label"]] += 1
    print("\nscene counts  class: GT(colour) | clustered(3D+class) | GT-with-3D")
    gt3_cnt = defaultdict(int)
    for o in gt_3d:
        gt3_cnt[o["raw_label"]] += 1
    for cls in sorted(gt_cnt, key=lambda c: -gt_cnt[c]):
        flag = "" if gt_cnt[cls] == cl_cnt[cls] else "  <-- diff"
        print(f"  {cls:16s} {gt_cnt[cls]:2d} | {cl_cnt[cls]:2d} | {gt3_cnt[cls]:2d}{flag}")

    if args.out:
        json.dump({"gt_scene_objects": gt,
                   "scene_counts_gt": dict(gt_cnt),
                   "n_clusters_3d_class": len(clusters)},
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
