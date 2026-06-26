#!/usr/bin/env python3
"""Cross-crop / cross-frame object merging (Phase 1, step 3).

lift3d turns one frame's detections into 3D object nodes, but the same physical
object appears multiple times: once per overlapping crop within a frame, and
again every time the robot re-observes it from a new viewpoint. This is the
`merge it with existing nodes if they represent the same physical instance` step
of the SysNav scene representation.

ObjectMap ingests lifted nodes (with their LiDAR points) and maintains a compact
set of persistent object nodes:
  - a new node merges into an existing one of the SAME label whose 3D box
    overlaps (IoU) or whose centre is close enough;
  - on merge the point clouds are unioned and the centroid + AABB are recomputed
    from the union, so the box CONVERGES to the true extent as views accumulate;
  - a final cross-class NMS drops near-duplicate boxes with conflicting labels
    (e.g. the same console detected as both "cabinet" and "shelf"), keeping the
    higher-evidence one.

    # single frame: collapses the cross-crop duplicates
    uv run python perception/objectmap.py --frames captures/run3/frames/000009 --source yolo
    # whole run: also merges across viewpoints
    uv run python perception/objectmap.py --frames "captures/run3/frames/*" --source yolo --out /tmp/map.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lift3d import dets_from_gt, lift_frame  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branchA_gt import LIDAR_GATE_M, robust_center  # noqa: E402

MERGE_IOU = 0.3        # same-label nodes merge if 3D IoU exceeds this ...
MERGE_DIST = 0.4       # ... or centres are within this many metres
NMS_IOU = 0.5          # cross-label: suppress the weaker of two boxes above this
PTS_CAP = 4000         # cap accumulated points per node (subsample beyond this)


def _aabb(pts: np.ndarray):
    return pts.min(0), pts.max(0)


def iou_3d(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(a_max - a_min); vb = np.prod(b_max - b_min)
    return float(inter / (va + vb - inter + 1e-9))


class _Node:
    __slots__ = ("label", "score", "n_obs", "pts", "cmin", "cmax", "center")

    def __init__(self, label, score, pts):
        self.label = label
        self.score = score
        self.n_obs = 1
        self.pts = pts
        self._recompute()

    def _recompute(self):
        if len(self.pts) > PTS_CAP:                       # keep memory bounded
            self.pts = self.pts[np.random.choice(len(self.pts), PTS_CAP, False)]
        self.cmin, self.cmax = _aabb(self.pts)
        c, _ = robust_center(self.pts)
        med = np.median(self.pts, axis=0)
        self.center = np.array(c) if c is not None else med

    def merge(self, label, score, pts):
        self.pts = np.vstack([self.pts, pts])
        self.score = max(self.score, score)
        self.n_obs += 1
        if score > self.score - 1e-9 and label != self.label:
            self.label = label                            # follow the stronger label
        self._recompute()


class ObjectMap:
    def __init__(self, merge_iou=MERGE_IOU, merge_dist=MERGE_DIST, nms_iou=NMS_IOU):
        self.nodes: list[_Node] = []
        self.merge_iou, self.merge_dist, self.nms_iou = merge_iou, merge_dist, nms_iou

    def add(self, label, score, pts):
        pmin, pmax = _aabb(pts)
        pc, _ = robust_center(pts)
        pcenter = np.array(pc) if pc is not None else np.median(pts, axis=0)
        best, best_key = None, 0.0
        for nd in self.nodes:
            if nd.label != label:
                continue
            iou = iou_3d(nd.cmin, nd.cmax, pmin, pmax)
            dist = float(np.linalg.norm(nd.center - pcenter))
            if iou >= self.merge_iou or dist <= self.merge_dist:
                key = iou + 1.0 / (dist + 1e-3)            # prefer the closest/most-overlapping
                if key > best_key:
                    best, best_key = nd, key
        if best is not None:
            best.merge(label, score, pts)
        else:
            self.nodes.append(_Node(label, score, pts))

    def add_frame(self, objects: list[dict]):
        for o in objects:
            if "pts" in o and len(o["pts"]) > 0:
                self.add(o["label"], o["score"], np.asarray(o["pts"]))

    def finalize(self):
        """Cross-label NMS: drop the weaker of two heavily-overlapping nodes."""
        order = sorted(range(len(self.nodes)),
                       key=lambda i: (self.nodes[i].n_obs, self.nodes[i].score),
                       reverse=True)
        keep, dead = [], set()
        for i in order:
            if i in dead:
                continue
            keep.append(self.nodes[i])
            for j in order:
                if j == i or j in dead:
                    continue
                a, b = self.nodes[i], self.nodes[j]
                if iou_3d(a.cmin, a.cmax, b.cmin, b.cmax) >= self.nms_iou:
                    dead.add(j)
        self.nodes = keep
        return self

    def prune(self, min_obs: int = 1, min_pts: int = 15):
        """Drop low-evidence nodes: a single-frame hit with very few LiDAR points
        is transient detector noise, not a trustworthy persistent object."""
        self.nodes = [nd for nd in self.nodes
                      if nd.n_obs > min_obs or len(nd.pts) >= min_pts]
        return self

    def export_nodes(self, min_pts: int = 15):
        """Non-destructive NMS+prune, returning the live _Node objects (with their
        accumulated points) -- for per-object point-cloud visualization."""
        view = ObjectMap(self.merge_iou, self.merge_dist, self.nms_iou)
        view.nodes = list(self.nodes)              # shared node objs, separate list
        view.finalize().prune(min_pts=min_pts)
        return view.nodes

    def export(self, min_pts: int = 15):
        """Non-destructive snapshot: NMS + prune on a copy of the node LIST so a
        live map can be summarized each cycle without losing accumulating nodes."""
        view = ObjectMap(self.merge_iou, self.merge_dist, self.nms_iou)
        view.nodes = list(self.nodes)              # shared node objs, separate list
        view.finalize().prune(min_pts=min_pts)
        return view.to_list()

    def to_list(self):
        out = []
        for nd in self.nodes:
            out.append({
                "label": nd.label, "score": round(float(nd.score), 4),
                "n_obs": nd.n_obs, "n_pts": int(len(nd.pts)),
                "center_3d": [round(float(x), 4) for x in nd.center],
                "bbox_aabb": {"min": [round(float(x), 4) for x in nd.cmin],
                              "max": [round(float(x), 4) for x in nd.cmax],
                              "size": [round(float(x), 4) for x in (nd.cmax - nd.cmin)]},
            })
        return sorted(out, key=lambda o: o["label"])


def _dets(frame: Path, source: str, names, device):
    if source == "gt":
        return dets_from_gt(frame)
    from detect import dets_from_yolo_sam
    return dets_from_yolo_sam(frame, names, device=device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True,
                    help="frame dir or glob, e.g. 'captures/run3/frames/*'")
    ap.add_argument("--source", choices=["gt", "yolo"], default="yolo")
    ap.add_argument("--names", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-pts", type=int, default=15,
                    help="prune single-frame nodes with fewer lidar points")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    frames = sorted(Path(p) for p in glob.glob(args.frames) if Path(p).is_dir()) \
        or [Path(args.frames)]
    names = None
    if args.source == "yolo":
        nj = args.names or (frames[0].parent.parent / "coco" / "annotations.json")
        names = [c["name"] for c in json.load(open(nj))["categories"]]

    omap = ObjectMap()
    total_raw = 0
    for fr in frames:
        dets = _dets(fr, args.source, names, args.device)
        res = lift_frame(fr, dets, keep_pts=True)
        total_raw += res["n_objects"]
        omap.add_frame(res["objects"])
        print(f"  {fr.name}: {len(dets)} dets -> {res['n_objects']} raw 3D "
              f"-> map now {len(omap.nodes)} nodes")
    omap.finalize().prune(min_pts=args.min_pts)

    objs = omap.to_list()
    print(f"\n{len(frames)} frame(s): {total_raw} raw 3D objects -> "
          f"{len(objs)} merged object nodes")
    print(f"  {'label':22s} {'#obs':>4s} {'#pts':>5s}  {'center (x,y,z)':>24s}   size")
    for o in objs:
        c, s = o["center_3d"], o["bbox_aabb"]["size"]
        print(f"  {o['label']:22s} {o['n_obs']:4d} {o['n_pts']:5d}  "
              f"({c[0]:6.2f},{c[1]:6.2f},{c[2]:6.2f})   "
              f"({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")
    if args.out:
        json.dump({"n_frames": len(frames), "objects": objs},
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
