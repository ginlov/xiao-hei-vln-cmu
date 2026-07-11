"""Cross-frame object fusion for the perception scene graph.

The lifter turns one frame's detection into a 3D point + its inlier LiDAR
cloud, but the same physical object is seen many times: once per overlapping
detection within a frame, and again every time the robot re-observes it from a
new viewpoint. ``SceneRepresentation.add_object`` merges these only by
"same label + close center → higher confidence wins", keeping a single frame's
median and no 3D box.

``ObjectMap`` fuses instead of replaces:
  - a new observation merges into an existing same-label node whose 3D box
    overlaps (IoU) or whose center is close enough;
  - on merge the **point clouds are unioned** and the centroid + AABB are
    recomputed from the union, so the box CONVERGES to the true extent as
    views accumulate (filling the ``bbox_min``/``bbox_max`` the scene graph
    otherwise leaves ``None``);
  - a final cross-label NMS drops near-duplicate boxes with conflicting labels
    (e.g. the same console detected as both "cabinet" and "shelf");
  - flat wall-decor labels whose mask grabbed the co-planar bare wall (a large,
    thin, vertical "sheet") are rejected as phantoms.

Each node carries a stable ``node_id`` (monotonic, never reused) so the scene
graph can upsert observations across ticks by identity.

This is a self-contained port of our offline ObjectMap prototype
(``robust_center`` + the ``LIDAR_GATE_M`` gate are inlined so the package has no
dependency on ``dataset_generator``).
"""

from __future__ import annotations

import numpy as np

# ── merge / prune tuning (identical to the offline objectmap) ──────────────────
MERGE_IOU = 0.3        # same-label nodes merge if 3D IoU exceeds this ...
MERGE_DIST = 0.4       # ... or centres are within this many metres
NMS_IOU = 0.5          # cross-label: suppress the weaker of two boxes above this
PTS_CAP = 4000         # cap accumulated points per node (subsample beyond this)

MIN_LIDAR_3D = 5       # need >= this many inlier lidar pts for a 3D centre
LIDAR_GATE_M = 1.5     # drop pts >this far from the instance median before averaging

# Flat wall-decor classes whose box-prompted masks frequently grab the co-planar
# bare wall, lifting to a large, thin, vertical "sheet" that is a phantom rather
# than the object. We reject only such sheets, and only for these labels.
FLAT_LABELS = {"wall decal", "picture", "painting", "photo", "poster",
               "mirror", "wall art", "framed picture", "frame"}
WALL_MIN_EXTENT = 2.0      # a real picture/decal's long side is well under this (m)
WALL_MAX_THICK = 0.12      # essentially planar (m)
WALL_NORMAL_MAX_Z = 0.4    # plane normal ~horizontal => a vertical wall surface


def robust_center(pts: np.ndarray):
    """Median-gate outlier rejection then mean. Returns (centre|None, n_inliers).
    None when too few trustworthy points -> the instance is 2D-only."""
    if len(pts) < MIN_LIDAR_3D:
        return None, len(pts)
    med = np.median(pts, axis=0)
    inl = pts[np.linalg.norm(pts - med, axis=1) <= LIDAR_GATE_M]
    if len(inl) < MIN_LIDAR_3D:
        return None, len(inl)
    return inl.mean(0).tolist(), len(inl)


def _aabb(pts: np.ndarray):
    return pts.min(0), pts.max(0)


def _pca_extents(pts: np.ndarray):
    """Return (extent-per-PCA-axis, plane-normal). Extents are along the principal
    axes (ascending eigenvalue), so ext.min() is the planar thickness and the
    smallest-variance eigenvector is the surface normal."""
    X = pts - pts.mean(0)
    _, V = np.linalg.eigh(X.T @ X)        # columns = axes, ascending eigenvalue
    proj = X @ V
    return proj.max(0) - proj.min(0), V[:, 0]


def _is_wall_sheet(pts: np.ndarray, label: str) -> bool:
    """True iff a flat-label node is actually a chunk of bare wall: large, planar,
    and vertical. Real pictures/decals stay (their long side is short)."""
    if label not in FLAT_LABELS or len(pts) < 8:
        return False
    ext, normal = _pca_extents(pts)
    return (ext.max() > WALL_MIN_EXTENT and ext.min() < WALL_MAX_THICK
            and abs(normal[2]) < WALL_NORMAL_MAX_Z)


def iou_3d(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(a_max - a_min); vb = np.prod(b_max - b_min)
    return float(inter / (va + vb - inter + 1e-9))


class _Node:
    __slots__ = ("node_id", "label", "score", "n_obs", "pts", "cmin", "cmax",
                 "center", "color_rgb", "color_name")

    def __init__(self, node_id, label, score, pts, color_rgb=None, color_name=None):
        self.node_id = node_id
        self.label = label
        self.score = score
        self.n_obs = 1
        self.pts = pts
        self.color_rgb = color_rgb
        self.color_name = color_name
        self._recompute()

    def _recompute(self):
        if len(self.pts) > PTS_CAP:                       # keep memory bounded
            self.pts = self.pts[np.random.choice(len(self.pts), PTS_CAP, False)]
        self.cmin, self.cmax = _aabb(self.pts)
        c, _ = robust_center(self.pts)
        med = np.median(self.pts, axis=0)
        self.center = np.array(c) if c is not None else med

    def merge(self, label, score, pts, color_rgb=None, color_name=None):
        self.pts = np.vstack([self.pts, pts])
        self.n_obs += 1
        if score >= self.score:                           # new best observation
            self.label = label                            # follow the stronger label
            if color_rgb is not None:                     # keep colour in step
                self.color_rgb = color_rgb
                self.color_name = color_name
        self.score = max(self.score, score)
        self._recompute()


class ObjectMap:
    def __init__(self, merge_iou=MERGE_IOU, merge_dist=MERGE_DIST, nms_iou=NMS_IOU):
        self.nodes: list[_Node] = []
        self.merge_iou, self.merge_dist, self.nms_iou = merge_iou, merge_dist, nms_iou
        self._next_id = 0

    def add(self, label, score, pts, color_rgb=None, color_name=None):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
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
            best.merge(label, score, pts, color_rgb, color_name)
        else:
            self.nodes.append(_Node(self._next_id, label, score, pts,
                                    color_rgb, color_name))
            self._next_id += 1

    def add_frame(self, objects: list[dict]):
        for o in objects:
            if "pts" in o and len(o["pts"]) > 0:
                self.add(o["label"], o["score"], np.asarray(o["pts"]),
                         o.get("color_rgb"), o.get("color_name"))

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

    def prune(self, min_obs: int = 1, min_pts: int = 15, drop_wall_sheets: bool = True):
        """Drop low-evidence nodes (a single-frame hit with very few LiDAR points
        is transient detector noise) and, optionally, flat-label "wall sheet"
        phantoms (large planar vertical patches that are bare wall, not objects)."""
        self.nodes = [nd for nd in self.nodes
                      if (nd.n_obs > min_obs or len(nd.pts) >= min_pts)
                      and not (drop_wall_sheets and _is_wall_sheet(nd.pts, nd.label))]
        return self

    def export(self, min_pts: int = 15):
        """Non-destructive snapshot: NMS + prune on a copy of the node LIST so a
        live map can be summarized each tick without losing accumulating nodes.
        Returns a list of dicts (see :meth:`to_list`)."""
        view = ObjectMap(self.merge_iou, self.merge_dist, self.nms_iou)
        view.nodes = list(self.nodes)              # shared node objs, separate list
        view.finalize().prune(min_pts=min_pts)
        return view.to_list()

    def to_list(self):
        out = []
        for nd in self.nodes:
            out.append({
                "node_id": int(nd.node_id),
                "label": nd.label, "score": round(float(nd.score), 4),
                "n_obs": nd.n_obs, "n_pts": int(len(nd.pts)),
                "center_3d": [round(float(x), 4) for x in nd.center],
                "bbox_aabb": {"min": [round(float(x), 4) for x in nd.cmin],
                              "max": [round(float(x), 4) for x in nd.cmax],
                              "size": [round(float(x), 4) for x in (nd.cmax - nd.cmin)]},
                "color_rgb": list(nd.color_rgb) if nd.color_rgb is not None else None,
                "color_name": nd.color_name,
            })
        return sorted(out, key=lambda o: o["node_id"])
