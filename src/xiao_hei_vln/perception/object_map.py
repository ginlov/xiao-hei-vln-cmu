"""Cross-frame object fusion for the perception scene graph.

The lifter turns one frame's detection into a 3D point + its inlier LiDAR
cloud, but the same physical object is seen many times: once per overlapping
detection within a frame, and again every time the robot re-observes it from a
new viewpoint. Picking one observation and discarding the rest — "same
label + close center → higher confidence wins" — keeps a single frame's median
and yields no 3D box at all.

``ObjectMap`` fuses instead of replaces:
  - a new observation merges into an existing same-label node whose 3D box
    overlaps (IoU) or whose center is close enough;
  - on merge the **point clouds are unioned** and the centroid + AABB are
    recomputed from the union, so the box CONVERGES to the true extent as
    views accumulate (filling the ``bbox_min``/``bbox_max`` the scene graph
    otherwise leaves ``None``);
  - a final suppression pass drops the weaker of two co-located nodes: by IoU
    for any label pair, and — for identical labels only — when their box
    surfaces nearly touch and their centres are close, which is what a
    fragmented object looks like once boxes are tight enough that IoU is 0;
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
MERGE_IOU = 0.3        # same-label nodes merge if box IoU exceeds this ...
MERGE_GAP_FRAC = 0.15  # ... or their surfaces are closer than this * max footprint size
FLAT_Z_M = 0.10        # a box this thin (m) in z is "flat" -> IoU falls back to 2D (BEV)
#
# The merge gate is `IoU >= MERGE_IOU OR surface_gap <= MERGE_GAP_FRAC * size`.
# Two design choices, both measured (TASK 31, arabic_room):
#   * IoU is 3D by default but falls back to the xy-footprint (BEV) IoU when
#     BOTH boxes are flat. Volumetric IoU collapses to 0 for a carpet (~0 m
#     thick), so the flat fragments a carpet shatters into could never satisfy
#     the IoU path; BEV IoU restores it without touching the volumetric path.
#   * The old distance gate (`centre_dist <= 0.4 m`) is replaced by a
#     surface-gap gate scaled by object size. A fixed centre radius merges two
#     genuinely distinct small same-label objects (two pillows 0.4 m apart) yet
#     is too tight for the loose, non-touching fragments of a large object. The
#     surface gap is ~0 for two views of one object and positive for two
#     objects; scaling the tolerance by the larger footprint lets a big carpet
#     absorb a nearby fragment while keeping small neighbours apart.
# `size` = max of the two footprint diagonals. FRAC is a starting value and
# should be swept end-to-end. An earlier size-scaled *distance* gate and a
# nested intersection-over-smaller rule were both rejected for merging distinct
# instances in dense scenes (chinese_room recall 0.500 -> 0.402); this gate is
# gap-based rather than centre-based precisely to avoid that, but the dense-scene
# regression is the thing to re-check when sweeping FRAC.
NMS_IOU = 0.5          # cross-label: suppress the weaker of two boxes above this
# IoU alone cannot catch cross-label duplicates once boxes are tight. Measured
# over 14 scenes, node pairs whose centres are within 0.5 m have median IoU
# 0.055 and NONE reach NMS_IOU — the rule never fires. Those pairs are not far
# apart, though: their box surfaces sit a median 3 cm apart, because a LiDAR
# sweep sees one face of an object and two viewpoints yield adjacent, disjoint
# slabs. So suppression also accepts "surfaces nearly touching AND centres
# close", which is what a fragmented object actually looks like.
# Restricted to IDENTICAL labels for now — across labels it would also
# collapse genuinely touching distinct objects and hide detector label
# instability. 0.0 on either disables the path. See docs/tasks/backlog.md B3.
# This is the deferred twin of the add()-time gate: `add` merges same-label
# nodes on the way in (IoU or surface-gap), but a node's centre and box MOVE as
# it accumulates points, so two nodes created apart can drift together with
# nothing re-checking. This is that re-check, deferred until they have settled.
# Measured over 14 scenes: 216 -> 204 redundant nodes, counting MAE
# 2.836 -> 2.813 (6 scenes better, 1 worse). Dropping to 0.3 m removes the
# effect entirely. The gap is not binding at this radius (0.05 and 0.15 give
# identical results) and is kept only as a guard against a large box whose
# centre happens to coincide with a small one.
NMS_DIST = 0.4         # centres within this (m) ...
NMS_GAP = 0.05         # ... and box surfaces no further apart than this (m)
PTS_CAP = 4000         # cap accumulated points per node (subsample beyond this)

MIN_LIDAR_3D = 5       # need >= this many inlier lidar pts for a 3D centre
LIDAR_GATE_M = 1.5     # drop pts >this far from the instance median before averaging

# Box estimation. Percentiles rather than min/max, because the node's cloud is
# the union of every observation ever merged into it and a single stray return
# would otherwise define a corner forever.
BOX_PCT = 2.0              # per-axis percentile for the box (2nd .. 98th)
CLUSTER_MAX_DIM_M = 3.0    # above this the cloud is probably two things ...
CLUSTER_VOXEL_M = 0.25     # ... so fall back to connected-component clustering

# Flat wall-decor classes whose box-prompted masks frequently grab the co-planar
# bare wall, lifting to a large, thin, vertical "sheet" that is a phantom rather
# than the object. We reject only such sheets, and only for these labels.
FLAT_LABELS = {"wall decal", "picture", "painting", "photo", "poster",
               "mirror", "wall art", "framed picture", "frame"}
WALL_MIN_EXTENT = 2.0      # a real picture/decal's long side is well under this (m)
WALL_MAX_THICK = 0.12      # essentially planar (m)
WALL_NORMAL_MAX_Z = 0.4    # plane normal ~horizontal => a vertical wall surface


def _is_structure(label: str) -> bool:
    """Architecture ("stuff") vs a real object instance.

    Imported lazily-ish here rather than duplicating the label set: the
    vocabulary module owns what counts as structure, this module only records
    the verdict on each node so consumers can filter.
    """
    from xiao_hei_vln.perception.vocab import is_structure

    return is_structure(label)


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


def _percentile_box(pts: np.ndarray):
    """AABB from per-axis percentiles instead of raw min/max.

    ``min``/``max`` are the least robust statistics there are: one stray
    return through a doorway sets a corner, and since the node's cloud is the
    union of every observation, one bad frame out of hundreds ruins the box
    permanently. Measured on livingroom_3, raw min/max gave a median size
    error of 6.8x and stretched one sofa to 58 m. Percentiles clip the tail
    while leaving a genuinely large object at its true extent.
    """
    if len(pts) < 8:                      # too few to have a meaningful tail
        return _aabb(pts)
    lo = np.percentile(pts, BOX_PCT, axis=0)
    hi = np.percentile(pts, 100.0 - BOX_PCT, axis=0)
    return lo, hi


def _voxel_largest_cluster(pts: np.ndarray, voxel_m: float) -> np.ndarray:
    """Keep the points of the largest 26-connected voxel component.

    The percentile box handles a thin tail of outliers, but not a node whose
    cloud is genuinely bimodal — a mask that straddled two rooms, or a merge
    that swallowed a second instance. There the outliers are a *cluster*, and
    trimming percentiles just shaves its edges. Connectivity separates them.
    """
    if len(pts) < 8:
        return pts

    keys = np.floor(pts / voxel_m).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(k): i for i, k in enumerate(uniq)}

    # Iterative flood fill over occupied voxels; 26-neighbourhood so a
    # diagonal contact still counts as one surface.
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) != (0, 0, 0)]
    comp = np.full(len(uniq), -1, dtype=np.int64)
    n_comp = 0
    for start in range(len(uniq)):
        if comp[start] != -1:
            continue
        stack = [start]
        comp[start] = n_comp
        while stack:
            cur = stack.pop()
            kx, ky, kz = uniq[cur]
            for dx, dy, dz in offsets:
                nb = index.get((kx + dx, ky + dy, kz + dz))
                if nb is not None and comp[nb] == -1:
                    comp[nb] = n_comp
                    stack.append(nb)
        n_comp += 1

    if n_comp <= 1:
        return pts
    point_comp = comp[inverse]
    counts = np.bincount(point_comp, minlength=n_comp)
    return pts[point_comp == int(np.argmax(counts))]


def _core_points(pts: np.ndarray) -> np.ndarray:
    """The subset of a node's cloud that plausibly belongs to one object.

    Two passes, cheapest first: a median-distance gate (the same
    ``LIDAR_GATE_M`` the centre already used), then — only if the result is
    still implausibly large for any indoor object — connected-component
    clustering to drop a whole second blob.
    """
    if len(pts) < MIN_LIDAR_3D:
        return pts
    med = np.median(pts, axis=0)
    gated = pts[np.linalg.norm(pts - med, axis=1) <= LIDAR_GATE_M]
    if len(gated) < MIN_LIDAR_3D:
        gated = pts

    lo, hi = _percentile_box(gated)
    if float(np.max(hi - lo)) <= CLUSTER_MAX_DIM_M:
        return gated
    clustered = _voxel_largest_cluster(gated, CLUSTER_VOXEL_M)
    return clustered if len(clustered) >= MIN_LIDAR_3D else gated


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


def box_gap(a_min, a_max, b_min, b_max) -> float:
    """Shortest distance between two AABB surfaces; 0.0 when they touch or
    overlap. Unlike IoU this stays informative for small boxes, where any
    non-overlap collapses IoU to exactly 0 with no gradient."""
    gap = np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)
    return float(np.linalg.norm(gap))


def iou_3d(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    va = np.prod(a_max - a_min); vb = np.prod(b_max - b_min)
    return float(inter / (va + vb - inter + 1e-9))


def iou_2d(a_min, a_max, b_min, b_max) -> float:
    """xy-footprint (bird's-eye) IoU — the flat-object fallback for ``iou_3d``.

    A carpet's box is ~0 m thick, so volumetric IoU is 0 even when two
    footprints fully overlap. Dropping z and comparing floor areas restores a
    meaningful overlap for planar objects."""
    lo = np.maximum(a_min[:2], b_min[:2])
    hi = np.minimum(a_max[:2], b_max[:2])
    inter = np.prod(np.clip(hi - lo, 0, None))
    if inter <= 0:
        return 0.0
    aa = np.prod(a_max[:2] - a_min[:2]); ab = np.prod(b_max[:2] - b_min[:2])
    return float(inter / (aa + ab - inter + 1e-9))


def _is_flat(bmin, bmax, thr: float = FLAT_Z_M) -> bool:
    """True when a box is thin enough in z to be treated as planar."""
    return float(bmax[2] - bmin[2]) <= thr


def _footprint_diag(bmin, bmax) -> float:
    """Diagonal of a box's xy footprint — the size scale for the gap gate."""
    return float(np.linalg.norm((np.asarray(bmax) - np.asarray(bmin))[:2]))


class _Node:
    __slots__ = ("node_id", "label", "score", "n_obs", "pts", "cmin", "cmax",
                 "center", "color_rgb", "color_name", "is_structure",
                 "obs_centers", "obs_extents", "obs_weights")

    def __init__(self, node_id, label, score, pts, color_rgb=None, color_name=None):
        self.node_id = node_id
        self.label = label
        self.is_structure = _is_structure(label)
        self.score = score
        self.n_obs = 1
        self.pts = pts
        self.color_rgb = color_rgb
        self.color_name = color_name
        self.obs_centers = []
        self.obs_extents = []
        self.obs_weights = []
        self._observe(pts)
        self._recompute()

    def _observe(self, pts):
        """Record one observation's own centre, extent, and point count.

        A single observation is already close to the right size — measured
        against ground truth its volume ratio is 0.95. What ruins the box is
        pooling the observations' points: each is offset from the true centre
        by ~0.23 m in a direction that depends on where the robot stood, so
        the union spans the object *plus* that scatter, and the volume comes
        out around 6x too big. Keeping the per-observation boxes lets the node
        average them instead of taking their union.

        The point count is kept per observation, not just summed, because it
        is the weight ``_recompute`` averages with: how much of the object a
        view saw is the natural measure of how much that view's box is worth.

        ``_core_points`` still runs, per observation: a mask that spans two
        surfaces has to be cut apart here, because an average over
        observations would faithfully return the bimodal box. The percentile
        trim does not — it exists to stop one bad frame out of hundreds from
        setting a corner of the pooled cloud, and averaging across
        observations already does that. Applying both shrank boxes to 0.66x of
        ground truth, trading one direction of error for the other.
        """
        core = _core_points(pts)
        lo, hi = _aabb(core)
        c, _ = robust_center(core)
        self.obs_centers.append(np.array(c) if c is not None else np.median(core, axis=0))
        self.obs_extents.append(hi - lo)
        self.obs_weights.append(max(len(core), 1))

    def _recompute(self):
        if len(self.pts) > PTS_CAP:                       # keep memory bounded
            self.pts = self.pts[np.random.choice(len(self.pts), PTS_CAP, False)]
        # Over the observations, not over the pooled cloud: the estimator that
        # does not accumulate each observation's centre error into the size.
        # Pooling was measured against this and is far worse — even when every
        # observation is filed under the right object by an oracle, one AABB
        # over the pooled points scores 0.138 of 2 against this estimator's
        # 0.443, because the extremes are set by the scatter rather than the
        # object.
        #
        # Weighted by how many points each observation contributed, rather
        # than a plain median: a view that saw 900 returns of the object knows
        # more about where it is than one that scraped 12 off its edge, and
        # unweighted the two count the same. Worth mIoU 0.203 -> 0.217 and
        # 37.9% -> 44.2% at IoU >= 0.25 over seven scenes, with recall and
        # precision unchanged — it is purely a better box. The gain is all in
        # near misses promoted to 1-pointers; the >= 0.5 rate does not move,
        # which is consistent with 2 points needing better point *selection*
        # rather than a better estimator over the same points.
        w = np.asarray(self.obs_weights, dtype=np.float64)
        w = w / w.sum()
        ext = np.array(self.obs_extents)
        ctrs = np.array(self.obs_centers)
        self.center = (ctrs * w[:, None]).sum(axis=0)
        # Extent estimator. The weighted MEAN of per-view boxes is right for
        # volumetric objects — it cancels the ~6x inflation of pooling near-face
        # LiDAR slabs. But it destroys FLAT objects: a carpet is coplanar, so
        # there is no inflation to cancel, and when the robot dwells at one pose
        # the many redundant partial views (each a thin slice) swamp the weighted
        # mean and the box shrinks below the object (measured: right carpet
        # width 1.44 -> 0.53, GT coverage 20%). For flat nodes we instead take
        # the MAX per-view extent in x/y — the reach of the single best view that
        # saw the whole object — which recovered coverage 20% -> 56% offline
        # (TASK 33). z stays the weighted mean (it is ~0 either way). `max` is
        # safe here because flat clouds are coplanar and _core_points already
        # rejects spill; a redundancy-aware gate + high-percentile is the more
        # robust eventual target (see the capture-time novelty gate, TASK 33).
        if np.median(ext[:, 2]) <= FLAT_Z_M:
            half_xy = ext[:, :2].max(axis=0) / 2.0
            half_z = float((ext[:, 2] * w).sum()) / 2.0
            half = np.array([half_xy[0], half_xy[1], half_z])
            # Flat CENTRE from the union midpoint of the per-view boxes, not the
            # weighted centroid. When navigation can only reach part of a large
            # carpet, every view's centroid clusters on the seen part, so the
            # centroid mean is biased toward it and — since the box is
            # centre ± max_half — the box is misplaced. The union midpoint
            # centres on the span the box actually covers, cancelling that bias
            # (measured, gate on: right carpet centre error 0.52 -> 0.24 m,
            # coverage 69 -> 82%; left 0.30 -> 0.12 m, 77 -> 84%). z keeps the
            # weighted centroid. xy only, so a spill-free coplanar cloud is
            # assumed — same regime the max-extent already relies on.
            lo_xy = (ctrs[:, :2] - ext[:, :2] / 2.0).min(axis=0)
            hi_xy = (ctrs[:, :2] + ext[:, :2] / 2.0).max(axis=0)
            self.center = np.array([
                (lo_xy[0] + hi_xy[0]) / 2.0,
                (lo_xy[1] + hi_xy[1]) / 2.0,
                self.center[2],
            ])
        else:
            half = (ext * w[:, None]).sum(axis=0) / 2.0
        self.cmin, self.cmax = self.center - half, self.center + half

    def merge(self, label, score, pts, color_rgb=None, color_name=None):
        self.pts = np.vstack([self.pts, pts])
        self.n_obs += 1
        self._observe(pts)
        if score >= self.score:                           # new best observation
            self.label = label                            # follow the stronger label
            self.is_structure = _is_structure(label)
            if color_rgb is not None:                     # keep colour in step
                self.color_rgb = color_rgb
                self.color_name = color_name
        self.score = max(self.score, score)
        self._recompute()


class ObjectMap:
    def __init__(self, merge_iou=MERGE_IOU, merge_gap_frac=MERGE_GAP_FRAC,
                 nms_iou=NMS_IOU, nms_dist=NMS_DIST, nms_gap=NMS_GAP):
        self.nodes: list[_Node] = []
        self.merge_iou, self.merge_gap_frac, self.nms_iou = merge_iou, merge_gap_frac, nms_iou
        self.nms_dist, self.nms_gap = nms_dist, nms_gap
        # node_id -> labels suppressed onto it, filled by finalize(). Kept off
        # the nodes themselves so export()'s throwaway view cannot leak into
        # the live map, which shares the same _Node objects.
        self._absorbed: dict[int, set[str]] = {}
        self._next_id = 0

    def add(self, label, score, pts, color_rgb=None, color_name=None):
        """Fuse one detection's cloud in; return the node id it landed in.

        The id lets a caller tie a 2D detection to the 3D node it became part
        of — the debug dumps use it to print the same number on the image
        overlay and the 3D box. ``None`` when the cloud was empty and nothing
        was recorded. Note a returned id can still be suppressed later by
        :meth:`finalize`, so it is a link, not a guarantee of survival.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return None
        pmin, pmax = _aabb(pts)
        p_flat = _is_flat(pmin, pmax)
        p_diag = _footprint_diag(pmin, pmax)
        best, best_key = None, 0.0
        for nd in self.nodes:
            if nd.label != label:
                continue
            # Flat-object fallback: BEV IoU when both boxes are planar, else 3D.
            if p_flat and _is_flat(nd.cmin, nd.cmax):
                iou = iou_2d(nd.cmin, nd.cmax, pmin, pmax)
            else:
                iou = iou_3d(nd.cmin, nd.cmax, pmin, pmax)
            gap = box_gap(nd.cmin, nd.cmax, pmin, pmax)
            gap_tol = self.merge_gap_frac * max(p_diag, _footprint_diag(nd.cmin, nd.cmax))
            if iou >= self.merge_iou or gap <= gap_tol:
                key = iou + 1.0 / (gap + 1e-3)             # prefer the closest/most-overlapping
                if key > best_key:
                    best, best_key = nd, key
        if best is not None:
            best.merge(label, score, pts, color_rgb, color_name)
            return best.node_id
        node = _Node(self._next_id, label, score, pts, color_rgb, color_name)
        self.nodes.append(node)
        self._next_id += 1
        return node.node_id

    def add_frame(self, objects: list[dict]):
        for o in objects:
            if "pts" in o and len(o["pts"]) > 0:
                self.add(o["label"], o["score"], np.asarray(o["pts"]),
                         o.get("color_rgb"), o.get("color_name"))

    def _suppresses(self, a: _Node, b: _Node) -> bool:
        """True when b is the same physical object as a and should be dropped.

        The gap/distance path is deliberately restricted to **identical
        labels**: it collapses fragments of one object that ``add`` could not
        merge because their tight boxes miss both ``merge_iou`` and the
        surface-gap gate. Applying it across labels would also collapse genuinely
        distinct touching objects (a pillow on a sofa) and would paper over
        detector label instability — that case is deferred; see backlog B3.
        """
        if iou_3d(a.cmin, a.cmax, b.cmin, b.cmax) >= self.nms_iou:
            return True                                  # legacy, any label
        if self.nms_dist <= 0.0 or self.nms_gap <= 0.0 or a.label != b.label:
            return False
        return (float(np.linalg.norm(a.center - b.center)) <= self.nms_dist
                and box_gap(a.cmin, a.cmax, b.cmin, b.cmax) <= self.nms_gap)

    def finalize(self):
        """Suppress the weaker of two co-located nodes.

        Two paths, see :meth:`_suppresses`: the legacy IoU rule (any label) and
        a gap/distance rule restricted to identical labels. Best-supported node
        wins — most observations, then score. When a cross-label suppression
        does fire, the loser's label is recorded on the winner rather than
        discarded, so detector label instability stays visible.
        """
        order = sorted(range(len(self.nodes)),
                       key=lambda i: (self.nodes[i].n_obs, self.nodes[i].score),
                       reverse=True)
        keep, dead = [], set()
        self._absorbed = {}
        for i in order:
            if i in dead:
                continue
            keep.append(self.nodes[i])
            for j in order:
                if j == i or j in dead:
                    continue
                a, b = self.nodes[i], self.nodes[j]
                if self._suppresses(a, b):
                    dead.add(j)
                    if b.label != a.label:
                        self._absorbed.setdefault(a.node_id, set()).add(b.label)
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

    def export(self, min_pts: int = 15, min_obs: int = 1):
        """Non-destructive snapshot: NMS + prune on a copy of the node LIST so a
        live map can be summarized each tick without losing accumulating nodes.
        Returns a list of dicts (see :meth:`to_list`)."""
        view = ObjectMap(self.merge_iou, self.merge_gap_frac, self.nms_iou,
                         self.nms_dist, self.nms_gap)
        view.nodes = list(self.nodes)              # shared node objs, separate list
        view.finalize().prune(min_obs=min_obs, min_pts=min_pts)
        return view.to_list()

    def to_list(self):
        out = []
        for nd in self.nodes:
            out.append({
                "node_id": int(nd.node_id),
                "label": nd.label, "score": round(float(nd.score), 4),
                "is_structure": bool(nd.is_structure),
                "n_obs": nd.n_obs, "n_pts": int(len(nd.pts)),
                "center_3d": [round(float(x), 4) for x in nd.center],
                "bbox_aabb": {"min": [round(float(x), 4) for x in nd.cmin],
                              "max": [round(float(x), 4) for x in nd.cmax],
                              "size": [round(float(x), 4) for x in (nd.cmax - nd.cmin)]},
                "color_rgb": list(nd.color_rgb) if nd.color_rgb is not None else None,
                "color_name": nd.color_name,
                "absorbed_labels": sorted(self._absorbed.get(nd.node_id, ())),
            })
        return sorted(out, key=lambda o: o["node_id"])
