"""Candidate inlier-filter strategies for the 3D lift (benchmark only).

The z-buffer gate in ``PointLifter`` rejects returns occluded *along a bearing*.
It cannot reject a co-visible neighbouring surface the detection mask spilled
onto — ceiling around a ceiling-mounted lamp, wall around a picture — because
for those pixels the neighbour is itself the nearest return. When the spill
outnumbers the object, the median tracks the contaminant and the lifted box
blows up (see docs/tasks/backlog.md, B1).

Each factory here returns an ``InlierFilter``: given the map-frame points that
survived mask + z-buffer, their camera-frame ranges, and the inlier floor, it
returns a boolean mask of points to keep. They live in the benchmark package,
not in ``xiao_hei_vln``, so strategies can be A/B'd on frozen captures without
committing production code to any of them.

**Selection rule is held constant across strategies** (nearest cluster with at
least ``min_inliers`` points, else the largest) so the comparison measures the
clustering, not the tie-break.

    uv run --extra perception python perception_benchmark/replay_score.py \
        --all --cluster voxel
"""
from __future__ import annotations

import numpy as np

# Beyond this, the O(n^2) strategies would allocate an uncomfortable distance
# matrix. Measured max over all 15 scenes is 2952 points per detection, so this
# never trips today — it is a guard, not a tuning knob.
PAIRWISE_CAP = 6000


# ── shared selection rule ─────────────────────────────────────────────────────

def _select(pts, depth, labels, min_inliers):
    """Keep the NEAREST cluster with >= min_inliers points, else the largest.

    The camera cannot see through the object, so under its own mask the object
    is the nearest substantial return. ``labels`` is an int array with -1 for
    noise (DBSCAN); noise is never selected but may be re-absorbed by nothing.
    """
    keep = np.zeros(len(pts), dtype=bool)
    ids = [c for c in np.unique(labels) if c >= 0]
    if not ids:
        return ~keep if labels.size == 0 else keep
    groups = [(c, np.flatnonzero(labels == c)) for c in ids]
    # nearest first, by the closest point in each cluster
    groups.sort(key=lambda g: depth[g[1]].min())
    chosen = next((g[1] for g in groups if len(g[1]) >= min_inliers),
                  max(groups, key=lambda g: len(g[1]))[1])
    keep[chosen] = True
    return keep


def _union_find(n, pairs):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
    return np.array([find(i) for i in range(n)])


# ── strategies ────────────────────────────────────────────────────────────────

def range_gap(gap_m: float = 0.3):
    """1-D split on gaps in range. Cheap, but blind to a plane viewed at a
    grazing angle, whose range recedes continuously with no gap to cut on."""
    def f(pts, depth, min_inliers):
        order = np.argsort(depth)
        labels = np.empty(len(pts), dtype=int)
        cuts = np.flatnonzero(np.diff(depth[order]) > gap_m) + 1
        for cid, chunk in enumerate(np.split(order, cuts)):
            labels[chunk] = cid
        return _select(pts, depth, labels, min_inliers)
    return f


def single_link(eps_m: float = 0.25):
    """Connected components of the eps-radius graph (DBSCAN with min_samples=1).
    Separates in 3D rather than along the ray, but chains through any thin
    bridge of points between two blobs."""
    def f(pts, depth, min_inliers):
        n = len(pts)
        if n > PAIRWISE_CAP:
            return np.ones(n, dtype=bool)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        labels = _union_find(n, zip(*np.where(np.triu(d <= eps_m, 1)), strict=True))
        return _select(pts, depth, labels, min_inliers)
    return f


def dbscan(eps_m: float = 0.25, min_samples: int = 4):
    """DBSCAN. The core-point requirement is the anti-chaining mechanism: a
    sparse bridge no longer connects two dense blobs. Non-core points join a
    neighbouring cluster; the rest are noise (-1) and never selected."""
    def f(pts, depth, min_inliers):
        n = len(pts)
        if n > PAIRWISE_CAP:
            return np.ones(n, dtype=bool)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        neigh = d <= eps_m
        core = neigh.sum(1) >= min_samples
        if not core.any():
            return np.ones(n, dtype=bool)
        # link core points to each other only — this is what stops chaining
        core_pairs = np.triu(neigh & core[:, None] & core[None, :], 1)
        labels = _union_find(n, zip(*np.where(core_pairs), strict=True))
        labels = np.where(core, labels, -1)
        # attach border points to any core neighbour
        for i in np.flatnonzero(~core):
            cn = np.flatnonzero(neigh[i] & core)
            if len(cn):
                labels[i] = labels[cn[np.argmin(d[i, cn])]]
        return _select(pts, depth, labels, min_inliers)
    return f


def voxel_cc(voxel_m: float = 0.10):
    """Connected components on a voxel grid (26-connectivity), PCL-style
    Euclidean cluster extraction. O(n) with no distance matrix, and density is
    normalised by construction — the cloud is already voxel-downsampled at 5 cm
    upstream, so far-away sparsity hurts less than a fixed metric eps."""
    offs = np.array([(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)])

    def f(pts, depth, min_inliers):
        keys = np.floor(pts / voxel_m).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        index = {tuple(k): i for i, k in enumerate(uniq)}
        vlab = np.full(len(uniq), -1, dtype=int)
        cid = 0
        for start in range(len(uniq)):
            if vlab[start] >= 0:
                continue
            stack, vlab[start] = [start], cid
            while stack:                       # flood fill
                cur = uniq[stack.pop()]
                for nb in map(tuple, cur + offs):
                    j = index.get(nb)
                    if j is not None and vlab[j] < 0:
                        vlab[j] = cid
                        stack.append(j)
            cid += 1
        return _select(pts, depth, vlab[inv], min_inliers)
    return f


def plane_voxel_cc(voxel_m: float = 0.10, plane_tol_m: float = 0.06,
                   min_plane_frac: float = 0.35, min_plane_extent_m: float = 1.0,
                   iters: int = 60, seed: int = 0):
    """RANSAC plane removal, then voxel connected components.

    Targets the physics rather than hoping for spatial separation: the
    contaminant is always a large planar surface (ceiling, wall, floor). If the
    dominant plane holds a large share of the points AND spans more than any
    plausible object, its inliers are dropped before clustering. This is the
    only strategy here that can help when the object is mounted *flush* against
    the surface, where no clustering can separate them.
    """
    base = voxel_cc(voxel_m)

    def f(pts, depth, min_inliers):
        n = len(pts)
        if n >= 3 * min_inliers:
            rng = np.random.default_rng(seed)
            best_in, best_n = None, 0
            for _ in range(iters):
                s = pts[rng.choice(n, 3, replace=False)]
                nrm = np.cross(s[1] - s[0], s[2] - s[0])
                mag = np.linalg.norm(nrm)
                if mag < 1e-9:
                    continue
                nrm = nrm / mag
                inl = np.abs((pts - s[0]) @ nrm) <= plane_tol_m
                if inl.sum() > best_n:
                    best_in, best_n = inl, int(inl.sum())
            if best_in is not None and best_n / n >= min_plane_frac:
                ext = pts[best_in].max(0) - pts[best_in].min(0)
                rest = ~best_in
                # only drop it if it is bigger than an object could be, and
                # enough survives to still lift something
                if np.sort(ext)[-1] >= min_plane_extent_m and rest.sum() >= min_inliers:
                    sub = base(pts[rest], depth[rest], min_inliers)
                    keep = np.zeros(n, dtype=bool)
                    keep[np.flatnonzero(rest)[sub]] = True
                    return keep
        return base(pts, depth, min_inliers)
    return f


STRATEGIES = {
    "none": lambda: None,
    "range": range_gap,
    "single": single_link,
    "dbscan": dbscan,
    "voxel": voxel_cc,
    "plane": plane_voxel_cc,
}
