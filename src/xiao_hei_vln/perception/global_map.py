"""Persistent 2D occupancy grid stitched from `terrain_ext` over time.

The challenge's ``/terrain_map_ext`` topic publishes a rolling 20 m
local map already expressed in the ``map`` frame. Combined with the
``/state_estimation`` pose, we can accumulate per-tick snapshots into a
single global grid — no SLAM, no extrinsics, no sensor fusion. The grid
is the substrate every later module (frontier planner, TSP coverage,
visibility checks) reads from.

Cells are one of three states:

* ``UNKNOWN`` (0) — never observed.
* ``FREE``    (1) — observed and judged traversable.
* ``OCCUPIED`` (2) — observed and judged non-traversable.

The origin of the grid is anchored to the robot's first observed pose
so we don't waste cells on far-away world coordinates; the grid spans
``±half_extent_m`` on each axis from the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from xiao_hei_vln.messages.sensors import OdomPose

UNKNOWN: int = 0
FREE: int = 1
OCCUPIED: int = 2


@dataclass(frozen=True)
class FrontierCluster:
    """A connected cluster of frontier cells, summarised for the planner."""

    centroid_xy: tuple[float, float]
    size: int
    # Representative cell index (i, j) closest to the centroid; useful for
    # downstream visibility checks. Optional — set to (-1, -1) if not used.
    cell_ij: tuple[int, int]


class GlobalMap:
    """Persistent 2D occupancy grid stitched across ticks.

    The grid is allocated lazily on the first ``update()`` so that its
    origin can be centred on wherever the robot actually spawns.
    """

    def __init__(
        self,
        half_extent_m: float = 30.0,
        resolution_m: float = 0.2,
        obstacle_cost_threshold: float = 0.5,
    ) -> None:
        if half_extent_m <= 0:
            raise ValueError(f"half_extent_m must be positive; got {half_extent_m}")
        if resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive; got {resolution_m}")
        self._half_extent = float(half_extent_m)
        self._res = float(resolution_m)
        self._obstacle_threshold = float(obstacle_cost_threshold)

        side = int(round(2 * self._half_extent / self._res))
        self._H = side
        self._W = side
        self._grid: np.ndarray = np.full((self._H, self._W), UNKNOWN, dtype=np.uint8)
        self._origin_xy: tuple[float, float] | None = None  # set on first update

    # ------------------------------------------------------------------ properties

    @property
    def grid(self) -> np.ndarray:
        """Read-only view of the underlying uint8 grid (H, W)."""
        return self._grid

    @property
    def resolution_m(self) -> float:
        return self._res

    @property
    def is_initialised(self) -> bool:
        return self._origin_xy is not None

    @property
    def origin_xy(self) -> tuple[float, float] | None:
        return self._origin_xy

    @property
    def shape(self) -> tuple[int, int]:
        return self._H, self._W

    # ------------------------------------------------------------------ coordinate helpers

    def world_to_idx(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x, y) → grid (i, j). May return out-of-bounds indices.

        ``i`` indexes rows (the y axis), ``j`` indexes columns (the x axis), to
        match the (H, W) shape convention.
        """
        if self._origin_xy is None:
            raise RuntimeError("GlobalMap not yet initialised; call update() first")
        ox, oy = self._origin_xy
        j = int((x - ox) / self._res)
        i = int((y - oy) / self._res)
        return i, j

    def idx_to_world(self, i: int, j: int) -> tuple[float, float]:
        """Cell centre in world coords."""
        if self._origin_xy is None:
            raise RuntimeError("GlobalMap not yet initialised; call update() first")
        ox, oy = self._origin_xy
        x = ox + (j + 0.5) * self._res
        y = oy + (i + 0.5) * self._res
        return x, y

    # ------------------------------------------------------------------ updates

    def update(self, terrain_points: np.ndarray, pose: OdomPose) -> None:
        """Stitch a single ``terrain_ext`` snapshot into the grid.

        ``terrain_points`` is an (N, 4) array of ``(x, y, z, cost)`` rows. The
        4th column is the traversability cost as published on
        ``/terrain_map_ext``; the threshold for "obstacle" is set at
        construction time and may need calibration against the actual
        simulator (see ``scripts/inspect_terrain.py``).
        """
        if terrain_points.size > 0 and (
            terrain_points.ndim != 2 or terrain_points.shape[1] != 4
        ):
            raise ValueError(
                f"terrain_points must be (N, 4); got {terrain_points.shape}",
            )

        # On the first call, anchor the grid origin to the robot's pose so the
        # grid is centred on wherever we spawn rather than on the world origin.
        if self._origin_xy is None:
            self._origin_xy = (
                pose.position.x - self._half_extent,
                pose.position.y - self._half_extent,
            )

        if terrain_points.shape[0] > 0:
            xs = terrain_points[:, 0]
            ys = terrain_points[:, 1]
            costs = terrain_points[:, 3]

            ox, oy = self._origin_xy
            js = ((xs - ox) / self._res).astype(np.int64)
            is_ = ((ys - oy) / self._res).astype(np.int64)

            # Drop points that fall outside the preallocated grid window.
            in_bounds = (
                (is_ >= 0)
                & (is_ < self._H)
                & (js >= 0)
                & (js < self._W)
            )
            is_ = is_[in_bounds]
            js = js[in_bounds]
            costs = costs[in_bounds]

            is_obstacle = costs > self._obstacle_threshold
            self._grid[is_[~is_obstacle], js[~is_obstacle]] = FREE
            self._grid[is_[is_obstacle], js[is_obstacle]] = OCCUPIED

        # The robot's own cell must be FREE — terrain_ext may not include it
        # because the sensor is mounted at the robot's centre.
        ri, rj = self.world_to_idx(pose.position.x, pose.position.y)
        if 0 <= ri < self._H and 0 <= rj < self._W:
            self._grid[ri, rj] = FREE

    def reset(self) -> None:
        """Wipe the grid back to all-UNKNOWN. Called at question change."""
        self._grid.fill(UNKNOWN)
        self._origin_xy = None

    # ------------------------------------------------------------------ frontier extraction

    def find_frontier_clusters(self, min_cluster_size: int = 4) -> list[FrontierCluster]:
        """Return clusters of frontier cells (FREE cells adjacent to UNKNOWN).

        A frontier cell is any ``FREE`` grid cell with at least one
        ``UNKNOWN`` 4-neighbour. We then group adjacent frontier cells via
        connected-component labelling and report each cluster's centroid in
        world coordinates plus its size in cells.
        """
        if not self.is_initialised:
            return []

        frontier_mask = _frontier_mask(self._grid)
        if not frontier_mask.any():
            return []

        labels = _label_4connected(frontier_mask)
        clusters: list[FrontierCluster] = []
        n_labels = int(labels.max())
        for lid in range(1, n_labels + 1):
            cells = np.argwhere(labels == lid)
            if cells.shape[0] < min_cluster_size:
                continue
            ci, cj = cells.mean(axis=0)
            # Pick the cluster cell nearest to the centroid as the representative
            # so it lies on FREE space (centroid_xy might fall on a non-cluster
            # cell for non-convex clusters).
            diffs = cells - np.array([ci, cj])
            idx = int(np.argmin((diffs ** 2).sum(axis=1)))
            rep_i, rep_j = int(cells[idx, 0]), int(cells[idx, 1])
            cx, cy = self.idx_to_world(rep_i, rep_j)
            clusters.append(
                FrontierCluster(
                    centroid_xy=(cx, cy),
                    size=int(cells.shape[0]),
                    cell_ij=(rep_i, rep_j),
                ),
            )
        return clusters

    def is_traversable(self, x: float, y: float) -> bool:
        if not self.is_initialised:
            return False
        i, j = self.world_to_idx(x, y)
        if not (0 <= i < self._H and 0 <= j < self._W):
            return False
        return bool(self._grid[i, j] == FREE)

    def compute_reachable_mask(self, from_x: float, from_y: float) -> np.ndarray:
        """4-connected BFS over FREE cells starting at ``(from_x, from_y)``.

        Returns a boolean (H, W) mask where True means the cell is FREE *and*
        connected to the seed via a path of FREE 4-neighbours. Cells
        immediately adjacent to FREE-but-not-yet-reachable terrain are
        included if their cluster is reached.

        The frontier planner uses this in place of straight-line line-of-sight
        so it can route the robot *around* furniture instead of through it.
        Cost: one BFS per tick, O(|FREE|) — typically a few hundred cells in
        a living-room-sized grid; trivial.
        """
        mask = np.zeros((self._H, self._W), dtype=bool)
        if not self.is_initialised:
            return mask

        si, sj = self.world_to_idx(from_x, from_y)
        if not (0 <= si < self._H and 0 <= sj < self._W):
            return mask
        if self._grid[si, sj] != FREE:
            # Seed isn't on FREE — still mark it so the robot's own cell shows up
            # as reachable for downstream lookups, but don't expand from here.
            mask[si, sj] = True
            return mask

        stack: list[tuple[int, int]] = [(si, sj)]
        mask[si, sj] = True
        while stack:
            ci, cj = stack.pop()
            for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                if (
                    0 <= ni < self._H
                    and 0 <= nj < self._W
                    and not mask[ni, nj]
                    and self._grid[ni, nj] == FREE
                ):
                    mask[ni, nj] = True
                    stack.append((ni, nj))
        return mask

    def line_passes_through(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        blocked_state: int = OCCUPIED,
    ) -> bool:
        """Cheap straight-line check: True if any cell on the segment is OCCUPIED.

        Used as a proxy for "can I walk in a straight line from A to B?". For
        non-convex obstacles this is conservative — replace with full A* if
        the planner starts picking unreachable frontiers in practice.
        """
        if not self.is_initialised:
            return False
        i0, j0 = self.world_to_idx(x0, y0)
        i1, j1 = self.world_to_idx(x1, y1)
        for i, j in _bresenham(i0, j0, i1, j1):
            if not (0 <= i < self._H and 0 <= j < self._W):
                return True  # segment leaves the known window — treat as blocked
            if int(self._grid[i, j]) == blocked_state:
                return True
        return False


# ----------------------------------------------------------------------- internals

def _frontier_mask(grid: np.ndarray) -> np.ndarray:
    """A FREE cell is a frontier iff at least one of its 4-neighbours is UNKNOWN."""
    free = grid == FREE
    unknown = grid == UNKNOWN
    # 4-connected dilation of UNKNOWN by shifting in each cardinal direction.
    # np.pad + slicing avoids wrapping at the borders.
    pad = np.pad(unknown, 1, mode="constant", constant_values=False)
    up = pad[:-2, 1:-1]
    down = pad[2:, 1:-1]
    left = pad[1:-1, :-2]
    right = pad[1:-1, 2:]
    unknown_dilated = unknown | up | down | left | right
    return free & unknown_dilated


def _label_4connected(mask: np.ndarray) -> np.ndarray:
    """Label connected components of a boolean mask using iterative BFS.

    Returns an (H, W) int32 array where 0 means "not in mask" and 1..N
    label distinct components. Avoids the scipy dependency; fast enough
    for the ~300×300 grids we use.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    # Visit in row-major order; for each unvisited mask cell, flood-fill.
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != 0:
                continue
            next_label += 1
            # BFS using a list as a queue (small components so overhead is fine)
            stack: list[tuple[int, int]] = [(i, j)]
            labels[i, j] = next_label
            while stack:
                ci, cj = stack.pop()
                for ni, nj in ((ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)):
                    if (
                        0 <= ni < h
                        and 0 <= nj < w
                        and mask[ni, nj]
                        and labels[ni, nj] == 0
                    ):
                        labels[ni, nj] = next_label
                        stack.append((ni, nj))
    return labels


def _bresenham(i0: int, j0: int, i1: int, j1: int) -> Iterable[tuple[int, int]]:
    """Yield every cell on the integer line segment between (i0, j0) and (i1, j1)."""
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i1 > i0 else -1
    sj = 1 if j1 > j0 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        yield i, j
        if i == i1 and j == j1:
            return
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj
