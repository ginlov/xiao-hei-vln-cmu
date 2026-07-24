"""2-D occupancy grid built incrementally from TerrainMap snapshots.

Cells are classified as FREE (traversable, cost ≤ threshold) or OCCUPIED
(high-cost obstacle). Cells never seen in any snapshot remain UNKNOWN.
Frontier cells are FREE cells that have at least one UNKNOWN neighbour.
"""

from __future__ import annotations

import math
from collections import deque

from xiao_hei_vln.messages.sensors import TerrainMap

# 4-connected neighbourhood offsets
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class OccupancyGrid:
    """Incrementally-updated 2-D grid in the map frame."""

    def __init__(self, resolution: float = 0.2) -> None:
        self._res = resolution
        self._free: set[tuple[int, int]] = set()
        self._occupied: set[tuple[int, int]] = set()
        # Cells explicitly suppressed via mark_occupied — terrain updates cannot re-free these.
        self._blacklisted: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Public update

    def update(self, terrain: TerrainMap, cost_threshold: float = 0.5) -> None:
        """Ingest one TerrainMap snapshot and update FREE / OCCUPIED sets."""
        pts = terrain.points  # (N, 4): x, y, z, cost
        for i in range(len(pts)):
            cell = self._to_grid(float(pts[i, 0]), float(pts[i, 1]))
            cost = float(pts[i, 3])
            if cost <= cost_threshold:
                if cell not in self._blacklisted:
                    self._free.add(cell)
                    self._occupied.discard(cell)
            else:
                # Only mark occupied if we haven't already confirmed it's free
                # from a previous snapshot with a better viewpoint.
                if cell not in self._free:
                    self._occupied.add(cell)

    # ------------------------------------------------------------------
    # Frontier extraction

    def frontier_cells(self) -> list[tuple[int, int]]:
        """Return FREE cells that have at least one UNKNOWN (unseen) neighbour."""
        frontiers: list[tuple[int, int]] = []
        known = self._free | self._occupied
        for cell in self._free:
            ix, iy = cell
            for dx, dy in _NEIGHBOURS:
                if (ix + dx, iy + dy) not in known:
                    frontiers.append(cell)
                    break
        return frontiers

    def unknown_region_size(
        self,
        cluster: list[tuple[int, int]],
        cap: int = 500,
        radius_cells: int = 15,
    ) -> int:
        """Local unseen area opened by ``cluster`` (information gain).

        Multi-source BFS through UNKNOWN neighbours, limited to
        ``radius_cells`` hops.
        """
        if not cluster or radius_cells <= 0:
            return 0
        known = self._free | self._occupied
        queue: deque[tuple[int, int]] = deque()
        dist: dict[tuple[int, int], int] = {}
        for ix, iy in cluster:
            for dx, dy in _NEIGHBOURS:
                nbr = (ix + dx, iy + dy)
                if nbr in known or nbr in dist:
                    continue
                dist[nbr] = 1
                queue.append(nbr)
        while queue and len(dist) < cap:
            cx, cy = queue.popleft()
            d = dist[(cx, cy)]
            if d >= radius_cells:
                continue
            for dx, dy in _NEIGHBOURS:
                nbr = (cx + dx, cy + dy)
                if nbr in known or nbr in dist:
                    continue
                dist[nbr] = d + 1
                queue.append(nbr)
                if len(dist) >= cap:
                    break
        return len(dist)

    def reachable_path_costs(self, x: float, y: float) -> dict[tuple[int, int], float]:
        """4-connected BFS distances (metres) over FREE cells from near ``(x, y)``."""
        if not self._free:
            return {}
        seed = self._seed_free_cell(x, y)
        if seed is None:
            return {}
        step = self._res
        costs: dict[tuple[int, int], float] = {seed: 0.0}
        queue: deque[tuple[int, int]] = deque([seed])
        while queue:
            cx, cy = queue.popleft()
            base = costs[(cx, cy)]
            for dx, dy in _NEIGHBOURS:
                nbr = (cx + dx, cy + dy)
                if nbr in self._free and nbr not in costs:
                    costs[nbr] = base + step
                    queue.append(nbr)
        return costs

    def _seed_free_cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Pick a FREE BFS seed at/near the robot pose."""
        origin = self._to_grid(x, y)
        if origin in self._free:
            return origin
        for radius in range(1, 9):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cand = (origin[0] + dx, origin[1] + dy)
                    if cand in self._free:
                        return cand
        return None

    def has_nav_clearance(self, cell: tuple[int, int], radius_cells: int = 1) -> bool:
        """True if the cell's immediate neighbourhood is not OCCUPIED (drivable)."""
        ix, iy = cell
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if (ix + dx, iy + dy) in self._occupied:
                    return False
        return True

    def wall_view_score(self, cell: tuple[int, int], *, inner: int = 2, outer: int = 6) -> int:
        """Occupied cells in an annulus — near walls but not colliding with them."""
        if not self.has_nav_clearance(cell, radius_cells=1):
            return 0
        ix, iy = cell
        n = 0
        for dx in range(-outer, outer + 1):
            for dy in range(-outer, outer + 1):
                if dx == 0 and dy == 0:
                    continue
                man = max(abs(dx), abs(dy))
                if man < inner:
                    continue
                if (ix + dx, iy + dy) in self._occupied:
                    n += 1
        return n

    def nearest_occupied_world(
        self,
        x: float,
        y: float,
        *,
        max_cells: int = 25,
    ) -> tuple[float, float] | None:
        """World XY of the nearest OCCUPIED cell (for look-at-wall headings)."""
        if not self._occupied:
            return None
        ox, oy = self._to_grid(x, y)
        best: tuple[int, int] | None = None
        best_d2 = None
        for ix, iy in self._occupied:
            if abs(ix - ox) > max_cells or abs(iy - oy) > max_cells:
                continue
            d2 = (ix - ox) ** 2 + (iy - oy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = (ix, iy)
        if best is None:
            return None
        return self.to_world(*best)

    # ------------------------------------------------------------------
    # Coordinate helpers

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        half = self._res * 0.5
        return (ix * self._res + half, iy * self._res + half)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return self._to_grid(x, y)

    # ------------------------------------------------------------------
    # Properties

    @property
    def free_cells(self) -> set[tuple[int, int]]:
        return self._free

    def mark_occupied(self, x: float, y: float, radius_cells: int = 1) -> None:
        """Mark a region around (x, y) as permanently occupied.

        Blacklisted cells are not restored by subsequent terrain updates, so
        skipped / visited frontier areas are not re-selected.
        """
        cx, cy = self._to_grid(x, y)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = (cx + dx, cy + dy)
                self._free.discard(cell)
                self._occupied.add(cell)
                self._blacklisted.add(cell)

    @property
    def resolution(self) -> float:
        return self._res

    # ------------------------------------------------------------------
    # Private

    def _to_grid(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self._res), math.floor(y / self._res))
