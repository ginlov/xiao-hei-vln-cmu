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
        ``radius_cells`` hops. A doorway into a large empty chamber fills
        many cells within that radius; a one-cell cavity scores ~1. A hop
        limit avoids treating the whole outdoor UNKNOWN ocean as equal gain
        for every perimeter frontier.
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
    def reachable_free_cells(self, x: float, y: float) -> set[tuple[int, int]]:
        """FREE cells connected to ``(x, y)`` via 4-connected FREE paths."""
        return set(self.reachable_path_costs(x, y))

    def reachable_path_costs(self, x: float, y: float) -> dict[tuple[int, int], float]:
        """4-connected BFS distances (metres) over FREE cells from near ``(x, y)``.

        Cost is grid hop count × resolution — a proxy for path length that
        routes around OCCUPIED / UNKNOWN instead of cutting through walls.
        If the robot cell is not FREE yet, seeds from the nearest FREE cell
        within a small search radius.
        """
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
        # Robot may sit on an unseen/blacklisted cell briefly — search nearby.
        for radius in range(1, 9):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cand = (origin[0] + dx, origin[1] + dy)
                    if cand in self._free:
                        return cand
        return None

    # ------------------------------------------------------------------
    # Coordinate helpers

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        half = self._res * 0.5
        return (ix * self._res + half, iy * self._res + half)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return self._to_grid(x, y)

    def is_clear_for_goal(self, ix: int, iy: int, clearance_cells: int = 1) -> bool:
        """True if ``(ix, iy)`` is FREE and no OCCUPIED cell lies within clearance.

        Used to keep published waypoints off walls / furniture so the local
        planner's inflated footprint can actually reach them.
        """
        if (ix, iy) not in self._free:
            return False
        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                if (ix + dx, iy + dy) in self._occupied:
                    return False
        return True

    def clearance_score(self, ix: int, iy: int, max_radius: int = 3) -> int:
        """Chebyshev distance to the nearest OCCUPIED cell, capped at ``max_radius``."""
        for radius in range(0, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    if (ix + dx, iy + dy) in self._occupied:
                        return radius
        return max_radius

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
        self.mark_cells([(cx, cy)], radius_cells=radius_cells)

    def mark_cells(
        self,
        cells: list[tuple[int, int]] | set[tuple[int, int]],
        radius_cells: int = 0,
    ) -> None:
        """Permanently blacklist ``cells`` (plus an optional dilation radius)."""
        for cx, cy in cells:
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
