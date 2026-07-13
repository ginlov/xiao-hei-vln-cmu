"""2-D occupancy grid built incrementally from TerrainMap snapshots.

Cells are classified as FREE (traversable, cost ≤ threshold) or OCCUPIED
(high-cost obstacle). Cells never seen in any snapshot remain UNKNOWN.
Frontier cells are FREE cells that have at least one UNKNOWN neighbour.
"""

from __future__ import annotations

import math

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

    # ------------------------------------------------------------------
    # Coordinate helpers

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        half = self._res * 0.5
        return (ix * self._res + half, iy * self._res + half)

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
