"""Ground-truth occupancy world from ``traversable_area.ply``.

Used by the offline exploration benchmark: voxelize the walkable floor,
treat the exterior of the free blob as occupied walls, ray-cast synthetic
``terrain_ext``, and A* between waypoints.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from pathlib import Path

import numpy as np

from xiao_hei_vln.scene.io import parse_traversable_ply

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class GTWorld:
    """2-D GT free/occupied grid in the map frame."""

    def __init__(
        self,
        free: set[tuple[int, int]],
        occupied: set[tuple[int, int]],
        resolution: float,
    ) -> None:
        self.free = free
        self.occupied = occupied
        self.resolution = resolution
        self._free_list = list(free)

    @classmethod
    def from_traversable_ply(
        cls,
        path: Path | str,
        resolution: float = 0.2,
        wall_thickness: int = 1,
    ) -> "GTWorld":
        pts = parse_traversable_ply(Path(path))
        if pts.size == 0:
            raise ValueError(f"empty traversable ply: {path}")
        free: set[tuple[int, int]] = set()
        for x, y in pts:
            free.add((math.floor(x / resolution), math.floor(y / resolution)))

        # Occupied = cells within wall_thickness of free that are not free
        # (forms a thin wall shell so raycasts stop at the room boundary).
        occupied: set[tuple[int, int]] = set()
        for ix, iy in free:
            for t in range(1, wall_thickness + 1):
                for dx in range(-t, t + 1):
                    for dy in range(-t, t + 1):
                        if max(abs(dx), abs(dy)) != t:
                            continue
                        cell = (ix + dx, iy + dy)
                        if cell not in free:
                            occupied.add(cell)
        return cls(free=free, occupied=occupied, resolution=resolution)

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        half = self.resolution * 0.5
        return (ix * self.resolution + half, iy * self.resolution + half)

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self.resolution), math.floor(y / self.resolution))

    def is_free(self, x: float, y: float) -> bool:
        return self.to_cell(x, y) in self.free

    def is_occupied(self, x: float, y: float) -> bool:
        cell = self.to_cell(x, y)
        if cell in self.free:
            return False
        if cell in self.occupied:
            return True
        # Outside known free and wall shell → treat as occupied (void).
        return True

    def start_pose(self) -> tuple[float, float]:
        """Centroid of the largest connected FREE component."""
        if not self.free:
            return (0.0, 0.0)
        seen: set[tuple[int, int]] = set()
        best: list[tuple[int, int]] = []
        for seed in self.free:
            if seed in seen:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([seed])
            while q:
                c = q.popleft()
                if c in seen or c not in self.free:
                    continue
                seen.add(c)
                comp.append(c)
                for dx, dy in _NEIGHBOURS:
                    q.append((c[0] + dx, c[1] + dy))
            if len(comp) > len(best):
                best = comp
        cx = sum(c[0] for c in best) / len(best)
        cy = sum(c[1] for c in best) / len(best)
        # Snap to nearest free cell to centroid.
        cell = min(best, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        return self.to_world(*cell)

    def sense(
        self,
        x: float,
        y: float,
        *,
        max_range_m: float = 8.0,
        n_rays: int = 180,
        sample_step_m: float | None = None,
    ) -> np.ndarray:
        """LiDAR-style ray-cast → (N, 4) terrain points [x,y,z,cost]."""
        step = sample_step_m or self.resolution
        pts: list[tuple[float, float, float, float]] = []
        for a in np.linspace(0.0, 2 * math.pi, n_rays, endpoint=False):
            dx, dy = math.cos(float(a)), math.sin(float(a))
            t = step
            while t <= max_range_m:
                px, py = x + t * dx, y + t * dy
                if self.is_occupied(px, py):
                    pts.append((px, py, 0.0, 1.0))
                    break
                pts.append((px, py, 0.0, 0.0))
                t += step
        if not pts:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def visible_gt_cells(
        self,
        x: float,
        y: float,
        *,
        max_range_m: float = 8.0,
        n_rays: int = 180,
        sample_step_m: float | None = None,
    ) -> set[tuple[int, int]]:
        """GT FREE cells hit by the current sensor sweep."""
        step = sample_step_m or self.resolution
        seen: set[tuple[int, int]] = set()
        for a in np.linspace(0.0, 2 * math.pi, n_rays, endpoint=False):
            dx, dy = math.cos(float(a)), math.sin(float(a))
            t = step
            while t <= max_range_m:
                px, py = x + t * dx, y + t * dy
                cell = self.to_cell(px, py)
                if cell in self.occupied or (
                    cell not in self.free and cell not in self.occupied
                ):
                    break
                if cell in self.free:
                    seen.add(cell)
                t += step
        return seen

    def astar(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> list[tuple[float, float]] | None:
        """A* on GT FREE cells. Returns world waypoints (goal inclusive)."""
        sc = self.to_cell(*start_xy)
        gc = self.to_cell(*goal_xy)
        if sc not in self.free:
            sc = self._nearest_free(sc)
        if gc not in self.free:
            gc = self._nearest_free(gc)
        if sc is None or gc is None:
            return None

        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, sc)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        gscore: dict[tuple[int, int], float] = {sc: 0.0}

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == gc:
                cells = [cur]
                while cells[-1] in came:
                    cells.append(came[cells[-1]])
                cells.reverse()
                return [self.to_world(ix, iy) for ix, iy in cells]
            for dx, dy in _NEIGHBOURS:
                nbr = (cur[0] + dx, cur[1] + dy)
                if nbr not in self.free:
                    continue
                tent = gscore[cur] + self.resolution
                if tent < gscore.get(nbr, math.inf):
                    came[nbr] = cur
                    gscore[nbr] = tent
                    h = self.resolution * math.hypot(nbr[0] - gc[0], nbr[1] - gc[1])
                    heapq.heappush(open_heap, (tent + h, nbr))
        return None

    def _nearest_free(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        if cell in self.free:
            return cell
        for r in range(1, 25):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cand = (cell[0] + dx, cell[1] + dy)
                    if cand in self.free:
                        return cand
        return None
