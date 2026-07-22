"""Map-free exploration strategies (belief map only — no GT floor plan).

These algorithms only use the online OccupancyGrid built from ``terrain_ext``:
wall-follow, next-best-view sampling, and RRT-based goal selection.
"""

from __future__ import annotations

import math
import random
from collections import deque

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class _MapFreeBase:
    """Shared belief-grid + reach/stuck loop for map-free explorers."""

    def __init__(
        self,
        *,
        max_waypoints: int = 100,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.45,
        cost_threshold: float = 0.5,
        stuck_timeout_s: float = 12.0,
        max_consecutive_skips: int = 25,
        seed: int = 0,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._cost_threshold = cost_threshold
        self._stuck_timeout_s = stuck_timeout_s
        self._max_consecutive_skips = max_consecutive_skips
        self._rng = random.Random(seed)

        self._grid = OccupancyGrid(grid_resolution)
        self._current_target: Waypoint | None = None
        self._visited: list[Waypoint] = []
        self._done = False
        self._target_set_time: float | None = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._blacklist: set[tuple[int, int]] = set()
        self._heading = 0.0

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        if self._done:
            return None
        if snapshot.terrain_ext is not None:
            self._grid.update(snapshot.terrain_ext, self._cost_threshold)
        pose = snapshot.pose
        if pose is None:
            return self._current_target
        rx, ry = pose.position.x, pose.position.y
        now = snapshot.tick_time.to_seconds()

        if self._current_target is not None and self._within(rx, ry, self._current_target):
            self._heading = self._current_target.heading
            self._visited.append(self._current_target)
            self._mark(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

        if self._current_target is not None and self._target_set_time is not None:
            if now - self._target_set_time > self._stuck_timeout_s:
                self._skip(self._current_target)
                if self._done:
                    return None

        if len(self._visited) >= self._max_waypoints:
            self._done = True
            return None

        if self._current_target is None:
            self._current_target = self._pick(rx, ry)
            if self._current_target is None:
                if not self._grid.free_cells or not self._grid.reachable_path_costs(rx, ry):
                    return None
                self._done = True
                return None
            self._target_set_time = now
        return self._current_target

    def force_skip(self) -> None:
        if self._current_target is not None:
            self._skip(self._current_target)

    def advance(self) -> None:
        if self._current_target is not None:
            self._heading = self._current_target.heading
            self._visited.append(self._current_target)
            self._mark(self._current_target)
            self._current_target = None
            self._target_set_time = None
            self._consecutive_skip_count = 0

    def is_complete(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._grid = OccupancyGrid(self._grid.resolution)
        self._current_target = None
        self._visited = []
        self._done = False
        self._target_set_time = None
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._blacklist.clear()
        self._heading = 0.0

    def get_visited_waypoints(self) -> list[Waypoint]:
        wps = list(self._visited)
        if self._current_target is not None:
            wps.append(self._current_target)
        return wps

    def get_grid(self) -> OccupancyGrid:
        return self._grid

    def get_viz_layers(self) -> dict:
        grid = self._grid
        current = None
        if self._current_target is not None:
            current = (self._current_target.x, self._current_target.y)
        return {
            "free": [grid.to_world(ix, iy) for ix, iy in grid.free_cells],
            "frontier": [grid.to_world(ix, iy) for ix, iy in grid.frontier_cells()],
            "soft_ban": [grid.to_world(ix, iy) for ix, iy in self._blacklist],
            "hard_ban": [],
            "visited": [(w.x, w.y) for w in self._visited],
            "current": current,
            "resolution": grid.resolution,
        }

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        raise NotImplementedError

    def _wp(self, cell: tuple[int, int], rx: float, ry: float) -> Waypoint:
        cx, cy = self._grid.to_world(*cell)
        return Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

    def _skip(self, wp: Waypoint) -> None:
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        self._blacklist.add(self._grid.world_to_grid(wp.x, wp.y))
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def _mark(self, wp: Waypoint) -> None:
        # radius_cells=3 (~0.6 m at 0.2 m res) suppresses nearby re-picks so
        # NBV/RRT/wall-follow do not thrash one doorway/corner forever.
        self._grid.mark_occupied(wp.x, wp.y, radius_cells=3)

    def _within(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist


class WallFollowExplorer(_MapFreeBase):
    """Left-hand wall follower on the belief occupancy map."""

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return None
        occupied = self._grid._occupied
        wall_cells = [
            c for c in costs
            if c not in self._blacklist
            and any((c[0] + dx, c[1] + dy) in occupied for dx, dy in _NEIGHBOURS)
        ]
        if not wall_cells:
            # Fallback: any frontier-adjacent free cell.
            known = self._grid.free_cells | occupied
            wall_cells = [
                c for c in costs
                if c not in self._blacklist
                and any((c[0] + dx, c[1] + dy) not in known for dx, dy in _NEIGHBOURS)
            ]
        if not wall_cells:
            return None

        # Prefer wall cell roughly to the left of current heading, then nearest.
        hx, hy = math.cos(self._heading), math.sin(self._heading)
        # Left normal.
        lx, ly = -hy, hx

        def score(cell: tuple[int, int]) -> tuple[float, float]:
            wx, wy = self._grid.to_world(*cell)
            dx, dy = wx - rx, wy - ry
            left = dx * lx + dy * ly
            ahead = dx * hx + dy * hy
            return (-left if left > 0 else 1.0, costs[cell] - 0.1 * ahead)

        cell = min(wall_cells, key=score)
        return self._wp(cell, rx, ry)


class NextBestViewExplorer(_MapFreeBase):
    """Sample candidate FREE poses; pick max unknown-gain / path cost."""

    def __init__(self, *, n_samples: int = 40, **kwargs) -> None:
        super().__init__(**kwargs)
        self._n_samples = n_samples

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return None
        candidates = [c for c in costs if c not in self._blacklist and costs[c] > self._reach_dist]
        if not candidates:
            candidates = [c for c in costs if c not in self._blacklist]
        if not candidates:
            return None

        # Bias samples toward frontier cells when available.
        frontiers = [
            c for c in self._grid.frontier_cells()
            if c in costs and c not in self._blacklist
        ]
        pool = frontiers * 3 + candidates if frontiers else candidates
        samples = [self._rng.choice(pool) for _ in range(min(self._n_samples, len(pool) * 2))]
        # Always consider a few true frontiers.
        samples.extend(frontiers[:10])

        best_cell = None
        best_score = -1.0
        for cell in samples:
            if cell not in costs or cell in self._blacklist:
                continue
            if costs[cell] <= self._reach_dist:
                continue
            gain = self._grid.unknown_region_size([cell], radius_cells=12)
            path = max(costs[cell], 1e-3)
            score = gain / (1.0 + path)
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            best_cell = min(candidates, key=lambda c: costs[c])
        return self._wp(best_cell, rx, ry)


class RRTExplorer(_MapFreeBase):
    """Grow a grid RRT in FREE space; pick the most informative node as goal.

    Samples may land in UNKNOWN; the tree steers to the FREE cell nearest the
    sample (classic exploration bias toward the frontier) — still no GT map.
    """

    def __init__(
        self,
        *,
        n_iter: int = 200,
        step_cells: int = 3,
        goal_bias: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._n_iter = n_iter
        self._step_cells = max(1, step_cells)
        self._goal_bias = goal_bias

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return None
        free = set(costs.keys())
        start = self._grid.world_to_grid(rx, ry)
        if start not in free:
            start = min(free, key=lambda c: (c[0] - start[0]) ** 2 + (c[1] - start[1]) ** 2)

        frontiers = [c for c in self._grid.frontier_cells() if c in free and c not in self._blacklist]
        xs = [c[0] for c in free]
        ys = [c[1] for c in free]
        pad = 8
        min_x, max_x = min(xs) - pad, max(xs) + pad
        min_y, max_y = min(ys) - pad, max(ys) + pad

        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        nodes = [start]

        for _ in range(self._n_iter):
            if frontiers and self._rng.random() < self._goal_bias:
                sample = self._rng.choice(frontiers)
            else:
                sample = (
                    self._rng.randint(min_x, max_x),
                    self._rng.randint(min_y, max_y),
                )
            # Nearest tree node.
            nearest = min(
                nodes,
                key=lambda n: (n[0] - sample[0]) ** 2 + (n[1] - sample[1]) ** 2,
            )
            steered = self._steer(nearest, sample, free)
            if steered is None or steered in parent or steered in self._blacklist:
                continue
            parent[steered] = nearest
            nodes.append(steered)
            if steered in free and steered not in frontiers:
                # Newly reached free near unknown becomes a soft frontier target.
                pass

        # Score tree nodes by local unknown gain / tree depth proxy (path hops).
        best_cell = None
        best_score = -1.0
        for node in nodes:
            if node == start or node in self._blacklist:
                continue
            if node not in free:
                continue
            gain = self._grid.unknown_region_size([node], radius_cells=10)
            depth = 0
            cur: tuple[int, int] | None = node
            while cur is not None and depth < 200:
                cur = parent.get(cur)
                depth += 1
            score = (gain + 1.0) / (1.0 + 0.25 * depth)
            if score > best_score:
                best_score = score
                best_cell = node

        if best_cell is None:
            # Fallback: farthest free cell in the RRT / reachable set.
            best_cell = max(free, key=lambda c: costs[c])
        return self._wp(best_cell, rx, ry)

    def _steer(
        self,
        nearest: tuple[int, int],
        sample: tuple[int, int],
        free: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        dx = sample[0] - nearest[0]
        dy = sample[1] - nearest[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None
        step = min(self._step_cells, int(dist))
        ux, uy = dx / dist, dy / dist
        last_free = None
        for s in range(1, step + 1):
            cell = (int(round(nearest[0] + ux * s)), int(round(nearest[1] + uy * s)))
            if cell in free:
                last_free = cell
            else:
                # Hit unknown/occupied — stop at last free (frontier push).
                break
        return last_free
