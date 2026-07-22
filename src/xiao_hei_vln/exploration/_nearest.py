"""Nearest-frontier and random-frontier exploration strategies."""

from __future__ import annotations

import math
import random
from collections import deque

from xiao_hei_vln.exploration._frontier import _cluster_frontier
from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint


class _BaseFrontierPick:
    """Shared grid + stuck/reach mechanics; subclasses only choose the cluster."""

    def __init__(
        self,
        *,
        max_waypoints: int = 100,
        grid_resolution: float = 0.2,
        waypoint_reach_dist: float = 0.4,
        min_frontier_size: int = 3,
        cost_threshold: float = 0.5,
        stuck_timeout_s: float = 12.0,
        max_consecutive_skips: int = 25,
        seed: int = 0,
    ) -> None:
        self._max_waypoints = max_waypoints
        self._reach_dist = waypoint_reach_dist
        self._min_frontier_size = min_frontier_size
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
            self._visited.append(self._current_target)
            self._mark_wp(self._current_target)
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
            self._visited.append(self._current_target)
            self._mark_wp(self._current_target)
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

    def _candidate_clusters(self, rx: float, ry: float):
        raw = self._grid.frontier_cells()
        clusters = [
            c for c in _cluster_frontier(raw)
            if len(c) >= self._min_frontier_size
            and not any(cell in self._blacklist for cell in c)
        ]
        costs = self._grid.reachable_path_costs(rx, ry)
        if not costs:
            return [], costs
        clusters = [c for c in clusters if any(cell in costs for cell in c)]
        return clusters, costs

    def _wp_from_cluster(self, cluster, costs, rx, ry) -> Waypoint:
        cell = min((c for c in cluster if c in costs), key=lambda c: costs[c])
        cx, cy = self._grid.to_world(*cell)
        return Waypoint(x=cx, y=cy, heading=math.atan2(cy - ry, cx - rx))

    def _skip(self, wp: Waypoint) -> None:
        self.skipped_count += 1
        self._consecutive_skip_count += 1
        cx, cy = self._grid.world_to_grid(wp.x, wp.y)
        self._blacklist.add((cx, cy))
        self._current_target = None
        self._target_set_time = None
        if self._consecutive_skip_count >= self._max_consecutive_skips:
            self._done = True

    def _mark_wp(self, wp: Waypoint) -> None:
        self._grid.mark_occupied(wp.x, wp.y, radius_cells=1)

    def _within(self, rx: float, ry: float, wp: Waypoint) -> bool:
        return math.hypot(rx - wp.x, ry - wp.y) < self._reach_dist


class NearestFrontierExplorer(_BaseFrontierPick):
    """Always drive to the path-nearest reachable frontier cluster."""

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        clusters, costs = self._candidate_clusters(rx, ry)
        if not clusters:
            return None
        best = None
        best_cost = math.inf
        for cluster in clusters:
            cell = min((c for c in cluster if c in costs), key=lambda c: costs[c])
            if costs[cell] < best_cost:
                best_cost = costs[cell]
                best = cluster
        assert best is not None
        return self._wp_from_cluster(best, costs, rx, ry)


class RandomFrontierExplorer(_BaseFrontierPick):
    """Uniform random among reachable frontier clusters (seeded)."""

    def _pick(self, rx: float, ry: float) -> Waypoint | None:
        clusters, costs = self._candidate_clusters(rx, ry)
        if not clusters:
            return None
        cluster = self._rng.choice(clusters)
        return self._wp_from_cluster(cluster, costs, rx, ry)
