"""Frontier-based exploration policy.

Given a populated ``GlobalMap`` plus the current robot pose, picks the
"most promising" frontier cluster to head toward this tick. Promising
is a weighted combination of:

* ``log(size)`` — bigger frontier clusters reveal more unknown area at once,
* negative distance — closer frontiers are cheaper to reach,
* an anti-loop penalty — recent targets get a soft decay-based discount so
  the planner doesn't oscillate between two equally-attractive frontiers.

The scorer is purely geometric in v0. VLM-based scoring (per VLFM and
Habibpour & Afghah 2025) is intentionally deferred to Phase 2 so we
can ship the geometric loop first and tune it against the simulator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from xiao_hei_vln.messages.sensors import OdomPose
from xiao_hei_vln.perception.global_map import FrontierCluster, GlobalMap


@dataclass(frozen=True)
class ScoredFrontier:
    """A frontier cluster plus its scoring breakdown.

    The breakdown is exposed mainly for debugging / logging — the planner
    itself returns the underlying ``FrontierCluster`` to the caller.
    """

    cluster: FrontierCluster
    distance_m: float
    size_term: float
    distance_term: float
    loop_term: float
    score: float


@dataclass(frozen=True)
class ScoringWeights:
    """Linear-combination weights for the v0 geometric scorer."""

    size: float = 1.0
    distance: float = 0.5
    loop: float = 3.0
    # Characteristic length (m) for the exponential anti-loop decay.
    loop_decay_m: float = 2.0


class FrontierPlanner:
    """Picks one frontier per tick from the GlobalMap.

    Stateful: keeps a short ring buffer of recently chosen target centroids
    so it can dampen oscillation. Reset between questions via ``reset()``.
    """

    def __init__(
        self,
        global_map: GlobalMap,
        weights: ScoringWeights | None = None,
        history_length: int = 5,
        min_cluster_size: int = 4,
    ) -> None:
        if history_length < 0:
            raise ValueError(f"history_length must be >= 0; got {history_length}")
        if min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be >= 1; got {min_cluster_size}")
        self._map = global_map
        self._weights = weights if weights is not None else ScoringWeights()
        self._history: deque[tuple[float, float]] = deque(maxlen=history_length)
        self._min_cluster_size = int(min_cluster_size)

    # ------------------------------------------------------------------ properties

    @property
    def history(self) -> tuple[tuple[float, float], ...]:
        """Recent selected target centroids (oldest first)."""
        return tuple(self._history)

    # ------------------------------------------------------------------ main API

    def select(self, pose: OdomPose) -> FrontierCluster | None:
        """Return the highest-scored reachable frontier cluster, or None."""
        scored = self.score_all(pose)
        if not scored:
            return None
        best = max(scored, key=lambda s: s.score)
        self._history.append(best.cluster.centroid_xy)
        return best.cluster

    def score_all(self, pose: OdomPose) -> list[ScoredFrontier]:
        """Return every reachable cluster with its score breakdown.

        Useful for debugging and for the future VLM-scored variant which
        layers a Qwen score on top of the geometric ranking.
        """
        clusters = self._map.find_frontier_clusters(self._min_cluster_size)
        if not clusters:
            return []

        px, py = pose.position.x, pose.position.y
        reachable = (c for c in clusters if self._reachable(pose, c.centroid_xy))
        return [self._score(c, px, py) for c in reachable]

    def reset(self) -> None:
        """Clear history at start of new question."""
        self._history.clear()

    # ------------------------------------------------------------------ internals

    def _reachable(self, pose: OdomPose, target_xy: tuple[float, float]) -> bool:
        """Cheap straight-line reachability check.

        For the current single-room scenario this is sufficient. Upgrade
        to an A* search on the occupancy grid if the planner starts
        picking obviously-unreachable targets through non-convex obstacles.
        """
        return not self._map.line_passes_through(
            pose.position.x, pose.position.y, target_xy[0], target_xy[1],
        )

    def _score(self, cluster: FrontierCluster, px: float, py: float) -> ScoredFrontier:
        cx, cy = cluster.centroid_xy
        dx = cx - px
        dy = cy - py
        d = math.hypot(dx, dy)
        size_term = self._weights.size * math.log1p(cluster.size)
        distance_term = -self._weights.distance * d
        loop_term = -self._weights.loop * self._loop_penalty(cx, cy)
        score = size_term + distance_term + loop_term
        return ScoredFrontier(
            cluster=cluster,
            distance_m=d,
            size_term=size_term,
            distance_term=distance_term,
            loop_term=loop_term,
            score=score,
        )

    def _loop_penalty(self, cx: float, cy: float) -> float:
        """Sum of exponentially-decayed proximity to past targets.

        A candidate within ``loop_decay_m`` of a recent target gets ~1
        penalty contribution; one ``3 * loop_decay_m`` away gets ~0.05.
        """
        if not self._history:
            return 0.0
        decay = self._weights.loop_decay_m
        return sum(
            math.exp(-math.hypot(cx - hx, cy - hy) / decay)
            for hx, hy in self._history
        )


__all__ = ["FrontierPlanner", "ScoredFrontier", "ScoringWeights"]
