"""Exploration quality metrics vs GT traversable floor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationScore:
    """Primary metric is ``gt_coverage`` (fraction of GT free cells sensed)."""

    gt_coverage: float
    mapped_coverage: float
    path_length_m: float
    coverage_per_meter: float
    n_gt_free: int
    n_seen: int
    n_mapped: int
    n_visited_waypoints: int
    ticks: int
    done_reason: str

    @property
    def primary(self) -> float:
        return self.gt_coverage


def score_exploration(
    *,
    gt_free: set[tuple[int, int]],
    seen_gt: set[tuple[int, int]],
    mapped_free: set[tuple[int, int]],
    path_length_m: float,
    n_visited_waypoints: int,
    ticks: int,
    done_reason: str,
) -> ExplorationScore:
    n_gt = max(len(gt_free), 1)
    n_seen = len(seen_gt & gt_free)
    n_mapped = len(mapped_free & gt_free)
    gt_cov = n_seen / n_gt
    map_cov = n_mapped / n_gt
    path = max(path_length_m, 1e-6)
    return ExplorationScore(
        gt_coverage=gt_cov,
        mapped_coverage=map_cov,
        path_length_m=path_length_m,
        coverage_per_meter=gt_cov / path,
        n_gt_free=len(gt_free),
        n_seen=n_seen,
        n_mapped=n_mapped,
        n_visited_waypoints=n_visited_waypoints,
        ticks=ticks,
        done_reason=done_reason,
    )
