"""Candidate viewpoint sampling, visibility computation, and greedy set cover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


@dataclass(frozen=True)
class CoverageResult:
    """Output of the greedy set cover."""

    selected_points: list[tuple[float, float]]
    object_coverage: float
    floor_coverage: float
    uncovered_object_ids: list[int] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def sample_candidates(
    polygon: Polygon | MultiPolygon,
    resolution: float = 0.5,
) -> np.ndarray:
    """Sample candidate viewpoints on a grid inside *polygon*."""
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.arange(min_x + resolution / 2, max_x, resolution)
    ys = np.arange(min_y + resolution / 2, max_y, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    all_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    from shapely import contains_xy

    mask = contains_xy(polygon, all_pts[:, 0], all_pts[:, 1])
    return all_pts[mask]


def compute_object_coverage_sets(
    candidates: np.ndarray,
    objects: dict[int, Any],
    holes: list[Polygon],
    radius: float = 3.0,
) -> list[set[int]]:
    """For each candidate, compute visible object IDs (within radius + LOS clear).

    Uses KD-tree for range query, then checks line-of-sight against *holes*.
    """
    if not objects:
        return [set() for _ in range(len(candidates))]

    obj_ids = list(objects.keys())
    obj_xy = np.array([[objects[oid].center.x, objects[oid].center.y] for oid in obj_ids])
    obj_tree = cKDTree(obj_xy)
    cand_tree = cKDTree(candidates)

    pairs = cand_tree.query_ball_tree(obj_tree, r=radius)

    if holes:
        from shapely import prepare

        for h in holes:
            prepare(h)

    # Pre-compute: for each object, find its "host" hole (the one it sits on)
    # so we can skip raycasting against it. An object is "on" a hole if its
    # 2D projection is inside or within 0.5m of the hole boundary.
    obj_host_hole: dict[int, int | None] = {}
    for j, _oid in enumerate(obj_ids):
        ox, oy = obj_xy[j]
        host = None
        if holes:
            op = Point(ox, oy)
            for hi, h in enumerate(holes):
                if h.contains(op) or h.distance(op) < 0.5:
                    host = hi
                    break
        obj_host_hole[j] = host

    coverage: list[set[int]] = []
    for ci, nearby_objs in enumerate(pairs):
        visible: set[int] = set()
        cx, cy = candidates[ci]
        for oj in nearby_objs:
            ox, oy = obj_xy[oj]
            skip_hole = obj_host_hole[oj]
            if _line_of_sight_clear(cx, cy, ox, oy, holes, skip_hole):
                visible.add(obj_ids[oj])
        coverage.append(visible)
    return coverage


def compute_floor_coverage_sets(
    candidates: np.ndarray,
    floor_cells: np.ndarray,
    holes: list[Polygon],
    radius: float = 3.0,
) -> list[set[int]]:
    """For each candidate, compute visible floor cell indices (within radius + LOS clear)."""
    if len(floor_cells) == 0:
        return [set() for _ in range(len(candidates))]

    floor_tree = cKDTree(floor_cells)
    cand_tree = cKDTree(candidates)
    pairs = cand_tree.query_ball_tree(floor_tree, r=radius)

    if not holes:
        return [set(p) for p in pairs]

    from shapely import prepare as _prepare

    for h in holes:
        _prepare(h)

    coverage: list[set[int]] = []
    for ci, nearby_cells in enumerate(pairs):
        visible: set[int] = set()
        cx, cy = candidates[ci]
        for fi in nearby_cells:
            fx, fy = floor_cells[fi]
            if _line_of_sight_clear(cx, cy, fx, fy, holes):
                visible.add(fi)
        coverage.append(visible)
    return coverage


def greedy_set_cover(
    candidates: np.ndarray,
    obj_sets: list[set[int]],
    floor_sets: list[set[int]],
    all_obj_ids: set[int],
    n_floor_cells: int,
    *,
    object_weight: float = 10.0,
    min_floor_coverage: float = 0.95,
) -> CoverageResult:
    """Greedy weighted set cover prioritizing object visibility."""
    uncovered_objs = set(all_obj_ids)
    uncovered_floor = set(range(n_floor_cells))
    selected_indices: list[int] = []
    used = np.zeros(len(candidates), dtype=bool)

    while uncovered_objs or (1 - len(uncovered_floor) / max(n_floor_cells, 1)) < min_floor_coverage:
        best_idx = -1
        best_score = -1.0

        for i in range(len(candidates)):
            if used[i]:
                continue
            obj_gain = len(obj_sets[i] & uncovered_objs)
            floor_gain = len(floor_sets[i] & uncovered_floor)
            score = object_weight * obj_gain + floor_gain
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score <= 0:
            break

        selected_indices.append(best_idx)
        used[best_idx] = True
        uncovered_objs -= obj_sets[best_idx]
        uncovered_floor -= floor_sets[best_idx]

    n_total = len(all_obj_ids) if all_obj_ids else 0
    n_covered = n_total - len(uncovered_objs)
    obj_cov = n_covered / n_total if n_total > 0 else 1.0
    floor_cov = 1 - len(uncovered_floor) / max(n_floor_cells, 1)

    return CoverageResult(
        selected_points=[(candidates[i][0], candidates[i][1]) for i in selected_indices],
        object_coverage=round(obj_cov, 4),
        floor_coverage=round(floor_cov, 4),
        uncovered_object_ids=sorted(uncovered_objs),
        stats={
            "num_candidates": len(candidates),
            "num_selected": len(selected_indices),
            "objects_total": n_total,
            "objects_covered": n_covered,
        },
    )


def _line_of_sight_clear(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    holes: list[Polygon] | None,
    skip_hole: int | None = None,
) -> bool:
    """Return True if the line from (x1,y1) to (x2,y2) doesn't cross any hole.

    *skip_hole* is the index of a hole to exclude (the object sits on it).
    """
    if not holes:
        return True
    line = LineString([(x1, y1), (x2, y2)])
    for i, h in enumerate(holes):
        if i == skip_hole:
            continue
        if line.intersects(h):
            return False
    return True
