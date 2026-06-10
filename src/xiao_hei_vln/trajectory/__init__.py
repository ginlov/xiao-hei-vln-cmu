"""Coverage trajectory generation for CMU VLN Challenge scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from xiao_hei_vln.messages.outputs import Waypoint

from ._coverage import (
    CoverageResult,
    compute_floor_coverage_sets,
    compute_object_coverage_sets,
    greedy_set_cover,
    sample_candidates,
)
from ._io import parse_traversable_ply_from_zip, read_objects_from_zip
from ._pathfinding import (
    build_visibility_graph,
    compute_geodesic_distances,
    find_collision_free_path,
    shortcut_path,
)
from ._polygon import build_polygon, erode_polygon
from ._tsp import compute_headings, path_length, solve_tsp


@dataclass(frozen=True)
class TrajectoryResult:
    """Full pipeline output."""

    waypoints: list[Waypoint]
    coverage: CoverageResult
    path_length_m: float
    stats: dict[str, Any] = field(default_factory=dict)


def plan_trajectory(
    traversable_points: np.ndarray,
    objects: dict[int, Any] | None = None,
    *,
    hull_ratio: float = 0.1,
    robot_radius: float = 0.3,
    grid_resolution: float = 0.25,
    coverage_radius: float = 3.0,
    object_weight: float = 10.0,
    min_floor_coverage: float = 0.95,
    floor_subsample: int = 10,
) -> TrajectoryResult:
    """Run the full coverage trajectory pipeline.

    1. Build polygon from point cloud
    2. Erode by robot radius
    3. Sample candidate viewpoints
    4. Compute object + floor coverage sets (with 2D raycasting)
    5. Greedy weighted set cover
    6. TSP ordering + headings
    """
    if objects is None:
        objects = {}

    poly = build_polygon(traversable_points, ratio=hull_ratio)
    eroded = erode_polygon(poly, robot_radius=robot_radius)
    candidates = sample_candidates(eroded, resolution=grid_resolution)

    if len(candidates) == 0:
        return TrajectoryResult(
            waypoints=[],
            coverage=CoverageResult([], 0.0, 0.0),
            path_length_m=0.0,
            stats={"error": "no candidates inside polygon"},
        )

    hole_polys = _extract_holes(poly)

    obj_sets = compute_object_coverage_sets(
        candidates, objects, hole_polys, radius=coverage_radius
    )

    floor_cells = traversable_points[::floor_subsample]
    floor_sets = compute_floor_coverage_sets(
        candidates, floor_cells, hole_polys, radius=coverage_radius
    )

    cov = greedy_set_cover(
        candidates,
        obj_sets,
        floor_sets,
        set(objects.keys()),
        len(floor_cells),
        object_weight=object_weight,
        min_floor_coverage=min_floor_coverage,
    )

    if not cov.selected_points:
        return TrajectoryResult(waypoints=[], coverage=cov, path_length_m=0.0)

    sel = np.array(cov.selected_points)

    all_nodes, adj, n_wp = build_visibility_graph(eroded, sel)
    geo_dist = compute_geodesic_distances(adj, len(all_nodes), n_wp)
    order = solve_tsp(sel, dist_matrix=geo_dist)

    routed = find_collision_free_path(all_nodes, adj, n_wp, order)
    routed = shortcut_path(routed, eroded, protected=sel)
    headings = compute_headings(routed)
    total_length = path_length(routed)

    waypoints = [
        Waypoint(x=float(routed[i, 0]), y=float(routed[i, 1]), heading=headings[i])
        for i in range(len(routed))
    ]

    return TrajectoryResult(
        waypoints=waypoints,
        coverage=cov,
        path_length_m=round(total_length, 2),
        stats={
            "hull_ratio": hull_ratio,
            "robot_radius": robot_radius,
            "grid_resolution": grid_resolution,
            "coverage_radius": coverage_radius,
            "num_holes": len(hole_polys),
            "num_coverage_viewpoints": len(sel),
        },
    )


def plan_trajectory_from_zip(
    zip_path: Path,
    scene_name: str,
    **kwargs: Any,
) -> TrajectoryResult:
    """Convenience: parse data from a scene zip, then plan."""
    pts = parse_traversable_ply_from_zip(zip_path, scene_name=scene_name)
    objs = read_objects_from_zip(zip_path, scene_name=scene_name)
    return plan_trajectory(pts, objs, **kwargs)


def _extract_holes(poly: ShapelyPolygon) -> list[ShapelyPolygon]:
    """Extract interior hole polygons from a Polygon or MultiPolygon."""
    holes: list[ShapelyPolygon] = []
    if poly.geom_type == "MultiPolygon":
        for g in poly.geoms:
            holes.extend(ShapelyPolygon(h) for h in g.interiors)
    elif hasattr(poly, "interiors"):
        holes.extend(ShapelyPolygon(h) for h in poly.interiors)
    return holes
