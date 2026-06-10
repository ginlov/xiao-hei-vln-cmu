"""Build walkable polygon from traversable-area point clouds."""

from __future__ import annotations

import numpy as np
from shapely import MultiPoint, concave_hull
from shapely.geometry import MultiPolygon, Polygon


def build_polygon(
    points: np.ndarray,
    *,
    ratio: float = 0.1,
    max_points: int = 5000,
) -> Polygon | MultiPolygon:
    """Return a concave hull of *points* with interior holes for obstacles.

    Subsamples to *max_points* for speed when the point cloud is large.
    *ratio* controls tightness: 0.0 = convex hull, 1.0 = tightest.
    """
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    mp = MultiPoint(points)
    return concave_hull(mp, ratio=ratio, allow_holes=True)


def erode_polygon(
    poly: Polygon | MultiPolygon,
    robot_radius: float = 0.3,
) -> Polygon | MultiPolygon:
    """Buffer the polygon inward by *robot_radius* for safe placement.

    Falls back to the original polygon if erosion empties it.
    """
    eroded = poly.buffer(-robot_radius)
    if eroded.is_empty:
        return poly
    return eroded
