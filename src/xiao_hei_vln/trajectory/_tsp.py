"""Nearest-neighbor TSP with 2-opt improvement, and heading computation."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import pdist, squareform


def solve_tsp(
    points: np.ndarray, dist_matrix: np.ndarray | None = None
) -> list[int]:
    """Return a visit order for *points* using nearest-neighbor + 2-opt.

    *points* is (N, 2).  If *dist_matrix* is given, it is used instead of
    Euclidean distances (e.g. geodesic distances from a visibility graph).
    Returns a permutation of ``range(N)``.
    """
    n = len(points)
    if n <= 2:
        return list(range(n))

    dist = dist_matrix if dist_matrix is not None else squareform(pdist(points))
    order = _nearest_neighbor(dist)
    order = _improve_2opt(order, dist)
    return order


def compute_headings(ordered_points: np.ndarray) -> list[float]:
    """Compute heading (radians, atan2 convention) toward the next waypoint.

    The last waypoint keeps the same heading as the second-to-last.
    """
    n = len(ordered_points)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    headings: list[float] = []
    for i in range(n - 1):
        dx = ordered_points[i + 1, 0] - ordered_points[i, 0]
        dy = ordered_points[i + 1, 1] - ordered_points[i, 1]
        headings.append(math.atan2(dy, dx))
    headings.append(headings[-1])
    return headings


def path_length(ordered_points: np.ndarray) -> float:
    """Total Euclidean path length."""
    if len(ordered_points) < 2:
        return 0.0
    diffs = np.diff(ordered_points, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def _nearest_neighbor(dist: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor starting from node 0."""
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        cur = order[-1]
        row = dist[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        order.append(nxt)
        visited[nxt] = True

    return order


def _improve_2opt(order: list[int], dist: np.ndarray) -> list[int]:
    """Apply 2-opt local search until no improvement."""
    route = list(order)
    n = len(route)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                d_old = dist[route[i], route[i + 1]] + (
                    dist[route[j], route[(j + 1) % n]] if j + 1 < n else 0
                )
                d_new = dist[route[i], route[j]] + (
                    dist[route[i + 1], route[(j + 1) % n]] if j + 1 < n else 0
                )
                if d_new < d_old - 1e-10:
                    route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                    improved = True

    return route
