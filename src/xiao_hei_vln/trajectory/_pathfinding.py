"""Visibility-graph pathfinding for collision-free trajectory routing."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree
from shapely import prepare
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


def build_visibility_graph(
    eroded_poly: Polygon | MultiPolygon,
    waypoints: np.ndarray,
    *,
    simplify_tol: float = 0.05,
    max_vis_dist: float = 10.0,
) -> tuple[list[tuple[float, float]], list[list[tuple[int, float]]], int]:
    """Build a visibility graph over *waypoints* and polygon boundary vertices.

    Returns ``(all_nodes, adjacency_list, n_waypoints)``.
    Visibility is checked against the **original** *eroded_poly* (not simplified).
    """
    simplified = eroded_poly.simplify(simplify_tol, preserve_topology=True)
    if simplified.is_empty:
        simplified = eroded_poly

    poly_verts, boundary_edges = _extract_vertices_and_edges(simplified)

    # Filter out simplified vertices that drifted outside the original polygon.
    valid_verts: list[tuple[float, float]] = []
    vert_remap: dict[int, int] = {}
    for i, v in enumerate(poly_verts):
        if eroded_poly.distance(Point(v)) < simplify_tol + 1e-6:
            vert_remap[i] = len(valid_verts)
            valid_verts.append(v)

    wp_list = [tuple(w) for w in waypoints]
    all_nodes = wp_list + valid_verts
    n_wp = len(wp_list)
    n = len(all_nodes)
    coords = np.array(all_nodes)

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    added: set[tuple[int, int]] = set()

    # Prepare the ORIGINAL polygon for fast visibility checks.
    prepare(eroded_poly)

    # Boundary edges from simplified polygon — validate against original.
    for i_local, j_local in boundary_edges:
        if i_local not in vert_remap or j_local not in vert_remap:
            continue
        i = vert_remap[i_local] + n_wp
        j = vert_remap[j_local] + n_wp
        line = LineString([all_nodes[i], all_nodes[j]])
        if eroded_poly.covers(line):
            d = float(np.hypot(coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]))
            adj[i].append((j, d))
            adj[j].append((i, d))
            added.add((min(i, j), max(i, j)))

    # Cross-edges within range — check against original polygon.
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_vis_dist)

    for i, j in pairs:
        key = (min(i, j), max(i, j))
        if key in added:
            continue
        line = LineString([all_nodes[i], all_nodes[j]])
        if eroded_poly.covers(line):
            d = float(np.hypot(coords[j, 0] - coords[i, 0], coords[j, 1] - coords[i, 1]))
            adj[i].append((j, d))
            adj[j].append((i, d))
            added.add(key)

    return all_nodes, adj, n_wp


def find_collision_free_path(
    all_nodes: list[tuple[float, float]],
    adj: list[list[tuple[int, float]]],
    n_wp: int,
    ordered_wp_indices: list[int],
) -> np.ndarray:
    """Route through the visibility graph following *ordered_wp_indices*.

    Returns an (M, 2) array: the waypoints plus intermediate polygon
    vertices needed to navigate around obstacles.
    """
    if len(ordered_wp_indices) < 2:
        return np.array([all_nodes[i] for i in ordered_wp_indices])

    n = len(all_nodes)
    full_path: list[tuple[float, float]] = [all_nodes[ordered_wp_indices[0]]]

    for k in range(len(ordered_wp_indices) - 1):
        src = ordered_wp_indices[k]
        dst = ordered_wp_indices[k + 1]
        path_indices = _dijkstra(adj, n, src, dst)
        if path_indices is None:
            raise RuntimeError(
                f"No collision-free path in visibility graph from {src} to {dst}; "
                "consider increasing max_vis_dist / simplify_tol or adding more nodes"
            )
        for idx in path_indices[1:]:
            full_path.append(all_nodes[idx])

    return np.array(full_path)


def shortcut_path(
    path: np.ndarray,
    eroded_poly: Polygon | MultiPolygon,
    protected: np.ndarray | None = None,
) -> np.ndarray:
    """Remove intermediate points where a direct segment is collision-free.

    Points in *protected* (coverage viewpoints) are never removed — only
    intermediate routing vertices inserted by the visibility graph are
    candidates for shortcutting.
    """
    if len(path) <= 2:
        return path.copy()

    prepare(eroded_poly)

    protected_set: set[tuple[float, float]] = set()
    if protected is not None:
        protected_set = {(round(p[0], 6), round(p[1], 6)) for p in protected}

    def _is_protected(pt: tuple[float, float]) -> bool:
        return (round(pt[0], 6), round(pt[1], 6)) in protected_set

    pts = [tuple(p) for p in path]

    changed = True
    while changed:
        changed = False
        i = 0
        new_pts: list[tuple[float, float]] = [pts[0]]
        while i < len(pts) - 1:
            best_j = i + 1
            for j in range(i + 2, len(pts)):
                # Never skip over a protected (coverage) waypoint
                if _is_protected(pts[j - 1]) and j - 1 != i:
                    break
                line = LineString([pts[i], pts[j]])
                if eroded_poly.covers(line):
                    best_j = j
            if best_j > i + 1:
                changed = True
            new_pts.append(pts[best_j])
            i = best_j
        pts = new_pts

    return np.array(pts)


def compute_geodesic_distances(
    adj: list[list[tuple[int, float]]],
    n_total: int,
    n_wp: int,
) -> np.ndarray:
    """Pairwise shortest distances between the first *n_wp* nodes."""
    dist_matrix = np.full((n_wp, n_wp), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)

    for src in range(n_wp):
        dists = _dijkstra_all(adj, n_total, src)
        for dst in range(n_wp):
            dist_matrix[src, dst] = dists[dst]

    return dist_matrix


def _extract_vertices_and_edges(
    poly: Polygon | MultiPolygon,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Extract vertices and boundary edges from polygon rings."""
    verts: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []
    geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    offset = 0
    for g in geoms:
        for ring in [g.exterior, *g.interiors]:
            ring_coords = list(ring.coords[:-1])
            n_ring = len(ring_coords)
            verts.extend(ring_coords)
            for k in range(n_ring):
                edges.append((offset + k, offset + (k + 1) % n_ring))
            offset += n_ring
    return verts, edges


def _dijkstra(
    adj: list[list[tuple[int, float]]], n: int, start: int, end: int
) -> list[int] | None:
    """Shortest path from *start* to *end*. Returns node index list or ``None``."""
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[end] == float("inf"):
        return None

    path: list[int] = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _dijkstra_all(
    adj: list[list[tuple[int, float]]], n: int, start: int
) -> list[float]:
    """Single-source shortest distances from *start* to all nodes."""
    dist = [float("inf")] * n
    dist[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist
