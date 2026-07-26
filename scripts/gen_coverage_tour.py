#!/usr/bin/env python3
"""Build a coverage tour over a scene's traversable area, for wp_driver.py.

The frontier explorer wedges often enough that it cannot be trusted to collect
a recording corpus, and its coverage varies run to run — which confounds any
comparison between scenes. This produces a deterministic route instead: every
reachable part of the floor, visited once, in a fixed order.

Two properties matter for the nav stack, and both come from routing over the
floor graph rather than emitting a straight-line tour:

  * consecutive waypoints stay within one grid step of each other (a jump of
    more than ~1.5 m makes the local planner give up), and
  * the segment between them stays on traversable floor, so the robot is never
    commanded through a wall.

    scripts/gen_coverage_tour.py ~/Downloads/unity_env_models/loft \\
        --out tours/loft.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

FINE_CELL_M = 0.25  # rasterisation of the floor; also the clearance resolution


def read_ply_xy(path: Path) -> np.ndarray:
    """The x/y of an ascii PLY's vertices. traversable_area.ply is planar."""
    with path.open("r", errors="replace") as fh:
        n = 0
        for line in fh:
            line = line.strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line == "end_header":
                break
        else:
            raise ValueError(f"{path}: no end_header")
        rows = [fh.readline().split()[:2] for _ in range(n)]
    return np.asarray(rows, dtype=float)


def _erode(mask: np.ndarray, steps: int) -> np.ndarray:
    """Shrink a boolean mask by `steps` cells in the 4-connected sense.

    Keeps waypoints off the walls: the robot has width, and a waypoint the
    planner cannot physically occupy burns the per-waypoint timeout.
    """
    out = mask
    for _ in range(max(steps, 0)):
        p = np.pad(out, 1, constant_values=False)
        out = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return out


def _segment_clear(free: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Does the straight line between two fine cells stay on free floor?"""
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    for i in range(n + 1):
        t = i / max(n, 1)
        r = int(round(a[0] + (b[0] - a[0]) * t))
        c = int(round(a[1] + (b[1] - a[1]) * t))
        if not free[r, c]:
            return False
    return True


def build_tour(
    xy: np.ndarray,
    *,
    spacing: float,
    clearance: float,
    max_hop: float,
    start: tuple[float, float],
) -> tuple[list[tuple[float, float]], dict]:
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    shape = (
        int(math.ceil((hi[1] - lo[1]) / FINE_CELL_M)) + 1,
        int(math.ceil((hi[0] - lo[0]) / FINE_CELL_M)) + 1,
    )
    free = np.zeros(shape, dtype=bool)
    idx = np.floor((xy - lo) / FINE_CELL_M).astype(int)
    free[idx[:, 1], idx[:, 0]] = True
    # A lidar-derived floor has pinholes where furniture legs occluded it;
    # close them before eroding or the clearance test eats the whole room.
    p = np.pad(free, 1, constant_values=False)
    dilated = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    free = _erode(dilated, 1)
    safe = _erode(free, int(round(clearance / FINE_CELL_M)))

    # Sampling on a fixed lattice loses whole rooms when the lattice lands on
    # furniture, so place nodes greedily instead: every safe cell is either a
    # node or within `spacing` of one.
    step = spacing / FINE_CELL_M
    cand = np.argwhere(safe)
    if cand.size == 0:
        raise SystemExit("no traversable cell survives the clearance margin")
    nodes: list[tuple[int, int]] = []
    taken = np.zeros(shape, dtype=bool)
    for r, c in cand:
        if taken[r, c]:
            continue
        nodes.append((int(r), int(c)))
        r0, r1 = max(int(r - step), 0), int(r + step) + 1
        c0, c1 = max(int(c - step), 0), int(c + step) + 1
        taken[r0:r1, c0:c1] = True

    # Connect over `free`, not `safe`: a doorway is often narrower than the
    # clearance margin, and refusing to route through it strands whole rooms.
    reach = max_hop / FINE_CELL_M
    adj: list[list[int]] = [[] for _ in nodes]
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if math.dist(a, b) > reach or not _segment_clear(free, a, b):
                continue
            adj[i].append(j)
            adj[j].append(i)

    start_rc = (
        int(round((start[1] - lo[1]) / FINE_CELL_M)),
        int(round((start[0] - lo[0]) / FINE_CELL_M)),
    )
    first = min(
        range(len(nodes)),
        key=lambda i: (nodes[i][0] - start_rc[0]) ** 2 + (nodes[i][1] - start_rc[1]) ** 2,
    )

    # Greedy nearest-unvisited, but "nearest" is measured along the floor and
    # the whole path is emitted — that is what keeps every commanded step short.
    visited = [False] * len(nodes)
    order = [first]
    visited[first] = True
    cur = first
    while True:
        prev = {cur: -1}
        q = deque([cur])
        target = -1
        while q:
            u = q.popleft()
            if not visited[u]:
                target = u
                break
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if target < 0:
            break
        path = []
        u = target
        while u != -1:
            path.append(u)
            u = prev[u]
        for u in reversed(path[:-1]):
            order.append(u)
            visited[u] = True
        cur = target

    pts = [
        (
            float(lo[0] + (nodes[i][1] + 0.5) * FINE_CELL_M),
            float(lo[1] + (nodes[i][0] + 0.5) * FINE_CELL_M),
        )
        for i in order
    ]
    hops = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    stats = {
        "traversable_m2": round(float(free.sum()) * FINE_CELL_M**2, 1),
        "nodes": len(nodes),
        "reached": sum(visited),
        "waypoints": len(pts),
        "max_hop_m": round(max(hops), 2) if hops else 0.0,
        "path_len_m": round(sum(hops), 1),
    }
    return pts, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path, help="unity_env_models/<scene>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spacing", type=float, default=0.8,
                    help="how far apart viewpoints are placed, in m; below the "
                         "hop cap, because sparser nodes fragment the floor "
                         "graph and strand parts of the scene")
    ap.add_argument("--clearance", type=float, default=0.25,
                    help="keep waypoints this far from untraversable cells; "
                         "traversable_area.ply is already the navigable region, "
                         "so this is a safety margin, not the robot's radius")
    ap.add_argument("--max-hop", type=float, default=1.6,
                    help="longest step the local planner is asked to take; much "
                         "beyond this it gives up and the waypoint burns its timeout")
    ap.add_argument("--start", default="0,0", help="robot spawn 'x,y' in map frame")
    args = ap.parse_args()

    ply = args.scene_dir / "traversable_area.ply"
    if not ply.is_file():
        raise SystemExit(f"missing {ply}")
    sx, sy = (float(v) for v in args.start.split(","))
    pts, stats = build_tour(
        read_ply_xy(ply),
        spacing=args.spacing,
        clearance=args.clearance,
        max_hop=args.max_hop,
        start=(sx, sy),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "scene": args.scene_dir.name,
        "spacing_m": args.spacing,
        "clearance_m": args.clearance,
        "max_hop_m": args.max_hop,
        "stats": stats,
        "waypoints": [{"x": round(x, 3), "y": round(y, 3)} for x, y in pts],
    }, indent=1))
    print(f"{args.scene_dir.name}: {stats}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
