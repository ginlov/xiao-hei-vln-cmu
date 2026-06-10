"""Synthetic scene → end-to-end run of the Phase A exploration loop → PNG.

This is a *visual* sanity check for ``PerceptionResponder``. It builds a
simulated 8 × 6 m living room, ray-casts a synthetic ``terrain_ext``
snapshot from the robot's pose each tick, feeds those snapshots into the
responder, and uses each emitted waypoint to step the robot forward —
exactly the loop that ``app/main.py`` runs against the real simulator,
minus ROS.

Run:

    .venv/bin/python scripts/visualize_explore_trajectory.py \\
        --ticks 200 --output /tmp/explore_trajectory.png

The output PNG has three panels:
    1. Ground-truth room + robot trajectory
    2. The responder's accumulated GlobalMap view + same trajectory
    3. Final-tick frontier clusters highlighted on the GlobalMap
"""

from __future__ import annotations

import argparse
import heapq
import math
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    OdomPose,
    Quaternion,
    Stamp,
    VLMInput,
    Vector3,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import TerrainMap
from xiao_hei_vln.perception.global_map import FREE, OCCUPIED, UNKNOWN
from xiao_hei_vln.perception_responder import PerceptionResponder


# ─── Scene definition ───────────────────────────────────────────────────

@dataclass(frozen=True)
class Room:
    """Ground-truth occupancy of the synthetic living room.

    `occ[i, j] == True` means the cell is an obstacle. `i` is the y axis,
    `j` is the x axis. World extent is `(width_m, height_m)` and
    `resolution_m` ties grid cells to world coordinates.
    """

    occ: np.ndarray
    width_m: float
    height_m: float
    resolution_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.occ.shape

    def world_to_idx(self, x: float, y: float) -> tuple[int, int]:
        return int(y / self.resolution_m), int(x / self.resolution_m)

    def is_obstacle(self, x: float, y: float) -> bool:
        i, j = self.world_to_idx(x, y)
        if not (0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]):
            return True  # outside the room → treat as a wall
        return bool(self.occ[i, j])


def build_living_room() -> Room:
    """A 8 × 6 m room with a sofa, coffee table, TV stand, and bookshelf.

    Grid resolution is 0.1 m → 80 × 60 cells. The layout is the same one
    used in the design conversations: furniture creates occluded zones
    behind which frontier exploration alone leaves blind spots.
    """
    res = 0.1
    W = int(8.0 / res)  # 80 cells wide  → x ∈ [0, 8] m
    H = int(6.0 / res)  # 60 cells high → y ∈ [0, 6] m
    occ = np.zeros((H, W), dtype=bool)

    # Walls (1 cell thick on all four sides)
    occ[0, :] = True
    occ[-1, :] = True
    occ[:, 0] = True
    occ[:, -1] = True

    # Sofa, lower-left of room (occupies x∈[1.5, 4.0], y∈[1.5, 2.2])
    occ[15:22, 15:40] = True
    # Coffee table, just above the sofa (x∈[2.5, 3.8], y∈[3.0, 3.6])
    occ[30:36, 25:38] = True
    # TV stand, right wall (x∈[6.5, 7.5], y∈[1.0, 1.7])
    occ[10:17, 65:75] = True
    # Bookshelf, top-right (x∈[6.0, 6.5], y∈[3.5, 5.5])
    occ[35:55, 60:65] = True
    return Room(occ=occ, width_m=8.0, height_m=6.0, resolution_m=res)


# ─── terrain_ext synthesis ──────────────────────────────────────────────

def plan_path(
    room: Room,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]] | None:
    """A* through the ground-truth FREE cells. Stands in for the system's nav stack.

    Returns a list of world (x, y) waypoints from start (exclusive) to goal,
    or ``None`` if goal is unreachable.
    """
    si, sj = room.world_to_idx(*start_xy)
    gi, gj = room.world_to_idx(*goal_xy)
    H, W = room.occ.shape

    def in_bounds(i: int, j: int) -> bool:
        return 0 <= i < H and 0 <= j < W

    if not in_bounds(si, sj) or room.occ[si, sj]:
        return None
    if not in_bounds(gi, gj) or room.occ[gi, gj]:
        # Goal is in an obstacle — snap to the nearest FREE cell so we still
        # produce a reasonable plan toward the frontier.
        candidates = np.argwhere(~room.occ)
        if candidates.size == 0:
            return None
        diffs = candidates - np.array([gi, gj])
        idx = int(np.argmin((diffs ** 2).sum(axis=1)))
        gi, gj = int(candidates[idx, 0]), int(candidates[idx, 1])

    open_heap: list[tuple[float, int, int]] = [(0.0, si, sj)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    gscore: dict[tuple[int, int], float] = {(si, sj): 0.0}

    while open_heap:
        _, ci, cj = heapq.heappop(open_heap)
        if (ci, cj) == (gi, gj):
            # Reconstruct path
            path_cells = [(ci, cj)]
            while path_cells[-1] in came_from:
                path_cells.append(came_from[path_cells[-1]])
            path_cells.reverse()
            return [
                (
                    (j + 0.5) * room.resolution_m,
                    (i + 0.5) * room.resolution_m,
                )
                for i, j in path_cells[1:]
            ]
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if not in_bounds(ni, nj) or room.occ[ni, nj]:
                continue
            tentative = gscore[(ci, cj)] + 1.0
            if tentative < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = tentative
                came_from[(ni, nj)] = (ci, cj)
                heuristic = math.hypot(ni - gi, nj - gj)
                heapq.heappush(open_heap, (tentative + heuristic, ni, nj))
    return None


def simulate_terrain_ext(
    robot_xy: tuple[float, float],
    room: Room,
    *,
    max_range_m: float = 20.0,
    n_rays: int = 360,
    sample_step_m: float = 0.2,
) -> np.ndarray:
    """Cheap LiDAR-style ray-cast that returns a (N, 4) point cloud.

    For each ray:
      * walk along it in `sample_step_m` increments
      * emit a FREE point (cost=0) at each free step
      * emit an OCCUPIED point (cost=1) at the first hit, then stop

    The 4th column is the "cost" field consumed by ``GlobalMap.update``.
    """
    rx, ry = robot_xy
    angles = np.linspace(0.0, 2 * math.pi, n_rays, endpoint=False)
    pts: list[tuple[float, float, float, float]] = []

    for a in angles:
        dx = math.cos(a)
        dy = math.sin(a)
        t = sample_step_m
        while t <= max_range_m:
            x = rx + t * dx
            y = ry + t * dy
            if room.is_obstacle(x, y):
                pts.append((x, y, 0.0, 1.0))
                break
            pts.append((x, y, 0.0, 0.0))
            t += sample_step_m

    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


# ─── driver ─────────────────────────────────────────────────────────────

def run_exploration(
    room: Room,
    *,
    start_xy: tuple[float, float] = (1.0, 1.0),
    max_ticks: int = 200,
    step_size_m: float = 0.25,
    waypoint_reach_m: float = 0.3,
) -> dict:
    """Drive ``PerceptionResponder`` for ``max_ticks`` and return a record.

    The robot moves in straight lines toward each emitted waypoint, up to
    ``step_size_m`` per tick. This is a simple stand-in for the challenge's
    low-level navigation stack.
    """
    responder = PerceptionResponder(min_cluster_size=3)

    rx, ry = start_xy
    trajectory: list[tuple[float, float]] = [(rx, ry)]
    waypoints: list[tuple[float, float]] = []
    standing_still_ticks = 0

    for tick in range(max_ticks):
        # If the robot has been "standing still" for several ticks in a row,
        # exploration is exhausted in this synthetic scene → stop early.
        if standing_still_ticks > 5:
            break

        stamp = Stamp.from_seconds(tick * 0.5)
        pose = OdomPose(
            header=Header(stamp=stamp, frame_id="map"),
            position=Vector3(x=rx, y=ry, z=0.5),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        terrain = TerrainMap(
            header=pose.header,
            points=simulate_terrain_ext((rx, ry), room),
            range="ext_20m",
        )
        question = ChallengeQuestion.from_text("How many cups", stamp)
        snap = VLMInput(
            tick_id=tick,
            tick_time=stamp,
            pose=pose,
            terrain_ext=terrain,
            question=question,
        )

        out = responder.respond(snap)
        if not isinstance(out, WaypointPathResponse):
            continue
        wp = out.waypoints[0]
        waypoints.append((wp.x, wp.y))

        # Plan an obstacle-aware path to the waypoint via A* on the ground-truth
        # map — this stands in for the challenge's low-level nav stack, which
        # the responder relies on to route around furniture. Without it the
        # naive straight-line driver walks through walls.
        path = plan_path(room, (rx, ry), (wp.x, wp.y))
        if not path:
            standing_still_ticks += 1
            trajectory.append((rx, ry))
            continue

        # Walk along the planned path up to step_size_m this tick.
        budget = step_size_m
        for tx, ty in path:
            seg = math.hypot(tx - rx, ty - ry)
            if seg <= budget:
                rx, ry = tx, ty
                budget -= seg
            else:
                # Step partway along the segment and stop.
                rx += budget * (tx - rx) / seg
                ry += budget * (ty - ry) / seg
                budget = 0.0
                break
            if budget <= 0.0:
                break

        if math.hypot(wp.x - rx, wp.y - ry) < waypoint_reach_m:
            standing_still_ticks += 1
        else:
            standing_still_ticks = 0
        trajectory.append((rx, ry))

    return {
        "responder": responder,
        "trajectory": np.array(trajectory),
        "waypoints": np.array(waypoints) if waypoints else np.zeros((0, 2)),
        "final_tick": tick,
    }


# ─── plotting ───────────────────────────────────────────────────────────

def _extract_obstacle_rects(room: Room) -> list[tuple[float, float, float, float]]:
    """Find axis-aligned rectangles of contiguous OCCUPIED cells in the room.

    Returns a list of ``(x0, y0, x1, y1)`` rectangles in world coordinates,
    used to draw furniture as "holes" in the polygon (PR #10 style).
    """
    occ = room.occ
    res = room.resolution_m
    visited = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[float, float, float, float]] = []
    H, W = occ.shape

    for i in range(H):
        for j in range(W):
            if not occ[i, j] or visited[i, j]:
                continue
            # Skip the room's own wall border — those form the outer polygon,
            # not "holes". Detect by checking whether the cell sits on the
            # outer ring of the grid.
            if i == 0 or i == H - 1 or j == 0 or j == W - 1:
                visited[i, j] = True
                continue
            # Greedy rectangle expansion right then down.
            j_end = j
            while j_end + 1 < W and occ[i, j_end + 1] and not visited[i, j_end + 1] \
                    and not (j_end + 1 == W - 1):
                j_end += 1
            i_end = i
            while i_end + 1 < H:
                row_ok = all(
                    occ[i_end + 1, k] and not visited[i_end + 1, k]
                    and not (i_end + 1 == H - 1)
                    for k in range(j, j_end + 1)
                )
                if not row_ok:
                    break
                i_end += 1
            visited[i:i_end + 1, j:j_end + 1] = True
            rects.append((j * res, i * res, (j_end + 1) * res, (i_end + 1) * res))
    return rects


def render(room: Room, record: dict, output_path: str) -> None:
    responder: PerceptionResponder = record["responder"]
    traj: np.ndarray = record["trajectory"]
    waypoints: np.ndarray = record["waypoints"]
    final_tick = record["final_tick"]

    gm = responder._map  # noqa: SLF001 — intentional internal access for visualisation
    if gm.origin_xy is None:
        raise RuntimeError("GlobalMap was never initialised; nothing to plot")

    # Stats for the title line and the printed summary.
    n_free = int((gm.grid == FREE).sum())
    n_occ = int((gm.grid == OCCUPIED).sum())
    n_unk = int((gm.grid == UNKNOWN).sum())
    n_known = n_free + n_occ
    clusters = gm.find_frontier_clusters(min_cluster_size=3)

    # Path length: sum of consecutive-segment lengths in the recorded trajectory.
    seg_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    path_length_m = float(seg_lengths.sum())

    matplotlib.use("Agg")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # ── Traversable area: sparse sample of FREE cells from the responder's view
    free_ij = np.argwhere(gm.grid == FREE)
    if free_ij.size > 0:
        ox, oy = gm.origin_xy
        free_xy = np.column_stack([
            ox + (free_ij[:, 1] + 0.5) * gm.resolution_m,
            oy + (free_ij[:, 0] + 0.5) * gm.resolution_m,
        ])
        ax.scatter(
            free_xy[::3, 0], free_xy[::3, 1],
            s=0.3, c="lightgray", alpha=0.6, label="Traversable area (observed)",
        )

    # ── Outer polygon boundary: the room walls
    boundary_x = [0.0, room.width_m, room.width_m, 0.0, 0.0]
    boundary_y = [0.0, 0.0, room.height_m, room.height_m, 0.0]
    ax.plot(boundary_x, boundary_y, "b-", linewidth=0.8, label="Polygon boundary")

    # ── Furniture as polygon "holes" — salmon fill, red border
    hole_label_used = False
    for (x0, y0, x1, y1) in _extract_obstacle_rects(room):
        poly = MplPolygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            closed=True, facecolor="salmon", alpha=0.5,
            edgecolor="red", linewidth=0.6,
            label=None if hole_label_used else "Furniture hole",
        )
        ax.add_patch(poly)
        hole_label_used = True

    # ── Remaining frontier clusters — red ×, analogous to PR #10 "uncovered objects"
    if clusters:
        cx = [c.centroid_xy[0] for c in clusters]
        cy = [c.centroid_xy[1] for c in clusters]
        ax.plot(
            cx, cy, "x", color="red", markersize=8, alpha=0.9,
            label=f"Remaining frontier ({len(clusters)})",
        )

    # ── Trajectory path + waypoints (PR #10 colour palette)
    if traj.size > 0:
        ax.plot(
            traj[:, 0], traj[:, 1],
            "-", color="royalblue", linewidth=1.5, alpha=0.7,
            zorder=5, label="Trajectory path",
        )
        if waypoints.size > 0:
            ax.scatter(
                waypoints[:, 0], waypoints[:, 1],
                c="royalblue", s=20, zorder=6, label="Waypoint emitted",
            )
        ax.plot(
            traj[0, 0], traj[0, 1], "s", color="lime", markersize=10,
            zorder=7, label="Start",
        )
        ax.plot(
            traj[-1, 0], traj[-1, 1], "D", color="orangered", markersize=8,
            zorder=7, label="End",
        )

    # ── Cosmetics matching PR #10
    cells_explored_pct = 100.0 * n_known / max(n_known + n_unk, 1)
    title_main = "Phase A frontier exploration — synthetic living room"
    title_stats = (
        f"Ticks={final_tick + 1} "
        f"FreeCells={n_free} "
        f"FrontiersLeft={len(clusters)} "
        f"PathLen={path_length_m:.1f}m"
    )
    ax.set_title(f"{title_main}\n{title_stats}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-0.5, room.width_m + 0.5)
    ax.set_ylim(-0.5, room.height_m + 0.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    # Stats summary so the script doubles as a smoke test.
    print(
        f"  ticks ran         = {final_tick + 1}\n"
        f"  cells FREE        = {n_free}\n"
        f"  cells OCCUPIED    = {n_occ}\n"
        f"  cells UNKNOWN     = {n_unk}\n"
        f"  cells explored    = {cells_explored_pct:.1f}% of observable area\n"
        f"  waypoints emitted = {len(waypoints)}\n"
        f"  trajectory length = {path_length_m:.1f} m"
    )


# ─── entry ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ticks", type=int, default=200, help="Max ticks to simulate.")
    p.add_argument("--start-x", type=float, default=1.0)
    p.add_argument("--start-y", type=float, default=1.0)
    p.add_argument(
        "--output", type=str, default="/tmp/explore_trajectory.png",
        help="Output PNG path.",
    )
    args = p.parse_args()

    room = build_living_room()
    record = run_exploration(
        room,
        start_xy=(args.start_x, args.start_y),
        max_ticks=args.ticks,
    )
    render(room, record, args.output)


if __name__ == "__main__":
    main()
