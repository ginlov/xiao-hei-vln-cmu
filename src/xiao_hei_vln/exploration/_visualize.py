"""Offline visualisation for the exploration module.

Produces a single PNG showing the accumulated traversable area and the
path the robot followed during the exploration phase.  Call this after
exploration is complete (e.g. when FrontierExplorer.is_complete() turns
True) to get a plot for visual debugging.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.outputs import Waypoint


def save_exploration_plot(
    visited_waypoints: list[Waypoint],
    grid: OccupancyGrid,
    output_path: Path | str,
    title: str = "Exploration Path",
) -> None:
    """Save a debug PNG of the explored map and robot path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    res = grid.resolution
    free = list(grid.free_cells)

    fig, ax = plt.subplots(figsize=(12, 10))

    if free:
        half = res * 0.5
        fx = [ix * res + half for ix, iy in free]
        fy = [iy * res + half for ix, iy in free]
        ax.scatter(fx, fy, s=0.5, c="lightgray", alpha=0.6, label="Explored area", rasterized=True)

    if visited_waypoints:
        wx = [w.x for w in visited_waypoints]
        wy = [w.y for w in visited_waypoints]
        ax.plot(wx, wy, "-", color="royalblue", linewidth=1.5, alpha=0.8, zorder=5,
                label="Path")
        ax.scatter(wx, wy, c="royalblue", s=30, zorder=6, label="Waypoint")
        ax.plot(wx[0], wy[0], "s", color="limegreen", markersize=10, zorder=7,
                label="Start")
        ax.plot(wx[-1], wy[-1], "D", color="orangered", markersize=8, zorder=7,
                label="End")

        for w in visited_waypoints:
            ax.annotate(
                "",
                xy=(w.x + 0.3 * math.cos(w.heading), w.y + 0.3 * math.sin(w.heading)),
                xytext=(w.x, w.y),
                arrowprops={"arrowstyle": "->", "color": "steelblue", "lw": 0.8},
                zorder=7,
            )

    n_free = len(free)
    n_wps = len(visited_waypoints)
    path_m = _path_length(visited_waypoints)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{title}\n"
        f"Waypoints={n_wps}  PathLength={path_m:.1f} m  FreeCells={n_free}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_rviz_style_plot(
    layers: dict[str, Any],
    output_path: Path | str,
    *,
    title: str = "Exploration (RViz-style)",
    gt_free_xy: list[tuple[float, float]] | None = None,
    score_text: str = "",
) -> None:
    """Save a PNG matching the live ``/exploration/markers`` color legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    if gt_free_xy:
        gx = [p[0] for p in gt_free_xy]
        gy = [p[1] for p in gt_free_xy]
        ax.scatter(
            gx, gy, s=1.0, c="#e8e8e8", alpha=0.5, label="GT free",
            rasterized=True, zorder=1,
        )

    def _scatter(key: str, color: str, label: str, size: float = 4.0, alpha: float = 0.85, z: int = 3):
        pts = layers.get(key) or []
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=size, c=color, alpha=alpha, label=label, rasterized=True, zorder=z)

    _scatter("free", "#bfbfbf", "free (explored)", size=2.5, alpha=0.45, z=2)
    _scatter("frontier", "#1ad1ff", "frontier", size=8.0, z=4)
    _scatter("soft_ban", "#ffd91a", "soft-ban", size=10.0, z=5)
    _scatter("hard_ban", "#ff261a", "hard-ban", size=10.0, z=5)

    visited = layers.get("visited") or []
    if visited:
        wx = [p[0] for p in visited]
        wy = [p[1] for p in visited]
        ax.plot(wx, wy, "-", color="#3366ff", linewidth=1.8, alpha=0.9, zorder=6, label="path")
        ax.scatter(wx, wy, c="#33e64d", s=40, zorder=7, label="visited")
        ax.plot(wx[0], wy[0], "s", color="limegreen", markersize=10, zorder=8)
        ax.plot(wx[-1], wy[-1], "D", color="orangered", markersize=8, zorder=8)

    current = layers.get("current")
    if current is not None:
        ax.scatter(
            [current[0]], [current[1]], c="#f233f2", s=120,
            marker="*", zorder=9, label="goal",
        )

    subtitle = score_text.strip()
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}" + (f"\n{subtitle}" if subtitle else ""))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def layers_from_explorer(explorer: Any) -> dict[str, Any]:
    """Build RViz-style layers from an explorer (viz API or grid fallback)."""
    if hasattr(explorer, "get_viz_layers"):
        return explorer.get_viz_layers()

    grid = explorer.get_grid()
    visited = explorer.get_visited_waypoints() if hasattr(explorer, "get_visited_waypoints") else []
    vis_xy = [(w.x, w.y) for w in visited]
    current = None
    if hasattr(explorer, "_current_target") and explorer._current_target is not None:
        current = (explorer._current_target.x, explorer._current_target.y)
    return {
        "free": [grid.to_world(ix, iy) for ix, iy in grid.free_cells],
        "frontier": [grid.to_world(ix, iy) for ix, iy in grid.frontier_cells()],
        "soft_ban": [],
        "hard_ban": [],
        "visited": vis_xy,
        "current": current,
        "resolution": grid.resolution,
    }


def _path_length(waypoints: list[Waypoint]) -> float:
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        total += (dx * dx + dy * dy) ** 0.5
    return total
