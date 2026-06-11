"""Offline visualisation for the exploration module.

Produces a single PNG showing the accumulated traversable area and the
path the robot followed during the exploration phase.  Call this after
exploration is complete (e.g. when FrontierExplorer.is_complete() turns
True) to get a plot for visual debugging.
"""

from __future__ import annotations

import math
from pathlib import Path

from xiao_hei_vln.exploration._grid import OccupancyGrid
from xiao_hei_vln.messages.outputs import Waypoint


def save_exploration_plot(
    visited_waypoints: list[Waypoint],
    grid: OccupancyGrid,
    output_path: Path | str,
    title: str = "Exploration Path",
) -> None:
    """Save a debug PNG of the explored map and robot path.

    Parameters
    ----------
    visited_waypoints:
        Ordered list of waypoints the robot visited (from
        ``FrontierExplorer.get_visited_waypoints()``).
    grid:
        The OccupancyGrid accumulated during exploration.
    output_path:
        Destination file (PNG).  Parent directory must exist.
    title:
        Plot title prefix.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    res = grid.resolution
    free = list(grid.free_cells)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Explored (free) cells as a grey point cloud
    if free:
        half = res * 0.5
        fx = [ix * res + half for ix, iy in free]
        fy = [iy * res + half for ix, iy in free]
        ax.scatter(fx, fy, s=0.5, c="lightgray", alpha=0.6, label="Explored area", rasterized=True)

    # Robot path and waypoints
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

        # Heading arrows at each waypoint
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


def _path_length(waypoints: list[Waypoint]) -> float:
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i].x - waypoints[i - 1].x
        dy = waypoints[i].y - waypoints[i - 1].y
        total += (dx * dx + dy * dy) ** 0.5
    return total
