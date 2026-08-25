"""Render the exploration occupancy grid to a top-down PNG for the VLM.

Unlike :mod:`xiao_hei_vln.gemini.scene_rep` (which renders the perception
``GlobalMap``), this works directly off the exploration
:class:`~xiao_hei_vln.exploration._grid.OccupancyGrid`, so the proposer is
self-contained and runs on the same belief map the geometric explorers use —
including in the dummy/capture path where no perception map exists.

Matplotlib/Pillow are imported lazily so the package stays importable without
the ``nav-vlm`` extra.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.exploration._grid import OccupancyGrid
    from xiao_hei_vln.messages.sensors import ImageFrame

# Cell codes for the rendered array.
_UNKNOWN, _FREE, _OCCUPIED = 0, 1, 2


def render_grid_png(
    grid: OccupancyGrid,
    *,
    robot_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    failed_xy: tuple[float, float] | None = None,
    failed_points: list[tuple[float, float]] | None = None,
    trajectory_xy: list[tuple[float, float]] | None = None,
    objects: list[tuple[float, float, str]] | None = None,
    dpi: int = 100,
) -> bytes:
    """Top-down PNG: free/occupied/unknown cells + robot + trajectory.

    ``failed_xy`` (drawn as a red star) marks the waypoint that just failed,
    so the model can visibly avoid re-proposing it. ``objects`` is a list of
    ``(x, y, label)`` detected scene-graph objects, drawn as labelled orange
    markers so the model can route toward a target it can SEE on the map
    rather than correlating a bare text coordinate with the grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    free = grid.free_cells
    # Occupied cells aren't exposed as a public property; fall back gracefully.
    occupied = getattr(grid, "_occupied", set())
    known = set(free) | set(occupied)

    if not known:
        return _placeholder_png("Map not initialised", dpi=dpi)

    res = grid.resolution
    ixs = [c[0] for c in known]
    iys = [c[1] for c in known]
    ix_min, ix_max = min(ixs), max(ixs)
    iy_min, iy_max = min(iys), max(iys)
    W = ix_max - ix_min + 1
    H = iy_max - iy_min + 1

    arr = np.full((H, W), _UNKNOWN, dtype=np.uint8)
    for (ix, iy) in occupied:
        arr[iy - iy_min, ix - ix_min] = _OCCUPIED
    for (ix, iy) in free:  # free wins over occupied on overlap (confirmed traversable)
        arr[iy - iy_min, ix - ix_min] = _FREE

    # World extent of the rendered window (cell origin is a corner; grids use
    # centre = ix*res + res/2, so the corner is ix*res).
    extent = (
        ix_min * res,
        (ix_max + 1) * res,
        iy_min * res,
        (iy_max + 1) * res,
    )
    cmap = ListedColormap(["#d8d8d8", "#ffffff", "#1a1a1a"])  # UNKNOWN, FREE, OCCUPIED

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(arr, cmap=cmap, origin="lower", extent=extent, vmin=_UNKNOWN, vmax=_OCCUPIED)

    if trajectory_xy:
        xs = [p[0] for p in trajectory_xy]
        ys = [p[1] for p in trajectory_xy]
        ax.plot(xs, ys, "-", color="#1d4ed8", linewidth=1.6, label="path")

    if objects:
        ox = [o[0] for o in objects]
        oy = [o[1] for o in objects]
        ax.scatter(ox, oy, marker="s", s=42, color="#f59e0b",
                   edgecolor="#7c4a03", linewidths=0.5, zorder=4,
                   label="detected object")
        for x, y, label in objects:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3),
                        fontsize=6, color="#7c4a03", zorder=4)

    # All recently-rejected waypoints as red stars — the blocked frontier the
    # model must route AROUND, not just the single latest one.
    fails = list(failed_points) if failed_points else []
    if failed_xy is not None and tuple(failed_xy) not in fails:
        fails.append(failed_xy)
    if fails:
        ax.scatter([p[0] for p in fails], [p[1] for p in fails], marker="*",
                   s=240, color="#dc2626", zorder=6, label="unreachable (avoid)")

    if target_xy is not None:
        ax.plot(target_xy[0], target_xy[1], "P", color="#7c3aed",
                markersize=12, label="current target", zorder=5)

    if robot_xy is not None:
        ax.plot(robot_xy[0], robot_xy[1], "o", color="#16a34a",
                markersize=11, label="robot", zorder=7)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Occupancy (white=free, black=wall, grey=unknown)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def panorama_to_jpeg(frame: ImageFrame, *, long_edge: int) -> bytes:
    """Downscale + JPEG-encode the current panorama for the API."""
    from xiao_hei_vln.image_utils import image_frame_to_pil, resize_pil

    pil = resize_pil(image_frame_to_pil(frame), long_edge=long_edge)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def trajectory_summary(traj: list[tuple[float, float]]) -> str:
    """One-line textual recap of the recent path (mirrors scene_rep style)."""
    if not traj:
        return "Trajectory: (none yet)."
    if len(traj) <= 8:
        rendered = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj)
    else:
        head = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[:2])
        tail = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[-4:])
        rendered = f"{head}, ... ({len(traj) - 6} omitted) ..., {tail}"
    return f"Recent trajectory ({len(traj)} samples): {rendered}."


def _placeholder_png(text: str, *, dpi: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
