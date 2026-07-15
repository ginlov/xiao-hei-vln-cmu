"""Pack the shared :class:`SceneRepresentation` for Gemini consumption.

The scene graph itself is built by
:class:`xiao_hei_vln.scene.SceneRepresentation` (cherry-picked from the
``feat/scene-representation`` branch): a SysNav-style three-level
hierarchy of Room → Viewpoints → Objects, updated tick-by-tick as the
robot explores.

This module's job is the *serialization* layer: turn that scene graph
plus the latest sensor snapshot into a (text, [image_bytes]) bundle the
:class:`GeminiEngine` can pass to the API in one multimodal call.

Two images are produced:

  1. **Current panorama JPEG** (downscaled) — Gemini's primary visual
     context for the current view.
  2. **Occupancy + trajectory PNG** — top-down render of the
     :class:`GlobalMap` overlay with viewpoint markers. Gemini uses
     this to reason about *spatial coverage* — which areas the robot
     has and hasn't observed.

Matplotlib and Pillow imports are deferred so this module stays
importable in environments that don't have the ``gemini`` extra.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from xiao_hei_vln.messages import VLMInput
from xiao_hei_vln.messages.sensors import ImageFrame, OdomPose
from xiao_hei_vln.perception.global_map import OCCUPIED, UNKNOWN, GlobalMap
from xiao_hei_vln.scene import SceneRepresentation


@dataclass(frozen=True)
class GeminiSceneBundle:
    """Multimodal context packet for a single Gemini call.

    Holds the JSON dict produced by :meth:`SceneRepresentation.to_dict`
    plus the rendered images. ``trajectory_xy`` is the recent pose
    history that gets drawn onto the occupancy PNG.
    """

    scene_dict: dict[str, Any]
    trajectory_xy: list[tuple[float, float]]
    occupancy_png_bytes: bytes
    panorama_jpg_bytes: bytes | None


# --- public builders -------------------------------------------------------


def build_bundle(
    snapshot: VLMInput,
    scene: SceneRepresentation,
    global_map: GlobalMap,
    trajectory_xy: list[tuple[float, float]],
    *,
    panorama_long_edge: int = 1280,
    occupancy_dpi: int = 100,
) -> GeminiSceneBundle:
    """Produce a :class:`GeminiSceneBundle` from current responder state.

    ``scene`` should already be ``update()``-ed for this tick before
    calling — the bundle just snapshots it via ``to_dict()``.
    """
    occ_png = render_occupancy_png(
        global_map,
        trajectory_xy=trajectory_xy,
        robot_xy=_pose_xy(snapshot.pose),
        viewpoint_xys=[
            (vp.position.x, vp.position.y) for vp in scene.viewpoints
        ],
        dpi=occupancy_dpi,
    )
    pano_jpg = (
        _panorama_to_jpeg(snapshot.image, long_edge=panorama_long_edge)
        if snapshot.image is not None
        else None
    )
    return GeminiSceneBundle(
        scene_dict=scene.to_dict(),
        trajectory_xy=list(trajectory_xy),
        occupancy_png_bytes=occ_png,
        panorama_jpg_bytes=pano_jpg,
    )


def serialize_for_gemini(bundle: GeminiSceneBundle) -> tuple[str, list[bytes]]:
    """Pack the bundle into the ``(text, [image_bytes])`` shape the engine wants.

    The text block contains a JSON dump of the scene graph plus a brief
    trajectory summary. The image list is ``[panorama, occupancy_map]``
    (panorama omitted if the snapshot had no image).
    """
    scene_text = (
        "Current scene graph (JSON from SceneRepresentation.to_dict):\n"
        f"```json\n{json.dumps(bundle.scene_dict, indent=2)}\n```\n"
        + _trajectory_summary(bundle.trajectory_xy)
    )
    images: list[bytes] = []
    if bundle.panorama_jpg_bytes is not None:
        images.append(bundle.panorama_jpg_bytes)
    images.append(bundle.occupancy_png_bytes)
    return scene_text, images


# --- occupancy PNG renderer ------------------------------------------------


def render_occupancy_png(
    global_map: GlobalMap,
    *,
    trajectory_xy: list[tuple[float, float]] | None = None,
    robot_xy: tuple[float, float] | None = None,
    viewpoint_xys: list[tuple[float, float]] | None = None,
    dpi: int = 100,
) -> bytes:
    """Render the occupancy grid + trajectory + viewpoint markers to PNG.

    Color legend:
      - white       → FREE cell
      - black       → OCCUPIED cell
      - light grey  → UNKNOWN cell
      - blue line   → robot trajectory
      - orange dot  → scene-graph viewpoint node (one per
        :class:`xiao_hei_vln.scene.ViewpointNode`)
      - green dot   → robot's current pose
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if not global_map.is_initialised:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.text(
            0.5, 0.5, "Map not initialised",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    H, W = global_map.shape
    ox, oy = global_map.origin_xy
    res = global_map.resolution_m
    extent = (ox, ox + W * res, oy, oy + H * res)
    cmap = ListedColormap(["#d8d8d8", "#ffffff", "#1a1a1a"])  # UNKNOWN, FREE, OCCUPIED

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(
        global_map.grid,
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=UNKNOWN,
        vmax=OCCUPIED,
    )

    if trajectory_xy:
        xs = [p[0] for p in trajectory_xy]
        ys = [p[1] for p in trajectory_xy]
        ax.plot(xs, ys, "-", color="#1d4ed8", linewidth=1.6, label="trajectory")
        ax.plot(xs[0], ys[0], "s", color="#22c55e", markersize=8, label="start")

    if viewpoint_xys:
        vxs = [v[0] for v in viewpoint_xys]
        vys = [v[1] for v in viewpoint_xys]
        ax.scatter(vxs, vys, c="#f59e0b", s=40, edgecolor="black",
                   zorder=4, label="viewpoint")

    if robot_xy is not None:
        ax.plot(robot_xy[0], robot_xy[1], "o", color="#16a34a",
                markersize=10, label="robot", zorder=5)

    # Crop to the observed area + a small margin.
    known = global_map.grid != UNKNOWN
    if known.any():
        ys_idx, xs_idx = np.where(known)
        x_min = ox + xs_idx.min() * res
        x_max = ox + (xs_idx.max() + 1) * res
        y_min = oy + ys_idx.min() * res
        y_max = oy + (ys_idx.max() + 1) * res
        pad = 0.5
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Occupancy + trajectory + viewpoints")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# --- helpers ---------------------------------------------------------------


def _trajectory_summary(traj: list[tuple[float, float]]) -> str:
    if not traj:
        return "Trajectory: (none yet)"
    if len(traj) <= 8:
        rendered = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj)
    else:
        head = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[:2])
        tail = ", ".join(f"({x:.1f},{y:.1f})" for x, y in traj[-4:])
        rendered = f"{head}, ... ({len(traj) - 6} omitted) ..., {tail}"
    return f"Recent trajectory ({len(traj)} samples): {rendered}"


def _pose_xy(pose: OdomPose | None) -> tuple[float, float] | None:
    if pose is None:
        return None
    p = pose.position
    return (p.x, p.y)


def _panorama_to_jpeg(frame: ImageFrame, *, long_edge: int) -> bytes:
    from xiao_hei_vln.image_utils import image_frame_to_pil, resize_pil
    pil_image = resize_pil(image_frame_to_pil(frame), long_edge=long_edge)
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
