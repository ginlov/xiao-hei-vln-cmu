"""Render SceneRepresentation snapshots to base64 PNGs + HTML tables.

Three pure functions consumed by ``scripts/generate_report.py``:

- :func:`render_topdown_png` — 2D top-down spatial view of the graph.
- :func:`render_graph_png`   — hierarchical scene-graph topology
  (Room → Viewpoints (left-to-right by tick_id) → Objects grouped
  under their first-observing viewpoint, with Object↔Object "near"
  arcs across the bottom).
- :func:`render_node_tables` — HTML tables for Room / Viewpoints /
  Objects, suitable for inline embedding.

All three accept a ``scene_dict`` exactly as returned by
:meth:`SceneRepresentation.to_dict`. The PNG functions return
base64-encoded payloads ready to drop into ``data:image/png;base64,…``
URLs. No file I/O.

The graph layout is **deterministic and stable across frames** — new
viewpoints append to the right, and objects sit under the viewpoint
that first observed them, so nothing jitters between ticks.
"""

from __future__ import annotations

import base64
import io
import math
from html import escape
from typing import Any

# matplotlib is provided by the `replay` extra; importing at top-level is fine
# because this module is only loaded by scripts/generate_report.py which
# already requires that extra.
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Top-down spatial view
# ---------------------------------------------------------------------------


def render_topdown_png(
    scene_dict: dict[str, Any],
    *,
    current_pose: dict[str, float] | None = None,
    pose_history: list[tuple[float, float]] | None = None,
    planned_waypoints: list[dict[str, float]] | None = None,
    label_radius: float = 3.0,
    near_edge_radius: float = 3.0,
    view_bounds: tuple[float, float, float, float] | None = None,
) -> str:
    """Render the spatial view (positions, paths, edges) as a base64 PNG.

    ``view_bounds`` is an optional ``(xmin, xmax, ymin, ymax)`` window.
    When given, the axes are pinned to it (and the image is saved at a
    fixed size) so the plotted area stays **stationary across frames** —
    pass the same session-wide bounds to every tick to stop the view
    from zooming/panning as objects and viewpoints accumulate.
    """
    fig, ax = plt.subplots(figsize=(8, 8) if view_bounds else (9, 7))

    # Planned coverage path (faint reference line + dots).
    if planned_waypoints:
        xs = [w["x"] for w in planned_waypoints]
        ys = [w["y"] for w in planned_waypoints]
        ax.plot(xs, ys, "-", color="#bbbbbb", linewidth=1.0, zorder=1)
        ax.scatter(xs, ys, s=20, c="#999999", marker="o", zorder=2,
                   label=f"planned waypoints ({len(planned_waypoints)})")

    # Scene bounds rectangle (dashed).
    sb = scene_dict.get("room", {}).get("scene_bounds")
    if sb is not None:
        (mnx, mny, _), (mxx, mxy, _) = sb
        ax.add_patch(Rectangle(
            (mnx, mny), mxx - mnx, mxy - mny,
            fill=False, edgecolor="#888888", linestyle="--",
            linewidth=1.0, zorder=2, label="scene_bounds",
        ))

    # Object nodes.
    objs = scene_dict.get("objects", [])
    if objs:
        ox = [o["position"][0] for o in objs]
        oy = [o["position"][1] for o in objs]
        sizes = [20 + 40 * o.get("confidence", 1.0) for o in objs]
        ax.scatter(ox, oy, s=sizes, c="#2e7d32", marker="o", alpha=0.7,
                   edgecolors="white", linewidths=0.5, zorder=4,
                   label=f"objects ({len(objs)})")

    # Local object labels — only when we have a pose to anchor on.
    # Object↔Object "near" edges intentionally not drawn.
    if current_pose is not None:
        px = float(current_pose["x"])
        py = float(current_pose["y"])
        _draw_local_labels(ax, objs, px, py, label_radius)

    # Viewpoint nodes (numbered).
    vps = scene_dict.get("viewpoints", [])
    if vps:
        vx = [v["position"][0] for v in vps]
        vy = [v["position"][1] for v in vps]
        ax.scatter(vx, vy, s=80, c="#1565c0", marker="s", alpha=0.8,
                   edgecolors="white", linewidths=1.0, zorder=5,
                   label=f"viewpoints ({len(vps)})")
        for i, v in enumerate(vps):
            ax.annotate(str(i), (v["position"][0], v["position"][1]),
                        color="white", fontsize=7, fontweight="bold",
                        ha="center", va="center", zorder=6)

    # Executed path + current pose.
    if pose_history and len(pose_history) >= 2:
        hxs = [p[0] for p in pose_history]
        hys = [p[1] for p in pose_history]
        ax.plot(hxs, hys, "-", color="#c62828", linewidth=1.5, alpha=0.7,
                zorder=7, label="executed path")

    if current_pose is not None:
        px = float(current_pose["x"])
        py = float(current_pose["y"])
        yaw = float(current_pose.get("yaw", 0.0))
        ax.add_patch(FancyArrowPatch(
            (px, py),
            (px + 0.6 * math.cos(yaw), py + 0.6 * math.sin(yaw)),
            arrowstyle="->", mutation_scale=15,
            color="#c62828", linewidth=2.0, zorder=8,
        ))
        ax.scatter([px], [py], s=80, c="#c62828", marker="o",
                   edgecolors="white", linewidths=1.0, zorder=8)

    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Pin the window so the boundary doesn't shift between frames.
    if view_bounds is not None:
        xmin, xmax, ymin, ymax = view_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    counts = (
        f"tick {scene_dict.get('tick_id', '?')}   |   "
        f"viewpoints {len(vps)}   objects {len(objs)}   "
        f"near-edges {sum(len(o.get('spatial_relations', [])) for o in objs)}"
    )
    ax.set_title(counts, fontsize=10)
    if (planned_waypoints or vps or objs or sb is not None or
            pose_history or current_pose is not None):
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    # Fixed window → fixed image size (no tight-crop) so the plotted
    # area is identical every frame; otherwise crop-to-content makes
    # the boundary appear to drift as the legend grows.
    return _fig_to_base64(fig, tight=view_bounds is None)


def _draw_local_near_edges(
    ax, objs: list[dict], px: float, py: float, radius: float,
) -> None:
    label_to_idx = {o["label"]: i for i, o in enumerate(objs)}
    for src in objs:
        if math.hypot(src["position"][0] - px, src["position"][1] - py) > radius:
            continue
        for rel in src.get("spatial_relations", []):
            if rel.get("relation") != "near":
                continue
            tgt_idx = label_to_idx.get(rel.get("target_label"))
            if tgt_idx is None:
                continue
            tgt = objs[tgt_idx]
            if math.hypot(tgt["position"][0] - px, tgt["position"][1] - py) > radius:
                continue
            ax.plot(
                [src["position"][0], tgt["position"][0]],
                [src["position"][1], tgt["position"][1]],
                "-", color="#ff9800", alpha=0.4, linewidth=0.7, zorder=3,
            )


def _draw_local_labels(
    ax, objs: list[dict], px: float, py: float, radius: float,
) -> None:
    for o in objs:
        if math.hypot(o["position"][0] - px, o["position"][1] - py) <= radius:
            ax.annotate(
                o["label"], (o["position"][0], o["position"][1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#1b5e20",
            )


# ---------------------------------------------------------------------------
# Hierarchical scene-graph topology
# ---------------------------------------------------------------------------


def render_graph_png(scene_dict: dict[str, Any]) -> str:
    """Render the three-level scene graph as a base64 PNG.

    Layout:

        Row 0:   Room
        Row 1:   Viewpoints (sorted left-to-right by tick_id)
        Row 2:   Objects, grouped under their first-observing viewpoint

    Edges:
        Room → Viewpoint     faint grey
        Viewpoint → Object   light grey thin
        Object ↔ Object near orange arc across the bottom
    """
    vps = scene_dict.get("viewpoints", [])
    objs = scene_dict.get("objects", [])

    g = nx.DiGraph()
    g.add_node("room", kind="room", label=scene_dict.get("room", {}).get("label", "scene"))
    vp_ids: set[int] = set()
    for vp in vps:
        g.add_node(f"vp:{vp['tick_id']}", kind="vp", label=str(vp["tick_id"]))
        g.add_edge("room", f"vp:{vp['tick_id']}", kind="r-v")
        vp_ids.add(int(vp["tick_id"]))
    for i, o in enumerate(objs):
        g.add_node(f"obj:{i}", kind="obj", label=o["label"])
        # observing_viewpoint_ids stores viewpoint tick_ids only (set by
        # SceneRepresentation.add_object via _current_viewpoint_id), so
        # every entry should match a vp:* node. The membership guard
        # below is defensive — catches malformed external JSON without
        # crashing the renderer.
        for vp_tick in o.get("observing_viewpoint_ids", []):
            if int(vp_tick) in vp_ids:
                g.add_edge(f"vp:{vp_tick}", f"obj:{i}", kind="v-o")
    near_pairs: list[tuple[str, str]] = []
    for i, o in enumerate(objs):
        for rel in o.get("spatial_relations", []):
            if rel.get("relation") == "near":
                near_pairs.append((f"obj:{i}", f"obj:{rel['target_index']}"))

    pos = _hierarchical_positions(vps, objs)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_axis_off()

    _draw_edges(ax, g, pos, kind="r-v", color="#cccccc", linewidth=0.5,
                alpha=0.5, zorder=1)
    _draw_edges(ax, g, pos, kind="v-o", color="#bbbbbb", linewidth=0.4,
                alpha=0.5, zorder=2)
    # Object↔Object "near" arcs intentionally not drawn — the graph shows
    # only the Room→Viewpoint→Object hierarchy. (near_pairs is still
    # computed above for the count in the title.)

    _draw_node_group(ax, pos, [n for n in g.nodes if g.nodes[n]["kind"] == "room"],
                     color="#9e9e9e", size=1400, marker="o", zorder=4)
    _draw_node_group(ax, pos, [n for n in g.nodes if g.nodes[n]["kind"] == "vp"],
                     color="#1565c0", size=550, marker="s", zorder=5)
    _draw_node_group(ax, pos, [n for n in g.nodes if g.nodes[n]["kind"] == "obj"],
                     color="#2e7d32", size=180, marker="o", zorder=6)

    for n, (x, y) in pos.items():
        kind = g.nodes[n]["kind"]
        label = g.nodes[n]["label"]
        if kind == "room":
            ax.text(x, y, label, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold", zorder=7)
        elif kind == "vp":
            ax.text(x, y, label, ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold", zorder=7)
        else:
            ax.text(x, y - 0.18, label, ha="center", va="top",
                    color="#1b5e20", fontsize=6.5, zorder=7)

    near_count = len(near_pairs)
    ax.set_title(
        f"tick {scene_dict.get('tick_id', '?')}   |   "
        f"Room 1   Viewpoints {len(vps)}   Objects {len(objs)}   "
        f"near-edges {near_count}",
        fontsize=10,
    )
    fig.tight_layout()
    return _fig_to_base64(fig)


def _hierarchical_positions(
    vps: list[dict], objs: list[dict],
) -> dict[str, tuple[float, float]]:
    """Lay out Room at top, Viewpoints in a row, Objects below their VP.

    Positions are deterministic — adding new nodes appends to the right
    without shuffling earlier ones, so animation is jitter-free.
    """
    pos: dict[str, tuple[float, float]] = {"room": (0.5, 1.0)}
    width = 1.0

    vp_xs = (
        []
        if not vps else
        [width / 2] if len(vps) == 1
        else [i / (len(vps) - 1) for i in range(len(vps))]
    )
    vp_id_to_x: dict[int, float] = {}
    sorted_vp_ticks: list[int] = []
    for vp, x in zip(vps, vp_xs, strict=True):
        pos[f"vp:{vp['tick_id']}"] = (x, 0.55)
        vp_id_to_x[vp["tick_id"]] = x
        sorted_vp_ticks.append(int(vp["tick_id"]))
    sorted_vp_ticks.sort()

    by_vp: dict[float, list[int]] = {}
    for i, o in enumerate(objs):
        observed = o.get("observing_viewpoint_ids") or []
        first_obs = int(observed[0]) if observed else None
        anchor_x = _resolve_anchor(first_obs, vp_id_to_x, sorted_vp_ticks, width / 2)
        by_vp.setdefault(anchor_x, []).append(i)

    vp_span = (1.0 / max(len(vps), 1)) * (0.9 if vps else 1.0)
    for anchor_x, obj_idxs in by_vp.items():
        n = len(obj_idxs)
        cols = max(1, math.ceil(math.sqrt(n)))
        for k, idx in enumerate(obj_idxs):
            col = k % cols
            row = k // cols
            dx = (col - (cols - 1) / 2) * (vp_span / max(cols, 1))
            dy = -0.08 * row
            pos[f"obj:{idx}"] = (anchor_x + dx, 0.1 + dy)
    return pos


def _resolve_anchor(
    first_obs: int | None,
    vp_id_to_x: dict[int, float],
    sorted_vp_ticks: list[int],
    fallback: float,
) -> float:
    """Pick the viewpoint column an object should hang under.

    ``first_obs`` is the first entry in ``observing_viewpoint_ids`` —
    a real viewpoint tick_id (set by ``SceneRepresentation.add_object``
    via ``_current_viewpoint_id``). The fast path is a direct dict
    lookup. The "latest viewpoint at-or-before" fallback only triggers
    for malformed external JSON (e.g. an old log that pre-dates the
    rename and still carries raw tick_ids) — kept for resilience.
    """
    if first_obs is None or not sorted_vp_ticks:
        return fallback
    if first_obs in vp_id_to_x:
        return vp_id_to_x[first_obs]
    # Largest vp tick that is <= first_obs.
    candidate: int | None = None
    for t in sorted_vp_ticks:
        if t <= first_obs:
            candidate = t
        else:
            break
    return vp_id_to_x.get(candidate, fallback) if candidate is not None else fallback


def _draw_edges(
    ax, g: nx.DiGraph, pos: dict[str, tuple[float, float]],
    *, kind: str, color: str, linewidth: float, alpha: float, zorder: int,
) -> None:
    for u, v, data in g.edges(data=True):
        if data.get("kind") != kind:
            continue
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            "-", color=color, linewidth=linewidth, alpha=alpha, zorder=zorder,
        )


def _draw_near_arcs(
    ax, pos: dict[str, tuple[float, float]],
    pairs: list[tuple[str, str]], *, zorder: int,
) -> None:
    for u, v in pairs:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle="arc3,rad=-0.25",
            arrowstyle="-", color="#ff9800",
            alpha=0.35, linewidth=0.6, zorder=zorder,
        ))


def _draw_node_group(
    ax, pos: dict[str, tuple[float, float]], nodes: list[str],
    *, color: str, size: int, marker: str, zorder: int,
) -> None:
    if not nodes:
        return
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    ax.scatter(xs, ys, s=size, c=color, marker=marker, alpha=0.9,
               edgecolors="white", linewidths=1.0, zorder=zorder)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def render_node_tables(scene_dict: dict[str, Any]) -> str:
    """Return three HTML tables (Room, Viewpoints, Objects) as a single string."""
    room = scene_dict.get("room", {}) or {}
    vps = scene_dict.get("viewpoints", []) or []
    objs = scene_dict.get("objects", []) or []

    return (
        "<div class='scene-tables'>"
        + _room_table(room)
        + _viewpoints_table(vps, objs)
        + _objects_table(objs)
        + "</div>"
    )


def _room_table(room: dict) -> str:
    sb = room.get("scene_bounds")
    bounds = (
        f"[{sb[0][0]:.1f}, {sb[0][1]:.1f}] → [{sb[1][0]:.1f}, {sb[1][1]:.1f}]"
        if sb is not None else "—"
    )
    best = room.get("best_image_tick_id")
    n_vps = len(room.get("viewpoint_tick_ids", []) or [])
    return (
        "<h3>Room</h3>"
        "<table class='scene-table'>"
        "<thead><tr><th>label</th><th>scene_bounds (xy)</th>"
        "<th>best_image_tick</th><th>#viewpoints</th></tr></thead>"
        "<tbody><tr>"
        f"<td>{escape(str(room.get('label', '')))}</td>"
        f"<td>{escape(bounds)}</td>"
        f"<td>{escape(str(best)) if best is not None else '—'}</td>"
        f"<td>{n_vps}</td>"
        "</tr></tbody></table>"
    )


def _viewpoints_table(vps: list[dict], objs: list[dict]) -> str:
    obs_count: dict[int, int] = {}
    for o in objs:
        for tid in o.get("observing_viewpoint_ids", []) or []:
            obs_count[tid] = obs_count.get(tid, 0) + 1

    rows = "".join(
        "<tr>"
        f"<td>{vp['tick_id']}</td>"
        f"<td>{vp['position'][0]:.2f}</td>"
        f"<td>{vp['position'][1]:.2f}</td>"
        f"<td>{math.degrees(vp['yaw']):.0f}°</td>"
        f"<td>{obs_count.get(vp['tick_id'], 0)}</td>"
        "</tr>"
        for vp in vps
    )
    if not rows:
        rows = "<tr><td colspan='5' class='empty'>— no viewpoints yet —</td></tr>"
    return (
        f"<h3>Viewpoints ({len(vps)})</h3>"
        "<table class='scene-table'>"
        "<thead><tr><th>tick_id</th><th>x</th><th>y</th>"
        "<th>yaw</th><th>#objects observed</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _objects_table(objs: list[dict]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(o['label'])}</td>"
        f"<td>{o['position'][0]:.2f}</td>"
        f"<td>{o['position'][1]:.2f}</td>"
        f"<td>{o.get('confidence', 1.0):.2f}</td>"
        f"<td>{o['first_tick_id']} → {o['last_tick_id']}</td>"
        f"<td>{len(o.get('observing_viewpoint_ids', []) or [])}</td>"
        f"<td>{_near_summary(o)}</td>"
        "</tr>"
        for o in objs
    )
    if not rows:
        rows = "<tr><td colspan='7' class='empty'>— no objects yet —</td></tr>"
    return (
        f"<h3>Objects ({len(objs)})</h3>"
        "<table class='scene-table'>"
        "<thead><tr><th>label</th><th>x</th><th>y</th><th>conf</th>"
        "<th>tick span</th><th>#obs</th><th>near</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _near_summary(obj: dict) -> str:
    near = [
        escape(r["target_label"])
        for r in obj.get("spatial_relations", []) or []
        if r.get("relation") == "near"
    ]
    if not near:
        return "—"
    if len(near) <= 4:
        return ", ".join(near)
    return ", ".join(near[:3]) + f" (+{len(near) - 3})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SCENE_TABLE_CSS = """
.scene-tables h3 { color: #283593; margin: 16px 0 6px 0; font-size: 14px; }
.scene-table { border-collapse: collapse; width: 100%; font-size: 12px;
               margin-bottom: 12px; }
.scene-table th { background: #5c6bc0; color: white; padding: 6px 8px;
                  text-align: left; }
.scene-table td { padding: 4px 8px; border-bottom: 1px solid #e0e0e0; }
.scene-table td.empty { color: #999; font-style: italic; text-align: center; }
"""


def _fig_to_base64(fig, *, tight: bool = True) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png",
        bbox_inches="tight" if tight else None,
        dpi=100,
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
