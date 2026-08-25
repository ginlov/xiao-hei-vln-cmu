"""Animate one class's detections, lifts and *merges* over a whole run.

``make_video.py`` answers "what does the graph look like at viewpoint N" and
draws every node one colour per label. That hides the exact thing
fragmentation is about: a carpet that ends the run as 28 separate nodes for 2
real carpets did not fail once — each frame it either fused a lift into an
existing node or spawned a brand new one, and the question is *which, and
why*. This viewer colours every lift and every box by the **node it fused
into** (via the ``node_id`` the offline ObjectMap already wrote into
``viz.json``), so a single physical carpet painted in six colours *is* the
fragmentation, drawn.

Two stacked panels, per viewpoint:

  * top    — the panorama detection overlay (``vp_NNN.png``) the sidecar drew,
             so you can see what the detector actually saw that frame;
  * bottom — a top-down (x-y) accumulation of every lift of the class so far,
             coloured by ``node_id``; current-frame lifts are ringed, node
             AABB footprints are drawn, GT footprints dashed, and a freshly
             spawned node is flagged with a line to its nearest same-class
             neighbour and the gap in metres — a small gap is a merge that
             *should* have happened and did not.

    uv run --with imageio --with imageio-ffmpeg python \
        perception_benchmark/merge_video.py --scene arabic_room \
        --label carpet --run perception_benchmark/debug

Reads a ``dump_debug.py`` directory only — no sidecar, no GPU. What it can
show is the **cross-frame ObjectMap fusion**; the *seam* merge runs pre-lift
inside a frame and is not in this dump (see ``--help`` epilogue).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

matplotlib.use("Agg")          # headless — nothing here needs a display

# A qualitative cycle long enough that adjacent fragments rarely collide on
# colour; node ids are mapped into it by order of first appearance, not by the
# id itself, so the palette stays dense however sparse the ids are.
_CMAP = (list(plt.get_cmap("tab20").colors)
         + list(plt.get_cmap("tab20b").colors)
         + list(plt.get_cmap("tab20c").colors))
GT_COLOR = "black"
NEW_RING = "red"


class _Mp4Writer:
    """Streams frames straight to h264 (same recipe as make_video.py)."""

    def __init__(self, path: Path, fps: float, quality: int):
        try:
            import imageio.v2 as iio
        except ImportError:
            raise SystemExit(
                "MP4 output needs imageio + imageio-ffmpeg:\n"
                "  uv run --with imageio --with imageio-ffmpeg python "
                "perception_benchmark/merge_video.py ...") from None
        self._w = iio.get_writer(
            str(path), fps=fps, codec="libx264", quality=quality,
            pixelformat="yuv420p", macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"])

    def append(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        self._w.append_data(rgb[:h - h % 2, :w - w % 2])

    def close(self) -> None:
        self._w.close()


def _matches(label: str, wanted: str) -> bool:
    low = label.lower()
    w = wanted.lower()
    return w == low or w in low


def _footprint(bmin, bmax):
    """(x, y, width, height) of an AABB's floor footprint, for a Rectangle."""
    return bmin[0], bmin[1], bmax[0] - bmin[0], bmax[1] - bmin[1]


def render(args) -> Path:
    run = Path(args.run)
    src = run / args.scene / "viz.json"
    if not src.is_file():
        raise SystemExit(f"no dump at {src}")
    data = json.loads(src.read_text())
    vps = data["viewpoints"]
    label = args.label

    gt = [g for g in data["gt"] if _matches(g["label"], label)]
    n_det_total = sum(1 for v in vps for d in v["detections"]
                      if _matches(d["label"], label))
    if n_det_total == 0:
        every = sorted({d["label"] for v in vps for d in v["detections"]})
        raise SystemExit(f"no {label!r} detections. Labels present: {every}")

    # Stable colour per node id, ordered by first appearance so the palette is
    # used densely regardless of how large / sparse the ids are.
    order: list[int] = []
    for v in vps:
        for d in v["detections"]:
            if (_matches(d["label"], label) and d["node_id"] is not None
                    and d["node_id"] not in order):
                order.append(d["node_id"])
    color_of = {nid: _CMAP[i % len(_CMAP)] for i, nid in enumerate(order)}
    print(f"{label}: {n_det_total} detections over {len(vps)} vps · "
          f"{len(order)} distinct node ids · {len(gt)} GT")

    # First frame a node id is written, so a spawn can be flagged the instant
    # it happens rather than every frame it persists.
    first_seen: dict[int, int] = {}
    for i, v in enumerate(vps):
        for d in v["detections"]:
            if _matches(d["label"], label) and d["node_id"] is not None:
                first_seen.setdefault(d["node_id"], i)

    # Fixed x-y limits over all class content + GT + path, so the floor plan
    # does not breathe as fragments appear.
    anchor = [d["position"][:2] for v in vps for d in v["detections"]
              if _matches(d["label"], label) and d["lifted"] and d["position"]]
    anchor += [g["bmin"][:2] for g in gt] + [g["bmax"][:2] for g in gt]
    anchor += [v["pose"][:2] for v in vps]
    A = np.array(anchor, dtype=float)
    (x0, y0), (x1, y1) = A.min(0) - 0.6, A.max(0) + 0.6

    frames_idx = list(range(0, len(vps), args.stride))
    if frames_idx[-1] != len(vps) - 1:
        frames_idx.append(len(vps) - 1)

    out = Path(args.out) if args.out else (
        Path("perception_benchmark/videos")
        / f"{args.scene}_{run.name}_{label.replace(' ', '_')}_merge.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = _Mp4Writer(out, args.fps, args.quality)

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[args.pano_ratio, 1.0],
                          hspace=0.12, left=0.02, right=0.98,
                          top=0.93, bottom=0.06)
    ax_pano = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])

    # Accumulated across frames: (x, y, node_id) for every lift so far.
    acc_xy: list[tuple[float, float]] = []
    acc_nid: list[int] = []
    last_i = 0

    for f, i in enumerate(frames_idx):
        # Fold in every intervening frame's lifts — stride skips *drawing*,
        # never *evidence*.
        for v in vps[last_i:i + 1]:
            for d in v["detections"]:
                if (_matches(d["label"], label) and d["lifted"]
                        and d["position"]):
                    acc_xy.append((d["position"][0], d["position"][1]))
                    acc_nid.append(d["node_id"])
        last_i = i + 1
        vp = vps[i]

        # ---- top: panorama overlay ---------------------------------------
        ax_pano.clear()
        ax_pano.axis("off")
        png = run / args.scene / f"{vp['id']}.png"
        if png.is_file():
            ax_pano.imshow(Image.open(png))
        cur = [d for d in vp["detections"] if _matches(d["label"], label)]
        n_lift_cur = sum(1 for d in cur if d["lifted"])
        ax_pano.set_title(
            f"{vp['id']}  ({i + 1}/{len(vps)})   ·   "
            f"{len(cur)} {label} det this frame, {n_lift_cur} lifted",
            fontsize=9)

        # ---- bottom: top-down accumulation coloured by node --------------
        ax_map.clear()
        # GT footprints, dashed.
        for g in gt:
            fx, fy, fw, fh = _footprint(g["bmin"], g["bmax"])
            ax_map.add_patch(Rectangle((fx, fy), fw, fh, fill=False,
                                       ec=GT_COLOR, ls="--", lw=1.4))
        # Every lift so far, coloured by the node it fused into.
        if acc_xy:
            q = np.array(acc_xy)
            cols = [color_of.get(n, "0.7") for n in acc_nid]
            ax_map.scatter(q[:, 0], q[:, 1], s=10, c=cols, alpha=0.45,
                           linewidths=0)
        # Current frame's lifts, ringed so the eye tracks what just landed.
        cur_pts = [(d["position"][0], d["position"][1], d["node_id"])
                   for d in cur if d["lifted"] and d["position"]]
        if cur_pts:
            cq = np.array([(x, y) for x, y, _ in cur_pts])
            cc = [color_of.get(n, "0.7") for _, _, n in cur_pts]
            ax_map.scatter(cq[:, 0], cq[:, 1], s=55, c=cc,
                           edgecolors="black", linewidths=0.7, zorder=5)
        # Where each node's accumulated lifts actually sit, so the id label
        # lands *on its own point cluster* rather than at a box centre that
        # can be metres away and buried under overlapping footprints.
        lift_centroid: dict[int, np.ndarray] = {}
        if acc_xy:
            arr = np.array(acc_xy)
            nid_arr = np.array(acc_nid)
            for nid in set(acc_nid):
                if nid is None:
                    continue
                lift_centroid[nid] = np.median(arr[nid_arr == nid], axis=0)

        # Node footprints present now, id-labelled on the cluster.
        nodes = [n for n in vp["nodes"] if _matches(n["label"], label)]
        for n in nodes:
            fx, fy, fw, fh = _footprint(n["bmin"], n["bmax"])
            c = color_of.get(n["node_id"], "0.5")
            fresh = i - first_seen.get(n["node_id"], 0) < args.new_window
            # Solid + opaque only for a freshly spawned box; the standing
            # ones go thin and translucent so 28 overlapping footprints do
            # not bury the lift clusters that are the actual evidence.
            ax_map.add_patch(Rectangle(
                (fx, fy), fw, fh, fill=False, ec=c,
                lw=2.4 if fresh else 0.9, alpha=1.0 if fresh else 0.4,
                zorder=4 if fresh else 2))
            lx, ly = lift_centroid.get(n["node_id"], n["center"][:2])
            ax_map.text(lx, ly, str(n["node_id"]), fontsize=7,
                        ha="center", va="center", color="black", zorder=10,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec=c, lw=1.6 if fresh else 0.8, alpha=0.85))

        # Flag nodes that *first appeared this frame*: draw a line to the
        # nearest existing same-class node and label the gap. A small gap is
        # a merge the ObjectMap declined — i.e. a fragment.
        spawned = [n for n in nodes if first_seen.get(n["node_id"]) == i]
        centers = {n["node_id"]: np.array(n["center"][:2]) for n in nodes}
        event = ""
        for n in spawned:
            c0 = centers[n["node_id"]]
            others = [(nid, p) for nid, p in centers.items()
                      if nid != n["node_id"]]
            if others:
                nid_near, p_near = min(
                    others, key=lambda kp: np.linalg.norm(kp[1] - c0))
                dist = float(np.linalg.norm(p_near - c0))
                ax_map.plot([c0[0], p_near[0]], [c0[1], p_near[1]],
                            color=NEW_RING, lw=1.2, ls=":", zorder=7)
                mid = (c0 + p_near) / 2
                ax_map.text(mid[0], mid[1], f"{dist:.2f}m", fontsize=6.5,
                            color=NEW_RING, ha="center", zorder=8)
                event = (f"NEW node {n['node_id']} — "
                         f"{dist:.2f} m from node {nid_near}")

        # Robot pose + heading.
        ax_map.scatter(*vp["pose"][:2], s=40, c="red", marker="o", zorder=9)
        if vp.get("yaw") is not None:
            dx, dy = np.cos(vp["yaw"]), np.sin(vp["yaw"])
            ax_map.arrow(vp["pose"][0], vp["pose"][1], 0.6 * dx, 0.6 * dy,
                         head_width=0.12, color="red", zorder=9)

        ax_map.set_xlim(x0, x1)
        ax_map.set_ylim(y0, y1)
        ax_map.set_aspect("equal")
        ax_map.grid(True, lw=0.3, alpha=0.4)
        ax_map.set_title(
            f"top-down · coloured by fused node id  ·  "
            f"{len(nodes)} {label} nodes vs {len(gt)} GT  ·  "
            f"{len(acc_xy)} lifts"
            + (f"   ·   {event}" if event else ""),
            fontsize=8.5, color=NEW_RING if event else "black")

        handles = [
            Line2D([], [], marker="o", ls="", mfc="0.6", mec="0.6",
                   label="lift (colour = fused node)"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="black",
                   label="lift this frame"),
            Line2D([], [], color=GT_COLOR, ls="--", label="GT footprint"),
            Line2D([], [], color=NEW_RING, ls=":",
                   label="new node → nearest (gap)"),
        ]
        ax_map.legend(handles=handles, loc="upper right", fontsize=6.5,
                      framealpha=0.8)

        fig.canvas.draw()
        writer.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        if f % 20 == 0:
            print(f"  frame {f + 1}/{len(frames_idx)}  ({vp['id']}, "
                  f"{len(nodes)} nodes)", flush=True)

    plt.close(fig)
    writer.close()
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(frames_idx)} frames, {mb:.1f} MB, "
          f"{len(frames_idx) / args.fps:.0f}s at {args.fps:g} fps)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Note: this shows the cross-frame ObjectMap fusion (node_id). "
               "The panorama-seam merge runs pre-lift inside a single frame "
               "and is not recorded in viz.json; visualising it needs a "
               "re-dump that logs the union-find groups.")
    p.add_argument("--scene", default="arabic_room")
    p.add_argument("--run", default="perception_benchmark/debug",
                   help="a dump_debug.py output directory")
    p.add_argument("--label", default="carpet",
                   help="class to follow (exact, else substring)")
    p.add_argument("--out", default=None)
    p.add_argument("--stride", type=int, default=1,
                   help="draw every Nth viewpoint (skipped lifts still count)")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--quality", type=int, default=8)
    p.add_argument("--new-window", type=int, default=6,
                   help="a node's box is bold for this many vps after it "
                        "first appears")
    p.add_argument("--pano-ratio", type=float, default=1.1,
                   help="height of the panorama panel relative to the map")
    p.add_argument("--width", type=float, default=11.0)
    p.add_argument("--height", type=float, default=8.5)
    p.add_argument("--dpi", type=int, default=100)
    render(p.parse_args())


if __name__ == "__main__":
    main()
