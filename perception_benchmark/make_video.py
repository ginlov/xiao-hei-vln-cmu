"""Animate the cumulative 3D object graph as a video, filtered to a class.

The viewer answers "what does the graph look like at viewpoint N". This
answers "how did it get there" — which is the question fragmentation raises,
because a run that ends with 54 `potted plant` nodes for 5 real plants did not
find 54 plants at once; it spawned them one at a time, and the animation shows
where and when.

    uv run --with imageio --with imageio-ffmpeg python \
        perception_benchmark/make_video.py --scene arabic_room \
        --labels "potted plant"

MP4 by default, so the result has a scrubber, a pause and a speed control — a
GIF has none of those, and 300 viewpoints is too long to watch without them.
`--format gif` is still there for pasting into somewhere that will not take a
video. The MP4 encoder comes from the `imageio-ffmpeg` wheel (it ships its own
binary); the GIF path needs nothing beyond Pillow.

Reads a dump_debug.py directory only — no sidecar, no GPU. Point --run at any
of a sweep's dumps to animate that setting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

matplotlib.use("Agg")          # headless — nothing here needs a display

# Palette-quantising each frame is what keeps a 300-frame GIF in the low
# megabytes; 128 colours is plenty for wireframes on a white ground.
GIF_COLORS = 128
PRED_COLOR = "crimson"
GT_COLOR = "0.45"
NEW_COLOR = "gold"


class _Mp4Writer:
    """Streams frames straight to h264 — no PNG round-trip, no frame buffer.

    yuv420p is what browsers and Streamlit's `st.video` will actually play, and
    it requires even dimensions; frames are cropped rather than rescaled so the
    plot is never resampled. `+faststart` moves the index to the front so the
    file seeks immediately instead of after a full download.
    """

    def __init__(self, path: Path, fps: float, quality: int, compat: bool):
        try:
            import imageio.v2 as iio
        except ImportError as e:                        # noqa: F841
            raise SystemExit(
                "MP4 output needs imageio + imageio-ffmpeg:\n"
                "  uv run --with imageio --with imageio-ffmpeg python "
                "perception_benchmark/make_video.py ...\n"
                "(or pass --format gif, which needs only Pillow)") from None
        extra = ["-movflags", "+faststart"]
        if compat:
            # libx264 defaults to High profile. Baseline is what the oldest
            # decoders and the most stripped-down browser builds accept; it
            # costs some bitrate and nothing else.
            extra += ["-profile:v", "baseline", "-level", "3.0"]
        self._w = iio.get_writer(
            str(path), fps=fps, codec="libx264", quality=quality,
            pixelformat="yuv420p",
            macro_block_size=1,          # we hand it even dimensions ourselves
            ffmpeg_params=extra)

    def append(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        self._w.append_data(rgb[:h - h % 2, :w - w % 2])

    def close(self) -> None:
        self._w.close()


class _GifWriter:
    """Collects palette-quantised frames; Pillow can only write a GIF whole."""

    def __init__(self, path: Path, fps: float):
        self.path, self.fps = path, fps
        self.frames: list[Image.Image] = []

    def append(self, rgb: np.ndarray) -> None:
        self.frames.append(Image.fromarray(rgb).convert(
            "P", palette=Image.Palette.ADAPTIVE, colors=GIF_COLORS))

    def close(self) -> None:
        # `disposal=2` clears each frame before the next: without it a
        # shrinking set of boxes would leave the old ones painted underneath.
        self.frames[0].save(
            self.path, save_all=True, append_images=self.frames[1:],
            duration=int(1000 / self.fps), loop=0, optimize=True, disposal=2)


def _box_segments(bmin, bmax):
    """The 12 edges of an axis-aligned box, as (start, end) point pairs."""
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b]) for a, b in e]


def _matches(label: str, wanted: list[str]) -> bool:
    """Case-insensitive exact match, falling back to substring.

    Substring is deliberate: the class list carries both `door` and
    `door frame`, and asking for "plant" should not require knowing that the
    dataset spells it "potted plant".
    """
    if not wanted:
        return True
    low = label.lower()
    return any(w == low or w in low for w in wanted)


def _limits(pts: np.ndarray, pad: float = 0.4):
    """Per-axis ranges around the content, plus the aspect they imply.

    Matplotlib's 3D axes default to a cube regardless of the data, which turns
    a room — wide and flat — into something unreadable: 1 m of height is drawn
    as long as 9 m of floor, so every box reads as a tall column. Feeding the
    true ranges to ``set_box_aspect`` fixes the proportions.
    """
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    span = np.maximum(hi - lo, 1e-3)
    return [(float(lo[i]), float(hi[i])) for i in range(3)], span


def render(args) -> Path:
    run = Path(args.run)
    src = run / args.scene / "viz.json"
    if not src.is_file():
        raise SystemExit(f"no dump at {src}")
    data = json.loads(src.read_text())
    vps = data["viewpoints"]
    wanted = [w.strip().lower() for w in args.labels.split(",") if w.strip()]

    gt = [g for g in data["gt"] if _matches(g["label"], wanted)]
    seen = sorted({n["label"] for v in vps for n in v["nodes"]
                   if _matches(n["label"], wanted)})
    if not seen and not gt:
        every = sorted({n["label"] for v in vps for n in v["nodes"]})
        raise SystemExit(f"nothing matches {wanted!r}. Predicted labels: {every}")
    print(f"labels: {seen or '(none predicted)'}  ·  GT objects: {len(gt)}")

    # One colour per label, so a multi-label GIF stays readable. A single
    # label keeps the viewer's crimson.
    colors = ({seen[0]: PRED_COLOR} if len(seen) <= 1 else
              {lab: c for lab, c in zip(
                  seen, plt.get_cmap("tab10").colors * 4, strict=False)})

    # Fixed limits for the whole animation — recomputing per frame would make
    # the room breathe as nodes appear, which destroys any sense of where
    # things are.
    anchors = [v["pose"] for v in vps]
    anchors += [b for g in gt for b in (g["bmin"], g["bmax"])]
    anchors += [b for n in vps[-1]["nodes"] if _matches(n["label"], wanted)
                for b in (n["bmin"], n["bmax"])]
    (xl, yl, zl), span = _limits(np.array(anchors, dtype=float))
    # A true aspect leaves a room almost flat on screen; z is stretched by a
    # documented factor so the boxes are readable, with the floor keeping short
    # scenes from collapsing entirely.
    aspect = (span[0], span[1], max(span[2] * args.z_scale,
                                    0.25 * max(span[0], span[1])))

    # When each cumulative node id first appeared. Node ids are stable in the
    # cumulative map, so this is what turns a static pile of boxes into a
    # story: the ones that just spawned are drawn gold.
    first_seen: dict[int, int] = {}
    for i, v in enumerate(vps):
        for nd in v["nodes"]:
            first_seen.setdefault(nd["node_id"], i)

    frames_idx = list(range(0, len(vps), args.stride))
    if frames_idx[-1] != len(vps) - 1:
        frames_idx.append(len(vps) - 1)        # always end on the final graph

    ext = args.format
    out = Path(args.out) if args.out else (
        Path("perception_benchmark/videos") /
        f"{args.scene}_{run.name}_"
        f"{'-'.join(wanted).replace(' ', '_') or 'all'}.{ext}")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = (_Mp4Writer(out, args.fps, args.quality, args.compat)
              if ext == "mp4" else _GifWriter(out, args.fps))

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    # A 3D axes fills its rect, so the pane's top edge is drawn straight
    # through a multi-line title. Shrinking the rect is the only thing that
    # actually clears it; ax.clear() leaves the position alone.
    fig.subplots_adjust(top=0.86)
    lifts: list[list[float]] = []               # accumulated lift positions
    last_i = 0

    for f, i in enumerate(frames_idx):
        # Every lift of a matching class since the previous rendered frame —
        # stride must not drop evidence, only skip drawing it.
        for v in vps[last_i:i + 1]:
            lifts += [d["position"] for d in v["detections"]
                      if d["lifted"] and d["position"]
                      and _matches(d["label"], wanted)]
        last_i = i + 1

        vp = vps[i]
        nodes = [n for n in vp["nodes"] if _matches(n["label"], wanted)]
        ax.clear()

        if args.gt and gt:
            segs = [s for g in gt for s in _box_segments(g["bmin"], g["bmax"])]
            ax.add_collection3d(Line3DCollection(
                segs, colors=GT_COLOR, linewidths=1.3, linestyles="--"))

        if args.path:
            p = np.array([v["pose"] for v in vps[:i + 1]], dtype=float)
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="orange", lw=1.2, alpha=0.8)

        if lifts and args.points:
            q = np.array(lifts, dtype=float)
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=7, c="royalblue",
                       alpha=0.5, depthshade=False, linewidths=0)

        fresh = [n for n in nodes
                 if i - first_seen.get(n["node_id"], 0) < args.new_window]
        for lab in seen:
            sub = [n for n in nodes if n["label"] == lab and n not in fresh]
            if not sub:
                continue
            segs = [s for n in sub for s in _box_segments(n["bmin"], n["bmax"])]
            ax.add_collection3d(Line3DCollection(
                segs, colors=colors[lab], linewidths=0.8, alpha=0.55))
            ctr = np.array([n["center"] for n in sub], dtype=float)
            ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], s=8, c=colors[lab],
                       depthshade=False, linewidths=0)
        if fresh:
            segs = [s for n in fresh for s in _box_segments(n["bmin"], n["bmax"])]
            ax.add_collection3d(Line3DCollection(
                segs, colors=NEW_COLOR, linewidths=2.0))
            ctr = np.array([n["center"] for n in fresh], dtype=float)
            ax.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], s=28, c=NEW_COLOR,
                       edgecolors="black", depthshade=False, linewidths=0.4)

        ax.scatter(*vp["pose"], s=45, c="red", marker="o", depthshade=False)
        if vp.get("yaw") is not None:
            # Yaw is a rotation about +z; the arrow is the vehicle's forward
            # axis, i.e. what the 360° camera is centred on.
            dx, dy = np.cos(vp["yaw"]), np.sin(vp["yaw"])
            ax.quiver(*vp["pose"], dx, dy, 0.0, length=0.8, color="red", lw=1.6)

        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_zlim(*zl)
        ax.set_box_aspect(aspect)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=args.elev,
                     azim=args.azim + args.spin * f / max(len(frames_idx) - 1, 1))
        # "54 of 327" rather than "54": with dozens of overlapping boxes the
        # frame looks unfiltered, and the denominator is what says otherwise.
        #
        # suptitle, not ax.set_title: a 3D projection overflows its axes rect,
        # and the pane edges draw *over* an axes-level title (an opaque bbox
        # loses too — it is inside the same axes). Figure artists always draw
        # last, so this is the only placement that stays clean.
        fig.suptitle(
            f"{args.scene} · {run.name} · {vp['id']}  ({i + 1}/{len(vps)})\n"
            f"{', '.join(seen) or args.labels}: "
            f"{len(nodes)} of {len(vp['nodes'])} nodes"
            + (f" vs {len(gt)} GT" if gt else "")
            + f"  ·  {len(lifts)} lifts"
            + "\n(other classes hidden)",
            fontsize=9, y=0.985, va="top")
        handles = [Line2D([], [], color=colors[lab], lw=1.5, label=lab)
                   for lab in seen]
        handles.append(Line2D([], [], color=NEW_COLOR, lw=2.0,
                              label=f"new (<{args.new_window} vp)"))
        if args.gt and gt:
            handles.append(Line2D([], [], color=GT_COLOR, lw=1.2, ls="--",
                                  label="ground truth"))
        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=8,
                      framealpha=0.7)

        # Straight off the canvas — a PNG round-trip per frame costs more than
        # the plotting does. The figure size is fixed (no bbox_inches="tight",
        # which crops to content and would make the video size jitter).
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="arabic_room")
    p.add_argument("--run", default="perception_benchmark/debug_k10",
                   help="a dump_debug.py output directory")
    p.add_argument("--labels", default="potted plant",
                   help="comma-separated; exact match, else substring. "
                        "Empty string = every class.")
    p.add_argument("--out", default=None)
    p.add_argument("--format", choices=("mp4", "gif"), default="mp4",
                   help="mp4 gives playback controls; gif is for pasting "
                        "somewhere that will not take a video")
    p.add_argument("--compat", action="store_true",
                   help="mp4 only: H.264 baseline profile, for a player that "
                        "shows nothing on the default High profile")
    p.add_argument("--quality", type=int, default=8,
                   help="mp4 only, 0-10 (imageio scale); 8 is visually clean "
                        "for line art")
    p.add_argument("--stride", type=int, default=1,
                   help="render every Nth viewpoint (skipped lifts are still "
                        "counted). 1 keeps the scrubber aligned to viewpoints; "
                        "raise it to shrink a GIF.")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--elev", type=float, default=38.0)
    p.add_argument("--new-window", type=int, default=8,
                   help="a node is drawn gold for this many viewpoints after "
                        "it first appears (0 disables the highlight)")
    p.add_argument("--z-scale", type=float, default=2.0,
                   help="vertical exaggeration of the plot box (1 = true)")
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--spin", type=float, default=0.0,
                   help="degrees of azimuth swept across the whole animation")
    p.add_argument("--width", type=float, default=7.0)
    p.add_argument("--height", type=float, default=5.5)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--no-gt", dest="gt", action="store_false")
    p.add_argument("--no-path", dest="path", action="store_false")
    p.add_argument("--no-points", dest="points", action="store_false",
                   help="hide the accumulated lift positions")
    render(p.parse_args())


if __name__ == "__main__":
    main()
