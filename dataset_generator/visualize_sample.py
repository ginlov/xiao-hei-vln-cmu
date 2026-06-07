"""Visualize a generated QA sample with target / anchor / distractor highlighted.

Two modes:

  Interactive (default):
      uv run python dataset_generator/visualize_sample.py \\
          --jsonl dataset/vla3d_ref.jsonl --idx 42

  Save to PNG:
      uv run python dataset_generator/visualize_sample.py \\
          --jsonl dataset/vla3d_ref.jsonl --idx 42 --save out/

  Batch (spot-check N per scene, requires --save):
      uv run python dataset_generator/visualize_sample.py \\
          --jsonl dataset/vla3d_ref.jsonl --sample-per-scene 20 --save out/

Color code:
  target      red    (the `target` / `answer.object_id` field)
  anchors     yellow (objects referenced in the question, e.g. "closest to X")
  distractors cyan   (same-class non-answers — only set on ref samples)
  other       gray   (everything else in object_list; hidden by default,
                      pass --show-other to draw these context boxes)

OBB wireframes (heading-aware), so bbox-rotation bugs are visible at a glance.
Defaults: point cloud overlay ON (--no-pointcloud to disable), dark background
ON (--no-dark-bg for white), 0.02 m voxel downsample, 2 cm tube OBB edges.

Install once:   uv sync --extra viz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import signal
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit(
        "open3d/pillow not installed. Run:\n  uv sync --extra viz\n",
    )

HERE = Path(__file__).parent
DEFAULT_SCENE_ROOT = HERE / "vla-3d" / "Unity"

# RGB in [0, 1]
COLOR_TARGET = (1.00, 0.15, 0.15)
COLOR_ANCHOR = (1.00, 0.85, 0.00)
COLOR_DISTRACTOR = (0.00, 0.75, 1.00)
COLOR_OTHER = (0.55, 0.55, 0.55)
COLOR_POINTCLOUD = (0.78, 0.78, 0.78)


@dataclass
class Box:
    obj_id: int
    center: np.ndarray
    extent: np.ndarray
    heading: float
    label: str


# object_list line: `id cx cy cz lx ly lz heading "label"`
_OBJ_LINE_RE = re.compile(
    r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.+)"\s*$',
)


def parse_object_list(lines: list[str]) -> list[Box]:
    boxes: list[Box] = []
    for line in lines:
        m = _OBJ_LINE_RE.match(line)
        if m:
            obj_id = int(m.group(1))
            cx, cy, cz, lx, ly, lz, heading = (float(x) for x in m.groups()[1:8])
            label = m.group(9)
        else:
            parts = line.split()
            if len(parts) < 9:
                continue
            obj_id = int(parts[0])
            cx, cy, cz, lx, ly, lz, heading = map(float, parts[1:8])
            label = " ".join(parts[8:]).strip('"')
        boxes.append(
            Box(
                obj_id=obj_id,
                center=np.array([cx, cy, cz], dtype=float),
                extent=np.array([lx, ly, lz], dtype=float),
                heading=heading,
                label=label,
            ),
        )
    return boxes


def box_to_obb(box: Box, color: tuple[float, float, float]) -> o3d.geometry.OrientedBoundingBox:
    # heading is rotation about Z (see vla3d_loader.py: `bbox orientation about Z`).
    c, s = np.cos(box.heading), np.sin(box.heading)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    extent = np.maximum(box.extent, 1e-3)  # zero-extent guard
    obb = o3d.geometry.OrientedBoundingBox(center=box.center, R=R, extent=extent)
    obb.color = color
    return obb


def _cylinder_between(
    p0: np.ndarray, p1: np.ndarray, radius: float,
    color: tuple[float, float, float],
) -> o3d.geometry.TriangleMesh | None:
    """A solid cylinder spanning p0->p1, used as one thick wireframe edge."""
    v = p1 - p0
    h = float(np.linalg.norm(v))
    if h < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=8)
    cyl.paint_uniform_color(color)
    # Rotate the cylinder's +Z axis onto the edge direction.
    z = np.array([0.0, 0.0, 1.0])
    d = v / h
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, d))
    if s < 1e-9:                                   # already (anti)parallel to Z
        if c < 0:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]),
                       center=(0, 0, 0))
    else:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * np.arctan2(s, c))
        cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((p0 + p1) / 2.0)
    cyl.compute_vertex_normals()
    return cyl


def box_to_tube(
    box: Box, color: tuple[float, float, float], radius: float,
) -> o3d.geometry.TriangleMesh:
    """Render an OBB's 12 edges as solid cylinders of `radius` meters.

    Open3D's legacy `Visualizer` ignores `RenderOption.line_width` on the
    macOS OpenGL backend (verified: width 1 and 8 render identically), so a
    LineSet can't be thickened. Drawing the edges as tubes gives a real,
    backend-independent line thickness the user can dial in.
    """
    obb = box_to_obb(box, color)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    pts = np.asarray(ls.points)
    lines = np.asarray(ls.lines)
    mesh = o3d.geometry.TriangleMesh()
    for i, j in lines:
        seg = _cylinder_between(pts[i], pts[j], radius, color)
        if seg is not None:
            mesh += seg
    return mesh


def _crop_ceiling(
    pcd: o3d.geometry.PointCloud,
    ceiling_cut: float,
) -> o3d.geometry.PointCloud:
    """Drop the top `ceiling_cut` meters of the point cloud so the roof
    stops occluding the interior from an isometric view. Object-bbox
    based cropping doesn't work here because VLA-3D walls extend up to
    ~2.9 m, the same height as the ceiling itself, so there's no margin
    between the tallest box top and the ceiling layer.
    """
    if ceiling_cut <= 0:
        return pcd
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z_max = pts[:, 2].max() - ceiling_cut
    box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[pts[:, 0].min() - 1, pts[:, 1].min() - 1, pts[:, 2].min() - 1],
        max_bound=[pts[:, 0].max() + 1, pts[:, 1].max() + 1, z_max],
    )
    return pcd.crop(box)


def _gamma_boost(pcd: o3d.geometry.PointCloud, gamma: float) -> None:
    """In-place gamma correction on per-point colors. VLA-3D native RGB
    averages ~0.16 (very dark); gamma 0.45 maps that to ~0.43 — readable
    while preserving texture variation.
    """
    if not pcd.has_colors() or gamma == 1.0:
        return
    cols = np.asarray(pcd.colors)
    cols = np.clip(cols ** gamma, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(cols)


def load_pointcloud(
    scene: str,
    scene_root: Path,
    voxel_size: float = 0.05,
    uniform_color: bool = False,
    ceiling_cut: float = 0.5,
    gamma: float = 0.45,
) -> o3d.geometry.PointCloud | None:
    """Load `<scene>_pc_result.ply`.

    - `voxel_size` (m): downsample for speed; VLA-3D Unity clouds are
      several million points and 1-pixel rendering looks like noise.
    - `uniform_color`: override per-point RGB with a flat pale gray.
    - `ceiling_cut` (m): drop the top N meters so the ceiling stops
      occluding the interior. 0 to disable.
    - `gamma`: <1 brightens dark native RGB. Skipped when `uniform_color`.
    """
    ply = scene_root / scene / f"{scene}_pc_result.ply"
    if not ply.exists():
        print(f"  [warn] point cloud not found: {ply}", file=sys.stderr)
        return None
    pcd = o3d.io.read_point_cloud(str(ply))
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd = _crop_ceiling(pcd, ceiling_cut)
    if uniform_color or not pcd.has_colors():
        pcd.paint_uniform_color(COLOR_POINTCLOUD)
    else:
        _gamma_boost(pcd, gamma)
    return pcd


def build_geometries(
    sample: dict,
    scene_root: Path,
    *,
    with_pointcloud: bool,
    voxel_size: float = 0.05,
    uniform_color: bool = False,
    ceiling_cut: float = 0.5,
    line_radius: float = 0.0,
    hide_other: bool = True,
) -> list:
    boxes = parse_object_list(sample["object_list"])
    target_id = sample.get("target")
    anchor_ids = set(sample.get("anchors") or [])
    distractor_ids = set(sample.get("distractor_ids") or [])

    geoms: list = []
    if with_pointcloud:
        pc = load_pointcloud(
            sample["scene"], scene_root,
            voxel_size=voxel_size,
            uniform_color=uniform_color,
            ceiling_cut=ceiling_cut,
        )
        if pc is not None:
            geoms.append(pc)

    for box in boxes:
        if target_id is not None and box.obj_id == target_id:
            color = COLOR_TARGET
        elif box.obj_id in anchor_ids:
            color = COLOR_ANCHOR
        elif box.obj_id in distractor_ids:
            color = COLOR_DISTRACTOR
        elif hide_other:
            continue  # skip context boxes — keep only target/anchor/distractor
        else:
            color = COLOR_OTHER
        # line_radius > 0: thick tube edges (works on every backend);
        # otherwise a 1px LineSet OBB (faster, but un-thickenable on macOS).
        if line_radius > 0:
            geoms.append(box_to_tube(box, color, line_radius))
        else:
            geoms.append(box_to_obb(box, color))

    return geoms


def print_color_legend() -> None:
    """ANSI-colored legend printed once at script start so the interactive
    window (which has no baked-in legend) still has a visible color → role map.
    """
    # ANSI 24-bit foreground for the swatch character.
    def swatch(rgb: tuple[float, float, float]) -> str:
        r, g, b = (int(round(c * 255)) for c in rgb)
        return f"\x1b[38;2;{r};{g};{b}m■\x1b[0m"  # filled square

    print("Color legend:")
    print(f"  {swatch(COLOR_TARGET)} RED    = target (the answer object)")
    print(f"  {swatch(COLOR_ANCHOR)} YELLOW = anchors (referenced in the question)")
    print(f"  {swatch(COLOR_DISTRACTOR)} CYAN   = distractors (same-class non-answers, ref only)")
    print(f"  {swatch(COLOR_OTHER)} GRAY   = other scene objects (context)")
    print()


def print_sample_summary(sample: dict, idx: int) -> None:
    print(f"--- sample #{idx} ---")
    print(f"  scene    : {sample['scene']}")
    print(f"  type     : {sample.get('type')}  source={sample.get('source')}")
    print(f"  question : {sample['question']}")
    print(f"  answer   : {sample['answer']}")
    if sample.get("target") is not None:
        print(f"  target   : id={sample['target']}  (red)")
    if sample.get("anchors"):
        print(f"  anchors  : {sample['anchors']}  (yellow)")
    if sample.get("distractor_ids"):
        print(f"  distract : {sample['distractor_ids']}  (cyan)")
    if sample.get("relation"):
        print(f"  relation : {sample['relation']} ({sample.get('relation_type')})")


def _apply_render_options(vis, *, point_size: float, dark_bg: bool) -> None:
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    if dark_bg:
        opt.background_color = np.array([0.08, 0.08, 0.10])
    ctr = vis.get_view_control()
    ctr.set_front([0.4, -0.4, 0.6])
    ctr.set_up([0.0, 0.0, 1.0])


def visualize_interactive(
    sample: dict, idx: int, geoms: list,
    *, point_size: float = 2.5, dark_bg: bool = False,
) -> None:
    title = f"#{idx} {sample['scene']} | {sample['question'][:60]}"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)
    _apply_render_options(vis, point_size=point_size, dark_bg=dark_bg)

    # `vis.run()` enters Open3D's own blocking C++ loop, which swallows the
    # terminal's SIGINT — Ctrl+C can't close the window. Drive the loop
    # manually instead and install a SIGINT handler that flips a flag; the
    # handler runs between poll_events() iterations, so Ctrl+C breaks out
    # cleanly. `poll_events()` returns False when the window is closed via
    # the X button or Q/Esc, so those paths still work.
    interrupted = {"flag": False}

    def _on_sigint(signum, frame):  # noqa: ARG001
        interrupted["flag"] = True

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        while not interrupted["flag"]:
            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        signal.signal(signal.SIGINT, prev_handler)
        vis.destroy_window()
    if interrupted["flag"]:
        print("  (closed by Ctrl+C)")


def visualize_headless(
    geoms: list, out_path: Path,
    *, size: tuple[int, int] = (1280, 800),
    point_size: float = 2.5, dark_bg: bool = False,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size[0], height=size[1])
    for g in geoms:
        vis.add_geometry(g)
    _apply_render_options(vis, point_size=point_size, dark_bg=dark_bg)
    vis.poll_events()
    vis.update_renderer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(out_path), do_render=True)
    vis.destroy_window()


# --- PNG metadata overlay (PIL) ---------------------------------------

def _to_255(rgb_float: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb_float)


_LEGEND_SPEC = [
    ("target", _to_255(COLOR_TARGET), "RED  — target (the answer object)"),
    ("anchors", _to_255(COLOR_ANCHOR), "YELLOW — anchors (referenced in the question)"),
    ("distractors", _to_255(COLOR_DISTRACTOR), "CYAN — distractors (same-class non-answers)"),
    ("other", _to_255(COLOR_OTHER), "GRAY — other scene objects (context)"),
]

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",                # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    # Linux
    "/Library/Fonts/Arial.ttf",                           # older macOS
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if font.getlength(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _roles_present(sample: dict, hide_other: bool = True) -> set[str]:
    roles: set[str] = set() if hide_other else {"other"}
    if sample.get("target") is not None:
        roles.add("target")
    if sample.get("anchors"):
        roles.add("anchors")
    if sample.get("distractor_ids"):
        roles.add("distractors")
    return roles


def overlay_metadata(
    png_path: Path, sample: dict, idx: int, *, hide_other: bool = True,
) -> None:
    """Composite header (scene + Q + A) and legend onto a saved PNG in-place."""
    img = Image.open(png_path).convert("RGBA")
    W, H = img.size
    pad = 12
    font_title = _load_font(18)
    font_body = _load_font(15)
    font_legend = _load_font(14)
    line_h_title = font_title.size + 6
    line_h_body = font_body.size + 6
    line_h_legend = font_legend.size + 8

    # Build header text lines
    title = f"#{idx}   scene: {sample['scene']}   type: {sample.get('type')}"
    q_lines = _wrap_text(f"Q: {sample['question']}", font_body, W - 2 * pad)
    answer_text = json.dumps(sample["answer"], ensure_ascii=False)
    a_lines = _wrap_text(f"A: {answer_text}", font_body, W - 2 * pad)
    header_h = pad + line_h_title + len(q_lines) * line_h_body + len(a_lines) * line_h_body + pad

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Translucent header bar
    draw.rectangle([0, 0, W, header_h], fill=(0, 0, 0, 190))
    y = pad
    draw.text((pad, y), title, font=font_title, fill=(255, 255, 255, 255))
    y += line_h_title
    for line in q_lines:
        draw.text((pad, y), line, font=font_body, fill=(255, 255, 255, 255))
        y += line_h_body
    for line in a_lines:
        draw.text((pad, y), line, font=font_body, fill=(160, 255, 160, 255))
        y += line_h_body

    # Legend in bottom-right (only the roles present in this sample)
    roles = _roles_present(sample, hide_other=hide_other)
    items = [(rgb, label) for key, rgb, label in _LEGEND_SPEC if key in roles]
    sw = 18  # color swatch size
    label_w = max(int(font_legend.getlength(label)) for _, label in items)
    legend_w = pad + sw + 8 + label_w + pad
    legend_h = pad + len(items) * line_h_legend + pad
    lx = W - legend_w - pad
    ly = H - legend_h - pad
    draw.rectangle([lx, ly, lx + legend_w, ly + legend_h], fill=(0, 0, 0, 190))
    for i, (rgb, label) in enumerate(items):
        cy = ly + pad + i * line_h_legend
        draw.rectangle(
            [lx + pad, cy + 2, lx + pad + sw, cy + 2 + sw],
            fill=(*rgb, 255),
        )
        draw.text(
            (lx + pad + sw + 8, cy),
            label,
            font=font_legend,
            fill=(255, 255, 255, 255),
        )

    Image.alpha_composite(img, overlay).convert("RGB").save(png_path)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_targets(
    rows: list[dict],
    *,
    idx: int | None,
    random_one: bool,
    sample_per_scene: int | None,
    seed: int,
) -> Iterable[tuple[int, dict]]:
    rng = random.Random(seed)
    if idx is not None:
        yield idx, rows[idx]
        return
    if random_one:
        i = rng.randrange(len(rows))
        yield i, rows[i]
        return
    if sample_per_scene is not None:
        by_scene: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_scene.setdefault(r["scene"], []).append(i)
        for scene in sorted(by_scene):
            picks = by_scene[scene]
            if len(picks) > sample_per_scene:
                picks = rng.sample(picks, sample_per_scene)
            picks.sort()
            for i in picks:
                yield i, rows[i]
        return
    yield 0, rows[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--idx", type=int, default=None, help="row index (0-based)")
    ap.add_argument("--random", action="store_true", help="pick one random row")
    ap.add_argument(
        "--sample-per-scene", type=int, default=None,
        help="batch: up to N samples per scene (requires --save)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pointcloud", action=argparse.BooleanOptionalAction, default=True,
        help="overlay the scene point cloud (default on; --no-pointcloud to "
             "disable for a lighter boxes-only view)",
    )
    ap.add_argument(
        "--point-size", type=float, default=2.5,
        help="point cloud point size in pixels (default 2.5; bump to 4-5 for chunky)",
    )
    ap.add_argument(
        "--voxel-size", type=float, default=0.02,
        help="voxel downsample size in meters (default 0.02; 0 disables)",
    )
    ap.add_argument(
        "--gray-points", action="store_true",
        help="force uniform gray (else use VLA-3D native per-point RGB)",
    )
    ap.add_argument(
        "--dark-bg", action=argparse.BooleanOptionalAction, default=True,
        help="dark background — point colors and OBBs pop more (default on; "
             "--no-dark-bg for a white background)",
    )
    ap.add_argument(
        "--ceiling-cut", type=float, default=0.5,
        help="crop top N meters of point cloud so ceiling stops occluding "
             "(default 0.5; 0 to disable)",
    )
    ap.add_argument(
        "--line-radius", type=float, default=0.02,
        help="OBB edge thickness in meters, drawn as solid tubes (default 0.02 "
             "= 2 cm; 0 = thin 1px wireframe). Use this instead of a line "
             "width: Open3D's legacy renderer ignores line_width on macOS.",
    )
    ap.add_argument(
        "--show-other", action="store_true",
        help="also draw the gray context OBBs for every other object. Off by "
             "default — only target / anchor / distractor boxes are shown "
             "(the point cloud already gives spatial context)",
    )
    ap.add_argument(
        "--save", type=Path, default=None,
        help="write PNG to PATH; if PATH is a directory, files are auto-named",
    )
    ap.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    args = ap.parse_args()

    if not args.jsonl.exists():
        ap.error(f"jsonl not found: {args.jsonl}")
    rows = load_jsonl(args.jsonl)
    if not rows:
        ap.error(f"empty jsonl: {args.jsonl}")
    if args.idx is not None and not (0 <= args.idx < len(rows)):
        ap.error(f"--idx {args.idx} out of range [0, {len(rows)})")
    if args.sample_per_scene is not None and args.save is None:
        ap.error("--sample-per-scene requires --save (would open thousands of windows)")

    targets = list(iter_targets(
        rows,
        idx=args.idx,
        random_one=args.random,
        sample_per_scene=args.sample_per_scene,
        seed=args.seed,
    ))

    print_color_legend()

    for i, sample in targets:
        print_sample_summary(sample, i)
        geoms = build_geometries(
            sample, args.scene_root,
            with_pointcloud=args.pointcloud,
            voxel_size=args.voxel_size,
            uniform_color=args.gray_points,
            ceiling_cut=args.ceiling_cut,
            line_radius=args.line_radius,
            hide_other=not args.show_other,
        )
        if args.save is None:
            visualize_interactive(
                sample, i, geoms,
                point_size=args.point_size, dark_bg=args.dark_bg,
            )
        else:
            if args.save.suffix.lower() == ".png":
                out = args.save
            else:
                out = args.save / f"{sample['scene']}_{i:05d}.png"
            visualize_headless(
                geoms, out,
                point_size=args.point_size, dark_bg=args.dark_bg,
            )
            overlay_metadata(out, sample, i, hide_other=not args.show_other)
            print(f"  -> wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
