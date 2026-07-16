"""Offline prototype: project VLA-3D CSV 3D boxes into an equirectangular
(360deg) view and overlay them on a panorama rendered from the scene point
cloud.

NO simulator needed. This validates the equirect projection + box-corner +
seam-wrap logic against the static ply+csv, using a *synthetic* camera pose
you place yourself. It is the de-risking first step of TASK 10 Phase 3
(section F): get the projection math right before spending sim time.

What it fakes vs. the real pipeline:
  - T (CSV-world -> sim-map)      : identity (the ply *is* in CSV-world)
  - robot pose (map -> sensor)    : you set --cam / --yaw
  - camera extrinsic (->camera)   : skipped (camera placed directly in world)
  - RGB image                     : rendered from the point cloud (a stand-in,
                                    NOT the sim's real camera image)

If a drawn box sits on the right cluster of points, the projection is correct.
If every box is uniformly off, a transform/axis convention is wrong.

World convention (verified from the CSVs): Z is up; bbox heading is a yaw
(radians) about Z. Equirect: azimuth = atan2(dy, dx) over full 360deg,
elevation = atan2(dz, hypot(dx,dy)) clipped to the vertical FOV.

Usage:
    uv run --extra qwen python dataset_generator/project_gt_prototype.py \
        --scene livingroom_1 --yaw 0 --out /tmp/gt_proj.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
UNITY = HERE / "vla-3d" / "Unity"


def read_ply_xyzrgb(path: Path, subsample: int = 0):
    """Parse a binary-little-endian ply of (float x,y,z + uchar r,g,b)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a ply file"
        fmt = f.readline().strip()
        assert b"binary_little_endian" in fmt, f"unexpected format: {fmt!r}"
        n = 0
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line == b"end_header":
                break
        dt = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        data = np.fromfile(f, dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    finite = np.isfinite(xyz).all(axis=1)        # drop NaN/Inf returns
    xyz, rgb = xyz[finite], rgb[finite]
    if subsample and subsample < len(xyz):
        idx = np.random.default_rng(42).choice(len(xyz), subsample, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def load_objects(scene: str):
    p = UNITY / scene / f"{scene}_object_result.csv"
    objs = []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                c = np.array([float(row["object_bbox_cx"]),
                              float(row["object_bbox_cy"]),
                              float(row["object_bbox_cz"])])
                s = np.array([float(row["object_bbox_xlength"]),
                              float(row["object_bbox_ylength"]),
                              float(row["object_bbox_zlength"])])
            except ValueError:
                continue
            h_raw = row.get("object_bbox_heading", "_")
            try:
                h = float(h_raw)
            except ValueError:
                h = 0.0
            objs.append({"id": row["object_id"], "label": row["raw_label"],
                         "c": c, "s": s, "h": h})
    return objs


def _yaw_rotate(d: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate vectors about Z by -yaw (turn the world opposite the camera)."""
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    x = d[:, 0] * cy - d[:, 1] * sy
    y = d[:, 0] * sy + d[:, 1] * cy
    return np.stack([x, y, d[:, 2]], axis=1)


def equirect(d: np.ndarray, W: int, H: int, fov_v_deg: float):
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    az = np.arctan2(dy, dx)
    el = np.arctan2(dz, np.hypot(dx, dy))
    u = (az / (2 * math.pi) + 0.5) * W
    fov = math.radians(fov_v_deg)
    v = (0.5 - el / fov) * H
    depth = np.linalg.norm(d, axis=1)
    valid = np.abs(el) <= fov / 2
    return u, v, depth, valid


def render_panorama(xyz, rgb, cam, yaw, W, H, fov_v, point_size=1):
    d = _yaw_rotate(xyz - cam, yaw)
    u, v, depth, valid = equirect(d, W, H, fov_v)
    ui = np.floor(u).astype(np.int32)
    vi = np.floor(v).astype(np.int32)
    m = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, depth, col = ui[m], vi[m], depth[m], rgb[m]
    if point_size > 1:                    # splat each point as a square block
        r = point_size // 2
        uu, vv, dd, cc = [], [], [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                uu.append(ui + dx); vv.append(vi + dy); dd.append(depth); cc.append(col)
        ui, vi = np.concatenate(uu), np.concatenate(vv)
        depth, col = np.concatenate(dd), np.concatenate(cc)
        m2 = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, depth, col = ui[m2], vi[m2], depth[m2], col[m2]
    order = np.argsort(-depth)            # far first -> near overwrites
    img = np.zeros((H, W, 3), np.uint8)
    img[vi[order], ui[order]] = col[order]
    return img


def box_corners(c, s, h):
    hx, hy, hz = s / 2
    signs = np.array([[sx, sy, sz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=float)
    local = signs * np.array([hx, hy, hz])
    ch, sh = math.cos(h), math.sin(h)
    R = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
    return (R @ local.T).T + c


ARCH_LABELS = {
    "wall", "floor", "ceiling", "celling", "window", "door", "curtain",
    "column", "carpet", "unknown",
}


def draw_boxes(img, objs, cam, yaw, W, H, fov_v, max_range=None, skip_arch=False):
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    drawn = 0
    for o in objs:
        if skip_arch and o["label"].lower() in ARCH_LABELS:
            continue
        if max_range is not None and np.linalg.norm(o["c"] - cam) > max_range:
            continue
        d = _yaw_rotate(box_corners(o["c"], o["s"], o["h"]) - cam, yaw)
        u, v, _, valid = equirect(d, W, H, fov_v)
        if valid.sum() < 2:
            continue
        u, v = u[valid], v[valid]
        if u.max() - u.min() > W / 2:     # spans the +/-180deg seam
            u = np.where(u < W / 2, u + W, u)
        umin, umax = float(u.min()), float(u.max())
        vmin = max(0.0, float(v.min()))
        vmax = min(H - 1.0, float(v.max()))
        segments = ([(umin, W - 1), (0, umax - W)] if umax > W
                    else [(umin, umax)])
        for a, b in segments:
            dr.rectangle([a, vmin, b, vmax], outline=(255, 0, 0), width=2)
        dr.text((umin % W, max(0, vmin - 10)), o["label"], fill=(255, 255, 0))
        drawn += 1
    return pil, drawn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="livingroom_1")
    ap.add_argument("--cam", type=float, nargs=3, default=None,
                    help="camera x y z (world). Default: object centroid, "
                         "0.8 m above the floor.")
    ap.add_argument("--yaw", type=float, default=0.0, help="camera yaw (deg)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--fov-v", type=float, default=120.0)
    ap.add_argument("--subsample", type=int, default=0,
                    help="use N random points (0 = all)")
    ap.add_argument("--point-size", type=int, default=1,
                    help="splat each point as a NxN block (2-3 fills the gaps)")
    ap.add_argument("--max-range", type=float, default=None,
                    help="only box objects within this distance (m) of cam")
    ap.add_argument("--skip-arch", action="store_true",
                    help="skip architectural labels (wall/floor/ceiling/...)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ply = UNITY / args.scene / f"{args.scene}_pc_result.ply"
    xyz, rgb = read_ply_xyzrgb(ply, args.subsample)
    objs = load_objects(args.scene)

    if args.cam is not None:
        cam = np.array(args.cam, dtype=float)
    else:
        centers = np.array([o["c"] for o in objs])
        floor = np.percentile(xyz[:, 2], 2.0)
        cam = np.array([centers[:, 0].mean(), centers[:, 1].mean(), floor + 0.8])

    yaw = math.radians(args.yaw)
    img = render_panorama(xyz, rgb, cam, yaw, args.width, args.height, args.fov_v,
                          point_size=args.point_size)
    pil, drawn = draw_boxes(img, objs, cam, yaw,
                            args.width, args.height, args.fov_v,
                            max_range=args.max_range, skip_arch=args.skip_arch)

    out = args.out or f"/tmp/gt_proj_{args.scene}.png"
    pil.save(out)
    print(f"scene={args.scene}  points={len(xyz):,}  objects={len(objs)} "
          f"(drawn={drawn})")
    print(f"camera={cam.round(2).tolist()}  yaw={args.yaw}deg  "
          f"fov_v={args.fov_v}deg  size={args.width}x{args.height}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
