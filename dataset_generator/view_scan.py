#!/usr/bin/env python3
"""Visualize a captured lidar scan (TASK 10).

Loads a frame's scan.npy (+ meta.json pose), drops [0,0,0] filler / NaN points,
and either:
  --png OUT   : save a static multi-view image (3D + top-down BEV)
  (default)   : open an interactive open3d window you can rotate (needs a display;
                run locally with `uv run --extra viz`)
  --ply OUT   : also export a .ply (open in CloudCompare / MeshLab)

Usage:
    uv run --extra viz python dataset_generator/view_scan.py \
        --frame dataset_generator/captures/run1/frames/000001 --png /tmp/scan.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_scan(frame: Path):
    xyz = np.load(frame / "scan.npy")[:, :3].astype(np.float64)
    keep = ~(np.all(xyz == 0, axis=1) | np.isnan(xyz).any(axis=1))
    xyz = xyz[keep]
    cam = None
    mp = frame / "meta.json"
    if mp.exists():
        p = json.load(open(mp))["pose"]["position"]
        cam = np.array([p["x"], p["y"], p["z"]])
    return xyz, cam


def save_png(xyz, cam, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = xyz[:, 2]
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=z, s=1.5, cmap="viridis")
    if cam is not None:
        ax.scatter(*cam, c="red", s=120, marker="*", label="robot")
        ax.legend()
    ax.set_title("3D point cloud (colour = height)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=35, azim=-60)
    ax2 = fig.add_subplot(1, 2, 2)
    s = ax2.scatter(xyz[:, 0], xyz[:, 1], c=z, s=2, cmap="viridis")
    if cam is not None:
        ax2.plot(cam[0], cam[1], "r*", ms=18)
    ax2.set_aspect("equal"); ax2.set_title("top-down BEV")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    plt.colorbar(s, ax=ax2, label="z (m)")
    plt.tight_layout(); plt.savefig(out, dpi=90)
    print(f"wrote {out}  ({len(xyz)} points)")


def show_open3d(xyz, cam):
    import open3d as o3d
    z = xyz[:, 2]
    zn = (z - z.min()) / (np.ptp(z) + 1e-9)
    import matplotlib.cm as cm
    colours = cm.viridis(zn)[:, :3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colours)
    geoms = [pc]
    if cam is not None:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sph.translate(cam); sph.paint_uniform_color([1, 0, 0])
        geoms.append(sph)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    o3d.visualization.draw_geometries(geoms)


def export_ply(xyz, out: Path):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(out), pc)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--ply", type=Path, default=None)
    args = ap.parse_args()
    xyz, cam = load_scan(args.frame)
    if args.ply:
        export_ply(xyz, args.ply)
    if args.png:
        save_png(xyz, cam, args.png)
    if not args.png and not args.ply:
        show_open3d(xyz, cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
