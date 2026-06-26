#!/usr/bin/env python3
"""Visualize lift3d output: LiDAR point cloud + 3D bounding boxes (Phase 1 viz).

Draggable Open3D window (or --png) showing a frame's scan with the lifted 3D
boxes drawn on top, so you can eyeball whether boxes wrap their objects. Pass
two lift jsons to overlay them for comparison (e.g. GT vs YOLO+SAM):

    # produce the jsons first
    uv run python perception/lift3d.py --frame F --source gt   --out /tmp/gt.json
    uv run python perception/lift3d.py --frame F --source yolo --out /tmp/yolo.json
    # then view both (GT green, YOLO red)
    uv run python perception/viz3d.py --frame F --lift /tmp/gt.json --lift2 /tmp/yolo.json

drag=orbit, scroll=zoom, R=reset, Q=quit. Boxes use the AABB from each object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _scan(frame: Path) -> np.ndarray:
    xyz = np.load(frame / "scan.npy")[:, :3].astype(np.float64)
    return xyz[~(np.all(xyz == 0, axis=1) | np.isnan(xyz).any(axis=1))]


def _boxes(lift_json: Path, colour):
    import open3d as o3d
    objs = json.load(open(lift_json))["objects"]
    geoms = []
    for o in objs:
        b = o["bbox_aabb"]
        box = o3d.geometry.AxisAlignedBoundingBox(np.array(b["min"]), np.array(b["max"]))
        box.color = colour
        geoms.append(box)
    return geoms, objs


def _boxes_map(map_json: Path, min_obs: int):
    """Boxes from an objectmap export, coloured by evidence (n_obs)."""
    import open3d as o3d
    objs = [o for o in json.load(open(map_json))["objects"] if o["n_obs"] >= min_obs]
    geoms = []
    for o in objs:
        b = o["bbox_aabb"]
        box = o3d.geometry.AxisAlignedBoundingBox(np.array(b["min"]), np.array(b["max"]))
        t = min(o["n_obs"], 6) / 6.0
        box.color = (1 - t, 0.3 + 0.6 * t, 0.1)      # red(weak) -> green(strong)
        geoms.append(box)
    return geoms, objs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--scans", type=str, default=None,
                    help="glob of frame dirs whose scans are accumulated (map view)")
    ap.add_argument("--lift", type=Path, default=None, help="lift3d json (green)")
    ap.add_argument("--lift2", type=Path, default=None, help="2nd lift json (red)")
    ap.add_argument("--map", dest="omap", type=Path, default=None,
                    help="objectmap json; boxes coloured by n_obs")
    ap.add_argument("--min-obs", type=int, default=2)
    ap.add_argument("--png", type=Path, default=None)
    args = ap.parse_args()
    import glob as _glob
    import open3d as o3d
    import matplotlib.cm as cm

    if args.scans:
        parts = [_scan(Path(d)) for d in sorted(_glob.glob(args.scans)) if Path(d).is_dir()]
        xyz = np.vstack(parts)
        if len(xyz) > 200000:
            xyz = xyz[np.random.choice(len(xyz), 200000, False)]
    else:
        xyz = _scan(args.frame)
    z = xyz[:, 2]
    zn = (z - z.min()) / (np.ptp(z) + 1e-9)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.colors = o3d.utility.Vector3dVector(cm.gray(0.3 + 0.5 * zn)[:, :3])
    geoms = [pc, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]

    if args.omap:
        gm, om = _boxes_map(args.omap, args.min_obs)
        geoms += gm
        print(f"map ({args.omap.name}): {len(om)} objects (n_obs>={args.min_obs}), "
              f"colour red=weak->green=strong")
        for o in sorted(om, key=lambda o: -o["n_obs"]):
            print(f"  {o['label']:18s} obs={o['n_obs']}")
    if args.lift:
        g1, o1 = _boxes(args.lift, (0.1, 0.9, 0.1))
        geoms += g1
        print(f"green ({args.lift.name}): {len(o1)} boxes")
    if args.lift2:
        g2, o2 = _boxes(args.lift2, (0.95, 0.1, 0.1))
        geoms += g2
        print(f"red   ({args.lift2.name}): {len(o2)} boxes")

    if args.png:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1400, height=900)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(str(args.png), do_render=True)
        vis.destroy_window()
        print("wrote", args.png)
    else:
        print("Open3D: drag=orbit, scroll=zoom, R=reset, Q=quit")
        o3d.visualization.draw_geometries(geoms, window_name=str(args.frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
