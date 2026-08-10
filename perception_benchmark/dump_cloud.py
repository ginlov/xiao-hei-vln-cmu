"""Cumulative LiDAR cloud per scene, for the viewer's scene-graph panel.

The scene graph shown at viewpoint *i* is cumulative — every observation up to
and including *i*. The point cloud it should be read against is therefore also
cumulative, not the single sweep the lift panel shows.

Builds the union of all `registered_scan` sweeps (already map-frame, so they
concatenate directly), voxel-downsampled, and records for each surviving point
the index of the FIRST viewpoint that saw it. The viewer slices on that index to
draw the cloud as it stood at any viewpoint.

Sidecar-free and fast — it only reads the captured sweeps.

    uv run python perception_benchmark/dump_cloud.py --all
    uv run python perception_benchmark/dump_cloud.py --scene arabic_room --voxel 0.06
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))
DEBUG_DIR = Path("perception_benchmark/debug")


def build(scene: str, *, voxel_m: float, out_root: Path) -> int:
    vp_dirs = sorted(glob.glob(str(CAP_DIR / scene / "vp_*")))
    if not vp_dirs:
        print(f"[{scene}] no captures — skip")
        return 0

    seen: dict[tuple[int, int, int], tuple[np.ndarray, int]] = {}
    for idx, vp_dir in enumerate(vp_dirs):
        pts = np.load(Path(vp_dir) / "registered_scan.npy")[:, :3]
        keys = np.floor(pts / voxel_m).astype(np.int64)
        uniq, first = np.unique(keys, axis=0, return_index=True)
        for k, i in zip(map(tuple, uniq), first, strict=True):
            if k not in seen:                       # keep the first sighting
                seen[k] = (pts[i], idx)

    if not seen:
        return 0
    points = np.array([v[0] for v in seen.values()], dtype=np.float32)
    first_vp = np.array([v[1] for v in seen.values()], dtype=np.int16)
    out = out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cloud.npz", points=points, first_vp=first_vp,
                        voxel_m=voxel_m, n_viewpoints=len(vp_dirs))
    print(f"[{scene}] {len(points)} voxels over {len(vp_dirs)} viewpoints "
          f"-> {out}/cloud.npz")
    return len(points)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.08,
                    help="downsample resolution (m); coarser keeps the viewer responsive")
    ap.add_argument("--out", type=Path, default=DEBUG_DIR)
    args = ap.parse_args()

    scenes = (sorted(os.path.basename(os.path.dirname(m))
                     for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
              if args.all else [args.scene])
    if not scenes or scenes == [None]:
        ap.error("pass --scene <name> or --all")
    for s in scenes:
        build(s, voxel_m=args.voxel, out_root=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
