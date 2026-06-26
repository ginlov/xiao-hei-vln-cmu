"""Solve the CSV-world -> sim-map rigid transform T (TASK 10, Step 0).

T places VLA-3D CSV object coordinates (which live in the scene's *world*
frame, the same frame as `_pc_result.ply`) into the simulator's *map* frame,
so captured GT boxes line up with the robot's frames. Until T is known, GT
cannot be projected — and a wrong T shows up as every box being uniformly
offset/rotated (vs. genuine per-object error = the residual after alignment).

Method (scan is partial, ply is full -> register partial onto full):
  source = captured /registered_scan (map frame, partial)
  target = scene _pc_result.ply       (world frame, full)
  voxel-downsample both -> FPFH global RANSAC for a coarse init
  -> point-to-plane ICP refine  => returns map->world
  T (world->map) = inverse(map->world)

Reports fitness (higher better) and inlier RMSE (lower better). Eyeball the
overlay before trusting it.

Self-test (NO sim needed): register the ply against a known rotated+translated
subsample of itself and check the recovered transform matches:
    uv run --extra viz python dataset_generator/solve_alignment.py \
        --scene livingroom_1 --selftest

Real use (on the box, after capturing a scan):
    uv run --extra viz python dataset_generator/solve_alignment.py \
        --scene livingroom_1 --scan captured_registered_scan.npy --out T.npy
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).parent
UNITY = HERE / "vla-3d" / "Unity"


def load_ply_points(scene: str) -> o3d.geometry.PointCloud:
    """Load the scene point cloud (world frame) via our own ply reader (fast)."""
    from project_gt_prototype import read_ply_xyzrgb  # local module
    xyz, rgb = read_ply_xyzrgb(UNITY / scene / f"{scene}_pc_result.ply")
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    return pc


def load_scan(path: Path) -> o3d.geometry.PointCloud:
    """Load a captured registered_scan (.npy (N,>=3) or .pcd/.ply)."""
    pc = o3d.geometry.PointCloud()
    if path.suffix == ".npy":
        arr = np.load(path)
        pc.points = o3d.utility.Vector3dVector(np.asarray(arr)[:, :3].astype(np.float64))
    else:
        pc = o3d.io.read_point_cloud(str(path))
    return pc


def _prep(pc: o3d.geometry.PointCloud, voxel: float):
    down = pc.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, fpfh


def register(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
             voxel: float = 0.05):
    """Return (T_source_to_target, fitness, rmse) via FPFH-RANSAC + ICP."""
    src_d, src_f = _prep(source, voxel)
    tgt_d, tgt_f = _prep(target, voxel)
    coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))
    fine = o3d.pipelines.registration.registration_icp(
        src_d, tgt_d, voxel * 1.5, coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return fine.transformation, fine.fitness, fine.inlier_rmse


def random_rigid(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.uniform(0, math.pi)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * ang)
    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = rng.uniform(-3, 3, size=3)
    return G


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="livingroom_1")
    ap.add_argument("--scan", type=Path, default=None,
                    help="captured /registered_scan (.npy/.pcd/.ply), map frame")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true",
                    help="register the ply vs a transformed subsample of itself")
    ap.add_argument("--out", type=Path, default=None, help="save 4x4 T (.npy)")
    args = ap.parse_args()

    target = load_ply_points(args.scene)  # world frame (full)

    if args.selftest:
        full = np.asarray(target.points)
        rng = np.random.default_rng(42)
        sub = full[rng.choice(len(full), len(full) // 5, replace=False)]
        G = random_rigid(seed=7)                       # pretend "world->map"
        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(
            (G[:3, :3] @ sub.T).T + G[:3, 3])           # subsample in "map" frame
        src = scan
    elif args.scan is not None:
        src = load_scan(args.scan)
    else:
        ap.error("provide --scan <file> or --selftest")

    map_to_world, fit, rmse = register(src, target, args.voxel)
    T = np.linalg.inv(map_to_world)                     # world -> map
    print(f"scene={args.scene}  fitness={fit:.3f}  inlier_rmse={rmse:.4f} m")
    print("T (world->map):\n", np.array2string(T, precision=4, suppress_small=True))

    if args.selftest:
        # recovered map->world should match G^{-1}; i.e. T (world->map) ~ G
        err = np.linalg.norm(T - G)
        print(f"\n[selftest] ||T - G_true|| = {err:.4f}  "
              f"({'PASS' if err < 0.05 else 'FAIL — check voxel / overlap'})")

    if args.out is not None:
        np.save(args.out, T)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
