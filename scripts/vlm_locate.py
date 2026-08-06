#!/usr/bin/env python3
"""Part B end to end: referring expression → box → ray → lidar depth → metres.

`vlm_probe.py` answers "can it point at the thing", which the bearing check
already showed is not the hard part — at 1920 px over 360° the lateral budget
is ±86 px at 2 m. The risk is depth: a box edge that slips past the object's
silhouette puts the ray on the wall behind, and the error along the ray is
unbounded. This measures that, against the acceptance bar of 0.58 m.

Two estimators are reported side by side, because they fail differently:

  ray     median range of the returns inside a small angular cone about the
          box centre, then a point that far along the ray. This is B1, the
          bare-pointing variant, and it is what breaks on a background-catching
          box.
  patch   centroid of those same returns. Robust to a slightly-off ray
          direction, but pulled off-target by any background in the cone.

A `--gt x,y` argument turns the run into a scored one.

    uv run --with anthropic python scripts/vlm_locate.py snaps/start \\
        "the tea table with the elephant figurine on it" --gt -0.02,-2.20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from vlm_probe import (NAMES, ask_claude, ask_gemini, build_prompt,  # noqa: E402
                       load_faces, parse, to_pixels)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from xiao_hei_vln.perception.geometry import sensor_to_camera_transform  # noqa: E402


def rot_from_quat(q: list[float]) -> np.ndarray:
    """XYZW quaternion → R such that ``p_map = R @ p_sensor``."""
    x, y, z, w = q
    n = float(np.linalg.norm([x, y, z, w]))
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def scan_to_camera(scan_map: np.ndarray, pose: dict) -> np.ndarray:
    """Registered scan (map frame) → camera frame, same chain as PointLifter."""
    R_sc, t_sc = sensor_to_camera_transform()
    R_ms = rot_from_quat(pose["orientation"])
    with np.errstate(all="ignore"):
        xyz_sensor = (scan_map[:, :3] - np.asarray(pose["position"])) @ R_ms
        return xyz_sensor @ R_sc.T + t_sc


def dominant_cluster(d: np.ndarray, gap_m: float = 0.35) -> np.ndarray:
    """Largest run of sorted depths with no gap wider than ``gap_m``.

    The same idea `PointLifter._dominant_depth_cluster` uses: when a cone
    catches both the object and the wall behind it, the two populations are
    separated by a gap far larger than the object's own depth spread.
    """
    if len(d) < 2:
        return d
    s = np.sort(d)
    cuts = np.flatnonzero(np.diff(s) > gap_m) + 1
    runs = np.split(s, cuts)
    return max(runs, key=len)


def locate(box_px: list[float], face_idx: int, scan_cam: np.ndarray, *,
           cone_deg: float, pose: dict, min_pts: int = 12) -> dict:
    ymin, xmin, ymax, xmax = box_px
    d = G.face_pixel_to_world_dir(np.array([(xmin + xmax) / 2]),
                                  np.array([(ymin + ymax) / 2]), face_idx)[0]
    d = d / np.linalg.norm(d)

    rng = np.linalg.norm(scan_cam, axis=1)
    live = rng > 1e-3          # the scan carries exact-zero rows; 0/0 is NaN
    with np.errstate(all="ignore"):
        cos = (scan_cam[live] / rng[live, None]) @ d

    # One registered scan is ~1000 points per steradian, so a 2° cone expects
    # under four returns and routinely catches none — the first run lost the
    # table that way and it looked like a depth failure rather than a sampling
    # one. Widen until there is enough to take a median of, and report which
    # cone was actually used so a wide one can be read as the warning it is.
    used = cone_deg
    for c in (cone_deg, cone_deg * 1.5, cone_deg * 2.5, cone_deg * 4):
        sel = cos > np.cos(np.deg2rad(c))
        used = c
        if int(sel.sum()) >= min_pts:
            break

    pts_cam = scan_cam[live][sel]
    if len(pts_cam) == 0:
        return {"n": 0, "cone": used}

    depths = rng[live][sel]
    keep = dominant_cluster(depths)
    lo, hi = keep.min(), keep.max()
    inl = (depths >= lo) & (depths <= hi)

    R_ms = rot_from_quat(pose["orientation"])
    R_sc, t_sc = sensor_to_camera_transform()
    def cam_to_map(p_cam: np.ndarray) -> np.ndarray:
        return ((p_cam - t_sc) @ R_sc) @ R_ms.T + np.asarray(pose["position"])

    return {
        "cone": used,
        "n": int(len(depths)),
        "n_inlier": int(inl.sum()),
        "range": float(np.median(keep)),
        "spread": float(hi - lo),
        "ray_xy": cam_to_map(d * float(np.median(keep)))[:2],
        "patch_xy": cam_to_map(pts_cam[inl].mean(axis=0))[:2],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot")
    ap.add_argument("phrase")
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cone-deg", type=float, default=2.0,
                    help="half-angle of the depth-sampling cone (default 2°)")
    ap.add_argument("--gt", default=None, metavar="X,Y",
                    help="ground-truth centre, to score the result")
    args = ap.parse_args()

    d = Path(args.snapshot)
    pose = json.loads((d / "pose.json").read_text())
    scan_map = np.load(d / "scan.npy")
    scan_cam = scan_to_camera(scan_map, pose)
    print(f"pose {np.round(pose['position'], 2)}  quat "
          f"{np.round(pose['orientation'], 3)}  scan {scan_map.shape}",
          file=sys.stderr)

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else "gemini-2.5-flash")
    fn = ask_claude if args.backend == "claude" else ask_gemini
    reply = parse(fn(build_prompt(args.phrase), load_faces(d), model))
    if reply is None:
        print("unparseable reply", file=sys.stderr)
        return 1
    if not reply.get("visible"):
        e = reply.get("explore") or {}
        print(f"NOT_VISIBLE — explore heading {e.get('heading_deg')}°")
        print(f"  {reply.get('evidence', '')}")
        return 0

    i = int(reply["image_index"])
    space = reply.get("coord_space")
    print(f"\n{args.phrase!r}\n  -> image {i} ({NAMES[i]}), "
          f"confidence {reply.get('confidence')}, coords={space or 'inferred'}")
    print(f"  {reply.get('evidence', '')}\n")

    gt = None
    if args.gt:
        gt = np.array([float(v) for v in args.gt.split(",")])

    for key in ("box_2d", "feature_box_2d"):
        if not reply.get(key):
            continue
        r = locate(to_pixels(reply[key], space, G.FACE_SIZE), i, scan_cam,
                   cone_deg=args.cone_deg, pose=pose)
        if r["n"] == 0:
            print(f"  {key:16s} no lidar returns even at {r['cone']:.1f}° "
                  f"— nothing to lift")
            continue
        print(f"  {key:16s} cone {r['cone']:.1f}°, {r['n']:3d} returns, "
              f"{r['n_inlier']} in cluster, range {r['range']:.2f} m "
              f"(spread {r['spread']:.2f} m)")
        for est in ("ray", "patch"):
            xy = r[f"{est}_xy"]
            line = f"    {est:6s} ({xy[0]:+.2f}, {xy[1]:+.2f})"
            if gt is not None:
                err = float(np.linalg.norm(xy - gt))
                line += f"   error {err:.3f} m   {'PASS' if err < 0.58 else 'FAIL'}"
            print(line)
    if gt is not None:
        print(f"\n  ground truth ({gt[0]:+.2f}, {gt[1]:+.2f}); bar is 0.58 m, "
              f"the median distance the reference trajectories keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
