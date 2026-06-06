"""CLI for coverage trajectory generation.

Usage:
    python -m xiao_hei_vln.trajectory <scene.zip> [options]
    python -m xiao_hei_vln.trajectory --batch <dir_of_zips> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _visualize(
    result: object,
    traversable_points: np.ndarray,
    poly: object,
    objects: dict,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    ax.scatter(
        traversable_points[::5, 0], traversable_points[::5, 1],
        s=0.3, c="lightgray", alpha=0.5,
    )

    geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, "b-", linewidth=0.8)
        for interior in g.interiors:
            hx, hy = interior.xy
            ax.fill(hx, hy, color="salmon", alpha=0.3)
            ax.plot(hx, hy, "r-", linewidth=0.5)

    for oid, o in objects.items():
        covered = oid not in result.coverage.uncovered_object_ids
        color = "green" if covered else "red"
        marker = "o" if covered else "x"
        ax.plot(o.center.x, o.center.y, marker, color=color, markersize=3, alpha=0.7)

    wps = result.waypoints
    if wps:
        wx = [w.x for w in wps]
        wy = [w.y for w in wps]
        ax.plot(wx, wy, "b-", linewidth=1.5, alpha=0.7, zorder=5)
        ax.scatter(wx, wy, c="blue", s=20, zorder=6)
        ax.plot(wx[0], wy[0], "s", color="lime", markersize=10, zorder=7, label="start")

        for w in wps:
            circle = Circle((w.x, w.y), 3.0, fill=False, edgecolor="blue", linewidth=0.3, alpha=0.3)
            ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.set_title(
        f"{title}\n"
        f"ObjCov={result.coverage.object_coverage*100:.1f}% "
        f"FloorCov={result.coverage.floor_coverage*100:.1f}% "
        f"Waypoints={len(wps)} "
        f"Path={result.path_length_m:.1f}m"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_scene(zip_path: Path, scene_name: str, out_dir: Path, **kwargs) -> dict:
    from xiao_hei_vln.trajectory import plan_trajectory_from_zip
    from xiao_hei_vln.trajectory._io import parse_traversable_ply_from_zip, read_objects_from_zip
    from xiao_hei_vln.trajectory._polygon import build_polygon

    result = plan_trajectory_from_zip(zip_path, scene_name, **kwargs)

    traj_data = {
        "scene": scene_name,
        "waypoints": [{"x": w.x, "y": w.y, "heading": w.heading} for w in result.waypoints],
        "object_coverage": result.coverage.object_coverage,
        "floor_coverage": result.coverage.floor_coverage,
        "uncovered_object_ids": result.coverage.uncovered_object_ids,
        "path_length_m": result.path_length_m,
        "num_waypoints": len(result.waypoints),
        "stats": {**result.coverage.stats, **result.stats},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{scene_name}.json"
    json_path.write_text(json.dumps(traj_data, indent=2))

    pts = parse_traversable_ply_from_zip(zip_path, scene_name=scene_name)
    objs = read_objects_from_zip(zip_path, scene_name=scene_name)
    poly = build_polygon(pts, ratio=kwargs.get("hull_ratio", 0.1))
    png_path = out_dir / f"{scene_name}.png"
    _visualize(result, pts, poly, objs, png_path, scene_name)

    return traj_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage trajectory generator")
    parser.add_argument("input", help="Scene zip file, or directory of zips with --batch")
    parser.add_argument("--batch", action="store_true", help="Process all zips in directory")
    parser.add_argument("--out", default="trajectories", help="Output directory")
    parser.add_argument("--coverage-radius", type=float, default=3.0)
    parser.add_argument("--robot-radius", type=float, default=0.3)
    parser.add_argument("--grid-resolution", type=float, default=0.5)
    parser.add_argument("--hull-ratio", type=float, default=0.1)
    args = parser.parse_args()

    kwargs = {
        "coverage_radius": args.coverage_radius,
        "robot_radius": args.robot_radius,
        "grid_resolution": args.grid_resolution,
        "hull_ratio": args.hull_ratio,
    }
    out_dir = Path(args.out)

    if args.batch:
        zips = sorted(Path(args.input).glob("*.zip"))
        if not zips:
            print(f"No zip files found in {args.input}", file=sys.stderr)
            sys.exit(1)
        results = []
        for zp in zips:
            scene = zp.stem
            print(f"Processing {scene}...", end=" ", flush=True)
            try:
                data = _run_scene(zp, scene, out_dir, **kwargs)
                results.append(data)
                print(
                    f"ObjCov={data['object_coverage']*100:.1f}% "
                    f"FloorCov={data['floor_coverage']*100:.1f}% "
                    f"Waypoints={data['num_waypoints']} "
                    f"Path={data['path_length_m']:.1f}m"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"scene": scene, "error": str(e)})

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2))

        cols = f"{'Scene':<40} {'Objs':>5} {'ObjCov%':>7} {'FlrCov%':>7} {'Wpts':>5} {'Path':>7}"
        header = f"\n{cols}"
        print(header)
        print("-" * 75)
        for r in results:
            if "error" in r:
                print(f"{r['scene']:<40} ERROR: {r['error']}")
            else:
                n_objs = r["stats"].get("objects_total", 0)
                print(
                    f"{r['scene']:<40} {n_objs:>5} "
                    f"{r['object_coverage']*100:>6.1f}% "
                    f"{r['floor_coverage']*100:>6.1f}% "
                    f"{r['num_waypoints']:>5} "
                    f"{r['path_length_m']:>7.1f}"
                )
    else:
        zp = Path(args.input)
        scene = zp.stem
        print(f"Processing {scene}...")
        data = _run_scene(zp, scene, out_dir, **kwargs)
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
