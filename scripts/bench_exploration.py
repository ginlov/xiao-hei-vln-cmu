#!/usr/bin/env python3
"""Offline exploration benchmark across all GT scenes.

Loads each scene's ``traversable_area.ply``, simulates sensing + A* navigation,
runs multiple exploration algorithms / hyperparameter configs, and scores
``gt_coverage`` = fraction of GT free cells observed by the robot sensor.

Usage (inside ai_module image with scenes mounted at /scenes)::

    PYTHONPATH=src python3 scripts/bench_exploration.py \\
        --scenes-root /scenes \\
        --out /opt/xiao_hei_vln/exploration_logs/bench_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._gt_world import GTWorld
from xiao_hei_vln.exploration._lawnmower import LawnmowerExplorer
from xiao_hei_vln.exploration._mapfree import (
    NextBestViewExplorer,
    RRTExplorer,
    WallFollowExplorer,
)
from xiao_hei_vln.exploration._metric import ExplorationScore, score_exploration
from xiao_hei_vln.exploration._nearest import NearestFrontierExplorer, RandomFrontierExplorer
from xiao_hei_vln.messages import Header, OdomPose, Quaternion, Stamp, VLMInput, Vector3
from xiao_hei_vln.messages.sensors import TerrainMap


AlgoFactory = Callable[[], Any]


def _algo_registry() -> dict[str, AlgoFactory]:
    """Named algorithms / hyperparameter variants under test.

    All strategies are map-free at runtime: they only see the online belief
    grid from simulated ``terrain_ext``. GT ``traversable_area.ply`` is used
    solely for scoring after the run.
    """

    def frontier(**kwargs):
        defaults = dict(
            max_waypoints=80,
            waypoint_reach_dist=0.45,
            stuck_timeout_s=10.0,
            max_waypoint_dist=3.0,
            max_consecutive_skips=25,
            min_frontier_size=3,
            max_soft_rescues=1,
            unknown_gain_weight=1.0,
        )
        defaults.update(kwargs)
        return FrontierExplorer(**defaults)

    return {
        # Frontier family
        "frontier_baseline": lambda: frontier(),
        "frontier_near": lambda: frontier(max_waypoint_dist=1.5),
        "frontier_far": lambda: frontier(max_waypoint_dist=5.0),
        "frontier_info_heavy": lambda: frontier(unknown_gain_weight=5.0, max_waypoint_dist=4.0),
        "frontier_no_rescue": lambda: frontier(max_soft_rescues=0),
        "nearest_frontier": lambda: NearestFrontierExplorer(max_waypoints=80, seed=0),
        "random_frontier": lambda: RandomFrontierExplorer(max_waypoints=80, seed=0),
        # Non-frontier / map-free families
        "lawnmower": lambda: LawnmowerExplorer(max_waypoints=80, row_stride_cells=4),
        "wall_follow": lambda: WallFollowExplorer(max_waypoints=80, seed=0),
        "nbv": lambda: NextBestViewExplorer(max_waypoints=80, n_samples=40, seed=0),
        "rrt": lambda: RRTExplorer(max_waypoints=80, n_iter=220, seed=0),
    }


def _snapshot(tick: int, t: float, xy: tuple[float, float], terrain: np.ndarray) -> VLMInput:
    hdr = Header(stamp=Stamp.from_seconds(t), frame_id="map")
    return VLMInput(
        tick_id=tick,
        tick_time=Stamp.from_seconds(t),
        pose=OdomPose(
            header=hdr,
            position=Vector3(x=xy[0], y=xy[1], z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
        terrain_ext=TerrainMap(header=hdr, points=terrain, range="ext_20m"),
    )


def run_one(
    world: GTWorld,
    explorer: Any,
    *,
    max_ticks: int = 350,
    step_m: float = 0.45,
    sense_range_m: float = 8.0,
    n_rays: int = 120,
) -> tuple[ExplorationScore, Any]:
    rx, ry = world.start_pose()
    seen_gt: set[tuple[int, int]] = set()
    path_len = 0.0
    prev = (rx, ry)
    goal: tuple[float, float] | None = None
    path: list[tuple[float, float]] = []
    path_i = 0
    stuck_ticks = 0
    done_reason = "max_ticks"
    tick = -1

    for tick in range(max_ticks):
        t = tick * 0.5
        terrain = world.sense(rx, ry, max_range_m=sense_range_m, n_rays=n_rays)
        seen_gt |= world.visible_gt_cells(rx, ry, max_range_m=sense_range_m, n_rays=n_rays)
        snap = _snapshot(tick, t, (rx, ry), terrain)

        # Nav-stack stand-ins for FrontierExplorer helpers.
        if goal is not None and math.hypot(rx - goal[0], ry - goal[1]) < 0.5:
            if hasattr(explorer, "advance"):
                explorer.advance()
            goal = None
            path = []
            path_i = 0
            stuck_ticks = 0

        wp = explorer.update(snap)
        if explorer.is_complete():
            done_reason = "complete"
            break
        if wp is None:
            continue

        new_goal = (wp.x, wp.y)
        if goal is None or math.hypot(new_goal[0] - goal[0], new_goal[1] - goal[1]) > 0.15:
            goal = new_goal
            path = world.astar((rx, ry), goal) or []
            path_i = 0
            stuck_ticks = 0
            if not path:
                if hasattr(explorer, "force_skip"):
                    explorer.force_skip()
                goal = None
                continue

        # Move along path.
        moved = False
        budget = step_m
        while path_i < len(path) and budget > 0:
            tx, ty = path[path_i]
            dx, dy = tx - rx, ty - ry
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                path_i += 1
                continue
            if dist <= budget:
                rx, ry = tx, ty
                budget -= dist
                path_i += 1
                moved = True
            else:
                rx += dx / dist * budget
                ry += dy / dist * budget
                budget = 0
                moved = True

        path_len += math.hypot(rx - prev[0], ry - prev[1])
        if not moved or math.hypot(rx - prev[0], ry - prev[1]) < 0.02:
            stuck_ticks += 1
        else:
            stuck_ticks = 0
        prev = (rx, ry)

        if stuck_ticks >= 8 and hasattr(explorer, "force_skip"):
            explorer.force_skip()
            goal = None
            path = []
            path_i = 0
            stuck_ticks = 0

    mapped = set()
    if hasattr(explorer, "get_grid"):
        mapped = set(explorer.get_grid().free_cells)
    n_vis = len(explorer.get_visited_waypoints()) if hasattr(explorer, "get_visited_waypoints") else 0
    final_ticks = tick + 1

    score = score_exploration(
        gt_free=world.free,
        seen_gt=seen_gt,
        mapped_free=mapped,
        path_length_m=path_len,
        n_visited_waypoints=n_vis,
        ticks=final_ticks,
        done_reason=done_reason,
    )
    return score, explorer


def discover_scenes(root: Path) -> list[Path]:
    scenes = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "traversable_area.ply").is_file():
            scenes.append(p)
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=float, default=0.25)
    ap.add_argument("--max-ticks", type=int, default=300)
    ap.add_argument("--algos", nargs="*", default=None, help="Subset of algo names")
    ap.add_argument("--scenes", nargs="*", default=None, help="Subset of scene names")
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If set, save RViz-style PNGs here (one per scene/algo)",
    )
    args = ap.parse_args()

    registry = _algo_registry()
    algo_names = args.algos or list(registry.keys())
    scenes = discover_scenes(args.scenes_root)
    if args.scenes:
        want = set(args.scenes)
        scenes = [s for s in scenes if s.name in want]

    results: dict[str, Any] = {
        "metric": "gt_coverage = |sensor_seen ∩ GT_free| / |GT_free|",
        "secondary": "coverage_per_meter, mapped_coverage",
        "max_ticks": args.max_ticks,
        "resolution": args.resolution,
        "algorithms": algo_names,
        "scenes": {},
        "summary": {},
        "plots": {},
    }

    # algo -> list of primary scores
    scores_by_algo: dict[str, list[float]] = {a: [] for a in algo_names}

    t0 = time.time()
    for scene in scenes:
        print(f"=== scene {scene.name} ===", flush=True)
        try:
            world = GTWorld.from_traversable_ply(
                scene / "traversable_area.ply",
                resolution=args.resolution,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL load: {exc}", flush=True)
            results["scenes"][scene.name] = {"error": str(exc)}
            continue

        gt_xy = [world.to_world(ix, iy) for ix, iy in world.free]
        scene_row: dict[str, Any] = {"n_gt_free": len(world.free), "algos": {}}
        for name in algo_names:
            print(f"  algo {name} ...", flush=True)
            try:
                explorer = registry[name]()
                score, explorer = run_one(world, explorer, max_ticks=args.max_ticks)
                scene_row["algos"][name] = asdict(score)
                scores_by_algo[name].append(score.gt_coverage)
                print(
                    f"    cov={score.gt_coverage:.3f}  map={score.mapped_coverage:.3f}  "
                    f"path={score.path_length_m:.1f}m  ticks={score.ticks}",
                    flush=True,
                )
                if args.plot_dir is not None:
                    from xiao_hei_vln.exploration._visualize import (
                        layers_from_explorer,
                        save_rviz_style_plot,
                    )

                    plot_path = args.plot_dir / scene.name / f"{name}.png"
                    save_rviz_style_plot(
                        layers_from_explorer(explorer),
                        plot_path,
                        title=f"{scene.name} — {name}",
                        gt_free_xy=gt_xy,
                        score_text=(
                            f"gt_coverage={score.gt_coverage:.3f}  "
                            f"path={score.path_length_m:.1f}m  ticks={score.ticks}"
                        ),
                    )
                    results["plots"].setdefault(scene.name, {})[name] = str(plot_path)
                    print(f"    plot -> {plot_path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                scene_row["algos"][name] = {"error": str(exc)}
        results["scenes"][scene.name] = scene_row

    ranking = []
    for name, vals in scores_by_algo.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        ranking.append((mean, name, len(vals), min(vals), max(vals)))
    ranking.sort(reverse=True)

    results["summary"] = {
        "ranking_mean_gt_coverage": [
            {
                "algo": name,
                "mean": mean,
                "n_scenes": n,
                "min": mn,
                "max": mx,
            }
            for mean, name, n, mn, mx in ranking
        ],
        "best_algo": ranking[0][1] if ranking else None,
        "best_mean_gt_coverage": ranking[0][0] if ranking else None,
        "elapsed_s": time.time() - t0,
    }

    # Per-scene winners
    winners: dict[str, str] = {}
    for sname, row in results["scenes"].items():
        if "algos" not in row:
            continue
        best_name, best_cov = None, -1.0
        for aname, sc in row["algos"].items():
            if isinstance(sc, dict) and "gt_coverage" in sc and sc["gt_coverage"] > best_cov:
                best_cov = sc["gt_coverage"]
                best_name = aname
        if best_name:
            winners[sname] = best_name
    results["summary"]["per_scene_winner"] = winners

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
