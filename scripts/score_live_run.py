#!/usr/bin/env python3
"""Score a live exploration dump vs GT traversable_area.ply.

Reads ``exploration_state.json`` (belief free cells + pose_trace) written by
the AI module, and computes the same metrics as the offline bench:

  gt_coverage     = |sensor_seen ∩ GT_free| / |GT_free|
                    (raycast from live pose_trace through GT)
  mapped_coverage = |belief_free ∩ GT_free| / |GT_free|

Usage:
  python3 scripts/score_live_run.py \\
    --state exploration_logs/exploration_state.json \\
    --scene /home/ubuntu/Downloads/unity_env_models/chinese_room \\
    --out exploration_logs/live_rviz/chinese_room/nbv_score.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xiao_hei_vln.exploration._gt_world import GTWorld  # noqa: E402
from xiao_hei_vln.exploration._metric import score_exploration  # noqa: E402


def _path_length(poses: list[list[float]]) -> float:
    total = 0.0
    for i in range(1, len(poses)):
        total += math.hypot(poses[i][0] - poses[i - 1][0], poses[i][1] - poses[i - 1][1])
    return total


def score_state(
    state: dict,
    scene_dir: Path,
    *,
    sense_range_m: float = 8.0,
    n_rays: int = 120,
    done_reason: str = "live",
) -> dict:
    ply = scene_dir / "traversable_area.ply"
    if not ply.is_file():
        raise FileNotFoundError(ply)

    resolution = float(state.get("resolution", 0.2))
    world = GTWorld.from_traversable_ply(ply, resolution=resolution)

    mapped = {tuple(c) for c in state.get("free_cells", [])}
    poses = state.get("pose_trace") or []
    if not poses:
        poses = [[w["x"], w["y"]] for w in state.get("visited", [])]

    seen: set[tuple[int, int]] = set()
    for xy in poses:
        if len(xy) < 2:
            continue
        seen |= world.visible_gt_cells(
            float(xy[0]), float(xy[1]), max_range_m=sense_range_m, n_rays=n_rays
        )

    visited = state.get("visited") or []
    path_len = _path_length(poses)
    if path_len < 1e-6 and len(visited) >= 2:
        path_len = _path_length([[w["x"], w["y"]] for w in visited])

    score = score_exploration(
        gt_free=world.free,
        seen_gt=seen,
        mapped_free=mapped,
        path_length_m=path_len,
        n_visited_waypoints=int(state.get("n_visited", len(visited))),
        ticks=len(poses),
        done_reason=done_reason,
    )
    return {
        "gt_coverage": score.gt_coverage,
        "mapped_coverage": score.mapped_coverage,
        "path_length_m": score.path_length_m,
        "coverage_per_meter": score.coverage_per_meter,
        "n_gt_free": score.n_gt_free,
        "n_seen": score.n_seen,
        "n_mapped": score.n_mapped,
        "n_visited_waypoints": score.n_visited_waypoints,
        "ticks": score.ticks,
        "done_reason": score.done_reason,
        "skipped": int(state.get("skipped", 0)),
        "resolution": resolution,
        "scene": scene_dir.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--done-reason", default="live")
    args = ap.parse_args()

    state = json.loads(args.state.read_text())
    result = score_state(state, args.scene, done_reason=args.done_reason)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
