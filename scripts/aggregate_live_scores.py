#!/usr/bin/env python3
"""Aggregate live_rviz per-run score JSONs into mean/median rankings."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("exploration_logs/live_rviz"),
        help="Directory with <scene>/*_score.json files",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    by_algo: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []
    for path in sorted(args.root.glob("*/*_score.json")):
        data = json.loads(path.read_text())
        # filename: {algo}_{status}_{stamp}_score.json or {algo}_score.json
        name = path.name
        algo = name.split("_")[0]
        cov = float(data["gt_coverage"])
        by_algo[algo].append(cov)
        rows.append({"scene": path.parent.name, "algo": algo, "path": str(path), **data})

    ranking = []
    for algo, vals in by_algo.items():
        ranking.append(
            {
                "algo": algo,
                "n": len(vals),
                "mean_gt_coverage": statistics.mean(vals),
                "median_gt_coverage": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    ranking.sort(key=lambda r: -r["mean_gt_coverage"])

    out = {
        "metric": "live gt_coverage (pose_trace raycast ∩ GT_free) / |GT_free|",
        "n_scores": len(rows),
        "ranking": ranking,
        "runs": rows,
    }
    text = json.dumps(out, indent=2)
    print(text)
    out_path = args.out or (args.root / "live_summary.json")
    out_path.write_text(text + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
