"""Box-quality metrics for scored scene graphs.

``replay_score.py`` reports distance-based metrics: mAP/precision/recall all ask
whether a node's *centre* is close to a GT centre. They are nearly blind to box
extent — a 2 m³ contaminated box and a correct 0.01 m³ box score identically
when their centres are equally close. That blindness hid a large-box regression
for the whole first round of lifter work (docs/tasks/backlog.md, B1).

This scores the extent instead, over one or more scored output directories:

  over_1m3 / over_5m3   the TAIL — what "the boxes are too big" actually means
  max_volume            the single worst node
  inflation             median (pred volume / median GT volume for that label).
                        Guards against the opposite failure: a filter that
                        fixes the tail by eating the object. Read it as a
                        RELATIVE signal between arms, not a 1.0 target --
                        LiDAR only samples surfaces facing the robot, so a
                        correct lift sits below 1.0 by construction.
  pathology_rate        share of nodes above 10x their label's GT volume

Accept a change only if the tail improves AND inflation does not collapse AND
recall (from replay_score) does not regress. Optimising the tail alone selects
for over-aggressive filters.

    uv run python perception_benchmark/box_quality.py \
        --label baseline perception_benchmark/scores \
        --label voxel-0.08 /path/to/other/scores
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from replay_score import scoreable
from viewgen import load_objects

CAP_DIR = Path(os.environ.get("PERCEPTION_CAP_DIR",
                              "perception_benchmark/captures"))


def gt_volumes(scene: str) -> dict[str, list[float]]:
    """Median GT volume per label, for the scoreable objects only."""
    out: dict[str, list[float]] = {}
    for e in load_objects(scene).values():
        if scoreable(e.label):
            out.setdefault(e.label, []).append(e.size.x * e.size.y * e.size.z)
    return out


def gt_centres(scene: str) -> dict[int, np.ndarray]:
    return {i: np.array([e.center.x, e.center.y, e.center.z])
            for i, e in load_objects(scene).items() if scoreable(e.label)}


def score_dir(path: Path, scenes: list[str]) -> dict:
    vols: list[float] = []
    inflation: list[float] = []
    dup_objects = redundant = 0
    for scene in scenes:
        f = path / f"{scene}_scene.json"
        if not f.is_file():
            continue
        gt = gt_volumes(scene)
        with open(f) as fh:
            nodes = json.load(fh)["objects"]
        # duplicates: GT objects claimed by more than one node (nearest within 0.5 m)
        claims: dict[int, int] = {}
        centres = gt_centres(scene)
        for node in nodes:
            c = np.array(node["center_3d"])
            if centres:
                i = min(centres, key=lambda k: np.linalg.norm(centres[k] - c))
                if np.linalg.norm(centres[i] - c) <= 0.5:
                    claims[i] = claims.get(i, 0) + 1
        dup_objects += sum(1 for v in claims.values() if v > 1)
        redundant += sum(v - 1 for v in claims.values() if v > 1)
        for node in nodes:
            v = float(np.prod(node["bbox_aabb"]["size"]))
            vols.append(v)
            ref = gt.get(node["label"])
            if ref and v > 0:
                inflation.append(v / max(float(np.median(ref)), 1e-6))
    if not vols:
        return {}
    a, inf = np.array(vols), np.array(inflation)
    return {
        "nodes": len(a),
        "over_1m3": int((a > 1).sum()),
        "over_5m3": int((a > 5).sum()),
        "max_volume": float(a.max()),
        "median_volume": float(np.median(a)),
        "p95_volume": float(np.percentile(a, 95)),
        "inflation": float(np.median(inf)) if inf.size else None,
        "pathology_rate": float(np.mean(inf > 10)) if inf.size else None,
        "dup_objects": dup_objects,
        "redundant": redundant,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", nargs=2, metavar=("NAME", "DIR"),
                    required=True, help="repeatable: a name and a scored output dir")
    args = ap.parse_args()

    scenes = sorted(os.path.basename(os.path.dirname(m))
                    for m in glob.glob(str(CAP_DIR / "*" / "manifest.json")))
    print(f"scenes: {len(scenes)}\n")
    hdr = (f"{'arm':16s}{'nodes':>7s}{'>1m3':>7s}{'>5m3':>7s}"
           f"{'max':>10s}{'inflate':>9s}{'pathol':>8s}{'dupObj':>8s}{'redund':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name, d in args.label:
        m = score_dir(Path(d), scenes)
        if not m:
            print(f"{name:16s}  (no scored scenes in {d})")
            continue
        print(f"{name:16s}{m['nodes']:7d}{m['over_1m3']:7d}{m['over_5m3']:7d}"
              f"{m['max_volume']:10.1f}{m['inflation']:9.2f}"
              f"{100 * m['pathology_rate']:7.1f}%{m['dup_objects']:8d}{m['redundant']:8d}")
    print("\ninflation 1.0 = boxes match GT volume; <1 means the filter is "
          "eating the object, >1 means contamination remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
