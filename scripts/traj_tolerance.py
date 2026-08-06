#!/usr/bin/env python3
"""How close does the reference trajectory actually come to the objects it names?

Instruction-following is 6 of the 17 points available per scene -- 70% of the
challenge score -- and it is answered with waypoints, not boxes. So the
question that decides how much perception it needs is not "what is our IoU" but
"how near is near": if the reference path passes within a metre of the objects
its instruction names, our measured 0.236 m centre error is already sufficient
and box quality is irrelevant to this question type.

The challenge ships `trajectory_q4.ply` and `trajectory_q5.ply` per scene --
the two instruction-following questions, densely sampled at robot height. This
measures, for every object the instruction mentions, how close that path comes
to the object's ground-truth box.

Reads the ground-truth objects out of `viz/data/<scene>.json` (produced by
`scripts/export_viz.py`), so it needs no scene zips and runs anywhere.

    uv run python scripts/traj_tolerance.py
    uv run python scripts/traj_tolerance.py --scene loft --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ_DATA = Path("viz/data")
# questions.json lists, per scene, 1 numerical + 2 object_reference +
# 2 instruction_following; the shipped trajectories are named q4 and q5, which
# is the 4th and 5th question in that order.
TRAJ_FOR = {0: "trajectory_q4.ply", 1: "trajectory_q5.ply"}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "at", "in", "on", "near", "by",
    "go", "then", "first", "stop", "take", "path", "between", "with", "that",
    "is", "are", "from", "next", "avoid", "avoiding", "through", "toward",
    "towards", "past", "along", "your", "you", "it", "its", "two", "second",
    "one", "closest", "furthest", "farthest", "nearest", "left", "right",
    "side", "other", "same", "small", "large", "big", "little", "front",
    "back", "behind", "under", "above", "below", "over", "beside", "around",
    "start", "starting", "end", "ending", "walk", "move", "head", "turn",
    "while", "without", "not", "do", "does", "get", "come", "coming", "make",
    "sure", "keep", "stay", "continue", "proceed", "finally", "lastly", "but",
    "which", "where", "when", "there", "here", "this", "these", "those",
}


def read_ply_ascii(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header") + 1
    pts = [list(map(float, ln.split()[:3])) for ln in lines[start:] if ln.strip()]
    return np.asarray(pts, dtype=float)


def point_to_box_xy(p_xy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Planar distance from a point to a box footprint; 0 inside it.

    Planar because the robot drives on the floor: its height never approaches a
    ceiling lamp's, and counting that vertical gap would report every overhead
    object as far away when the robot passed directly beneath it.
    """
    d = np.maximum(np.maximum(lo[:2] - p_xy, p_xy - hi[:2]), 0.0)
    return float(np.hypot(d[0], d[1]))


def traj_to_box(traj: np.ndarray, lo, hi) -> tuple[float, int]:
    d = [point_to_box_xy(p[:2], lo, hi) for p in traj]
    i = int(np.argmin(d))
    return d[i], i


def mentioned(question: str, labels: set[str]) -> list[str]:
    """Ground-truth labels the instruction names, longest match first.

    Matching the annotation's own vocabulary rather than parsing noun phrases:
    the question says "the columns" and the scene calls them `column`, so a
    singular/plural fold over the real label set is both simpler and less
    likely to invent an object that is not there.
    """
    q = " " + re.sub(r"[^a-z ]", " ", question.lower()) + " "
    q = re.sub(r"\s+", " ", q)
    hits = []
    for lab in sorted(labels, key=len, reverse=True):
        if lab in STOPWORDS:
            continue
        for form in (lab, lab + "s", lab + "es"):
            if f" {form} " in q:
                hits.append(lab)
                break
    # drop labels wholly contained in a longer hit ("table" inside "side table")
    return [h for h in hits if not any(h != o and h in o for o in hits)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="all")
    ap.add_argument("--challenge", default=str(CHALLENGE))
    ap.add_argument("--viz-data", default=str(VIZ_DATA))
    ap.add_argument("--verbose", action="store_true",
                    help="list every named object, not just the summary")
    args = ap.parse_args()

    root = Path(args.challenge)
    qs = {e["scene"]: e["questions"]
          for e in json.loads((root / "questions/questions.json").read_text())}
    viz = Path(args.viz_data)

    scenes = sorted(qs) if args.scene == "all" else [args.scene]
    rows, per_q = [], []
    for scene in scenes:
        vf = viz / f"{scene}.json"
        if not vf.is_file():
            continue
        data = json.loads(vf.read_text())
        labels = {n.strip().lower() for n in data["labels"]}
        objs = [{"label": data["labels"][o["l"]].strip().lower(),
                 "lo": np.array(o["lo"]), "hi": np.array(o["hi"]),
                 "c": np.array(o["c"])} for o in data["gt"]]
        by_label: dict[str, list] = {}
        for o in objs:
            by_label.setdefault(o["label"], []).append(o)

        for qi, question in enumerate(qs[scene]["instruction_following"]):
            tp = root / "questions" / scene / TRAJ_FOR.get(qi, "")
            if not tp.is_file():
                continue
            traj = read_ply_ascii(tp)
            seg = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
            names = mentioned(question, labels)

            named_d = []
            for lab in names:
                cands = by_label.get(lab, [])
                if not cands:
                    continue
                # The instruction names one of possibly several; the nearest is
                # the one the path was drawn to, so it bounds the tolerance.
                d = min(traj_to_box(traj, o["lo"], o["hi"])[0] for o in cands)
                named_d.append((lab, d, len(cands)))
                rows.append(d)

            # How close the path comes to everything else, for contrast: if the
            # named objects are no nearer than the rest, naming is not what the
            # trajectory tracks and this whole measure means nothing.
            others = [traj_to_box(traj, o["lo"], o["hi"])[0]
                      for o in objs if o["label"] not in names]
            per_q.append({
                "scene": scene, "q": qi + 4, "len": seg, "n_pts": len(traj),
                "named": named_d, "others": others, "text": question,
            })

    print(f"{'scene':14s} {'q':>2s} {'path m':>7s}  "
          "named objects (min planar distance to box)")
    print("-" * 100)
    for r in per_q:
        nd = "  ".join(f"{lab}={d:.2f}" + (f"[{n}]" if n > 1 else "")
                       for lab, d, n in r["named"]) or "(none matched)"
        print(f"{r['scene']:14s} {r['q']:2d} {r['len']:7.1f}  {nd}")
        if args.verbose:
            print(f"{'':18s}{r['text']}")

    d = np.array(rows)
    if len(d):
        print(f"\n{len(d)} named-object distances over "
              f"{len({r['scene'] for r in per_q})} scenes")
        for q in (50, 75, 90, 95, 100):
            print(f"  p{q:<4d} {np.percentile(d, q):.2f} m")
        print(f"  mean  {d.mean():.2f} m     within 1.0 m: "
              f"{(d <= 1.0).mean():.0%}     within 1.5 m: {(d <= 1.5).mean():.0%}")

    allo = np.concatenate([r["others"] for r in per_q if r["others"]])
    print(f"\nfor contrast, the {len(allo)} objects the instructions do NOT name:")
    print(f"  median {np.median(allo):.2f} m   "
          f"within 1.0 m: {(allo <= 1.0).mean():.0%}")
    print("\nIf named objects are not markedly closer than unnamed ones, the "
          "path\ndoes not track the named objects and this measure proves "
          "nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
