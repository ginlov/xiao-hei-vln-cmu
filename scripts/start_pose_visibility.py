#!/usr/bin/env python3
"""How much of an instruction can be answered from the pose the robot starts in?

The organizers ruled out a separate exploration phase for instruction
following: the robot's motion is the answer. That makes one number decide the
architecture -- of the objects an instruction names, how many are already
detected before the robot has moved at all. If most are, the path can be
planned up front and refined en route; if few are, the path itself has to do
the discovering, and every waypoint is a bet.

We start with an advantage worth measuring rather than assuming: the camera is
a 360-degree equirect, so the first frame is a full panorama. Nothing has to be
turned toward.

Both corpora and reference trajectories start at (0, 0, 0.75), so frame 0 of a
recorded tour is the pose the challenge would start from.

Reads `viz/data/<scene>.json` (from `scripts/export_viz.py`), so it runs
locally with no scene zips.

    uv run python scripts/start_pose_visibility.py
    uv run python scripts/start_pose_visibility.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vlm_sweep import family_key  # noqa: E402
from traj_tolerance import STOPWORDS, mentioned  # noqa: E402  (same directory)

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ_DATA = Path("viz/data")
# A detection counts as localising a named object when its box centre lands
# within this of the annotation's -- the tolerance the reference trajectories
# themselves keep to named objects (median 0.58 m, p75 0.81 m).
HIT_M = 1.0


def label_of(data, i):
    return data["labels"][i].strip().lower()


def key_of(lab: str, families: bool):
    """Identity used to match a detection to an annotation.

    Exact strings understate what the robot can see: in arabic_room's first
    frame a `coffee table` detection sits 0.17 m from the annotated `table` and
    a `lamp` 0.07 m from the annotated `ceiling lamp`. The object is visible and
    well localised; only the word differs. `family_key` folds the synonym sets
    read off the label confusion matrix in TASK 23.
    """
    return family_key(lab) if families else lab


def detected_labels(data, upto_frame: int, families: bool) -> dict:
    """label -> detection box centres, over frames [0, upto_frame]."""
    out: dict = {}
    for f in data["frames"][:upto_frame + 1]:
        for d in f["dets"]:
            c = (np.array(d["lo"]) + np.array(d["hi"])) / 2
            out.setdefault(key_of(label_of(data, d["l"]), families), []).append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenge", default=str(CHALLENGE))
    ap.add_argument("--viz-data", default=str(VIZ_DATA))
    ap.add_argument("--families", action="store_true",
                    help="match detections to annotations through the TASK 23 "
                         "synonym families instead of exact label strings")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.challenge)
    qs = {e["scene"]: e["questions"]
          for e in json.loads((root / "questions/questions.json").read_text())}
    viz = Path(args.viz_data)

    # Frame 0 is the start pose; the later cuts show what a little motion buys.
    cuts = [0, 5, 15, 40]
    tally = {c: [0, 0] for c in cuts}          # [found, total]
    rows = []

    for scene in sorted(qs):
        vf = viz / f"{scene}.json"
        if not vf.is_file():
            continue
        data = json.loads(vf.read_text())
        labels = {label_of(data, i) for i in range(len(data["labels"]))}
        gt_by: dict = {}
        for o in data["gt"]:
            gt_by.setdefault(key_of(label_of(data, o["l"]), args.families),
                             []).append(np.array(o["c"]))

        seen_at = {c: detected_labels(data, c, args.families) for c in cuts}
        for qi, question in enumerate(qs[scene]["instruction_following"]):
            names = [n for n in mentioned(question, labels)
                     if key_of(n, args.families) in gt_by]
            per_cut = {}
            for c in cuts:
                found = []
                for lab in names:
                    k = key_of(lab, args.families)
                    cands = seen_at[c].get(k, [])
                    ok = any(float(np.linalg.norm(g - d)) <= HIT_M
                             for g in gt_by[k] for d in cands)
                    if ok:
                        found.append(lab)
                per_cut[c] = found
                tally[c][0] += len(found)
                tally[c][1] += len(names)
            rows.append({"scene": scene, "q": qi + 4, "names": names,
                         "per_cut": per_cut, "text": question})

    print(f"named objects localised within {HIT_M} m, by how many frames the "
          f"robot has taken\n")
    hdr = f"{'scene':14s} {'q':>2s} {'named':>5s} " + " ".join(
        f"{'f0' if c == 0 else f'f0-{c}':>7s}" for c in cuts)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        n = len(r["names"])
        cells = " ".join(f"{len(r['per_cut'][c]):3d}/{n:<3d}" for c in cuts)
        print(f"{r['scene']:14s} {r['q']:2d} {n:5d} {cells}")
        if args.verbose:
            missing = [x for x in r["names"] if x not in r["per_cut"][0]]
            print(f"{'':18s}{r['text']}")
            print(f"{'':18s}not visible at start: "
                  f"{', '.join(missing) if missing else '(none)'}")

    print()
    for c in cuts:
        f, t = tally[c]
        tag = "at the start pose" if c == 0 else f"after {c} frames"
        print(f"  {tag:22s} {f:3d}/{t:3d} = {f / max(t, 1):5.1%}")

    whole = sum(1 for r in rows if len(r["per_cut"][0]) == len(r["names"]))
    print(f"\n  questions whose every named object is already localised at the "
          f"start: {whole}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
