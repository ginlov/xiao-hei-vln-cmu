#!/usr/bin/env python3
"""How many of the official relations rest on an anchor we cannot lift?

`resolve_relation` settles "closest to X" by lifting X and measuring. It needs
at least one anchor with lidar returns in a cone a few degrees wide. On loft the
anchor is a TV remote — 0.05 x 0.20 x 0.02 m, six metres away — and it came back
with zero anchors on every call of every run, so the comparison was never made
and the loop bound the wrong cup.

Anchor size, not target size, is the gate on that whole mechanism. This measures
it across the official question set, from the start pose the robot actually
begins each question at.

Covers the seven scenes with exported ground truth (`viz/data/<scene>.json`).

    uv run python scripts/anchor_size.py
    uv run python scripts/anchor_size.py --verbose
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ = Path("viz/data")
# The robot begins every question at the map origin, camera 0.75 m up.
START = np.array([0.0, 0.0, 0.75])

RELATION = re.compile(
    r"\b(?:closest to|nearest to|closest|nearest|farthest from|furthest from|"
    r"farthest|furthest|between|near|next to|beside|with)\b", re.I)

STOP = {"the", "a", "an", "of", "on", "in", "it", "that", "is", "are", "and",
        "then", "to", "two", "second", "one", "other", "same", "its", "his",
        "her", "their", "this", "these", "those", "there"}


def anchor_phrases(sentence: str) -> list[str]:
    """The noun phrases a relation word points at.

    Deliberately crude: everything from the relation word to the next clause
    break. A relation whose anchor this misses is reported as unmatched rather
    than silently scored.
    """
    out = []
    for m in RELATION.finditer(sentence):
        tail = sentence[m.end():]
        tail = re.split(r"[,.]| then | and then | avoid", tail)[0]
        if tail.strip():
            out.append(tail.strip())
    return out


def gt_matches(phrase: str, gt: list, labels: list) -> list[int]:
    """Indices whose label the phrase names, preferring the most specific.

    Token overlap alone is wrong here in the one direction that matters: "the
    TV remote" overlaps both `tv remote` and `tv`, and taking either would
    silently replace a 0.20 m anchor with a 1.02 m one — exactly the case this
    script exists to count. So multi-word labels that appear whole in the
    phrase win, and single tokens are the fallback.
    """
    text = re.sub(r"[^a-z0-9 ]", " ", phrase.lower())
    text = " " + " ".join(text.split()) + " "
    whole = [(len(labels[g["l"]]), i) for i, g in enumerate(gt)
             if not g.get("struct")
             and f" {labels[g['l']].lower()} " in text]
    if whole:
        longest = max(w[0] for w in whole)
        return [i for n, i in whole if n == longest]

    toks = [t for t in text.split() if t not in STOP]
    return [i for i, g in enumerate(gt)
            if not g.get("struct")
            and any(t == labels[g["l"]].lower() for t in toks)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qs = json.loads((CHALLENGE / "questions/questions.json").read_text())
    qs = qs if isinstance(qs, list) else qs["questions"]

    rows, unmatched = [], 0
    for entry in qs:
        scene = entry["scene"]
        f = VIZ / f"{scene}.json"
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        gt, labels = d["gt"], d["labels"]
        for kind, sentences in entry["questions"].items():
            for s in sentences:
                for phrase in anchor_phrases(s):
                    hits = gt_matches(phrase, gt, labels)
                    if not hits:
                        unmatched += 1
                        continue
                    # The nearest instance is the one the robot would try to
                    # lift first, and the most favourable case for us.
                    best = min(hits, key=lambda i: np.linalg.norm(
                        np.array(gt[i]["c"]) - START))
                    g = gt[best]
                    size = np.array(g["hi"]) - np.array(g["lo"])
                    rng = float(np.linalg.norm(np.array(g["c"]) - START))
                    # What the box subtends at the start pose, using the largest
                    # face-on dimension — the generous reading.
                    ang = float(np.rad2deg(2 * np.arctan(max(size[:2].max(),
                                                             size[2]) / 2 / rng)))
                    rows.append({"scene": scene, "kind": kind, "phrase": phrase,
                                 "label": labels[g["l"]], "size": size,
                                 "range": rng, "ang": ang})

    rows.sort(key=lambda r: r["ang"])
    print(f"{len(rows)} anchors matched to ground truth across "
          f"{len({r['scene'] for r in rows})} scenes ({unmatched} phrases "
          f"matched nothing and are not scored)\n")

    if args.verbose:
        print(f"{'scene':14s} {'anchor':22s} {'largest':>8s} {'range':>7s} {'subtends':>9s}")
        for r in rows:
            print(f"{r['scene']:14s} {r['label'][:22]:22s} "
                  f"{max(r['size'][:2].max(), r['size'][2]):8.2f} "
                  f"{r['range']:7.2f} {r['ang']:8.2f}°")
        print()

    # Calibrated on what was observed, not chosen in advance:
    #   tv remote     0.20 m at 6.40 m -> 1.8 deg   0 anchors on every call
    #   fan decor.    0.78 m at 1.65 m -> 26  deg   lifted, comparison made
    #   projector scr 1.35 m at 4.63 m -> 16  deg   lifted to 0.02 m
    for lo, hi, tag in ((0, 2, "below the TV remote — no chance"),
                        (2, 5, "marginal: a 1-5 deg box is 6-30 px on a face"),
                        (5, 1e9, "comfortably liftable")):
        sel = [r for r in rows if lo <= r["ang"] < hi]
        print(f"{len(sel):3d} / {len(rows)}  {100*len(sel)/len(rows):4.0f}%   "
              f"subtends {lo}-{hi if hi < 1e9 else '∞'}°   {tag}")
        for r in sel[:6] if lo < 5 else []:
            print(f"        {r['scene']}: {r['label']!r} "
                  f"{max(r['size'][:2].max(), r['size'][2]):.2f} m at "
                  f"{r['range']:.1f} m = {r['ang']:.1f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
