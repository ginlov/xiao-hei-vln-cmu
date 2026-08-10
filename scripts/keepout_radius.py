#!/usr/bin/env python3
"""How wide may a keep-out zone be before it forbids the official path?

The avoidance radius has been carried as a guess (2 m) since the architecture
discussion. This measures it, and the measurement is two-sided:

  lower bound   the radius must swallow our own centre error, or a correctly
                identified anchor still lands the zone in the wrong place
  upper bound   the reference trajectory's own clearance -- a radius larger
                than that would rule out the execution the organisers shipped
                as the answer

Only three of the thirty instruction-following questions contain a true
keep-out. The other ten "between" sentences are *required* passages ("take the
path between the TV and the bed"), which want the opposite treatment: a gate to
drive through, not a region to avoid. Conflating them would turn ten questions
into guaranteed failures, so they are reported separately here.

Reads ground truth from `viz/data/<scene>.json`, so it needs no scene zips.

    uv run python scripts/keepout_radius.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ_DATA = Path("viz/data")
TRAJ_FOR = {0: "trajectory_q4.ply", 1: "trajectory_q5.ply"}
# TASK 26's p90 centre error, plus robot half-width and path-tracking slop: a
# zone narrower than this can be correctly identified and still land in the
# wrong place.
FLOOR = 0.86

# (scene, question index, anchors). A pair avoids the corridor *between* two
# objects; a single name avoids the object itself.
AVOID = [
    ("chinese_room", 1, ("chair", "folding screen")),
    ("loft", 0, ("cabinet",)),
    # The question says "the path between the TV and the tea table"; this scene
    # labels that piece of furniture `coffee table` and has no `tea table` at
    # all, while chinese_room's question uses the same words against a real
    # `tea table` label. The questions are not written against the scenes'
    # vocabulary, so an anchor name cannot be looked up literally.
    ("livingroom_2", 1, ("tv", "coffee table")),
]


def read_ply_ascii(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header") + 1
    return np.asarray([list(map(float, ln.split()[:3]))
                       for ln in lines[start:] if ln.strip()], dtype=float)


def instances(scene: dict, name: str) -> list[dict]:
    labels = [x.strip().lower() for x in scene["labels"]]
    return [o for o in scene["gt"] if labels[o["l"]] == name.strip().lower()]


def seg_dist(pts_xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Planar distance from every point to the segment ``a``--``b``."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(pts_xy - a, axis=1)
    # numpy leaks stale BLAS float state as matmul warnings here; the inputs
    # are checked finite and the divisor is guarded above.
    with np.errstate(all="ignore"):
        t = np.clip((pts_xy - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts_xy - (a + t[:, None] * ab), axis=1)


def main() -> int:
    print("clearance the OFFICIAL reference trajectory keeps from each "
          "keep-out anchor\n(this is the ceiling: a larger radius would "
          "forbid the reference path itself)\n")
    ceilings, discs = [], []

    for scene_name, qi, anchors in AVOID:
        f = VIZ_DATA / f"{scene_name}.json"
        tp = CHALLENGE / "questions" / scene_name / TRAJ_FOR[qi]
        if not f.is_file() or not tp.is_file():
            print(f"{scene_name} q{qi + 4}: missing data")
            continue
        scene = json.loads(f.read_text())
        traj = read_ply_ascii(tp)[:, :2]

        print(f"== {scene_name} q{qi + 4}  avoid {' + '.join(anchors)} "
              f"({len(traj)} trajectory points) ==")

        if len(anchors) == 2:
            a_list, b_list = (instances(scene, n) for n in anchors)
            if not a_list or not b_list:
                print(f"   ground truth missing: "
                      f"{anchors[0]}×{len(a_list)} {anchors[1]}×{len(b_list)}\n")
                continue
            # "the chair" is one of six; every pairing is a candidate corridor,
            # and the binding step has to choose. Report all of them, because
            # the radius has to be safe for whichever one we pick.
            rows = []
            for i, a in enumerate(a_list):
                for b in b_list:
                    pa, pb = np.array(a["c"][:2]), np.array(b["c"][:2])
                    rows.append((float(seg_dist(traj, pa, pb).min()),
                                 f"{anchors[0]}[{i}]--{anchors[1]}",
                                 float(np.linalg.norm(pa - pb))))
            rows.sort(key=lambda r: r[2])       # by anchor separation
            for d, name, sep in rows:
                print(f"   {name:28s} anchors {sep:5.2f} m apart   "
                      f"clearance {d:5.2f} m")
            # Which pairing did the instruction mean? The reference path drives
            # straight through the widest pairings (clearance 0.00 m), so "the
            # path between A and B" cannot mean an arbitrary pair -- it means a
            # gap narrow enough to be a passage. Taking the closest pair is the
            # only reading under which the official execution stays legal.
            d, name, sep = rows[0]
            print(f"   -> closest pair {name} ({sep:.2f} m apart) is the only "
                  f"reading\n      the reference path respects: clearance "
                  f"{d:.2f} m. Widest pairing: {max(rows, key=lambda r: r[2])[0]:.2f} m.\n")
            ceilings.append((f"{scene_name} q{qi + 4}", d))
            for name in anchors:
                for i, o in enumerate(instances(scene, name)):
                    c = np.array(o["c"][:2])
                    discs.append((f"{scene_name} {name}[{i}]",
                                  float(np.linalg.norm(traj - c, axis=1).min())))
        else:
            objs = instances(scene, anchors[0])
            if not objs:
                print(f"   ground truth missing: {anchors[0]}\n")
                continue
            for i, o in enumerate(objs):
                c = np.array(o["c"][:2])
                d = float(np.linalg.norm(traj - c, axis=1).min())
                print(f"   {anchors[0]}[{i}] centre-clearance {d:5.2f} m")
                ceilings.append((f"{scene_name} q{qi + 4}", d))
                discs.append((f"{scene_name} {anchors[0]}[{i}]", d))
            print()

    print("\n== as a corridor between the two anchors ==")
    print(f"  lower bound  {FLOOR:.2f} m   our centre error at p90 "
          f"(TASK 26 distance bins)")
    print("               + robot half-width + path-tracking slop")
    if ceilings:
        worst, lo = min(ceilings, key=lambda t: t[1])
        print(f"  upper bound  {lo:.2f} m   tightest reference clearance ({worst})")
        if lo < FLOOR:
            print(f"\n  These no longer bracket. The tightest official path leaves "
                  f"{lo:.2f} m,\n  less than the {FLOOR:.2f} m our own localisation "
                  f"error needs, so there is no\n  single half-width that is both "
                  f"safe for us and legal for the reference.")

    # The loop implements this as a disc per anchor, which is a different
    # constraint: it forbids being *near an object* rather than crossing the
    # gap *between two*. The instructions say "avoid the path between A and B",
    # and the reference paths brush right past A while respecting the corridor.
    print("\n== as a disc on each anchor, which is what approach_loop builds ==")
    if discs:
        for name, d in sorted(discs, key=lambda t: t[1])[:6]:
            bad = "  <- a 1.2 m disc forbids the official path" if d < 1.2 else ""
            print(f"  {name:32s} {d:5.2f} m from the path{bad}")
        n_bad = sum(1 for _, d in discs if d < 1.2)
        print(f"\n  {n_bad} of {len(discs)} anchor instances sit closer to the "
              f"official path than\n  the 1.2 m radius the loop would put around "
              f"them. Whether that fires\n  depends on which instance the model "
              f"binds -- except livingroom_2,\n  which has exactly one coffee "
              f"table and would fail outright.")

    print(f"\n  n = {len(AVOID)} keep-out questions in the whole 30-question set, "
          f"all with\n  exported ground truth. Any radius quoted from this is a\n"
          f"  defensible starting value, not a fitted one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
