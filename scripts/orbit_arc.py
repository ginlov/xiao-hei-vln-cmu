#!/usr/bin/env python3
"""How much of an orbit does each object-reference target actually allow?

`answer_reference` orbits the target in fixed steps -- 75 degrees five times,
or whatever `--orbit-deg` says. That presumes a ring of standable floor around
the object, and these targets do not have one: they are small things, and small
things sit against walls and on furniture. loft's potted plant is 0.06 m from a
wall, 0.16 m from a vase and 0.21 m from the TV cabinet.

So this measures, for every released question, the azimuths on a ring of radius
R where a robot of radius `--robot` would fit, against the VLA-3D boxes that
fall in its height band. Two numbers per radius: how much of the ring is free
at all, and the longest CONTIGUOUS arc -- only the second one is an orbit.

    uv run python scripts/score_reference.py --json key.json
    uv run python scripts/orbit_arc.py key.json

An AABB is only the object when the object is roughly convex, and VLA-3D's
`unknown` and L-shaped `wall` annotations are neither: japanese_room has an
`unknown` 8.6 x 10.7 m and home_building_1 one 57 x 55 m, whose boxes cover the
whole scene. Keeping them made six targets look unreachable at every radius,
which is an artefact of the annotation rather than the room. They are dropped,
which makes every number here OPTIMISTIC -- a lower bound on how blocked the
ring is, never an upper one. The simulator's own terrain map is the honest
source, and it only exists where the robot has already driven.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from score_numerical import load  # noqa: E402

# Structure the robot drives on or under, rather than into.
SKIP = {"floor", "ceiling", "rug", "carpet", "mat"}
# A body that tall clears nothing; a box entirely below or above it is not in
# the way. 0.15 m is the wheels, 1.2 m the top of the sensor mast.
Z_LO, Z_HI = 0.15, 1.2
MAX_FOOTPRINT_M2 = 4.0      # past this an AABB is a room, not an object
WALL_MAX_THICKNESS_M = 0.5  # past this a `wall` box is an L or a U
STEP_DEG = 10


def obstacles(objs: list[dict], target_id: str) -> list[tuple]:
    """Footprints a robot would hit, with the un-boxable ones dropped."""
    out = []
    for o in objs:
        if o["id"] == target_id or o["label"].lower() in SKIP:
            continue
        if o["hi"][2] < Z_LO or o["lo"][2] > Z_HI:
            continue
        w, d = float(o["sz"][0]), float(o["sz"][1])
        if o["label"].lower() == "wall":
            if min(w, d) > WALL_MAX_THICKNESS_M:
                continue
        elif w * d > MAX_FOOTPRINT_M2:
            continue
        out.append((np.asarray(o["lo"][:2]), np.asarray(o["hi"][:2])))
    return out


def free_bins(centre: np.ndarray, obs: list[tuple], radius: float,
              robot: float) -> list[int]:
    """The azimuths, in degrees, where the robot's disc clears every box."""
    free = []
    for a in range(0, 360, STEP_DEG):
        p = centre + radius * np.array([np.cos(np.radians(a)),
                                        np.sin(np.radians(a))])
        if all(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))) >= robot
               for lo, hi in obs):
            free.append(a)
    return free


def longest_arc(free: list[int]) -> int:
    """The longest contiguous run of free azimuths, in degrees, wrapping.

    The number that matters: a ring that is 200 degrees free in four scattered
    pieces is not an orbit, it is four viewpoints that each need their own
    drive around the furniture.
    """
    if not free:
        return 0
    s = set(free)
    n_bins = 360 // STEP_DEG
    if len(s) == n_bins:
        return 360
    best = 0
    for a in free:
        if (a - STEP_DEG) % 360 in s:
            continue                       # not the start of a run
        n, b = 0, a
        while b % 360 in s and n <= n_bins:
            n, b = n + 1, b + STEP_DEG
        best = max(best, n)
    return best * STEP_DEG


def census(key: list[dict], radii: tuple[float, ...], robot: float) -> list[dict]:
    rows, cache = [], {}
    for r in key:
        res = r.get("resolved") or {}
        if res.get("n") != 1:
            continue                        # no single target to orbit
        objs = cache.setdefault(r["scene"], load(r["scene"]))
        t = next((o for o in objs if o["id"] == res.get("id")), None)
        if t is None:
            continue
        obs = obstacles(objs, t["id"])
        c = np.asarray(t["c"][:2], float)
        row = {"scene": r["scene"], "i": r["i"], "label": t["label"], "arc": {}}
        for R in radii:
            f = free_bins(c, obs, R, robot)
            row["arc"][R] = (len(f) * STEP_DEG, longest_arc(f))
        rows.append(row)
    return rows


def report(rows: list[dict], radii: tuple[float, ...]) -> None:
    head = f"{'scene':<18}{'q':<3}{'target':<17}"
    print(head + "".join(f"{f'r={R:.1f}  free / arc':>21}" for R in radii))
    for row in sorted(rows, key=lambda r: max(r["arc"][R][1] for R in radii)):
        print(f"{row['scene']:<18}{row['i']:<3}{row['label'][:16]:<17}"
              + "".join(f"{row['arc'][R][0]:>12}°{row['arc'][R][1]:>7}°"
                        for R in radii))
    print(f"\n{len(rows)} targets — the second number is the orbit")
    for R in radii:
        a = np.array([r["arc"][R][1] for r in rows])
        print(f"  r={R:.1f} m   median {np.median(a):3.0f}°"
              f"   >=180° for {(a >= 180).sum():2d}/{len(a)}"
              f"   <=90° for {(a <= 90).sum():2d}/{len(a)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", help="written by score_reference.py --json")
    ap.add_argument("--radius", type=float, nargs="+",
                    default=[1.2, 1.5, 1.8],
                    help="standoffs to test (default: the shipped 1.8 and two nearer)")
    ap.add_argument("--robot", type=float, default=0.35, help="robot radius, m")
    ap.add_argument("--json", metavar="PATH", help="write the census")
    a = ap.parse_args()

    rows = census(json.loads(Path(a.key).read_text()), tuple(a.radius), a.robot)
    if not rows:
        print("no targets resolved to exactly one object", file=sys.stderr)
        return 1
    report(rows, tuple(a.radius))
    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=1))
        print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
