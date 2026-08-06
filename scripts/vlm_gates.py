#!/usr/bin/env python3
"""What the reliability gates buy, and what they cost.

The v3 sweep asks the model for three things it can see but the scanner cannot
report: whether the target is behind glass, whether the beam can pass through
it, and roughly how far away it is. This applies gates built from those answers
to the already-scored rows and reports the trade every gate makes, because a
gate that removes a false positive by also removing two hits is not a gain.

Reads `artifacts/vlm_sweep.json`; makes no API calls, so thresholds can be
swept freely.

    uv run python scripts/vlm_gates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# A folding screen 2 m tall seen at 2.5 m subtends ~45°; the same box at 9.5 m
# would imply a 7 m object. The gate fires on that kind of contradiction, not
# on a 20% disagreement, so the band is deliberately wide.
SIZE_RATIO = 3.0
DIST_RATIO = 2.5


def implied_size(lift_range: float, box_deg: float) -> float:
    return 2.0 * lift_range * np.tan(np.deg2rad(box_deg) / 2.0)


def gates(r: dict) -> str | None:
    """Name of the first gate that rejects this row, or None to keep it."""
    if r.get("same_space") is False:
        return "not same space"
    lift, vd = r.get("lift_range"), r.get("vlm_dist")
    if lift is None:
        return "no lidar return"
    if vd:
        if lift > vd * DIST_RATIO:
            return "lift >> vlm distance"
        if lift < vd / DIST_RATIO:
            return "lift << vlm distance"
    rng = r.get("vlm_range") or []
    if len(rng) == 2 and rng[1]:
        if r.get("occlusion") == "see_through" and lift > rng[1]:
            return "see_through, lift beyond range"
        if r.get("occlusion") == "partly_occluded" and lift < rng[0]:
            return "occluded, lift short of range"
    return None


def report(rows: list[dict], name: str, keep) -> None:
    live = [r for r in rows if keep(r)]
    drop = [r for r in rows if not keep(r)]
    n = len(rows)
    hit = sum(r["hit"] for r in live)
    fp = sum(r["claimed"] and not r["hit"] for r in live)
    dh = sum(r["hit"] for r in drop)
    df = sum(r["claimed"] and not r["hit"] for r in drop)
    print(f"{name:34s} hits {hit:2d}/{n} = {hit / n:5.1%}   "
          f"FP {fp:2d}/{n} = {fp / n:5.1%}   "
          f"(dropped {df} FP, {dh} hits)")


def main() -> int:
    rows = json.loads(Path("artifacts/vlm_sweep.json").read_text())
    claimed = [r for r in rows if r["claimed"]]
    print(f"{len(rows)} pairs, {len(claimed)} claimed visible\n")

    print("== self-reported occlusion ==")
    for v in sorted({str(r.get("occlusion")) for r in claimed}):
        sub = [r for r in claimed if str(r.get("occlusion")) == v]
        h = sum(r["hit"] for r in sub)
        print(f"  {v:18s} n={len(sub):2d}   hit {h}/{len(sub)} = {h / len(sub):4.0%}")

    print("\n== self-reported same_space ==")
    for v in (True, False, None):
        sub = [r for r in claimed if r.get("same_space") is v]
        if not sub:
            continue
        h = sum(r["hit"] for r in sub)
        print(f"  {str(v):18s} n={len(sub):2d}   hit {h}/{len(sub)} = {h / len(sub):4.0%}")

    pairs = [(r["vlm_dist"], r["true_range"]) for r in claimed
             if r.get("vlm_dist") and r.get("true_range")]
    if pairs:
        v, t = np.array(pairs).T
        print(f"\n== VLM distance vs true range ==  n={len(pairs)}")
        print(f"  correlation      {np.corrcoef(v, t)[0, 1]:.3f}")
        print(f"  median |error|   {np.median(np.abs(v - t)):.2f} m")
        print(f"  median ratio     {np.median(v / t):.2f}x  "
              f"(1.0 = unbiased)")
        within = np.mean(np.abs(np.log(v / t)) < np.log(1.5))
        print(f"  within 1.5x      {within:.0%}")

    print("\n== gates ==")
    report(rows, "no gate", lambda r: True)
    report(rows, "same_space only", lambda r: r.get("same_space") is not False)
    report(rows, "distance cross-check only",
           lambda r: not (r.get("lift_range") and r.get("vlm_dist")
                          and (r["lift_range"] > r["vlm_dist"] * DIST_RATIO
                               or r["lift_range"] < r["vlm_dist"] / DIST_RATIO)))
    report(rows, "all gates", lambda r: gates(r) is None)

    print("\n== what each gate actually rejected ==")
    for r in rows:
        g = gates(r)
        if g and r["claimed"]:
            e = "no lift" if r["err"] is None else f"{r['err']:.2f} m"
            tag = "correctly" if not r["hit"] else "WRONGLY  "
            print(f"  {tag} dropped {r['scene']:14s} {r['phrase']:18s} "
                  f"err {e:>8s}  [{g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
