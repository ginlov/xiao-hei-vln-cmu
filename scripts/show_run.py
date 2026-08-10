#!/usr/bin/env python3
"""Read a run directory back: what the robot saw, chose, and did, per step.

`steps.jsonl` holds one JSON object per grounding call, and the interesting
fields are nested three deep inside a reply that also carries several hundred
words the executor never reads. This prints only the parts that decide where
the robot went, so a leg that failed can be read in one screen instead of
scrolled through.

    uv run python scripts/show_run.py runs/hm1_q2_reserve
    uv run python scripts/show_run.py runs/hm1_q2_reserve --leg 1
    uv run python scripts/show_run.py runs/hm1_q2_reserve --full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def xy(pose) -> tuple[float, float] | None:
    """Planar position out of either pose shape the log carries."""
    if isinstance(pose, dict):
        pose = pose.get("position")
    if isinstance(pose, list) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--leg", type=int, default=None,
                    help="only this clause number")
    ap.add_argument("--full", action="store_true",
                    help="do not truncate the model's reasoning")
    a = ap.parse_args()

    plan = a.run / "plan.json"
    if plan.exists():
        p = json.loads(plan.read_text())
        print(f"Q: {p['question']}\n")
        for i, (c, r) in enumerate(zip(p["plan"], p["results"]), 1):
            mark = "OK  " if r["ok"] else "FAIL"
            where = ("" if not r.get("xy")
                     else f"  at ({r['xy'][0]:+.2f}, {r['xy'][1]:+.2f})")
            print(f"  {mark} {i}. {c}\n         {r['why']}{where}")
        print()

    steps = a.run / "steps.jsonl"
    if not steps.exists():
        print(f"no steps.jsonl in {a.run}")
        return 1

    cut = 10_000 if a.full else 130
    for line in steps.read_text().splitlines():
        r = json.loads(line)
        k = r.get("clause")
        if a.leg is not None and k != a.leg:
            continue
        rep = r.get("reply") or {}
        act = r.get("action") or {}
        o = xy(r.get("pose"))
        pos = f"({o[0]:+.2f},{o[1]:+.2f})" if o else "?"
        flags = "".join([
            f"  LOOP#{r['looped']}" if r.get("looped") else "",
            "  ADRIFT" if r.get("adrift") else "",
            f"  STOP:{r['stopped']}" if r.get("stopped") else "",
            f"  ARRIVED:{r['arrived']}" if r.get("arrived") else "",
        ])
        print(f"[{r.get('step')}] c{k} {r.get('kind') or ''} {pos}  "
              f"vis={rep.get('visible')} conf={rep.get('confidence')}{flags}")
        if rep.get("here"):
            print(f"      here:  {rep['here'][:cut]}")
        if act:
            g = act.get("goal") or []
            print(f"      go:    heading {act.get('heading_deg')}°  "
                  f"reach {act.get('reach_m')}  off {act.get('delta_deg')}°  "
                  f"-> ({g[0]:+.2f}, {g[1]:+.2f})" if g else "      go:")
        why = (rep.get("explore") or {}).get("why")
        if why:
            print(f"      why:   {why[:cut]}")
        if r.get("relation"):
            print(f"      rel:   {str(r['relation'])[:cut]}")
        if r.get("binding"):
            print(f"      bind:  {r['binding']}")
        d = r.get("drive") or {}
        if d:
            print(f"      drove: {d.get('status')}  "
                  f"moved {d.get('moved_m', 0):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
