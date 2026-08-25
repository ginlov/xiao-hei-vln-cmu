#!/usr/bin/env python3
"""§6.2's table, generated from the runs rather than transcribed from them.

Three of four questions were wrong the last time anything in this project was
typed out of a log by hand, so nothing that reaches the paper is typed twice.

    uv run python scripts/paper_table.py --runs 'cv_*'
    uv run python scripts/paper_table.py --runs 'cv_*' --md > /tmp/t.md

Reports, per question: the proxy score, the destinations reached, the model
calls, the wall clock against the budget, and the cost. Then the aggregates §6.2
asks for -- the mean, the per-question spread, and the share of the budget used
-- plus the miss decomposition §6.3 rests on.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from approach_loop import COST_PER_CALL  # noqa: E402
from score_if import (TAU, driven_track, load_scene,  # noqa: E402
                      official_questions, reference_pins, run_question,
                      score_trajectory)

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
NEAR_M = 2.5           # the split §6.3 reports; see the caveat it carries


def facts(run: Path) -> dict:
    """Calls, wall clock and budget, read off the log rather than the console."""
    out = {"calls": 0, "wall_s": None, "drive_steps": 0}
    sj = run / "steps.jsonl"
    if not sj.is_file():
        return out
    first = last = None
    for line in sj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step") == "settings":
            first = r.get("started")
            continue
        out["drive_steps"] += 1
        if "reply" in r:
            out["calls"] += 1
        if r.get("t") or r.get("time"):
            last = r.get("t") or r.get("time")
    # `steps.jsonl` has no end stamp, so fall back to the file's own mtime --
    # it is written at close, which is the end of the run.
    if first is not None:
        end = last if last else sj.stat().st_mtime
        out["wall_s"] = max(0.0, float(end) - float(first))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="cv_*", help="glob over run directory names")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    ap.add_argument("--strict-instance", action="store_true",
                    help="a GOTO must reach the instance the organisers' own "
                         "reference path went to, not merely something with "
                         "the same name")
    args = ap.parse_args()

    qmap = official_questions(CHALLENGE)
    viz, scenes_gt, rows = REPO / "viz/data", {}, []
    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir() or not fnmatch.fnmatch(run.name, args.runs):
            continue
        if not (run / "plan.json").is_file():
            continue
        q = run_question(run)
        if q is None or q not in qmap:
            continue
        scene, qn = qmap[q]
        if scene not in scenes_gt:
            got = load_scene(scene, viz)
            if got is None:
                continue
            scenes_gt[scene] = got
        by_label, labels = scenes_gt[scene]
        track = driven_track(run)
        if track is None:
            continue
        pins = (reference_pins(q, scene, qn, by_label, labels)
                if args.strict_instance else None)
        res = score_trajectory(track, q, by_label, labels, TAU, pin=pins)
        cons = res.get("constraints", [])
        rows.append({"run": run.name, "scene": scene,
                     # `official_questions` indexes from 0; the challenge calls
                     # them 4 and 5, and the paper must use the challenge's
                     # numbers or a reader cannot look the question up.
                     "q": qn + 4,
                     "score": res["score"], "constraints": len(cons),
                     "done": sum(1 for c in cons if c.get("achieved")),
                     "ooo": sum(1 for c in cons if c.get("achieved")
                                and not c.get("in_order", True)),
                     "ko": sum(1 for k in res.get("keepouts", [])
                               if k.get("violated")),
                     "why": [c.get("why", "") for c in cons],
                     **facts(run)})

    if not rows:
        print(f"no scoreable runs match {args.runs!r}")
        return 1

    scenes = sorted({r["scene"] for r in rows})
    scores = [r["score"] for r in rows]
    mean = sum(scores) / len(scores)

    if args.md:
        print("| scene | q | /6 | reached | calls | wall (s) | $ |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            w = "—" if r["wall_s"] is None else f"{r['wall_s']:.0f}"
            print(f"| {r['scene']} | {r['q']} | {r['score']:.2f} | "
                  f"{r['done']}/{r['constraints']} | {r['calls']} | {w} | "
                  f"{r['calls'] * COST_PER_CALL:.2f} |")
    else:
        print(f"{'scene':18s} {'q':>2s} {'/6':>5s} {'reached':>8s} "
              f"{'calls':>6s} {'wall':>6s} {'$':>5s}  run")
        for r in rows:
            w = "—" if r["wall_s"] is None else f"{r['wall_s']:.0f}"
            print(f"{r['scene']:18s} {r['q']:>2} {r['score']:5.2f} "
                  f"{r['done']:>4}/{r['constraints']:<3} {r['calls']:6d} {w:>6s} "
                  f"{r['calls'] * COST_PER_CALL:5.2f}  {r['run']}")

    print(f"\n{len(rows)} questions on {len(scenes)} scenes"
          f"   mean {mean:.2f}/6   total {sum(scores):.2f}/{6 * len(rows)}")
    full = sum(1 for s in scores if s >= 5.999)
    zero = sum(1 for s in scores if s <= 0.001)
    print(f"full marks {full}/{len(rows)}   nothing scored {zero}/{len(rows)}")

    # Per-question spread, which §6.2 calls the first result rather than a
    # caveat. Only questions driven more than once can show it.
    by_q: dict[tuple, list] = {}
    for r in rows:
        by_q.setdefault((r["scene"], r["q"]), []).append(r["score"])
    rep = {k: v for k, v in by_q.items() if len(v) > 1}
    if rep:
        spreads = [max(v) - min(v) for v in rep.values()]
        print(f"\nrepeated questions: {len(rep)}   "
              f"widest spread {max(spreads):.2f}/6   "
              f"mean spread {sum(spreads) / len(spreads):.2f}")
        for (sc, qn), v in sorted(rep.items()):
            if max(v) - min(v) > 0:
                print(f"   {sc} q{qn}: " + ", ".join(f"{x:.2f}" for x in sorted(v)))
    else:
        print("\nrepeated questions: none — one pass, so the spread §6.2 needs "
              "is not in this corpus and must come from a second pass.")

    # §6.3's decomposition.
    misses = []
    for r in rows:
        for w in r["why"]:
            m = re.search(r"closest approach ([\d.]+) m", w or "")
            if m:
                misses.append(float(m.group(1)))
    if misses:
        misses.sort()
        near = [d for d in misses if d < NEAR_M]
        far = [d for d in misses if d >= NEAR_M]
        print(f"\nmissed destinations: {len(misses)}   "
              f"near (<{NEAR_M} m) {len(near)}   far {len(far)}")
        print("   " + "  ".join(f"{d:.1f}" for d in misses))
        if near:
            print(f"   near band {min(near):.2f}–{max(near):.2f} m")
        if far:
            print(f"   far  band {min(far):.2f}–{max(far):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
