#!/usr/bin/env python3
"""What removing the visited block did, measured on more than the score.

    uv run python scripts/memory_ablation.py

The score is one number per question, so on 16 questions it resolves roughly
13-point effects and nothing smaller. The prior per-call work (TASK 45) puts
this block's effect well below that: it moves the model's answer past noise
(22/30 steps, p = 0.016) but moves it 0.06 m in the direction it asks for, and
two samples of the same prompt differ by 0.06 m too. So a null on the score was
the expected outcome and would say very little.

The quantities below are recorded once per STEP rather than once per question,
which is an order of magnitude more samples, and they measure the thing the
block actually claims to do: keep the robot from searching the same place twice.

  revisit rate   share of steps whose pose is within REVISIT_M of a pose the
                 same run already stood at. This is the block's stated goal --
                 "Prefer somewhere it has not stood" -- scored directly.
  spread         mean pairwise distance between the poses a run stood at. A
                 run that circles one spot has a small spread; one that sweeps
                 the building has a large one. Read WITH the revisit rate: a
                 low spread is only a fault if the target was elsewhere.
  steps          grounding calls per question. Losing the record should cost
                 steps before it costs marks.
  visible rate   share of steps where the model reported the target visible.
                 Included because the same gradient appeared across the
                 geometry arms and it is cheap to check here.
"""
from __future__ import annotations

import fnmatch
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from approach_loop import REVISIT_M  # noqa: E402
from score_if import run_question  # noqa: E402

ARMS = {"Ours": "cv_0*", "w/o Memory": "vo_0825_1958*"}

# The memory arm only ever ran the 8 kept scenes, so the control has to be cut
# to the same questions. Comparing 26 questions of one arm against 16 of another
# measures the question set, not the ablation -- and it does so invisibly,
# because both numbers look like perfectly ordinary means.
# Both arms now have the full 26; the subset restriction this once carried is gone.
DROP_SCENES: set[str] = set()


def kept_questions() -> set[str]:
    """The A/B list minus the dropped scenes.

    Deriving this from `official_questions` instead would silently include
    `hotel_room_1` and `livingroom_4`, which the challenge ships but the A/B
    sweeps never drove -- harmless in the mean, since no run matches them, but
    it makes the printed sample size a lie.
    """
    import subprocess
    out = subprocess.run(
        ["bash", "-c", f'source {REPO}/scripts/ab_questions.sh; '
         'for i in "${!CV_Q[@]}"; do echo "${CV_S[$i]}|${CV_Q[$i]}"; done'],
        capture_output=True, text=True).stdout.strip().splitlines()
    return {l.split("|", 1)[1] for l in out if l.split("|", 1)[0] not in DROP_SCENES}


def per_run(run: Path) -> dict | None:
    f = run / "steps.jsonl"
    if not f.is_file():
        return None
    poses, vis, n = [], 0, 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step") == "settings":
            continue
        n += 1
        p = (r.get("pose") or {}).get("position")
        if p:
            poses.append(np.asarray(p[:2], float))
        if (r.get("reply") or {}).get("visible"):
            vis += 1
    if n == 0:
        return None
    # A revisit is judged against everywhere the run stood BEFORE this step,
    # excluding the step immediately prior: a leg that legitimately made a short
    # move would otherwise read as a return, which is the same exclusion
    # `returned_to` makes in the drive loop.
    revisits = sum(1 for i in range(2, len(poses))
                   if any(float(np.linalg.norm(poses[i] - q)) < REVISIT_M
                          for q in poses[:i - 1]))
    judged = max(0, len(poses) - 2)
    spread = (st.mean(float(np.linalg.norm(a - b))
                      for a, b in combinations(poses, 2)) if len(poses) > 1 else 0.0)
    return {"steps": n, "visible": vis / n, "spread": spread,
            "revisit": revisits / judged if judged else None}


def main() -> int:
    keep = kept_questions()
    print(f"revisit threshold {REVISIT_M} m, the same one the drive loop uses")
    print(f"restricted to the {len(keep)} questions both arms have\n")
    print(f"{'arm':<14}{'runs':>6}{'steps/q':>9}{'revisit':>10}{'spread':>9}{'visible':>9}")
    print("-" * 57)
    for name, g in ARMS.items():
        rows = [r for run in sorted((REPO / "runs").iterdir())
                if run.is_dir() and fnmatch.fnmatch(run.name, g)
                and run_question(run) in keep
                for r in [per_run(run)] if r]
        if not rows:
            print(f"{name:<14}{'0':>6}   no runs")
            continue
        rv = [r["revisit"] for r in rows if r["revisit"] is not None]
        print(f"{name:<14}{len(rows):>6}{st.mean(r['steps'] for r in rows):>9.1f}"
              f"{st.mean(rv):>9.0%}{st.mean(r['spread'] for r in rows):>9.2f}"
              f"{st.mean(r['visible'] for r in rows):>9.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
