#!/usr/bin/env python3
"""Path-fidelity metrics from the VLN literature, over trajectories we already drove.

    uv run python scripts/path_fidelity.py --runs 'cv_0*'
    uv run python scripts/path_fidelity.py --runs 'cv_0*,nv_0*,cm_0825_1549*' --json out.json

WHY THIS EXISTS

Our own scorer is the weakest instrument in the paper: the official evaluator is
closed, so both the 1.25 m arrival tolerance and the instance-matching rule are
ours, and both move the result. The obvious reviewer question is whether the
ranking survives a threshold we did not choose.

nDTW does not use either. It compares the shape of two curves -- the trajectory
we drove and the organisers' own reference path for the same question -- and
nothing else. So it is an INDEPENDENT check on the same corpus, computed from
data already on disk, with no simulator and no model call.

METRICS

  TL     trajectory length, metres. Reported because every fidelity measure has
         to be read against how far the vehicle actually went.
  nDTW   normalised Dynamic Time Warping (Ilharco et al. 2019):
             exp( -DTW / (|R| * d_th) )
         where DTW is the optimal alignment cost between our track and the
         reference, |R| the number of reference points and d_th a success
         threshold. 1.0 is a perfect trace of the reference; it decays smoothly.
  CLS    Coverage weighted by Length Score (Jain et al. 2019): how much of the
         reference the track covers, times how close its length is to the
         covered portion. Penalises both skipping and dawdling.
  SDTW   nDTW gated on success. We do not have the official success flag, so
         this is nDTW * (our rubric score / 6) and is labelled as such
         everywhere -- it inherits our scorer and is NOT the independent number.

WHAT THIS DELIBERATELY DOES NOT COMPUTE

  SPL    needs the shortest traversable path, which needs a navmesh we do not
         have. Approximating it with Euclidean distance would flatter every arm
         equally and mean nothing. We say we cannot compute it rather than
         computing something else and calling it SPL.

The reference paths live in the challenge repo, not this one:
`<CHALLENGE>/questions/<scene>/trajectory_q{4,5}.ply`.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from score_if import (CHALLENGE, TRAJ_FOR, driven_track,  # noqa: E402
                      official_questions, read_ply_ascii, run_question)

# The same tolerance the rubric uses, so the two instruments are read on one
# scale. nDTW is not sensitive to it the way the rubric is -- it appears once,
# inside an exponential, rather than as a pass/fail edge.
D_TH = 1.25


def resample(track: np.ndarray, step: float = 0.25) -> np.ndarray:
    """Arc-length resampling, so DTW is not dominated by where points are dense.

    `robot_io` subsamples at 0.10 m while the reference `.ply` is whatever the
    organisers exported. Comparing them raw makes the cost depend on sampling
    rate rather than on shape, which is the one thing DTW is supposed to ignore.
    """
    p = np.asarray(track, float)[:, :2]
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return p[:1]
    n = max(2, int(total / step) + 1)
    want = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(want, s, p[:, 0]),
                            np.interp(want, s, p[:, 1])])


def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Optimal alignment cost. O(len(a) * len(b)); both are resampled first."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        # Vectorising the inner loop is not possible: each cell depends on the
        # one to its left. n and m are ~100 after resampling, so this is fine.
        for j in range(1, m + 1):
            acc[i, j] = d[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1],
                                              acc[i - 1, j - 1])
    return float(acc[n, m])


def ndtw(track: np.ndarray, ref: np.ndarray, d_th: float = D_TH) -> float:
    a, b = resample(track), resample(ref)
    return float(np.exp(-dtw_cost(a, b) / (len(b) * d_th)))


def cls(track: np.ndarray, ref: np.ndarray, d_th: float = D_TH) -> float:
    """Coverage weighted by Length Score (Jain et al. 2019).

    coverage: mean over reference points of exp(-min distance to our track / d_th)
    expected length: coverage * length(reference)
    LS: expected / (expected + |expected - length(ours)|)
    CLS = coverage * LS
    """
    a, b = resample(track), resample(ref)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    dmin = np.min(np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2), axis=1)
    cov = float(np.mean(np.exp(-dmin / d_th)))
    len_ref = float(np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1)))
    len_ours = float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))
    exp_len = cov * len_ref
    ls = exp_len / (exp_len + abs(exp_len - len_ours)) if exp_len > 0 else 0.0
    return float(cov * ls)


def path_len(track: np.ndarray) -> float:
    p = np.asarray(track, float)[:, :2]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def rows_for(glob_: str, qmap: dict) -> list[dict]:
    out = []
    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir() or not fnmatch.fnmatch(run.name, glob_):
            continue
        q = run_question(run)
        if q is None or q not in qmap:
            continue
        scene, qi = qmap[q]
        rp = CHALLENGE / "questions" / scene / TRAJ_FOR[qi]
        track = driven_track(run)
        if track is None or not rp.is_file():
            continue
        ref = read_ply_ascii(rp)
        if ref is None or len(ref) < 2:
            continue
        out.append({"run": run.name, "scene": scene, "q": qi + 4,
                    "TL": path_len(track), "ref_TL": path_len(ref),
                    "nDTW": ndtw(track, ref), "CLS": cls(track, ref)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="cv_0*",
                    help="comma-separated globs over run directory names")
    ap.add_argument("--json", help="write per-run rows here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qmap = official_questions(CHALLENGE)
    allrows = {}
    for g in args.runs.split(","):
        g = g.strip()
        if g:
            allrows[g] = rows_for(g, qmap)

    print(f"{'arm (glob)':<20}{'runs':>6}{'nDTW':>9}{'CLS':>9}"
          f"{'TL':>9}{'ref TL':>9}{'TL ratio':>10}")
    print("-" * 72)
    for g, rows in allrows.items():
        if not rows:
            print(f"{g:<20}{'0':>6}   no scorable runs")
            continue
        nd = [r["nDTW"] for r in rows]
        cl = [r["CLS"] for r in rows]
        tl = [r["TL"] for r in rows]
        rt = [r["TL"] / r["ref_TL"] for r in rows if r["ref_TL"] > 0]
        print(f"{g:<20}{len(rows):>6}{st.mean(nd):>9.3f}{st.mean(cl):>9.3f}"
              f"{st.mean(tl):>9.1f}{st.mean([r['ref_TL'] for r in rows]):>9.1f}"
              f"{st.mean(rt):>10.2f}")

    if args.verbose:
        for g, rows in allrows.items():
            print(f"\n--- {g}")
            for r in sorted(rows, key=lambda r: -r["nDTW"]):
                print(f"  {r['scene']:<18}q{r['q']}  nDTW {r['nDTW']:.3f}  "
                      f"CLS {r['CLS']:.3f}  TL {r['TL']:6.1f} / {r['ref_TL']:.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(allrows, indent=1))
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
