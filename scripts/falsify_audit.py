#!/usr/bin/env python3
"""Noise floor for the cross-step consistency tests.

Reads recorded runs only. No model call, no simulator.

For every pair of consecutive steps *within one leg*, the binding made at the
earlier step is a hypothesis. Two residuals test it:

  T1  range vs odometry     | ||P - b|| - r_B |
  T2  implied size          | log( (r_A*h_A) / (r_B*h_B) ) |

A residual is never zero even when the binding is right, so the question this
answers is how big it gets when nothing is wrong. Pairs are labelled by the
binding's distance to the nearest ground-truth instance of the phrase's head
noun -- taken from the scene annotation, not from our own scorer, so the label
does not depend on the thing being calibrated.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from score_if import load_scene, head_label, official_questions, run_question  # noqa
from vlm_approach import (box_angular_size, cam_dir_to_map,  # noqa
                          ray_from_box, triangulate)

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
GOOD_M, BAD_M = 1.0, 2.5     # the bimodal split found by score_if


def pos(p):
    if isinstance(p, dict):
        return np.asarray(p.get("position", [np.nan] * 3), float)
    return np.asarray(p, float)


def collect():
    qmap = official_questions(CHALLENGE)
    viz = REPO / "viz/data"
    scenes: dict = {}
    rows = []
    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir():
            continue
        q = run_question(run)
        if q is None or q not in qmap:
            continue
        scene, _ = qmap[q]
        if scene not in scenes:
            got = load_scene(scene, viz)
            if got is None:
                continue
            scenes[scene] = got
        by_label, labels = scenes[scene]

        sj = run / "steps.jsonl"
        if not sj.is_file():
            continue
        steps = []
        for line in sj.read_text().splitlines():
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for a, b in zip(steps, steps[1:]):
            if a.get("clause") != b.get("clause"):
                continue                      # a leg boundary changes the target
            ra, rb = a.get("reply") or {}, b.get("reply") or {}
            wa, wb = a.get("waypoint") or {}, b.get("waypoint") or {}
            bind = (a.get("binding") or {}).get("xy")
            pa, pb = pos(a.get("pose")), pos(b.get("pose"))
            if not (ra.get("box_2d") and rb.get("box_2d") and bind is not None):
                continue
            if ra.get("image_index") is None or rb.get("image_index") is None:
                continue
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            r_A, r_B = wa.get("range_m"), wb.get("range_m")
            P = np.asarray(bind, float)[:2]

            # label from the annotation, independent of our scorer
            lab = head_label(a.get("phrase") or "", labels)
            gts = by_label.get(lab or "", [])
            if not gts:
                continue
            err = min(float(np.linalg.norm(P - g["c"][:2])) for g in gts)

            rec = {"run": run.name, "scene": scene, "bind_err_m": err,
                   "moved_m": float(np.linalg.norm(pb[:2] - pa[:2]))}

            # T1 -- range against odometry
            if r_B is not None:
                rec["t1"] = abs(float(np.linalg.norm(P - pb[:2])) - float(r_B))

            # T2 -- implied physical size across the two views
            if r_A and r_B:
                _, hA = box_angular_size(ra["box_2d"], int(ra["image_index"]))
                _, hB = box_angular_size(rb["box_2d"], int(rb["image_index"]))
                sA = 2 * float(r_A) * np.tan(np.deg2rad(hA) / 2)
                sB = 2 * float(r_B) * np.tan(np.deg2rad(hB) / 2)
                if sA > 1e-3 and sB > 1e-3:
                    rec["t2"] = abs(float(np.log(sA / sB)))

            # T3 -- bearings only. No lidar anywhere in this test, so it is
            # the one path independent of the lift that T1 depends on, and it
            # survives on pairs where the lift was refused and there is no range.
            dA = ray_from_box(ra["box_2d"], int(ra["image_index"]))
            dB = ray_from_box(rb["box_2d"], int(rb["image_index"]))
            oa, ob = a.get("pose"), b.get("pose")
            if isinstance(oa, dict) and isinstance(ob, dict) and oa.get("orientation"):
                mA = cam_dir_to_map(dA, oa)
                mB = cam_dir_to_map(dB, ob)
                t = triangulate(pa, mA, pb, mB)
                rec["parallax_deg"] = t["parallax_deg"]
                if t.get("range_m") and t["range_m"] > 0:
                    # T3a: do the two rays actually meet? If the two boxes are
                    # the same physical point they must nearly intersect.
                    rec["t3a"] = float(t["miss_m"]) / max(float(t["range_m"]), 0.5)
                    # T3b: bearing-only range against the lidar lift.
                    if r_A:
                        rec["t3b"] = abs(float(np.log(
                            max(t["range_m"], 1e-3) / max(float(r_A), 1e-3))))
            else:
                rec["parallax_deg"] = float(np.rad2deg(np.arccos(
                    np.clip(abs(float(dA @ dB)), -1, 1))))
            rows.append(rec)
    return rows


def report(rows):
    good = [r for r in rows if r["bind_err_m"] <= GOOD_M]
    bad = [r for r in rows if r["bind_err_m"] >= BAD_M]
    print(f"{len(rows)} in-leg step pairs from "
          f"{len({r['run'] for r in rows})} runs\n"
          f"  binding within {GOOD_M} m of a ground-truth instance : {len(good)}\n"
          f"  binding beyond {BAD_M} m                             : {len(bad)}\n"
          f"  in between (unlabelled, excluded)                    : "
          f"{len(rows) - len(good) - len(bad)}")

    for key, name, unit in (("t1", "T1  range vs odometry", "m"),
                            ("t2", "T2  implied-size |log ratio|", ""),
                            ("t3a", "T3a ray miss / range  (bearings only)", ""),
                            ("t3b", "T3b triangulated vs lifted |log ratio|", "")):
        g = np.array([r[key] for r in good if key in r])
        b = np.array([r[key] for r in bad if key in r])
        print(f"\n{name}   (good n={len(g)}, bad n={len(b)})")
        if len(g) < 5 or len(b) < 5:
            print("   too few to characterise")
            continue
        print(f"   {'':6s} {'good':>9s} {'bad':>9s}")
        for p in (50, 75, 90, 95):
            print(f"   p{p:<5d} {np.percentile(g, p):9.3f} {np.percentile(b, p):9.3f}")
        thr = float(np.percentile(g, 95))
        det = float((b > thr).mean())
        print(f"   threshold at good-p95 = {thr:.3f} {unit}"
              f"  ->  catches {det:.0%} of bad, 5% false alarm")
        sep = np.median(b) / max(np.median(g), 1e-9)
        print(f"   median separation: {sep:.1f}x")

    par = np.array([r["parallax_deg"] for r in rows])
    print(f"\nparallax between the two bearings (for a triangulation test)")
    for p in (25, 50, 75, 90):
        print(f"   p{p:<5d} {np.percentile(par, p):6.1f} deg")
    print(f"   pairs with < 5 deg (triangulation degenerate): {(par < 5).mean():.0%}")


GATES = [
    ("t1", "T1  range vs odometry", lambda r: r["moved_m"] >= 0.5, "moved >= 0.5 m"),
    ("t1", "T1  range vs odometry *", lambda r: r["moved_m"] >= 0.75, "moved >= 0.75 m  <- deployed"),
    ("t1", "T1  range vs odometry", lambda r: r["moved_m"] >= 1.0, "moved >= 1.0 m"),
    ("t2", "T2  implied size", lambda r: r["moved_m"] >= 1.0, "moved >= 1.0 m"),
    ("t3a", "T3a ray miss", lambda r: r.get("parallax_deg", 0) >= 20, "parallax >= 20 deg"),
    ("t3b", "T3b triangulated range", lambda r: r.get("parallax_deg", 0) >= 10, "parallax >= 10 deg"),
    ("t3b", "T3b triangulated range", lambda r: r.get("parallax_deg", 0) >= 20, "parallax >= 20 deg"),
]


def conditional(rows):
    """Each test scored only where its own geometry gives it leverage.

    A test evaluated over pairs it cannot judge is being blamed for cases it
    should have declined. Letting it abstain is not a concession -- for T1 it
    is the difference between a weak gate and a strong one.
    """
    print("\n\nEach test applied ONLY where its geometry is non-degenerate")
    print("-" * 78)
    print(f"{'test':26s} {'valid when':20s} {'abst':>5s} {'n g/b':>8s} "
          f"{'catch':>6s} {'sep':>7s}")
    print("-" * 78)
    N = len(rows)
    for key, name, cond, lab in GATES:
        sub = [r for r in rows if cond(r)]
        g = np.array([r[key] for r in sub if key in r and r["bind_err_m"] <= GOOD_M])
        b = np.array([r[key] for r in sub if key in r and r["bind_err_m"] >= BAD_M])
        ab = 1 - len(sub) / N
        if len(g) < 8 or len(b) < 8:
            print(f"{name:26s} {lab:20s} {ab:5.0%} {'':>8s} too few")
            continue
        thr = float(np.percentile(g, 95))
        print(f"{name:26s} {lab:20s} {ab:5.0%} {len(g):3d}/{len(b):<4d} "
              f"{(b > thr).mean():6.0%} {np.median(b) / max(np.median(g), 1e-9):6.1f}x")
    print("\nThreshold is each test's own good-p95, so every row buys 5% false alarm.")


if __name__ == "__main__":
    rows = collect()
    if not rows:
        print("no usable pairs")
        raise SystemExit(1)
    report(rows)
    conditional(rows)
