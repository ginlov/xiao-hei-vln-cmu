#!/usr/bin/env python3
"""Geometry for the trajectory atlas: one plan view per question.

Pulls together the four things a reader needs to judge a run without trusting
our scorer: the sentence, the organisers' own reference trajectory, the two
trajectories we drove, and the ground-truth boxes of the objects the sentence
names. Everything is in the map frame, which is where the reference paths, the
annotations and `/state_estimation` all already live.

    uv run python scripts/export_traj_figures.py --out artifacts/atlas.json

The output is one JSON the artifact embeds directly. Paths are decimated to a
fixed spacing rather than to a fixed count, so a long corridor keeps its shape
and a short leg does not get inflated into one.

Every run carries **two** verdicts, because the scorer has two readings and the
gap between them is a finding rather than a caveat. The loose reading credits a
GOTO for reaching any instance of the phrase's head noun; the strict reading
credits it only for the instance the organisers' own reference path went to
(`score_if.reference_pins`). Boxes are marked accordingly, so a reader can see
at a glance which "potted plant" the sentence meant and which one we drove to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from score_if import (CHALLENGE, TAU, TRAJ_FOR, driven_track,  # noqa: E402
                      load_scene, official_questions, reference_pins,
                      run_question, score_trajectory)
from traj_tolerance import read_ply_ascii  # noqa: E402

STEP_M = 0.12          # decimation spacing along a path
PAD_M = 1.5            # margin around the drawn extent


def thin(track: np.ndarray, step: float = STEP_M) -> list[list[float]]:
    """Decimate by arc length, always keeping the first and last point."""
    if track is None or len(track) == 0:
        return []
    out = [track[0][:2]]
    for p in track[1:]:
        if float(np.linalg.norm(p[:2] - out[-1])) >= step:
            out.append(p[:2])
    if len(track) > 1 and not np.allclose(out[-1], track[-1][:2]):
        out.append(track[-1][:2])
    return [[round(float(x), 3), round(float(y), 3)] for x, y in out]


def box_xy(o: dict) -> list[float]:
    return [round(float(o["lo"][0]), 3), round(float(o["lo"][1]), 3),
            round(float(o["hi"][0]), 3), round(float(o["hi"][1]), 3)]


def key_of(o: dict) -> tuple:
    """Identity for a ground-truth box, so a pinned instance is recognisable."""
    return tuple(box_xy(o))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="cv_0818_", help="run-name prefix to include")
    ap.add_argument("--out", default="artifacts/atlas.json")
    args = ap.parse_args()

    qmap = official_questions(CHALLENGE)
    viz = REPO / "viz/data"
    scenes: dict = {}
    raw: dict = {}

    # our driven trajectories, grouped by question
    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir() or not run.name.startswith(args.runs):
            continue
        if not (run / "plan.json").is_file():
            continue
        q = run_question(run)
        if q is None or q not in qmap:
            continue
        t = driven_track(run)
        if t is None:
            continue
        raw.setdefault(q, []).append((run.name, t))

    out = []
    for q, (scene, qi) in sorted(qmap.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        if q not in raw:
            continue
        if scene not in scenes:
            got = load_scene(scene, viz)
            if got is None:
                continue
            scenes[scene] = got
        by_label, labels = scenes[scene]

        # the reference answer, as the organisers drove it
        rp = CHALLENGE / "questions" / scene / TRAJ_FOR[qi]
        ref = read_ply_ascii(rp) if rp.is_file() else None

        # which instance the reference path itself picked, per ambiguous GOTO
        pins = reference_pins(q, scene, qi, by_label, labels)
        chosen = {key_of(b) for boxes in pins.values() for b in boxes}

        runs = []
        # A destination is scored by the tolerance ball; a passage anchor is
        # not -- PASS is decided by crossing the segment between two anchors.
        # Keeping them apart is what stops the drawing from putting a 1.25 m
        # ring around an object no ring applies to.
        dests: set[str] = set()
        anchors: set[str] = set()
        for name, t in raw[q]:
            loose = score_trajectory(t, q, by_label, labels, TAU)
            strict = score_trajectory(t, q, by_label, labels, TAU, pin=pins)
            tight = {i: c for i, c in enumerate(strict.get("constraints", []))}
            cons = []
            for i, c in enumerate(loose.get("constraints", [])):
                sc = tight.get(i, {})
                cons.append({"kind": c["kind"], "text": c["text"],
                             "ok": bool(c.get("achieved")),
                             "ok_strict": bool(sc.get("achieved")),
                             "why": c.get("why", ""),
                             "why_strict": sc.get("why", ""),
                             "pinned": i in pins,
                             "in_order": bool(c.get("in_order", True))})
                if c.get("label"):
                    dests.add(c["label"])
                for a in (c.get("anchors") or []):
                    if a:
                        anchors.add(a)
            runs.append({"run": name, "score": round(loose["score"], 2),
                         "score_strict": round(strict["score"], 2),
                         "path": thin(t), "constraints": cons,
                         "keepouts": [{"text": k.get("text", ""),
                                       "violated": bool(k.get("violated"))}
                                      for k in loose.get("keepouts", [])]})

        # boxes: the objects the sentence names, and the rest for context.
        # A named object in an ambiguous scene is either the instance the
        # reference went to (`role: "pinned"`) or one of its rivals
        # (`role: "rival"`); unambiguous ones carry no role at all.
        targets, context = [], []
        for lab, boxes in by_label.items():
            for o in boxes:
                rec = {"label": lab, "b": box_xy(o)}
                if lab in dests:
                    rec["use"] = "goto"
                elif lab in anchors:
                    rec["use"] = "anchor"
                else:
                    context.append(rec)
                    continue
                if chosen:
                    same = [k for k in chosen
                            if k in {key_of(x) for x in by_label.get(lab, [])}]
                    if same:
                        rec["role"] = ("pinned" if key_of(o) in chosen
                                       else "rival")
                targets.append(rec)

        # frame the drawing on what matters: the paths and the named objects
        pts = []
        for r in runs:
            pts += r["path"]
        if ref is not None:
            pts += thin(ref)
        for t in targets:
            pts += [[t["b"][0], t["b"][1]], [t["b"][2], t["b"][3]]]
        if not pts:
            continue
        a = np.asarray(pts, float)
        lo = (a.min(axis=0) - PAD_M).round(2).tolist()
        hi = (a.max(axis=0) + PAD_M).round(2).tolist()

        # context furniture only inside the frame, so a scene's far wing does
        # not shrink the room the question is actually about
        context = [c for c in context
                   if c["b"][2] >= lo[0] and c["b"][0] <= hi[0]
                   and c["b"][3] >= lo[1] and c["b"][1] <= hi[1]]

        out.append({"scene": scene, "q": qi + 4, "question": q,
                    "lo": lo, "hi": hi, "tau": TAU,
                    "reference": thin(ref) if ref is not None else [],
                    "runs": runs, "targets": targets, "context": context})

    p = REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")))
    n_runs = sum(len(o["runs"]) for o in out)
    n_amb = sum(1 for o in out
                for t in o["targets"] if t.get("role") == "rival")
    n_anchor = sum(1 for o in out
                   for t in o["targets"] if t.get("use") == "anchor")
    split = sum(1 for o in out for r in o["runs"]
                if r["score"] != r["score_strict"])
    print(f"{len(out)} questions, {n_runs} driven trajectories, "
          f"{sum(1 for o in out if o['reference'])} with a reference path")
    print(f"{n_amb} rival same-name instances marked, "
          f"{n_anchor} passage anchors held back from the tolerance layer; "
          f"{split}/{n_runs} runs score differently under the strict reading")
    print(f"{p}  ({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
