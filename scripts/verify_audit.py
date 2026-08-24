#!/usr/bin/env python3
"""What the semantic verifier would have said, on runs already driven.

Replays recorded steps: the faces on disk, the pose on disk, the binding the
leg actually held. No simulator, so it costs one model call per fired step and
nothing else, and it can be re-run after any change to the prompt or the gate.

The label is the same one `falsify_audit` uses -- the binding's distance to the
nearest ground-truth instance of the phrase's head noun, read from the scene
annotation rather than from our own scorer, so what the verifier is graded
against does not depend on the thing being calibrated.

Three numbers come out of it, and only the third decides whether the mechanism
is worth its call:

  fire rate     what fraction of grounding steps the gate fires on -- the cost
  refutation    of the bindings that were WRONG, how many it refuted -- the win
  false alarm   of the bindings that were RIGHT, how many it refuted -- the risk

A verifier that refutes everything scores perfectly on the second and destroys
the system on the third, which is why abstention is reported alongside.

    uv run python scripts/verify_audit.py                 # every run
    uv run python scripts/verify_audit.py --runs 'cv_*'   # one corpus
    uv run python scripts/verify_audit.py --dry           # what would fire, no calls
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from score_if import head_label, load_scene, official_questions, run_question  # noqa: E402
from verify_binding import (CARRIED_STEPS, REFUTE_CONF,  # noqa: E402
                            should_verify, verify_binding, where_is)

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
GOOD_M, BAD_M = 1.0, 2.5          # the bimodal split found by score_if
MODEL = os.environ.get("XIAO_HEI_MODEL", "claude-opus-5")


def steps_of(run: Path) -> list[dict]:
    sj = run / "steps.jsonl"
    if not sj.is_file():
        return []
    out = []
    for line in sj.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("step") != "settings":
            out.append(r)
    return out


def fired(steps: list[dict]) -> list[tuple[dict, str, int]]:
    """The steps the gate fires on, with the trigger and the carried run.

    Reconstructed exactly as `run_goto` maintains it: the run of carried steps
    resets on any binding the lift agreed with, and never crosses a leg.
    """
    out, carried, clause = [], 0, None
    for r in steps:
        if r.get("clause") != clause:
            clause, carried = r.get("clause"), 0
        b = r.get("binding")
        if isinstance(b, dict) and b.get("carried"):
            carried += 1
        elif isinstance(b, dict):
            carried = 0
        if not isinstance(b, dict) or b.get("xy") is None:
            continue
        bound = {"xy": np.asarray(b["xy"], float), "conf": b.get("conf", 0.0),
                 "verified": b.get("verified", True)}
        why = should_verify(carried_run=carried,
                            rejected="binding_rejected" in r,
                            arriving=bool(r.get("arrived")), bound=bound)
        if why is not None:
            out.append((r, why, carried))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="*", help="glob over run directory names")
    ap.add_argument("--dry", action="store_true",
                    help="report what would fire; make no model calls")
    ap.add_argument("--out", default="artifacts/verify_audit.json")
    args = ap.parse_args()

    qmap = official_questions(CHALLENGE)
    viz, scenes, rows = REPO / "viz/data", {}, []
    n_steps = 0

    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir() or not fnmatch.fnmatch(run.name, args.runs):
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

        steps = steps_of(run)
        n_steps += sum(1 for r in steps if "binding" in r or "reply" in r)
        for r, why, carried in fired(steps):
            phrase = r.get("phrase") or ""
            pose = r.get("pose")
            if not isinstance(pose, dict) or not pose.get("orientation"):
                continue
            P = np.asarray(r["binding"]["xy"], float)[:2]

            lab = head_label(phrase, labels)
            gts = by_label.get(lab or "", [])
            if not gts:
                continue                       # unlabelled: no ground to grade on
            err = min(float(np.linalg.norm(P - g["c"][:2])) for g in gts)

            row = {"run": run.name, "scene": scene, "step": r.get("step"),
                   "phrase": phrase, "why": why, "carried": carried,
                   "bind_err_m": err}
            at = where_is(P, pose)
            row["visible_face"] = None if at is None else at["face"]
            row["range_m"] = None if at is None else at["range_m"]

            if not args.dry:
                faces = [(run / f"step{r['step']}_face{i}.jpg") for i in range(4)]
                if not all(f.is_file() for f in faces):
                    row["skipped"] = "faces not on disk"
                else:
                    v = verify_binding([f.read_bytes() for f in faces], phrase,
                                       P, pose, why=why, model=MODEL,
                                       carried_steps=carried)
                    row.update({k: v.get(k) for k in
                                ("verdict", "confidence", "what_is_there",
                                 "acted", "skipped", "called")})
            rows.append(row)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    report(rows, n_steps, dry=args.dry)
    print(f"\nwritten: {out}")
    return 0


def report(rows: list[dict], n_steps: int, *, dry: bool) -> None:
    print(f"\ngrounding steps in corpus        {n_steps}")
    if n_steps:
        print(f"gate fired on                   {len(rows)} "
              f"({100 * len(rows) / n_steps:.1f}%)   "
              f"[CARRIED_STEPS={CARRIED_STEPS}]")
    by_why: dict[str, int] = {}
    for r in rows:
        by_why[r["why"]] = by_why.get(r["why"], 0) + 1
    for k, n in sorted(by_why.items(), key=lambda kv: -kv[1]):
        print(f"    {k:10s} {n}")

    good = [r for r in rows if r["bind_err_m"] <= GOOD_M]
    bad = [r for r in rows if r["bind_err_m"] >= BAD_M]
    mid = len(rows) - len(good) - len(bad)
    print(f"\nof those, the binding was       "
          f"right {len(good)}   wrong {len(bad)}   in between {mid}"
          f"   (<={GOOD_M} m / >={BAD_M} m from the annotation)")
    if dry:
        print("\n--dry: no model calls made, so no verdicts to report.")
        return

    def cut(rs, pred):
        n = sum(1 for r in rs if pred(r))
        return f"{n}/{len(rs)}" + (f" = {100 * n / len(rs):.0f}%" if rs else "")

    acted = lambda r: bool(r.get("acted"))                       # noqa: E731
    said = lambda r, v: r.get("verdict") == v                    # noqa: E731
    print(f"\n{'':24s}{'wrong bindings':>18s}{'right bindings':>18s}")
    print(f"{'refuted and acted':24s}{cut(bad, acted):>18s}{cut(good, acted):>18s}"
          f"     <- win / risk   (conf >= {REFUTE_CONF})")
    for v in ("refuted", "holds", "cannot_tell"):
        print(f"{'said ' + v:24s}"
              f"{cut(bad, lambda r: said(r, v)):>18s}"
              f"{cut(good, lambda r: said(r, v)):>18s}")
    skipped = [r for r in rows if r.get("skipped")]
    if skipped:
        why: dict[str, int] = {}
        for r in skipped:
            why[str(r["skipped"])] = why.get(str(r["skipped"]), 0) + 1
        print("\nskipped: " + ", ".join(f"{k} ({n})" for k, n in why.items()))


if __name__ == "__main__":
    raise SystemExit(main())
