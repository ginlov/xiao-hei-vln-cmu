#!/usr/bin/env python3
"""What the counting call would have said, on views the robot has already taken.

The gate. If the model cannot count reliably from a clean, close, fully-framed
view, nothing downstream is worth building -- no amount of driving, merging or
abstention recovers a call that miscounts what it can plainly see. This answers
that before any of the harder machinery exists, using the 1,980 face images the
instruction sweep left on disk, at zero simulator cost.

Grading needs a truth for *this view*, not for the scene, because a robot
standing in a doorway legitimately sees three of four pillows. So each view
carries its own expectation:

  visible   how many of the key's counted objects project into a face from
            this pose within `MAX_M`. It ignores occlusion, so it is an UPPER
            bound on what could honestly be counted.
  full      whether `visible` equals the scene's whole answer -- the subset
            where the model has no excuse and must return the exact integer.

Three cells decide the gate, and the third is the one that matters:

  exact on full views       can it count what it can plainly see -- the gate
  over-count               a number above `visible` is a real error: a double
                           count, or something that is not the target at all
  confident and wrong      `sufficient: true` with the wrong integer. This is
                           the cell that loses questions, because nothing
                           downstream gets a chance to fix it.

    uv run python scripts/count_audit.py --dry              # what would fire
    uv run --with anthropic python scripts/count_audit.py --scenes hotel_room_2
    uv run --with anthropic python scripts/count_audit.py --full-only --limit 40
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
from count_view import count_view, lift_instances, merge  # noqa: E402
from score_numerical import SPECS, counted_objects, key  # noqa: E402
from verify_binding import where_is  # noqa: E402

MODEL = os.environ.get("XIAO_HEI_MODEL", "claude-opus-5")
# Beyond this a pillow is a smudge, and asking the model to count is asking it
# to guess. The plan's stop rule aims at 2-3 m; this is deliberately looser so
# the audit can report how accuracy falls off with range rather than assuming
# a cut-off.
MAX_M = 5.0
# `where_is` already answers "is it in a face at all" -- it walks the four and
# returns None when the bearing falls in none of them. There is nothing left to
# filter, and an earlier version of this file filtered anyway: it read `across`
# as a signed offset from the face centre and kept `|across| < 0.42`, when
# `across` is `u / FACE_SIZE` in [0, 1) with 0.5 AT the centre. That threw away
# everything right of centre, which under-counted `visible` on more than half
# the views and, worse, made "sees every counted object" mean "every counted
# object is in the left 42% of its face". Two conclusions had to be withdrawn.
MARGIN = 0.02          # ignore a bearing sitting exactly on a face seam


def views_of(run: Path, scene: str) -> list[dict]:
    """Every recorded step in a run that has faces, a scan and a pose."""
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
        step = r.get("step")
        pose = r.get("pose")
        if r.get("step") == "settings" or not isinstance(pose, dict):
            continue
        if not pose.get("orientation"):
            continue
        faces = [run / f"step{step}_face{i}.jpg" for i in range(4)]
        scan = run / f"step{step}_scan.npy"
        if not all(f.is_file() for f in faces) or not scan.is_file():
            continue
        out.append({"run": run.name, "scene": scene, "step": step, "pose": pose,
                    "faces": faces, "scan": scan})
    return out


def visible_from(objs: list[dict], pose: dict) -> list[dict]:
    """Which counted objects project into a face from here, inside `MAX_M`."""
    out = []
    for o in objs:
        at = where_is(np.asarray(o["c"], float)[:2], pose)
        if at and at["range_m"] <= MAX_M and MARGIN <= at["across"] <= 1 - MARGIN:
            out.append({**o, "at": at})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="cv_0818_*", help="glob over run names")
    ap.add_argument("--scenes", default="*",
                    help="comma-separated globs over scene names")
    ap.add_argument("--dry", action="store_true",
                    help="report what would be called on; make no model calls")
    ap.add_argument("--full-only", action="store_true",
                    help="only views that can see every counted object")
    ap.add_argument("--limit", type=int, default=0, help="cap the calls made")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="artifacts/count_audit.json")
    args = ap.parse_args()

    answers = {r["scene"]: r for r in key()}
    # A list rather than one glob: the useful selection is usually "these four
    # scenes", and `fnmatch` has no brace expansion, so `{a,b}` matches nothing
    # and reports zero views without saying why.
    wanted = [w.strip() for w in args.scenes.split(",") if w.strip()]
    scene_of = {s: s for s in SPECS
                if any(fnmatch.fnmatch(s, w) for w in wanted)}
    if not scene_of:
        print(f"no scene matches {args.scenes!r}; have "
              f"{', '.join(sorted(SPECS))}", file=sys.stderr)
        return 1

    # Runs are named `<prefix>_<scene>_<qtag>_p<pass>`; the scene is whichever
    # spec name the directory contains. Matching on the name rather than
    # re-reading every plan.json keeps the dry pass instant.
    picked: list[dict] = []
    for run in sorted((REPO / "runs").iterdir()):
        if not run.is_dir() or not fnmatch.fnmatch(run.name, args.runs):
            continue
        scene = next((s for s in sorted(scene_of, key=len, reverse=True)
                      if s in run.name), None)
        if scene is None:
            continue
        objs = counted_objects(scene)
        if not objs:
            continue
        for v in views_of(run, scene):
            vis = visible_from(objs, v["pose"])
            if not vis:
                continue
            v["visible"] = len(vis)
            v["truth"] = answers[scene]["answer"]
            v["full"] = len(vis) == v["truth"]
            v["nearest_m"] = round(min(x["at"]["range_m"] for x in vis), 2)
            picked.append(v)

    if args.full_only:
        picked = [v for v in picked if v["full"]]
    picked.sort(key=lambda v: (v["scene"], v["nearest_m"]))
    if args.limit:
        # Nearest first within each scene, so a capped run spends its calls on
        # the views the gate is actually about.
        by: dict[str, list] = {}
        for v in picked:
            by.setdefault(v["scene"], []).append(v)
        per = max(1, args.limit // max(1, len(by)))
        picked = [v for vs in by.values() for v in vs[:per]][:args.limit]

    print(f"{len(picked)} views over {len({v['scene'] for v in picked})} scenes"
          f"   ({sum(1 for v in picked if v['full'])} see every counted object)")
    for scene in sorted({v["scene"] for v in picked}):
        vs = [v for v in picked if v["scene"] == scene]
        print(f"   {scene:18s} {len(vs):>3} views, "
              f"{sum(1 for v in vs if v['full']):>3} full, "
              f"truth {answers[scene]['answer']} "
              f"({answers[scene]['reading']}), "
              f"nearest {min(v['nearest_m'] for v in vs):.1f} m")
    if args.dry:
        print("\n--dry: no model calls made.")
        return 0

    spec_of = {s: SPECS[s] for s in SPECS}
    rows = []
    for i, v in enumerate(picked, 1):
        spec = spec_of[v["scene"]]
        faces = [f.read_bytes() for f in v["faces"]]
        r = count_view(faces, spec["q"], spec["target"], spec.get("anchor"),
                       model=args.model)
        lifted = lift_instances(r, np.load(v["scan"]), v["pose"]) \
            if not r.get("error") else []
        row = {"run": v["run"], "scene": v["scene"], "step": v["step"],
               "nearest_m": v["nearest_m"], "visible": v["visible"],
               "truth": v["truth"], "full": v["full"],
               "count": r.get("count_here"), "sufficient": r.get("sufficient"),
               "why_not": r.get("why_not"), "confidence": r.get("confidence"),
               "placed": sum(1 for x in lifted if x["xy"] is not None),
               # Among the instances that HAVE a position. Counting
               # `len(lifted) - len(merge(...))` made an unplaced instance look
               # like a merged one, which reported the cups on livingroom_2 as
               # de-duplicated when in fact the scanner never reached them.
               "self_merged": (lambda pl: len(pl) - len(merge([], pl)))(
                   [x for x in lifted if x["xy"] is not None]),
               "error": r.get("error"), "evidence": r.get("evidence"),
               # Kept so the report can be recomputed without paying for the
               # calls again -- every metric above is derivable from these.
               "lifted": [None if x["xy"] is None
                          else [round(float(x["xy"][0]), 3),
                                round(float(x["xy"][1]), 3)] for x in lifted]}
        rows.append(row)
        mark = "=" if row["count"] == v["truth"] else (
            "+" if (row["count"] or 0) > v["visible"] else "-")
        print(f"  [{i}/{len(picked)}] {mark} {v['scene']:16s} {v['run'][-24:]:24s} "
              f"step {v['step']:>3}  {v['nearest_m']:>4.1f} m  "
              f"count {row['count']}  visible {v['visible']}  truth {v['truth']}"
              f"  {'SUFF' if row['sufficient'] else 'more'}"
              f"{'  ' + row['error'] if row['error'] else ''}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    report(rows)
    print(f"\nwritten: {out}")
    return 0


def report(rows: list[dict]) -> None:
    live = [r for r in rows if not r.get("error")]
    if not live:
        print("\nno usable replies.")
        return
    full = [r for r in live if r["full"]]
    print(f"\n{'':34s}{'n':>5}{'exact':>9}")

    def line(name, rs, pred):
        if not rs:
            print(f"{name:34s}{0:>5}{'—':>9}")
            return
        n = sum(1 for r in rs if pred(r))
        print(f"{name:34s}{len(rs):>5}{n:>4} = {100 * n / len(rs):3.0f}%")

    exact = lambda r: r["count"] == r["truth"]                    # noqa: E731
    # Against what is IN FRAME rather than against the scene's answer. On a
    # partial view the two differ, and only this one is a question about the
    # model: a robot that can see three of six pillows and says three has
    # counted correctly and answered the question wrongly, and those are
    # different failures with different fixes.
    inframe = lambda r: r["count"] == r["visible"]                 # noqa: E731
    line("exact, views seeing everything", full, exact)            # THE GATE
    line("exact, all views", live, exact)
    line("exact, within 3 m", [r for r in live if r["nearest_m"] <= 3.0], exact)
    print()
    line("counted what was in frame, all views", live, inframe)
    line("  ... where 4 or more were in frame",
         [r for r in live if r["visible"] >= 4], inframe)

    over = [r for r in live if (r["count"] or 0) > r["visible"]]
    print(f"\nover-count (above what is in frame)   {len(over)}/{len(live)}"
          f" = {100 * len(over) / len(live):.0f}%")
    dup = sum(r["self_merged"] for r in live)
    print(f"placed instances two lifts put in one spot   {dup}"
          f"   (double counts caught within one view)")
    want = sum(r["count"] or 0 for r in live)
    got = sum(r["placed"] for r in live)
    print(f"instances the scanner could place            {got}/{want}"
          f" = {100 * got / want:.0f}%"
          f"   <- the de-duplication only works on these")
    for r in sorted(live, key=lambda r: r["placed"] - (r["count"] or 0)):
        if r["placed"] < (r["count"] or 0):
            print(f"    {r['scene']:16s} step {r['step']:>3}  "
                  f"placed {r['placed']}/{r['count']} at {r['nearest_m']:.1f} m")

    say_yes = [r for r in live if r["sufficient"]]
    say_no = [r for r in live if not r["sufficient"]]
    print(f"\n{'':34s}{'n':>5}{'right':>9}")
    line("said sufficient", say_yes, exact)
    line("said not sufficient", say_no, exact)
    bad = [r for r in say_yes if not exact(r)]
    print(f"\nCONFIDENT AND WRONG                   {len(bad)}/{len(live)}"
          f" = {100 * len(bad) / len(live):.0f}%   <- the cell that loses questions")
    why: dict[str, int] = {}
    for r in say_no:
        why[str(r["why_not"])] = why.get(str(r["why_not"]), 0) + 1
    if why:
        print("reasons given for needing more:  "
              + ", ".join(f"{k} ({n})" for k, n in sorted(why.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    raise SystemExit(main())
