#!/usr/bin/env python3
"""Part A + B over the 60 pairs, scored against the retrieval baseline.

`scripts/retrieval_probe.py` measures what the fused object map can offer a
language query: 48.3% of named objects localised from the start pose, 67% at
k=3 after the whole tour. This asks the same 60 (question, named object) pairs
of a VLM looking at the same frame, so the two numbers mean the same thing and
can be put side by side.

Scored exactly as retrieval scores: a hit is a lifted position within
`HIT_M` = 1.0 m of any ground-truth instance of that phrase's synonym family.
Everything else is a miss, and a claimed sighting that lands further away is
counted separately as a false positive — the number the acceptance criteria
care about most, since a confident wrong answer corrupts the ordering of every
constraint after it.

Replies are cached per (scene, phrase) so a re-run costs nothing and a crash
mid-sweep loses only the call in flight.

    export ANTHROPIC_API_KEY=...
    uv run --with anthropic python scripts/vlm_sweep.py
    uv run --with anthropic python scripts/vlm_sweep.py --scene chinese_room -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from fusion_sweep import family_key  # noqa: E402
from grab_faces import build_luts  # noqa: E402
from traj_tolerance import mentioned  # noqa: E402
from vlm_locate import locate, scan_to_camera  # noqa: E402
from vlm_probe import ask_claude, ask_gemini, build_prompt, parse, to_pixels  # noqa: E402

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ = Path("viz/data")
FRAMES = Path("frames_first")
CACHE = Path("artifacts/vlm_sweep_cache.json")
HIT_M = 1.0          # same as retrieval_probe, so the numbers compare
# Bump whenever the prompt changes, so cached replies are not reused
# across a change in what was actually asked.
PROMPT_VER = "v3-occlusion-distance"


def scene_frame(scene: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """First frame of the corpus: equirect, scan, pose."""
    d = FRAMES / f"{scene}_tour"
    rec = json.loads((d / "frames.jsonl").read_text().splitlines()[0])
    eq = cv2.imread(str(d / rec["image"]), cv2.IMREAD_COLOR)
    scan = np.load(d / rec["scan"])
    pos = rec.get("image_pose_position") or rec["position"]
    quat = rec.get("image_pose_orientation") or rec["orientation"]
    pose = {"position": list(pos), "orientation": list(quat)}
    return eq, scan, pose


def faces_of(eq: np.ndarray) -> list[bytes]:
    out = []
    for map_x, map_y in build_luts(G.FACE_SIZE):
        f = cv2.remap(eq, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_WRAP)
        out.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="all")
    ap.add_argument("--backend", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    model = args.model or ("claude-opus-5" if args.backend == "claude"
                           else "gemini-2.5-flash")
    ask = ask_claude if args.backend == "claude" else ask_gemini
    cache = json.loads(CACHE.read_text()) if CACHE.is_file() and not args.no_cache else {}

    qs = {e["scene"]: e["questions"] for e in
          json.loads((CHALLENGE / "questions/questions.json").read_text())}
    scenes = sorted(qs) if args.scene == "all" else [args.scene]
    rows: list[dict] = []
    t0 = time.time()

    for scene in scenes:
        vf = VIZ / f"{scene}.json"
        if not vf.is_file() or not (FRAMES / f"{scene}_tour").is_dir():
            continue
        d = json.loads(vf.read_text())
        labels = [x.strip().lower() for x in d["labels"]]
        vocab = set(labels)
        gt_by: dict[str, list[np.ndarray]] = {}
        for o in d["gt"]:
            gt_by.setdefault(family_key(labels[o["l"]]), []).append(np.array(o["c"]))

        eq, scan, pose = scene_frame(scene)
        images = faces_of(eq)
        scan_cam = scan_to_camera(scan, pose)
        print(f"\n=== {scene} === pose {np.round(pose['position'][:2], 2)} "
              f"scan {scan.shape[0]}", flush=True)

        seen: set[str] = set()
        for qi, question in enumerate(qs[scene]["instruction_following"]):
            for phrase in mentioned(question, vocab):
                key = family_key(phrase)
                if key not in gt_by or (scene, phrase) in seen:
                    continue
                seen.add((scene, phrase))
                ck = f"{PROMPT_VER}|{args.backend}|{model}|{scene}|{phrase}"
                if ck not in cache:
                    cache[ck] = ask(build_prompt(phrase), images, model)
                    CACHE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE.write_text(json.dumps(cache))
                r = parse(cache[ck]) or {}

                truth = gt_by[key]
                row = {"scene": scene, "q": qi + 4, "phrase": phrase,
                       "claimed": bool(r.get("visible")),
                       "conf": r.get("confidence"), "err": None, "hit": False,
                       "n_gt": len(truth),
                       "occlusion": r.get("occlusion"),
                       "same_space": r.get("same_space"),
                       "vlm_dist": r.get("distance_m"),
                       "vlm_range": r.get("distance_range_m"),
                       "lift_range": None, "true_range": None,
                       "box_deg": None}
                origin = np.asarray(pose["position"])[:2]
                row["true_range"] = min(float(np.linalg.norm(g[:2] - origin))
                                        for g in truth)
                if row["claimed"] and r.get("image_index") is not None:
                    box = r.get("feature_box_2d") or r.get("box_2d")
                    # Prefer the distinguishing feature: it is compact, so its
                    # cone catches less background than the whole object's.
                    if box:
                        res = locate(to_pixels(box, r.get("coord_space"), G.FACE_SIZE),
                                     int(r["image_index"]), scan_cam,
                                     cone_deg=2.0, pose=pose)
                        if res.get("n"):
                            px = to_pixels(box, r.get("coord_space"), G.FACE_SIZE)
                            row["lift_range"] = res["range"]
                            # Angular height of the box, for the size-prior gate.
                            row["box_deg"] = float(
                                (px[2] - px[0]) / G.FACE_SIZE
                                * np.rad2deg(G.FACE_FOV))
                            xy = res["patch_xy"]
                            row["err"] = min(float(np.linalg.norm(xy - g[:2]))
                                             for g in truth)
                            row["hit"] = row["err"] <= HIT_M
                rows.append(row)
                mark = "HIT " if row["hit"] else ("FP  " if row["claimed"] else "miss")
                err = f"{row['err']:.2f} m" if row["err"] is not None else "  —  "
                print(f"  {mark} {phrase:22s} {err}  conf={row['conf']}", flush=True)
                if args.verbose and r.get("evidence"):
                    print(f"        {r['evidence'][:150]}")

    n = len(rows)
    if not n:
        print("no pairs scored", file=sys.stderr)
        return 1
    hit = sum(r["hit"] for r in rows)
    claimed = sum(r["claimed"] for r in rows)
    fp = sum(r["claimed"] and not r["hit"] for r in rows)
    errs = [r["err"] for r in rows if r["hit"]]

    print(f"\n\n== {n} pairs, {args.backend}/{model}, start pose, "
          f"{time.time() - t0:.0f}s ==")
    print(f"  localised within {HIT_M} m   {hit / n:6.1%}   ({hit}/{n})")
    print(f"  claimed visible          {claimed / n:6.1%}   ({claimed}/{n})")
    print(f"  false positives          {fp / n:6.1%}   ({fp}/{n})  "
          f"claimed but >{HIT_M} m off")
    if errs:
        print(f"  error of the hits        median {np.median(errs):.2f} m   "
              f"p90 {np.percentile(errs, 90):.2f} m")
    print("\n  baseline, same 60 pairs, same frame "
          "(scripts/retrieval_probe.py):")
    print("    object map at the start pose       48.3%")
    print("    object map after the whole tour    67%  (k=3)")
    out = Path("artifacts/vlm_sweep.json")
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
