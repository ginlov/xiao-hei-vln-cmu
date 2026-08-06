#!/usr/bin/env python3
"""Does asking the detector for the question's own words beat a fixed vocabulary?

Retrieval finds only 67% of the objects the fourteen instruction-following
questions name (`scripts/retrieval_probe.py`). Splitting the misses shows the
cause is not detection but vocabulary: 14 of the 19 unfound objects -- `stairs`,
`whiteboard`, `exit sign`, `water cooler`, `projector screen` -- are simply
absent from the 110-class list we push at startup, so no amount of detector
quality could have produced them.

YOLO-World is open-vocabulary; the 110 classes are our own choice, and
`PerceptionPipeline.detect` already accepts a per-call class list at ~50 ms to
re-encode. Since the challenge publishes the question before the run, the words
to look for are known in advance. This measures what that is worth, and what it
costs.

The cost is the part worth measuring honestly: with only a handful of classes
every table-ish surface has nowhere to go but `table`, so recall can rise while
the ranking fills with phantoms. Hence two configurations, and two acceptance
numbers rather than one.

Stage A output goes to `detections_<tag>.jsonl`, never to the 110-class cache
the rest of the harness depends on.

    uv run python scripts/narrow_vocab_probe.py --scene chinese_room
    uv run python scripts/narrow_vocab_probe.py --scene all --configs base,q,q+base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_probe import CLUSTER_M, HIT_M  # noqa: E402
from traj_tolerance import mentioned  # noqa: E402

from xiao_hei_vln.perception import replay  # noqa: E402
from xiao_hei_vln.perception.lifter import PointLifter  # noqa: E402
from xiao_hei_vln.perception.object_map import ObjectMap  # noqa: E402
from xiao_hei_vln.perception.vocab import DEFAULT_PRIOR, is_structure  # noqa: E402
from xiao_hei_vln.scene.io import read_objects_from_zip  # noqa: E402

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
GT_ROOT = Path.home() / "workspace/dataset/unity-scene"
DEFAULT_SCENES = ("arabic_room", "chinese_room", "japanese_room",
                  "livingroom_3", "loft", "office_1", "office_2")
LIFT = dict(min_move_m=0.15, min_rot_deg=10.0, min_score=0.35, min_inliers=10)
KS = (1, 3, 5)

# `map wall decal` and `sphere decoration` are the kind of compound noun CLIP
# text encoders handle badly, so each phrase is also offered as its head noun.
# Costs nothing -- extra classes are one text encoding -- and lets the report
# say which spelling actually fired.
def expand(phrase: str) -> list[str]:
    words = phrase.split()
    out = [phrase]
    if len(words) > 1:
        out.append(words[-1])
    return out


def question_classes(question: str, vocab: set[str]) -> list[str]:
    """The words this question names, plus head-noun fallbacks."""
    seen, out = set(), []
    for p in mentioned(question, vocab):
        for v in expand(p):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def detect_to(frames_dir: Path, out_path: Path, classes: list[str], *,
              base_url: str) -> Path:
    """Stage A into a named file, leaving the shared cache alone."""
    from xiao_hei_vln.perception.client import HTTPPerceptionClient

    frames = replay.load_frames(frames_dir, min_move_m=LIFT["min_move_m"],
                                min_rot_deg=LIFT["min_rot_deg"])
    client = HTTPPerceptionClient(base_url=base_url)
    client.set_classes(tuple(classes))
    t0 = time.time()
    with out_path.open("w", buffering=1) as fh:
        for i, frame in enumerate(frames):
            # The client hands back decoded masks; the cache format is RLE, so
            # re-encode rather than invent a second on-disk shape that stage B
            # would have to special-case.
            dets = client.detect(frame.image_bgr())
            fh.write(json.dumps({
                "tick_id": frame.tick_id,
                "detections": [
                    {"label": d.label, "score": round(d.score, 4),
                     "mask_rle": replay._encode_mask(d.mask)} for d in dets]}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(frames)} frames "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return out_path


def fuse(frames_dir: Path, det_path: Path) -> list[dict]:
    """Stage B against a specific detections file."""
    frames = replay.load_frames(frames_dir, min_move_m=LIFT["min_move_m"],
                                min_rot_deg=LIFT["min_rot_deg"])
    dets: dict[int, list[dict]] = {}
    for line in det_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            dets[int(r["tick_id"])] = r["detections"]
    lifter = PointLifter(min_inliers=LIFT["min_inliers"])
    omap = ObjectMap()
    for frame in frames:
        d = dets.get(frame.tick_id)
        if not d:
            continue
        scan = frame.scan()
        for det in d:
            if float(det["score"]) < LIFT["min_score"]:
                continue
            res = lifter.lift(mask=replay.decode_mask(det["mask_rle"]),
                              scan_points_map=scan,
                              pose_position=frame.position,
                              pose_orientation=frame.orientation)
            if res.position is not None:
                omap.add(det["label"], float(det["score"]), res.inlier_points)
    return omap.export()


def retrieve(nodes: list[dict], phrase: str, top_k: int) -> list[np.ndarray]:
    """Same shape as retrieval_probe.find, but keyed on the literal phrase.

    Under a question-specific vocabulary the detector answers with the word it
    was asked for, so the synonym families that exact-string matching needs on
    the 110-class run have nothing left to do here -- which is precisely the
    effect being measured.
    """
    want = phrase.strip().lower()
    live = []
    for nd in nodes:
        lab = nd["label"].strip().lower()
        if lab != want and lab != want.split()[-1]:
            continue
        live.append((np.array(nd["center_3d"]), nd["n_obs"]))
    live.sort(key=lambda t: -t[1])
    merged: list[list] = []
    for c, n in live:
        for g in merged:
            if float(np.linalg.norm(g[0][:2] - c[:2])) < CLUSTER_M:
                g[1] += n
                break
        else:
            merged.append([c, n])
    merged.sort(key=lambda g: -g[1])
    return [g[0] for g in merged[:top_k]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="all")
    ap.add_argument("--frames-root", default="frames")
    ap.add_argument("--gt-root", default=str(GT_ROOT))
    ap.add_argument("--challenge", default=str(CHALLENGE))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--configs", default="q,q+base",
                    help="comma-separated: q (question words only), "
                         "q+base (question words plus the 110 classes)")
    ap.add_argument("--reuse", action="store_true",
                    help="skip stage A when the detections file already exists")
    args = ap.parse_args()

    qs = {e["scene"]: e["questions"] for e in
          json.loads((Path(args.challenge) / "questions/questions.json").read_text())}
    scenes = DEFAULT_SCENES if args.scene == "all" else (args.scene,)
    base = [c.strip().lower() for c in DEFAULT_PRIOR]
    rows = []

    for scene in scenes:
        fdir = Path(args.frames_root) / f"{scene}_tour"
        if not fdir.is_dir() or scene not in qs:
            print(f"{scene}: skipped (no corpus or no questions)", file=sys.stderr)
            continue
        gt_all = [e for e in read_objects_from_zip(
            Path(args.gt_root) / f"{scene}.zip", scene_name=scene).values()
            if not is_structure(e.label)]
        vocab = {e.label.strip().lower() for e in gt_all}

        for qi, question in enumerate(qs[scene]["instruction_following"]):
            phrases = mentioned(question, vocab)
            if not phrases:
                continue
            qcls = question_classes(question, vocab)
            for cfg in [c.strip() for c in args.configs.split(",")]:
                classes = qcls if cfg == "q" else sorted(set(base) | set(qcls))
                tag = f"{cfg.replace('+', '_')}_q{qi + 4}"
                out = fdir / f"detections_{tag}.jsonl"
                print(f"\n== {scene} q{qi + 4} [{cfg}] {len(classes)} classes ==",
                      flush=True)
                if not (args.reuse and out.is_file()):
                    detect_to(fdir, out, classes, base_url=args.base_url)
                nodes = fuse(fdir, out)
                print(f"   {len(nodes)} nodes", flush=True)

                for phrase in phrases:
                    truth = [np.array([e.center.x, e.center.y, e.center.z])
                             for e in gt_all
                             if e.label.strip().lower() == phrase]
                    if not truth:
                        continue
                    got = retrieve(nodes, phrase, max(KS))
                    rows.append({
                        "scene": scene, "q": qi + 4, "cfg": cfg, "phrase": phrase,
                        "hits": {k: any(
                            float(np.linalg.norm(c[:2] - g[:2])) <= HIT_M
                            for g in truth for c in got[:k]) for k in KS},
                        "n_nodes": len(nodes),
                    })

    print("\n\n== recall of named objects, by configuration ==")
    print("baseline (110 classes, measured separately): k=1 60%  k=3 67%\n")
    cfgs = sorted({r["cfg"] for r in rows})
    print(f"{'config':10s} {'pairs':>6s} " + " ".join(f"{'k=' + str(k):>7s}" for k in KS)
          + f" {'nodes/scene':>12s}")
    print("-" * 52)
    for cfg in cfgs:
        sub = [r for r in rows if r["cfg"] == cfg]
        cells = " ".join(
            f"{np.mean([r['hits'][k] for r in sub]):7.0%}" for k in KS)
        nn = np.mean([r["n_nodes"] for r in sub]) if sub else 0
        print(f"{cfg:10s} {len(sub):6d} {cells} {nn:12.0f}")

    print("\naccept if k=3 reaches 80% and k=1 does not fall below 60%: recall "
          "bought\nby forcing every table-ish surface into the one class asked "
          "for is not a gain.")
    out = Path("artifacts/narrow_vocab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, default=str, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
