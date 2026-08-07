#!/usr/bin/env python3
"""What the object map can offer a language query, and when.

Instruction following is 6 of the 17 points per scene and every one of its
statements names objects: "the small table farthest from the columns". Before
any of that can be reasoned about, the named thing has to be findable in the
map. Whatever an LLM does downstream, it cannot pick an answer that retrieval
never returned -- so this measures the ceiling.

Two knobs matter and they trade against each other:

  k         how many candidates to return. Instructions routinely name one of
            several ("the chair closest to the TV" -- six chairs), so k = 1 is
            structurally wrong for those.
  distance  the map grows as the robot drives, and the robot's motion *is* the
            answer, so a candidate that only appears after twelve metres is
            worth less than one available at the start.

Candidates and anchors want opposite filters, which is why `find` takes them
as arguments rather than baking one policy in:

  candidate ("the small table")   missing it loses the question outright; a
                                  spare duplicate only costs a slot -> loose
  anchor    ("from the columns")  one phantom column corrupts every distance
                                  the predicate computes -> strict

Runs entirely off `viz/data/<scene>.json`, so no server and no GPU.

    uv run python scripts/retrieval_probe.py
    uv run python scripts/retrieval_probe.py --strict-anchor --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vlm_sweep import family_key                       # noqa: E402
from traj_tolerance import mentioned                      # noqa: E402

CHALLENGE = Path.home() / "Workspace/vln-challenge/CMU-VLN-Challenge-2026"
VIZ = Path("viz/data")
# The reference trajectories keep a median 0.58 m from the objects they name,
# so a candidate this close is close enough to drive to.
HIT_M = 1.0
CLUSTER_M = 0.8


class Scene:
    """One exported scene, queryable as of a given frame."""

    def __init__(self, path: Path):
        d = json.loads(path.read_text())
        self.labels = [x.strip().lower() for x in d["labels"]]
        self.vocab = set(self.labels)
        self.objects = d["objects"]
        self.gt = d["gt"]
        # Cumulative distance travelled by each exported frame, so results can
        # be reported in metres -- the unit the reference trajectories use --
        # rather than in frame indices, which two layers of gating make opaque.
        p = np.array([f["p"] for f in d["frames"]])
        self.dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        self.n_frames = len(d["frames"])

    def frame_at(self, metres: float) -> int:
        return int(np.searchsorted(self.dist, metres, side="right") - 1)

    def gt_of(self, phrase: str) -> list[np.ndarray]:
        k = family_key(phrase)
        return [np.array(o["c"]) for o in self.gt
                if family_key(self.labels[o["l"]]) == k]


def find(scene: Scene, phrase: str, *, upto: int, top_k: int,
         min_vol: float = 0.0, rel_evidence: float = 0.0) -> list[dict]:
    """Ranked candidate positions for a phrase, as of frame ``upto``.

    Nodes are grouped by synonym family, restricted to those already observed,
    merged by proximity, and ranked by how much evidence supports them. The
    evidence count is recomputed for the cut rather than taken from the
    finished map: early on every node has been seen once, and a filter that
    needs three observations would return nothing at all.
    """
    key = family_key(phrase)
    live = []
    for o in scene.objects:
        if family_key(scene.labels[o["l"]]) != key:
            continue
        seen = [f for f in o.get("fs", []) if f <= upto]
        if not seen:
            continue
        size = np.array(o["hi"]) - np.array(o["lo"])
        if float(np.prod(size)) < min_vol:
            continue
        live.append({"c": np.array(o["c"]), "n": len(seen), "size": size,
                     "label": scene.labels[o["l"]], "first": seen[0]})
    live.sort(key=lambda x: -x["n"])

    merged: list[dict] = []
    for cand in live:
        for g in merged:
            if float(np.linalg.norm(g["c"][:2] - cand["c"][:2])) < CLUSTER_M:
                g["n"] += cand["n"]
                g["members"] += 1
                break
        else:
            merged.append({**cand, "members": 1})

    if merged and rel_evidence > 0:
        cut = rel_evidence * max(m["n"] for m in merged)
        merged = [m for m in merged if m["n"] >= cut]
    merged.sort(key=lambda x: -x["n"])
    return merged[:top_k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--viz-data", default=str(VIZ))
    ap.add_argument("--challenge", default=str(CHALLENGE))
    ap.add_argument("--strict-anchor", action="store_true",
                    help="apply the anchor policy (rel>=30%% + volume) instead "
                         "of the loose candidate policy")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qs = {e["scene"]: e["questions"] for e in
          json.loads((Path(args.challenge) / "questions/questions.json").read_text())}
    viz = Path(args.viz_data)

    cfg = dict(min_vol=0.001, rel_evidence=0.30) if args.strict_anchor \
        else dict(min_vol=0.0, rel_evidence=0.0)
    metres = [0.0, 1.3, 4.8, 12.4, 1e9]
    ks = [1, 3, 5, 10]
    grid = {(k, m): [0, 0] for k in ks for m in metres}
    misses: dict[str, int] = {}

    for scene_name in sorted(qs):
        f = viz / f"{scene_name}.json"
        if not f.is_file():
            continue
        sc = Scene(f)
        for question in qs[scene_name]["instruction_following"]:
            for phrase in mentioned(question, sc.vocab):
                truth = sc.gt_of(phrase)
                if not truth:
                    continue
                for m in metres:
                    upto = sc.frame_at(min(m, sc.dist[-1]))
                    got = find(sc, phrase, upto=upto, top_k=max(ks), **cfg)
                    for k in ks:
                        hit = any(
                            float(np.linalg.norm(c["c"][:2] - g[:2])) <= HIT_M
                            for g in truth for c in got[:k])
                        grid[(k, m)][0] += int(hit)
                        grid[(k, m)][1] += 1
                        if not hit and k == max(ks) and m == metres[-1]:
                            misses[phrase] = misses.get(phrase, 0) + 1

    policy = "anchor (rel>=30% + vol)" if args.strict_anchor else "candidate (loose)"
    print(f"top-k recall of the objects the 14 instruction questions name "
          f"-- policy: {policy}\n")
    hdr = f"{'':>6s} " + " ".join(
        f"{('full' if m > 1e8 else f'{m:.1f} m'):>8s}" for m in metres)
    print(hdr)
    print("-" * len(hdr))
    for k in ks:
        cells = []
        for m in metres:
            h, t = grid[(k, m)]
            cells.append(f"{h / max(t, 1):7.0%} ")
        print(f"k={k:<4d} " + " ".join(cells))

    n = grid[(max(ks), metres[-1])][1]
    print(f"\n  {n} (question, named object) pairs; a hit means a ground-truth "
          f"instance\n  sits within {HIT_M} m of one of the top k candidates.")
    if misses:
        print(f"\n  never retrieved even at k={max(ks)} on the full map:")
        for phrase, c in sorted(misses.items(), key=lambda kv: -kv[1]):
            print(f"    {phrase:26s} x{c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
