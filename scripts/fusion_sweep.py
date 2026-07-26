#!/usr/bin/env python3
"""Sweep the fusion rules without paying for the lift each time.

89% of our nodes sit within 1 m of a real object and 65% have their centre
inside a ground-truth box, so they are not hallucinations — they are
duplicates: 1374 surplus nodes stacked on the 595 real ones. The autopsy puts
the opposite error, two objects fused into one, at 1.8%. A rule that
over-segments forty times more often than it over-merges is mistuned, and the
question is by how much.

Sweeping that answer used to cost a full replay per value. It does not have to:
the median-box estimator only ever uses each observation's own centre and
extent, so one lift pass can cache `(label, score, centre, extent, n_pts)` per
observation and every later fusion runs from that in memory.

    uv run python scripts/fusion_sweep.py                  # all scenes
    uv run python scripts/fusion_sweep.py --cache-only     # just build caches
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from xiao_hei_vln.perception import replay
from xiao_hei_vln.perception import object_map as om
from xiao_hei_vln.perception.eval import iou_3d, load_gt_from_zip
from xiao_hei_vln.perception.vocab import is_structure

DEFAULT_SCENES = ("arabic_room", "chinese_room", "japanese_room",
                  "livingroom_3", "loft", "office_1", "office_2")
GT_ROOT = Path.home() / "workspace/dataset/unity-scene"
CACHE_DIR = Path("artifacts/obs_cache")
LIFT = dict(min_move_m=0.15, min_rot_deg=10.0, min_score=0.35)
MATCH_M = 1.0

# Empirical synonym families, read off the label confusion matrix: each of
# these is a set of words the detector uses for one kind of thing, verified by
# where the nodes actually landed. `lantern` and `wall lamp` nodes match a
# ground-truth `focus light` 8 and 7 times respectively and their own spelling
# never or twice, which is why merging by name has to be many-to-many rather
# than a one-to-one translation table.
FAMILIES: tuple[frozenset[str], ...] = tuple(frozenset(f) for f in (
    {"lamp", "lantern", "wall lamp", "ceiling lamp", "ceiling light",
     "focus light", "bedroom light", "floor lamp", "table lamp"},
    {"couch", "sofa", "loveseat"},
    {"picture", "painting", "poster", "calligraphy painting", "photo",
     "wall art", "framed picture"},
    {"table", "dining table", "coffee table", "side table", "desk"},
    {"chair", "bench", "stool", "armchair", "office chair"},
    {"tv", "computer monitor", "monitor", "screen", "television"},
    {"cup", "coffee cup", "mug", "paper cup"},
    {"box", "paper box", "carton"},
    {"cabinet", "cupboard", "sideboard", "dresser", "wardrobe"},
    {"shelf", "bookcase", "bookshelf", "display ledge"},
    {"potted plant", "plant", "potted cactus", "flowers", "flower"},
    {"rug", "carpet", "mat"},
))


def norm(s: str) -> str:
    return s.strip().lower().replace("_", " ")


_FAMILY_OF = {}
for _f in FAMILIES:
    for _w in _f:
        _FAMILY_OF[_w] = _f


def family_key(label: str):
    """A hashable identity that groups synonyms; the label itself otherwise."""
    return _FAMILY_OF.get(label, label)


# ---------------------------------------------------------------------------
# Stage 1 — cache one lift pass per scene
# ---------------------------------------------------------------------------

def build_cache(scene: str, frames_root: Path) -> list[dict]:
    """Record every observation `ObjectMap.add` would have received."""
    obs: list[dict] = []
    original = om.ObjectMap.add

    def recording_add(self, label, score, pts, color_rgb=None, color_name=None):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0]:
            core = om._core_points(pts)
            lo, hi = om._aabb(core)
            c, _ = om.robust_center(core)
            centre = np.array(c) if c is not None else np.median(core, axis=0)
            obs.append(dict(label=norm(label), score=float(score),
                            centre=centre.astype(np.float32),
                            extent=(hi - lo).astype(np.float32),
                            n_pts=int(len(pts))))
        return original(self, label, score, pts, color_rgb, color_name)

    om.ObjectMap.add = recording_add
    try:
        replay.replay_lift(str(frames_root / f"{scene}_tour"), **LIFT)
    finally:
        om.ObjectMap.add = original
    return obs


# ---------------------------------------------------------------------------
# Stage 2 — fuse from the cache under a given rule
# ---------------------------------------------------------------------------

class Node:
    # `ptlist` keeps each observation's own point count, not just their sum:
    # how much of the object a given view saw is the natural weight for any
    # estimator that prefers the better looks over the average of all of them.
    __slots__ = ("key", "labels", "score", "centres", "extents", "ptlist")

    def __init__(self, key, label, score, centre, extent, n_pts):
        self.key, self.labels = key, collections.Counter({label: 1})
        self.score = score
        self.centres, self.extents, self.ptlist = [centre], [extent], [n_pts]

    def add(self, label, score, centre, extent, n_pts):
        self.labels[label] += 1
        self.score = max(self.score, score)
        self.centres.append(centre)
        self.extents.append(extent)
        self.ptlist.append(n_pts)

    @property
    def pts(self):
        return sum(self.ptlist)

    @property
    def n_obs(self):
        return len(self.centres)

    def box(self):
        c = np.median(np.array(self.centres), axis=0)
        h = np.median(np.array(self.extents), axis=0) / 2.0
        return c, h * 2.0, c - h, c + h


def fuse(obs, *, merge_dist, merge_iou, relative, use_families, nms_iou):
    """Group observations into nodes.

    `relative` scales the merge radius by the object's own size — a 0.4 m
    absolute threshold is generous for a mug and stingy for a sofa, and the
    duplicates concentrate on the large repeated furniture.
    """
    nodes: list[Node] = []
    for o in obs:
        key = family_key(o["label"]) if use_families else o["label"]
        lo, hi = o["centre"] - o["extent"] / 2, o["centre"] + o["extent"] / 2
        best, best_score = None, 0.0
        for nd in nodes:
            if nd.key != key:
                continue
            c, size, nlo, nhi = nd.box()
            dist = float(np.linalg.norm(c - o["centre"]))
            thr = merge_dist
            if relative:
                thr = max(merge_dist, relative * float(np.mean(size)))
            iou = iou_3d(nlo, nhi, lo, hi)
            if iou >= merge_iou or dist <= thr:
                s = iou + 1.0 / (dist + 1e-3)
                if s > best_score:
                    best, best_score = nd, s
        if best is None:
            nodes.append(Node(key, o["label"], o["score"], o["centre"],
                              o["extent"], o["n_pts"]))
        else:
            best.add(o["label"], o["score"], o["centre"], o["extent"], o["n_pts"])

    # Cross-label suppression, but keeping the loser's name as an alias — the
    # variant measured in TASK 23, which reaches "keep both" recall at
    # "suppress" object count.
    order = sorted(range(len(nodes)),
                   key=lambda i: (nodes[i].n_obs, nodes[i].score), reverse=True)
    dead: set[int] = set()
    for i in order:
        if i in dead:
            continue
        _, _, alo, ahi = nodes[i].box()
        for j in order:
            if j == i or j in dead:
                continue
            _, _, blo, bhi = nodes[j].box()
            if iou_3d(alo, ahi, blo, bhi) >= nms_iou:
                dead.add(j)
                nodes[i].labels.update(nodes[j].labels)
    return [nd for k, nd in enumerate(nodes) if k not in dead]


def recluster(nodes, *, radius, relative, rounds=8):
    """Agglomerative pass over finished nodes.

    `fuse` is greedy, online and never revisits a decision: once one chair has
    become two nodes, no later evidence can put them back together, however
    close their centres drift as observations accumulate. Every threshold in
    the sweep changes when that first decision is taken, not whether it can be
    undone — so a re-clustering pass is testing something none of them reach.

    Merging pools the observations, so the surviving node's median box is
    recomputed over both halves rather than picked from one.
    """
    for _ in range(rounds):
        merged = False
        i = 0
        while i < len(nodes):
            a = nodes[i]
            ac, asize, _, _ = a.box()
            j = i + 1
            while j < len(nodes):
                b = nodes[j]
                if b.key != a.key:
                    j += 1
                    continue
                bc, bsize, _, _ = b.box()
                thr = radius
                if relative:
                    thr = max(radius, relative * float(np.mean((asize + bsize) / 2)))
                if float(np.linalg.norm(ac - bc)) <= thr:
                    a.labels.update(b.labels)
                    a.score = max(a.score, b.score)
                    a.centres.extend(b.centres)
                    a.extents.extend(b.extents)
                    a.ptlist.extend(b.ptlist)
                    nodes.pop(j)
                    ac, asize, _, _ = a.box()
                    merged = True
                else:
                    j += 1
            i += 1
        if not merged:
            break
    return nodes


def prune(nodes, *, min_obs, min_pts):
    return [nd for nd in nodes if nd.n_obs >= min_obs and nd.pts >= min_pts]


# ---------------------------------------------------------------------------
# Stage 3 — score
# ---------------------------------------------------------------------------

def score(gt, nodes, *, drop_unknown):
    """Greedy nearest-centre match; a node hits if any of its labels fits."""
    truth = [g for g in gt if not (drop_unknown and norm(g["label"]) == "unknown")]
    by_label = collections.defaultdict(list)
    for g in truth:
        by_label[norm(g["label"])].append(g)
    used, ious = set(), []
    for nd in sorted(nodes, key=lambda n: -n.n_obs):
        c, size, lo, hi = nd.box()
        cands = [(lab, g) for lab in nd.labels for g in by_label[lab]
                 if id(g) not in used]
        if not cands:
            continue
        lab, best = min(cands, key=lambda t: np.linalg.norm(
            np.array(t[1]["center_3d"]) - c))
        d = float(np.linalg.norm(np.array(best["center_3d"]) - c))
        if d > MATCH_M:
            continue
        used.add(id(best))
        gc = np.array(best["center_3d"])
        gs = np.array(best["bbox_aabb"]["size"])
        ious.append(iou_3d(lo, hi, gc - gs / 2, gc + gs / 2))
    pred_n: collections.Counter = collections.Counter()
    for nd in nodes:
        pred_n[nd.labels.most_common(1)[0][0]] += 1
    gt_n = collections.Counter(norm(g["label"]) for g in truth)
    errs = [abs(pred_n.get(k, 0) - gt_n.get(k, 0)) for k in set(pred_n) | set(gt_n)]
    return dict(n=len(nodes), R=len(ious) / max(len(truth), 1),
                P=len(ious) / max(len(nodes), 1),
                iou=float(np.mean(ious or [0])),
                p25=float(np.mean([x >= .25 for x in ious] or [0])),
                p50=float(np.mean([x >= .5 for x in ious] or [0])),
                cmae=float(np.mean(errs or [0])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="*", default=list(DEFAULT_SCENES))
    ap.add_argument("--frames-root", default="frames")
    ap.add_argument("--gt-root", default=str(GT_ROOT))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--keep-unknown", action="store_true",
                    help="score the `unknown` ground-truth objects too; they "
                         "are 8.9% of GT and no label can ever match them")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenes, cached, gts = list(args.scenes or DEFAULT_SCENES), {}, {}
    for scene in scenes:
        cf = cache_dir / f"{scene}.pkl"
        if cf.is_file():
            cached[scene] = pickle.loads(cf.read_bytes())
            print(f"{scene:14s} {len(cached[scene]):6d} observations (cached)",
                  flush=True)
        else:
            frames = Path(args.frames_root)
            if not (frames / f"{scene}_tour").is_dir():
                print(f"{scene}: no corpus", file=sys.stderr)
                continue
            obs = build_cache(scene, frames)
            cf.write_bytes(pickle.dumps(obs))
            cached[scene] = obs
            print(f"{scene:14s} {len(obs):6d} observations (built)", flush=True)
        gts[scene] = [g for g in load_gt_from_zip(
            Path(args.gt_root) / f"{scene}.zip", scene)
            if not is_structure(g["label"])]
    if args.cache_only or not cached:
        return 0

    base = dict(merge_dist=om.MERGE_DIST, merge_iou=om.MERGE_IOU,
                relative=0.0, use_families=False, nms_iou=om.NMS_IOU)
    # (name, fuse overrides, prune overrides, recluster kwargs or None)
    configs: list[tuple] = [("greedy 0.4 (baseline)", {}, {})]
    for d in (0.6, 0.8, 1.0):
        configs.append((f"merge_dist={d}", {"merge_dist": d}, {}))
    for r in (0.5, 0.75, 1.0):
        configs.append((f"relative={r}", {"relative": r}, {}))
    configs.append(("families", {"use_families": True}, {}))
    configs.append(("families+0.8", {"use_families": True, "merge_dist": 0.8}, {}))
    configs.append(("families+rel0.75",
                    {"use_families": True, "relative": 0.75}, {}))
    configs.append(("fam+0.8+n_obs>=3",
                    {"use_families": True, "merge_dist": 0.8},
                    {"min_obs": 3}))
    # The agglomerative pass: undo the greedy splits rather than delay them.
    for r in (0.5, 0.8):
        configs.append((f"recluster={r}", {}, {}, {"radius": r, "relative": 0.0}))
    configs.append(("recluster rel0.75", {}, {}, {"radius": 0.4, "relative": 0.75}))
    configs.append(("fam+recluster0.8", {"use_families": True}, {},
                    {"radius": 0.8, "relative": 0.0}))
    configs.append(("fam+recluster rel0.75", {"use_families": True}, {},
                    {"radius": 0.4, "relative": 0.75}))
    configs.append(("fam+recl rel0.75+n_obs>=3", {"use_families": True},
                    {"min_obs": 3}, {"radius": 0.4, "relative": 0.75}))

    hdr = (f"\n{'config':20s} {'#obj':>5s} {'P':>6s} {'R':>6s} {'mIoU':>6s} "
           f"{'>=.25':>6s} {'>=.5':>6s} {'cMAE':>6s} {'score/2':>8s}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    results = {}
    for cfg_row in configs:
        name, over, pover = cfg_row[0], cfg_row[1], cfg_row[2]
        recl = cfg_row[3] if len(cfg_row) > 3 else None
        cfg = {**base, **over}
        pcfg = {"min_obs": 1, "min_pts": 15, **pover}
        rows = []
        for scene in cached:
            nodes = fuse(cached[scene], **cfg)
            if recl is not None:
                nodes = recluster(nodes, **recl)
            nodes = prune(nodes, **pcfg)
            rows.append(score(gts[scene], nodes,
                              drop_unknown=not args.keep_unknown))
        m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        results[name] = {k: m(k) for k in
                         ("n", "P", "R", "iou", "p25", "p50", "cmae")}
        print(f"{name:20s} {m('n'):5.0f} {m('P'):6.3f} {m('R'):6.3f} "
              f"{m('iou'):6.3f} {m('p25'):6.1%} {m('p50'):6.1%} {m('cmae'):6.2f} "
              f"{2 * m('p50') + (m('p25') - m('p50')):8.3f}", flush=True)

    out = Path("artifacts/fusion_sweep.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
