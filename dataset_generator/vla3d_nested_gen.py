"""Deterministic nested Q&A generation (no LLM).

Combines two VLA-3D primitives into one question:
    inner relation:  on / above / below   (from scene_graph)
    outer relation:  closest / farthest   (computed per-class from positions)

Produces patterns like:
    Find the bowl on the table closest to the folding screen.    (ref)
    Find the lamp above the table farthest from the window.       (ref)
    How many pillows are on the sofa closest to the TV cabinet?   (num)
    Find the picture between the door and the window.             (ref, between)

Every emitted pair has 100% verified geometry by construction —
no LLM hallucination, no backward-verification needed.

Output: vla3d_nested.jsonl
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from geometry import wall_between
from phrasing import apply_omit_find
from vla3d_loader import VLAObject, VLAScene, load_all_vla_scenes, render_object_list
from vla3d_num_gen import BAD_LABELS, pluralize

# No RNG is needed: this generator is deterministic by traversal — it walks
# scenes / regions / relationships / labels in insertion order (dicts preserve
# it in Python 3.7+) and accepts/rejects candidates by geometric thresholds.
# Same VLA-3D input → byte-identical jsonl, no seed argument needed.

HERE = Path(__file__).parent

# Target sizes after a deterministic, pattern-stratified downsample. The raw
# pools are far larger now that `near` is also an inner relation, so we sample
# down for per-pattern / per-anchor diversity and to hit the corpus's nested
# target (~50% of the 12k object_reference goal).
TARGET_NESTED_REF = 4000
TARGET_NESTED_NUM = 40
DOWNSAMPLE_SEED = 42
# Fraction of nested ref questions rephrased to drop the leading "Find".
OMIT_FIND_FRAC = 0.10

# Relation-word reweight (moderate, not an exact fit to the 66-occurrence
# official sample). The official ref+num set is `on`/`closest`-dominated with
# `near` ~8% and `farthest` ~5%; our supply-driven default was ~29% near and
# ~29% farthest (the `near`-as-inner supply unlock + farthest being the free
# symmetric twin of closest). We make `closest` the primary outer by taking the
# full supply of every `*_closest` template (and `between`), while hard-capping
# the over-supplied minority relations so they stay minority regardless of how
# much geometric supply exists:
#   - `*_farthest` templates  -> FARTHEST_TEMPLATE_CAP each
#   - `*_near` (near as outer) -> NEAR_OUTER_TEMPLATE_CAP each
#   - `near_closest` / `near_farthest` (near as inner, ~10k combined supply)
#     -> NEAR_INNER_CAP each (near_farthest falls under the farthest cap first)
# `near`-as-inner is still used (capped) to clear the non-near geometric ceiling.
NEAR_INNER_CAP = 320
FARTHEST_TEMPLATE_CAP = 45
NEAR_OUTER_TEMPLATE_CAP = 30


def _template_cap(tpl: str, supply: int) -> int:
    """Per-template sample cap implementing the relation reweight above."""
    if tpl == "NEST_between":
        return supply
    if tpl.endswith("_farthest"):          # incl. near_farthest
        return min(supply, FARTHEST_TEMPLATE_CAP)
    if tpl.endswith("_near"):              # near as the OUTER relation
        return min(supply, NEAR_OUTER_TEMPLATE_CAP)
    if tpl.startswith("NEST_near_"):       # near_closest (near as inner)
        return min(supply, NEAR_INNER_CAP)
    return supply                          # every *_closest with a non-near inner

# Architectural / structural labels that are fine as ANCHORS ("farthest from
# the floor", "between a door frame and a window") but nonsensical as the
# answer TARGET — "the ceiling above the spoon" is vacuous since the ceiling is
# above everything. Excluded from target selection only (includes the VLA-3D
# misspelling "celling" and plural variants the BAD_LABELS set misses).
NON_TARGET_LABELS = {
    "celling", "ceiling", "windows", "window", "wall", "walls",
    "exterior walls", "interior wall", "floor", "stair", "stairs",
    "column", "columns", "door", "door frame", "doorframe",
    "curtain", "curtains", "ceiling beam",
}

# Anchor2 must be uniquely the closest/farthest by this much (meters)
CLOSEST_MARGIN_M = 0.3
# 'near' outer: anchor1 is the ONLY mate within this radius (m) of anchor2.
# Distinct geometry from 'closest' (not just reworded) — avoids duplicate
# samples while matching the official set's frequent loose "near" phrasing.
NEAR_RADIUS_M = 1.5
# Cap per (pattern, anchor1) to avoid one anchor dominating. Raised from 3
# to lift the nested share toward the official ~50% (the geometric ceiling
# of the on/above/below × closest/farthest patterns is otherwise ~4.7k).
MAX_PER_PATTERN_PER_ANCHOR = 12
# Numerical answers in [2, 6] (matches official Q&A range)
MIN_COUNT, MAX_COUNT = 2, 6


def xy_dist(a: VLAObject, b: VLAObject) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def build_by_label(sc: VLAScene) -> dict[str, list[VLAObject]]:
    out: dict[str, list[VLAObject]] = defaultdict(list)
    for o in sc.objects:
        if o.region_id >= 0:
            out[o.raw_label].append(o)
    return dict(out)


def closest_of_label_to(label_instances: list[VLAObject], anchor: VLAObject
                        ) -> tuple[VLAObject | None, float, float]:
    """Return (closest_instance, closest_dist, second_dist). second_dist is inf
    if only one instance."""
    cands = [o for o in label_instances if o.id != anchor.id]
    if not cands:
        return None, math.inf, math.inf
    sorted_cands = sorted(cands, key=lambda o: xy_dist(o, anchor))
    closest = sorted_cands[0]
    closest_d = xy_dist(closest, anchor)
    second_d = xy_dist(sorted_cands[1], anchor) if len(sorted_cands) > 1 else math.inf
    return closest, closest_d, second_d


def farthest_of_label_to(label_instances: list[VLAObject], anchor: VLAObject
                         ) -> tuple[VLAObject | None, float, float]:
    """Return (farthest_instance, farthest_dist, second_max_dist)."""
    cands = [o for o in label_instances if o.id != anchor.id]
    if not cands:
        return None, -math.inf, -math.inf
    sorted_cands = sorted(cands, key=lambda o: xy_dist(o, anchor), reverse=True)
    farthest = sorted_cands[0]
    farthest_d = xy_dist(farthest, anchor)
    second_d = xy_dist(sorted_cands[1], anchor) if len(sorted_cands) > 1 else -math.inf
    return farthest, farthest_d, second_d


def unique_near_of_label_to(label_instances: list[VLAObject], anchor: VLAObject,
                            radius: float) -> VLAObject | None:
    """The single mate within `radius` (xy) of anchor, or None if 0 or >1 are.
    Used for the 'near' outer relation so "the X near the Y" is unambiguous."""
    within = [o for o in label_instances
              if o.id != anchor.id and xy_dist(o, anchor) <= radius]
    return within[0] if len(within) == 1 else None


# ── Pair constructors ────────────────────────────────────────────────────────

def make_ref(scene: str, template: str, question: str,
             target_id: int, target_label: str, anchors: list[int],
             object_list: list[str], region_id: int) -> dict:
    return {
        "scene": scene,
        "type": "object_reference",
        "source": "vla3d_nested",
        "template": template,
        "question": question,
        "object_list": object_list,
        "answer": {"object_id": target_id, "label": target_label},
        "target": target_id,
        "anchors": anchors,
        "region_id": region_id,
    }


def make_num(scene: str, template: str, question: str,
             count: int, anchors: list[int],
             object_list: list[str], region_id: int) -> dict:
    return {
        "scene": scene,
        "type": "numerical",
        "source": "vla3d_nested",
        "template": template,
        "question": question,
        "object_list": object_list,
        "answer": count,
        "target": None,
        "anchors": anchors,
        "region_id": region_id,
    }


# ── Pattern: <target> <inner_rel> <anchor1> <outer_rel> <anchor2> ────────────

def _emit_inner_outer(sc: VLAScene, by_label: dict[str, list[VLAObject]],
                      inner_rel: str, inner_prep: str,
                      outer: str) -> list[dict]:
    """
    inner_rel ∈ {on, above, below}    → relation in scene_graph
    outer ∈ {closest, farthest}        → computed per-class
    Emits ref or num pairs.
    """
    pairs: list[dict] = []
    template_id = f"NEST_{inner_rel}_{outer}"
    outer_word = {"closest": "closest to", "farthest": "farthest from",
                  "near": "near"}[outer]

    for region_id, rel_dict in sc.relationships.items():
        if region_id < 0:
            continue
        inner_map = rel_dict.get(inner_rel, {})

        for anchor1_id, target_ids in inner_map.items():
            anchor1 = sc.by_id.get(anchor1_id)
            if anchor1 is None or anchor1.raw_label in BAD_LABELS:
                continue

            # We need ≥2 instances of anchor1's label for the outer relation
            # to actually narrow things down.
            anchor1_mates = by_label.get(anchor1.raw_label, [])
            if len(anchor1_mates) < 2:
                continue

            # Group targets on this anchor1 by their own label
            targets_by_label: dict[str, list[VLAObject]] = defaultdict(list)
            for tid in target_ids:
                t = sc.by_id.get(tid)
                if (t and t.raw_label not in BAD_LABELS
                        and t.raw_label not in NON_TARGET_LABELS
                        and t.region_id >= 0):
                    targets_by_label[t.raw_label].append(t)
            if not targets_by_label:
                continue

            # Try each singleton anchor2 candidate
            emitted_for_anchor1 = 0
            for anchor2_label, anchor2_list in by_label.items():
                if len(anchor2_list) != 1:
                    continue
                anchor2 = anchor2_list[0]
                if anchor2.raw_label in BAD_LABELS or anchor2.id == anchor1.id:
                    continue
                if anchor2.raw_label == anchor1.raw_label:
                    continue

                # Is anchor1 uniquely closest/farthest among its mates?
                if outer == "closest":
                    selected, sel_d, second_d = closest_of_label_to(anchor1_mates, anchor2)
                    if selected is None or selected.id != anchor1.id:
                        continue
                    if second_d - sel_d < CLOSEST_MARGIN_M:
                        continue
                elif outer == "farthest":
                    selected, sel_d, second_d = farthest_of_label_to(anchor1_mates, anchor2)
                    if selected is None or selected.id != anchor1.id:
                        continue
                    if sel_d - second_d < CLOSEST_MARGIN_M:
                        continue
                else:  # near — anchor1 is the only mate within NEAR_RADIUS_M
                    selected = unique_near_of_label_to(anchor1_mates, anchor2, NEAR_RADIUS_M)
                    if selected is None or selected.id != anchor1.id:
                        continue

                # Wall-between gate (closest/near outer): "the X on anchor1
                # closest/near to anchor2" isn't navigable proximity if a wall
                # separates anchor1 and anchor2. `farthest` is exempt (naturally
                # across the room). Same shared gate as single-layer ref.
                if outer in ("closest", "near") and wall_between(sc, anchor1, anchor2):
                    continue

                # Keep nested object_list scene-wide: anchor2 is a *scene*
                # singleton and may live in a different region than the
                # target. A region filter would drop it from object_list,
                # making the question unanswerable.
                ol = render_object_list(sc)

                # Emit one pair per target_label
                for tgt_label, tgt_objs in targets_by_label.items():
                    # Target must not share a label with either anchor, or the
                    # question reads "Find the X ... near the X" — ambiguous.
                    if tgt_label in (anchor1.raw_label, anchor2.raw_label):
                        continue
                    n = len(tgt_objs)
                    # Redundancy gate (ref only): skip if the target class is
                    # already unique scene-wide — then "Find the X" needs no
                    # constraint, unlike the official set (29/30 non-unique).
                    # Counting (num) questions are exempt: a count is meaningful
                    # regardless of how many same-class objects exist.
                    if n == 1 and len(by_label.get(tgt_label, [])) < 2:
                        continue
                    if n == 1:
                        t = tgt_objs[0]
                        q = (f"Find the {tgt_label} {inner_prep} the "
                             f"{anchor1.raw_label} {outer_word} the {anchor2.raw_label}.")
                        pairs.append(make_ref(sc.name, template_id, q,
                                              t.id, tgt_label,
                                              [anchor1.id, anchor2.id], ol, region_id))
                        emitted_for_anchor1 += 1
                    elif MIN_COUNT <= n <= MAX_COUNT:
                        q = (f"How many {pluralize(tgt_label)} are {inner_prep} the "
                             f"{anchor1.raw_label} {outer_word} the {anchor2.raw_label}?")
                        pairs.append(make_num(sc.name, template_id, q, n,
                                              [anchor1.id, anchor2.id], ol, region_id))
                        emitted_for_anchor1 += 1

                    if emitted_for_anchor1 >= MAX_PER_PATTERN_PER_ANCHOR:
                        break
                if emitted_for_anchor1 >= MAX_PER_PATTERN_PER_ANCHOR:
                    break
    return pairs


# ── Pattern: between two singleton anchors ───────────────────────────────────

def _emit_between(sc: VLAScene, by_label: dict[str, list[VLAObject]]) -> list[dict]:
    """Use scene_graph['between'] precomputed list. Anchors must both be
    singletons so 'between X and Y' is unambiguous."""
    pairs: list[dict] = []
    for region_id, rel_dict in sc.relationships.items():
        if region_id < 0:
            continue
        between_map = rel_dict.get("between", {})
        for target_id, anchor_pairs in between_map.items():
            target = sc.by_id.get(target_id)
            if target is None or target.raw_label in BAD_LABELS:
                continue
            # anchor_pairs is a list of [a,b] anchor id pairs
            for ap in anchor_pairs:
                if not isinstance(ap, list) or len(ap) != 2:
                    continue
                a1 = sc.by_id.get(ap[0]); a2 = sc.by_id.get(ap[1])
                if a1 is None or a2 is None:
                    continue
                if a1.raw_label in BAD_LABELS or a2.raw_label in BAD_LABELS:
                    continue
                if a1.raw_label == a2.raw_label:
                    continue
                # Both anchors must be singleton
                if len(by_label.get(a1.raw_label, [])) != 1: continue
                if len(by_label.get(a2.raw_label, [])) != 1: continue
                # Target must be unique among its label-mates as being between THESE two
                mates = by_label.get(target.raw_label, [])
                same_role = []
                for m in mates:
                    m_pairs = between_map.get(m.id, [])
                    if any(set(p) == {a1.id, a2.id} for p in m_pairs if isinstance(p, list)):
                        same_role.append(m.id)
                if len(same_role) != 1 or same_role[0] != target.id:
                    continue
                # Redundancy gate: skip if the target class is already unique
                # scene-wide — "between X and Y" would be a vacuous constraint.
                if len(by_label.get(target.raw_label, [])) < 2:
                    continue
                q = (f"Find the {target.raw_label} between the "
                     f"{a1.raw_label} and the {a2.raw_label}.")
                # Scene-wide: both anchors are scene singletons; one or both
                # may live in a different region than target.
                ol = render_object_list(sc)
                pairs.append(make_ref(sc.name, "NEST_between", q,
                                      target.id, target.raw_label,
                                      [a1.id, a2.id], ol, region_id))
    return pairs


# ── Driver ────────────────────────────────────────────────────────────────────

def _stratified_downsample(pairs: list[dict], target: int,
                           rng: random.Random) -> list[dict]:
    """Cap each template to `_template_cap` (the relation reweight) and keep all
    survivors. `target` is an upper bound: if the capped pool still exceeds it,
    trim with a round-robin across templates so no pattern dominates the trim.
    In practice the caps bind well below `target`, so this returns the capped
    pool — `farthest`/`near` stay minority by construction, `closest` leads."""
    by_tpl: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_tpl[p["template"]].append(p)
    for lst in by_tpl.values():
        rng.shuffle(lst)

    capped: dict[str, list[dict]] = {
        t: lst[:_template_cap(t, len(lst))] for t, lst in by_tpl.items()
    }
    kept: list[dict] = [p for lst in capped.values() for p in lst]
    if len(kept) <= target:
        rng.shuffle(kept)
        return kept

    # Over target even after caps: round-robin trim down to `target`.
    out: list[dict] = []
    tpls = sorted(capped)
    while len(out) < target:
        progressed = False
        for t in tpls:
            if capped[t]:
                out.append(capped[t].pop())
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
    rng.shuffle(out)
    return out


def generate_scene(sc: VLAScene) -> tuple[list[dict], dict]:
    by_label = build_by_label(sc)
    counter: Counter = Counter()
    out: list[dict] = []
    inner_rels = [("on", "on"), ("above", "above"), ("below", "below"),
                  ("hanging_on", "hanging on"), ("in", "in"), ("near", "near")]
    for inner_rel, inner_prep in inner_rels:
        for outer in ("closest", "farthest", "near"):
            if inner_rel == "near" and outer == "near":
                continue  # avoid the awkward "X near the Y near the Z"
            ps = _emit_inner_outer(sc, by_label, inner_rel, inner_prep, outer)
            out.extend(ps)
            counter[f"{inner_rel}+{outer}"] = len(ps)
    bt = _emit_between(sc, by_label)
    out.extend(bt)
    counter["between"] = len(bt)
    return out, dict(counter)


def main(out_path: Path = HERE.parent / "dataset" / "vla3d_nested.jsonl") -> None:
    scenes = load_all_vla_scenes()
    print(f"Loaded {len(scenes)} scenes\n")

    all_pairs: list[dict] = []
    per_scene = {}
    per_pattern: Counter = Counter()
    for name, sc in scenes.items():
        pairs, c = generate_scene(sc)
        all_pairs.extend(pairs)
        per_scene[name] = len(pairs)
        per_pattern.update(c)
        print(f"  {name:<25} {len(pairs):>4} pairs   {c}")

    print(f"\nTotal raw nested pairs: {len(all_pairs)}")
    print("\nPer pattern (raw):")
    for p, n in sorted(per_pattern.items()):
        print(f"  {p:<25} {n:>5}")

    # Deterministic, pattern-stratified downsample to the target sizes.
    rng = random.Random(DOWNSAMPLE_SEED)
    ref_raw = [p for p in all_pairs if p["type"] == "object_reference"]
    # Nested num is kept small and excludes any `near` (inner OR outer): the
    # official numerical set is on-dominated and `near`-light, so near-bearing
    # compositional counts would pull the merged num distribution off-target.
    num_raw = [p for p in all_pairs
               if p["type"] == "numerical" and "near" not in p["template"]]
    ref_kept = _stratified_downsample(ref_raw, TARGET_NESTED_REF, rng)
    num_kept = _stratified_downsample(num_raw, TARGET_NESTED_NUM, rng)
    # Rephrase ~10% of ref to drop "Find" (official style); num is untouched.
    n_omit = apply_omit_find(ref_kept, rng, OMIT_FIND_FRAC)
    all_pairs = ref_kept + num_kept
    print(f"\nDownsampled: ref {len(ref_raw)}→{len(ref_kept)}, "
          f"num {len(num_raw)}→{len(num_kept)}  (omit-Find applied to {n_omit})")

    # Split type counts
    from collections import Counter as C
    type_counts = C(p["type"] for p in all_pairs)
    print(f"By type: {dict(type_counts)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n=> wrote {len(all_pairs)} pairs to {out_path}")

    # Sample
    print("\nSamples (one per pattern):")
    seen = set()
    for p in all_pairs:
        if p["template"] in seen: continue
        seen.add(p["template"])
        ans = p["answer"]["object_id"] if isinstance(p["answer"], dict) else p["answer"]
        print(f"  [{p['template']:<22}] ({p['scene']:18s}) {p['question']}  → {ans}")
        if len(seen) >= 7: break


if __name__ == "__main__":
    main()
