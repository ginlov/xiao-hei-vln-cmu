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
from collections import Counter, defaultdict
from pathlib import Path

from vla3d_loader import load_all_vla_scenes, render_object_list, VLAScene, VLAObject
from vla3d_num_gen import pluralize, BAD_LABELS

# No RNG is needed: this generator is deterministic by traversal — it walks
# scenes / regions / relationships / labels in insertion order (dicts preserve
# it in Python 3.7+) and accepts/rejects candidates by geometric thresholds.
# Same VLA-3D input → byte-identical jsonl, no seed argument needed.

HERE = Path(__file__).parent

# Anchor2 must be uniquely the closest/farthest by this much (meters)
CLOSEST_MARGIN_M = 0.3
# Cap per (pattern, anchor1) to avoid one anchor dominating
MAX_PER_PATTERN_PER_ANCHOR = 3
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


# ── Pair constructors ────────────────────────────────────────────────────────

def make_ref(scene: str, template: str, question: str,
             target_id: int, target_label: str, anchors: list[int],
             object_list: list[str]) -> dict:
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
    }


def make_num(scene: str, template: str, question: str,
             count: int, anchors: list[int],
             object_list: list[str]) -> dict:
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
    outer_word = "closest to" if outer == "closest" else "farthest from"

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
                if t and t.raw_label not in BAD_LABELS and t.region_id >= 0:
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
                else:
                    selected, sel_d, second_d = farthest_of_label_to(anchor1_mates, anchor2)
                    if selected is None or selected.id != anchor1.id:
                        continue
                    if sel_d - second_d < CLOSEST_MARGIN_M:
                        continue

                ol = render_object_list(sc)

                # Emit one pair per target_label
                for tgt_label, tgt_objs in targets_by_label.items():
                    n = len(tgt_objs)
                    if n == 1:
                        t = tgt_objs[0]
                        q = (f"Find the {tgt_label} {inner_prep} the "
                             f"{anchor1.raw_label} {outer_word} the {anchor2.raw_label}.")
                        pairs.append(make_ref(sc.name, template_id, q,
                                              t.id, tgt_label,
                                              [anchor1.id, anchor2.id], ol))
                        emitted_for_anchor1 += 1
                    elif MIN_COUNT <= n <= MAX_COUNT:
                        q = (f"How many {pluralize(tgt_label)} are {inner_prep} the "
                             f"{anchor1.raw_label} {outer_word} the {anchor2.raw_label}?")
                        pairs.append(make_num(sc.name, template_id, q, n,
                                              [anchor1.id, anchor2.id], ol))
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
                q = (f"Find the {target.raw_label} between the "
                     f"{a1.raw_label} and the {a2.raw_label}.")
                ol = render_object_list(sc)
                pairs.append(make_ref(sc.name, "NEST_between", q,
                                      target.id, target.raw_label,
                                      [a1.id, a2.id], ol))
    return pairs


# ── Driver ────────────────────────────────────────────────────────────────────

def generate_scene(sc: VLAScene) -> tuple[list[dict], dict]:
    by_label = build_by_label(sc)
    counter: Counter = Counter()
    out: list[dict] = []
    for inner_rel, inner_prep in [("on", "on"), ("above", "above"), ("below", "below")]:
        for outer in ("closest", "farthest"):
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

    print(f"\nTotal nested pairs: {len(all_pairs)}")
    print("\nPer pattern:")
    for p, n in sorted(per_pattern.items()):
        print(f"  {p:<25} {n:>5}")

    # Split type counts
    from collections import Counter as C
    type_counts = C(p["type"] for p in all_pairs)
    print(f"\nBy type: {dict(type_counts)}")

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
