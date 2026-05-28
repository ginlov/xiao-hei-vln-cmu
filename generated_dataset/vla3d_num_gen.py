"""Derive numerical (counting) Q&A pairs from VLA-3D scene_graph + colors.

We use the same N1–N5 templates as our earlier Phase 1, but powered by
VLA-3D's richer data: 8 relation types in `scene_graph.json` and 3 dominant
colors per object in `object_result.csv`.

Templates:

  N1  How many <X-plural> are on the <anchor>?              # uses scene_graph['on']
  N2  How many <X-plural> are near the <anchor>?            # ['near']
  N3  How many <X-plural> are above the <anchor>?           # ['above']
  N4  How many <X-plural> are there in the room?            # total count
  N5  How many <impossible-X> are there?  →  0              # refusal sample
  N6  How many <color> <X-plural> are on the <anchor>?      # color-conditioned
  N7  How many <X-plural> are below the <anchor>?           # ['below']
  N8  How many <X-plural> are hanging on the <anchor>?      # ['hanging_on']

Same output schema as template_generator.py — JSONL with answer as int.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from vla3d_loader import load_all_vla_scenes, render_object_list, VLAScene, VLAObject

random.seed(42)

# ── Filters & caps ────────────────────────────────────────────────────────────

STRUCTURAL_LABELS = {
    "wall", "exterior walls", "interior wall", "ceiling", "floor",
    "carpet", "unknown",
}
BORING_LABELS = {
    "focus light", "ceiling lamp", "spot light",
    "light switch", "ceiling light", "_",
}
BAD_LABELS = STRUCTURAL_LABELS | BORING_LABELS

# nyu40 classes that make sensible counting anchors (must be furniture-like)
GOOD_ANCHOR_NYU40 = {
    "table", "desk", "counter", "shelves", "bookshelf", "dresser",
    "cabinet", "night stand", "nightstand", "sofa", "chair", "bed",
    "ottoman", "stool", "bench", "picture",
}

MAX_PER_TEMPLATE_PER_SCENE = 20      # was 12; we have 8 templates now
MAX_REFUSALS_PER_SCENE = 4

# ── Pluralization (reused from template_generator) ────────────────────────────
ALREADY_PLURAL = {
    "books", "chopsticks", "clothes", "curtains", "drawers", "files",
    "flowers", "headphones", "notecards", "pillows", "shoes",
    "sofa pillows", "stairs", "sticky notes", "utensils", "windows",
    "eye glasses", "bathroom walls", "exterior walls",
}
SINGULAR_WITH_S = {"glass", "canvas", "chess", "mattress",
                   "bowl of apples", "potted cactus", "wine glass", "toilet glass"}
# words where -f / -fe → -ves; checked against actual VLA-3D label set
F_TO_VES = {"knife": "knives", "kitchen knife": "kitchen knives", "shelf": "shelves"}


def pluralize(noun: str) -> str:
    if noun in ALREADY_PLURAL:
        return noun
    if noun in F_TO_VES:
        return F_TO_VES[noun]
    if noun in SINGULAR_WITH_S:
        return noun + "es"
    if noun.endswith(("s", "x", "sh", "ch")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def usable_anchor(o: VLAObject) -> bool:
    if o.raw_label in BAD_LABELS or o.region_id < 0:
        return False
    return (
        o.nyu40_label in GOOD_ANCHOR_NYU40
        or "table" in o.raw_label
        or "cabinet" in o.raw_label
    )


def usable_target_label(label: str) -> bool:
    return label not in BAD_LABELS and not label.startswith("_")


def good_color(c: str) -> bool:
    return c and c not in ("N/A", "_", "")


# ── Per-scene utilities ───────────────────────────────────────────────────────

def label_singletons_in_scene(sc: VLAScene) -> dict[str, VLAObject]:
    """Labels that appear exactly once across the whole scene → unambiguous anchor."""
    counts = Counter(o.raw_label for o in sc.objects)
    return {o.raw_label: o for o in sc.objects if counts[o.raw_label] == 1}


def collect_related(sc: VLAScene, rel: str, anchor: VLAObject) -> list[VLAObject]:
    """All objects related to `anchor` via `rel` (within the anchor's region)."""
    rel_dict = sc.relationships.get(anchor.region_id, {}).get(rel, {})
    tgt_ids = rel_dict.get(anchor.id, [])
    return [sc.by_id[t] for t in tgt_ids if t in sc.by_id]


def group_by_label(objects: list[VLAObject]) -> dict[str, list[VLAObject]]:
    out: dict[str, list[VLAObject]] = defaultdict(list)
    for o in objects:
        out[o.raw_label].append(o)
    return out


# ── Template emitters ─────────────────────────────────────────────────────────

def make_pair(scene: str, template: str, question: str, answer: int,
              anchors: list[int], object_list: list[str]) -> dict:
    return {
        "scene": scene,
        "type": "numerical",
        "source": "vla3d_num",
        "template": template,
        "question": question,
        "object_list": object_list,
        "answer": answer,
        "anchors": anchors,
        "target": None,
    }


def emit_relation_count(sc: VLAScene, template_id: str, rel: str,
                        preposition: str) -> list[dict]:
    """N1 (on), N2 (near), N3 (above), N7 (below), N8 (hanging on).

    For each singleton anchor in the scene, count objects of each label related
    via `rel`, emit a question if count >= 2.
    """
    out: list[dict] = []
    singletons = label_singletons_in_scene(sc)
    for anchor_label, anchor in singletons.items():
        if not usable_anchor(anchor):
            continue
        related = collect_related(sc, rel, anchor)
        groups = group_by_label(related)
        for tgt_label, tgt_objs in groups.items():
            if not usable_target_label(tgt_label):
                continue
            n = len(tgt_objs)
            if n < 2:
                continue       # not very interesting if 0 or 1
            q = f"How many {pluralize(tgt_label)} are {preposition} the {anchor_label}?"
            ol = render_object_list(sc)
            out.append(make_pair(sc.name, template_id, q, n, [anchor.id], ol))
    return out


def emit_total_count(sc: VLAScene) -> list[dict]:
    """N4: total count of each label in the scene."""
    out: list[dict] = []
    label_counts = Counter(o.raw_label for o in sc.objects
                           if usable_target_label(o.raw_label) and o.region_id >= 0)
    ol = render_object_list(sc)
    for label, n in label_counts.items():
        if n < 3 or n > 30:   # uninteresting at extremes
            continue
        q = f"How many {pluralize(label)} are there in the room?"
        out.append(make_pair(sc.name, "N4", q, n, [], ol))
    return out


def emit_refusal(sc: VLAScene) -> list[dict]:
    """N5: ask for a category we KNOW is not in the scene → answer 0.

    We mine plausible-sounding nouns from OTHER scenes that DON'T appear here.
    """
    present = {o.raw_label for o in sc.objects}
    candidates = [
        "guitar", "piano", "elevator", "telescope", "treadmill", "bathtub",
        "stove", "fridge", "dishwasher", "microwave", "blender",
        "kettle", "umbrella", "skateboard", "bicycle", "helmet",
        "trophy", "globe", "fan", "projector", "printer",
    ]
    available = [c for c in candidates if c not in present]
    random.shuffle(available)
    ol = render_object_list(sc)
    out = []
    for label in available[:MAX_REFUSALS_PER_SCENE]:
        q = f"How many {pluralize(label)} are there in the room?"
        out.append(make_pair(sc.name, "N5", q, 0, [], ol))
    return out


def emit_color_on(sc: VLAScene) -> list[dict]:
    """N6: How many <color> <X> are on the <anchor>?

    Use VLA-3D's color_scheme1 (dominant color, named).
    """
    out: list[dict] = []
    singletons = label_singletons_in_scene(sc)
    for anchor_label, anchor in singletons.items():
        if not usable_anchor(anchor):
            continue
        on_objs = collect_related(sc, "on", anchor)
        # Group by (color, label)
        by_color_label: dict[tuple[str, str], list[VLAObject]] = defaultdict(list)
        for o in on_objs:
            if not usable_target_label(o.raw_label):
                continue
            c = o.colors[0]
            if not good_color(c):
                continue
            by_color_label[(c, o.raw_label)].append(o)
        # Only emit when (color, label) is unique enough to be a useful Q
        # AND the same label has at least 1 object of a different color
        # (otherwise color is redundant)
        labels_seen = Counter(o.raw_label for o in on_objs)
        for (color, label), objs in by_color_label.items():
            n = len(objs)
            if n < 1:
                continue
            if labels_seen[label] == n:
                # all objects of this label share the same color → color is redundant
                continue
            q = f"How many {color} {pluralize(label)} are on the {anchor_label}?"
            ol = render_object_list(sc)
            out.append(make_pair(sc.name, "N6", q, n, [anchor.id], ol))
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

def generate_scene(sc: VLAScene, rng: random.Random) -> list[dict]:
    bucket: dict[str, list[dict]] = defaultdict(list)

    bucket["N1"] = emit_relation_count(sc, "N1", "on", "on")
    bucket["N2"] = emit_relation_count(sc, "N2", "near", "near")
    bucket["N3"] = emit_relation_count(sc, "N3", "above", "above")
    bucket["N4"] = emit_total_count(sc)
    bucket["N5"] = emit_refusal(sc)
    bucket["N6"] = emit_color_on(sc)
    bucket["N7"] = emit_relation_count(sc, "N7", "below", "below")
    bucket["N8"] = emit_relation_count(sc, "N8", "hanging_on", "hanging on")

    out: list[dict] = []
    for tid, pairs in bucket.items():
        rng.shuffle(pairs)
        cap = MAX_REFUSALS_PER_SCENE if tid == "N5" else MAX_PER_TEMPLATE_PER_SCENE
        out.extend(pairs[:cap])
    return out


def main(out_path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    scenes = load_all_vla_scenes()
    print(f"Loaded {len(scenes)} scenes\n")

    all_pairs: list[dict] = []
    per_scene_counts = {}
    per_template_counts: Counter = Counter()
    for name, sc in scenes.items():
        sc_pairs = generate_scene(sc, rng)
        all_pairs.extend(sc_pairs)
        per_scene_counts[name] = len(sc_pairs)
        per_template_counts.update(p["template"] for p in sc_pairs)
        print(f"  {name:<25} {len(sc_pairs):>4} pairs")

    print(f"\nTotal: {len(all_pairs)} numerical pairs")
    print("\nPer-template counts:")
    for tid, c in sorted(per_template_counts.items()):
        print(f"  {tid:<3} {c:>5}")

    with out_path.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n=> wrote {len(all_pairs)} pairs to {out_path}")

    # Sample
    print("\nSamples (one per template):")
    seen = set()
    for p in all_pairs:
        if p["template"] in seen:
            continue
        seen.add(p["template"])
        print(f"  [{p['template']}] ({p['scene']:18s}) {p['question']!r} → {p['answer']}")
        if len(seen) == 8:
            break


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "vla3d_num.jsonl"
    main(out)
