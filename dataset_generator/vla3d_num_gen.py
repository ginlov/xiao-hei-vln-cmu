"""Derive numerical (counting) Q&A pairs from VLA-3D scene_graph + colors.

We start from the N1–N5 templates of our earlier Phase 1 and extend them
to N1–N8 here, powered by VLA-3D's richer data: 8 relation types in
`scene_graph.json` and 3 dominant colors per object in `object_result.csv`.

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

from phrasing import apply_count_phrasing
from vla3d_loader import VLAObject, VLAScene, load_all_vla_scenes, render_object_list

# Randomness is threaded explicitly through `rng = random.Random(seed)` in
# `main()` → `generate_scene` → `emit_refusal`. We deliberately do NOT call
# `random.seed(...)` at module scope: that would mutate the global RNG at
# import time, making byte-identical reproducibility depend on import order
# in any module that does `from vla3d_num_gen import ...`.

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
COUNT_PHRASING_FRAC = 0.09           # ~1/15 official numericals use "Count …"

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
              anchors: list[int], object_list: list[str],
              region_id: int | None) -> dict:
    """`region_id=None` marks the sample as scene-wide (N4 total count,
    N5 refusal) — `object_list` is the full scene. For anchor-based
    counts (N1/N2/N3/N6/N7/N8) the anchor's region is used and the
    `object_list` is filtered to that region."""
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
        "region_id": region_id,
    }


def emit_relation_count(sc: VLAScene, template_id: str, rel: str,
                        preposition: str, min_count: int = 2) -> list[dict]:
    """N1 (on), N2 (near), N3 (above), N7 (below), N8 (hanging on).

    For each singleton anchor in the scene, count objects of each label related
    via `rel`, emit a question if count >= `min_count`. The official set's
    "on" counts include answer 1 ("How many red pillows are on the sofa?"),
    so `on` is emitted with min_count=1; the others keep 2 to stay non-trivial.
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
            if n < min_count:
                continue       # not interesting below the threshold
            q = f"How many {pluralize(tgt_label)} are {preposition} the {anchor_label}?"
            ol = render_object_list(sc, region_ids={anchor.region_id})
            out.append(make_pair(sc.name, template_id, q, n, [anchor.id], ol,
                                 region_id=anchor.region_id))
    return out


def emit_total_count(sc: VLAScene) -> list[dict]:
    """N4: total count of each label in the scene. Scene-wide by design, so
    object_list stays unfiltered and region_id is None.

    Re-enabled (capped) to *balance* N5: N4 and N5 share the exact "...are there
    in the room?" phrasing, but N4 answers are >=1 and N5 answers are 0. Without
    N4 that phrasing only ever maps to 0, so the model could learn the shortcut
    "in the room => 0" instead of actually counting. Answers are kept in [2, 8]
    to stay near the official small-count range (no "15 books" outliers)."""
    out: list[dict] = []
    label_counts = Counter(o.raw_label for o in sc.objects
                           if usable_target_label(o.raw_label) and o.region_id >= 0)
    ol = render_object_list(sc)
    for label, n in label_counts.items():
        if n < 2 or n > 8:   # non-trivial, but within the official answer range
            continue
        q = f"How many {pluralize(label)} are there in the room?"
        out.append(make_pair(sc.name, "N4", q, n, [], ol, region_id=None))
    return out


def emit_refusal(sc: VLAScene, rng: random.Random) -> list[dict]:
    """N5: ask for a category we KNOW is not in the scene → answer 0.

    We mine plausible-sounding nouns from OTHER scenes that DON'T appear here.
    Scene-wide; object_list stays unfiltered and region_id is None.
    """
    present = {o.raw_label for o in sc.objects}
    candidates = [
        "guitar", "piano", "elevator", "telescope", "treadmill", "bathtub",
        "stove", "fridge", "dishwasher", "microwave", "blender",
        "kettle", "umbrella", "skateboard", "bicycle", "helmet",
        "trophy", "globe", "fan", "projector", "printer",
    ]
    available = [c for c in candidates if c not in present]
    rng.shuffle(available)
    ol = render_object_list(sc)
    out = []
    for label in available[:MAX_REFUSALS_PER_SCENE]:
        q = f"How many {pluralize(label)} are there in the room?"
        out.append(make_pair(sc.name, "N5", q, 0, [], ol, region_id=None))
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
        # Emit "How many <color> <X> are on the <anchor>?" — the official set
        # uses these even when the colour is redundant ("How many red pillows
        # are on the sofa?" with all pillows red), so we don't filter on
        # colour-redundancy; that also keeps colour supply near the 13% target.
        for (color, label), objs in by_color_label.items():
            n = len(objs)
            if n < 1:
                continue
            q = f"How many {color} {pluralize(label)} are on the {anchor_label}?"
            ol = render_object_list(sc, region_ids={anchor.region_id})
            out.append(make_pair(sc.name, "N6", q, n, [anchor.id], ol,
                                 region_id=anchor.region_id))
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

# Per-template per-scene caps, tuned to the official numerical distribution:
# `on` dominates (11/15), color ~13% (all "on"), `near` rare (1/15). The
# The official set has NO pure totals ("How many X in the room?") and NO
# refusals — both N4 and N5 are out-of-distribution. We keep a small, BALANCED
# slice of each: N5 (answer 0) gives refusal robustness, and N4 (answer >=1)
# exists only to counterweight it so the shared "...in the room?" phrasing
# doesn't collapse to a "=> 0" shortcut. N4 is capped slightly above N5 so 0 is
# a minority answer within that phrasing.
NUM_TEMPLATE_CAP = {
    "N1": 9999,   # on — keep all (the priority relation)
    "N6": 3,      # color-on — capped so colour ≈ 13% (these are also "on" Qs)
    "N3": 10,     # above
    "N2": 1,      # near (official: 1/15)
    "N7": 5,      # below
    "N8": 5,      # hanging on
    "N4": 2,      # total-in-room (answer >=1) — balances N5's zeros
    "N5": 1,      # refusal (answer 0) — token retention for robustness
}


def generate_scene(sc: VLAScene, rng: random.Random) -> list[dict]:
    bucket: dict[str, list[dict]] = defaultdict(list)

    bucket["N1"] = emit_relation_count(sc, "N1", "on", "on", min_count=1)
    bucket["N2"] = emit_relation_count(sc, "N2", "near", "near")
    bucket["N3"] = emit_relation_count(sc, "N3", "above", "above")
    # N4 re-enabled (capped) purely to balance N5 — see NUM_TEMPLATE_CAP. Both
    # share the "...in the room?" phrasing; N4 answers >=1, N5 answers 0, so the
    # phrasing no longer predicts 0 and the model must actually count.
    bucket["N4"] = emit_total_count(sc)
    bucket["N5"] = emit_refusal(sc, rng)
    bucket["N6"] = emit_color_on(sc)
    bucket["N7"] = emit_relation_count(sc, "N7", "below", "below")
    bucket["N8"] = emit_relation_count(sc, "N8", "hanging_on", "hanging on")

    out: list[dict] = []
    for tid, pairs in bucket.items():
        rng.shuffle(pairs)
        out.extend(pairs[: NUM_TEMPLATE_CAP.get(tid, MAX_PER_TEMPLATE_PER_SCENE)])
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

    # ~7% to "Count the number of …" phrasing (official has 1/15 such).
    n_count = apply_count_phrasing(all_pairs, rng, COUNT_PHRASING_FRAC)

    print(f"\nTotal: {len(all_pairs)} numerical pairs  "
          f"(count-phrasing applied to {n_count})")
    print("\nPer-template counts:")
    for tid, c in sorted(per_template_counts.items()):
        print(f"  {tid:<3} {c:>5}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "dataset" / "vla3d_num.jsonl"
    main(out)
