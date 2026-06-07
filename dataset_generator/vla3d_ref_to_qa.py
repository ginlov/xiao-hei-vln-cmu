"""Convert VLA-3D referential_statements.json → our object_reference JSONL.

Two jobs:
1.  Rebalance the heavily-skewed relation distribution so it roughly tracks
    the official Q&A (which is dominated by `on` and `closest`, not by VLA-3D's
    `farthest`).
2.  Rewrite descriptive form ("the X that is above Y")
              → imperative form ("Find the X above Y.")  to match
    the CMU-VLN-Challenge eval style.

Output schema (one JSON object per line):

    {
      "scene": "loft",
      "type": "object_reference",
      "source": "vla3d_ref",
      "question": "Find the book above the big table.",
      "answer": {"object_id": 72, "label": "book"},
      "target": 72,
      "anchors": [22],
      "distractor_ids": [58],
      "relation": "above",
      "relation_type": "binary",
      "region_id": 0
    }
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from phrasing import apply_omit_find
from vla3d_loader import VLAScene, load_all_vla_scenes, render_object_list

# Statements containing any of these substrings get dropped — they're
# either weirdly phrased or duplicate of a kept variant.
DROP_SUBSTRINGS = (
    "in the middle of",   # "in the middle of X and Y" sounds awkward when imperative
    "most distant",       # "third most distant from" — keep "farthest" variant only
)

# Max words allowed in the *final* imperative question.
MAX_WORDS = 18

# Ordinal-ranked phrasings ("second closest", "third farthest", ...). The
# official CMU-VLN question set (questions.json, 30 object_reference items)
# uses only superlatives (closest / farthest / near) and contains ZERO
# ordinals, so these 1,248 samples are out-of-distribution. All of them are
# an ordinal word immediately followed by a ranking word, so this is exact.
_ORDINAL_RANK_RE = re.compile(
    r"\b(second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(closest|farthest|furthest|nearest)\b",
    re.IGNORECASE,
)


def is_ordinal_ranked(stmt: str) -> bool:
    return _ORDINAL_RANK_RE.search(stmt) is not None


# Final single-layer sizing + feature quotas. The merged corpus is now smaller
# (~7.5k) after the relation-word reweight pulls `farthest`/`near` down toward
# the official shape; nested contributes ~0 color / ~0 "a"-anchors so those
# quotas live here. Color ~7% of the merged total; "a"-anchor is supply-capped
# (~540 after the redundancy gate) so its quota just takes all available.
TARGET_SINGLE = 3500
COLOR_QUOTA = 350        # ~7% of the ~5k merged corpus
A_ANCHOR_QUOTA = 1560    # supply-capped well below this; effectively "take all"
OMIT_FIND_FRAC = 0.10

# Per-relation caps on the single-layer pool, applied before bucketing. VLA-3D
# referential statements are `farthest`/`near`-heavy; the official set is
# `closest`-led with little `farthest`. Capping these two lets `closest` /
# `between` / `on`-family fill the rest. Relations not listed are uncapped.
REL_CAP = {"farthest": 300, "near": 380}

# A superlative ("the X closest/farthest/near to Y") is only well-posed if the
# target is *visibly* the closest/farthest among its same-class candidates.
# VLA-3D statements are uniquely-referring but can tie in the xy-plane the robot
# navigates in — e.g. two files stacked at the same (x,y) are equidistant from
# any anchor, so "the file nearest the plant" has no determinate answer (~14% of
# these samples are within 5 cm). Require the target to be THE xy-nearest /
# -farthest by this margin, matching the nested generator's CLOSEST_MARGIN_M.
SUPERLATIVE_MARGIN_M = 0.30
_SUPERLATIVE_RELS = {"closest", "near", "farthest"}


# Wall-occlusion gate (closest/near only). Straight-line xy distance through a
# wall doesn't reflect navigable proximity: "the cabinet closest to the fire
# alarm" picks a cabinet on the far side of a glass partition. `farthest` is
# exempt — the farthest object is naturally across the room / a wall.
_WALL_RELS = {"closest", "near"}
# Keep the crossing away from both segment ends so a wall-mounted anchor
# (picture / light switch / fire alarm, which sit *on* a wall, crossing near
# t=1) or an object backed against its own wall (t≈0) is not mistaken for a
# divider strictly *between* target and anchor.
_WALL_T_LO, _WALL_T_HI = 0.08, 0.92
_SCENE_WALLS: dict[str, list] = {}


def _scene_walls(sc: VLAScene) -> list:
    """Structural walls (incl. glass partitions — real navigation barriers).
    Filtered by height so flat 'wall decal' / 'wall lamp' fixtures don't count."""
    if sc.name not in _SCENE_WALLS:
        ws = []
        for o in sc.objects:
            lbl = o.raw_label.lower()
            if ("wall" in lbl
                    and not any(x in lbl for x in
                                ("decal", "lamp", "clock", "art", "paper",
                                 "mount", "sticker", "switch"))
                    and o.lz >= 1.2):
                ws.append(o)
        _SCENE_WALLS[sc.name] = ws
    return _SCENE_WALLS[sc.name]


def _seg_obb_interval(p0, p1, w) -> tuple[float, float] | None:
    """Liang-Barsky clip of segment p0->p1 to wall w's xy OBB. Returns the
    crossing parameter interval (t0, t1) along the segment, or None."""
    ca, sa = math.cos(-w.heading), math.sin(-w.heading)
    hx, hy = w.lx / 2, w.ly / 2

    def loc(p):
        dx, dy = p[0] - w.x, p[1] - w.y
        return (ca * dx - sa * dy, sa * dx + ca * dy)

    a = loc(p0)
    b = loc(p1)
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for pdir, q in ((-dx, a[0] + hx), (dx, hx - a[0]),
                    (-dy, a[1] + hy), (dy, hy - a[1])):
        if abs(pdir) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / pdir
            if pdir < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
    return (t0, t1) if t0 < t1 else None


def wall_between_ok(sc: VLAScene, raw: dict) -> bool:
    """False if a closest/near target is separated from its anchor by a wall
    (the xy segment crosses a structural wall in its middle portion)."""
    if raw["relation"] not in _WALL_RELS:
        return True
    anchors = raw["anchors"]
    if not anchors:
        return True
    t = sc.by_id.get(raw["target_id"])
    a = sc.by_id.get(anchors[0]["id"])
    if t is None or a is None:
        return True
    for w in _scene_walls(sc):
        if w.region_id != t.region_id:
            continue
        iv = _seg_obb_interval((t.x, t.y), (a.x, a.y), w)
        if iv and iv[0] < _WALL_T_HI and iv[1] > _WALL_T_LO:
            return False
    return True


def superlative_margin_ok(sc: VLAScene, raw: dict) -> bool:
    """For closest/near/farthest, the target must be the unique xy-nearest (or
    -farthest) candidate by SUPERLATIVE_MARGIN_M. Non-superlative relations and
    cases with <2 candidates or a missing anchor pass through unchanged."""
    if raw["relation"] not in _SUPERLATIVE_RELS:
        return True
    anchors = raw["anchors"]
    if not anchors:
        return True
    a = sc.by_id.get(anchors[0]["id"])
    cands = [raw["target_id"], *raw["distractor_ids"]]
    objs = [sc.by_id.get(c) for c in cands]
    if a is None or len(objs) < 2 or any(o is None for o in objs):
        return True
    ranked = sorted(
        ((math.hypot(o.x - a.x, o.y - a.y), c)
         for o, c in zip(objs, cands, strict=True)),
        key=lambda t: t[0],
    )
    if raw["relation"] == "farthest":
        ranked.reverse()
    best_id = ranked[0][1]
    margin = abs(ranked[1][0] - ranked[0][0])
    return best_id == raw["target_id"] and margin >= SUPERLATIVE_MARGIN_M

# Real colour words only. Deliberately excludes "dark"/"light": those match
# object labels like "light switch" / "spot light", not colour modifiers.
_COLOR_RE = re.compile(
    r"\b(red|blue|green|black|white|orange|yellow|brown|gray|grey|purple|"
    r"pink|silver|gold|golden|beige|tan|aqua|maroon|navy|teal|violet)\b",
    re.IGNORECASE,
)


def has_color(question: str) -> bool:
    return _COLOR_RE.search(question) is not None


def _indefinite_article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _looks_plural(label: str) -> bool:
    """Rough plural test so we don't emit "a books". Head noun = last word."""
    head = label.split()[-1].lower() if label.split() else label.lower()
    return head.endswith("s") and not head.endswith(("ss", "us", "is"))


def soften_anchor(question: str, anchor_class: str) -> str:
    """Replace the first "the {anchor_class}" with "a/an {anchor_class}" so an
    anchor with several same-class instances reads as an indefinite reference
    (the official set's style, e.g. "closest to a window") instead of an
    ill-formed definite "the window"."""
    art = _indefinite_article(anchor_class)
    # Word-boundary match so "the box" doesn't fire inside "the boxes".
    pat = re.compile(r"\bthe " + re.escape(anchor_class) + r"\b")
    return pat.sub(f"{art} {anchor_class}", question, count=1)


def rewrite_imperative(stmt: str) -> str | None:
    """'the X that is above the Y' -> 'Find the X above the Y.'

    Returns None if we cannot confidently rewrite — caller drops the sample.
    """
    s = stmt.strip()
    if not s:
        return None
    s = s[0].lower() + s[1:]                # normalize leading capital

    # Strip leading article if present, then prepend imperative.
    if s.startswith("the "):
        body = s
    elif s.startswith("a "):
        body = "the " + s[2:]
    else:
        body = "the " + s

    # Remove "that is" / "which is" — they're verbose; "Find the X above Y" reads cleaner.
    body = re.sub(r"\bthat is\b", "", body)
    body = re.sub(r"\bwhich is\b", "", body)

    # Collapse repeated whitespace
    body = re.sub(r"\s+", " ", body).strip()

    if not body:
        return None
    # Capitalize "Find " and ensure trailing period
    out = "Find " + body
    if not out.endswith("."):
        out += "."
    return out


def is_acceptable_statement(stmt: str) -> bool:
    if any(sub in stmt for sub in DROP_SUBSTRINGS):
        return False
    return True


# Cache per (scene, region): raw_label counts + id->raw_label, used to decide
# whether an anchor phrased as a bare "the X" is ill-formed (X not unique).
_REGION_RAW_COUNT: dict[tuple[str, int], Counter] = {}
_REGION_ID2RAW: dict[tuple[str, int], dict[int, str]] = {}

# Tally of why pairs were dropped, printed at the end of main().
DROP_COUNTS: Counter = Counter()


def _region_info(sc: VLAScene, region_id: int) -> tuple[Counter, dict[int, str]]:
    key = (sc.name, region_id)
    if key not in _REGION_RAW_COUNT:
        cnt: Counter = Counter()
        id2raw: dict[int, str] = {}
        for o in sc.objects:
            if o.region_id == region_id:
                cnt[o.raw_label] += 1
                id2raw[o.id] = o.raw_label
        _REGION_RAW_COUNT[key] = cnt
        _REGION_ID2RAW[key] = id2raw
    return _REGION_RAW_COUNT[key], _REGION_ID2RAW[key]


def ill_formed_anchor_classes(sc: VLAScene, region_id: int, raw: dict) -> list[str]:
    """Statement-vocab classes of anchors phrased as a bare "the X" while >1
    object of that class sits in the region (e.g. "the keyboard" when the
    office has 6). These are softened to "a X" rather than dropped.

    Anchors carrying a color/size disambiguator ("the BLUE book") are not
    returned: the modifier already makes the reference unique.
    """
    cnt, id2raw = _region_info(sc, region_id)
    out: list[str] = []
    for a in raw["anchors"]:
        rawlbl = id2raw.get(a["id"])
        if rawlbl is None:
            continue
        # Anchor sharing the target's class is phrased "the other X" by VLA-3D
        # (well-formed when there are exactly two); never soften it — and
        # softening would hit the target's leading "the X" anyway.
        if a["class"] == raw["target_class"]:
            continue
        # Plural labels ("books") can't take "a" — leave them as "the books".
        if _looks_plural(a["class"]):
            continue
        has_modifier = bool(a["color_used"] or a["size_used"])
        if not has_modifier and cnt[rawlbl] > 1:
            out.append(a["class"])
    return out


def build_pair(sc: VLAScene, region_id: int, raw: dict) -> dict | None:
    """Turn one VLA-3D ref-statement dict into our QA pair dict."""
    if not is_acceptable_statement(raw["statement"]):
        DROP_COUNTS["unacceptable_statement"] += 1
        return None
    # Drop ordinal-ranked phrasings ("second/third closest") — the official
    # question set has none; these are out-of-distribution.
    if is_ordinal_ranked(raw["statement"]):
        DROP_COUNTS["ordinal_ranked"] += 1
        return None
    # Redundancy gate: if the target's class is already unique within the
    # object_list this sample carries, no spatial/attribute constraint is
    # needed to identify it ("Find the dvd beside the chair" when only one dvd
    # is in view). The official set never does this — 29/30 of its
    # object_reference targets are non-unique, so the relation is doing real
    # disambiguation. Single-layer object_list is region-filtered (below), so
    # the operative scope is the region: require >=2 same-class instances there.
    tgt_obj = sc.by_id.get(raw["target_id"])
    if tgt_obj is not None:
        region_cnt, _ = _region_info(sc, region_id)
        if region_cnt[tgt_obj.raw_label] < 2:
            DROP_COUNTS["target_unique_in_view"] += 1
            return None
    # Superlative margin: drop "closest/farthest/near" questions where the
    # target isn't the unique xy-nearest/-farthest candidate by a visible margin
    # (stacked / tied objects make the answer indeterminate).
    if not superlative_margin_ok(sc, raw):
        DROP_COUNTS["superlative_tie"] += 1
        return None
    # Wall between target and anchor (closest/near only) — straight-line
    # distance through a wall isn't navigable proximity.
    if not wall_between_ok(sc, raw):
        DROP_COUNTS["wall_between"] += 1
        return None
    question = rewrite_imperative(raw["statement"])
    if question is None:
        DROP_COUNTS["rewrite_failed"] += 1
        return None
    if len(question.split()) > MAX_WORDS:
        DROP_COUNTS["too_long"] += 1
        return None
    # Soften ill-formed definite anchors ("the keyboard" with 6 keyboards) to
    # "a keyboard". The target stays geometrically unique, and the indefinite
    # article no longer (wrongly) presupposes a unique keyboard — this is the
    # official set's "closest to a window" style. If an ill-formed anchor's
    # phrase isn't found in the rewritten question, drop (can't safely soften).
    a_anchor = False
    for cls in ill_formed_anchor_classes(sc, region_id, raw):
        softened = soften_anchor(question, cls)
        if softened != question:
            question = softened
            a_anchor = True
        else:
            DROP_COUNTS["ill_formed_unfixable"] += 1
            return None

    # Align object_list labels with the statement's class vocabulary so the
    # model can ground the question noun. VLA-3D's referential statements use
    # a normalized class ('television') that often differs from
    # object_result.csv's raw_label ('tv') — 46% of single-layer ref samples
    # had object_list[target] != answer.label before this fix. Relabel:
    #   target + same-class distractors -> target_class  (relation, not label,
    #     must disambiguate — prevents trivial label-matching shortcuts)
    #   each anchor -> its statement class
    label_overrides: dict[int, str] = {raw["target_id"]: raw["target_class"]}
    for d in raw["distractor_ids"]:
        label_overrides[d] = raw["target_class"]
    for a in raw["anchors"]:
        label_overrides[a["id"]] = a["class"]

    return {
        "scene": sc.name,
        "type": "object_reference",
        "source": "vla3d_ref",
        "question": question,
        # Filter object_list to the target's region: VLA-3D ref statements
        # carry per-region disambiguators (e.g. "the BIG table" is unique in
        # its region but not scene-wide). Audit confirmed 0/6730 single-layer
        # ref samples have an anchor outside the target's region, so this
        # is lossless. Shrinks the prompt and removes the region-ambiguity
        # bug surfaced by visualize_sample.py.
        "object_list": render_object_list(sc, region_ids={region_id},
                                          label_overrides=label_overrides),
        "answer": {"object_id": raw["target_id"], "label": raw["target_class"]},
        "target": raw["target_id"],
        "anchors": [a["id"] for a in raw["anchors"]],
        "distractor_ids": raw["distractor_ids"],
        "relation": raw["relation"],
        "relation_type": raw["relation_type"],
        "region_id": region_id,
        # keep original wording for paraphrase diversity if useful later
        "original_statement": raw["statement"],
        # transient flags for the quota sampler; stripped before writing.
        "_a_anchor": a_anchor,
        "_has_color": has_color(question),
    }


def convert_scene(sc: VLAScene, rng: random.Random) -> list[dict]:
    """All eligible pairs for one scene, before global rebalancing."""
    pairs: list[dict] = []
    for region_id, stmts in sc.ref_statements.items():
        if region_id < 0:
            continue                       # skip unassigned
        for raw in stmts:
            p = build_pair(sc, region_id, raw)
            if p is not None:
                pairs.append(p)
    rng.shuffle(pairs)
    return pairs


def sample_single_layer(pairs: list[dict], rng: random.Random) -> list[dict]:
    """Sample TARGET_SINGLE pairs while hitting the color / "a"-anchor quotas.

    Three disjoint buckets so the feature shares are controlled directly:
      color      → up to COLOR_QUOTA
      a-anchor   → up to A_ANCHOR_QUOTA   (non-color)
      plain      → fills the remainder
    Backfills from leftovers if any bucket is short.
    """
    rng.shuffle(pairs)
    # Relation reweight: cap the over-supplied `farthest` / `near` statements so
    # `closest` / `between` / `on`-family carry more of the single-layer share,
    # tracking the official `closest`-led shape (see REL_CAP).
    seen: Counter = Counter()
    capped: list[dict] = []
    for p in pairs:
        rel = p["relation"]
        if rel in REL_CAP and seen[rel] >= REL_CAP[rel]:
            continue
        seen[rel] += 1
        capped.append(p)
    pairs = capped
    # a-anchor is the priority bucket (its supply is scarcer than color's), so
    # an "a X" sample that also contains a color word still counts as a-anchor.
    a_anchor = [p for p in pairs if p["_a_anchor"]]
    color = [p for p in pairs if p["_has_color"] and not p["_a_anchor"]]
    plain = [p for p in pairs if not p["_has_color"] and not p["_a_anchor"]]

    kept = a_anchor[:A_ANCHOR_QUOTA] + color[:COLOR_QUOTA]
    kept += plain[: max(0, TARGET_SINGLE - len(kept))]
    if len(kept) < TARGET_SINGLE:                      # backfill
        leftover = a_anchor[A_ANCHOR_QUOTA:] + color[COLOR_QUOTA:] + plain[TARGET_SINGLE:]
        rng.shuffle(leftover)
        kept += leftover[: TARGET_SINGLE - len(kept)]
    rng.shuffle(kept)
    return kept[:TARGET_SINGLE]


def main(out_path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    scenes = load_all_vla_scenes()
    print(f"Loaded {len(scenes)} VLA-3D scenes\n")

    all_pairs: list[dict] = []
    for name, sc in scenes.items():
        sc_pairs = convert_scene(sc, rng)
        print(f"  {name:<25} eligible_pairs={len(sc_pairs)}")
        all_pairs.extend(sc_pairs)
    print(f"\nTotal eligible (post-filter, pre-rebalance): {len(all_pairs)}")
    if DROP_COUNTS:
        print("Dropped during build_pair:")
        for reason, n in DROP_COUNTS.most_common():
            print(f"  {reason:<24} {n}")

    kept = sample_single_layer(all_pairs, rng)
    n_color = sum(p["_has_color"] for p in kept)
    n_a = sum(p["_a_anchor"] for p in kept)
    n_omit = apply_omit_find(kept, rng, OMIT_FIND_FRAC)
    print(f"\nSampled single-layer: {len(kept)}  "
          f"(color={n_color}, a-anchor={n_a}, omit-Find={n_omit})")

    rel_counts = Counter(p["relation"] for p in kept)
    print("Relation distribution:")
    for r, c in rel_counts.most_common():
        print(f"  {r:<20} {c:>6}  ({100 * c / len(kept):.1f}%)")

    # Strip transient sampler flags before writing.
    for p in kept:
        p.pop("_a_anchor", None)
        p.pop("_has_color", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n=> wrote {len(kept)} pairs to {out_path}")

    # Show 6 random examples
    print("\nSamples:")
    for s in rng.sample(kept, min(6, len(kept))):
        print(f"  [{s['scene']}/r{s['region_id']}] {s['question']!r}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "dataset" / "vla3d_ref.jsonl"
    main(out)
