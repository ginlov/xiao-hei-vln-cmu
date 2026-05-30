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
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from vla3d_loader import load_all_vla_scenes, render_object_list, VLAScene

# How many ref statements to keep per relation, scene-wide pooled.
# Roughly matches the official Q&A distribution.
# Relations not listed are kept in full (rare ones we WANT more of).
PER_RELATION_CAP = {
    "closest": 1500,
    "farthest": 500,
    "near":    2000,
    "between": 1500,
    # on / above / below / beside / in / hanging_on: keep ALL (already rare)
}

# Statements containing any of these substrings get dropped — they're
# either weirdly phrased or duplicate of a kept variant.
DROP_SUBSTRINGS = (
    "in the middle of",   # "in the middle of X and Y" sounds awkward when imperative
    "most distant",       # "third most distant from" — keep "farthest" variant only
)

# Max words allowed in the *final* imperative question.
MAX_WORDS = 18


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


def build_pair(sc: VLAScene, region_id: int, raw: dict) -> dict | None:
    """Turn one VLA-3D ref-statement dict into our QA pair dict."""
    if not is_acceptable_statement(raw["statement"]):
        return None
    question = rewrite_imperative(raw["statement"])
    if question is None:
        return None
    if len(question.split()) > MAX_WORDS:
        return None

    return {
        "scene": sc.name,
        "type": "object_reference",
        "source": "vla3d_ref",
        "question": question,
        "object_list": render_object_list(sc),
        "answer": {"object_id": raw["target_id"], "label": raw["target_class"]},
        "target": raw["target_id"],
        "anchors": [a["id"] for a in raw["anchors"]],
        "distractor_ids": raw["distractor_ids"],
        "relation": raw["relation"],
        "relation_type": raw["relation_type"],
        "region_id": region_id,
        # keep original wording for paraphrase diversity if useful later
        "original_statement": raw["statement"],
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


def rebalance(pairs: list[dict], rng: random.Random) -> list[dict]:
    """Per-relation downsampling so distribution approaches official."""
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_rel[p["relation"]].append(p)

    kept: list[dict] = []
    for rel, lst in by_rel.items():
        cap = PER_RELATION_CAP.get(rel)
        if cap is None or len(lst) <= cap:
            kept.extend(lst)
        else:
            rng.shuffle(lst)
            kept.extend(lst[:cap])
    rng.shuffle(kept)
    return kept


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

    rebalanced = rebalance(all_pairs, rng)
    print(f"After rebalancing: {len(rebalanced)}")

    rel_counts = Counter(p["relation"] for p in rebalanced)
    print("\nRelation distribution AFTER rebalance:")
    for r, c in rel_counts.most_common():
        pct = 100 * c / len(rebalanced)
        print(f"  {r:<20} {c:>6}  ({pct:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in rebalanced:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n=> wrote {len(rebalanced)} pairs to {out_path}")

    # Show 6 random examples
    print("\nSamples:")
    for s in rng.sample(rebalanced, min(6, len(rebalanced))):
        print(f"  [{s['scene']}/r{s['region_id']}] {s['question']!r}")
        print(f"      target={s['target']} ({s['answer']['label']}) rel={s['relation']} distractors={len(s['distractor_ids'])}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "dataset" / "vla3d_ref.jsonl"
    main(out)
