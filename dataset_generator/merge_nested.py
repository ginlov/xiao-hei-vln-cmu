"""Fold ``vla3d_nested.jsonl`` back into ``vla3d_ref.jsonl`` and
``vla3d_num.jsonl`` by the runtime ``type`` field, then drop the
intermediate file.

Why this step exists
--------------------

The official challenge has three question types — ``numerical`` /
``object_reference`` / ``instruction_following`` — and the runtime
``classify_question()`` heuristic routes by question prefix only
(``"how many"``, ``"find"``, else). It does NOT know about "nested".

Our nested generator produces samples that combine an inner relation
(``on`` / ``above`` / ``below``) with an outer relation (``closest`` /
``farthest`` / ``between``). Each of those samples is *intrinsically* a
``numerical`` or ``object_reference`` pair — the "nested" label is a
generator-side taxonomy, not a runtime concept.

Keeping them in a third jsonl created two issues:

1. The training distribution per file was uniformly single-layer or
   uniformly nested, so any batch sampling that pulled from one file
   at a time saw a hard distribution shift between files.
2. Downstream pipelines (split_and_dump, check_question_types,
   evaluator) had to special-case a third file even though the
   runtime contract recognises two types.

After this merge:

- ``vla3d_ref.jsonl`` = original ``vla3d_ref`` rewrites + nested ref
- ``vla3d_num.jsonl`` = original ``vla3d_num`` templates + nested num
- ``vla3d_nested.jsonl`` is removed
- The ``source`` field is preserved verbatim, so anything that wants
  nested-only metrics can still filter by ``source == "vla3d_nested"``

The merge is deterministic given ``seed`` — same inputs + seed produce
byte-identical jsonl, matching the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DATASET_DIR = HERE.parent / "dataset"

REF_FILE = "vla3d_ref.jsonl"
NUM_FILE = "vla3d_num.jsonl"
NESTED_FILE = "vla3d_nested.jsonl"

NESTED_SOURCE_TAG = "vla3d_nested"


def _read_jsonl(path: Path) -> list[dict]:
    pairs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def _write_jsonl(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def merge(dataset_dir: Path, seed: int = 42) -> dict[str, int]:
    """Merge nested.jsonl into ref/num jsonl, shuffle, drop nested file.

    Returns a small summary dict for the caller / tests.
    """
    ref_path = dataset_dir / REF_FILE
    num_path = dataset_dir / NUM_FILE
    nested_path = dataset_dir / NESTED_FILE

    if not nested_path.exists():
        # Already merged or never generated — nothing to do.
        return {
            "ref_added": 0,
            "num_added": 0,
            "ref_total": len(_read_jsonl(ref_path)) if ref_path.exists() else 0,
            "num_total": len(_read_jsonl(num_path)) if num_path.exists() else 0,
            "nested_present": False,
        }

    ref_pairs = _read_jsonl(ref_path) if ref_path.exists() else []
    num_pairs = _read_jsonl(num_path) if num_path.exists() else []
    nested = _read_jsonl(nested_path)

    nested_ref = [r for r in nested if r.get("type") == "object_reference"]
    nested_num = [r for r in nested if r.get("type") == "numerical"]

    unexpected = len(nested) - len(nested_ref) - len(nested_num)
    if unexpected:
        # Surface but don't crash — runtime classifier is the source of
        # truth for which topic a sample would route to anyway.
        print(f"  [warn] {unexpected} nested pairs have a type other than "
              "object_reference/numerical; they will be dropped")

    full_ref = ref_pairs + nested_ref
    full_num = num_pairs + nested_num

    # Use one rng for both shuffles so the merge is deterministic given seed.
    # Independent shuffles via two rngs derived from the same seed would also
    # work; this is simpler and equally reproducible.
    rng = random.Random(seed)
    rng.shuffle(full_ref)
    rng.shuffle(full_num)

    _write_jsonl(ref_path, full_ref)
    _write_jsonl(num_path, full_num)
    nested_path.unlink()

    return {
        "ref_added": len(nested_ref),
        "num_added": len(nested_num),
        "ref_total": len(full_ref),
        "num_total": len(full_num),
        "nested_present": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                    help=f"Directory containing the jsonl files (default: {DEFAULT_DATASET_DIR})")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = merge(args.dataset_dir, seed=args.seed)
    if not summary["nested_present"]:
        print(f"  [skip] {args.dataset_dir / NESTED_FILE} not found — nothing to merge")
        return

    print("Merged nested pairs into ref/num:")
    print(f"  vla3d_ref.jsonl  +{summary['ref_added']}  →  {summary['ref_total']} total")
    print(f"  vla3d_num.jsonl  +{summary['num_added']}  →  {summary['num_total']} total")
    print("  vla3d_nested.jsonl removed")

    # Per-source breakdown for the user, since nested-origin samples are now
    # mixed in and the only way to count them is via the `source` field.
    for path in (args.dataset_dir / REF_FILE, args.dataset_dir / NUM_FILE):
        pairs = _read_jsonl(path)
        sources = Counter(p.get("source", "?") for p in pairs)
        print(f"  {path.name} by source: {dict(sources)}")


if __name__ == "__main__":
    main()
