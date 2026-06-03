"""Sanity-check: do our `type` labels agree with the runtime question router?

The xiao-hei-vln-cmu runtime decides which ROS topic to publish on by running
this exact heuristic on the question string:

    "how many ..." -> numerical          -> /numerical_response
    "find ..."     -> object_reference   -> /selected_object_marker
    else           -> instruction_following -> /way_point_with_heading

Our jsonl pairs carry a `type` field set at generation time. If the two ever
disagree, the model would be trained on one route but answers would be
published on another. This script asserts they agree.

Exit 0 on full agreement, 1 otherwise. Run after regenerating any jsonl.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DATASET_DIR = HERE.parent / "dataset"
# After `merge_nested.py` runs, nested samples live inside vla3d_ref.jsonl
# and vla3d_num.jsonl (identified via the `source` field), so we only check
# the two type-aligned files here.
SOURCES = ("vla3d_ref.jsonl", "vla3d_num.jsonl")
NESTED_INTERMEDIATE = "vla3d_nested.jsonl"


def runtime_classifier(text: str) -> str:
    """1:1 mirror of `xiao_hei_vln.messages.question.classify_question`."""
    head = text.lstrip().lower()
    if head.startswith("how many"):
        return "numerical"
    if head.startswith("find"):
        return "object_reference"
    return "instruction_following"


def main() -> int:
    nested_path = DATASET_DIR / NESTED_INTERMEDIATE
    if nested_path.exists():
        print(
            f"[error] Found {NESTED_INTERMEDIATE} in {DATASET_DIR}. "
            "Run merge_nested.py before check_question_types.py."
        )
        return 1

    total = 0
    mismatches = 0
    loaded = 0
    buckets: Counter[tuple[str, str]] = Counter()
    examples: list[tuple[str, str, str]] = []

    for fn in SOURCES:
        path = DATASET_DIR / fn
        if not path.exists():
            print(f"  [skip] {fn} (not found)")
            continue
        loaded += 1
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            ours = p["type"]
            theirs = runtime_classifier(p["question"])
            buckets[(ours, theirs)] += 1
            total += 1
            if ours != theirs:
                mismatches += 1
                if len(examples) < 10:
                    examples.append((ours, theirs, p["question"]))

    print(f"Checked {total} pairs across {loaded} files")
    print(f"Mismatches: {mismatches}")
    print("\n(ours, runtime) bucket counts:")
    for (ours, theirs), n in sorted(buckets.items()):
        marker = " " if ours == theirs else " <-- MISMATCH"
        print(f"  {ours:<22} {theirs:<22} {n:>5}{marker}")

    if examples:
        print("\nFirst mismatches:")
        for o, t, q in examples:
            print(f"  ours={o:<22} runtime={t:<22} {q!r}")

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
