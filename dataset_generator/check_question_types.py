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
# After the nested merge step runs (either explicitly via merge_nested.py or
# implicitly via maybe_merge() below), nested samples live inside the two
# type-aligned files and are identifiable via the `source` field.
SOURCES = ("vla3d_ref.jsonl", "vla3d_num.jsonl")


def runtime_classifier(text: str) -> str:
    """1:1 mirror of `xiao_hei_vln.messages.question.classify_question`."""
    head = text.lstrip().lower()
    if head.startswith("how many") or head.startswith("count "):
        return "numerical"
    if head.startswith("find") or head.startswith("the "):
        return "object_reference"
    return "instruction_following"


def main() -> int:
    # Fold any leftover intermediate nested.jsonl into ref/num first so this
    # check is always against the full corpus, not partway through the
    # pipeline. No-op when nested.jsonl is already gone.
    from merge_nested import maybe_merge

    maybe_merge(DATASET_DIR)

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
