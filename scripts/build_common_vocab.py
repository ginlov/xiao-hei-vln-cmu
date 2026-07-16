"""Consolidate ONE deduped, synonym-merged class list from every scene's object_list.

The perception detector (YOLO-World v2) needs a class-list *prior* so the scene
graph accumulates objects during exploration even before any question arrives.
The prior baked into ``perception/vocab.py`` (``DEFAULT_PRIOR``) is arabic_room
-specific, so it under-detects the other 14 scenes.

This script walks every scene's ``object_list.txt``, takes the **union** of GT
labels, and consolidates it in four deterministic stages:

1. **Normalise** — lower-case, collapse internal whitespace.
2. **Synonym / typo map** — an explicit, auditable table (``SYNONYM_MAP``)
   folds typos (``celling`` → ``ceiling``) and true synonyms
   (``couch`` → ``sofa``, ``fridge`` → ``refrigerator``).
3. **Spacing/hyphen variants** — labels that are identical once spaces and
   hyphens are stripped collapse together (``ash tray`` ≡ ``ashtray``,
   ``door frame`` ≡ ``doorframe``, ``night stand`` ≡ ``nightstand``); the
   canonical form is the variant seen in the most scenes.
4. **Plural folding** — a plural collapses to its singular *only when the
   singular is also present* (``books`` → ``book``), so we never mangle a
   label that only ever appears plural (``clothes``, ``utensils``).

Outputs two files next to ``--out``:

* ``<out>``            — the final vocab, one label per line, sorted.
* ``<out>.audit.txt``  — every collapse group ``canonical  <=  raw, raw…``
                         so the merges are reviewable / adjustable.

Usage::

    uv run python scripts/build_common_vocab.py \\
        --scenes-dir /home/ubuntu/workspace/dataset/vla-3d/Unity \\
        --out perception/common_vocab.txt

Idempotent: re-running regenerates the same files.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

# Ground-truth placeholder for un-annotated objects — never a real class.
_DROP = {"unknown", ""}

# Stage 2: explicit typo fixes + true semantic synonyms. Left side is a
# normalised raw label; right side is the canonical to fold it into. Kept
# deliberately small and conservative — distinct object *types* (painting vs
# poster vs photo) are NOT merged; only spelling errors and genuine synonyms.
SYNONYM_MAP: dict[str, str] = {
    # --- typos --------------------------------------------------------
    "celling": "ceiling",
    "symbol decoraion": "symbol decoration",
    "refridgerator": "refrigerator",
    # --- synonyms -----------------------------------------------------
    "fridge": "refrigerator",
    "couch": "sofa",
    "night stand": "nightstand",
    "bedside table": "nightstand",
    "trash bin": "trash can",
    "cupboard": "cabinet",
    "ceiling lamp": "ceiling light",
}


def _normalise(label: str) -> str:
    """Lower-case and collapse internal whitespace (matches Vocabulary)."""
    return re.sub(r"\s+", " ", label.strip().lower())


def _squash_key(label: str) -> str:
    """Key that ignores spaces and hyphens: 'ash tray' -> 'ashtray'."""
    return label.replace(" ", "").replace("-", "")


def _depluralise(word: str) -> str:
    """Conservative singular of the last word (mirrors Vocabulary._depluralise)."""
    if len(word) < 4:
        return word
    if word.endswith(("ses", "xes", "zes", "shes", "ches")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _labels_from_object_list(path: Path) -> set[str]:
    """Extract the normalised quoted labels from one ``object_list.txt``."""
    labels: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r'"([^"]*)"', raw)
        if m is None:
            continue
        label = _normalise(m.group(1))
        if label not in _DROP:
            labels.add(label)
    return labels


def consolidate(
    scenes_dir: Path,
) -> tuple[list[str], dict[str, set[str]], dict[str, int]]:
    """Return (final sorted vocab, canonical -> {raw labels}, per-scene counts).

    ``doc_freq`` (scenes each raw label appears in) drives canonical choice
    for spacing/hyphen variants.
    """
    doc_freq: dict[str, int] = defaultdict(int)
    per_scene: dict[str, int] = {}
    for scene_dir in sorted(p for p in scenes_dir.iterdir() if p.is_dir()):
        obj_list = scene_dir / "object_list.txt"
        if not obj_list.is_file():
            per_scene[scene_dir.name] = 0
            continue
        labels = _labels_from_object_list(obj_list)
        per_scene[scene_dir.name] = len(labels)
        for label in labels:
            doc_freq[label] += 1

    # Track, for each final canonical label, which raw labels folded in.
    provenance: dict[str, set[str]] = defaultdict(set)

    # Stage 2: synonym / typo map.
    mapped_freq: dict[str, int] = defaultdict(int)
    raw_to_mapped: dict[str, str] = {}
    for raw, freq in doc_freq.items():
        mapped = SYNONYM_MAP.get(raw, raw)
        raw_to_mapped[raw] = mapped
        mapped_freq[mapped] += freq

    # Stage 3: collapse spacing/hyphen variants; canonical = highest doc_freq
    # (tie-break: the form containing a space, else the lexically first).
    groups: dict[str, list[str]] = defaultdict(list)
    for label in mapped_freq:
        groups[_squash_key(label)].append(label)
    squash_canon: dict[str, str] = {}
    for members in groups.values():
        canonical = sorted(
            members,
            key=lambda m: (-mapped_freq[m], " " not in m, m),
        )[0]
        for m in members:
            squash_canon[m] = canonical

    # Post-stage-3 frequencies (needed for the plural-fold presence test).
    after_squash_freq: dict[str, int] = defaultdict(int)
    for label, freq in mapped_freq.items():
        after_squash_freq[squash_canon[label]] += freq
    present = set(after_squash_freq)

    # Stage 4: plural -> singular only when the singular is also present.
    def plural_fold(label: str) -> str:
        words = label.split()
        if not words:
            return label
        singular_last = _depluralise(words[-1])
        candidate = " ".join(words[:-1] + [singular_last])
        return candidate if candidate != label and candidate in present else label

    # Compose the full raw -> final mapping and gather provenance.
    for raw in doc_freq:
        final = plural_fold(squash_canon[raw_to_mapped[raw]])
        provenance[final].add(raw)

    final_vocab = sorted(provenance)
    return final_vocab, provenance, per_scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("/home/ubuntu/workspace/dataset/vla-3d/Unity"),
        help="Directory with one subdir per scene, each holding object_list.txt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("perception/common_vocab.txt"),
        help="Where to write the consolidated one-label-per-line vocab",
    )
    args = ap.parse_args()

    vocab, provenance, per_scene = consolidate(args.scenes_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    audit_path = args.out.with_suffix(args.out.suffix + ".audit.txt")
    audit_lines: list[str] = []
    for canonical in vocab:
        raws = provenance[canonical]
        if raws != {canonical}:  # something actually folded in
            merged = ", ".join(sorted(raws))
            audit_lines.append(f"{canonical}  <=  {merged}")
    audit_path.write_text(
        "# collapse groups (canonical  <=  raw variants folded in)\n"
        + "\n".join(audit_lines) + "\n",
        encoding="utf-8",
    )

    print(f"scenes scanned    : {len(per_scene)}")
    for scene, n in per_scene.items():
        print(f"  {scene:<18} {n:>4} labels")
    print(f"final vocab       : {len(vocab)} labels  ({len(audit_lines)} collapse groups)")
    print(f"written           : {args.out}")
    print(f"audit             : {audit_path}")


if __name__ == "__main__":
    main()
