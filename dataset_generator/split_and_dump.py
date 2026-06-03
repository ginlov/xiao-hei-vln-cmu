"""Scene-level train/val/test split + dump.

Two modes:

1. Single split (default):
       python split_and_dump.py --seed 42
   →   splits/train.jsonl  splits/val.jsonl  splits/test.jsonl
       (10 / 3 / 2 scenes for our 15-scene dataset)

2. K-fold CV mode:
       python split_and_dump.py --kfold 5 --seed 42
   →   splits/fold_0/train.jsonl  splits/fold_0/val.jsonl  splits/fold_0/test.jsonl
       splits/fold_1/...
       splits/fold_2/...
       splits/fold_3/...
       splits/fold_4/...

Why scene-level? Samples generated from the same scene share the same object
inventory, the same anchors, and the same geometric layout. If they were split
randomly across train/test, the model would essentially see every scene's
layout during training and the test accuracy would be optimistically biased.
Scene-level splitting (a.k.a. Group K-Fold) forces the model to generalize to
NEW scenes — which is what the CMU VLN Challenge actually evaluates.

Inputs (any subset, present in this directory):
  - vla3d_ref.jsonl    (includes nested ref pairs after merge_nested.py)
  - vla3d_num.jsonl    (includes nested num pairs after merge_nested.py)
  - phase1_raw.jsonl   (kept as fallback / style comparison)

Output: each fold contains all source pairs partitioned by the pair's `scene`
field. A summary `manifest.json` records which scenes landed in which split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATASET_DIR = HERE.parent / "dataset"

DEFAULT_SOURCES = ("vla3d_ref.jsonl", "vla3d_num.jsonl")
OPTIONAL_SOURCES = ("phase1_raw.jsonl", "vla3d_hard.jsonl")

# For 15 scenes; only used in single-split mode
SINGLE_SPLIT = {"train": 10, "val": 3, "test": 2}


def load_pairs(paths: list[Path]) -> list[dict]:
    pairs: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"  [skip] {p.name} (not found)")
            continue
        with p.open() as f:
            n = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pairs.append(json.loads(line))
                n += 1
        print(f"  [load] {p.name}: {n} pairs")
    return pairs


def by_scene(pairs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        out[p["scene"]].append(p)
    return out


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def summarize(pairs: list[dict]) -> dict:
    by_type = Counter(p.get("type", "?") for p in pairs)
    by_source = Counter(p.get("source", "?") for p in pairs)
    return {
        "n": len(pairs),
        "by_type": dict(by_type),
        "by_source": dict(by_source),
        "scenes": sorted({p["scene"] for p in pairs}),
    }


def single_split(scenes: list[str], rng: random.Random) -> dict[str, list[str]]:
    shuffled = scenes.copy()
    rng.shuffle(shuffled)
    n_train, n_val, n_test = SINGLE_SPLIT["train"], SINGLE_SPLIT["val"], SINGLE_SPLIT["test"]
    if len(shuffled) < n_train + n_val + n_test:
        # adapt proportionally if we don't have exactly 15
        total = len(shuffled)
        n_test = max(1, total * 2 // 15)
        n_val = max(1, total * 3 // 15)
        n_train = total - n_test - n_val
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def kfold_splits(scenes: list[str], k: int, rng: random.Random) -> list[dict[str, list[str]]]:
    """Group K-Fold: rotate which `~len/k` scenes form the test set each fold.

    Each fold reserves a small val set (disjoint from its test) so we can
    early-stop without leaking from test. The val set is re-randomized per
    fold using a per-fold seed derived from `rng`, so val rotates too —
    otherwise the first scenes after the shuffle would always be val.
    """
    shuffled = scenes.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    fold_size = n // k
    folds = []
    for i in range(k):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else n
        test = shuffled[test_start:test_end]
        remaining = shuffled[:test_start] + shuffled[test_end:]

        # Shuffle remaining with a per-fold seed so val differs across folds
        fold_rng = random.Random(rng.random())
        fold_rng.shuffle(remaining)

        n_val = max(1, len(remaining) // 5)
        val = remaining[:n_val]
        train = remaining[n_val:]
        folds.append({"train": train, "val": val, "test": test})
    return folds


def dump_split(out_dir: Path, split_scenes: dict[str, list[str]],
               by_scene_pairs: dict[str, list[dict]]) -> dict:
    manifest = {}
    for split_name, scene_list in split_scenes.items():
        pairs = [p for s in scene_list for p in by_scene_pairs.get(s, [])]
        write_jsonl(out_dir / f"{split_name}.jsonl", pairs)
        manifest[split_name] = {
            "scenes": scene_list,
            **summarize(pairs),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def print_manifest(label: str, manifest: dict) -> None:
    print(f"\n[{label}]")
    for split, info in manifest.items():
        print(f"  {split:>5}: {info['n']:>5} pairs   scenes={info['scenes']}")
        if info["by_type"]:
            tparts = " ".join(f"{t}={c}" for t, c in info["by_type"].items())
            print(f"         types: {tparts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=0,
                    help="If >0, produce K-fold splits instead of a single one")
    ap.add_argument("--out", type=Path, default=DATASET_DIR / "splits")
    ap.add_argument("--include-legacy", action="store_true",
                    help="Also include phase1_raw.jsonl (template-based fallback)")
    args = ap.parse_args()

    # Fold any leftover intermediate nested.jsonl into ref/num first so the
    # split sees the full corpus instead of silently under-counting by ~1.2k
    # nested pairs. No-op when nested.jsonl is already gone.
    from merge_nested import maybe_merge

    maybe_merge(DATASET_DIR, seed=args.seed)

    sources = [DATASET_DIR / s for s in DEFAULT_SOURCES]
    if args.include_legacy:
        sources += [DATASET_DIR / s for s in OPTIONAL_SOURCES]

    print("Loading source jsonl files:")
    pairs = load_pairs(sources)
    print(f"Total pairs loaded: {len(pairs)}")

    grouped = by_scene(pairs)
    scenes = sorted(grouped.keys())
    print(f"Distinct scenes: {len(scenes)}  → {scenes}")

    rng = random.Random(args.seed)
    out = args.out

    if args.kfold > 0:
        print(f"\nGenerating {args.kfold}-fold splits  (seed={args.seed})")
        folds = kfold_splits(scenes, args.kfold, rng)
        for i, fold in enumerate(folds):
            manifest = dump_split(out / f"fold_{i}", fold, grouped)
            print_manifest(f"fold_{i}", manifest)
        # top-level manifest summarizing all folds
        (out / "kfold_manifest.json").write_text(
            json.dumps({"k": args.kfold, "seed": args.seed,
                        "folds": [{"train": f["train"], "val": f["val"], "test": f["test"]}
                                  for f in folds]}, indent=2)
        )
        print(f"\n=> wrote {args.kfold} folds under {out}/")
    else:
        print(f"\nGenerating single 10/3/2 split  (seed={args.seed})")
        split = single_split(scenes, rng)
        manifest = dump_split(out, split, grouped)
        print_manifest("single-split", manifest)
        print(f"\n=> wrote train/val/test.jsonl under {out}/")


if __name__ == "__main__":
    main()
