# dataset_generator

Generates the training Q&A corpus for Team Xiao Hei's VLM from the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) Unity subset. The
generators live here; the data they emit lands in a sibling `dataset/`
directory (gitignored).

The 15 Unity scenes covered by VLA-3D are the same scenes the CMU VLN
Challenge uses for training, with object IDs aligned 1:1 to the
challenge's `object_list.txt`. We exploit that alignment to derive
8,299 grounded Q&A pairs across the challenge's two scoreable runtime
types (the third type, `instruction_following`, requires the official
forbidden-zone labels and is tracked separately):

| Source file            | Pairs  | Composition |
|------------------------|-------:|---|
| `vla3d_ref.jsonl`      | 7,708  | 6,730 single-layer `object_reference` rewrites + 978 nested-pattern ref pairs |
| `vla3d_num.jsonl`      |   591  | 386 numerical templates (N1–N8) + 205 nested-pattern num pairs |
| **Total**              | **8,299** | — |

Nested patterns (inner relation × outer closest/farthest, plus
`between`) are *not* a runtime type — the official challenge has only
three categories, and runtime `classify_question()` routes by question
prefix only. They live inside the two type-aligned files above and are
identifiable via the `source` field (`source == "vla3d_nested"`), so
nested-only metrics are still possible at evaluation time.

All `object_list` lines and question routing match the runtime
contract in `src/xiao_hei_vln/messages/`:

- `object_list` line schema: `id cx cy cz lx ly lz heading label` —
  matches `src/xiao_hei_vln/dummy/data/object_list.txt`.
- `type` field on every pair agrees 100% with the runtime
  `classify_question()` heuristic (see `check_question_types.py`).

## One-time setup: download VLA-3D

The 2 GB raw data is **not committed** (it isn't in the VLA-3D GitHub
repo either — it lives on CMU AirLab's public bucket). Fetch it once
with the helper, which downloads `Unity.zip` and unpacks it into the
default location:

```bash
uv run python dataset_generator/download_vla3d.py
# → dataset_generator/vla-3d/Unity/<scene>/...   (the loader's default)
```

To put it elsewhere and point the scripts at it instead:

```bash
uv run python dataset_generator/download_vla3d.py --dest /data/vla-3d
export VLA3D_ROOT=/data/vla-3d/Unity
```

The pipeline expects each scene to be a directory under `$VLA3D_ROOT/`
containing at minimum:

```
<scene>_object_result.csv
<scene>_scene_graph.json
<scene>_referential_statements.json
```

## Generate the corpus

One shot:

```bash
./dataset_generator/regen.sh
```

Or step-by-step (equivalent):

```bash
uv run python dataset_generator/vla3d_ref_to_qa.py     # → dataset/vla3d_ref.jsonl     (~67 MB)
uv run python dataset_generator/vla3d_num_gen.py       # → dataset/vla3d_num.jsonl     (~3 MB)
uv run python dataset_generator/vla3d_nested_gen.py    # → dataset/vla3d_nested.jsonl  (~14 MB, intermediate)
uv run python dataset_generator/merge_nested.py        # folds nested → ref/num, drops nested.jsonl
uv run python dataset_generator/check_question_types.py # sanity check
```

The pipeline order matters: the three generators must run first, then
`merge_nested.py` reads `vla3d_nested.jsonl`, splits its rows by
`type`, appends them to `vla3d_ref.jsonl` / `vla3d_num.jsonl`,
deterministically shuffles each (so single-layer and nested pairs are
interleaved rather than block-segregated), and deletes the nested
file. The merge is idempotent — re-running it after nested.jsonl is
gone is a no-op.

All steps are deterministic — seed `42` is fixed, so the same VLA-3D
source data + this code reproduces byte-identical jsonl. The
`dataset/` output is gitignored; regenerate or fetch it from a release
artifact when training.

## Build train/val/test splits

```bash
# Single 10/3/2 scene split
uv run python dataset_generator/split_and_dump.py --seed 42

# Or 5-fold scene-level cross-validation
uv run python dataset_generator/split_and_dump.py --kfold 5 --seed 42
```

Splits are written under `dataset/splits/`.

Scene-level (Group K-Fold) splitting prevents leakage: every sample
from a given scene lands in the same split, so test accuracy reflects
true generalization to unseen scenes.

## Layout

```
dataset_generator/          # generator code (committed)
├── download_vla3d.py       Fetch + unpack the VLA-3D Unity subset (~2 GB) into vla-3d/
├── vla3d_loader.py         Unified VLA-3D scene loader (objects, scene_graph, ref statements)
├── vla3d_ref_to_qa.py      Rewrite VLA-3D ref statements → 6,730 "Find …" pairs
├── vla3d_num_gen.py        8 numerical templates (count, color-conditioned, refusal)
├── vla3d_nested_gen.py     Two-stage nested patterns: inner relation × outer closest/farthest
├── merge_nested.py         Fold nested.jsonl into ref/num jsonl by type, then delete it
├── regen.sh                One-shot driver: runs the four generators + merge + sanity check
├── noise_augment.py        Perception-noise library (drop/swap/jitter, target-protected)
├── split_and_dump.py       Scene-level single split + K-Fold
├── check_question_types.py CI sanity: jsonl `type` ↔ runtime classify_question()
└── vla-3d/Unity/           VLA-3D source data — INPUT (NOT committed; download per above)

dataset/                    # generated output (NOT committed)
├── vla3d_ref.jsonl         single-layer ref + nested ref (identifiable via `source`)
├── vla3d_num.jsonl         numerical templates + nested num (identifiable via `source`)
└── splits/                 train/val/test (or fold_*/) + manifest.json
```

## Integrating into a responder

Each jsonl row carries everything a fine-tuned LLM needs:

```jsonc
{
  "scene": "loft",
  "type": "object_reference",
  "question": "Find the lamp on the table closest to the potted bamboo.",
  "object_list": [
    "0 5.91 -1.46 0.31 0.40 0.23 0.05 -0.02 dvd",
    "5 -0.84 -3.55 4.06 0.71 0.17 0.65 -2.80 lamp",
    // ...
  ],
  "answer": {"object_id": 5, "label": "lamp"},
  "target": 5,
  "anchors": [22, 30]
}
```

At inference time, the responder feeds the perception module's
`object_list` plus the question to the model; the model emits an
`object_id` (or `count` / `waypoints`), and the responder looks the
bbox up in the perception list before publishing
`ObjectReferenceResponse`.

## Assumptions worth knowing

- **Full-scan scene context.** Every `object_list` baked into the
  jsonl is the *whole scene*, on the assumption that the deployed
  robot does a full sweep before answering. If the exploration
  strategy changes later (region-driven, etc.), `noise_augment.py`
  is the seed for a viewpoint sampler that perturbs this view at
  training time.
- **Instruction-following pairs are not generated yet.** Building
  those out is tracked separately.
