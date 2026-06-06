# dataset_generator

Generates the training Q&A corpus for Team Xiao Hei's VLM from the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) Unity subset. The
generators live here; the data they emit lands in a sibling `dataset/`
directory (gitignored).

The 15 Unity scenes covered by VLA-3D are the same scenes the CMU VLN
Challenge uses for training, with object IDs aligned 1:1 to the
challenge's `object_list.txt`. We exploit that alignment to derive
12,190 grounded Q&A pairs across the challenge's two scoreable runtime
types (the third type, `instruction_following`, requires the official
forbidden-zone labels and is tracked separately):

| Source file            | Pairs  | Composition |
|------------------------|-------:|---|
| `vla3d_ref.jsonl`      | 12,000 | 6,000 single-layer `object_reference` rewrites + 6,000 nested-pattern ref pairs |
| `vla3d_num.jsonl`      |   190  | 150 numerical templates (N1–N8) + 40 nested-pattern num pairs |
| **Total**              | **12,190** | — |

### Matching the official phrasing distribution

Both halves are shaped to the official question set
(`../CMU-VLN-Challenge-2026/questions/questions.json`, 30 object_reference
+ 15 numerical graded items). `object_reference` (measured vs target):

| Feature                         | Official | Ours (12k) |
|---------------------------------|---------:|-----------:|
| compositional (≥2 relations)    |   57%    |   56%      |
| color modifier ("the red X")    |    7%    |    7%      |
| indefinite "a X" anchor         |   13%    |   13%      |
| omit-"Find" ("The X …")         |   10%    |   10%      |
| ordinal ("second closest")      |    0%    |    0%      |

`numerical` (measured vs target):

| Feature                         | Official | Ours |
|---------------------------------|---------:|-----:|
| `on` relation                   |   73%    |  67% |
| `near` relation                 |    7%    |   6% |
| color modifier                  |   13%    |  14% |
| compositional (≥2 relations)    |   20%    |  22% |
| "Count the number of …" phrasing|    7%    |   7% |
| pure totals / refusals          |    0%    |   8% |

The `on` share is capped by VLA-3D supply: only ~84 scene-unique "on"
anchors exist, so num is intentionally small (190) and faithful rather
than padded with abundant `near` counts (the old 591 was 36% pure totals
+ refusals and `near`-heavy). The official set has no pure totals
("How many X in the room?"), so the `N4` emitter is disabled; a token
`N5` refusal slice (answer 0) is kept for robustness.

The knobs live in `vla3d_ref_to_qa.py` (`TARGET_SINGLE`, `COLOR_QUOTA`,
`A_ANCHOR_QUOTA`, `OMIT_FIND_FRAC`), `vla3d_num_gen.py`
(`NUM_TEMPLATE_CAP`, `COUNT_PHRASING_FRAC`) and `vla3d_nested_gen.py`
(`TARGET_NESTED_REF`, `TARGET_NESTED_NUM`). Nested ref supply is lifted
past its old ~4.7k geometric ceiling by using `near` as an *inner*
relation ("the cup near the laptop closest to the window"), then
pattern-stratified-downsampled to 6,000 for diversity.

Single-layer ref pairs whose anchor is an **ill-formed definite
description** — a bare "the X" when the region holds more than one X and
VLA-3D supplied no color/size disambiguator (e.g. "the keyboard" in an
office with 6 keyboards) — are **softened to "a X"** rather than dropped:
the target stays geometrically unique, and the indefinite article no
longer wrongly presupposes a unique keyboard. This is the official set's
own style ("closest to a window"). Anchors with a disambiguator ("the
BLUE book") keep the definite article.

**Ordinal-ranked phrasings** ("second closest", "third farthest", …) are
dropped: the official set uses only superlatives and contains zero
ordinals, so these were out-of-distribution.

A `~10%` slice of object_reference questions drop the leading "Find"
("The red pillow closest to the sushi.") to mirror the 3/30 official
items phrased that way. This required teaching the runtime
`classify_question()` that a leading "The …" is object_reference (the
official instruction_following items always start with an action verb —
Go / First / Take — never "the").

Both `object_list` labels and the question are normalized to
VLA-3D's class vocabulary so the model can ground the question noun
(object_result.csv's `raw_label` of `tv` is rendered as the statement's
`television`); same-class distractors share the target's label so the
spatial relation, not a unique label, must disambiguate.

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
uv run python dataset_generator/check_question_types.py # auto-merges nested + sanity-checks
```

Nested samples are not a separate runtime type (`classify_question()`
only knows `numerical` / `object_reference` / `instruction_following`),
so the nested generator's output is an *intermediate file* that gets
folded back into the two type-aligned jsonl files. Both downstream
consumers — `check_question_types.py` and `split_and_dump.py` —
**auto-merge** `vla3d_nested.jsonl` into `vla3d_ref.jsonl` /
`vla3d_num.jsonl` at startup if they find it still on disk, so the user
never has to remember the merge step. `merge_nested.py` also exists as
an explicit CLI entry point for CI / batch-style invocation; it is
idempotent (no-op when nested.jsonl is already absent).

All steps are deterministic — seed `42` is fixed, so the same VLA-3D
source data + this code reproduces byte-identical jsonl. The
`dataset/` output is gitignored; regenerate or fetch it from a release
artifact when training.

## Visualize a sample (sanity-check generation)

Each jsonl row baked an `object_list` (all OBBs in the scene) plus a
`target` / `anchors` / `distractor_ids`. To verify these line up with
real geometry — i.e. target is on the right object, anchors actually
match the `closest to X` relation, heading isn't flipped — render any
row as a 3D OBB wireframe scene:

```bash
# One-time install (heavy ~400 MB Open3D wheel; opt-in only)
uv sync --extra viz

# Interactive (drag to rotate, scroll to zoom):
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42

# Random row:
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --random

# Single PNG for sharing in a PR/issue:
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42 --save out/

# Spot-check N samples per scene (headless batch):
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --sample-per-scene 20 --save out/

# Overlay the full scene point cloud (heavier, slower):
uv run python dataset_generator/visualize_sample.py \
    --jsonl dataset/vla3d_ref.jsonl --idx 42 --pointcloud
```

Color code:

- **red** — `target` (the answer object)
- **yellow** — `anchors` (objects referenced in the question)
- **cyan** — `distractor_ids` (same-class non-answers, ref samples only)
- **gray** — everything else in `object_list`, for spatial context

Wireframes are **OBB** (heading-aware), so a wrong `heading` shows up as
a visibly mis-rotated box. Note: our in-repo evaluator computes IoU as
**AABB** (`src/xiao_hei_vln/evaluator/metrics/object_reference.py`); the
official challenge does not publish its evaluation code, so the AABB/OBB
choice on the leaderboard side is not confirmed.

Saved PNGs come with a baked-in **header strip** (`#idx`, `scene`,
`type`, the question, and the answer) plus a **legend** in the
bottom-right corner showing only the role colors actually present in
that sample. Interactive windows show `scene` + question prefix in the
title bar; the full Q/A and role IDs are printed to terminal.

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
├── merge_nested.py         Fold nested.jsonl into ref/num jsonl by type (auto-called by consumers; CLI also)
├── regen.sh                One-shot driver: runs the three generators + sanity check (merge is implicit)
├── noise_augment.py        Perception-noise library (drop/swap/jitter, target-protected)
├── split_and_dump.py       Scene-level single split + K-Fold
├── check_question_types.py CI sanity: jsonl `type` ↔ runtime classify_question()
├── visualize_sample.py     Render one or many jsonl rows as 3D OBB wireframe (Open3D; opt-in viz extras)
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
