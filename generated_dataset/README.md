# generated_dataset

Generates the training Q&A corpus for Team Xiao Hei's VLM from the
[VLA-3D](https://github.com/HaochenZ11/VLA-3D) Unity subset.

The 15 Unity scenes covered by VLA-3D are the same scenes the CMU VLN
Challenge uses for training, with object IDs aligned 1:1 to the
challenge's `object_list.txt`. We exploit that alignment to derive
8,299 grounded Q&A pairs across the challenge's three question types:

| Source file            | Pairs  | Type                                  |
|------------------------|-------:|---------------------------------------|
| `vla3d_ref.jsonl`      | 6,730  | object_reference                      |
| `vla3d_num.jsonl`      |   386  | numerical                             |
| `vla3d_nested.jsonl`   | 1,183  | mixed (978 object_reference, 205 numerical) — nested |
| **Total**              | **8,299** | — |

All `object_list` lines and question routing match the runtime
contract in `src/xiao_hei_vln/messages/`:

- `object_list` line schema: `id cx cy cz lx ly lz heading label` —
  matches `src/xiao_hei_vln/dummy/data/object_list.txt`.
- `type` field on every pair agrees 100% with the runtime
  `classify_question()` heuristic (see `check_question_types.py`).

## One-time setup: download VLA-3D

The 2 GB raw data is **not committed**. Download it once:

```bash
# Option A — from the VLA-3D GitHub repo
git clone https://github.com/HaochenZ11/VLA-3D /tmp/vla-3d
# The 15 Unity scenes live under /tmp/vla-3d/Unity/

# Either move/symlink it into the default location ...
mv /tmp/vla-3d/Unity generated_dataset/vla-3d/Unity

# ... or point the scripts at it via env var:
export VLA3D_ROOT=/tmp/vla-3d/Unity
```

The pipeline expects each scene to be a directory under `$VLA3D_ROOT/`
containing at minimum:

```
<scene>_object_result.csv
<scene>_scene_graph.json
<scene>_referential_statements.json
```

## Generate the corpus

```bash
uv run python generated_dataset/vla3d_ref_to_qa.py     # → vla3d_ref.jsonl     (~67 MB)
uv run python generated_dataset/vla3d_num_gen.py       # → vla3d_num.jsonl     (~3 MB)
uv run python generated_dataset/vla3d_nested_gen.py    # → vla3d_nested.jsonl  (~14 MB)
uv run python generated_dataset/check_question_types.py # sanity check
```

All three generators are deterministic — seed `42` is fixed, so the
same VLA-3D source data + this code reproduces byte-identical jsonl.
The generated files are gitignored; regenerate or fetch them from a
release artifact when training.

## Build train/val/test splits

```bash
# Single 10/3/2 scene split
uv run python generated_dataset/split_and_dump.py --seed 42

# Or 5-fold scene-level cross-validation
uv run python generated_dataset/split_and_dump.py --kfold 5 --seed 42
```

Scene-level (Group K-Fold) splitting prevents leakage: every sample
from a given scene lands in the same split, so test accuracy reflects
true generalization to unseen scenes.

## Layout

```
generated_dataset/
├── vla3d_loader.py         Unified VLA-3D scene loader (objects, scene_graph, ref statements)
├── vla3d_ref_to_qa.py      Rewrite VLA-3D ref statements → 6,730 "Find …" pairs
├── vla3d_num_gen.py        8 numerical templates (count, color-conditioned, refusal)
├── vla3d_nested_gen.py     Two-stage nested patterns: inner relation × outer closest/farthest
├── noise_augment.py        Perception-noise library (drop/swap/jitter, target-protected)
├── split_and_dump.py       Scene-level single split + K-Fold
├── check_question_types.py CI sanity: jsonl `type` ↔ runtime classify_question()
└── vla-3d/Unity/           VLA-3D source data (NOT committed; download per above)
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
