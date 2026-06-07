# TASK 7 — Evaluation pipeline integration

## Problem

All three evaluation components exist but are not connected end-to-end:

| Component | Status | Format |
|---|---|---|
| Ground-truth data | Ready | VLA-3D JSONL: `{type, question, answer, object_list, scene}` |
| System output | Ready | predictions JSONL: `{question, prediction: VLMOutput}` |
| Evaluation pipeline | Ready | `eval_pipeline --gt <GT> --pred <predictions>` → metrics |

The pipeline works by matching GT entries to predictions **by exact question text**.
The gap is that the question texts in the two files never match in practice:

- `predictions.jsonl` (from a live challenge run) contains the **75 official challenge
  questions** from `questions.json`, e.g. `"Find the pillow closest to the book on the stool."`.
- The GT data (`vla3d_ref.jsonl`, `vla3d_num.jsonl`) contains **VLA-3D-generated questions**,
  e.g. `"Find the lamp on the table closest to the potted bamboo."`.

These are entirely different strings. The `eval_sampler` finds zero matches and the
evaluation produces no results.

## Root cause

`questions.json` (the challenge's official 75 questions across 15 training scenes) has
no corresponding ground-truth JSONL file. It only stores question texts, not answers.
There is no `challenge_gt.jsonl` that would let the eval pipeline score a real run.

## What is missing

A **`challenge_gt.jsonl`** file whose question texts exactly mirror `questions.json`,
with answers and object lists filled in from scene data:

```jsonc
// object_reference entry
{
  "scene": "arabic_room",
  "type": "object_reference",
  "question": "Find the pillow closest to the book on the stool.",
  "answer": {"object_id": 2, "label": "pillow"},
  "object_list": [
    "0 -3.77 2.05 0.61 0.67 0.64 1.23 0.008 \"potted plant\"",
    "2 1.94 -2.09 0.41 0.42 0.21 0.36 0.199 \"pillow\"",
    // … all objects in the scene …
  ]
}

// numerical entry
{
  "scene": "arabic_room",
  "type": "numerical",
  "question": "How many sofas are below a window?",
  "answer": 1,
  "object_list": [/* same scene object list */]
}
```

The `object_list` lines are pulled directly from each scene's `object_list.txt`
(available in the challenge scene zips under
`CMU-VLN-Challenge-data/unity_env_models/<scene>.zip`). Their format is identical
to `dummy/data/object_list.txt` and is already understood by `eval_sampler.object_list`.

The `answer` values require annotation against the scene data:

- **`object_reference`**: determine which `object_id` from the scene's `object_list.txt`
  is the target, using the VLA-3D scene graph for spatial reasoning
  (relations such as `closest`, `between`, `on`, etc.).
- **`numerical`**: count the qualifying objects from the `object_list.txt` and scene graph.
- **`instruction_following`**: skipped — not scored by our offline pipeline.

## Planned fix

Create `dataset_generator/challenge_gt_gen.py` that:

1. Reads `questions/questions.json` from the challenge repo.
2. For each scene, extracts and reads `object_list.txt` from the corresponding scene zip.
3. For each scoreable question (`numerical` / `object_reference`), produces a GT entry
   in the eval-sampler format.
4. Writes `dataset/challenge_gt.jsonl`.

Once `challenge_gt.jsonl` exists, the full offline evaluation chain is:

```
live challenge run
  → predictions.jsonl  (logger.write_prediction(), question texts = questions.json)
  + challenge_gt.jsonl (question texts = questions.json, answers annotated)
  → eval_pipeline --gt dataset/challenge_gt.jsonl --pred predictions.jsonl
  → metrics (accuracy, mean IoU, challenge score)
```

Answer annotations will be computed programmatically using the VLA-3D scene graphs
where possible; any question that cannot be resolved automatically will be flagged
for manual review.
