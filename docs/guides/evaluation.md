# Evaluation

The offline evaluation pipeline scores model predictions against ground-truth
data without the simulator running.  It supports `numerical` and
`object_reference` questions; `instruction_following` requires the official
closed-source evaluator.

## End-to-end pipeline

```
live challenge run
  → predictions.jsonl   (VLMLogger.write_prediction(), question texts = questions.json)
  + challenge_gt.jsonl  (question texts = questions.json, answers from scene graph)
  → eval_pipeline --gt dataset/challenge_gt.jsonl --pred predictions.jsonl
  → metrics (accuracy, mean IoU, challenge score)
```

The key requirement is that **question texts in both files are identical** — the
`eval_sampler` matches GT entries to predictions by exact question string.

### Ground-truth files

| File | Contents | Source |
|---|---|---|
| `dataset/challenge_gt.jsonl` | 45 official challenge questions + answers | `challenge_gt_gen.py` |
| `dataset/vla3d_ref.jsonl` | 7 708 VLA-3D generated object-reference Q&A | `vla3d_ref_to_qa.py` |
| `dataset/vla3d_num.jsonl` | 591 VLA-3D generated numerical Q&A | `vla3d_num_gen.py` |

`challenge_gt.jsonl` is what you need for scoring real challenge runs.
The VLA-3D files are used for training and ablation.

## Generating `challenge_gt.jsonl`

```bash
uv run python dataset_generator/challenge_gt_gen.py
```

This reads `questions/questions.json` from the challenge repo, resolves each
scoreable question against the VLA-3D scene graph, and writes
`dataset/challenge_gt.jsonl` (45 entries, one per question).

Verify answers before committing:

```bash
uv run python dataset_generator/challenge_gt_gen.py --verify
```

Sample output:

```
  arabic_room          numerical            How many sofas are below a window?       → 3
  arabic_room          object_reference     Find the pillow closest to the book …    → id=73 label='pillow'
  …
Resolved 45 questions  (errors/warnings: 0)
```

### Ground-truth format

```jsonc
// object_reference entry
{
  "scene": "arabic_room",
  "type": "object_reference",
  "question": "Find the pillow closest to the book on the stool.",
  "answer": {"object_id": 73, "label": "pillow"},
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
  "answer": 3,
  "object_list": [ /* same scene object list */ ]
}
```

`object_list` lines come directly from each scene's `object_list.txt` in the
challenge scene zips — format: `id cx cy cz lx ly lz heading "label"`.

## Running offline evaluation

### From the CLI

```bash
uv run xiao-hei-eval \
  --gt dataset/challenge_gt.jsonl \
  --pred vlm_logs/<session>/predictions.jsonl
```

### From Python

```python
from xiao_hei_vln.eval_sampler import assemble_samples
from xiao_hei_vln.evaluator import Evaluator

samples = assemble_samples(
    gt_path="dataset/challenge_gt.jsonl",
    pred_path="vlm_logs/my_session/predictions.jsonl",
)
report = Evaluator().evaluate(samples)
print(report)
```

## Metrics

### Numerical questions

| Metric | Description |
|---|---|
| Exact match | Fraction where `prediction == ground_truth` |
| Mean absolute error | Average `|prediction − ground_truth|` |
| Within-1 accuracy | Fraction where `|prediction − ground_truth| ≤ 1` |

### Object reference questions

| Metric | Description |
|---|---|
| IoU ≥ 0.50 | 2 points — predicted bbox overlaps GT bbox by ≥ 50 % |
| IoU ≥ 0.25 | 1 point — weaker overlap |
| IoU < 0.25 | 0 points |

### Instruction following

Requires the official challenge evaluator (simulator-based).  Not scored
offline.

## Predictions format

The `VLMLogger` writes one JSON line per question to `predictions.jsonl`:

```json
{"question": "How many sofas are below a window?", "prediction": { ... VLMOutput ... }}
```

The question text must exactly match the text in `challenge_gt.jsonl`.  Both
files use the question texts from `questions/questions.json`, so they align
automatically when the logger and GT generator are used as-is.
