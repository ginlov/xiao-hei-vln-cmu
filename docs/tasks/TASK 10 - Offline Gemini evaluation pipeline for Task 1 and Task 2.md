# TASK 10 — Offline Gemini evaluation pipeline for Task 1 and Task 2

## Purpose

Score Gemini on **Task 1 (numerical)** and **Task 2 (object_reference)**
with our existing offline evaluator, without bringing up the ROS
simulator. The evaluator chain (`eval_pipeline --gt … --pred …`) was
already complete; the only missing piece was a way to produce a Gemini
`predictions.jsonl`. This task adds that producer.

## What it does

`src/xiao_hei_vln/gemini/batch.py` reconstructs, for each GT entry, the
**same scene-graph representation** Gemini sees at test time and asks
Gemini to answer from it:

```
object_list (VLA-3D text) → SceneRepresentation → to_dict() JSON → Gemini → VLMOutput
```

- The JSON is byte-for-byte the shape
  `SceneRepresentation.to_dict()` emits in the live path (Room →
  Viewpoints → Objects, each with a 3D bbox + spatial relations), so
  Gemini reasons over the same representation as production — minus the
  rendered panorama / occupancy images, which don't exist offline.
- Same-label merging is disabled (`merge_radius=0`) so every GT object
  survives; `near` spatial edges are derived; scene bounds are set.
- Because we send a graph instead of images, the live image-centric
  prompts in `gemini/prompts.py` don't apply — `batch.py` carries its
  own scene-graph-driven system prompts (numerical / object_reference).
- `instruction_following` entries are skipped (not scored offline).

## Design choices

- **Path chosen: offline batch** (vs. live simulator run). Fast, cheap,
  no simulator; isolates Gemini's reasoning over the scene graph with
  perception assumed perfect.
- **Scene input includes bboxes.** object_reference is scored by 3D bbox
  IoU, so this measures Gemini's ability to *pick the referred object
  and copy its bbox* from the graph; numerical measures spatial-relation
  filtering + counting. It is an upper bound given perfect perception,
  not an end-to-end system score.
- Only `gemini/batch.py` is new; `SceneRepresentation`, `parse_object_list`,
  `GeminiEngine`, `eval_pipeline`, `evaluator`, and `gt_converter` are
  reused unchanged.

## How to use

```bash
export XIAO_HEI_GEMINI_API_KEY=<key>

# Task 2 (object_reference)
uv run python -m xiao_hei_vln.gemini.batch \
  --gt /home/ubuntu/workspace/dataset/tmp/vla3d_ref.jsonl \
  --out pred_ref.jsonl --limit 50
uv run python -m xiao_hei_vln.eval_pipeline \
  --gt /home/ubuntu/workspace/dataset/tmp/vla3d_ref.jsonl --pred pred_ref.jsonl

# Task 1 (numerical)
uv run python -m xiao_hei_vln.gemini.batch \
  --gt /home/ubuntu/workspace/dataset/tmp/vla3d_num.jsonl \
  --out pred_num.jsonl --limit 50
uv run python -m xiao_hei_vln.eval_pipeline \
  --gt /home/ubuntu/workspace/dataset/tmp/vla3d_num.jsonl --pred pred_num.jsonl
```

Question texts are generated from the GT entries, so they align exactly —
the eval-sampler match step has no gap. `--limit` controls API cost;
`--near-threshold` tunes the `near` edges added to the graph.

## Tests

`tests/test_gemini_batch.py` — 13 cases, all with an injected fake engine
(no API key, no network): scene reconstruction (bbox, no-merge, near
edges, bounds), JSON serialisation, prompt dispatch, `predict_entry`
routing + skips, and `run` (write shape, `--limit`, engine-error
survival).

End-to-end smoke on **real** `vla3d_{ref,num}.jsonl` with an oracle
engine (returns GT) → Task 1 accuracy 100 % / Task 2 mean IoU 1.000,
confirming the full plumbing. `--help` works without an API key.

Full related suite: 40 passed (batch + responder + eval metrics).
