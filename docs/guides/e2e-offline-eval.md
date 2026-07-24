# End-to-end offline eval (GT + live)

This guide covers the VLA-3D offline harness for Task 1 (numerical) and
Task 2 (object_reference):

1. **GT offline** — perfect-perception upper bound (`--object-source gt`)
2. **Live offline** — explore a Unity scene, dump the perception scene
   graph, then score the same questions against that dump
   (`--object-source live`)

Harness entry points:

- `scripts/run_e2e_offline_eval.sh` / `scripts/run_scene_vla3d_eval.sh`
- `python -m xiao_hei_vln.gemini.batch`
- `scripts/export_live_scene_for_offline_eval.py`
- `scripts/freeze_bench_gt.py` (freeze first N Qs per scene/split)

Live explore uses an **isolated eval compose overlay**
(`docker/compose.eval.yml`, project `xiao_hei_eval` by default) so a second
stack can run next to a shared `compose_scene_gemini` stack without
clobbering container names, ROS domain, or the perception port.

## Prerequisites

```bash
export XIAO_HEI_GEMINI_API_KEY=...   # or source .env
export DISPLAY=:0
xhost +local:
# Optional: export GT_DIR=/path/to/dir/with/vla3d_{ref,num}.jsonl
# Optional: export SCENES_DIR=~/Downloads/unity_env_models
```

Unity scenes under `$SCENES_DIR/<scene>/environment`
(default `~/Downloads/unity_env_models`).

## Verified smoke commands

Artifact root: `artifacts/e2e_harness_smoke`. Run from the repo root.

### 1) GT-only offline (5 ref + 5 num on studio)

```bash
cd /path/to/xiao-hei-vln-cmu
set -a && source .env && set +a
export DISPLAY=:0
export OUT_DIR=$PWD/artifacts/e2e_harness_smoke
export SPLITS=ref,num LIMIT_Q=5
scripts/run_e2e_offline_eval.sh --gt-only --limit 5 --splits ref,num studio
```

Expected:

- `artifacts/e2e_harness_smoke/gt/studio_{ref,num}.jsonl` (5 lines each)
- `artifacts/e2e_harness_smoke/preds/studio_{ref,num}.jsonl`
- `artifacts/e2e_harness_smoke/metrics/studio_{ref,num}.json`

### 2) Live explore + offline (same 5+5 on studio, frontier)

```bash
cd /path/to/xiao-hei-vln-cmu
set -a && source .env && set +a
export DISPLAY=:0
export OUT_DIR=$PWD/artifacts/e2e_harness_smoke
export STRATEGY=frontier MAX_SECONDS=180 TIMEOUT=780 SPLITS=ref,num LIMIT_Q=5
scripts/run_e2e_offline_eval.sh --limit 5 --splits ref,num studio
```

Expected:

- RViz / robot motion on the isolated eval containers
  (`xiao_hei_eval_iros2026_system`, `xiao_hei_eval_ai_module`, `ROS_DOMAIN_ID=42`)
- `artifacts/e2e_harness_smoke/explored_scenes/studio/scene.json`
- preds + metrics under the same `OUT_DIR` as above

### 3) Reuse an existing live dump

```bash
cd /path/to/xiao-hei-vln-cmu
set -a && source .env && set +a
export OUT_DIR=$PWD/artifacts/e2e_harness_smoke
export SPLITS=ref,num LIMIT_Q=5
scripts/run_e2e_offline_eval.sh --skip-explore --limit 5 --splits ref,num studio
```

## Freeze a multi-scene bench slice

```bash
uv run python scripts/freeze_bench_gt.py \
  --scenes studio chinese_room livingroom_3 office_2 home_building_1 \
  --num 10 --ref 10 \
  --out artifacts/bench_5scene_100q/gt
```

`--limit N` on the eval scripts also freezes the **first N GT rows** per
scene/split (not only a prediction cap), so scored IDs stay fixed.

## Direct `gemini.batch` usage

```bash
uv run python -m xiao_hei_vln.gemini.batch \
  --gt artifacts/e2e_harness_smoke/gt/studio_ref.jsonl \
  --out artifacts/e2e_harness_smoke/preds/studio_ref.jsonl \
  --object-source gt --limit 5

uv run python -m xiao_hei_vln.gemini.batch \
  --gt artifacts/e2e_harness_smoke/gt/studio_ref.jsonl \
  --out artifacts/e2e_harness_smoke/preds/studio_ref_live.jsonl \
  --object-source live \
  --live-scenes-dir artifacts/e2e_harness_smoke/explored_scenes \
  --limit 5

uv run python -m xiao_hei_vln.eval_pipeline \
  --gt artifacts/e2e_harness_smoke/gt/studio_ref.jsonl \
  --pred artifacts/e2e_harness_smoke/preds/studio_ref_live.jsonl \
  --out artifacts/e2e_harness_smoke/metrics/studio_ref_live.json
```

## Smoke results

Verified on the e2e-eval-harness branch (studio, `LIMIT_Q=5`).

| Arm | Split | Metric | Value |
|---|---|---|---|
| GT-only | ref | mean IoU | 0.600 |
| GT-only | ref | SR@0.5 | 0.600 |
| GT-only | num | accuracy | 0.600 |
| GT-only | num | MAE | 0.400 |
| Live frontier (`MAX_SECONDS=180`) | ref | mean IoU | 0.001 |
| Live frontier | num | accuracy | 0.000 |
| Live frontier | num | MAE | 1.400 |

Live explore ended with `DONE visited=3 skipped=21 reason=max_consecutive_skips`
(short smoke cap). Dump present at
`artifacts/e2e_harness_smoke/explored_scenes/studio/scene.json`.

## Isolated eval stack knobs

| Env | Default | Meaning |
|---|---|---|
| `COMPOSE_PROJECT` | `xiao_hei_eval` | `docker compose -p …` project name |
| `XIAO_HEI_EVAL_PREFIX` | `xiao_hei_eval` | container name prefix |
| `ROS_DOMAIN_ID` | `42` | ROS domain for the eval stack |
| `PERCEPTION_PORT` | `8002` | perception uvicorn port (base often 8001) |
| `GT_DIR` / `XIAO_HEI_GT_DIR` | auto | directory with `vla3d_{ref,num}.jsonl` |
