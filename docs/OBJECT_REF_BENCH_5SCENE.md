# Object-reference 5-scene bench (100 questions)

Frozen GT: first 10 ref + first 10 num per scene for
`studio`, `chinese_room`, `livingroom_3`, `office_2`, `home_building_1`
(50 + 50 = 100 questions). Same IDs scored on baseline and improved dumps
(re-explore each arm separately).

Worktree: `/home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref`
Branch: `aryan/object-ref-improvements`

## Freeze command

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
uv run python scripts/freeze_bench_gt.py \
  --scenes studio chinese_room livingroom_3 office_2 home_building_1 \
  --num 10 --ref 10 \
  --out artifacts/bench_5scene_100q/gt
```

## Baseline (frontier, spatial off)

Exact command run:

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
set -a && source .env && set +a
export DISPLAY=:0
export OUT_DIR=$PWD/artifacts/bench_5scene_100q/baseline
export STRATEGY=frontier MAX_SECONDS=540 TIMEOUT=1200 SPLITS=ref,num LIMIT_Q=10
export XIAO_HEI_REF_SPATIAL=0
export XIAO_HEI_WP_REACHED_M=1.35
scripts/run_scene_vla3d_eval.sh --limit 10 --splits ref,num \
  studio chinese_room livingroom_3 office_2 home_building_1
```

Artifacts: `artifacts/bench_5scene_100q/baseline/{explored_scenes,gt,preds,metrics,summary.csv}`

## Improved (NBV + spatial on)

Exact command run:

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
set -a && source .env && set +a
export DISPLAY=:0
export OUT_DIR=$PWD/artifacts/bench_5scene_100q/improved
export STRATEGY=nbv MAX_SECONDS=540 TIMEOUT=1200 SPLITS=ref,num LIMIT_Q=10
export XIAO_HEI_REF_SPATIAL=1
export XIAO_HEI_WP_REACHED_M=1.35
scripts/run_scene_vla3d_eval.sh --limit 10 --splits ref,num \
  studio chinese_room livingroom_3 office_2 home_building_1
```

`chinese_room` timed out waiting for `DONE` on the first NBV pass; scoring used the
periodic `exploration_logs/chinese_room/scene_live.json` dump via:

```bash
mkdir -p artifacts/bench_5scene_100q/improved/explored_scenes/chinese_room
cp exploration_logs/chinese_room/scene_live.json \
  artifacts/bench_5scene_100q/improved/explored_scenes/chinese_room/scene.json
export OUT_DIR=$PWD/artifacts/bench_5scene_100q/improved
export STRATEGY=nbv SPLITS=ref,num LIMIT_Q=10 XIAO_HEI_REF_SPATIAL=1
scripts/run_scene_vla3d_eval.sh --skip-explore --limit 10 --splits ref,num chinese_room
```

Artifacts: `artifacts/bench_5scene_100q/improved/{explored_scenes,gt,preds,metrics,summary.csv}`
Aggregate: `artifacts/bench_5scene_100q/aggregate.json`

## Aggregate results (n=50 each split)

| Arm | Ref mean IoU | Ref SR@0.5 | Ref SR@0.25 | Ref center dist (m) | Ref challenge /2 | Num accuracy | Num MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline frontier | 0.0211 | 0.000 | 0.040 | 4.705 | 0.040 | 0.120 | 1.900 |
| Improved NBV+spatial | 0.0019 | 0.000 | 0.000 | 4.737 | 0.000 | 0.140 | 1.980 |

### Per-scene object-reference mean IoU

| Scene | Baseline | Improved |
|---|---:|---:|
| studio | 0.0000 | 0.0063 |
| chinese_room | 0.0552 | 0.0018 |
| livingroom_3 | 0.0000 | 0.0000 |
| office_2 | 0.0504 | 0.0000 |
| home_building_1 | 0.0000 | 0.0012 |

### Per-scene numerical accuracy / MAE

| Scene | Baseline acc | Baseline MAE | Improved acc | Improved MAE |
|---|---:|---:|---:|---:|
| studio | 0.00 | 1.700 | 0.20 | 1.500 |
| chinese_room | 0.10 | 0.900 | 0.10 | 0.900 |
| livingroom_3 | 0.30 | 1.600 | 0.20 | 1.900 |
| office_2 | 0.10 | 3.100 | 0.10 | 3.500 |
| home_building_1 | 0.10 | 2.200 | 0.10 | 2.100 |

## Notes

- Compare mode is **re-explore**: dumps differ by design; question IDs are fixed.
- Improved arm defaults: `STRATEGY=nbv`, `XIAO_HEI_REF_SPATIAL=1`, `XIAO_HEI_WP_REACHED_M=1.35`, NBV wall-view bias.
- On this frozen 100-Q slice, improved ref mean IoU did **not** beat baseline; num accuracy edged up slightly (0.12 → 0.14). Studio-tuned focus-light gains from earlier work do not generalize to this multi-scene head-10 freeze.
