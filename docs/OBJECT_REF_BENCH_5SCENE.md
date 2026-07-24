# Object-reference 5-scene bench (100 questions)

Frozen GT: first 10 ref + first 10 num per scene for
`studio`, `chinese_room`, `livingroom_3`, `office_2`, `home_building_1`
(50 + 50 = 100 questions). Same IDs scored across arms.

Worktree: `/home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref`
Branch: `aryan/object-ref-improvements`

## Default stack (fair NBV)

After this PR merges, `scripts/run_scene_vla3d_eval.sh` + `docker/compose.aryan.yml`
default to the **fair NBV** configuration that scored **ref mean IoU 0.033**
on the frozen 50-ref set:

| Knob | Default |
|---|---|
| `STRATEGY` / `XIAO_HEI_EXPLORATION_STRATEGY` | `nbv` |
| `MAX_SECONDS` / `XIAO_HEI_EXPLORATION_MAX_SECONDS` | `540` (hard-stop in `main.py`) |
| NBV ctor | `NextBestViewExplorer(max_waypoints=…)` only → `reach_dist=0.45`, skips=25 |
| `XIAO_HEI_WP_REACHED_M` | `1.35` |
| `XIAO_HEI_VIEWPOINT_RADIUS` | `0.8` |
| `XIAO_HEI_OBJECT_MAP` | `1` |
| `XIAO_HEI_REF_SPATIAL` | `1` |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST` | `3.0` |

## Freeze command

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
uv run python scripts/freeze_bench_gt.py \
  --scenes studio chinese_room livingroom_3 office_2 home_building_1 \
  --num 10 --ref 10 \
  --out artifacts/bench_5scene_100q/gt
```

## How to run (defaults = fair NBV, 50 ref)

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
set -a && source .env && set +a
export DISPLAY=:0
export OUT_DIR=$PWD/artifacts/bench_nbv_fair_50
# STRATEGY/MAX_SECONDS/REF_SPATIAL/OBJECT_MAP/WP/VIEWPOINT already default correctly
export TIMEOUT=1200 SPLITS=ref LIMIT_Q=10
scripts/run_scene_vla3d_eval.sh --limit 10 --splits ref \
  studio chinese_room livingroom_3 office_2 home_building_1
```

Equivalent explicit env (same as defaults):

```bash
export STRATEGY=nbv MAX_SECONDS=540 TIMEOUT=1200 SPLITS=ref LIMIT_Q=10
export XIAO_HEI_REF_SPATIAL=1
export XIAO_HEI_WP_REACHED_M=1.35
export XIAO_HEI_VIEWPOINT_RADIUS=0.8
export XIAO_HEI_OBJECT_MAP=1
```

## Comparison on the same frozen 50 ref questions

| Arm | Ref mean IoU | Ref SR@0.25 | Challenge /2 | Notes |
|---|---:|---:|---:|---|
| **Fair NBV (defaults after this PR)** | **0.0326** | **0.060** | **0.060** | hard-stop + NBV class defaults + spatial on |
| Old baseline frontier (spatial off) | 0.0211 | 0.040 | 0.040 | earlier A/B arm |
| Old “improved” NBV (broken wiring) | 0.0019 | 0.000 | 0.000 | env `MAX_SECONDS` ignored; NBV `reach_dist` forced to 0.3 |

### Fair NBV per-scene (n=10 each)

| Scene | Mean IoU | Stop reason | Visited |
|---|---:|---|---:|
| studio | 0.0363 | `max_consecutive_skips` | 5 |
| chinese_room | 0.0154 | `time_limit` | 9 |
| livingroom_3 | 0.0000 | `time_limit` | 24 |
| office_2 | 0.0511 | `time_limit` | 9 |
| home_building_1 | 0.0604 | `time_limit` | 12 |

### Older A/B arms (for history)

#### Baseline (frontier, spatial off) — exact command

```bash
export OUT_DIR=$PWD/artifacts/bench_5scene_100q/baseline
export STRATEGY=frontier MAX_SECONDS=540 TIMEOUT=1200 SPLITS=ref,num LIMIT_Q=10
export XIAO_HEI_REF_SPATIAL=0
scripts/run_scene_vla3d_eval.sh --limit 10 --splits ref,num \
  studio chinese_room livingroom_3 office_2 home_building_1
```

#### Broken improved arm (do not reproduce) — env set but agent ignored `MAX_SECONDS`

```bash
export OUT_DIR=$PWD/artifacts/bench_5scene_100q/improved
export STRATEGY=nbv MAX_SECONDS=540 TIMEOUT=1200 SPLITS=ref,num LIMIT_Q=10
export XIAO_HEI_REF_SPATIAL=1
# Pre-fix: main.py did not hard-stop; NBV forced waypoint_reach_dist=0.3
```

## Notes

- Fair NBV preds were verified 1:1 against the frozen first-10 questions per scene.
- Explore is still high-variance; expect run-to-run spread around ~0.03 on this slice.
- Do not hand-merge dumps; score single-explore live scenes only.
