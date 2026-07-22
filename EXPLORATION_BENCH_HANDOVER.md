# Exploration Benchmark Handover

**Date:** 2026-07-22  
**Branch:** `aryan/exploration`  
**Repo:** `/home/ubuntu/workspace/aryan/xiao-hei-vln-cmu`  
**Owner context:** Aryan — DCV + Cursor SSH on shared GPU box

---

## What was done (this session)

### Live exploration fixes (Unity / ROS)
- Soft-ban skips (not permanent wipe); hard-ban only after **3 fail poses ≥2 m apart**
- Soft-ban **rescue** before `no_frontiers`
- Score frontiers by `(size + unknown_gain_weight * unknown_region_size) / (1 + path_cost)`
- RViz markers on `/exploration/markers` (`exploration_markers.py`)
- Strategies registered in `app/main.py`: `frontier`, `nearest`, `random`, `lawnmower`

### Offline GT benchmark (autonomous loop)
- Metric + GT world + algos + full matrix over **18 scenes**
- Results: `exploration_logs/bench_results.json`, `exploration_logs/bench_results.csv`

---

## Metric

**Primary: `gt_coverage`**

\[
\text{gt\_coverage} = \frac{|\text{sensor\_seen} \cap \text{GT\_free}|}{|\text{GT\_free}|}
\]

- **GT_free:** voxelized cells from `traversable_area.ply` (default res `0.25 m`)
- **sensor_seen:** GT free cells hit by simulated LiDAR raycasts along the trajectory
- **Secondary:** `mapped_coverage` (explorer FREE ∩ GT), `path_length_m`, `coverage_per_meter`

This is **offline** (no Unity). It approximates “how much walkable floor did we see?” under perfect A* nav + raycast sensing.

---

## Algorithms evaluated

| Name | Description |
|------|-------------|
| `frontier_baseline` | Current FrontierExplorer (`max_waypoint_dist=3`, soft rescue, `unknown_gain_weight=1`) |
| `frontier_near` | `max_waypoint_dist=1.5` |
| `frontier_far` | `max_waypoint_dist=5.0` |
| `frontier_info_heavy` | `unknown_gain_weight=5`, `max_waypoint_dist=4` |
| `frontier_no_rescue` | `max_soft_rescues=0` |
| `nearest_frontier` | Always path-nearest reachable frontier |
| `random_frontier` | Seeded random among reachable frontiers |
| `lawnmower` | Serpentine sweep over known FREE |

Code:
- `exploration/_frontier.py`, `_nearest.py`, `_lawnmower.py`
- `exploration/_gt_world.py`, `_metric.py`
- Runner: `scripts/bench_exploration.py`

---

## How to re-run

```bash
docker run --rm \
  -v /home/ubuntu/workspace/aryan/xiao-hei-vln-cmu:/opt/xiao_hei_vln \
  -v /home/ubuntu/Downloads/unity_env_models:/scenes:ro \
  --entrypoint bash xiao-hei/ai_module:scene_gemini -c '
cd /opt/xiao_hei_vln && PYTHONPATH=src python3 scripts/bench_exploration.py \
  --scenes-root /scenes \
  --max-ticks 300 \
  --out /opt/xiao_hei_vln/exploration_logs/bench_results.json
'
```

Subset: `--scenes chinese_room arabic_room --algos frontier_far nearest_frontier`

---

## Live Unity run (DCV + SSH)

```bash
# SSH
export XIAO_HEI_GEMINI_API_KEY='…'
export XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST=3.0
export XIAO_HEI_SCENE_DIR_HOST=/home/ubuntu/Downloads/unity_env_models/<scene>
export XIAO_HEI_EXPLORATION_STRATEGY=frontier   # or nearest|random|lawnmower
cd /home/ubuntu/workspace/aryan/xiao-hei-vln-cmu
docker compose -f docker/compose_scene_gemini.yml -f docker/compose.scene.yml up -d --force-recreate
# After Unity up: recreate ai_module; ensure waypointConverter

# DCV
export DISPLAY=:0 && xhost +local:
docker exec -it iros2026_system \
  /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

RViz: add **MarkerArray** → `/exploration/markers`  
SSH key for git: `~/workspace/aryan/.ssh/id_aryan`

---

## Known caveats

- Offline bench ≠ real local-planner (no inflation / wall-push `nav=inf`)
- Early `complete` with tiny path (e.g. arabic `frontier_baseline` 3 ticks) means “no frontiers left” in the *belief* map, not full GT coverage
- `random_frontier` wins mean coverage but has catastrophic lows on large offices (~0.39)
- Uncommitted local changes on `aryan/exploration` — not pushed

---

## Results (18 scenes × 8 algos, max_ticks=300)

**Metric:** `gt_coverage` = fraction of GT traversable cells sensed.

### Ranking by mean gt_coverage

| Rank | Algo | Mean | Min | Max |
|------|------|------|-----|-----|
| 1 | **random_frontier** | **0.872** | 0.389 | 1.000 |
| 2 | **nearest_frontier** | **0.867** | 0.454 | 1.000 |
| 3 | frontier_far | 0.860 | 0.526 | 1.000 |
| 4 | frontier_info_heavy | 0.856 | 0.582 | 1.000 |
| 5 | frontier_near | 0.817 | 0.582 | 1.000 |
| 6 | lawnmower | 0.785 | 0.297 | 1.000 |
| 7 | frontier_baseline / no_rescue | 0.773 | 0.582 | 1.000 |

### Ranking by mean per-scene rank (1 = best on that scene)

1. **nearest_frontier** (3.06) ← most consistent winner  
2. random_frontier (3.33)  
3. frontier_near (4.11)  
…

### Recommended default

**`nearest_frontier`** — nearly best mean coverage, **best mean rank**, less catastrophic than random on large offices.

Live: `XIAO_HEI_EXPLORATION_STRATEGY=nearest`

Artifacts: `exploration_logs/bench_results.json`, `bench_results.csv`
