# Exploration Debugging Summary

## Problem (original)
Robot exploration completed 200 "waypoints" in seconds without actually moving. Count incremented every tick while the robot sat still.

## Root Causes Found & Fixed

### 1. False visits — FIXED
`waypoint_reach_dist=1.0m` was too large. Frontier centroid landed within reach of the stationary robot every tick. Fixed by: reducing to 0.3m, AND fixing `_select_frontier` to track `nearest_wp` fallback only from clusters OUTSIDE reach_dist (old code's fallback bypassed the within-reach filter).

### 2. Nav stack settles short of goal — FIXED
`/way_point_reached` (Float32) is a **continuous distance topic** — not a one-shot event. Nav stack navigates, then stabilizes at a fixed distance when it can't go further (obstacle inflation). Settled distance varies by environment:
- ~0.25m — open space, close goal
- ~0.37-0.65m — frontier near an obstacle
- ~0.75-0.90m — tighter obstacle clearance required
- ~1.0-1.8m — mostly blocked path

**Fix:** when `/way_point_reached` stays below `_WP_REACHED_THRESHOLD` for 3 consecutive ticks (~1.5s), call `explorer.advance()`. Reset `_wp_reached_state["value"]` and `["best"]` to `inf` after each advance so stale distances don't carry over to the next target.

### 3. Stuck in dead end → infinite reselect loop — FIXED
After a stuck skip, `_select_frontier` returned the same unreachable frontier. Fixed by `mark_occupied(..., radius_cells=3)` on skip.

### 4. Cycling through same frontier points — FIXED
After `advance()`, the frontier wasn't suppressed. Fixed by `mark_occupied(..., radius_cells=1)` in `advance()`.

### 5. Nav topic carry-over false advance — FIXED
After `advance()`, stale distance value immediately counted toward the new target's close_ticks, causing a false advance after 3 ticks. Fixed by resetting `value`, `best`, and `close_ticks` in both the advance block and WP_SET. Also reset `last_exploration_wp` after advance so the next target always triggers WP_SET.

### 6. mark_occupied undone by terrain updates — FIXED
`mark_occupied()` suppresses a frontier by adding cells to `_occupied`. But `OccupancyGrid.update()` runs every tick and re-adds those same cells to `_free` whenever the terrain sensor reports them as traversable (cost ≤ 0.5) — which it always does for open floor. This caused the robot to re-select the same unreachable frontier on every tick, spinning through the full `stuck_timeout_s × max_consecutive_skips` before giving up.

**Fix:** added `_blacklisted` set in `OccupancyGrid`. Cells passed to `mark_occupied()` are permanently blacklisted and skipped by `update()`, so terrain data can never restore them.

## Current Parameters (`_build_explorer` in `main.py`)
```python
FrontierExplorer(
    max_waypoints=N,             # set via XIAO_HEI_EXPLORATION_MAX_WAYPOINTS
    waypoint_reach_dist=0.3,     # odometry fallback threshold (rarely triggers)
    max_waypoint_dist=1.5,       # prefer frontiers within 1.5m
    stuck_timeout_s=12.0,        # skip waypoint if no progress in 12s
    max_consecutive_skips=20,    # give up after 20 skips in a row
)
_WP_REACHED_THRESHOLD = 0.92    # nav stack settles between 0.25-0.90m
_wp_reached_state = {"value": inf, "close_ticks": 0, "best": inf}
```

## exploration.log format
Written to `exploration_logs/exploration.log` — one structured event per line:
```
[timestamp_s] EVENT  key=value  key=value ...
```

| Event | Fields | What it means |
|---|---|---|
| `START` | max_waypoints, threshold, stuck_timeout, max_skips | Logged once when exploration begins |
| `WP_SET` | target, robot, dist | New frontier target selected; dist = robot→target |
| `WP_ADVANCE` | target, nav_dist, visited | Nav stack settled below threshold; waypoint counted as visited |
| `WP_SKIP` | target, robot, elapsed, best_nav_dist, last_nav_dist, consecutive | Stuck timeout fired |
| `DONE` | visited, skipped, reason | reason = budget_exhausted \| max_consecutive_skips \| no_frontiers |

**Reading WP_SKIP:**
- `best_nav_dist` — closest the nav stack got to the target
- `last_nav_dist` — distance at timeout; if `last > best`, the nav stack retreated after approaching
- If `best_nav_dist` consistently > 0.92m across many skips → raise `_WP_REACHED_THRESHOLD` or those frontiers are genuinely inaccessible

## Key Topics
| Topic | Type | What it is |
|---|---|---|
| `/way_point_with_heading` | Pose2D | We publish target waypoints here |
| `/way_point_reached` | Float32 | Nav stack's continuous distance to current waypoint (not a one-shot event) |
| `/state_estimation` | Odometry | Robot pose — takes 90-190s to arrive after container start |

## Files Changed
- `src/xiao_hei_vln/exploration/_frontier.py` — stuck detection, `advance()`, frontier filter fix, `mark_occupied` on skip and advance
- `src/xiao_hei_vln/exploration/_grid.py` — added `mark_occupied()`, `_blacklisted` set to permanently suppress frontier cells
- `src/xiao_hei_vln/app/main.py` — structured `exploration.log`, `/way_point_reached` subscriber, nav-distance-based advance

## Run Command
```
XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=200 XIAO_HEI_RESPONDER=dummy \
  docker compose -f docker/compose_gpu.yml up -d --build
```
