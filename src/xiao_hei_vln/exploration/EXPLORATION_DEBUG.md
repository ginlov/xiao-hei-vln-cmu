# Exploration Debugging Summary

## Problem (original)
Robot exploration completed 200 "waypoints" in seconds without actually moving. Count incremented every tick while the robot sat still.

## Root Causes Found & Fixed

### 1. False visits (primary bug) — FIXED
`waypoint_reach_dist=1.0m` was too large. Frontier centroid landed within reach of the stationary robot every tick. Fixed by: reducing to 0.3m, AND fixing `_select_frontier` to track `nearest_wp` fallback only from clusters OUTSIDE reach_dist (old code's fallback bypassed the within-reach filter).

### 2. Nav stack settles short of goal — FIXED (tuned)
`/way_point_reached` (Float32) is a **continuous distance topic** — not a one-shot event. Nav stack navigates, then stabilizes at a fixed distance when it can't go further (obstacle inflation). The settled distance varies widely:
- ~0.25m — open space, close goal
- ~0.37-0.65m — frontier near an obstacle
- ~0.75-0.88m — tighter obstacle clearance required
- ~1.0-1.8m — mostly blocked path

**Fix in `main.py` tick loop:** when `/way_point_reached` stays below `_WP_REACHED_THRESHOLD` for 3 consecutive ticks (~1.5s), call `explorer.advance()`. After advance fires, reset `_wp_reached_state["value"] = inf` so the stale distance doesn't carry over to the next target.

**Threshold evolution:**
- `0.35m` (original) — only worked for the very first waypoint; most targets settled at 0.37-0.88m so advance() never fired
- `0.75m` (current) — covers most reachable targets based on observed nav_debug.log values

### 3. Stuck in dead end → infinite reselect loop — FIXED
After a stuck skip, `_select_frontier` returned the same unreachable frontier. Fixed by `mark_occupied(..., radius_cells=3)` on skip.

### 4. Cycling through same 5-6 frontier points — FIXED
After `advance()`, the frontier wasn't suppressed. Fixed by `mark_occupied(..., radius_cells=1)` in `advance()`.

### 5. `max_consecutive_skips=3` too aggressive — FIXED
Terminated exploration when robot was genuinely exploring new areas (3 different skips in new space). Increased to 10.

### 6. Nav topic carry-over false advance — FIXED
After `advance()`, `_wp_reached_state["value"]` retained the old settled distance. Next tick would immediately count toward the new target's close_ticks, causing a false advance after just 3 ticks. Fixed by resetting to `inf` after each advance().

## Current Parameters (`_build_explorer` in `main.py`)
```python
FrontierExplorer(
    max_waypoints=N,             # set via XIAO_HEI_EXPLORATION_MAX_WAYPOINTS
    waypoint_reach_dist=0.3,     # odometry fallback threshold (rarely triggers now)
    max_waypoint_dist=1.5,       # prefer frontiers within 1.5m
    stuck_timeout_s=12.0,        # skip waypoint if no progress in 12s (nav needs time for far targets)
    max_consecutive_skips=20,    # give up after 20 skips in a row
)
_WP_REACHED_THRESHOLD = 0.92    # /way_point_reached threshold to trigger advance()
_wp_reached_state = {"value": inf, "close_ticks": 0}  # reset value after each advance
```

## Key Topics
| Topic | Type | What it is |
|---|---|---|
| `/way_point_with_heading` | Pose2D | We publish target waypoints here |
| `/way_point_reached` | Float32 | Nav stack's continuous distance to current waypoint (~2Hz, may publish less when idle) |
| `/traversable_area` | PointCloud2 | Nav stack's traversable map (logged periodically to nav_debug.log) |
| `/state_estimation` | Odometry | Robot pose — takes 90-190s to arrive after container start |

## Files Changed
- `src/xiao_hei_vln/exploration/_frontier.py` — stuck detection, `max_consecutive_skips`, `advance()` method, frontier filter fix, `mark_occupied` on skip and advance
- `src/xiao_hei_vln/exploration/_grid.py` — added `mark_occupied()` method
- `src/xiao_hei_vln/app/main.py` — `/way_point_reached` subscriber, nav-distance-based advance with carry-over fix, heartbeat logging, skip event logging

## Observation from Latest Run (max_waypoints=5000)
Robot physically traveled from (0,0) to (4.93, -3.74) — 6+m range. Only 4 "visited" (advance triggered) + 10 skips (stuck timeout). The robot DID explore the space; skips were doing genuine exploration because `mark_occupied` blacklisted each skipped area. The problem was just that `advance()` was barely triggering (0.35m threshold too tight).

`nav_debug.log` (at `exploration_logs/nav_debug.log`) shows raw `/way_point_reached` values per run — check this to tune `_WP_REACHED_THRESHOLD`.

## What to Try If Still Broken
1. Check `nav_debug.log` — if values are consistently above 0.75m, raise threshold to 0.9m
2. If robot stops exploring after 10 consecutive skips but the map isn't covered, increase `max_consecutive_skips`
3. Subscribe to `/traversable_area` and use it as the occupancy grid source — waypoints would then always be in nav-stack-reachable space

## Run Command
```
XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=200 XIAO_HEI_RESPONDER=dummy \
  docker compose -f docker/compose_gpu.yml up -d --build
```
