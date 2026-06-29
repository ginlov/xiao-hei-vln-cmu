# FrontierExplorer

`FrontierExplorer` is the default exploration strategy.  It builds a 2-D
occupancy grid from terrain sensor data and navigates toward the boundary
between known and unknown space (the "frontier"), systematically covering
the environment.

## Algorithm — per tick

```
1. If terrain_ext is available → update OccupancyGrid (FREE / OCCUPIED cells)
2. If pose is unavailable → return current target unchanged (wait)
3. If robot is within 0.3 m of current target (odometry) → mark visited, clear target
4. If stuck timeout exceeded without nav-stack advance → skip target, blacklist area
5. If visited count ≥ max_waypoints → done (budget exhausted)
6. If no current target → select best frontier cluster centroid
   └─ if none found and grid has data → done (no frontiers)
7. Return current target waypoint
```

## Inputs

| Source | Field / Topic | What it provides |
|---|---|---|
| `VLMInput.terrain_ext` | `/terrain_map_ext` (20 m PointCloud2) | Point cloud of `(x, y, z, cost)` used to classify cells as FREE (cost ≤ 0.5) or OCCUPIED |
| `VLMInput.pose` | `/state_estimation` (Odometry) | Robot `(x, y)` position in map frame — used for reach check, frontier scoring, and stuck detection |
| `VLMInput.tick_time` | snapshot timestamp | Wall-clock seconds for stuck timeout elapsed-time tracking |
| `/way_point_reached` | `std_msgs/Float32` (subscribed separately in `main.py`) | Nav stack's live distance to the current waypoint — drives advance and early-skip |

## Output

Each `update()` call returns a `Waypoint(x, y, heading)` pointing at the
centroid of the best frontier cluster, or `None` when there is nothing to
publish.  `heading` is the angle from the robot's current position to the
target centroid (`atan2`).

## OccupancyGrid

The grid is built incrementally from `/terrain_map_ext` snapshots at
0.2 m resolution.  Each cell is in one of three states:

- **FREE** — seen at least once with `cost ≤ 0.5` (traversable)
- **OCCUPIED** — seen with `cost > 0.5`, or manually blacklisted
- **UNKNOWN** — never seen in any snapshot

**Frontier cells** are FREE cells that have at least one UNKNOWN
4-connected neighbour.  These are the boundaries of explored space.

## Frontier selection

Frontier cells are grouped into 4-connected clusters.  Clusters smaller
than `min_frontier_size` (5 cells) are discarded.  Each remaining cluster
is scored by its centroid:

```
score = cluster_size / (1 + distance_to_robot)
```

Only clusters whose centroid is within `max_waypoint_dist` (1.5 m) of
the robot are scored.  If all clusters exceed that cap, the nearest valid
cluster is used as a fallback.  Clusters whose centroid is already within
`waypoint_reach_dist` (0.3 m) of the robot are skipped entirely — they
would be marked visited immediately without any movement.

## Robustness features

### Nav-stack advance
`main.py` subscribes to `/way_point_reached` (a `Float32` publishing the
nav stack's live distance to the current waypoint).  When that distance
stays **below 0.92 m for 3 consecutive ticks (~1.5 s)**, `advance()` is
called: the target is marked visited and the frontier area is blacklisted
with a 1-cell radius so it is not re-selected next tick.

### Early-skip
When the nav stack has settled **above 0.92 m with no improvement for
5 consecutive ticks (~2.5 s)** — after an initial 4 s settling window —
`force_skip()` is called.  This handles targets the nav stack can approach
but cannot reach (obstacle inflation, narrow gaps).  Minimum time before
an early-skip fires: **6.5 s**.

### Stuck timeout
As a last resort, if a target has been held for longer than
`stuck_timeout_s` (12 s) without either an advance or an early-skip
firing, the target is abandoned via the same skip path.  This covers edge
cases where `/way_point_reached` is not publishing.

### Blacklisting
Both skip paths call `mark_occupied(x, y, radius_cells=3)` on the grid.
Skipped frontier cells are permanently added to `_blacklisted` — a set
that `OccupancyGrid.update()` will never re-free, even when subsequent
`/terrain_map_ext` snapshots report those cells as traversable.  This
prevents the robot from cycling back to the same unreachable frontier.

On `advance()`, a smaller blacklist radius (1 cell) is applied to suppress
the visited area without over-blocking nearby valid frontiers.

### Consecutive skip limit
`_consecutive_skip_count` increments on every skip and resets to 0 on
every successful advance.  When it reaches `max_consecutive_skips` (20),
`is_complete()` returns `True` with reason `max_consecutive_skips` — the
robot has hit too many dead ends in a row and exploration is abandoned.

## Current parameters

These are set in `_build_explorer()` in `app/main.py`:

| Parameter | Default | Env var |
|---|---|---|
| `max_waypoints` | 100 | `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` |
| `max_waypoint_dist` | 1.5 m | `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST` |
| `waypoint_reach_dist` | 0.3 m | — |
| `stuck_timeout_s` | 12.0 s | — |
| `max_consecutive_skips` | 20 | — |
| `grid_resolution` | 0.2 m | — |
| `min_frontier_size` | 5 cells | — |
| `cost_threshold` | 0.5 | — |
| `_WP_REACHED_THRESHOLD` | 0.92 m | — |
