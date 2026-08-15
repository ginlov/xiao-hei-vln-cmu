# TASK 43 — Why both explorers stall, and which one to keep

Two sweeps were on disk with no verdict attached: `exploration_logs_frontier/`
(19 scenes, no wall-clock budget) and `exploration_logs_nbv/` (13 scenes, the
480 s budget from TASK 42). The question was which strategy to invest in, and
then to fix what the logs showed was broken.

## Verdict: frontier

The budgets differ, so absolute totals are not comparable. Normalising by wall
clock on the 12 scenes both strategies ran:

| | frontier | nbv |
|---|---|---|
| mean duration | 353 s | 389 s |
| mean path travelled | 87.8 m | 35.5 m |
| **metres per minute** | **9.17** | **5.01** |
| mean pose-bbox area | 135.8 m² | 43.7 m² |
| mean waypoints advanced | 16.9 | 4.1 |

Frontier wins metres-per-minute on 9 of 12 scenes. It covers ~1.8× more ground
per unit time *and* did so while terminating early on most scenes — it had no
time budget, so its skip-hatch fired at 130–250 s while nbv was still running
its full 480 s.

The decisive number is not the ratio, it is *why* each one fails. Splitting
skips by whether `/way_point_reached` ever published during the attempt:

| | frontier | nbv |
|---|---|---|
| skips where nav stack stayed silent (`best_nav_dist=inf`) | 409 / 611 | 248 / 487 |
| median robot displacement during those skips | **2.39 m** | **0.00 m** |

Frontier's failures are a robot that was driving and got interrupted. Nbv's are
a robot that could not move at all. The first is a bad supervisor heuristic,
fixable in the tick loop; the second is the strategy poisoning its own map.
Frontier is the one to keep, and nbv is worth keeping alive only as a
comparison baseline.

## Root causes and fixes

### 1. A silent nav stack read as "the robot has stopped" (both; biggest win)

`main.py`'s early-skip fires when `best_nav_dist` fails to improve for 5 ticks.
`best_nav_dist` initialises to `inf`, and `inf >= inf - 0.02` is true, so
whenever `/way_point_reached` published nothing the counter ran to 5 unopposed
and the waypoint was dropped ~6 s in. That is 97% of frontier's skips and 80%
of nbv's — they are almost all the early skip, not the 12 s stuck timeout.

Fix: track distance-to-target in odometry alongside the nav topic
(`best_odom` / `last_progress_time`) and refuse to early-skip while the robot
is visibly closing on the goal. Replaying the recorded skips against the new
predicate: **71% of frontier's early skips and 36% of nbv's were still closing
on their target** and would now be deferred to the stuck timeout.
`best_odom_dist` is added to the `WP_SKIP` log line so the next sweep is
diagnosable without this reconstruction.

### 2. Explorers stamped their own waypoints as obstacles (both; nbv-fatal)

Both strategies called `OccupancyGrid.mark_occupied()` on visited and skipped
waypoints to stop re-selecting them. That does not just suppress a goal — it
deletes the cells from `_free` and blacklists them against future terrain
updates. A waypoint in a doorway becomes a wall. Nbv picks targets through
`reachable_path_costs()`, a 4-connected BFS over free cells, so every visited
waypoint chopped a 1.4 m hole in its own reachability graph until nothing was
reachable. Hence the 0.00 m median displacement.

Fix: new `mark_no_target()` / `is_targetable()` on the grid, backed by a
separate `_no_target` set. Cells stay FREE and stay traversable; they are only
excluded from *selection*. `mark_occupied()` keeps its old meaning and is now
reserved for genuine obstacles.

### 3. Frontier re-issued one target 18 times in a row (`loft`)

Suppressing the cells under a rejected centroid leaves the rest of the cluster
averaging to nearly the same point, so `(3.62,0.62)` → `(3.64,0.61)` →
`(3.64,0.61)` … until the skip hatch closed the run at `visited=0`. Frontier
repeated a target 8.7 times per scene on average.

Fix: an explicit `_rejected` list of failed centroids in world coordinates,
with a `reject_radius` (0.8 m) test applied at selection time — independent of
what the grid does with the underlying cells.

### 4. One barren tick ended the sweep as `no_frontiers`

7 of 19 frontier scenes died this way at 130–250 s. A single tick where every
cluster fell under `min_frontier_size` set `_done`. Terrain arrives in bursts;
that is noise, not coverage.

Fix: `empty_ticks_before_done` (20 ticks) before declaring done, and a fallback
to under-sized clusters — a 3-cell frontier is still unseen space.

### 5. `max_consecutive_skips` fired while the map was still growing

The hatch is meant to catch a stalled sweep, but it counted skips with no
reference to whether exploration was still making progress.

Fix: on hitting the limit, compare the free-cell count against the count at the
last advance. If the map has grown by `skip_reset_free_cells` (60 cells ≈
2.4 m²) the sweep is still learning — bank the progress, reset the counter, and
continue. Otherwise terminate as before.

### 6. Nbv detail fixes

* Skip blacklisted a single 0.2 m cell, so the next pick landed 20 cm away and
  failed identically. Now suppresses a disc.
* `_pick` drew 40 uniform samples from a pool where frontier cells were merely
  weighted ×3. Frontier cells are the only candidates carrying information
  gain, so they are now all scored nearest-first up to `n_samples`, with random
  free-space draws only filling the remainder.

## Instrumentation for the next sweep

Every number in this report is a proxy. Coverage was reconstructed from pose
bounding boxes, path length from poses sampled only at `WP_SET`/`WP_SKIP`, and
"was this the early skip or the stuck timeout?" from an `elapsed < 11 s`
heuristic. The next sweep should not need any of that:

| Added | Answers |
|---|---|
| `MAP` heartbeat every 10 s (`XIAO_HEI_EXPLORATION_MAP_LOG_S`) — `free`, `occupied`, `frontier`, `frontier_open`, `no_target`, `free_m2`, `reachable`, `path_m` | The coverage curve, and the gaps between waypoints. `reachable` falling away from `free` is a strategy walling itself in — the nbv failure, visible directly instead of inferred |
| `kind=no_progress\|stuck_timeout` on `WP_SKIP` | Which of the two skip paths fired, instead of guessing from `elapsed` |
| `best_odom_dist` on `WP_SKIP` | Whether the robot was closing on the target when it was abandoned |
| `path_m` accumulated every tick, on `MAP`/`WP_SKIP`/`DONE` | True path length; the old pose sampling undercounts worst on the scenes that drive most |
| `NO_TARGET` (throttled to one line per reason-change or 5 s) with `why=` | Why the selector came back empty — `all_suppressed` (over-blacklisted) vs `no_frontier_cells` (real coverage) vs `unreachable` demand opposite responses and were indistinguishable |
| `why=` + candidate counts on `WP_SET` | What the selector was choosing between |
| `hatch_resets`, `select_why`, and the final grid counters on `DONE` | A one-line summary per run |

Supporting changes: `scripts/exploration_sweep.sh` collates the new fields into
`results_<strategy>.csv` and now measures duration from `CLOCK_START` rather
than `START`, so the 90-190 s wait for `/state_estimation` is not billed as
exploration time. `scripts/compare_exploration.py` prints the per-strategy and
paired per-scene tables above from any log directory — it reads the old logs
too, marking reconstructed columns with `~`, and excludes runs that produced no
data (`frontier/chinese_room` is a `START` line and nothing else) so a crashed
container cannot lose a head-to-head.

## Tests

`tests/test_exploration_targeting.py` — 20 tests. One per failure above:
suppression leaves a corridor traversable while `mark_occupied` still blocks
it; a visited nbv waypoint does not disconnect the map behind it; neither
strategy re-proposes a skipped target; a barren tick does not end a sweep; the
skip hatch resets while free cells are still arriving. Then one per diagnostic
field, since a log line that lies is worse than no log line: `stats()`
separates a real obstacle (`reachable` halves) from target suppression
(`reachable` unchanged), and each `why=` value is pinned to the state it
names.

Full suite: 521 passed. (`tests/test_execute_plan.py` does not import in this
environment — `scripts/approach_loop.py` needs `cv2`, pre-existing.)

## Not done

Nothing was re-run against the simulator; every number above is reconstructed
from the two logged sweeps. The next sweep should run **both** strategies under
the same 480 s budget — frontier never had one, which is the single largest
confound in the table at the top. Then:

```
uv run python scripts/compare_exploration.py exploration_logs
```

Three things to check in the result, in order:

1. **`kind=` on the skips.** `no_progress` should have collapsed as a share of
   the total. If it has not, the odometry guard is not doing its job and
   `_WP_ODOM_STALL_S` is the knob.
2. **`reachable` vs `free` in the `MAP` lines,** for nbv especially. They
   should now track each other for the whole run. Divergence means something
   else is still eating the reachable set.
3. **`free_m2` per minute,** which is the first honest coverage number either
   strategy will have produced. If it disagrees with metres-per-minute about
   which strategy is better, believe `free_m2` — path length rewards a robot
   that drives in circles.
