# Exploration Phase

Before any challenge question arrives, the system runs an autonomous
exploration phase to build up a map of the environment.  This page covers
how exploration fits into the tick loop, what data it consumes, and how
its lifecycle is managed.

## ExplorationStrategy protocol

Any object that implements three methods is a valid exploration strategy:

```python
class ExplorationStrategy(Protocol):
    def update(self, snapshot: VLMInput) -> Waypoint | None: ...
    def is_complete(self) -> bool: ...
    def reset(self) -> None: ...
```

`update()` is called once per tick and returns the next waypoint to
navigate to, or `None` when no waypoint should be published (e.g. waiting
for sensor data or exploration is done).  `is_complete()` signals that the
strategy has finished.  `reset()` wipes all internal state back to the
initial condition.

The protocol is exported as `ExplorationStrategy` from
`xiao_hei_vln.exploration` and is `@runtime_checkable`, so
`isinstance(obj, ExplorationStrategy)` works without inheriting from it.

## Inputs from VLMInput

The exploration strategy receives the same `VLMInput` snapshot as the
responder.  In practice, `FrontierExplorer` only reads three fields:

| `VLMInput` field | ROS topic | Used for |
|---|---|---|
| `terrain_ext` | `/terrain_map_ext` (20 m range) | Building the OccupancyGrid |
| `pose` | `/state_estimation` | Reach check, frontier scoring, stuck detection |
| `tick_time` | — (snapshot timestamp) | Stuck timeout elapsed-time tracking |

`terrain_ext` and `pose` may be `None` on early ticks before the
simulator has started publishing.  The explorer handles both gracefully:
it skips the grid update when `terrain_ext` is `None`, and returns the
current target unchanged when `pose` is `None`.

## Next-Best-View (`nbv`)

Set `XIAO_HEI_EXPLORATION_STRATEGY=nbv` to use `NextBestViewExplorer`
instead of frontier clustering.  It builds the same online
`OccupancyGrid` from `terrain_ext`, then repeatedly:

1. BFS reachable FREE cells from the robot pose.
2. Sample candidates (frontier-biased).
3. Score each as `unknown_gain / (1 + path_cost)`.
4. Drive to the best sample; blacklist visited cells with a small radius.

No ground-truth floor plan is used — only the live belief map.

## Output

Each call to `update()` returns a `Waypoint(x, y, heading)` or `None`.
When a waypoint is returned, the tick loop wraps it in a
`WaypointPathResponse` and publishes it to `/way_point_with_heading`
(`geometry_msgs/Pose2D`).  The nav stack then drives the robot toward
that goal.

## Where exploration runs

Exploration runs inside the `xiao_hei_ai_module` container — the same
Python process as the VLM responder.  No separate container or process is
involved.  The strategy object is created once at startup by
`_build_explorer()` in `app/main.py` and lives for the duration of the
run.

## Tick loop integration

Each 2 Hz tick the loop checks three conditions before deciding whether to
run exploration or the responder:

```
if explorer is not None       # exploration enabled
   and not explorer.is_complete()
   and snapshot.question is None:   # no active question
    → run exploration tick
    → return (responder is NOT called this tick)

# otherwise: run responder
```

The exploration block **returns early** — the responder is never called
on the same tick as exploration.

## Lifecycle

### Starting

Exploration starts automatically on the first tick where all three
conditions above are met.  In practice this means as soon as the
container is up and no question is active — typically within seconds of
`system_simulation.sh` being launched.  There is no manual trigger.

```
Container starts → first tick with no question → "Exploration started."
```

### Scene building on the fly

Exploration is not just movement — the scene graph is built *during* it.
On every exploration tick the loop also calls `scene.update(snapshot)`
(maintaining viewpoint and scene-bounds nodes) and, for responders that
expose it, `responder.ingest(snapshot)`. For the perception responder
`ingest()` runs the per-tick detect→lift→add_object cycle *without* emitting
an answer, so objects accumulate into the
[scene representation](../scene-representation.md) as the robot sweeps the
space while the explorer alone drives the waypoints. By the time exploration
completes, the scene graph already reflects everything the sweep saw, and the
responder answers from it directly (no separate trajectory walk needed).

`/state_estimation` takes 90–190 s to arrive after container start.
During that window `snapshot.pose` is `None` and the explorer publishes
its current target each tick without advancing state.

### Questions do not interrupt exploration

A challenge question arriving mid-exploration (`snapshot.question is not
None`) does **not** stop the sweep.  Exploration keeps running until the
strategy completes on its own terms (budget exhausted, consecutive-skip
hatch, or no frontiers remain).  The answer is **deferred**: the responder
only takes over once `explorer.is_complete()` is `True`, and by then it
answers from a scene graph that reflects the whole sweep — not just what
was visible when the question happened to arrive.

While the question is pending, its text still flows into the perception
detector's vocabulary (through `responder.ingest()`), so the queried object
is actively searched for during the remaining ticks.

### Stopping

Exploration ends when `is_complete()` returns `True`, or when the wall-clock
budget expires.  There are four reasons this can happen:

| Reason | Decided by | What happened |
|---|---|---|
| `budget_exhausted` | strategy | `N` waypoints visited, budget used up |
| `no_frontiers` | strategy | No frontier clusters remain — full coverage |
| `max_consecutive_skips` | strategy | Nav stack failed on too many targets in a row |
| `time_limit` | tick loop | `XIAO_HEI_EXPLORATION_MAX_SECONDS` elapsed (default 480 s) |

### The wall-clock cutoff

None of the strategy's own stop conditions is bounded in time, so a sweep that
keeps finding reachable frontiers in a large scene never hands over to the
responder.  `XIAO_HEI_EXPLORATION_MAX_SECONDS` (default **480 s = 8 minutes**,
`0` disables) puts a ceiling on it.

The clock starts on the first tick that has a pose, not at node boot:
`/state_estimation` takes 90-190 s to arrive and the robot cannot explore
before it does, so charging that dead time to the budget would silently
shorten it.  The tick that starts the clock logs `CLOCK_START`.

On expiry the loop logs `DONE  reason=time_limit` with the elapsed time,
saves `exploration.png` and `rviz.png`, and stops entering the exploration
branch — the responder takes over on that same tick's successor.  The strategy
object is never told; `is_complete()` stays `False` and its accumulated grid
remains readable.  Terminating through the same `DONE` event as every other
reason is deliberate: `scripts/vla3d_eval_sim.sh` polls the log for `" DONE "`,
so a timed-out sweep is detected exactly like a completed one.

The cutoff does **not** brake the robot.  The last commanded waypoint stays
with the nav stack until the responder publishes its own, which matches what
already happens when a sweep ends for any other reason.

After exploration ends, the tick loop falls through to the responder on
every subsequent tick.  The map built during exploration is not discarded
— it lives in the `FrontierExplorer` instance for the rest of the run
and is used to save the debug PNG (if configured).

On `DONE` the loop also saves two images into
`exploration_logs/<scene>/<strategy>/` (the scene name comes from
`XIAO_HEI_SCENE_DIR_HOST`, or `default_scene`):
`exploration.png` (the explorer's occupancy grid and visited waypoints) and
`rviz.png` (a screenshot of the simulator's RViz window — the traversed path
over the scene mesh).  The screenshot needs `DISPLAY` to be set and the X
socket mounted; without either it is skipped with an info log.  Both are
best-effort and never fail the run.

### Periodic snapshots

`DONE` is not guaranteed to arrive.  A sweep torn down mid-run, or one that
never terminates, would leave the text log but no images at all — exactly the
runs worth looking at.  So both PNGs are *also* re-written every
`XIAO_HEI_EXPLORATION_SNAPSHOT_S` seconds (default 30; `0` restores
end-of-sweep-only).  Each snapshot overwrites the last, so the file on disk is
always the newest view of the sweep, and the `DONE` save is simply the final
one.

Snapshots are written by a separate ROS timer, but rclpy's default executor is
single-threaded, so a snapshot never interleaves with a tick — the grid is read
between ticks, never mid-update.  Each file is written to a temp path and
renamed into place, so a reader tailing the directory never sees a half-written
PNG.  The timer stops firing once the strategy reports complete.

Two costs worth knowing.  Rendering the plot blocks the executor for its
duration, so the tick that coincides with a snapshot is delayed by roughly the
render time.  And the RViz grab raises the window before reading its pixels
(X11 without a compositor does not retain obscured regions), so an interactive
session will see RViz pop to the front on every interval — raise the interval,
or set it to `0`, if that is disruptive.

## Disabling exploration

Set `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0` to skip exploration entirely.
The tick loop then runs the responder from the very first tick.

## Structured log

Every exploration event is written to `exploration.log` in the configured
log directory (default: `/exploration_logs`, mounted to `exploration_logs/`
in the repo root).  See [Configuration](../getting-started/configuration.md)
for the env var.

| Event | When |
|---|---|
| `START` | Once, at the first exploration tick |
| `WP_SET` | New frontier target selected |
| `WP_ADVANCE` | Target marked visited (nav-stack or odometry) |
| `WP_SKIP` | Target abandoned (stuck timeout or early-skip) |
| `CLOCK_START` | First tick with a pose — the wall-clock budget starts here |
| `DONE` | Exploration over (`reason=` names which of the four) |
