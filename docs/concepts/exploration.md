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

Exploration ends when `is_complete()` returns `True`.  There are three
reasons this can happen:

| Reason | `exploration.log` field | What happened |
|---|---|---|
| `budget_exhausted` | `visited=N skipped=M` | `N` waypoints visited, budget used up |
| `no_frontiers` | `visited=N skipped=M` | No frontier clusters remain — full coverage |
| `max_consecutive_skips` | `visited=N skipped=M` | Nav stack failed on too many targets in a row |

After exploration ends, the tick loop falls through to the responder on
every subsequent tick.  The map built during exploration is not discarded
— it lives in the `FrontierExplorer` instance for the rest of the run
and is used to save the debug PNG (if configured).

On `DONE` the loop also saves two images into `exploration_logs/<scene>/`
(the scene name comes from `XIAO_HEI_SCENE_DIR_HOST`, or `default_scene`):
`exploration.png` (the explorer's occupancy grid and visited waypoints) and
`rviz.png` (a screenshot of the simulator's RViz window — the traversed path
over the scene mesh).  The screenshot needs `DISPLAY` to be set and the X
socket mounted; without either it is skipped with an info log.  Both are
best-effort and never fail the run.

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
| `DONE` | Exploration complete |
