# Configuration

All runtime configuration is done via environment variables. No config files
are required — docker-compose sets them for you, but they can be overridden.

## Core variables

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `dummy` | Which responder to use: `dummy`, `perception`, or `scene_gemini` (the submission stack) |
| `XIAO_HEI_VLM_TICK_HZ` | `1.0` | VLM tick rate in Hz |

## Gemini engine

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_GEMINI_API_KEY` | **(required)** | API key. `GeminiEngine.warmup()` validates it at boot, so an invalid key crash-loops the container |
| `XIAO_HEI_GEMINI_MODEL` | `gemini-2.5-flash` | Model id |
| `XIAO_HEI_GEMINI_TEMPERATURE` | `0.2` | Sampling temperature |
| `XIAO_HEI_GEMINI_MAX_OUTPUT_TOKENS` | `2048` | Response token cap |
| `XIAO_HEI_GEMINI_THINKING_BUDGET` | `0` | Thinking tokens. `0` disables (keeps the JSON answer from being truncated); `-1` = dynamic |
| `XIAO_HEI_GEMINI_IMAGE_LONG_EDGE` | `1280` | Downscale long edge before send |

## Responder loop

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_GEMINI_MAX_EXPLORE_TICKS` | `120` | Ticks without a reachable frontier before committing to an answer (60 s at 2 Hz) |
| `XIAO_HEI_GEMINI_MAX_TICKS` | `240` | Hard cap on ticks per question |

## Perception sidecar

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_PERCEPTION_BASE_URL` | `http://localhost:8001` | Sidecar URL the responder talks to |
| `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` | `0.25` | YOLO-World detection score gate. Lower → more detections, more noise |
| `XIAO_HEI_PERCEPTION_MIN_INLIERS` | `10` | LiDAR-return count below which a detection mask is dropped |
| `XIAO_HEI_SCAN_KEYFRAMES` | `10` | Sweeps accumulated to densify the sparse single sweep before lifting. One per tick, so this is a window in *ticks* — 10 ≈ 5 s at the 2 Hz default |
| `XIAO_HEI_SCAN_VOXEL_M` | `0.05` | Voxel size (m) for downsampling the accumulated scan |

## Exploration

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_EXPLORATION_STRATEGY` | `frontier` | Exploration algorithm: `frontier` (cluster the frontier, score by size/distance) or `nbv` (sample reachable poses, score by unknown-gain/path-cost). Also names the artefact subdirectory. Unknown values disable exploration with an error log — they do **not** fall back to `frontier` |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` | `500` | Waypoint budget. Set to `0` to disable exploration entirely and go straight to question answering. Exploration runs to completion (budget / skips / no frontiers) even if a question arrives first — the answer is deferred until it finishes |
| `XIAO_HEI_EXPLORATION_MAX_SECONDS` | `480` (8 min) | Wall-clock cutoff for the sweep. The clock starts on the first tick with a pose — not at node boot, since `/state_estimation` takes 90-190 s to arrive and the robot cannot explore before it does. On expiry the loop logs `DONE reason=time_limit`, saves both PNGs, and hands over to the responder. `0` disables |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST` | `1.5` | Preferred maximum distance (metres) to a frontier target. Closer targets score higher; if all exceed this cap the nearest valid one is used as a fallback |
| `XIAO_HEI_EXPLORATION_LOG_DIR` | `/exploration_logs` (GPU compose) | Base directory for the run's artefacts. The run lands in `<log dir>/<scene>/<strategy>/` — scene from the basename of `XIAO_HEI_SCENE_DIR_HOST` (`default_scene` when no scene is mounted), strategy from `XIAO_HEI_EXPLORATION_STRATEGY` (`no_exploration` when the waypoint budget is 0). Keying on both means neither a second scene nor a second algorithm clobbers an earlier run. The text log is always written; the two PNGs only when this variable is set |
| `XIAO_HEI_EXPLORATION_SNAPSHOT_S` | `30` | How often (seconds) `exploration.png` and `rviz.png` are re-written while the sweep is still running, so a run killed before `DONE` still leaves images. Each snapshot overwrites the last. Set to `0` to save only at the end |
| `XIAO_HEI_EXPLORATION_MAP_LOG_S` | `10` | How often (seconds) a `MAP` heartbeat line goes into `exploration.log`: elapsed, robot pose, path length, and the grid counters (`free`, `occupied`, `frontier`, `frontier_open`, `no_target`, `free_m2`, `reachable`). This is the coverage curve — `WP_*` events alone record decisions, not how much of the scene was seen. Watch for `reachable` falling away from `free`: that is the strategy walling itself off from its own map. Set to `0` to disable |
| `XIAO_HEI_SCENE_DIR_HOST` | (unset) | Host path of the extracted scene. Mounts the scene into the simulator, and its basename names the log dir — `.../chinese_room` → `exploration_logs/chinese_room/<strategy>/`. Passed to `ai_module` by every compose file |
| `DISPLAY` | (unset) | X display the simulator renders RViz into. When set (and the X socket is mounted), the node screenshots the RViz window to `rviz.png` as exploration finishes. When unset — a headless host, or the challenge submission — the screenshot is skipped with an info log. Needs `xhost +local:` on the host, since the containers run as a different user |

The exploration phase writes three artefacts into
`exploration_logs/<scene>/<strategy>/` — the log continuously, the two PNGs
every `XIAO_HEI_EXPLORATION_SNAPSHOT_S` seconds and again on `DONE`:

| File | What it is |
|---|---|
| `exploration.log` | Structured event log (`START` / `WP_SET` / `WP_ADVANCE` / `WP_SKIP` / `DONE`) |
| `exploration.png` | The explorer's own view: occupancy grid + visited waypoints |
| `rviz.png` | The simulator's view: the traversed path over the scene mesh, as RViz drew it |

The screenshot is strictly best-effort. A missing display, a missing
`python-xlib`, or an RViz that never opened all produce a warning and nothing
more — exploration has already finished by then, so a lost debug image never
fails the run.

## Logging

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_VLM_LOG_DIR` | `/vlm_logs` (GPU compose) | Directory for VLM tick logs. Unset = logging disabled |

## Docker build-time

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_EXTRA` | `perception` | pip optional extra to install at build time. The submission image uses `perception,gemini,exploration` |

## Configuration in code

All Gemini-related variables are loaded by `GeminiConfig.from_env()`, which
raises if the API key is missing:

```python
from xiao_hei_vln.gemini import GeminiConfig

config = GeminiConfig.from_env()
print(config.model)         # gemini-2.5-flash
print(config.temperature)   # 0.2
```
