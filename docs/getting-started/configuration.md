# Configuration

All runtime configuration is done via environment variables. No config files
are required — docker-compose sets them for you, but they can be overridden.

## Core variables

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `dummy` | Which responder to use: `dummy`, `perception`, or `scene_gemini` (the submission stack) |
| `XIAO_HEI_VLM_TICK_HZ` | `2.0` | VLM tick rate in Hz |

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
| `XIAO_HEI_OBJECT_MAP` | (off) | `1` fuses detections into converged 3D boxes (NMS + wall-sheet rejection) |
| `XIAO_HEI_SCAN_KEYFRAMES` | `10` | Keyframes accumulated to densify the sparse single sweep before lifting |
| `XIAO_HEI_SCAN_MIN_MOVE_M` | `0.25` | Minimum translation (m) between accumulated keyframes |
| `XIAO_HEI_SCAN_MIN_ROT_DEG` | `15` | Minimum rotation (deg) between accumulated keyframes |
| `XIAO_HEI_SCAN_VOXEL_M` | `0.05` | Voxel size (m) for downsampling the accumulated scan |

## Exploration

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_EXPLORATION_STRATEGY` | `frontier` | Exploration algorithm to use. Only `frontier` is currently implemented; unknown values disable exploration with an error log |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` | `100` | Waypoint budget. Set to `0` to disable exploration entirely and go straight to question answering. Exploration runs to completion (budget / skips / no frontiers) even if a question arrives first — the answer is deferred until it finishes |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST` | `1.5` | Preferred maximum distance (metres) to a frontier target. Closer targets score higher; if all exceed this cap the nearest valid one is used as a fallback |
| `XIAO_HEI_EXPLORATION_LOG_DIR` | `/exploration_logs` (GPU compose) | Directory for `exploration.log` and the debug PNG. The text log is always written (falls back to `/exploration_logs`); the PNG is only saved when this variable is explicitly set |

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
