# Configuration

All runtime configuration is done via environment variables. No config files
are required — docker-compose sets them for you, but they can be overridden.

## Core variables

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `qwen` (GPU) / `dummy` (CPU) | Which responder implementation to use |
| `XIAO_HEI_VLM_TICK_HZ` | `2.0` | VLM tick rate in Hz |

## Qwen engine

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_QWEN_VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM sidecar URL. Empty string = in-process mode |
| `XIAO_HEI_QWEN_MODEL` | `/models/Qwen3.5-4B` | Model path (local) or HuggingFace ID |
| `XIAO_HEI_QWEN_DTYPE` | `bfloat16` | Model dtype for vLLM |
| `XIAO_HEI_QWEN_MAX_MODEL_LEN` | `4096` | Maximum context length |
| `XIAO_HEI_QWEN_GPU_MEM_UTIL` | `0.85` | GPU memory fraction for vLLM |
| `XIAO_HEI_QWEN_TRUST_REMOTE` | `1` | Trust remote code (HuggingFace models) |

## Sampling parameters

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_QWEN_TEMPERATURE` | `0.0` | Sampling temperature (0 = greedy) |
| `XIAO_HEI_QWEN_MAX_OUTPUT_TOKENS` | `256` | Maximum tokens per response |
| `XIAO_HEI_QWEN_SEED` | `0` | Random seed for reproducibility |
| `XIAO_HEI_QWEN_IMAGE_LONG_EDGE` | `1280` | Downscale target for camera images |

## Responder loop

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_QWEN_MAX_TICKS` | `30` | Max ticks per question before timeout |

## Exploration

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_EXPLORATION_STRATEGY` | `frontier` | Exploration algorithm to use. Only `frontier` is currently implemented; unknown values disable exploration with an error log |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS` | `100` | Waypoint budget. Set to `0` to disable exploration entirely and go straight to question answering |
| `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST` | `1.5` | Preferred maximum distance (metres) to a frontier target. Closer targets score higher; if all exceed this cap the nearest valid one is used as a fallback |
| `XIAO_HEI_EXPLORATION_LOG_DIR` | `/exploration_logs` (GPU compose) | Directory for `exploration.log` and the debug PNG. The text log is always written (falls back to `/exploration_logs`); the PNG is only saved when this variable is explicitly set |

## Logging

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_VLM_LOG_DIR` | `/vlm_logs` (GPU compose) | Directory for VLM tick logs. Unset = logging disabled |

## Authentication

| Variable | Default | Description |
|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | (unset) | HuggingFace token for gated model downloads |

## Docker build-time

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_EXTRA` | `qwen` (GPU) | pip optional extra to install at build time |

## Configuration in code

All Qwen-related variables are loaded by `QwenConfig.from_env()`:

```python
from xiao_hei_vln.qwen import QwenConfig

config = QwenConfig.from_env()
print(config.model)           # /models/Qwen3.5-4B
print(config.vllm_base_url)   # http://localhost:8000/v1
print(config.temperature)     # 0.0
```
