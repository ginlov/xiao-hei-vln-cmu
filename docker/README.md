# Xiao Hei AI Module — Docker

This directory builds the container that hosts our Python VLM stack
(`xiao_hei_vln`) and connects it to the official CMU VLN Challenge
ROS 2 simulator.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `xiao-hei/ai_module:latest` by extending `zhangjicmu/ubuntu24_ros:ai_module`, installing `python3-pip`, then editable-installing this repo into the system Python 3.12 (same interpreter as `rclpy`). |
| `entrypoint.sh` | Sources ROS Jazzy and sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` before `exec`-ing the container CMD. |
| `compose.yml` | CPU-only / dummy-responder stack: `iros2026_system` + `xiao_hei_ai_module`. No GPU, no vLLM. |
| `compose_gpu.yml` | Full GPU stack: `iros2026_system` + `xiao_hei_vllm` (vLLM sidecar) + `xiao_hei_ai_module`. |

## Build

```bash
# Build the ai_module image (installs openai + pillow for the HTTP sidecar)
docker compose -f docker/compose_gpu.yml build ai_module
```

## Run the full stack with Qwen3.5 (sidecar mode)

```bash
xhost +local:
docker compose -f docker/compose_gpu.yml up -d

# The vLLM sidecar downloads Qwen3.5-4B weights on first boot (~8 GB).
# Watch its logs to see when it's ready:
docker logs -f xiao_hei_vllm
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

# Start the simulator (RViz opens on your host display)
docker exec -it iros2026_system /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# Our VLM auto-starts once vLLM is reachable. Tail its logs:
docker logs -f xiao_hei_ai_module

# Fire test questions from any container with ROS sourced:
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"How many cups\"}"'

# Tear down
docker compose -f docker/compose_gpu.yml down
```

## Run dummy-only (no GPU)

```bash
XIAO_HEI_RESPONDER=dummy docker compose -f docker/compose.yml up -d
```

## Architecture: sidecar vs in-process

**Sidecar (default, recommended)**: the `compose_gpu.yml` stack runs
three containers. The `vllm` service uses the official
`vllm/vllm-openai` image and exposes an OpenAI-compatible API at
`localhost:8000`. The `ai_module` calls it via HTTP using the
lightweight `openai` Python SDK. No CUDA deps are installed in the
ai_module image — this completely avoids the pip/apt package conflicts
with the ROS base image.

**In-process (legacy)**: set `XIAO_HEI_QWEN_VLLM_BASE_URL=""` and
build with `XIAO_HEI_EXTRA=qwen-local` to load vLLM directly inside
the ai_module container. This path requires manually resolving
pip/apt conflicts in the Dockerfile and is not recommended.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `qwen` (GPU compose) / `dummy` (CPU compose) | Which responder to use |
| `XIAO_HEI_QWEN_VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL. Unset for in-process mode |
| `XIAO_HEI_QWEN_MODEL` | `/models/Qwen3.5-4B` | Model path (local) or HuggingFace ID |
| `XIAO_HEI_QWEN_DTYPE` | `bfloat16` | Model dtype (vLLM server arg) |
| `XIAO_HEI_QWEN_MAX_MODEL_LEN` | `4096` | Max context length |
| `XIAO_HEI_QWEN_GPU_MEM_UTIL` | `0.85` | GPU memory fraction for vLLM |
| `XIAO_HEI_VLM_LOG_DIR` | `/vlm_logs` (GPU compose) / (unset, CPU) | Directory for VLM tick logs. When set, every tick is logged to JSONL + JPEG. |
| `XIAO_HEI_VLM_TICK_HZ` | `2.0` | VLM tick rate in Hz |
| `HUGGING_FACE_HUB_TOKEN` | (unset) | HF token if model weights are gated |
| `XIAO_HEI_EXTRA` | `qwen` (GPU compose) / (empty, CPU compose) | pip extra to install at build time |

## Publish to Docker Hub (for the challenge submission)

```bash
docker tag xiao-hei/ai_module:latest <your-dockerhub-user>/xiao-hei-ai-module:latest
docker push <your-dockerhub-user>/xiao-hei-ai-module:latest
```

Then in the [Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScdZAcw5S2nbfSKn8qB-kmNC3PEEQHTK64dU9Hqb5iKg0_jtA/viewform)
provide the Docker Hub link.

## Drop into the challenge repo

In `CMU-VLN-Challenge-2026/docker/compose.yml` (or `compose_gpu.yml`),
replace the `ai_module:` block with:

```yaml
  ai_module:
    image: <your-dockerhub-user>/xiao-hei-ai-module:latest
    container_name: iros2026_ai_module
    network_mode: host
    environment:
      - RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    stdin_open: true
    tty: true
```

The container will auto-start the VLM on `docker compose up` — no
need to `docker exec` and `ros2 launch` like the reference C++ dummy
required.

## Replacing with another VLM later

Only **`src/xiao_hei_vln/app/main.py`** needs to change: add a new
branch to `_build_responder()` returning anything that implements
`respond / is_done / reset`. Everything else stays put:

- The topic contract (`bind_subscribers`, `VLMOutputPublisher`)
- `LatestCache` and the 2 Hz tick
- The Docker image (just rebuild after the source change)
- The compose files and the env wiring

A real VLM that produces a `VLMOutput` per tick will plug straight
into the existing publisher and route to the right topic via the
discriminated union — no new code paths required.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker logs xiao_hei_ai_module` shows "Waiting for vLLM server…" indefinitely | vLLM container not started, or model download stalled | Check `docker logs xiao_hei_vllm` for download progress or OOM errors. |
| vLLM OOM on model load | Model too large for GPU | Use a smaller model (e.g. `XIAO_HEI_QWEN_MODEL=/models/Qwen3.5-2B`) or lower `XIAO_HEI_QWEN_GPU_MEM_UTIL`. |
| ai_module ready but `ros2 topic info /challenge_question` shows `Subscription count: 0` | DDS mismatch | Confirm both have `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (`docker exec <c> env \| grep RMW`). |
| Publisher sends a response but `ros2 topic echo --once` returns nothing | Late-subscriber + VOLATILE QoS | Run `ros2 topic echo /<topic>` *before* publishing the question. |

## VLM tick logging

When `XIAO_HEI_VLM_LOG_DIR` is set (default `/vlm_logs` in the GPU
compose), every VLM tick is recorded for post-run debugging:

```
vlm_logs/
  session_20260530_143022/
    session.json                  # Config snapshot (model, tick_hz, etc.)
    q_001_how_many_chairs/
      ticks.jsonl                 # One JSON line per tick
      images/
        tick_000003.jpg
      pointclouds/                # Lidar/terrain .npy arrays per tick
        tick_000003_registered.npy
        tick_000003_terrain_local.npy
        tick_000003_terrain_ext.npy
      report.html                 # Self-contained HTML report (generated)
    q_002_find_the_red_cup/
      ticks.jsonl
      images/
        tick_000007.jpg
```

The `compose_gpu.yml` bind-mounts `../vlm_logs` into the container, so
logs appear on the host automatically.

### Inspecting logs

```bash
# Text summary
python scripts/replay_session.py vlm_logs/session_20260530_143022

# Filter by question
python scripts/replay_session.py vlm_logs/session_20260530_143022 -q chairs

# Generate HTML reports (camera playback, pose trajectory, sensor BEV,
# per-tick I/O, latency charts)
pip install xiao-hei-vln[replay]  # adds matplotlib
python scripts/generate_report.py vlm_logs/session_20260530_143022/
open vlm_logs/session_20260530_143022/q_001_*/report.html

# Single question report
python scripts/generate_report.py vlm_logs/session_*/ -q chairs
```

To disable logging, unset the env var:

```bash
XIAO_HEI_VLM_LOG_DIR="" docker compose -f docker/compose_gpu.yml up -d
```
