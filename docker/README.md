# Xiao Hei AI Module — Docker

This directory builds the container that hosts our Python VLM stack
(`xiao_hei_vln`) and connects it to the official CMU VLN Challenge
ROS 2 simulator.

## TL;DR

Pick the responder with one environment variable; `docker/run` does the rest.

```bash
# Dummy responder (no sidecars)
XIAO_HEI_RESPONDER=dummy docker/run up -d

# Perception responder (YOLO-World + SAM 2.1 sidecar) — one scene env var
# drives the sim (Unity scene); optional trajectory walks coverage in Phase A.
SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room              # for system (Unity)
export XIAO_HEI_TRAJECTORY_JSON_HOST=$PWD/trajectories/arabic_room.json  # optional
XIAO_HEI_RESPONDER=perception docker/run up -d
```

All trailing args to `docker/run` are forwarded to `docker compose`:
`docker/run logs -f ai_module`, `docker/run down`, `docker/run build ai_module`, etc.

> **Switching scenes**: extract `<scene>.zip` once on the host (e.g.
> `unzip -o arabic_room.zip -d CMU-VLN-Challenge-data/unity_env_models/`)
> then set `XIAO_HEI_SCENE_DIR_HOST`. `docker/run up -d` swaps the bind
> mount; no more `docker cp` required.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `xiao-hei/ai_module:latest` by extending `zhangjicmu/ubuntu24_ros:ai_module` and editable-installing this repo into the system Python 3.12. Default build installs the lightweight `perception` extra (`httpx` + `pillow` + `pycocotools`) so the same image serves both responders. |
| `entrypoint.sh` | Sources ROS Jazzy and sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` before `exec`-ing the container CMD. |
| `compose.yml` | Single unified stack: `system` + `ai_module` always, `perception` only under `--profile perception`. Conditional trajectory bind mount via a host-side env var. |
| `run` | Wrapper script. Maps `XIAO_HEI_RESPONDER` to the right `--profile`, layers `compose.scene.yml` when `XIAO_HEI_SCENE_DIR_HOST` is set, forwards everything else to `docker compose`. |

## How the env-var-driven selection works

| `XIAO_HEI_RESPONDER` | Containers started | Extra env vars needed |
|---|---|---|
| `dummy` (default) | `system`, `ai_module` | — |
| `perception` | `system`, `ai_module`, `perception` | optional `XIAO_HEI_SCENE_DIR_HOST` to pick a sim scene. Runs YOLOv8x-World v2 + SAM 2.1 Hiera Tiny in the sidecar; the responder lifts each mask through the LiDAR scan and pushes detections into the scene graph. See [Perception (Sidecar + Responder)](../docs/perception-sidecar.md). |

The `perception` sidecar is profile-gated (`profiles: [perception]` in `compose.yml`), so a dummy run never instantiates it.

If you'd rather call `docker compose` directly:

```bash
# Equivalent to XIAO_HEI_RESPONDER=perception docker/run up -d
XIAO_HEI_RESPONDER=perception docker compose -f docker/compose.yml --profile perception up -d
```

## Build

```bash
docker/run build ai_module
```

## Full run with the perception sidecar

```bash
xhost +local:
XIAO_HEI_RESPONDER=perception docker/run up -d

# Wait for the sidecar to load YOLO-World + SAM weights.
docker logs -f xiao_hei_perception
# look for: "Uvicorn running on http://0.0.0.0:8001"

# Start the simulator (RViz opens on your host display)
docker exec -it iros2026_system /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# Tail our VLM
docker logs -f xiao_hei_ai_module

# Fire a test question
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"How many cups\"}"'

docker/run down
```

## Full run with the perception responder

The [perception responder](../docs/perception-sidecar.md) runs
YOLOv8x-World v2 + SAM 2.1 in a sidecar, lifts each detection to 3D
through the LiDAR scan, and answers from the live scene graph. Use it to
exercise the full pipeline (sim → detect → scene rep → logs → report).

```bash
# 1. Extract the scene zip once on the host (bind-mounted into the sim).
SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
unzip -oq $SCENES/arabic_room.zip -d $SCENES/

# 2. (Optional but recommended) Pre-plan a coverage trajectory so the
#    robot actually moves and the scene graph grows over the run.
uv run python -m xiao_hei_vln.trajectory $SCENES/arabic_room.zip --out trajectories

# 3. Point at the extracted dir (for the sim) + trajectory, then start.
#    --profile perception is activated by the wrapper automatically.
export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room
export XIAO_HEI_TRAJECTORY_JSON_HOST=$PWD/trajectories/arabic_room.json
XIAO_HEI_RESPONDER=perception docker/run up -d

# 4. Launch Unity in the sim container.
docker exec -it iros2026_system \
  /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# 5. Fire a question.
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String \
     "{data: \"How many chairs are in the scene?\"}"'
```

To switch scenes later, just change `XIAO_HEI_SCENE_DIR_HOST` (and the
trajectory) and run `docker/run up -d` again — the bind mount swaps, no
`docker cp`.

## Architecture: why sidecars

Heavy models run in **their own container**, never inside `ai_module`.
The perception sidecar ships its own CUDA/torch stack and exposes an
HTTP API on `localhost:8001`; `ai_module` calls it with the lightweight
`httpx` client. No CUDA deps are installed in the ai_module image, which
completely avoids the pip/apt package conflicts with the ROS base image.

The submission stack applies the same split to reasoning: `scene_gemini`
calls the Gemini API over HTTPS, so no inference weights live in
`ai_module` at all. See `compose_scene_gemini.yml`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `XIAO_HEI_RESPONDER` | `dummy` | Which responder to use: `dummy`, `perception`. The `docker/run` wrapper maps `perception` to `--profile perception` (starts the YOLO+SAM sidecar). The submission responder `scene_gemini` has its own compose file. |
| `XIAO_HEI_PERCEPTION_BASE_URL` | `http://localhost:8001` | Perception sidecar URL the responder talks to. |
| `XIAO_HEI_PERCEPTION_SCORE_THRESHOLD` | `0.25` | Forwarded to YOLO-World on every `/detect`. Lower → more detections, more noise. |
| `XIAO_HEI_PERCEPTION_MIN_INLIERS` | `10` | LiDAR-return count below which a detection mask is dropped (no 3D point committed). |
| `PERCEPTION_DEBUG` | (unset) | Set on the `perception` sidecar to dump per-step images (equirect → faces → bboxes → masks → reprojected equirect) to `perception/debug/` for each `/detect`. |
| `XIAO_HEI_VLM_LOG_DIR` | `/vlm_logs` | Directory for VLM tick logs. When set, every tick is logged to JSONL + JPEG. |
| `XIAO_HEI_VLM_TICK_HZ` | `2.0` | VLM tick rate in Hz. |
| `XIAO_HEI_SCENE_DIR_HOST` | (unset) | **Optional, any responder**: path to the *extracted* scene directory on the host (e.g. `…/unity_env_models/arabic_room`). When set, the wrapper layers `compose.scene.yml` to bind-mount `<dir>/environment/` over the sim's prebaked Unity environment, so the simulator loads that scene. Without it, the sim falls back to whatever scene is baked into the image. |
| `XIAO_HEI_TRAJECTORY_JSON_HOST` | (unset) | Host path to a pre-planned Task 7 trajectory JSON; bind-mounted as `/data/trajectory.json` inside `ai_module`. Walked in Phase A by the perception responder. Without it, the responder answers from tick 0 without moving. |
| `XIAO_HEI_EXTRA` | `perception` | pip extra to install at image build time. |

## Publish to Docker Hub (for the challenge submission)

```bash
docker tag xiao-hei/ai_module:latest <your-dockerhub-user>/xiao-hei-ai-module:latest
docker push <your-dockerhub-user>/xiao-hei-ai-module:latest
```

Then provide the Docker Hub link in the [Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScdZAcw5S2nbfSKn8qB-kmNC3PEEQHTK64dU9Hqb5iKg0_jtA/viewform).

## Drop into the challenge repo

In `CMU-VLN-Challenge-2026/docker/compose.yml`, replace the `ai_module:` block with:

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

The container auto-starts the VLM on `docker compose up` — no need to
`docker exec` and `ros2 launch` like the reference C++ dummy required.

## Replacing with another VLM later

Only **`src/xiao_hei_vln/app/main.py`** needs to change: add a new
branch to `_build_responder()` returning anything that implements
`respond / is_done / reset`. Everything else stays put:

- The topic contract (`bind_subscribers`, `VLMOutputPublisher`)
- `LatestCache` and the 2 Hz tick
- The Docker image (just rebuild)
- The compose file and env wiring

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ai_module` blocks at startup waiting on the perception sidecar | sidecar not started, or weights still downloading | Confirm you set `XIAO_HEI_RESPONDER=perception` (so `--profile perception` activates). Check `docker logs xiao_hei_perception`. |
| ai_module ready but `ros2 topic info /challenge_question` shows `Subscription count: 0` | DDS mismatch | Confirm both have `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (`docker exec <c> env \| grep RMW`). |
| Publisher sends a response but `ros2 topic echo --once` returns nothing | Late-subscriber + VOLATILE QoS | Run `ros2 topic echo /<topic>` *before* publishing the question. |
| Perception responder answers from the start pose without moving | No coverage trajectory provided | Set `XIAO_HEI_TRAJECTORY_JSON_HOST` to a pre-planned Task 7 trajectory JSON (see the full-run section above). |

## VLM tick logging

When `XIAO_HEI_VLM_LOG_DIR` is set (default `/vlm_logs` inside the container,
bind-mounted to `../vlm_logs` on the host), every VLM tick is recorded for
post-run debugging:

```
vlm_logs/
  session_20260530_143022/
    session.json                  # Config snapshot (model, tick_hz, etc.)
    predictions.jsonl             # Final answer per question (for offline eval)
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

# Offline evaluation (after a run)
python -m xiao_hei_vln.eval_pipeline \
    --gt dataset/vla3d_num.jsonl \
    --pred vlm_logs/session_20260530_143022/predictions.jsonl \
    --out results.json
```

To disable logging, unset the env var:

```bash
XIAO_HEI_VLM_LOG_DIR="" XIAO_HEI_RESPONDER=perception docker/run up -d
```
