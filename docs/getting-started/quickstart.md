# Quickstart

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker with NVIDIA Container Toolkit (for GPU mode)
- `xhost` (for simulator GUI forwarding)

## Install for development

```bash
git clone https://github.com/ginlov/xiao-hei-vln-cmu.git
cd xiao-hei-vln-cmu
uv sync
```

## Run tests

```bash
uv run pytest -q
```

No ROS or GPU required — all tests use pure Python with mock engines.

## Run the full stack

Pick the responder with one env var; `docker/run` handles compose profiles
and validates prerequisites.

```bash
# Allow X11 forwarding for the simulator GUI
xhost +local:

# Build and start all containers (the perception sidecar auto-starts
# under XIAO_HEI_RESPONDER=perception)
XIAO_HEI_RESPONDER=perception docker/run up -d --build

# Wait for the sidecar to load its models
docker logs -f xiao_hei_perception
# Wait for: "Uvicorn running on http://0.0.0.0:8001"

# Start the simulator
docker exec -it iros2026_system \
  /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# Watch AI module logs
docker logs -f xiao_hei_ai_module

# Send a test question
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && \
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question \
   std_msgs/msg/String "{data: \"How many chairs are in the room?\"}"'
```

## Run dummy mode (no sidecars)

```bash
XIAO_HEI_RESPONDER=dummy docker/run up -d --build
```

The dummy responder always returns a fixed answer — useful for testing
infrastructure without a model. No sidecar starts, because no profile is
activated.

## Run the perception responder (real models)

`XIAO_HEI_SCENE_DIR_HOST` drives the simulator (`system`) — it loads the
Unity scene from the extracted dir — and the perception sidecar
(YOLOv8x-World v2 + SAM 2.1 Hiera Tiny) runs real detection per tick:

```bash
SCENES=/path/to/CMU-VLN-Challenge-data/unity_env_models
unzip -oq $SCENES/arabic_room.zip -d $SCENES/

export XIAO_HEI_SCENE_DIR_HOST=$SCENES/arabic_room                # for system (Unity)
export XIAO_HEI_TRAJECTORY_JSON_HOST=$PWD/trajectories/arabic_room.json   # optional
XIAO_HEI_RESPONDER=perception docker/run up -d --build
```

First boot pulls + builds the perception image (~5 GB, mostly torch +
CUDA from the `ultralytics/ultralytics` base).

For the full submission stack (frontier exploration + perception + Gemini),
use `docker/compose_scene_gemini.yml` instead — see the
[project README](https://github.com/ginlov/xiao-hei-vln-cmu#4b-run-end-to-end--the-submission-stack-scene_gemini).

See [Perception (Sidecar + Responder)](../perception-sidecar.md) for the
full architecture, configuration, and tuning options.

## Next steps

- [Docker setup details](docker.md)
- [Configuration reference](configuration.md)
- [Plug in your own model](../guides/new-model.md)
