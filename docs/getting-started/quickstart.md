# Quickstart

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker with NVIDIA Container Toolkit (for GPU mode)
- `xhost` (for simulator GUI forwarding)

## Install for development

```bash
git clone https://github.com/longgiang2015bn/xiao-hei-vln-cmu.git
cd xiao-hei-vln-cmu
uv sync
```

## Run tests

```bash
uv run pytest -q
```

No ROS or GPU required — all tests use pure Python with mock engines.

## Run the full stack (GPU)

```bash
# Allow X11 forwarding for the simulator GUI
xhost +local:

# Build and start all containers
docker compose -f docker/compose_gpu.yml up -d --build

# Wait for vLLM to be ready
docker logs -f xiao_hei_vllm
# Wait for: "Uvicorn running on http://0.0.0.0:8000"

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

## Run dummy mode (no GPU)

```bash
XIAO_HEI_RESPONDER=dummy docker compose -f docker/compose.yml up -d --build
```

The dummy responder always returns a fixed answer — useful for testing the infrastructure without a GPU.

## Next steps

- [Docker setup details](docker.md)
- [Configuration reference](configuration.md)
- [Plug in your own model](../guides/new-model.md)
