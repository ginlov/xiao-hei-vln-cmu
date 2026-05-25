# Xiao Hei AI Module — Docker

This directory builds the container that hosts our Python VLM stack
(`xiao_hei_vln`) and connects it to the official CMU VLN Challenge
ROS 2 simulator. Today it runs a **dummy responder** ported from the
challenge reference `dummyVLM.cpp`; replacing the dummy with a real
VLM is a one-file change documented at the bottom.

## What's here

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `xiao-hei/ai_module:latest` by extending `zhangjicmu/ubuntu24_ros:ai_module`, installing `python3-pip`, then editable-installing this repo into the system Python 3.12 (same interpreter as `rclpy`). |
| `entrypoint.sh` | Sources ROS Jazzy and sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` before `exec`-ing the container CMD. |
| `compose.yml` / `compose_gpu.yml` | Bring up two services: `iros2026_system` (the unmodified challenge sim) and `xiao_hei_ai_module` (our image). Both share `network_mode: host` and cyclonedds. The GPU file adds NVIDIA reservations to both. |

## Build

```bash
cd docker
docker compose -f compose_gpu.yml build ai_module    # or compose.yml on CPU-only hosts
```

First build takes ~30 s on a warm cache (the heavy `ros-jazzy-desktop`
layer comes from the parent image). Subsequent builds skip everything
above the `COPY src` layer if only Python sources changed.

## Run the full stack end-to-end

```bash
xhost +local:                                                              # let containers reach your X server
docker compose -f docker/compose_gpu.yml up -d                             # bring up both containers

# Start the simulator (RViz opens on your host display)
docker exec -it iros2026_system /home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh

# Our VLM is already running. Tail its logs in a second terminal:
docker logs -f xiao_hei_ai_module

# Fire test questions from any container with ROS sourced:
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"How many cups\"}"'
# → publishes std_msgs/Int32 on /numerical_response

docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"Find the red cup\"}"'
# → publishes visualization_msgs/Marker on /selected_object_marker

docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"Take the path near the window\"}"'
# → publishes geometry_msgs/Pose2D on /way_point_with_heading; vehicle advances through waypoints.ply

# Tear down
docker compose -f docker/compose_gpu.yml down
```

## How code lands inside the container

- **Baked**: `COPY src ./src` + `pip install -e .` puts our package at
  `/opt/xiao_hei_vln/src/xiao_hei_vln/`. The console script
  `xiao-hei-dummy-vlm` is installed into `/usr/local/bin/`.
- **Bind-mounted** (dev): the compose files mount `../src` over
  `/opt/xiao_hei_vln/src:ro`. Because the install is editable, code
  changes on your host are visible the next time the container starts
  (`docker compose restart ai_module`). Comment that volume out for
  "what the submission will see" runs.
- **Env knobs**: `XIAO_HEI_VLM_TICK_HZ` overrides the 2 Hz default;
  `RMW_IMPLEMENTATION` defaults to `rmw_cyclonedds_cpp` to match the
  challenge stack.

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

## Replacing the dummy with the real VLM later

Only **`src/xiao_hei_vln/app/main.py`** needs to change: swap
`DummyResponder()` for the real model. Everything else stays put:

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
| `docker logs xiao_hei_ai_module` shows the "ready" line but `ros2 topic info /challenge_question` reports `Subscription count: 0` | Containers using different DDS implementations | Confirm both have `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` set (visible via `docker exec <c> env | grep RMW`). |
| Publisher sends a response but `ros2 topic echo --once` returns nothing | Late-subscriber + VOLATILE QoS — `echo --once` started after our publish | Run `ros2 topic echo /<topic>` *before* publishing the question. |
| `pip install -e .` fails complaining about NumPy uninstall | NumPy from apt has no RECORD file | Keep `numpy>=1.26` in `pyproject.toml` (already done) so the apt-installed version satisfies the requirement. |
