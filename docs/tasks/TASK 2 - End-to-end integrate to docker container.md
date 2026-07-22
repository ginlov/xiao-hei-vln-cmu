# TASK 2 — End-to-end integrate to docker container

## Goal

Package the Task-1 Python contract into a docker image that runs a
working VLM (a dummy for now, the real model later) against the
challenge ROS 2 simulator. The container is built to drop straight
into the challenge `docker/compose.yml` in place of the reference
`iros2026_ai_module` — same DDS, same network mode, same topic
contract.

## What was done

### Phase 1 — Investigate the challenge container

Reviewed in Task 1: `CMU-VLN-Challenge-2026/ai_module/docker/Dockerfile`
extends `zhangjicmu/ubuntu24_ros:ai_module`, copies `src/dummy_vlm`
over an existing `/home/docker/ai_module` workspace, colcon-builds it,
and relies on the user calling `ros2 launch dummy_vlm dummy_vlm.launch`
inside an interactive shell.

For Task 2 I additionally probed the base image:

```
$ docker run --rm zhangjicmu/ubuntu24_ros:ai_module bash -c 'python3 --version; python3 -c "import rclpy"; id'
Python 3.12.3
uid=1001(docker) gid=1001(docker) groups=1001(docker),27(sudo)
```

Findings the Dockerfile depends on:
- `python3` is system Python 3.12 with `rclpy` at
  `/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/`.
- **`pip` is NOT installed** — we add `python3-pip` via apt in our image.
- The base ships **NumPy 1.26 from apt**, with no `RECORD` file, so
  pip cannot uninstall it. We loosened our requirement to `numpy>=1.26`
  to coexist with the system package.
- User `docker` exists at `/home/docker`.

### Phase 2 — Plan a parallel Python stack

Documented in `/home/leo/.claude/plans/ticklish-napping-acorn.md`.
Decisions: extend the challenge base image (fastest, most aligned);
auto-launch the VLM via ENTRYPOINT; ship a compose that brings up both
`iros2026_system` and our `xiao_hei_ai_module` so `docker compose up`
stands up the full dev rig; port `dummyVLM.cpp` 1:1 in Python so it
reads the same `waypoints.ply` / `object_list.txt` fixtures.

### Phase 3 — Implement

**Python:**
- `src/xiao_hei_vln/dummy/` — `fixtures.read_waypoints_ply` /
  `read_object_list` + `DummyResponder` that mirrors the three branches
  of `dummyVLM.cpp`. Bundled data files are verbatim copies from the
  challenge `ai_module/src/dummy_vlm/data/`.
- `src/xiao_hei_vln/app/main.py` — rclpy entry point. Creates
  `LatestCache`, calls Task-1's `bind_subscribers(node, cache)`,
  constructs `VLMOutputPublisher(node)`, ticks at 2 Hz, drives
  `DummyResponder.respond(snapshot)`, clears the cache slot when done.
- `pyproject.toml` gains `[project.scripts] xiao-hei-dummy-vlm = "xiao_hei_vln.app.main:main"`.
- 15 new pytest cases (`tests/test_fixtures.py`, `tests/test_dummy_responder.py`)
  covering all three response branches, waypoint-advance logic with /
  without pose, and parser edge cases. **Total now: 47 passing tests.**

**Docker:**
- `docker/Dockerfile` — adds `python3-pip`, copies the package, `pip install -e .`
  into system Python, sources ROS and sets cyclonedds at runtime.
- `docker/entrypoint.sh` — sources `/opt/ros/jazzy/setup.bash`, exports
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `exec`s the CMD.
- `docker/compose.yml` + `docker/compose_gpu.yml` — both services
  (`iros2026_system`, `xiao_hei_ai_module`) on `network_mode: host`
  with cyclonedds; GPU file adds NVIDIA reservations to each.
- `docker/README.md` — build / run / push / drop-in / replace-the-dummy
  walkthrough, plus a troubleshooting table.

## End-to-end verification (live sim)

`docker compose -f docker/compose_gpu.yml up -d` brought up both
containers. The dummy VLM auto-started:

```
[INFO] xiao_hei_dummy_vlm ready (tick = 2.00 Hz, 7 subscribers)
```

Launched `system_simulation.sh` inside `iros2026_system`, then
published one test question of each type and observed the matching
output topic:

| Question | Output topic | Captured message |
|---|---|---|
| `How many cups` | `/numerical_response` | `std_msgs/msg/Int32 data: 2` |
| `Find the red cup` | `/selected_object_marker` | Marker, frame_id=`map`, ns=`sofa`, position (3.37, -2.09, 0.50), scale (2.86, 1.20, 1.02) — matches `object_list.txt` exactly |
| `Take the path near the window` | `/way_point_with_heading` | `Pose2D x=7.5 y=-1.0` (the first waypoint from `waypoints.ply`); vehicle then advanced through the path. |

For each, the container logged `Received question: '...'` followed by
`Response complete; awaiting next question.`. Tore down with
`docker compose down`; no orphaned containers.

## Files added

- `src/xiao_hei_vln/dummy/{__init__.py, fixtures.py, responder.py}`
- `src/xiao_hei_vln/dummy/data/{object_list.txt, waypoints.ply}` (copied from challenge)
- `src/xiao_hei_vln/app/{__init__.py, main.py}`
- `tests/test_fixtures.py`, `tests/test_dummy_responder.py`
- `docker/{Dockerfile, entrypoint.sh, compose.yml, compose_gpu.yml, README.md}`
- `pyproject.toml` — added `[project.scripts]`, loosened numpy pin
- This report.

## Open follow-ups

- **Real VLM**: only `src/xiao_hei_vln/app/main.py` needs to change —
  swap `DummyResponder()` for the real model. Topic contract, cache,
  publisher, container, compose all stay put.
- **Slim image for submission**: today the image inherits the
  ~9.4 GB challenge base. A multi-stage build on `ros:jazzy-ros-base`
  would cut this dramatically; revisit before the official submission.
- **CI**: hook `docker compose build ai_module` into CI so we don't
  ship a regression to the Hub by accident.
