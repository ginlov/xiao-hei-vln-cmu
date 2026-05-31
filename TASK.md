# About the challenge

The CMU Vision-Language-Navigation Challenge combines computer vision and natural language understanding for navigation autonomy. The challenge aims to push the limits of embodied AI in real environments and on real robots — providing a robot platform and a working autonomy system to move higher-level reasoning and learning models closer to real-world deployment.

The challenge requires developing a model that takes natural language queries or commands about a scene and generates appropriate navigation responses by reasoning about semantic and spatial relationships. The environment is initially unknown, so the system must navigate to informative viewpoints to discover and validate spatial relations and object attributes. A real-robot platform equipped with a 3D LiDAR and a 360° camera is provided and includes onboard base autonomy that estimates sensor pose, analyzes terrain, avoids collisions, and navigates to waypoints. For 2026, the challenge will run first in a custom simulation environment and transition to the real-robot system in a later phase.

# Tasks

## Task 1 — Consolidate input/output format for the VLM model

Because we communicate with a ROS system where different sensors publish data at different frequencies, we need to consolidate the VLM model's input and output formats to:

- Standardize the data format.
- Synchronize or resample sensor data to a unified frequency for the VLM pipeline.

This task includes three phases:

- Run examples to determine current sensor data formats and publishing frequencies; consult the challenge repository and website for documentation.
- Propose the input/output formats and the execution frequency for the VLM system.
- Implement the proposed formats as Python data classes in this repository for development and testing.

## Task 2 - End-to-end integrate to docker container

This task is to enable core AI module development, now input and output format are standardzed, I want to build docker container from this project in order to:

- Make use of new input/output format.
- Build my own dummy VLM which is later replace by real VLM.

This task includes three phases:

- Investigate how dummy container in the challenge repo was built.
- Plan of building a dummy system for this repo so that I get use it from now.
- Implement it and write report on how to use, how to build, how to integrate VLM later and how to replace the ai module container in challenge repo by that container.

## Task 4 - Integrate Qwen3.5VL for testing
This task is to integrate Qwen3.5VL into the model to get initial results.

This task includes three phases:

- Research the proper library, framework to run Qwen3.5 VL for this challenge since we are gonna receive data frequently depends on the frequency we set.
- Implement that serving service.
- Design the prompt for the type one question which is numerical question, when we receive the question, Qwen model need to return the waypoint for the robot to navigate and also at the end, answer that numerical questions.

### Task 4.1
It seems like the dependencies of vllm are too different from the ROS system. Setting up a separate container for vllm only is better approach. For this task.
- Let set up a new container for Qwen3.5 VL only.
- Integrate the system so that we just need to start all in one.

## Task 5 — VLM tick logger for debugging and visualization

During live runs the VLM tick loop processes camera frames, constructs prompts, and produces outputs at 2 Hz — but none of this is persisted. When something goes wrong (bad answer, missed object, waypoint loop) there is no way to replay what the model saw, what it was asked, or what it responded.

This task adds a file-based logger that records every VLM tick to disk so runs can be inspected, visualized, and compared after the fact.

### What to log per tick

- `tick_id`, `tick_time`, inference latency (ms)
- Question text and classified type
- Robot pose (position + orientation)
- System prompt and user text sent to the engine
- Full `VLMOutput` JSON (including rationale)
- Evidence log snapshot (for multi-tick numerical reasoning)
- Camera frame saved as JPEG

### File layout

```
vlm_logs/
  session_20260530_143022/
    session.json          # Config snapshot, model, responder, tick_hz
    ticks.jsonl           # One JSON line per tick — all metadata above
    images/
      tick_000001.jpg     # Camera frame at that tick (only when present)
```

### Phases

- **Phase 1 — VLMLogger core**: implement `VLMLogger` class that writes `session.json` on init, appends to `ticks.jsonl` per tick, and saves JPEG images. Toggled by `XIAO_HEI_VLM_LOG_DIR` env var.
- **Phase 2 — Integration**: wire the logger into `QwenResponder` so it captures prompts, input metadata, output, and latency. Add `log_dir` passthrough from `compose_gpu.yml` with a bind-mount volume.
- **Phase 3 — Replay viewer**: a standalone script that loads a session directory and prints a human-readable summary (tick table, question timeline, answer history). Optionally opens images.