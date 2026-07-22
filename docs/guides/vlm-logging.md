# VLM Logging & Replay

The VLM tick logger records every model tick to disk for post-run debugging
and visualization.

## Enabling logging

Set `XIAO_HEI_VLM_LOG_DIR` to a directory path:

```bash
# In docker-compose (already set as the default in docker/compose.yml)
XIAO_HEI_VLM_LOG_DIR=/vlm_logs

# Locally
XIAO_HEI_VLM_LOG_DIR=./vlm_logs uv run xiao-hei-vlm
```

## File layout

```
vlm_logs/
  session_20260530_143022/
    session.json                    # Config snapshot
    q_001_how_many_chairs/
      ticks.jsonl                   # One JSON line per tick
      images/
        tick_000003.jpg             # Camera frame
      pointclouds/
        tick_000003_registered.npy  # Lidar points
        tick_000003_terrain_local.npy
      report.html                   # Generated HTML report
    q_002_find_the_red_cup/
      ticks.jsonl
      images/
        tick_000007.jpg
```

## What's logged per tick

| Field | Description |
|---|---|
| `tick_id` | Sequential tick number |
| `tick_time` | ROS timestamp |
| `inference_ms` | Model inference latency |
| `question_text` | Current question |
| `question_type` | numerical / object_reference / instruction_following |
| `system_prompt` | Full system prompt sent to model |
| `user_text` | Full user message sent to model |
| `output` | Complete VLMOutput JSON |
| `evidence` | Evidence log snapshot |
| `pose` | Robot position + orientation |
| `image_path` | Relative path to saved JPEG |
| `pointclouds` | Dict of relative paths to .npy files |

## Text replay

Quick terminal-based review:

```bash
# All questions in a session
python scripts/replay_session.py vlm_logs/session_20260530_143022

# Filter by question keyword
python scripts/replay_session.py vlm_logs/session_20260530_143022 -q chairs
```

## HTML report generation

Rich HTML reports with camera playback, trajectory plots, and sensor diagrams:

```bash
# Install replay dependencies
uv pip install -e ".[replay]"

# Generate reports for all questions in a session
python scripts/generate_report.py vlm_logs/session_20260530_143022/

# Single question
python scripts/generate_report.py vlm_logs/session_*/q_001_*/

# Filter by keyword
python scripts/generate_report.py vlm_logs/session_*/ -q chairs

# Open the report
open vlm_logs/session_*/q_001_*/report.html
```

### Report contents

1. **Camera playback** — JS slider / play button to scrub through frames at tick rate
2. **Scene Representation** — per-tick top-down spatial view + three-level scene
   graph + cumulative Room/Viewpoint/Object tables (see below)
3. **Pose trajectory** — Robot path with waypoint outputs overlaid
4. **Sensor BEV** — Bird's-eye-view scatter plot of terrain + lidar
5. **Per-tick I/O table** — Expandable prompts, outputs, evidence
6. **Latency chart** — Inference time per tick

### Scene Representation playback

The Scene Representation section renders one frame per logged tick (top-down
spatial view on the left, scene-graph topology on the right, node tables below).

- **Synchronised with the camera.** When the session has camera frames, the
  scene view has **no controls of its own** — the camera playback slider / play
  button drives both, so the video and the scene graph advance together in
  lock-step. Each step shows the *cumulative* scene state at-or-before the
  current camera tick. (With no camera frames, the section falls back to its own
  slider.)
- **Stationary top-down window.** The top-down boundary is pinned to a single
  session-wide square window (union of all object/viewpoint/pose positions,
  padded and squared) so it does **not** zoom or pan between frames — only the
  objects, viewpoints, and robot marker move within a fixed frame. The LiDAR
  `scene_bounds` is excluded from the window (it spans the full ~20 m scan
  radius and would shrink the explored area to a dot); its dashed rectangle
  simply clips to the window.
- **Object↔Object edges are not drawn.** Both the top-down view and the scene
  graph show only the Room→Viewpoint→Object hierarchy. The `near` relation
  count still appears in the title as a statistic, but the edges themselves are
  omitted to keep the graph readable.

## Disabling logging

Unset the environment variable:

```bash
XIAO_HEI_VLM_LOG_DIR="" docker/run perception up -d
```

## Performance impact

Logging adds ~5 ms overhead per tick (JSONL write + JPEG save), well within
the 500 ms tick budget.
