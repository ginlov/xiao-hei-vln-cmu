# VLM Logging & Replay

The VLM tick logger records every model tick to disk for post-run debugging
and visualization.

## Enabling logging

Set `XIAO_HEI_VLM_LOG_DIR` to a directory path:

```bash
# In docker-compose (already set in compose_gpu.yml)
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

1. **Camera playback** — JS slider to scrub through frames at tick rate
2. **Pose trajectory** — Robot path with waypoint outputs overlaid
3. **Sensor BEV** — Bird's-eye-view scatter plot of terrain + lidar
4. **Per-tick I/O table** — Expandable prompts, outputs, evidence
5. **Latency chart** — Inference time per tick

## Disabling logging

Unset the environment variable:

```bash
XIAO_HEI_VLM_LOG_DIR="" docker compose -f docker/compose_gpu.yml up -d
```

## Performance impact

Logging adds ~5 ms overhead per tick (JSONL write + JPEG save), well within
the 500 ms tick budget.
