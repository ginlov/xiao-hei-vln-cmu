# TASK 5 — VLM tick logger for debugging and visualization

## Problem

During live runs the VLM tick loop at 2 Hz processes camera frames,
constructs prompts, and produces `VLMOutput` responses — but nothing is
persisted. When the model gives a wrong answer, explores too long, or
loops on waypoints, the only signal is a final ROS topic message and
whatever was in `docker logs`. There is no way to:

- See what camera frame the model was looking at on a given tick.
- Read the exact prompt (system + user text) that was sent.
- Compare outputs across ticks to spot drift or hallucination.
- Replay a full question session to understand multi-tick reasoning.

## Design

### Data captured per tick

| Field               | Source                          | Storage                  |
|---------------------|---------------------------------|--------------------------|
| `tick_id`           | `VLMInput.tick_id`              | JSONL                    |
| `tick_time`         | `VLMInput.tick_time`            | JSONL (float seconds)    |
| `question_text`     | `VLMInput.question.text`        | JSONL                    |
| `question_type`     | `VLMInput.question.type`        | JSONL                    |
| `pose`              | `VLMInput.pose`                 | JSONL (position + quat)  |
| `system_prompt`     | constructed in `QwenResponder`  | JSONL                    |
| `user_text`         | constructed in `QwenResponder`  | JSONL                    |
| `output`            | `VLMOutput` model dump          | JSONL                    |
| `inference_ms`      | wall-clock around `engine.infer`| JSONL                    |
| `evidence`          | responder evidence log snapshot | JSONL                    |
| `image_path`        | camera JPEG filename            | JSONL (relative path)    |
| camera frame        | `VLMInput.image`                | JPEG file in `images/`   |

### File layout

```
vlm_logs/
  session_20260530_143022/
    session.json          # One-time: config, model, responder, tick_hz
    ticks.jsonl           # Append-only: one JSON line per tick
    images/
      tick_000001.jpg     # Camera frame (only when image was present)
      tick_000004.jpg
```

- **JSONL** — one line per tick, cheap to append, easy to load into
  pandas (`pd.read_json("ticks.jsonl", lines=True)`).
- **Session JSON** — written once at startup, contains the full
  `QwenConfig` dump plus metadata (start time, responder name,
  tick frequency, model name).
- **JPEG images** — only saved when `VLMInput.image` is not `None`.
  BGR8 → RGB → JPEG quality 90, same conversion already in
  `engine._image_frame_to_pil`. Zero-padded tick IDs in filename
  so `ls` sorts correctly.

### Toggle

- `XIAO_HEI_VLM_LOG_DIR` env var → when set, logger is active.
- When unset or empty → no logger created, zero overhead.
- `compose_gpu.yml` adds a bind-mount volume mapping a host directory
  into the container at the configured path.

### Integration point

`QwenResponder.respond()` is the only place where both the constructed
prompts (system + user text) and the output are available. The logger
is injected into the responder as an optional dependency:

```python
class QwenResponder:
    def __init__(self, engine, config, *, logger: VLMLogger | None = None):
        self._logger = logger
```

The responder calls `logger.log_tick(...)` after inference completes.
The logger handles JPEG encoding and file I/O internally.

### Synchronous vs async

At 2 Hz with ~2 ms for JPEG encode + ~1 ms for file write, synchronous
logging adds < 5 ms per tick — well within the 500 ms budget. No
background thread needed. If latency becomes a concern on slower
storage, a buffered/async path can be added later without changing the
interface.

## Phases

### Phase 1 — VLMLogger core

New file `src/xiao_hei_vln/qwen/logger.py`:

- `VLMLogger.__init__(log_dir, config, responder_name, tick_hz)` —
  creates session directory, writes `session.json`, opens `ticks.jsonl`
  for appending.
- `VLMLogger.log_tick(snapshot, system_prompt, user_text, output,
  inference_ms, evidence)` — appends one JSON line, saves JPEG if image
  present.
- `VLMLogger.close()` — flushes and closes the JSONL file handle.
- Helper `_save_image(image_frame, path)` — reuses
  `_image_frame_to_pil` from `engine.py` for BGR→RGB conversion.
- Tests with `tmp_path` fixture — verify file layout, JSONL schema,
  image creation.

### Phase 2 — Integration

- Wire logger into `QwenResponder`: optional `logger` kwarg, called
  inside `respond()` after inference.
- Wire into `app/main.py`: read `XIAO_HEI_VLM_LOG_DIR`, create
  `VLMLogger` if set, pass to `QwenResponder`.
- `compose_gpu.yml`: add `XIAO_HEI_VLM_LOG_DIR=/vlm_logs` env var and
  bind-mount `../vlm_logs:/vlm_logs` on `ai_module`.
- Add `vlm_logs/` to `.gitignore`.

### Phase 3 — Replay viewer

A standalone script `scripts/replay_session.py` that:

- Loads `session.json` and `ticks.jsonl` from a session directory.
- Prints a summary table: tick_id, question, output kind, inference_ms.
- Groups ticks by question for multi-tick numerical debugging.
- Optionally opens JPEG images for specific ticks.

## Files to add / change

```
src/xiao_hei_vln/qwen/logger.py         new — VLMLogger class
src/xiao_hei_vln/qwen/__init__.py        + export VLMLogger
src/xiao_hei_vln/qwen/responder.py       + optional logger kwarg
src/xiao_hei_vln/app/main.py             + logger init from env
docker/compose_gpu.yml                   + log dir env + bind mount
.gitignore                               + vlm_logs/
tests/test_vlm_logger.py                 new — logger unit tests
scripts/replay_session.py                new — session replay CLI
docker/README.md                         + VLM tick logging section + env var docs
```

## Verification

```
$ uv run pytest -q
........................................................................ [ 63%]
.........................................                                [100%]
113 passed in 0.27s

$ uv run ruff check src tests scripts
All checks passed!
```

(94 from Tasks 1–4, +9 new logger tests, +10 others.)

## How to use

```bash
# GPU compose — logging is enabled by default (XIAO_HEI_VLM_LOG_DIR=/vlm_logs)
docker compose -f docker/compose_gpu.yml up -d

# After a run, logs appear on the host:
ls vlm_logs/session_*/
# → session.json  ticks.jsonl  images/

# Replay a session:
python scripts/replay_session.py vlm_logs/session_20260530_143022

# Disable logging:
XIAO_HEI_VLM_LOG_DIR="" docker compose -f docker/compose_gpu.yml up -d
```
