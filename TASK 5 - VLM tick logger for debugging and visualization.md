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

Logs are organized per question within each session:

```
vlm_logs/
  session_20260530_143022/
    session.json                    # One-time: config, model, responder, tick_hz
    q_001_how_many_chairs/
      ticks.jsonl                   # One JSON line per tick for this question
      images/
        tick_000003.jpg
    q_002_find_the_red_cup/
      ticks.jsonl
      images/
        tick_000007.jpg
```

- **Per-question directories** — named `q_{NNN}_{slug}` where NNN is
  sequential and slug is derived from the question text. Each question
  gets its own `ticks.jsonl` and `images/` directory, making it easy
  to inspect a single question's reasoning chain.
- **JSONL** — one line per tick, cheap to append, easy to load into
  pandas (`pd.read_json("ticks.jsonl", lines=True)`).
- **Session JSON** — written once at startup, contains the full
  `QwenConfig` dump plus metadata (start time, responder name,
  tick frequency, model name).
- **JPEG images** — only saved when `VLMInput.image` is not `None`.
  BGR8 → RGB → JPEG quality 90, using `image_frame_to_pil` from
  `image_utils.py`. Zero-padded tick IDs in filename so `ls` sorts
  correctly.

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

The responder calls `logger.new_question(text)` on the first tick of
each question to open a new subdirectory, then `logger.log_tick(...)`
after each inference. The logger handles JPEG encoding and file I/O
internally.

### Synchronous vs async

At 2 Hz with ~2 ms for JPEG encode + ~1 ms for file write, synchronous
logging adds < 5 ms per tick — well within the 500 ms budget. No
background thread needed. If latency becomes a concern on slower
storage, a buffered/async path can be added later without changing the
interface.

## Phases

### Phase 1 — VLMLogger core

New files:

- `src/xiao_hei_vln/qwen/image_utils.py` — shared `image_frame_to_pil`
  and `resize_pil` helpers, extracted from `engine.py` to avoid
  duplication between engine and logger.
- `src/xiao_hei_vln/qwen/logger.py`:
  - `VLMLogger.__init__(log_dir, config, responder_name, tick_hz)` —
    creates session directory, writes `session.json`.
  - `VLMLogger.new_question(question_text)` — opens a new per-question
    subdirectory (`q_001_<slug>/`) with its own `ticks.jsonl` and
    `images/` directory.
  - `VLMLogger.log_tick(snapshot, system_prompt, user_text, output,
    inference_ms, evidence)` — appends one JSON line to the current
    question's `ticks.jsonl`, saves JPEG if image present.
  - `VLMLogger.close()` — flushes and closes the current file handle.
  - Helper `_slugify(text)` — converts question text to a filesystem-
    safe slug for directory naming.
- Tests with `tmp_path` fixture — 13 tests covering session init,
  per-question dirs, JSONL records, image saving, multiple questions,
  slug special characters, edge cases.

### Phase 2 — Integration

- `QwenResponder`: optional `logger` kwarg; calls `logger.new_question()`
  on tick_count == 1, calls `logger.log_tick()` after inference with
  `time.perf_counter()` timing.
- `app/main.py`: reads `XIAO_HEI_VLM_LOG_DIR`, creates `VLMLogger`
  if set, passes to `QwenResponder`, closes on shutdown.
- `engine.py`: refactored to import from `image_utils.py`.
- `compose_gpu.yml`: `XIAO_HEI_VLM_LOG_DIR=/vlm_logs` env var +
  bind-mount `../vlm_logs:/vlm_logs` on `ai_module`.
- `.gitignore`: added `vlm_logs/`.

### Phase 3 — Replay viewer

Standalone script `scripts/replay_session.py`:

- Loads `session.json` and per-question `ticks.jsonl` files.
- Prints a summary table per question: tick_id, time, output kind,
  inference_ms, rationale.
- Shows latency stats (min/max/avg) per question.
- With `--images`: opens JPEG for each tick using PIL.

## Files to add / change

```
src/xiao_hei_vln/qwen/image_utils.py    new — shared image_frame_to_pil + resize_pil
src/xiao_hei_vln/qwen/logger.py         new — VLMLogger class (per-question dirs)
src/xiao_hei_vln/qwen/engine.py         refactored to use image_utils
src/xiao_hei_vln/qwen/__init__.py       + export VLMLogger
src/xiao_hei_vln/qwen/responder.py      + optional logger kwarg + timing
src/xiao_hei_vln/app/main.py            + logger init from env
docker/compose_gpu.yml                   + log dir env + bind mount
.gitignore                               + vlm_logs/
tests/test_vlm_logger.py                 new — 13 logger unit tests
scripts/replay_session.py                new — session replay CLI
docker/README.md                         + VLM tick logging section + env var docs
```

## Verification

```
$ uv run pytest -q
........................................................................ [ 61%]
.............................................                            [100%]
117 passed in 0.25s

$ uv run ruff check src tests scripts
All checks passed!
```

(94 from Tasks 1–4, +13 new logger tests, +10 others.)

## How to use

```bash
# GPU compose — logging is enabled by default (XIAO_HEI_VLM_LOG_DIR=/vlm_logs)
docker compose -f docker/compose_gpu.yml up -d

# After a run, logs appear on the host:
ls vlm_logs/session_*/
# → session.json  q_001_how_many_chairs/  q_002_find_the_red_cup/

# Each question dir contains:
ls vlm_logs/session_*/q_001_*/
# → ticks.jsonl  images/

# Replay a session:
python scripts/replay_session.py vlm_logs/session_20260530_143022

# Replay with images:
python scripts/replay_session.py vlm_logs/session_20260530_143022 --images

# Disable logging:
XIAO_HEI_VLM_LOG_DIR="" docker compose -f docker/compose_gpu.yml up -d
```
