# TASK 4 — Integrate Qwen3.5 VL for testing

## Goal

Plug Qwen3.5 into the Task-1 / Task-2 stack so the system answers
real challenge questions instead of the dummy fixtures, and lay the
groundwork for the type-1 (numerical) question pipeline: from a
question + a stream of camera frames, choose either to navigate the
robot to a better viewpoint or to commit to a final integer answer.

The dummy responder stays in place as a no-GPU fallback so dev /
compose-smoke / CI paths keep working unchanged.

## What was done

### Phase 1 — Framework research

- Confirmed (with a web check) that **Qwen 3.5 is natively
  multimodal** — text+image+video trained jointly, no separate `-VL`
  checkpoint. The HF collection ships 0.8B/2B/4B/9B/27B dense and
  35B-A3B/122B-A10B/397B-A17 MoE variants.
- Picked **`Qwen3.5-4B` at bf16** as the initial target for a single
  24 GB GPU (~8 GB weights → ~14 GB headroom for KV + vision tokens).
  Documented swap paths to `Qwen3.5-9B-Instruct-AWQ` and
  `Qwen3.5-2B` as quality/latency fallbacks.
- Initially chose vLLM in-process, later refactored to **HTTP sidecar**
  (Task 4.1) — vLLM runs in a separate container (`vllm/vllm-openai`)
  and the ai_module calls it via `HTTPQwenEngine` using the OpenAI-
  compatible API. This avoids pip/apt dependency conflicts with the
  ROS base image. In-process `QwenEngine` remains as a legacy
  fallback. Guided JSON decoding works via `response_format` on the
  HTTP API.
- Recorded the full decision and latency budget breakdown in
  **`docs/task3_phase1_framework.md`**.

### Phase 2 — Serving service

- New package **`src/xiao_hei_vln/qwen/`**:
  - `config.QwenConfig` — env-overridable dataclass (model, dtype,
    `max_model_len`, GPU util, sampling temperature / max tokens /
    seed, image long-edge, per-question tick cap, `vllm_base_url`).
    All knobs land via `XIAO_HEI_QWEN_*`.
  - `engine.HTTPQwenEngine` (default) — calls a vLLM
    OpenAI-compatible HTTP sidecar. Sends images as base64 JPEG data
    URLs. Uses `response_format` for guided JSON decoding. `warmup()`
    polls the sidecar until ready (up to 300 s).
  - `engine.QwenEngine` (legacy) — loads vLLM in-process. Requires
    `pip install .[qwen-local]` and a CUDA GPU in the same container.
  - `engine.EngineProtocol` — minimal Protocol so tests can inject a
    fake without importing vLLM / PIL.
  - `image_utils.py` — shared `image_frame_to_pil` and `resize_pil`
    helpers for BGR→RGB conversion and downscaling.
  - `prompts.build_system_prompt` + `build_user_message` — generic
    builders that cover all three question types (used by object-
    reference and instruction-following).
  - `responder.QwenResponder` — same shape as `DummyResponder`
    (`respond` / `is_done` / `reset`); holds the evidence log across
    ticks; terminal `NumericalResponse` / `ObjectReferenceResponse`
    set `is_done`; `WaypointPathResponse` is terminal only for
    instruction-following, otherwise the responder stays alive for a
    next exploration tick; engine exceptions skip the tick rather
    than crash the rclpy node; numerical timeout falls back to
    `NumericalResponse(value=0)`.
- **`src/xiao_hei_vln/app/main.py`** — `_build_responder(name)` picks
  `dummy` vs `qwen` from `XIAO_HEI_RESPONDER`. When `qwen`:
  selects `HTTPQwenEngine` if `vllm_base_url` is set, else falls back
  to `QwenEngine`. Logs the responder name in the ready line.
- **`pyproject.toml`** — two optional dep groups:
  `qwen = ["openai>=1.30", "pillow>=10"]` (lightweight HTTP client,
  default) and `qwen-local = ["vllm>=0.7,<0.10", "pillow>=10",
  "huggingface-hub>=0.24"]` (heavy in-process, legacy).
- **Docker:**
  - `docker/Dockerfile` — `ARG XIAO_HEI_EXTRA` controls which extra
    is installed (default `qwen` → openai + pillow only).
  - `docker/compose_gpu.yml` — three services: `system` (challenge
    sim), `vllm` (sidecar using `vllm/vllm-openai` image with local
    model weights bind-mounted), `ai_module` (our code). Default env:
    `XIAO_HEI_QWEN_VLLM_BASE_URL=http://localhost:8000/v1`.
  - `docker/compose.yml` — two services for CPU/dummy mode (no GPU).
  - `docker/README.md` — documents sidecar architecture, env vars,
    and troubleshooting.
- **Tests (no GPU required):**
  - `tests/test_qwen_config.py` — defaults match Phase-1 plan;
    `from_env()` overrides every knob.
  - `tests/test_qwen_prompts.py` — system prompt covers all three
    schemas; user message renders question + pose (incl. missing) +
    evidence (incl. first-tick); type-specific hint per question
    kind.
  - `tests/test_qwen_responder.py` — `ScriptedEngine` + `FailingEngine`
    cover terminal numerical / object-reference, multi-tick
    numerical-with-navigation, evidence-log growth, instruction-
    following terminal-after-one-tick, engine-failure tick-skip,
    numerical-timeout fallback, non-numerical-timeout silence, reset.

### Phase 3 — Numerical-question prompt design

- Added an optional **`rationale: str | None = None`** field to all
  three `VLMOutput` variants. The publisher is unaffected (already
  only reads the topic-relevant fields); the new field is the
  exclusive carrier for cross-tick reasoning in the numerical loop.
- New specialised prompts in `src/xiao_hei_vln/qwen/prompts.py`:
  - `NUMERICAL_SYSTEM_PROMPT` describes the per-tick protocol
    (estimate `view_count`, recover prior `running_total`, decide
    `explore` vs `commit`), the concrete commit predicate (≥ 2
    distinct viewpoints + last two tallies agree + nothing new in
    frame), the JSON shape for each action, and the
    `view_count=... running_total=... action=...` rationale
    convention.
  - `build_numerical_user_message(...)` renders the per-tick budget
    (`Tick i of at most N`), the full prior-rationale evidence log,
    and on the final tick switches to `BUDGET EXHAUSTED: you MUST
    commit on this tick` wording so the model gets the same signal
    the responder's safety net does.
- `QwenResponder` now dispatches by question type: numerical uses the
  numerical prompts and captures rationale verbatim (so the next tick
  sees the model's own scratchpad), other types continue to use the
  Phase-2 prompts and the coarse summary log.
- Design captured in **`docs/task3_phase3_prompt.md`**.

## Verification

```
$ uv run pytest -q
............................................................................   78 passed in 0.19s
$ uv run ruff check src tests
All checks passed!
```

(47 from Tasks 1+2, +20 Phase 2, +11 Phase 3.)

Live-system / GPU verification is deferred to a follow-up — this
machine is a Mac, so the responder loop is exercised end-to-end with
a scripted engine and the engine wrapper is held to a static contract
(`EngineProtocol`) plus a lint pass.

## Files added / changed

```
docs/
  task3_phase1_framework.md            new
  task3_phase3_prompt.md               new
src/xiao_hei_vln/
  messages/outputs.py                  + optional `rationale` on the 3 responses
  app/main.py                          + responder env switch
  qwen/__init__.py                     new
  qwen/config.py                       new
  qwen/engine.py                       new
  qwen/prompts.py                      new (generic + numerical builders)
  qwen/responder.py                    new
tests/
  test_qwen_config.py                  new
  test_qwen_prompts.py                 new
  test_qwen_responder.py               new
  test_qwen_numerical_loop.py          new
docker/
  Dockerfile                           + XIAO_HEI_EXTRA build arg + HF_HOME
  compose_gpu.yml                      + build args, env forwarding, hf_cache volume
  compose.yml                          + build args, responder env
  README.md                            + Qwen3.5 switching section
pyproject.toml                          + [qwen] extra, xiao-hei-vlm script alias
TASK 4 - Integrate Qwen3.5 VL for testing.md   this report
```

## How to run

```bash
# Local dev (Mac, no GPU): dummy responder, all tests, lint
uv sync
uv run pytest -q
uv run ruff check src tests

# Build the ai_module (lightweight: only openai + pillow)
docker compose -f docker/compose_gpu.yml build ai_module

# Bring up all three services (system + vllm sidecar + ai_module)
docker compose -f docker/compose_gpu.yml up -d

# Watch vLLM startup (wait for "Uvicorn running on ...")
docker logs -f xiao_hei_vllm

# Watch ai_module logs
docker logs -f xiao_hei_ai_module

# Ask a numerical question
docker exec iros2026_system bash -lc \
  'source /opt/ros/jazzy/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
   ros2 topic pub --once /challenge_question std_msgs/msg/String "{data: \"How many cups\"}"'
```

Model weights are bind-mounted from `../models/` into the vLLM
container. Download weights to that directory before starting.

## Open follow-ups

- **Eval-host latency measurement**: validate the 250–430 ms estimate
  on the real GPU; tune `XIAO_HEI_VLM_TICK_HZ` and / or quantization
  based on the gap.
- **Rationale parser**: the `view_count=... running_total=...` triplet
  is convention-only; a small parser + drift warning would harden the
  numerical loop.
- **Phase 4 — other question types**: object-reference and
  instruction-following currently use the Phase-2 generic prompt.
  Specialising them is the natural next task once we have numerical
  scores from the eval node.
- **Submission-grade image**: the sidecar architecture keeps the
  ai_module image lightweight (~9.4 GB challenge base + openai/pillow).
  For submission, the vLLM sidecar can be replaced with a smaller
  serving image or the model can be quantized.
