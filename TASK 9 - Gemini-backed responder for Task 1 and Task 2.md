# TASK 9 — Gemini-backed responder for Task 1 and Task 2

## Problem

The existing responders (`dummy`, `qwen`, `perception`) leave the
question-answering / route-planning loop open:

- `dummy` is a deterministic fallback with no reasoning.
- `qwen` answers but does no geometric exploration of the scene.
- `perception` (PR #12, Phase A frontier) explores but **never produces
  an answer** — every question type gets a navigate-only reply.

We have a Gemini API key, so the natural step is to fuse the two: keep
the proven frontier explorer driving the agent around the room, then on
a clear trigger ask Gemini for the actual answer (Task 1) or route
plan (Task 2) using the accumulated scene representation. That's the
minimum-mileage path to closing the loop on all three question types
without bringing up a local-GPU VLM.

## What this PR adds

| File | Purpose |
|---|---|
| `src/xiao_hei_vln/gemini/config.py` | `GeminiConfig` dataclass — API key, model, temperature, output-token cap, image long-edge, `max_explore_ticks`, `max_ticks_per_question`. `from_env()` reads `XIAO_HEI_GEMINI_API_KEY` + optional overrides. |
| `src/xiao_hei_vln/gemini/prompts.py` | Three system prompts (numerical / object_reference / waypoint_path) each embedding the canonical `VLMOutput` JSON schema. `build_system_prompt(qtype)` dispatches; `build_user_message(...)` formats pose + trajectory + terrain summary. |
| `src/xiao_hei_vln/gemini/scene_rep.py` | Wraps the `SceneRepresentation` (cherry-picked from `feat/scene-representation`) plus the `GlobalMap` into a `GeminiSceneBundle`. `serialize_for_gemini(bundle)` returns `(scene_json_text, [panorama_jpg, occupancy_png])`. The occupancy PNG is rendered with matplotlib's Agg backend (lazy import). |
| `src/xiao_hei_vln/gemini/engine.py` | `GeminiEngine` — thin wrapper around `google-genai` that issues one `generate_content` call per question with `response_mime_type="application/json"` and `response_schema=TypeAdapter(VLMOutput).json_schema()`. `GeminiClientProtocol` + `GeminiEngineProtocol` let tests inject fakes. SDK import is lazy. |
| `src/xiao_hei_vln/gemini/responder.py` | `GeminiResponder` — stateful per-question wrapper that composes a real `PerceptionResponder` for exploration and the engine for reasoning. Branches by `QuestionType`: explore-then-answer for Task 1, plan-once-then-step-waypoints for Task 2. |
| `src/xiao_hei_vln/scene/{__init__,representation}.py` | SysNav-style scene graph (Room → Viewpoints → Objects). Cherry-picked from `feat/scene-representation` and used as-is. |
| `src/xiao_hei_vln/app/main.py` | New dispatcher branch + node name for `XIAO_HEI_RESPONDER=gemini`. |
| `pyproject.toml` | New `gemini` optional-dependency group: `google-genai>=0.5`, `pillow>=10`, `matplotlib>=3.8`. |

## Architecture

Per tick:

```
snapshot
   │
   ├─ scene.update(snapshot)         # accumulate SysNav scene graph
   ├─ trajectory_xy.append(pose)     # ring buffer, last 200 poses
   │
   ▼
question.type == INSTRUCTION_FOLLOWING ?
   │
   ├─ YES ─▶ Task-2:
   │           if no plan yet:
   │             build scene bundle → call Gemini → store waypoint plan
   │           emit current waypoint; advance when within 0.5 m
   │           mark done when last waypoint reached
   │
   └─ NO  ─▶ Task-1:
              if committed answer exists → return it
              if trigger fires:
                 build scene bundle → call Gemini → commit + done
              else:
                 delegate to PerceptionResponder (frontier waypoint)
```

**Task-1 trigger** (whichever fires first):

1. `PerceptionResponder` rationale starts with `phase_a_exhausted`
   (frontier list empty), OR
2. `tick_count > config.max_explore_ticks` (default 120 ticks ≈ 60 s
   at 2 Hz), OR
3. `tick_count > config.max_ticks_per_question` (hard safety cap).

## Output guarantee

JSON-mode + `response_schema` means the SDK rejects responses that
don't match the canonical `VLMOutput` schema. The responder then
parses with `parse_vlm_output(...)` which is the same parser the Qwen
pipeline uses, so the discriminated union (`kind`:
numerical / object_reference / waypoint_path) is enforced end-to-end.

## Tests

All run **without `XIAO_HEI_GEMINI_API_KEY`** — engines and perception
are injected as fakes:

| File | Cases |
|---|---|
| `tests/test_gemini_config.py` | 5 — env-var defaults, overrides, missing-key error. |
| `tests/test_gemini_prompts.py` | 11 — each system prompt embeds its JSON schema; user-message formatter handles missing pose, missing terrain; routing dispatches by `QuestionType`. |
| `tests/test_gemini_scene_rep.py` | 7 — `GeminiSceneBundle` shape; `serialize_for_gemini` returns 2-image list; PNG render uses Agg backend; viewpoint markers actually change the rendered bytes. |
| `tests/test_gemini_engine.py` | 11 — MIME sniffing for PNG / JPEG; `infer_multimodal` sends `system_instruction` + `response_schema` config; multi-image contents shape; `warmup()` issues exactly one call and propagates errors; text fallback to `candidates[0].content.parts[0].text`. |
| `tests/test_gemini_responder.py` | 7 — Task-1 explore-then-commit at `max_explore_ticks+1`; subsequent ticks return committed answer; object_reference routing; Task-2 plan-then-step; Task-2 mis-typed Gemini reply marks done; `reset()` clears state for the next question. |

293 tests total pass (gemini + scene + the full pre-existing suite).

## Smoke test

```bash
XIAO_HEI_GEMINI_API_KEY=test XIAO_HEI_RESPONDER=gemini \
  .venv/bin/python -c "from xiao_hei_vln.app.main import _build_responder; \
                       print(_build_responder('gemini'))"
```

confirms the dispatcher branch loads, instantiates a real
`GeminiResponder`, and gates on the env var.

## Out of scope

- **No object detection / YOLO-World.** The scene rep is occupancy +
  pose + RGB + the SysNav viewpoint graph only.
- **No Phase B / TSP coverage.** Exploration is still Phase A frontier.
- **No live SDK call in CI.** The real Gemini call only happens when
  the env var is set; the test suite never touches the network.

## Files modified / created

**Created** — 8 new source files, 6 new test files (one cherry-picked).

**Modified** — `pyproject.toml`, `src/xiao_hei_vln/app/main.py`.

## Follow-ups

- Tune `max_explore_ticks` per question difficulty once we have logged
  cost / latency numbers from real runs.
- Wire `VLMLogger` JSONL into the responder so the full Gemini prompt
  + response is captured per tick — already plumbed in the dispatcher,
  just not yet exercised by `GeminiResponder.respond()`.
- Phase B coverage planner + object detection are still open and would
  let us shrink `max_explore_ticks` aggressively.
