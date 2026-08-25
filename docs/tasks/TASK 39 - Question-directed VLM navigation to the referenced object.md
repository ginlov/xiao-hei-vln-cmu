# TASK 39 — Question-directed VLM navigation to the referenced object (Opus 5)

## Purpose

A second VLM navigation mode, for the **first question type**
(`object_reference`, e.g. *"Find the red chair"* / *"the pillow closest to the
sushi"*). Where TASK 38's `nav_vlm` explores *undirected*, this mode is
**goal-directed**: given the question, drive the robot to the object it names,
then answer the question from the built scene graph.

The three constraints from the request map onto the pipeline as follows:

- **Scene graph still built at 2 Hz** — the app tick loop already calls
  `responder.ingest()` every tick during the exploration phase; the new
  `scene_claude` responder runs the perception detect → lift → fuse cycle there,
  unchanged.
- **Claude called only on waypoint reached or skipped** — the app's existing
  reach / skip / stuck-watchdog supervisor already drives `explorer.advance()` /
  `force_skip()` on exactly those events. The new explorer's model call fires on
  those hooks (inherited from TASK 38's event-driven trigger), never on a timer.
- **At arrival, dump the scene graph to Claude to answer** — when the model
  declares arrival the explorer completes, the tick loop hands over to
  `scene_claude`, which serialises the full scene graph and asks Claude to pick
  the answer `object_id`.

## What was built

Reuses TASK 38's `nav_vlm` engine/config/render and the app's navigation
supervisor — the new surface is small.

| Piece | Role |
|---|---|
| `nav_vlm/engine.py` | Generalised `AnthropicNavEngine`: a `call_tool(system, tool, user_text, images)` primitive with injectable `system_prompt`/`tool`; `propose()` now sits on top of it. Backward-compatible (TASK 38 tests unchanged). |
| `nav_vlm/task1_prompts.py` | The goal-directed nav prompt + `propose_or_arrive` tool (`done` = *arrived at the target object*), the object-reference **answer** prompt + `answer_object_reference` tool, and the user-text / scene-summary builders. |
| `exploration/_nav_task1.py` | `NavTask1Explorer(NavVLMExplorer)` — holds until a question arrives, feeds the question + objects-detected-so-far into each call, and completes when the model declares arrival. Inherits grid, reachability snapping, async off-thread calls, and the reach/skip/watchdog machinery. |
| `scene_claude/responder.py` | `SceneClaudeResponder` — builds the scene at 2 Hz via a wrapped `PerceptionResponder` (`ingest`), and at arrival dumps `scene.to_dict()` + a panorama + occupancy map to Claude to select the answer `object_id` → `ObjectReferenceResponse`. Non-`object_reference` questions fall back to the perception responder. |
| `app/main.py` | Wires `XIAO_HEI_EXPLORATION_STRATEGY=nav_task1` (threading the shared `scene` into `_build_explorer`) and `XIAO_HEI_RESPONDER=scene_claude`. |

## Design — the loop

```
question arrives ─► NavTask1Explorer (goal-directed)
   every tick: PerceptionResponder.ingest → scene graph grows (2 Hz)
   on reach (advance) / skip (force_skip / watchdog): ONE Claude call
       → propose_or_arrive: next waypoint toward target, OR done=arrived
   arrived ─► explorer.is_complete() ─► SceneClaudeResponder.respond
       → dump scene graph + images to Claude
       → answer_object_reference(object_id) ─► ObjectReferenceResponse
```

Before the question arrives the explorer **holds** (no target object to head
for) while the scene graph still fills in. Every waypoint the model proposes is
snapped to a grid-reachable free cell (TASK 38's anti-wedge safeguard), and a
cannot-reach outcome feeds its reason into the next prompt.

If the target object is not yet in the scene graph, the model is told so (the
prompt lists the detected objects) and proposes search waypoints toward
unobserved area until it appears — then approaches and declares arrival. Arrival
is **Claude's decision**, not a geometric threshold.

## Robustness

- **Answer always emitted.** If Claude fails, times out, or names an id not in
  the graph, the responder retries a few answer ticks (the scene may still be
  settling) and then falls back to the closest object whose label appears in the
  question — so the run never ends without an `ObjectReferenceResponse`.
- **Fails fast on a missing key** for the answerer (`scene_claude` raises at
  boot); the explorer fails **soft** (disables, like `nav_vlm`).
- **No SDK / key / GPU needed to build or test** — lazy `anthropic` import,
  injectable fake client/engine.

## Tests (no key / no network)

- `tests/test_nav_task1_explorer.py` — holds before a question; a question
  triggers nav and the prompt carries the question + scene graph; model arrival
  completes; reach re-triggers; skip feeds the failure forward.
- `tests/test_scene_claude_responder.py` — a valid model pick becomes the
  `ObjectReferenceResponse`; `ingest` builds without answering; invalid-id retry
  → label fallback; engine failure tolerated; non-`object_reference` delegates
  to perception.
- `tests/test_nav_vlm_engine.py` — added a `call_tool` custom-tool test.

29 mode tests pass; the full suite is 524 passed (the 2 collection errors are
pre-existing missing optional deps: networkx/scipy).

## To go live

1. Provide `ANTHROPIC_API_KEY` (env or repo `.env`).
2. `uv sync --extra nav-vlm` (adds `anthropic`; matplotlib/pillow already needed
   by the occupancy render).
3. Select both halves:
   `XIAO_HEI_RESPONDER=scene_claude` and
   `XIAO_HEI_EXPLORATION_STRATEGY=nav_task1`.
   The perception sidecar must be up (scene-graph building).

## Live validation (2026-08-24)

With the Anthropic key in `.env` (`XIAO_HEI_ANTHROPIC_API_KEY`), both Claude
paths were smoke-tested against the real `claude-opus-5`:

- **nav** (`propose_or_arrive`) — returned a goal-directed waypoint with the
  right reasoning ("no red chair detected yet, so move into unobserved area");
- **answer** (`answer_object_reference`) — correctly selected the matching
  `object_id` from a scene-graph dump;
- **`warmup()`** — succeeds (credentials + model + forced-tool + image).

Two API facts were found and handled in the engine/config:

- Opus 5 **deprecates `temperature`** — the request omits it by default (400
  otherwise); still settable via env for other models, sent through `extra_body`
  so it works across SDK builds (this environment ships `anthropic` 1.0.0, whose
  `messages.create()` has no `temperature` keyword).
- The vision API **rejects sub-minimum images** ("Could not process image"), so
  `warmup()` uses a 64×64 solid PNG, not a 1×1.

Still pending: a full sim run (perception sidecar + ROS) driving the loop
end-to-end.

## Status

Scaffold complete, unit-tested against a fake engine, and **both Claude paths
live-validated** against Opus 5. Full sim run pending. Default
responder/strategy are unchanged (`dummy` / `frontier`) — this mode is fully
opt-in.
