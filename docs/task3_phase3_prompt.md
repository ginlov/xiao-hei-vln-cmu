# Task 3 — Phase 3: Numerical question prompt design

!!! warning "Historical record"

    This document describes the **retired `qwen` responder** (Qwen3.5 via a
    vLLM sidecar), removed in
    [TASK 17](tasks/TASK%2017%20-%20Retire%20the%20qwen%20responder.md).
    It is kept for the recorded rationale — the sidecar-vs-in-process
    reasoning here still informs how the perception sidecar is structured.
    Nothing on this page describes code that currently exists.

This document records the prompt + responder-loop design used to
answer the **numerical** ("How many ...") question type with the
Qwen3.5 responder. The other two question types continue to use the
generic Phase-2 prompt; this document and the matching code path are
specific to numerical.

## Problem shape

For a numerical question the robot is, in general, **not** able to
answer from its spawn pose — the target objects are distributed
around a scene that requires the robot to translate and look in
multiple directions before it has full coverage.

The constraints from earlier tasks fix the operating envelope:

- The VLM is called once per tick (default 2 Hz = 500 ms).
- Each call sees one fresh camera frame, the robot's pose, and any
  earlier reasoning we choose to carry over.
- The only output the evaluation node scores is a **single integer**
  published on `/numerical_response`.
- The same `/way_point_with_heading` topic is used by both
  instruction-following navigation and (here) by the model to ask the
  base autonomy to take the robot to a better viewpoint before
  answering.

So the model needs to choose, on every tick, between two actions:

1. **Explore** — emit a `WaypointPathResponse` toward a new viewpoint.
2. **Commit** — emit a `NumericalResponse` with the final integer.

## Cross-tick state

vLLM is invoked once per tick with a fresh KV cache (we don't keep a
persistent conversation), so anything the model needs across ticks
must travel through the prompt. We use the optional `rationale: str |
None` field that was added to every `VLMOutput` variant in Phase 3:

- The field is part of the JSON schema fed to `guided_decoding`, so
  the model is required to fill it on every reply.
- The publisher reads only `value` / `label` / `center` / `size` /
  `waypoints` and is oblivious to `rationale`; nothing about the ROS
  contract changes.
- `QwenResponder._record_evidence()` appends the rationale (verbatim,
  prefixed by tick number) to the evidence log when the question is
  numerical, and the next tick's user prompt includes that log.

The model is asked to begin every rationale with a parseable triplet:

```
view_count=<int> running_total=<int> action=<explore|commit> <free text>
```

This is the running tally the next tick will read. We don't
post-process it — the model is responsible for keeping its own
arithmetic consistent. (Phase 4 may add a parser + sanity-check on
top, but that's out of scope here.)

## The two prompts

### System prompt (`NUMERICAL_SYSTEM_PROMPT`)

Defined in `src/xiao_hei_vln/qwen/prompts.py`. It states:

- The role and the I/O contract (image + pose in, JSON out).
- The 4-step per-tick protocol: estimate `view_count`, recover prior
  `running_total`, decide explore vs commit, emit JSON.
- A concrete commit predicate the model can apply: ≥ 2 distinct
  viewpoints AND last two tallies agree AND current frame shows
  nothing new.
- The exact JSON shape for each of the two actions, with the
  `view_count=... running_total=... action=...` rationale convention.
- Waypoint constraints (≤ 5 m from current pose so the local terrain
  map can certify reachability, distinct from prior viewpoints,
  `heading` in radians facing the inspection sector).

### User prompt (`build_numerical_user_message`)

Built fresh on every tick from the live snapshot. It carries:

1. The question text and a one-line pose summary (so the model knows
   where it is in map coordinates).
2. A one-line terrain / lidar summary (point counts at 5 m and 20 m
   ranges) so the model knows whether reachability info is present.
3. **Tick budget**: `Tick {i} of at most {N}`. On the final tick the
   prompt switches to `BUDGET EXHAUSTED: you MUST commit on this
   tick`. This is the model-visible side of
   `QwenConfig.max_ticks_per_question`.
4. The full prior-rationale evidence log (oldest → newest), with an
   explicit nudge to parse the most recent `view_count=` /
   `running_total=` line. On the first tick we instead print
   `This is the FIRST tick on this question — initialise your tally.`
5. A closing `Respond now with the JSON object.`

## Responder loop

`QwenResponder.respond()` dispatches by question type:

- `QuestionType.NUMERICAL` → numerical system prompt + numerical user
  prompt + rationale-only evidence capture.
- Other types → Phase-2 generic prompt + coarse evidence summary
  (with rationale appended after `::` when present).

Terminal logic is unchanged from Phase 2: a `NumericalResponse` sets
`is_done = True`; a `WaypointPathResponse` keeps the loop alive (the
publisher emits the first waypoint of the path and waits for
`/way_point_reached` to advance the base autonomy).

### Safety net

If the model never commits within `max_ticks_per_question` ticks
(default 30, i.e. 15 s at 2 Hz) the responder forces a
`NumericalResponse(value=0, rationale="timeout after N ticks")`.
This guarantees the evaluation node always sees an answer rather than
silence; a confidently-wrong zero is still strictly better-scoring
than no emission. The threshold is env-overridable
(`XIAO_HEI_QWEN_MAX_TICKS`).

The model's own prompt is kept in sync with this budget: the user
message always shows `Tick i of at most N`, and the final tick
switches to `BUDGET EXHAUSTED` wording so the model itself is
incentivised to commit before the responder's fallback fires.

## What we deliberately did NOT add

- **No frontier-based waypoint generation in the responder.** We rely
  on the model to propose waypoints from the image + pose + terrain
  summary. A geometric planner could be layered on later if
  measurements show the model wanders off-map, but the responder
  itself stays prompt-driven.
- **No external rationale parser.** The triplet (`view_count=...`
  etc.) is purely a convention the model uses to talk to its future
  self. We never validate it on the responder side.
- **No per-tick conversation history inside vLLM.** Each call is a
  fresh `LLM.generate()` with the evidence log embedded in the
  prompt. This keeps the engine wrapper stateless and avoids the
  KV-cache management work that a real chat would require.
- **No multi-image context.** Qwen3.5 supports multi-image input but
  we cap `limit_mm_per_prompt={"image": 1}` because the tick budget
  doesn't leave room for prefill on multiple images, and the evidence
  log already covers cross-tick memory.

## Open follow-ups

- Validate the budget on the real eval host — Phase 1 numbers are
  estimates. If the 4B model runs faster than budgeted we can raise
  `max_ticks_per_question` or lower it depending on accuracy/latency
  trade-off measurements.
- A small parser around the `view_count=... / running_total=...`
  triplet would let us surface drift (model says `running_total=3`
  one tick and `running_total=5` the next without seeing more
  objects) as a warning in the logs.
- The commit predicate is a heuristic; if the eval shows the model
  commits too eagerly or too cautiously, the predicate is a single
  paragraph in `NUMERICAL_SYSTEM_PROMPT` and is the cheapest knob to
  tune.
