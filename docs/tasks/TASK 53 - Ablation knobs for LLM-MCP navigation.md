# TASK 53 — Ablation knobs for LLM-MCP navigation

## Why

The paper's thesis is **LLM-MCP-based navigation**: the LLM drives the loop
through structured tool calls, over two channels — it *steers navigation*
(the waypoint tool) and it *steers perception* (naming objects it sees /
wants verified). To quantify each channel's contribution we need two clean,
env-gated ablations that keep the working `nav_task1` loop otherwise intact
(no swap to the unverified geometric explorers).

## What

Two independent knobs, both defaulting to the baseline behaviour.

### Ablation A — truly-naive navigation (`XIAO_HEI_NAV_RAW_REASONING=1`)

A genuinely naive LLM-navigation baseline: no MCP tool anywhere, and none of
the engineering scaffolding. **Vision steering and every scene/visual hint are
KEPT** — the naive-ness is the *lack of structure and guardrails*, not less
information. Concretely, with the flag on:

- **No MCP tool, navigation AND answering.** The waypoint proposer and the
  `answer_object_reference` step both go through `AnthropicNavEngine.call()`,
  which dispatches to `call_raw()` (free text → JSON parsed from a fenced
  block) instead of `call_tool()`. The format instruction is **derived from the
  tool's own JSON schema** so the raw and tool arms request identical fields.
  `call_raw` floors `max_tokens` at 4096 (the tool arm's small budget truncates
  free-text reasoning before the JSON) and logs `stop_reason`/block types on a
  parse miss (format-adherence failure = a measurable outcome).
- **No reachability snapping.** The model's raw waypoint is driven straight to
  the planner (`_apply_proposal_naive`) — a pick with no reachable path is
  exactly the naive failure mode we want to exhibit.
- **No reach/skip machinery.** `advance()`/`force_skip()` are inert; no
  stuck-watchdog, no blacklist, no failure feedback, no reach↔explore mode
  switch, no coverage-plateau backstop.
- **Fixed-cadence re-plan** instead of event-driven triggering
  (`XIAO_HEI_NAV_NAIVE_REPLAN_S`, default 5 s) — this is what "reach/skip
  triggers disappear" means, and it prevents an unreachable raw waypoint from
  stranding the robot until the time budget.

Stopping criteria (naive): model ARRIVED, the per-question nav budget
(`max_question_seconds`, default 540 s), the waypoint cap (`max_waypoints`,
default 100 — each re-plan counts), or 4 consecutive propose failures. The
plateau / skip-cap / watchdog stops of the full system are gone.

All of the above is gated on `self._naive` (= `config.raw_reasoning`), so with
the flag off the baseline path is byte-for-byte the original code.

### Ablation B — no perception steering

`XIAO_HEI_NAV_PERCEPTION_STEER=0`. Disables the navigator's in-loop control of
perception: no `visible_objects` vocab priming and no `verify_objects`
threshold-lowering. The detector runs on the static prior + question nouns
only.

- `NavTask1Explorer(perception_steer=...)`; the two `_absorb_*` calls in
  `_apply_proposal` are gated on it. Wired from env in `app/main.py`.

## How to run

Each ablation is one full 45-question run; output auto-tags so it never
clobbers the baseline (`artifacts/bench_task2_scenes_rawreason.jsonl`,
`..._nopercsteer.jsonl`):

    # baseline (already have it)
    ./scripts/benchmark_task2_scenes.sh

    # Ablation A
    XIAO_HEI_NAV_RAW_REASONING=1 ./scripts/benchmark_task2_scenes.sh

    # Ablation B
    XIAO_HEI_NAV_PERCEPTION_STEER=0 ./scripts/benchmark_task2_scenes.sh

The flags are forwarded to the ai_module in `docker/compose_scene_gemini.yml`;
because `restart_ai` uses `docker restart` (baked env), the flag must be
exported **before** the first `docker/run scene_claude up -d` of the run — the
benchmark's per-scene bring-up handles that.

## Testing

`uv run pytest -q --ignore=tests/test_execute_plan.py` → 586 passed. New:
- `tests/test_nav_vlm_engine.py`: fenced-JSON parse, unfenced fallback,
  no-tool request shape, unparseable→raises, env flag, and `call()` dispatch
  (raw vs. tool).
- `tests/test_nav_task1_explorer.py`: steer-off makes no vocab/threshold edits;
  steer-on primes vocab and lowers the threshold one step; naive drives the raw
  waypoint unsnapped; naive ignores reach/skip events; naive re-plans on the
  cadence; baseline (flag off) still snaps.
- `tests/test_scene_claude_responder.py`: fake engine updated for `call()`.
