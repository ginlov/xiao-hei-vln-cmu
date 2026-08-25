# TASK 38 — VLM-based navigation waypoint proposer (Opus 5)

## Purpose

Replace the geometric exploration strategy (frontier / NBV) with a
**vision-language model that proposes the next waypoint**, while keeping the
existing local planner, the `/way_point_with_heading` interface, and the whole
capture + offline-eval harness untouched. This is the "waypoint proposer only"
scope — the VLM decides *where to go next*; the autonomy stack still drives
there and does obstacle avoidance.

Motivation: the NBV explorer *wedges* — it repeatedly proposes waypoints the
local planner cannot reach (nav distance never drops below ~1.1 m; runs reached
0–4 of ~19 waypoints, with high run-to-run variance). A model that reasons over
the actual view + occupancy map, and that gets told when a target was
unreachable, can re-plan around dead zones instead of re-proposing them.

The Anthropic key is **not required to build, import, or unit-test** any of
this — it is only needed at runtime when the `nav_vlm` strategy is selected.

## What was built

New package `src/xiao_hei_vln/nav_vlm/` (backend-agnostic):

| Module | Role |
|---|---|
| `config.py` | `NavVLMConfig` — env-var knobs. Key from `ANTHROPIC_API_KEY` (fallback `XIAO_HEI_ANTHROPIC_API_KEY`); everything else `XIAO_HEI_NAV_VLM_*`. `from_env` only fails on a missing key, and only when called. |
| `engine.py` | `NavVLMEngineProtocol` (one method: `propose(...) -> WaypointProposal`) + `AnthropicNavEngine` (Opus 5 via forced `propose_waypoint` tool-use; lazy `anthropic` import; fake client injectable for tests). |
| `prompts.py` | System prompt + the `propose_waypoint(done, x, y, heading, rationale)` tool schema, and the per-call user text (pose + failure context). |
| `render.py` | Top-down occupancy PNG (free/occupied/unknown + robot + path + failed-waypoint star) and panorama JPEG — the spatial grounding, rendered off the exploration `OccupancyGrid` so it works even in the dummy/capture path. |

New strategy `src/xiao_hei_vln/exploration/_nav_vlm.py` — `NavVLMExplorer`,
a drop-in `ExplorationStrategy` that mirrors the app-facing surface the tick
loop drives (`advance()`, `force_skip()`, `_current_target`, `_visited`,
`skipped_count`, `_consecutive_skip_count`, `_max_waypoints`,
`_max_consecutive_skips`). Selected with
`XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm` (wired in `app/main.py:_build_explorer`,
fails soft to "exploration disabled" if the key is absent).

## Design — trigger policy

The model is asked for a waypoint **exactly once per waypoint outcome**, never
on a timer. Every waypoint's life ends in exactly one of these, so this is the
tightest possible policy:

- **cold start** — one call at episode start (no target yet);
- **reached** — the app calls `advance()` (nav stack settled on target);
- **cannot reach** — the app calls `force_skip()` (planner blocked short), OR
  the explorer's internal **stuck watchdog** fires (liveness backstop, so a
  silent stall still triggers a re-plan).

The call runs on a background thread (`ThreadPoolExecutor`, `max_workers=1`), so
the 2 Hz control loop never blocks on the multi-second Opus 5 round trip. While
a call is in flight `update()` returns `None` and the robot holds at its last
position.

## Two safeguards the geometric explorers lacked

1. **Reachability snapping.** Every proposal is snapped to the nearest
   grid-reachable FREE cell — BFS over free space from the current pose
   (`OccupancyGrid.reachable_path_costs`) — within a hop horizon (`max_hop_m`,
   defaults to `XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST`). The model cannot send
   the robot into a wall or an unreachable pocket. This directly attacks the NBV
   wedge.
2. **Failure feedback.** A cannot-reach outcome blacklists the dead zone and
   feeds the reason (`"target (x,y) was blocked …"`) into the next prompt, so
   the model re-plans around it instead of re-proposing it.

Termination: model `done=true`, waypoint budget exhausted, too many consecutive
skips, or too many failed/unreachable proposals in a row.

## Hyperparameters (defaults)

Model: `claude-opus-5`, temperature **omitted by default** (Opus 5 deprecates
`temperature` and the API 400s on it — set `XIAO_HEI_NAV_VLM_TEMPERATURE` only
for a model that still accepts it; it is sent via `extra_body`),
`max_output_tokens` 1024, thinking disabled (forced tool-use is incompatible
with extended thinking), panorama long-edge 1024 px, occupancy DPI 100, request
timeout 60 s.

Explorer: `max_hop_m` 3.0 m, `stuck_timeout_s` 25 s, `max_consecutive_skips` 6,
`max_propose_failures` 4, grid resolution 0.2 m.

## Tests (no key / no network)

- `tests/test_nav_vlm_config.py` — env parsing, missing-key raise, fallback key.
- `tests/test_nav_vlm_engine.py` — tool-use parsing, forced-tool request shape,
  1-vs-2 image blocks, missing-tool raise, occupancy-render smoke.
- `tests/test_nav_vlm_explorer.py` — cold start, reachability snapping,
  reached→re-propose, force-skip failure-feedback, watchdog skip, model-done,
  consecutive-skip + budget termination, async hold-then-deliver.

All pass; the package imports with `anthropic` **not** installed.

## To go live

1. Provide `ANTHROPIC_API_KEY` (env or repo `.env`).
2. Install the extra: `uv sync --extra nav-vlm` (adds `anthropic`).
3. Select it: `XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm`.
4. Capture / eval with the existing harness (`run_nav_capture.sh` accepts
   `STRATEGY=nav_vlm`).

## Status

Scaffold complete and tested against a fake engine. Live validation pending the
Anthropic key. Default strategy remains `frontier` — `nav_vlm` is opt-in.
