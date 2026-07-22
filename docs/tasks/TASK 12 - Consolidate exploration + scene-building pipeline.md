# TASK 12 - Consolidate exploration + scene-building pipeline

## Purpose

Unify the two halves of the intended runtime path — **frontier exploration →
scene representation built on the fly** — and make exploration run to its own
completion instead of pausing when a question arrives.

## Problem (before)

In `app/main.py` the tick loop made exploration and scene-building **mutually
exclusive**, gated on whether a question was active:

- **No question active** → exploration branch ran, published a frontier
  waypoint, then `return`ed early. It never called `scene.update()` and never
  ran perception, so the scene graph stayed empty throughout exploration.
- **Question arrives** → exploration was *paused* for that tick and every
  subsequent tick; the responder took over immediately and answered from
  whatever was visible at that moment. `PerceptionResponder` "Phase A"
  additionally walked a precomputed Task-7 trajectory JSON.

Net effect: the scene was never built during exploration, and a question cut
exploration short.

## Desired behavior

1. Build the scene graph **during** frontier exploration.
2. A question arriving mid-exploration must **not** stop the sweep. Exploration
   runs until it completes on its own terms (**budget kept**: budget exhausted,
   consecutive-skip hatch, or no frontiers remain). The answer is **deferred**
   until exploration finishes, then produced from the fully-built scene.

## Changes

### 1. Scene built on the fly, without answering — `perception/responder.py`

Added `PerceptionResponder.ingest(snapshot)` — runs only the
detect → lift → add_object cycle (`_inject_visible`) and emits **no** answer,
even when a question is already active. A pending question's text still flows
into the detector vocabulary, so the queried object is actively searched for
during the remaining sweep.

### 2. Exploration is never interrupted — `app/main.py`

The exploration branch condition dropped its `snapshot.question is None` gate:

```python
if explorer is not None and not explorer.is_complete():
    scene.update(snapshot)                 # viewpoint + scene-bounds nodes
    if hasattr(responder, "ingest"):
        responder.ingest(snapshot)         # perception scene-building, no answer
    ... run explorer, publish waypoint ...
    return
```

Because the branch still `return`s, the responder's answering path below is
only reached once `explorer.is_complete()` is `True`. So a question that
arrives mid-sweep is held; the responder resets and answers from the complete
scene graph on the first post-exploration tick. Responders without a scene
path (`dummy`/`qwen`) simply don't expose `ingest()` and are skipped.

Intended deployment runs **without** `XIAO_HEI_TRAJECTORY_JSON`, so the
perception responder skips Phase A (trajectory walk) and answers straight from
the accumulated graph — exploration has already done the movement.

### 3. Budget limit kept

The waypoint budget and consecutive-skip hatch are unchanged from `main`:
`FrontierExplorer` still stops on `budget_exhausted` (`max_waypoints`, default
100), `max_consecutive_skips` (20), or `no_frontiers`. `_frontier.py` is
byte-for-byte identical to `main` — no changes to the algorithm's termination.

## Files touched

- `src/xiao_hei_vln/perception/responder.py` — new `ingest()` method.
- `src/xiao_hei_vln/app/main.py` — exploration no longer gated on question
  absence; scene-building via `scene.update()` + `responder.ingest()`.
- `docs/concepts/exploration.md` — "Questions do not interrupt exploration" +
  "Scene building on the fly" sections.
- `docs/getting-started/configuration.md`, `docs/concepts/frontier-explorer.md`
  — note the deferred-answer behavior; budget defaults unchanged.

## Behavior notes / trade-offs

- Applies uniformly to all question types. For numerical / object-reference
  this is ideal (full coverage before answering). For instruction-following
  the robot completes the frontier sweep before executing the instruction —
  acceptable given the explicit "don't stop exploration" requirement, but worth
  revisiting if instruction latency matters.

## Verification

- `uv run pytest -q` → **309 passed, 1 skipped** (studio-zip test).
- `git diff` confirms `_frontier.py` unchanged vs `main` (budget intact).
- `ruff check` — no newly introduced violations on the changed source.

Not exercised here: full end-to-end run against the live sim + perception
sidecar (requires GPU containers, handled separately).
