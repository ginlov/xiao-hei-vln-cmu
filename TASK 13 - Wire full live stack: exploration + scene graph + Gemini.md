# TASK 13 - Wire the full live stack: frontier exploration → scene graph → Gemini

## Purpose

Turn the three layers that already existed in isolation into one live
pipeline so the `gemini` responder actually answers over a scene graph
built during exploration:

1. **Navigation** — app-level frontier exploration drives the robot.
2. **Scene graph** — YOLO+SAM detections are lifted to 3D and added to a
   shared `SceneRepresentation` *during* the sweep.
3. **Question answering** — Gemini reasons over that fully-built graph
   (plus the panorama + occupancy image) once exploration completes.

## Problem (before)

The live `gemini` path never realized the described stack:

- `GeminiResponder` built its **own** empty-object `SceneRepresentation`
  and its **own** internal frontier explorer
  (`perception_responder.PerceptionResponder`, geometry-only), ignoring
  the shared `scene` that `app/main.py` maintains.
- That internal explorer **never runs object detection** (no
  `add_object`), so the scene graph Gemini received live had Room +
  Viewpoints but **zero objects**. Gemini answered from images +
  viewpoint positions only.
- Because `GeminiResponder` exposed no `ingest()`, the app-level
  exploration loop skipped it. With the default
  `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=100`, the robot therefore explored
  **twice**: the app-level `FrontierExplorer` swept first (building the
  shared scene that Gemini discarded), then `GeminiResponder` ran its own
  second frontier sweep before answering.

The complete object pipeline (`detect → lift → add_object`) only existed
in the separate `perception` responder and in the **offline** batch
evaluator (`gemini/batch.py`, reconstructed from the GT `object_list`).

## Changes

### 1. `GeminiResponder` — two operating modes (`gemini/responder.py`)

- **External-scene mode** (new, production): the constructor now accepts
  the shared `scene` and a duck-typed `perception_ingestor` (the YOLO+SAM
  `perception.responder.PerceptionResponder` bound to that scene). New
  `ingest(snapshot)` — called every exploration tick by `app/main.py` —
  updates the occupancy `GlobalMap` + trajectory and delegates object
  detection to the ingestor. It emits **no** answer, so a question
  arriving mid-sweep stays deferred (matches TASK 12).
  `_should_commit` short-circuits to `True` in this mode: the app-level
  sweep has already run to completion before `respond()` is first called,
  so Gemini commits on the first tick over the fully-built graph — **no
  second exploration**. `reset()` preserves the shared scene / map /
  trajectory across questions (the sweep runs once per session).
- **Self-paced mode** (unchanged): when no `scene` is passed, the
  responder keeps its private scene + internal frontier fallback and the
  original `max_explore_ticks` trigger. Preserves existing behavior for
  unit tests and any caller without an app-level explorer.

### 2. `app/main.py` — gemini branch builds the ingestor

The `gemini` branch of `_build_responder` now builds the perception
client / lifter / vocab (same as the `perception` branch), wraps them in
a `PerceptionResponder` bound to the **shared** `scene`, and passes both
`scene=` and `perception_ingestor=` into `GeminiResponder`. This makes
`hasattr(responder, "ingest")` true, so the **existing** exploration loop
(unchanged since TASK 12) feeds it every tick. No tick-loop edits needed.

### 3. `docker/compose_gemini.yml` — perception sidecar added

The live Gemini path now depends on the YOLO+SAM sidecar, so:

- `ai_module` builds with `XIAO_HEI_EXTRA=gemini,perception` (adds the
  perception HTTP-client deps: httpx / pillow / pycocotools).
- New `XIAO_HEI_PERCEPTION_BASE_URL` env + `depends_on: perception` +
  an `exploration_logs` mount.
- New `perception` service (GPU) mirroring the one in `compose.yml`.

Full stack is now: `system` (sim) + `perception` (YOLO+SAM) + `ai_module`
(gemini).

## Files touched

- `src/xiao_hei_vln/gemini/responder.py` — external/self-paced modes,
  `ingest()`, commit-on-first-tick, scene-preserving `reset()`,
  ingestor-closing `close()`.
- `src/xiao_hei_vln/app/main.py` — gemini branch wires the shared-scene
  ingestor.
- `docker/compose_gemini.yml` — perception sidecar + extra + env.
- `tests/test_gemini_responder.py` — new `TestExternalSceneMode` (5
  tests) covering ingest-without-answering, commit-on-first-respond,
  shared-scene identity, reset-preserves-scene, close-releases-ingestor,
  and external-mode Gemini-failure (no fallback).

## Verification

- `uv run pytest -q` → **418 passed, 1 skipped** (was 413; +5 new tests).
- `ruff check` on changed source → **no newly introduced violations**
  (baseline pre-existing E501/SIM115/SIM103 unchanged; the 3 new import
  blocks were auto-sorted).

**Not exercised here** (needs the challenge box): the true end-to-end
live run — GPU perception sidecar + Unity sim + a real
`XIAO_HEI_GEMINI_API_KEY`. Smoke-test checklist for the box:

1. `export XIAO_HEI_GEMINI_API_KEY=…` and bring up `compose_gemini.yml`.
2. Watch `exploration_logs/exploration.log` for `START … DONE` (a full
   sweep completing).
3. Confirm the per-question `vlm_logs/` tick shows a scene-graph JSON
   with a **non-empty `objects` list** in the Gemini user prompt.
4. Confirm exactly **one** Gemini call per Task-1 question, fired on the
   first post-exploration tick.
