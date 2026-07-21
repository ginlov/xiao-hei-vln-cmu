# TASK 14 - Consolidate the submission pipeline into the `scene_gemini` responder

## Purpose

Wire the three named submission components — **frontier exploration +
scene-graph building + Gemini question answering** — into a single coherent
responder, and retire the old `gemini` responder that was not actually wired to
either the shared explorer or the object scene graph.

## Problem (before)

The `gemini` responder (`GeminiResponder`) and the app-level pipeline were two
parallel, unwired designs:

- **Double exploration.** `compose_gemini.yml` left `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS`
  at its default (100), so the app-level `FrontierExplorer` ran to completion
  *before* `GeminiResponder.respond()` was ever called (the explore branch
  `return`s early and `GeminiResponder` has no `ingest()`). `GeminiResponder`
  then explored **again** internally (its own `perception_responder` +
  `GlobalMap`, up to `max_explore_ticks`) before calling Gemini. Redundant and
  slow, with two disconnected scenes.
- **Empty object graph.** `GeminiResponder` served an object-less scene: its
  internal explorer was pure geometric (no detector), and `compose_gemini.yml`
  ran **no perception sidecar**. Gemini had to infer everything from the
  panorama — weak for object-reference (which must return a 3D box) and
  counting.

## What changed

New responder **`SceneGeminiResponder`** (`XIAO_HEI_RESPONDER=scene_gemini`,
module `src/xiao_hei_vln/scene_gemini/`), driven by the **shared** app-level
explorer:

```
FrontierExplorer (app loop) drives movement
  └─ every tick → responder.ingest(snapshot)
        └─ perception sidecar (YOLO-World + SAM) → 3D lift → SceneRepresentation
  └─ on explorer.is_complete() → responder.respond(snapshot)
        └─ serialize the POPULATED scene graph + panorama + occupancy PNG → Gemini
```

- **One exploration pass.** The responder exposes `ingest()`, so the Task-12
  app loop feeds it during the sweep; it never runs its own explorer. `respond()`
  is only reached once the sweep is complete, so Task 1 asks Gemini **once** on
  the first answer tick and commits (no internal `max_explore_ticks` dance).
- **Real object graph.** `ingest()` delegates to the sidecar-backed
  `perception.PerceptionResponder`, writing labelled, 3D-positioned (and, with
  `XIAO_HEI_OBJECT_MAP=1`, 3D-boxed) objects into the **shared**
  `SceneRepresentation` — the same instance `app/main.py` owns and the same one
  serialized to Gemini. So the app-level `scene.update` (viewpoints/bounds) and
  the perception object layer always agree.
- **Task 2** unchanged in spirit: plan a route via Gemini once, step waypoints.
- **API-key leak fixed.** The scene_gemini logger config strips `api_key`
  before writing `session.json` (the old gemini branch wrote it verbatim).
- **rviz markers** now also publish for `scene_gemini` (fused object boxes on
  `/perception/objects`).

Reused as-is: `GeminiEngine` / `GeminiConfig` / prompts / `scene_rep`
(`build_bundle` + `serialize_for_gemini`) / `trace`, and the whole offline
`gemini.batch` evaluator.

## Retired

- `src/xiao_hei_vln/gemini/responder.py` (`GeminiResponder`) — **deleted**.
- `tests/test_gemini_responder.py` — deleted; replaced by `tests/test_scene_gemini.py`.
- `docker/compose_gemini.yml` — deleted (set the dead `gemini` responder).
- `gemini/__init__.py` no longer exports `GeminiResponder`; `app/main.py`'s
  `gemini` branch is replaced by the `scene_gemini` branch; the
  `expected one of:` error message updated.

## Files

New: `src/xiao_hei_vln/scene_gemini/__init__.py`, `.../responder.py`,
`tests/test_scene_gemini.py`, `docker/compose_scene_gemini.yml`.
Modified: `src/xiao_hei_vln/app/main.py`, `src/xiao_hei_vln/gemini/__init__.py`,
`.../gemini/engine.py` (docstring), `.../gemini/batch.py` (docstring),
`README.md`.
Deleted: `src/xiao_hei_vln/gemini/responder.py`,
`tests/test_gemini_responder.py`, `docker/compose_gemini.yml`.

## Verification

- `uv run pytest -q` → **461 passed, 1 skipped** (studio-zip test). New
  `test_scene_gemini.py` covers: `ingest` builds the shared scene without
  answering; Task-1 numerical/object-reference commit on the first answer tick
  with the populated graph reaching Gemini; Task-2 plan-once-then-step;
  Gemini-failure hold+retry; logger lifecycle; reset.
- `ruff check` clean on the new source (`src/xiao_hei_vln/scene_gemini/`,
  `app/main.py`).
- `docker compose -f docker/compose_scene_gemini.yml config` validates.

## Not yet done

- **Tier-2 live end-to-end** on the sim (needs a valid `XIAO_HEI_GEMINI_API_KEY`
  and the perception + sim containers sharing the GPU). This is the real
  submission smoke test: bring up `compose_scene_gemini.yml`, pub one question
  per task type, confirm the answer routes to the right topic and
  `predictions.jsonl` is written.
- Deeper `docs/` pages still describe the old `gemini` responder
  architecturally (no broken run instructions — those were README-only).
