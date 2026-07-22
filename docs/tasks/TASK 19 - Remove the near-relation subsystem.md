# TASK 19 - Remove the near-relation subsystem

## Purpose

Delete the Object↔Object spatial-relation machinery — `SpatialRelation`,
`spatial_relations`, `add_spatial_relation*`, `derive_near_relations`, the
`near_threshold` plumbing and the renderer's `near` features. Nothing consumed
the edges.

## Why it was safe

Traced every producer and consumer before touching anything:

**Producers.** `derive_near_relations` was the *only* thing in the codebase
that ever created a spatial relation. The generic `add_spatial_relation` /
`add_spatial_relation_by_index` primitives were never called by production
code — only from inside `derive_near_relations` and from their own tests.

**Consumers — none that affect an answer:**

| path | does it use `near` edges? |
|---|---|
| live `scene_gemini` (the submission stack) | **No** — never calls `derive_near_relations` at all; `to_dict()["objects"][i]["spatial_relations"]` was always `[]` |
| offline `gemini.batch` | **No** — `build_scene` derived edges, but the prompt is built by `_compact_objects()`, which emits only `id`/`label`/`center`/`size`. The edges reached only the `--debug-dir` dump |
| `perception` responder Phase B | **No** — `_answer_numerical` counts by label match; `_answer_object_reference` sorts candidates by XY distance to pose. Neither reads relations. The `derive_near_relations` call existed, by its own comment, so "the logged snapshot has them" |
| `scene/render.py` | Yes — but debug visualisation only |

The committed artifact confirms it: `dataset/detected/livingroom_3_scene_graph.json`,
produced by the live pipeline, has `"spatial_relations": []` on every object.

So the subsystem was a **debug/visualisation feature**, not answer logic.
Proximity is inferred from object coordinates, which every consumer already
has — which is exactly what the offline prompt already did deliberately.

## What changed

**Removed** — `scene/representation.py`: `SpatialRelation` dataclass, the
`spatial_relations` field, its carry-over on object re-add, all three methods,
and the `to_dict()` key. `scene/__init__.py`: the `SpatialRelation` re-export.
`scene/render.py`: `_draw_near_arcs`, `_draw_local_near_edges`,
`_near_summary`, the `near_edge_radius` parameter, the `near` table column, and
the `near-edges` counters in both plot titles. `perception/responder.py`:
`DEFAULT_NEAR_THRESHOLD`, the `near_threshold` parameter and the derive call.
`gemini/batch.py`: the `near_threshold` parameters and the `--near-threshold`
CLI flag. `app/main.py`: `near_threshold` from `_PerceptionSettings` and the
log config.

Three of the removed render helpers (`_draw_near_arcs`,
`_draw_local_near_edges`, `near_edge_radius`) were **already dead** — defined
but never called, before this task.

**Schema change.** `SceneRepresentation.to_dict()` no longer emits a
`spatial_relations` key. It was always `[]` in the live path, so the JSON sent
to Gemini loses nothing but a few tokens. The committed
`livingroom_3_scene_graph.json` still carries the old empty key; readers use
`.get()`, so it stays loadable.

## Two mistakes I made and caught

Both came from deleting code by "slice from this definition to the next one",
which is not safe when the thing being removed is adjacent to something else:

1. **`SCENE_TABLE_CSS` was destroyed.** Deleting `_near_summary` up to the next
   `def` also swallowed the `# Helpers` banner and the `SCENE_TABLE_CSS`
   constant that sat between them. Caught by an `ImportError` in
   `test_scene_render.py`; restored verbatim.
2. **Eight unrelated tests were destroyed.** `TestEdgeStorage` was a *mixed*
   class — Room→Viewpoint and Viewpoint→Object edge tests, *then* the
   Object→Object ones. Deleting the class removed all 8 viewpoint-edge tests
   too. Restored by taking the original class up to its
   `# Object → Object (spatial_relations)` section marker.

The second would have passed CI silently — the tests simply cease to exist, so
nothing goes red. It was caught by diffing the set of `class`/`def` names
against `HEAD` rather than trusting the suite to stay green. That diff is the
check worth repeating on any deletion of this shape.

## Verification

- `uv run pytest -q` → **405 passed, 1 skipped**. Baseline was 427; the 22-test
  delta is exactly the relation tests (`TestDeriveNearRelations`,
  `TestRelationTargetObjectId`, the Object→Object half of `TestEdgeStorage`,
  `test_build_scene_derives_near_relations`, and the near-threshold assertions
  in `test_perception_settings.py`).
- Definition-set diff against `HEAD` for every edited source and test file:
  only near-related names removed, nothing added, nothing else lost.
- `ruff check src` — findings unchanged from before the task (all pre-existing).
- `mkdocs build --strict` → exit 0.
- Repo-wide grep for `near_threshold` / `derive_near` / `spatial_relation` is
  clean across `src`, `tests`, `scripts`, `perception` and all live docs.

## Deliberately kept

`dataset_generator`'s `"near"` is a **different thing** — it is the challenge's
own question vocabulary ("How many X are near the Y?", `vla3d_num_gen.py`
template N2, the `REL_CAP`/`_WALL_RELS` relation sets). That is dataset
semantics required to generate and score the official question types, and is
untouched.
