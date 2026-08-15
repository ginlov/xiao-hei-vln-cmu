# TASK 48 - Artefact paths keyed on scene and strategy

## Purpose

Stop exploration runs from destroying each other's evidence.

## Problem

Every run wrote to `<log dir>/<scene>/` with three fixed filenames —
`exploration.log`, `exploration.png`, `rviz.png`. Nothing in the path depended
on the algorithm, so a `frontier` run and an `nbv` run over the same scene
landed on the same three files.

The collision is a deletion, not a merge: `exploration.log` is opened with mode
`"w"` at node startup, so the *previous* run's log is destroyed the moment the
new container boots — before the new run has produced anything to replace it
with. Comparing two strategies therefore required remembering to copy artefacts
out between runs, and forgetting was unrecoverable.

Under the submission stack it was worse than per-strategy collision. Both
`compose.yml` and `compose.eval.yml` pass `XIAO_HEI_SCENE_DIR_HOST` through to
`ai_module`, but `compose_scene_gemini.yml` did not — it only reaches `system`,
via the `compose.scene.yml` mount overlay. So `_EXPLORATION_SCENE` fell back to
`default_scene` for *every* scene, and all runs of all strategies over all
scenes shared one directory.

## Change

### 1. Strategy in the path — `app/main.py`

```python
_EXPLORATION_RUN_LABEL = (
    _EXPLORATION_STRATEGY if _EXPLORATION_MAX_WAYPOINTS > 0 else "no_exploration"
)

def _exploration_dir() -> Path:
    return (
        Path(_EXPLORATION_LOG_DIR or "/exploration_logs")
        / _EXPLORATION_SCENE
        / _EXPLORATION_RUN_LABEL
    )
```

`exploration_logs/chinese_room/frontier/` and `.../nbv/`.

The `no_exploration` label matters: `XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0`
disables exploration but leaves `XIAO_HEI_EXPLORATION_STRATEGY` at its
`frontier` default, so without the guard a run that explores nothing would
claim `frontier/` and truncate a real frontier run's log.

### 2. Scene passthrough — `compose_scene_gemini.yml`

Added `XIAO_HEI_SCENE_DIR_HOST` to `ai_module`'s environment, matching the
other two compose files. The node never opens the path — it is a *host* path,
meaningful only to the sim's bind mount — it only takes the basename to name
the directory.

### 3. Consumers updated

Three scripts read these paths and would otherwise have looked in the wrong
place:

- `scripts/vla3d_eval_sim.sh` — `wait_exploration_done` polls
  `<scene>/${STRATEGY:-frontier}/exploration.log`. The default matches the
  node's, for callers that source this file without setting `STRATEGY`.
- `scripts/run_scene_vla3d_eval.sh` — the mkdir/rm/touch before a run, the
  `exploration.png` copy into `explored_scenes/`, and `scene_live.json` (both
  the `XIAO_HEI_SCENE_DUMP_PATH` export and the read back).
- `scripts/exploration_sweep.sh` — gained `STRATEGY` (default `frontier`),
  exports it so the sweep can drive either algorithm, and writes
  `results_<strategy>.csv` so the collation does not collide either.

## Files touched

- `src/xiao_hei_vln/app/main.py` — `_EXPLORATION_RUN_LABEL`, `_exploration_dir`.
- `docker/compose_scene_gemini.yml` — scene passthrough.
- `scripts/exploration_sweep.sh`, `scripts/run_scene_vla3d_eval.sh`,
  `scripts/vla3d_eval_sim.sh` — path updates.
- `tests/test_exploration_snapshot.py` — 3 new tests.
- `docs/concepts/exploration.md`, `docs/getting-started/configuration.md`,
  `docker/README.md` — documented; also corrected the configuration table's
  claim that "only `frontier` is currently implemented" and its note that an
  unknown strategy falls back (it disables exploration instead).

## Verification

- `uv run pytest -q --ignore=tests/test_execute_plan.py` → **494 passed,
  1 skipped**. (`test_execute_plan.py` fails to collect on a missing `cv2`,
  pre-existing and unrelated.)
- `bash -n` clean on all three scripts.
- `uv run ruff check` on the changed Python reports the same 4 pre-existing
  findings as `HEAD`.

Not exercised here: a live sweep. The path change is covered by unit tests; the
shell edits are covered by inspection and a syntax check.

## Migration note

Existing artefacts under `exploration_logs/<scene>/` stay where they are — the
node writes to the new nested location and never reads the old one. Anything
worth keeping from a previous run should be moved into a
`<scene>/<strategy>/` directory, or it will simply sit unused alongside the new
tree.
