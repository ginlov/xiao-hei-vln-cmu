# TASK 15 - Retire the unreachable Phase-A exploration stack

## Purpose

Delete the dead second frontier-exploration implementation and the legacy
`perception_responder/` package it backed. Both were superseded by the
Task 12/14 pipeline (`FrontierExplorer` driving `scene_gemini`) and were no
longer reachable from any shipping code path.

## Problem (before)

Two parallel frontier implementations coexisted with no shared code:

| | live path | retired path |
|---|---|---|
| strategy | `exploration/_frontier.py` → `FrontierExplorer` | `exploration/frontier.py` → `FrontierPlanner` |
| map | `exploration/_grid.py` → `OccupancyGrid` | `perception/global_map.py` → `GlobalMap` |
| driver | `app/main.py` tick loop | `perception_responder/responder.py` |

`perception_responder` also exported a class named **`PerceptionResponder`** —
the same name as the live sidecar-backed `perception.PerceptionResponder`, so
the two were trivially confusable at an import site.

Reachability: `exploration/frontier.py` was imported only by
`perception_responder/responder.py` and its own test;
`perception_responder/` was imported only by
`scripts/visualize_explore_trajectory.py`, which was itself referenced by no
doc, test, or CI job.

## What changed

Pure deletion — no behaviour change to any shipping path.

Deleted:

- `src/xiao_hei_vln/perception_responder/` (`__init__.py`, `responder.py`)
- `src/xiao_hei_vln/exploration/frontier.py`
  (`FrontierPlanner`, `ScoredFrontier`, `ScoringWeights`)
- `tests/test_frontier_planner.py` (7 tests)
- `scripts/visualize_explore_trajectory.py` — synthetic-room visual harness
  for the retired responder; unreferenced anywhere
- `docs/assets/exploration/synthetic_livingroom.png` — its only output,
  linked from no page

Kept deliberately:

- `perception/global_map.py` — `GlobalMap` is still used by
  `scene_gemini/responder.py` and `gemini/scene_rep.py` (occupancy PNG for
  the Gemini prompt).
- `exploration/_frontier.py`, `_grid.py`, `_visualize.py` — the live path.

## Gotcha worth remembering

`git rm -r` deletes the tracked files but leaves the **empty directory** on
disk. Because `src/` is on the path, Python then resolved
`xiao_hei_vln.perception_responder` as an implicit namespace package
(PEP 420) — `import` succeeded silently and returned an empty module instead
of raising `ModuleNotFoundError`. Any deletion of a package directory must be
followed by removing the empty dir, or the retirement is invisible to a
smoke-test import.

## Verification

- `uv run pytest -q` → **454 passed, 1 skipped** (was 461; the 7 removed are
  exactly `test_frontier_planner.py`).
- `import xiao_hei_vln.perception_responder` and
  `import xiao_hei_vln.exploration.frontier` both raise `ModuleNotFoundError`.
- `xiao_hei_vln.exploration` still exports `ExplorationStrategy`,
  `FrontierExplorer`, `save_exploration_plot`; `FrontierExplorer()` still
  satisfies the `ExplorationStrategy` protocol at runtime.
- Repo-wide grep for the removed symbols across `src tests scripts docker docs
  perception dataset_generator config pyproject.toml mkdocs.yml README.md` is
  clean. (Historical `TASK 8` / `TASK 11` reports still mention them; those are
  point-in-time records and were left as written.)

## Follow-ups this opens

- `GlobalMap.find_frontier_clusters`, `FrontierCluster`,
  `compute_reachable_mask`, `is_traversable` and `line_passes_through` now have
  **no production caller** — only `tests/test_global_map.py` exercises them.
  `GlobalMap` itself must stay, but that surface could be trimmed as a separate
  decision.
- The live `FrontierExplorer` has no offline visual harness now. The in-run PNG
  (`save_exploration_plot`, via `XIAO_HEI_EXPLORATION_LOG_DIR`) remains the
  only visualisation.
