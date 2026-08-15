# TASK 52 — Locking the exploration config, and switching the default to nbv

Two changes: every exploration hyperparameter moves out of the source and into
`config/exploration.env`, and the default strategy becomes `nbv`.

The first is uncontroversial plumbing. **The second rests on weaker evidence
than anything else in this series, and this report says so plainly** so that
whoever reads it later can weigh it properly.

## The default change, and what it is actually based on

On coverage — the metric used throughout TASK 50 and 51 — **frontier wins**:
2516 m² against nbv's 1657 m² over 13 scenes, and it wins the two large
multi-room scenes by 86% and 130%.

The case for nbv is that `free_m2` may be measuring the wrong thing. Comparing
`home_building_2`:

| | free_m2 | waypoints reached | m² mapped per place the robot stood |
|---|---|---|---|
| frontier | 617 | 3 | **206** |
| nbv | 268 | 9 | **30** |

Frontier maps ~7× more area per vantage point. Its coverage is real — 99% of its
free cells are reachable-connected — but it is **long-range LiDAR sweeping
across open space from a handful of positions**, not the robot getting near
things. The plots show it directly: frontier's `home_building_2` run is three
long straight legs with a thin scatter over a huge extent; nbv's is a connected
tree with visibly denser cells around the trajectory.

Which matters depends on what the map is for. Mapping the floor plan: frontier,
clearly. Observing objects well enough to answer questions about them:
plausibly nbv, because detection and segmentation need the object at usable
range, and a cell grazed from 8 m away is on the map without its contents being
recoverable.

**The challenge grades answers, not maps.** That is the reasoning behind the
switch.

### What would settle it, and has not been run

Run the responder against a frontier map and an nbv map of the same scene, on
the same questions, and compare answer accuracy. Until that exists this default
is a judgement call from a density argument and one ratio — not a measurement.
If it turns out frontier answers as well or better, revert the default; nothing
else in this branch depends on the choice.

## The config lock

Every knob the explorers take is now read from the environment, defaults living
in `config/exploration.env`, which all three compose files load via `env_file`.
Nothing is hard-coded at the call site. Porting to the submission repo is
copying that one file.

Precedence is `environment:` → shell → `config/exploration.env`.

One trap, verified against docker compose rather than assumed. A **bare**
`- VAR` entry under `environment:` resolves to an **empty string** when the
shell has that variable unset, and it overrides `env_file`:

```
env_file: FOO=from_env_file, environment: [- FOO], shell unset
  -> container sees FOO=""      (not "from_env_file")
```

For `XIAO_HEI_EXPLORATION_STRATEGY` an empty value is not a recognised strategy,
so exploration would be **disabled with an error** on every run — silently, in
the sense that the sweep still produces logs and a CSV. The compose entries
therefore keep their `${VAR:-default}` form, which means the handful the sweep
scripts override are declared twice. `tests/test_exploration_config.py` asserts
those defaults still agree with the env file, and that no bare pass-through
creeps back in.

`compose.eval.yml` and `compose_scene_gemini.yml` deliberately cap the sweep
harder than a benchmarking run (fewer waypoints, and 540 s for eval). Those
divergences are enumerated in `DECLARED_OVERRIDES` in the test, so a deliberate
difference is a one-line edit and an accidental one fails.

## Tests

`tests/test_exploration_config.py` — 12 tests: the env file parses and covers
every knob `main.py` reads; each compose file loads it; declared defaults agree;
no bare pass-through; every hyperparameter is settable from the environment; and
no profile defaults to a strategy other than nbv, since a comparison across
profiles running different algorithms would mean nothing.

Full suite: 541 passed.
