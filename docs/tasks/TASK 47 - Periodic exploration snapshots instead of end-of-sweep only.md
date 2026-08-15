# TASK 47 - Periodic exploration snapshots instead of end-of-sweep only

## Purpose

Make `exploration.png` and `rviz.png` survive a run that never reaches `DONE`.

## Problem

Both images were written in exactly one place: the `if explorer.is_complete():`
branch of the tick loop (`app/main.py`). That branch only fires when the
strategy terminates on its own terms — budget exhausted, consecutive-skip
hatch, or no frontiers left.

Every other way a run ends produced no image at all:

- torn down mid-sweep (`docker/run down`, container recreate, Ctrl-C),
- a sweep that never terminates,
- a crash or an OOM in the responder path.

Those are disproportionately the runs worth looking at, and the text log — which
*is* written continuously, line-buffered — showed the waypoint sequence but not
the map. The failure was silent: no warning, just a missing file.

The overwrite behaviour made it worse. `exploration.log` is opened with mode
`"w"` and every artefact lands in `<log dir>/<scene>/`, so starting the next run
destroys the previous run's log immediately and its PNGs at the next `DONE`.
Losing the images to an early teardown therefore meant losing them for good.

## Change

### 1. `_publish_atomically(out_path, write)` — `app/main.py`

Runs `write(tmp)` then renames tmp over the target. Snapshots are read while
the run is still going (an `scp`, an auto-reloading image viewer), so a
half-written PNG must never be observable.

The temp name keeps the `.png` suffix (`.exploration.partial.png`): both
matplotlib and pillow infer the output format from the extension, so a
`.tmp` suffix would raise instead of writing. A failed write leaves the
previous good snapshot in place, since the rename never happens.

### 2. `quiet=` on both save helpers

`_maybe_save_png` / `_maybe_save_rviz` gained a `quiet` flag that demotes the
routine "saved to …" and "skipped" lines to debug. Without it a 30 s timer
narrates itself into the run log for the whole sweep.

A *failed* grab still warns in both modes — mid-run is when you want to find
out the snapshots are not landing, rather than discovering it at the end.

### 3. The timer

```python
if _EXPLORATION_SNAPSHOT_S > 0 and explorer is not None:
    def _snapshot_exploration() -> None:
        if explorer.is_complete():
            return                       # DONE already wrote the final pair
        _maybe_save_png(explorer, node, quiet=True)
        _maybe_save_rviz(node, quiet=True)
    node.create_timer(_EXPLORATION_SNAPSHOT_S, _snapshot_exploration)
```

Interval from `XIAO_HEI_EXPLORATION_SNAPSHOT_S` (default 30 s, `0` restores
end-of-sweep-only). Each snapshot overwrites the last, so the file on disk is
always the newest view and `DONE` is simply the final write.

Concurrency needs no lock: rclpy's default executor is single-threaded, so the
timer can never interleave with `tick()`. The grid and the visited list are
read between ticks, never mid-update.

## Applies to both strategies

`_maybe_save_png` silently no-ops on a strategy that lacks
`get_visited_waypoints()` / `get_grid()`, which would have made periodic
snapshots quietly vanish for `nbv` with no error anywhere. Both
`FrontierExplorer` and `NextBestViewExplorer` implement them over the same
`OccupancyGrid`, and a test now pins that for both.

## Costs

- **Tick jitter.** Rendering blocks the single-threaded executor, so the tick
  coinciding with a snapshot is delayed by roughly the render time. At 30 s
  intervals against a 1 Hz tick this is a small fraction of ticks.
- **RViz window raise.** `save_rviz_screenshot` raises the window before
  reading its pixels — X11 without a compositor does not retain obscured
  regions — so an interactive session sees RViz pop to the front each
  interval. Raise the interval or set `0` if that is disruptive.

## Files touched

- `src/xiao_hei_vln/app/main.py` — `_publish_atomically`, `quiet=` on both
  helpers, `_EXPLORATION_SNAPSHOT_S`, the timer.
- `tests/test_exploration_snapshot.py` — new, 14 tests.
- `docker/compose.yml`, `docker/compose_scene_gemini.yml` — expose the knob.
- `docs/concepts/exploration.md`, `docs/getting-started/configuration.md`,
  `docker/README.md` — document it.

## Verification

- `uv run pytest tests/test_exploration_snapshot.py -q` → **14 passed**.
- `uv run pytest -q --ignore=tests/test_execute_plan.py` → **488 passed,
  1 skipped**. (`test_execute_plan.py` fails to collect on a missing `cv2`,
  pre-existing and unrelated — it imports `scripts/approach_loop.py`.)
- `uv run ruff check` on the changed files reports the same 4 pre-existing
  findings as `HEAD` (3 × E501, 1 × SIM115); the added lines introduce none.

Not exercised here: a live sweep against the sim. The timer registration
itself is only reachable under rclpy, so it is covered by inspection; every
function it calls is covered by the new tests.

## Related gap — fixed in TASK 41

`compose_scene_gemini.yml` did not pass `XIAO_HEI_SCENE_DIR_HOST` to
`ai_module` (`compose.yml:87` and `compose.eval.yml:61` both do), so under the
submission stack `_EXPLORATION_SCENE` always resolved to `default_scene` and
every scene's artefacts collided in one directory — periodic snapshots
included. That, and the same collision between strategies, is fixed in
[TASK 41](TASK%2048%20-%20Artefact%20paths%20keyed%20on%20scene%20and%20strategy.md):
artefacts now land in `<scene>/<strategy>/`.
