# TASK 49 - An eight-minute wall-clock cutoff for exploration

## Purpose

Bound the exploration sweep in time. Default 8 minutes.

## Problem

Exploration ended for exactly three reasons, all owned by the strategy:
`budget_exhausted` (waypoint count), `max_consecutive_skips`, `no_frontiers`.
None is bounded in time. A large scene where the explorer keeps finding
reachable frontiers can burn an arbitrary amount of wall clock at 1 Hz without
ever reaching any of them — and because a question arriving mid-sweep is
deferred until the sweep ends (the TASK 12 contract), an unbounded sweep is an
unbounded delay before the robot answers anything.

`XIAO_HEI_EXPLORATION_MAX_SECONDS` already existed as a name — set in
`docker/compose.eval.yml` (540) and exported by
`scripts/run_scene_vla3d_eval.sh` (540) — but **nothing read it**. A grep over
`src/` returned no consumers. The eval harness enforced only a shell-level
`timeout` of `MAX_SECONDS + 600` around the whole run, and
`wait_exploration_done` additionally polled for a `HARD_STOP` marker that no
code has ever emitted. So the setting looked live and was not.

## Change — `app/main.py`

### The knob

```python
_EXPLORATION_MAX_SECONDS = float(os.environ.get("XIAO_HEI_EXPLORATION_MAX_SECONDS", "480"))
```

Reusing the existing name rather than adding a second one, so the eval
harness's long-dead setting becomes live at its intended value.

### When the clock starts

Not at node boot. `/state_estimation` takes 90-190 s to arrive, and the robot
cannot explore before it does — charging that dead time to the budget would
silently shorten an 8-minute sweep by up to 3 minutes. The clock starts on the
**first exploration tick that has a pose**, logged as `CLOCK_START`.

### The decision

Extracted rather than left inline, because the condition is the whole feature
and `tick()` needs rclpy to run:

```python
def _budget_expired(clock_start: float | None, now_s: float, budget_s: float) -> bool:
    if budget_s <= 0 or clock_start is None:
        return False
    return now_s - clock_start >= budget_s
```

`>=`, not `>`: at 1 Hz a sample can land exactly on the deadline.

### What expiry does

Logs `DONE  reason=time_limit  elapsed=…  budget=…`, saves `exploration.png`
and `rviz.png` (non-quiet, same as any other termination), and returns.

Terminating through the same `DONE` event as every other reason is deliberate:
`scripts/vla3d_eval_sim.sh` polls the log for `" DONE "`, so a timed-out sweep
is detected exactly like a completed one, with the reason field distinguishing
them. The `exploration_sweep.sh` results.csv awk reads `reason` the same way.

### How the sweep stops

The strategy is never told — `is_complete()` stays `False`, and its
accumulated grid stays readable for the final plot. A loop-level flag
`state["exploration_timed_out"]` takes the exploration branch out of the tick
loop instead, via a new `_exploration_over()` helper that both the branch
condition and the snapshot timer now consult. Reaching into
`explorer._done` would have worked for both shipped strategies but would have
made the `ExplorationStrategy` protocol a lie.

The snapshot timer registration moved below `state` so it can read that flag;
without it a timed-out sweep would have gone on writing snapshots forever.

## Deliberately not done

**The cutoff does not brake the robot.** The last commanded waypoint stays with
the nav stack until the responder publishes its own. This matches what already
happens when a sweep ends for any other reason, so the cutoff introduces no new
motion behaviour. Stopping on expiry would be a different change with its own
risk — the robot halting mid-corridor when a question is pending.

## Files touched

- `src/xiao_hei_vln/app/main.py` — knob, `_budget_expired`, `CLOCK_START` /
  `DONE reason=time_limit`, `_exploration_over()`, timer reorder.
- `tests/test_exploration_snapshot.py` — 7 new tests.
- `docker/compose.yml`, `docker/compose_scene_gemini.yml` — expose the knob.
- `docs/concepts/exploration.md` (four stop reasons + a section on the cutoff,
  `CLOCK_START` in the event table), `docs/getting-started/configuration.md`,
  `docker/README.md`.

## Verification

- `uv run pytest -q --ignore=tests/test_execute_plan.py` → **501 passed,
  1 skipped**. (`test_execute_plan.py` fails to collect on a missing `cv2`,
  pre-existing and unrelated.)
- `uv run ruff check` on the changed files → the same 4 pre-existing findings
  as `HEAD`.

Not exercised here: a live 8-minute sweep. `_budget_expired` is covered
directly (before / at / after the deadline, clock not started, disabled); the
tick-loop wiring around it is covered by inspection, since it only runs under
rclpy.

## Behaviour change to flag

Implementing the variable makes two existing configurations live that
previously had no effect: `compose.eval.yml` and `run_scene_vla3d_eval.sh` both
set 540 s. Runs through the eval harness now genuinely stop at 9 minutes
instead of exploring until a strategy condition fires. That is what those
settings were written to mean, but it is a real change in what those runs do.
