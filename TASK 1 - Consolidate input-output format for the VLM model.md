# TASK 1 — Consolidate input/output format for the VLM model

## Goal

Define a stable contract between the CMU VLN Challenge ROS 2 stack and
Team Xiao Hei's VLM so the model can be developed and tested
independently of ROS, and so multi-rate sensor inputs are resampled
onto a single VLM execution cadence.

## What was done

### Phase 1 — Observe the live system

- Pulled and launched the official challenge docker stack
  (`compose_gpu.yml` → `iros2026_system` + `iros2026_ai_module`) and
  ran `system_simulation.sh` inside the bundled Unity scene.
- Used `ros2 topic list / info -v / hz / echo --once` (with
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, as the challenge configures)
  to capture each sensor topic's type, frame, QoS, message structure,
  and live rate.
- Findings are recorded in **`docs/task1_phase1_measurements.md`**.
  Headline notes:
  - `/registered_scan`, `/sensor_scan`, `/terrain_map`,
    `/terrain_map_ext` all publish at **~10 Hz**, not the 5 Hz quoted
    in the README.
  - `/state_estimation` publishes at the upper end of its quoted band
    (~200 Hz).
  - `/camera/image` is `bgr8`, 1920×640, ~9.3 Hz.
  - `/terrain_map` uses `point_step=32` (16 bytes of padding after
    intensity); we read fields by offset to stay robust to this.

### Phase 2 — Propose I/O format & execution frequency

- Wrote **`docs/task1_io_spec.md`** specifying:
  - Seven input channels (one per allowed sensor topic + the question
    channel) and three output variants (numerical / object reference /
    waypoint path), each mapped to a pydantic class.
  - **VLM tick rate: 2 Hz** (configurable 0.5 – 5 Hz). Rationale: 2 Hz
    leaves ~400–500 ms of inference budget per tick, guarantees the
    snapshot is never more than ~110 ms stale relative to the slowest
    input, and halves the worst-case reaction latency vs the 1 Hz
    question publish rate.
  - **Synchronization: latest-cache + tick snapshot.** ROS subscriber
    callbacks overwrite per-channel slots; the VLM loop atomically
    snapshots all slots into a single `VLMInput`. Simpler than
    `ApproximateTimeSynchronizer` and a better fit for a slow VLM.

### Phase 3 — Implement as Python data classes

- Bootstrapped a Python 3.12 project under `uv` (per repo
  `CLAUDE.md`): `pyproject.toml` declares `pydantic>=2.7`, `numpy>=2`
  as runtime deps and `pytest`, `ruff` as dev deps; `src/` layout
  installed editable via `uv pip install -e .`.
- Added the package under `src/xiao_hei_vln/`:
  - `messages/common.py` — `Stamp`, `Header`, `Vector3`, `Quaternion`
  - `messages/sensors.py` — `ImageFrame`, `LidarScan`, `TerrainMap`,
    `OdomPose` (with shape/dtype validators)
  - `messages/question.py` — `QuestionType`, `ChallengeQuestion`,
    `classify_question` (matching the dummy VLM's heuristic, swappable
    later)
  - `messages/inputs.py` — `VLMInput` (the tick snapshot)
  - `messages/outputs.py` — `NumericalResponse`,
    `ObjectReferenceResponse`, `Waypoint`, `WaypointPathResponse`,
    `VLMOutput` discriminated union, `parse_vlm_output`
  - `sync/latest_cache.py` — thread-safe `LatestCache` with sticky
    question semantics and atomic `snapshot()`
  - `adapters/ros/subscribers.py` — `bind_subscribers(node, cache)`
    that creates QoS-matched subscriptions and pushes into the cache;
    `rclpy`/`sensor_msgs` imported lazily so the core stays
    ROS-free
  - `adapters/ros/publishers.py` — `VLMOutputPublisher` that routes a
    `VLMOutput` to the correct topic (`/numerical_response`,
    `/selected_object_marker`, `/way_point_with_heading`)
- Added **32 pytest cases** under `tests/` covering: stamp
  conversions, shape/dtype/encoding validators, the question
  classifier, `VLMInput.is_ready`, every `VLMOutput` discriminator
  branch + JSON round-trip + rejection of empty waypoint lists, and a
  concurrent writer/snapshot stress test for `LatestCache`.

## Verification

```
$ uv run pytest -q
................................   32 passed in 0.39s
$ uv run ruff check src tests
All checks passed!
$ uv run python -c "from xiao_hei_vln.messages import VLMInput; print(sorted(VLMInput.model_fields))"
['image', 'pose', 'question', 'registered_scan', 'sensor_scan',
 'terrain_ext', 'terrain_local', 'tick_id', 'tick_time']
```

ROS-side smoke test of the adapters is deferred to the follow-up task
that wires an actual `rclpy` node — the lazy import pattern was
chosen so we don't need a ROS install to validate the contract today,
and Phase 1 already confirmed that every message field we consume is
present and shaped as expected on the live topics.

## Files added

- `pyproject.toml`, `.python-version`, `uv.lock`
- `src/xiao_hei_vln/` (8 modules)
- `tests/test_messages.py`, `tests/test_sync.py`
- `docs/task1_phase1_measurements.md`
- `docs/task1_io_spec.md`
- This report.

## Open follow-ups (out of scope here)

- Wire a Python `ai_module` ROS node that constructs a `LatestCache`,
  calls `bind_subscribers`, ticks the VLM at 2 Hz, and uses
  `VLMOutputPublisher` to emit responses — replacing the dummy
  `dummyVLM.cpp` once the actual VLM is ready.
- For instruction-following questions, decide how a `WaypointPath`
  is fed waypoint-by-waypoint into `/way_point_with_heading` driven by
  `/way_point_reached` (the publisher already exposes a
  `publish_waypoint(waypoint)` helper for this).
- Question-type classification is a keyword heuristic today; revisit
  once we have a learned model.
