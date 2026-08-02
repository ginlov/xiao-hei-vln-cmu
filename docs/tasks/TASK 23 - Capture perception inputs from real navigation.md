# TASK 23 — Capture perception inputs from real navigation

## Why

Every perception number to date (TASK 20–22, the clustering benchmark, the 2D
detector audit) was measured on `perception_benchmark/captures/`, which
`capture_viewpoints.py` produced by driving the robot to the geometric k-cover
viewpoints from `viewgen.py`. That dataset answers **"how good could perception
be given near-ideal coverage?"** — 88–100% of scoreable objects are guaranteed
observable from at least one captured pose, by construction.

It does **not** answer what the submission stack actually sees. Live, the robot
follows the frontier explorer, which optimises for map coverage, not object
coverage: it stops where frontiers are, at whatever range and angle that
implies, and it never revisits a pose for a second look. The scene graph the
challenge is scored on is built from *that* stream.

So the two datasets can disagree in both directions — a viewpoint-set object may
never be seen live, and the live run produces many more, closer, partially
redundant views. Task 23 builds the second dataset so the gap can be measured
instead of assumed.

## What was built

### `perception_benchmark/record_navigation.py`

A **passive** rclpy recorder. It publishes nothing — the ai_module container
drives with the real `FrontierExplorer`, and this node only watches
`/state_estimation` and the sensor topics.

It reproduces the **live ingest cadence**, which is the point. In `app/main.py`
the node ticks at `XIAO_HEI_VLM_TICK_HZ` (default 2.0 → every 0.5 s), and every
exploration tick calls `responder.ingest()` → `_inject_visible()`, which runs
detect → lift → `add_object` *unconditionally*. There is no distance, rotation
or motion-blur gate on that path: a tick is skipped only when the snapshot lacks
an image / pose / scan, and a blurred frame simply returns no detections and
leaves the scene unchanged. So the recorder samples at a fixed `--rate-hz`
(default 2.0) and applies **no quality filter**.

The first draft gated keyframes on 0.6 m / 30° of motion with a motion-blur
speed gate. That was wrong — it is a *capture policy*, not the system, and it
would have handed the offline replay ~10× fewer, sharper, better-spaced frames
than the live stack ever sees, quietly flattering every downstream number. Those
gates were then kept as opt-in knobs, and finally removed outright: an option
that can only make the dataset diverge from the system is not worth the config
surface, and its presence invites exactly the silent divergence the task exists
to eliminate.

A warm-up delay went the same way, and was the subtlest of the three. The live
node starts its timer immediately and skips ticks whose snapshot is incomplete
(`_inject_visible` returns early on a missing image / pose / scan). Spinning for
3 s before the first sample would have suppressed ticks the live stack *would*
have attempted. The `skipped.incomplete` counter reproduces that behaviour
faithfully instead — a small non-zero value at startup is correct, not a fault.

`--rate-hz` now defaults to `XIAO_HEI_VLM_TICK_HZ` itself, so the recorder tracks
the live tick rate rather than hard-coding a copy of it.

The only frames dropped are ones where the camera has not published since the
last sample. That needs a stalled camera — `/camera/image` runs ~10 Hz
(`docs/architecture.md:68`) against a 2 Hz tick — so in a healthy run this never
fires. It is skipped rather than written as a byte-identical copy, and
**counted**: a non-zero count means the live stack would have re-ingested and
re-fused that image, so it is a real divergence worth seeing. All three skip
counters (`incomplete`, `stale_image`, `downsampled`) land in `manifest.json`.

An earlier draft paired this with an `--allow-duplicate-frames` escape hatch to
restore live fidelity. It was removed: at 10 Hz vs 2 Hz the case it guards
against cannot arise outside a fault, so it was a knob bought for nothing. The
counter stays, since that is where the diagnostic value actually was.

### The full flag list, and why each survives

`--scene` and `--out` name the output. `--rate-hz` *is* a system parameter
(`XIAO_HEI_VLM_TICK_HZ`). `--stop-file` is mechanism, not policy: the live node
ends when the explorer completes, and a passive recorder has to be told. That
leaves exactly one flag with no counterpart in the live stack — `--max-seconds`,
a disk backstop, since at ~500 MB/min an unattended recorder is a real hazard.
When it fires the manifest records `stop_reason: max_seconds`, so a truncated
dataset can never be mistaken for a complete one.

There is no motion gate, no blur gate, no warm-up and no frame cap.

The output layout is byte-identical to `capture_viewpoints.py`'s
(`vp_XXX/{image,registered_scan,sensor_scan,terrain_map,terrain_map_ext}.npy`
`+ pose.json + meta.json`, `+ manifest.json`), so every offline consumer reads it
unchanged. The wire decoders (`image_to_array`, `cloud_to_array`) are imported
from `capture_viewpoints.py` rather than copied — the intensity-column handling
is subtle enough that a second copy would drift.

`manifest.json` is rewritten after **every** frame via temp-file + rename, so an
abrupt kill costs at most the frame in flight.

**Disk cost is the real constraint.** A frame is ~4.2 MB, 3.7 MB of which is the
raw 640×1920 uint8 image. At 2 Hz that is ~500 MB per minute of exploration per
scene, so a 10-minute scene costs ~5 GB and a 15-scene sweep runs into tens of
GB. `RATE_HZ` is the knob; check free space first.

### `perception_benchmark/run_nav_capture.sh`

Scene sweep. Per scene: extract, bring up a fresh sim + ai_module
(`XIAO_HEI_RESPONDER=dummy`, so capture needs no GPU and no perception sidecar),
launch the sim, start the recorder inside the sim container, then poll
`exploration_logs/<scene>/exploration.log` for the explorer's `DONE` line —
which is the only authoritative end-of-trajectory signal, since the recorder is
passive. On DONE it touches the recorder's stop file, waits for the flush, and
copies the captures to the host.

Resumable (skips scenes with an existing manifest unless `FORCE=1`), and the
containers are removed on normal exit **and** on Ctrl-C.

### `PERCEPTION_CAP_DIR`

`CAP_DIR` in the eight offline scripts (`replay_score`, `dump_debug`,
`dump_cloud`, `dump_lift_input`, `detect_2d_eval`, `dump_detections`,
`box_quality`, `debug_viewpoint`) now reads `PERCEPTION_CAP_DIR`, defaulting to
the existing `perception_benchmark/captures`. This mirrors the
`PERCEPTION_DEBUG_DIR` knob `viz_app.py` already had, and means the two datasets
coexist instead of one overwriting the other — the TASK 20–22 numbers stay
reproducible.

## Run

```bash
perception_benchmark/run_nav_capture.sh arabic_room       # one scene first
perception_benchmark/run_nav_capture.sh                   # all 15

export PERCEPTION_CAP_DIR=perception_benchmark/captures_nav
export PERCEPTION_DEBUG_DIR=perception_benchmark/debug_nav
uv run --extra perception python perception_benchmark/replay_score.py --all
uv run --extra perception python perception_benchmark/dump_debug.py --all --out "$PERCEPTION_DEBUG_DIR"
uv run --extra perception python perception_benchmark/dump_cloud.py  --all --out "$PERCEPTION_DEBUG_DIR"
uv run --with streamlit --with plotly streamlit run perception_benchmark/viz_app.py
```

Knobs: `RATE_HZ` (default 2.0 — match the live tick), `MAX_WAYPOINTS` (explorer
budget), `TIMEOUT` (per scene, default 1800 s), `FORCE`, `KEEP_UP`.

## Verification

The sampler is exercised offline with ROS stubbed out (scratchpad
`test_gate.py`): the 0.5 s period fires at 0.0/0.5 s and not at 0.2/0.4 s; a
**stationary** robot still yields a frame every tick (the property the first
draft got wrong); an incomplete snapshot is skipped and counted, matching
`_inject_visible`'s own guard; and a stale camera frame is skipped and counted.

Two assertions guard against the mistake this task kept making: a barely-moved
and a fast-moved robot are both still sampled (no motion may ever gate a frame),
and the CLI source is checked to contain none of `--min-move`, `--min-rot`,
`--warmup-s`, `--max-frames`, `--speed-max` or `--allow-duplicate-frames` while
`--rate-hz` still defaults from `XIAO_HEI_VLM_TICK_HZ`. All pass.

Not yet run against the sim — that is the next step, and the notes below are
what to check when it is.

## What to look for once it runs

- **Frame count vs exploration duration.** At 2 Hz the count should be
  ~2× the seconds between the explorer's `START` and `DONE` lines. A large
  shortfall means skipped ticks — read the `skipped` block and `stop_reason` in
  the manifest before concluding anything about perception.
- **That the explorer actually explored.** The one real log on disk
  (`exploration_logs/studio`) ended `visited=0 skipped=20
  reason=max_consecutive_skips` after 261 s. A run like that produces a
  perfectly valid recording of the robot going nowhere. Check `visited` before
  scoring.
- **Recall vs the viewpoint arm.** A drop is the real cost of coverage-driven
  navigation and is the number that matters for the challenge.
- **Precision and counting MAE.** More, closer, more redundant views push
  fragmentation up. This is the same axis the scan-accumulator A/B moved
  (P@1 0.327 → 0.408 with the accumulator off), so the two interact and should
  be read together.
- **`skipped.stale_image`.** Should be 0. Non-zero means the camera stalled
  below the 2 Hz tick, so the live stack was re-ingesting and re-fusing the same
  image — a finding about the live stack, not about this recorder.

## Related

- TASK 20 — the viewpoint-based benchmark this complements (does not replace).
- TASK 21 / 22 — the lift and fusion fixes whose gains should be re-measured on
  this dataset.
- `docs/tasks/backlog.md` — B2, B3 remain open and are unaffected.
