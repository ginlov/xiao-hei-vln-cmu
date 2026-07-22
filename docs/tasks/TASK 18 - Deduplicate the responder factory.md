# TASK 18 - Deduplicate the responder factory

!!! note "Partly superseded by TASK 19"

    The "Behaviour change: none" section below reasons at length about
    `near_threshold`. TASK 19 then removed the near-relation subsystem
    outright, so `near_threshold`, `DEFAULT_NEAR_THRESHOLD` and the
    `test_log_config_omits_near_threshold` test referenced here no longer
    exist. The reasoning is what led to that removal; everything else in this
    report still describes the current code.

## Purpose

Collapse the copy-pasted perception-stack construction in `_build_responder`
into one code path, and give the env-var parsing its first test coverage.

## Problem (before)

The `perception` and `scene_gemini` branches of
`src/xiao_hei_vln/app/main.py` each built the same perception stack from
scratch — **52 identical lines**, including two copies of:

- the same eight lazy imports,
- reads of the same **nine** env vars (`XIAO_HEI_PERCEPTION_*`,
  `XIAO_HEI_SCAN_*`, `XIAO_HEI_OBJECT_MAP`), each with its default repeated,
- `HTTPPerceptionClient` + `wait_until_ready()`, `PointLifter`,
  `ScanAccumulator`, `Vocabulary`, and the conditional `ObjectMap`,
- the `PerceptionResponder` construction itself.

The failure mode this invites is silent: a default tuned in one branch and not
the other gives two responders that disagree about how perception behaves, with
nothing failing to signal it. The branches had **already drifted** — see below.

Compounding it, `_build_responder` has **never had a test**. It blocks on the
sidecar's `/healthz` and issues a live Gemini `warmup()`, so it isn't
unit-testable as written, and the env parsing — the part most likely to carry a
config bug — was the least protected code in the file.

## What changed

Two new module-level pieces in `app/main.py`:

- **`_PerceptionSettings`** — a frozen dataclass with `from_env()` and
  `as_log_config()`. Pure: no I/O, no network, so it is directly testable. The
  perception-package imports stay lazy inside `from_env()`, preserving the
  property that the package imports without the `perception` extra installed.
- **`_build_perception_responder(scene, settings, ...)`** — does the wiring,
  parameterised by the only three things that actually differed between the
  branches: `trajectory_path`, `take_waypoint_reached_signals`, `logger`.

Result:

| | before | after |
|---|---|---|
| `_build_responder` | 222 lines | **75 lines** |
| env vars read in both branches | 10 | **1** (`XIAO_HEI_VLM_LOG_DIR`) |
| tests covering this code | 0 | **20** |

`app/main.py` net: 676 → 647 lines, despite the added helper and docstrings.

## Behaviour change: none

`session.json` comes out byte-identical to before for both responders.

An earlier revision of this refactor got that wrong, and the reason is worth
recording. The two logger configs *look* like they had drifted — the
`perception` branch recorded `near_threshold_m`, `scene_gemini` did not — so
the first attempt unified them and added the key to both. That was incorrect:

`near_threshold` is consumed in exactly one place,
`PerceptionResponder._compute_output` (the Phase-B answer path). But
`scene_gemini` never calls `respond()` on its perception responder — it drives
it purely through `ingest()` and answers via Gemini (TASK 14). **The threshold
therefore has no effect whatsoever in the submission path**, and the scene
graph handed to Gemini carries no derived `near` edges at all.

So the asymmetry was right, and unifying it would have documented a setting
that does nothing. `as_log_config()` now carries only the four knobs that apply
to any responder building the perception stack; the `perception` branch adds
`near_threshold_m` and `trajectory_json` itself, both being Phase-B-only.
`tests/test_perception_settings.py::test_log_config_omits_near_threshold` pins
the reasoning so it does not get "fixed" again.

The general lesson: an asymmetry between two copies is not automatically drift.
Here it encoded a real difference in which code path each responder runs.

## Verification

- `uv run pytest -q` → **427 passed, 1 skipped** (407 + 20 new). No existing
  test changed.
- New `tests/test_perception_settings.py` asserts defaults **against the
  imported constants** (`DEFAULT_NEAR_THRESHOLD` etc.) rather than literals, so
  bumping a default in the perception package cannot silently desync the
  factory. Also covers numeric coercion (env gives strings; these feed
  arithmetic), all seven truthy spellings of `XIAO_HEI_OBJECT_MAP`, the
  off-by-default rule for unrecognised values, and that `as_log_config()` stays
  JSON-serialisable and carries no key/token field.
- Wiring confirmed end-to-end with `wait_until_ready` patched: every knob
  (`near_threshold`, `score_threshold`, lifter `min_inliers`, scan
  `max_keyframes`, `ObjectMap` enablement) reaches its destination object, and
  the `/healthz` block is still issued.
- `ruff check` on `app/main.py`: **identical 4 findings before and after**
  (line numbers shifted only) — all pre-existing (`E501` ×3, `SIM115`), none
  introduced. New test file is clean.

## Not done

- `_build_responder` still isn't testable as a whole — the `scene_gemini`
  branch's `GeminiEngine.warmup()` and the sidecar `/healthz` block are real
  I/O in a factory function. Injecting those would make the whole factory
  unit-testable; the pure part is now carved out and covered.
- `_PerceptionSettings` lives in `app/main.py`. Moving it to
  `xiao_hei_vln/perception/settings.py` would put the env schema next to the
  defaults it reads, at the cost of the perception package owning its own
  configuration surface. Left as-is; the app is the only consumer.
