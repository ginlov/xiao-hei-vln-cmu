"""Tests for `_PerceptionSettings` — the env-var contract shared by the
`perception` and `scene_gemini` responders.

These knobs previously lived inline in two copies of `_build_responder`, where
nothing exercised them: the factory blocks on the sidecar's `/healthz` and
issues a live Gemini `warmup()`, so it has never been unit-testable. Parsing is
now pure, so a default drifting (or a bool flag silently not taking effect) is
caught here.
"""

from __future__ import annotations

import pytest

from xiao_hei_vln.app.main import _PerceptionSettings
from xiao_hei_vln.perception.client import DEFAULT_BASE_URL
from xiao_hei_vln.perception.lifter import DEFAULT_MIN_INLIERS
from xiao_hei_vln.perception.responder import DEFAULT_SCORE_THRESHOLD

_ALL_VARS = (
    "XIAO_HEI_PERCEPTION_BASE_URL",
    "XIAO_HEI_PERCEPTION_SCORE_THRESHOLD",
    "XIAO_HEI_PERCEPTION_MIN_INLIERS",
    "XIAO_HEI_SCAN_KEYFRAMES",
    "XIAO_HEI_SCAN_MIN_MOVE_M",
    "XIAO_HEI_SCAN_MIN_ROT_DEG",
    "XIAO_HEI_SCAN_VOXEL_M",
)


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every knob so a stray value in the caller's shell can't leak in."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)


# --- defaults ---------------------------------------------------------------


def test_defaults_track_the_source_constants(clean_env: None) -> None:
    """With nothing set, defaults come from the perception package itself.

    Asserted against the imported constants rather than literals, so bumping a
    default in one place can't silently desync the factory.
    """
    s = _PerceptionSettings.from_env()

    assert s.base_url == DEFAULT_BASE_URL
    assert s.score_threshold == DEFAULT_SCORE_THRESHOLD
    assert s.min_inliers == DEFAULT_MIN_INLIERS


def test_scan_accumulator_defaults(clean_env: None) -> None:
    s = _PerceptionSettings.from_env()

    # Accumulation is on by default at main's value. This branch measured 0 as
    # better on the two scenes it has corpora for; see the note in
    # `app/main.py`, where that disagreement is recorded rather than settled.
    assert s.scan_keyframes == 10
    assert s.scan_min_move_m == pytest.approx(0.25)
    assert s.scan_min_rot_deg == pytest.approx(15.0)
    assert s.scan_voxel_m == pytest.approx(0.05)


# --- overrides + coercion ---------------------------------------------------


def test_every_knob_is_overridable_and_typed(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XIAO_HEI_PERCEPTION_BASE_URL", "http://sidecar:9999")
    monkeypatch.setenv("XIAO_HEI_PERCEPTION_SCORE_THRESHOLD", "0.4")
    monkeypatch.setenv("XIAO_HEI_PERCEPTION_MIN_INLIERS", "25")
    monkeypatch.setenv("XIAO_HEI_SCAN_KEYFRAMES", "4")
    monkeypatch.setenv("XIAO_HEI_SCAN_MIN_MOVE_M", "1.5")
    monkeypatch.setenv("XIAO_HEI_SCAN_MIN_ROT_DEG", "30")
    monkeypatch.setenv("XIAO_HEI_SCAN_VOXEL_M", "0.1")

    s = _PerceptionSettings.from_env()

    assert s.base_url == "http://sidecar:9999"
    # Env vars arrive as strings; the numeric fields must be coerced, not
    # passed through — PointLifter/ScanAccumulator do arithmetic on them.
    assert isinstance(s.score_threshold, float) and s.score_threshold == pytest.approx(0.4)
    assert isinstance(s.min_inliers, int) and s.min_inliers == 25
    assert isinstance(s.scan_keyframes, int) and s.scan_keyframes == 4
    assert s.scan_min_move_m == pytest.approx(1.5)
    assert s.scan_min_rot_deg == pytest.approx(30.0)
    assert s.scan_voxel_m == pytest.approx(0.1)


def test_object_map_fusion_is_not_configurable(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ObjectMap fusion is unconditional; the old opt-in flag is gone.

    A stale ``XIAO_HEI_OBJECT_MAP`` in someone's shell must not resurrect a
    second code path — there is only one now, so the var is inert.
    """
    monkeypatch.setenv("XIAO_HEI_OBJECT_MAP", "0")

    assert not hasattr(_PerceptionSettings.from_env(), "use_object_map")


# --- logging ----------------------------------------------------------------


def test_log_config_is_json_safe_and_complete(clean_env: None) -> None:
    """`as_log_config()` feeds `session.json`, so it must carry the knobs that
    change results and stay JSON-serialisable."""
    import json

    cfg = _PerceptionSettings.from_env().as_log_config()

    assert set(cfg) == {
        "perception_base_url",
        "score_threshold",
        "min_inliers",
    }
    json.loads(json.dumps(cfg))  # raises if a value isn't JSON-native


def test_log_config_reflects_overrides(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XIAO_HEI_PERCEPTION_SCORE_THRESHOLD", "0.75")
    monkeypatch.setenv("XIAO_HEI_PERCEPTION_MIN_INLIERS", "25")

    cfg = _PerceptionSettings.from_env().as_log_config()

    assert cfg["score_threshold"] == pytest.approx(0.75)
    assert cfg["min_inliers"] == 25


def test_api_key_never_reachable_through_log_config(clean_env: None) -> None:
    """Guards the TASK 14 fix from regressing via this path: the perception
    settings must not carry anything secret into session.json."""
    cfg = _PerceptionSettings.from_env().as_log_config()
    assert not any("key" in k.lower() or "token" in k.lower() for k in cfg)
