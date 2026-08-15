"""The exploration configuration is locked in config/exploration.env.

Two things need guarding:

* every hyperparameter reaches the explorers from the environment, so porting
  to the submission repo is copying one file rather than re-reading the source;
* the handful of values that compose *also* declares under `environment:` (the
  ones sweep scripts override) still agree with that file. They are duplicated
  out of necessity — a bare `- VAR` entry resolves to an EMPTY string when the
  shell has it unset, which overrides env_file and, for STRATEGY, silently
  disables exploration.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / "config" / "exploration.env"
COMPOSE = [
    REPO / "docker" / "compose.yml",
    REPO / "docker" / "compose_scene_gemini.yml",
    REPO / "docker" / "compose.eval.yml",
]


def _env_file_values() -> dict[str, str]:
    out = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def test_env_file_exists_and_parses() -> None:
    vals = _env_file_values()
    assert vals, "config/exploration.env has no settings"
    assert all(k.startswith("XIAO_HEI_") for k in vals), vals


def test_default_strategy_is_nbv() -> None:
    assert _env_file_values()["XIAO_HEI_EXPLORATION_STRATEGY"] == "nbv"


@pytest.mark.parametrize("compose_path", COMPOSE, ids=lambda p: p.name)
def test_compose_loads_the_env_file(compose_path: Path) -> None:
    spec = yaml.safe_load(compose_path.read_text())
    ai = spec["services"]["ai_module"]
    assert "env_file" in ai, f"{compose_path.name}: ai_module does not load the env file"
    assert any("exploration.env" in str(e) for e in ai["env_file"]), ai["env_file"]


# Deliberate per-profile divergences from config/exploration.env. The
# submission and eval stacks cap the sweep harder than a benchmarking run does.
# Anything NOT listed here is drift and fails the test.
DECLARED_OVERRIDES = {
    "compose_scene_gemini.yml": {"XIAO_HEI_EXPLORATION_MAX_WAYPOINTS"},
    "compose.eval.yml": {
        "XIAO_HEI_EXPLORATION_MAX_WAYPOINTS",
        "XIAO_HEI_EXPLORATION_MAX_SECONDS",
        "XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST",
    },
}


@pytest.mark.parametrize("compose_path", COMPOSE, ids=lambda p: p.name)
def test_compose_defaults_agree_with_the_env_file(compose_path: Path) -> None:
    """`- VAR=${VAR:-default}` entries must not drift from config/exploration.env.

    compose.yml is the reference and may not diverge at all. The other profiles
    may, but only for variables listed in DECLARED_OVERRIDES — so a deliberate
    difference is a one-line edit here and an accidental one is a failure.
    """
    env_vals = _env_file_values()
    allowed = DECLARED_OVERRIDES.get(compose_path.name, set())
    mismatches = []
    for var, default in re.findall(
        r"(XIAO_HEI_EXPLORATION_[A-Z_]+):-([^}]*)\}", compose_path.read_text()
    ):
        if var in env_vals and env_vals[var] != default and var not in allowed:
            mismatches.append(f"{var}: compose={default!r} env_file={env_vals[var]!r}")
    assert not mismatches, (
        f"{compose_path.name} disagrees with config/exploration.env:\n  "
        + "\n  ".join(mismatches)
        + "\n(if deliberate, add the variable to DECLARED_OVERRIDES)"
    )


def test_strategy_is_never_a_declared_override() -> None:
    """Every profile must run the same algorithm, or a comparison means nothing."""
    for path in COMPOSE:
        for default in re.findall(
            r"XIAO_HEI_EXPLORATION_STRATEGY:-([^}]*)\}", path.read_text()
        ):
            assert default == "nbv", f"{path.name} defaults to {default!r}, not nbv"


def test_no_bare_exploration_passthrough_in_compose() -> None:
    """A bare `- XIAO_HEI_EXPLORATION_X` nulls out the env_file value.

    Verified against docker compose: with the shell variable unset the container
    receives an empty string, not the env_file default. For STRATEGY that means
    exploration is disabled with an "unknown strategy" error.
    """
    offenders = []
    for path in COMPOSE:
        for line in path.read_text().splitlines():
            s = line.strip()
            if re.fullmatch(r"- XIAO_HEI_EXPLORATION_[A-Z_]+", s):
                offenders.append(f"{path.name}: {s}")
    assert not offenders, (
        "bare pass-through overrides env_file with an empty string:\n  "
        + "\n  ".join(offenders)
    )


def test_every_hyperparameter_is_read_from_the_environment(monkeypatch) -> None:
    """No explorer knob may be hard-coded at the call site."""
    overrides = {
        "XIAO_HEI_EXPLORATION_GRID_RESOLUTION": "0.5",
        "XIAO_HEI_EXPLORATION_COST_THRESHOLD": "0.9",
        "XIAO_HEI_EXPLORATION_REACH_DIST": "1.25",
        "XIAO_HEI_EXPLORATION_WP_TIMEOUT_S": "33",
        "XIAO_HEI_EXPLORATION_NBV_N_SAMPLES": "77",
        "XIAO_HEI_EXPLORATION_NBV_MAX_SKIPS": "11",
        "XIAO_HEI_EXPLORATION_NBV_SEED": "5",
        "XIAO_HEI_EXPLORATION_FRONTIER_MIN_CLUSTER": "9",
        "XIAO_HEI_EXPLORATION_FRONTIER_REJECT_RADIUS": "2.5",
    }
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    main = importlib.reload(importlib.import_module("xiao_hei_vln.app.main"))
    try:
        assert main._EXPLORATION_GRID_RESOLUTION == 0.5
        assert main._EXPLORATION_COST_THRESHOLD == 0.9
        assert main._EXPLORATION_REACH_DIST == 1.25
        assert main._EXPLORATION_WP_TIMEOUT_S == 33.0
        assert main._NBV_N_SAMPLES == 77
        assert main._NBV_MAX_SKIPS == 11
        assert main._NBV_SEED == 5
        assert main._FRONTIER_MIN_CLUSTER == 9
        assert main._FRONTIER_REJECT_RADIUS == 2.5
    finally:
        for k in overrides:
            monkeypatch.delenv(k, raising=False)
        importlib.reload(main)


def test_env_file_covers_every_knob_the_node_reads() -> None:
    """A knob the code reads but the env file omits is an undocumented default."""
    src = (REPO / "src" / "xiao_hei_vln" / "app" / "main.py").read_text()
    read = set(re.findall(r'"(XIAO_HEI_EXPLORATION_[A-Z_]+)"', src))
    documented = set(_env_file_values())
    # Scene dir and log dir are deployment paths, not tuning; they stay out.
    read -= {"XIAO_HEI_EXPLORATION_LOG_DIR"}
    missing = read - documented
    assert not missing, f"read by main.py but absent from config/exploration.env: {sorted(missing)}"
