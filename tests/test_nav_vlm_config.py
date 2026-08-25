"""Env-var parsing for NavVLMConfig (no key/network needed)."""

from __future__ import annotations

import pytest

from xiao_hei_vln.nav_vlm.config import NavVLMConfig

_KEYS = [
    "ANTHROPIC_API_KEY",
    "XIAO_HEI_ANTHROPIC_API_KEY",
    "XIAO_HEI_NAV_VLM_MODEL",
    "XIAO_HEI_NAV_VLM_TEMPERATURE",
    "XIAO_HEI_NAV_VLM_IMAGE_LONG_EDGE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        NavVLMConfig.from_env()


def test_primary_key_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = NavVLMConfig.from_env()
    assert cfg.api_key == "sk-test"
    assert cfg.model == "claude-opus-5"
    assert cfg.thinking_budget == 0  # forced-tool default


def test_namespaced_fallback_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XIAO_HEI_ANTHROPIC_API_KEY", "sk-fallback")
    cfg = NavVLMConfig.from_env()
    assert cfg.api_key == "sk-fallback"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("XIAO_HEI_NAV_VLM_MODEL", "claude-opus-5-mini")
    monkeypatch.setenv("XIAO_HEI_NAV_VLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("XIAO_HEI_NAV_VLM_IMAGE_LONG_EDGE", "768")
    cfg = NavVLMConfig.from_env()
    assert cfg.model == "claude-opus-5-mini"
    assert cfg.temperature == 0.7
    assert cfg.image_long_edge == 768
