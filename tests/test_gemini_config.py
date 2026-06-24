"""Tests for ``xiao_hei_vln.gemini.config.GeminiConfig``."""

from __future__ import annotations

import pytest

from xiao_hei_vln.gemini.config import GeminiConfig


class TestDefaults:
    def test_required_api_key(self) -> None:
        c = GeminiConfig(api_key="abc")
        assert c.api_key == "abc"
        assert c.model == "gemini-2.5-flash"
        assert c.temperature == 0.2
        assert c.max_output_tokens == 1024
        assert c.image_long_edge == 1280
        assert c.max_explore_ticks == 120
        assert c.max_ticks_per_question == 240


class TestFromEnv:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XIAO_HEI_GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="XIAO_HEI_GEMINI_API_KEY"):
            GeminiConfig.from_env()

    def test_empty_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XIAO_HEI_GEMINI_API_KEY", "")
        with pytest.raises(ValueError, match="XIAO_HEI_GEMINI_API_KEY"):
            GeminiConfig.from_env()

    def test_uses_env_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XIAO_HEI_GEMINI_API_KEY", "sk-fake")
        monkeypatch.setenv("XIAO_HEI_GEMINI_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("XIAO_HEI_GEMINI_TEMPERATURE", "0.5")
        monkeypatch.setenv("XIAO_HEI_GEMINI_MAX_OUTPUT_TOKENS", "2048")
        monkeypatch.setenv("XIAO_HEI_GEMINI_IMAGE_LONG_EDGE", "640")
        monkeypatch.setenv("XIAO_HEI_GEMINI_MAX_EXPLORE_TICKS", "30")
        monkeypatch.setenv("XIAO_HEI_GEMINI_MAX_TICKS", "60")
        c = GeminiConfig.from_env()
        assert c.api_key == "sk-fake"
        assert c.model == "gemini-2.5-pro"
        assert c.temperature == pytest.approx(0.5)
        assert c.max_output_tokens == 2048
        assert c.image_long_edge == 640
        assert c.max_explore_ticks == 30
        assert c.max_ticks_per_question == 60

    def test_partial_env_keeps_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XIAO_HEI_GEMINI_API_KEY", "sk-fake")
        # Other vars unset
        monkeypatch.delenv("XIAO_HEI_GEMINI_MODEL", raising=False)
        monkeypatch.delenv("XIAO_HEI_GEMINI_MAX_EXPLORE_TICKS", raising=False)
        c = GeminiConfig.from_env()
        assert c.api_key == "sk-fake"
        assert c.model == "gemini-2.5-flash"
        assert c.max_explore_ticks == 120
