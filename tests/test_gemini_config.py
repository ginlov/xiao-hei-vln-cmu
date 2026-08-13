"""Tests for ``xiao_hei_vln.gemini.config.GeminiConfig``."""

from __future__ import annotations

import pytest

from xiao_hei_vln.gemini.config import GeminiConfig


class TestTheDefaultModelIsPinned:
    """A moving default makes every A/B meaningless.

    `gemini-flash-latest` moved from 3.5 to 3.6 while this was being written,
    and a benchmark whose model changed between its two halves measured
    nothing. The previous default, `gemini-2.5-flash`, is still listed by
    `models.list` and answers 404 "no longer available to new users" — so the
    catalogue is not evidence a model can be called, and a bump here has to be
    a decision someone made after calling it.
    """

    def test_it_is_a_version_and_not_an_alias(self) -> None:
        assert not GeminiConfig(api_key="x").model.endswith("-latest")

    def test_the_two_defaults_agree(self) -> None:
        """`scripts/` and `src/` are separate entry points onto one API."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from vlm_probe import DEFAULT_GEMINI_MODEL
        assert GeminiConfig(api_key="x").model == DEFAULT_GEMINI_MODEL


class TestDefaults:
    def test_required_api_key(self) -> None:
        c = GeminiConfig(api_key="abc")
        assert c.api_key == "abc"
        assert c.model == "gemini-3.6-flash"
        assert c.temperature == 0.2
        assert c.max_output_tokens == 2048
        assert c.thinking_budget == 0
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
        assert c.model == "gemini-3.6-flash"
        assert c.max_explore_ticks == 120
