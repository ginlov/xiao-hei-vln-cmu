"""Tests for `QwenConfig` env-var loading."""

from __future__ import annotations

import pytest

from xiao_hei_vln.qwen import QwenConfig


def test_defaults_match_phase1_plan() -> None:
    cfg = QwenConfig()
    assert cfg.model == "Qwen/Qwen3.5-4B-Instruct"
    assert cfg.dtype == "bfloat16"
    assert cfg.max_model_len == 4096
    assert cfg.temperature == 0.0
    assert cfg.image_long_edge == 1280
    assert cfg.vllm_base_url is None


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XIAO_HEI_QWEN_MODEL", "Qwen/Qwen3.5-9B-Instruct-AWQ")
    monkeypatch.setenv("XIAO_HEI_QWEN_DTYPE", "float16")
    monkeypatch.setenv("XIAO_HEI_QWEN_MAX_MODEL_LEN", "8192")
    monkeypatch.setenv("XIAO_HEI_QWEN_GPU_MEM_UTIL", "0.7")
    monkeypatch.setenv("XIAO_HEI_QWEN_TEMPERATURE", "0.2")
    monkeypatch.setenv("XIAO_HEI_QWEN_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("XIAO_HEI_QWEN_IMAGE_LONG_EDGE", "896")
    monkeypatch.setenv("XIAO_HEI_QWEN_MAX_TICKS", "50")
    monkeypatch.setenv("XIAO_HEI_QWEN_SEED", "123")
    monkeypatch.setenv("XIAO_HEI_QWEN_TRUST_REMOTE", "0")

    cfg = QwenConfig.from_env()
    assert cfg.model == "Qwen/Qwen3.5-9B-Instruct-AWQ"
    assert cfg.dtype == "float16"
    assert cfg.max_model_len == 8192
    assert cfg.gpu_memory_utilization == pytest.approx(0.7)
    assert cfg.temperature == pytest.approx(0.2)
    assert cfg.max_output_tokens == 512
    assert cfg.image_long_edge == 896
    assert cfg.max_ticks_per_question == 50
    assert cfg.seed == 123
    assert cfg.trust_remote_code is False


def test_from_env_loads_vllm_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XIAO_HEI_QWEN_VLLM_BASE_URL", "http://localhost:8000/v1")
    cfg = QwenConfig.from_env()
    assert cfg.vllm_base_url == "http://localhost:8000/v1"


def test_empty_vllm_base_url_treated_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XIAO_HEI_QWEN_VLLM_BASE_URL", "")
    cfg = QwenConfig.from_env()
    assert cfg.vllm_base_url is None
