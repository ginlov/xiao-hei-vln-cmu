"""Engine / responder configuration loaded from environment variables.

All knobs are env-overridable so docker-compose can swap them without a
rebuild. Defaults match the Phase 1 plan in
`docs/task3_phase1_framework.md`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QwenConfig:
    """Engine + responder knobs."""

    # --- model / vLLM engine ----------------------------------------------
    model: str = "/models/Qwen3.5-4B"
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    trust_remote_code: bool = True

    # --- sidecar (HTTPQwenEngine) -----------------------------------------
    # When set, the responder calls vLLM over HTTP instead of loading the
    # engine in-process. Expected value: "http://localhost:8000/v1".
    vllm_base_url: str | None = None

    # --- sampling ---------------------------------------------------------
    temperature: float = 0.0
    max_output_tokens: int = 256
    seed: int | None = 0

    # --- image preprocessing ---------------------------------------------
    image_long_edge: int = 1280  # downscale target before handoff to vLLM

    # --- responder loop ---------------------------------------------------
    # Phase 3: a numerical-question loop accumulates "evidence" across
    # ticks before committing to an integer answer. This caps the loop so
    # we always return a final answer even if the model never converges.
    max_ticks_per_question: int = 30

    @classmethod
    def from_env(cls) -> QwenConfig:
        """Build a config from `XIAO_HEI_QWEN_*` env vars (all optional)."""
        env = os.environ.get
        raw_url = env("XIAO_HEI_QWEN_VLLM_BASE_URL", "") or None
        return cls(
            vllm_base_url=raw_url,
            model=env("XIAO_HEI_QWEN_MODEL", cls.model),
            dtype=env("XIAO_HEI_QWEN_DTYPE", cls.dtype),
            max_model_len=int(env("XIAO_HEI_QWEN_MAX_MODEL_LEN", str(cls.max_model_len))),
            gpu_memory_utilization=float(
                env("XIAO_HEI_QWEN_GPU_MEM_UTIL", str(cls.gpu_memory_utilization)),
            ),
            trust_remote_code=env("XIAO_HEI_QWEN_TRUST_REMOTE", "1") != "0",
            temperature=float(env("XIAO_HEI_QWEN_TEMPERATURE", str(cls.temperature))),
            max_output_tokens=int(
                env("XIAO_HEI_QWEN_MAX_OUTPUT_TOKENS", str(cls.max_output_tokens)),
            ),
            seed=(
                int(env("XIAO_HEI_QWEN_SEED", "0"))
                if "XIAO_HEI_QWEN_SEED" in os.environ
                else cls.seed
            ),
            image_long_edge=int(env("XIAO_HEI_QWEN_IMAGE_LONG_EDGE", str(cls.image_long_edge))),
            max_ticks_per_question=int(
                env("XIAO_HEI_QWEN_MAX_TICKS", str(cls.max_ticks_per_question)),
            ),
        )
