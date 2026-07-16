"""Engine / responder configuration loaded from environment variables.

Mirrors the shape of :class:`xiao_hei_vln.qwen.config.QwenConfig` — all
knobs are env-overridable so docker-compose can swap them without a
rebuild. The only required env var is ``XIAO_HEI_GEMINI_API_KEY``; the
constructor fails fast if it is missing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiConfig:
    """Engine + responder knobs."""

    # --- credentials ------------------------------------------------------
    api_key: str

    # --- model -----------------------------------------------------------
    # Default to a fast multimodal model; "flash" generations are fast
    # enough to keep the responder near the 2 Hz tick budget even with
    # network round-trip.
    model: str = "gemini-2.5-flash"

    # --- sampling --------------------------------------------------------
    temperature: float = 0.2
    # gemini-2.5-flash is a *thinking* model: reasoning tokens count against
    # ``max_output_tokens``. With a small cap, thinking eats the whole budget
    # and the JSON answer gets truncated ("Unterminated string" at parse).
    # Give the answer real headroom.
    max_output_tokens: int = 2048
    # Thinking budget (tokens). 0 disables thinking entirely — correct for
    # our structured-output task, and it stops thinking from truncating the
    # JSON. Set >0 (or -1 for dynamic) to re-enable via env.
    thinking_budget: int = 0

    # --- image preprocessing --------------------------------------------
    image_long_edge: int = 1280  # downscale long-edge before send

    # --- exploration → answer trigger (Task-1 only) ---------------------
    # If exploration hasn't surfaced any reachable frontier this many
    # ticks in a row, commit to a Gemini answer call. 120 ticks = 60s at
    # 2 Hz; matches the literature notes' "single living room takes
    # ~30-60s to fully explore" finding.
    max_explore_ticks: int = 120

    # --- safety cap ------------------------------------------------------
    # Hard cap on total ticks per question — even if exploration never
    # exhausts, we commit a Gemini answer at this tick.
    max_ticks_per_question: int = 240

    @classmethod
    def from_env(cls) -> GeminiConfig:
        """Build a config from ``XIAO_HEI_GEMINI_*`` env vars.

        Raises:
            ValueError: if ``XIAO_HEI_GEMINI_API_KEY`` is unset or empty.
        """
        env = os.environ.get
        api_key = env("XIAO_HEI_GEMINI_API_KEY", "") or ""
        if not api_key:
            raise ValueError(
                "XIAO_HEI_GEMINI_API_KEY env var is required for the gemini responder",
            )
        return cls(
            api_key=api_key,
            model=env("XIAO_HEI_GEMINI_MODEL", cls.model),
            temperature=float(env("XIAO_HEI_GEMINI_TEMPERATURE", str(cls.temperature))),
            max_output_tokens=int(
                env("XIAO_HEI_GEMINI_MAX_OUTPUT_TOKENS", str(cls.max_output_tokens)),
            ),
            thinking_budget=int(
                env("XIAO_HEI_GEMINI_THINKING_BUDGET", str(cls.thinking_budget)),
            ),
            image_long_edge=int(
                env("XIAO_HEI_GEMINI_IMAGE_LONG_EDGE", str(cls.image_long_edge)),
            ),
            max_explore_ticks=int(
                env("XIAO_HEI_GEMINI_MAX_EXPLORE_TICKS", str(cls.max_explore_ticks)),
            ),
            max_ticks_per_question=int(
                env("XIAO_HEI_GEMINI_MAX_TICKS", str(cls.max_ticks_per_question)),
            ),
        )
