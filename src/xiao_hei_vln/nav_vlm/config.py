"""Configuration for the VLM navigation waypoint proposer.

All knobs are env-overridable so docker-compose can swap them without a
rebuild. The API key is read from ``ANTHROPIC_API_KEY`` (the SDK's own
convention) with ``XIAO_HEI_ANTHROPIC_API_KEY`` as a namespaced fallback.
Everything else is prefixed ``XIAO_HEI_NAV_VLM_*``.

``from_env`` only fails on a missing key — and only when *called*, which the
explorer defers until the ``nav_vlm`` strategy is actually selected. So the
whole package imports and unit-tests without a key in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class NavVLMConfig:
    """Model + request knobs for the navigation proposer."""

    # --- credentials ------------------------------------------------------
    api_key: str

    # --- model -----------------------------------------------------------
    # Opus 5 is the strongest reasoner for spatial "where next" decisions.
    # The proposer fires at most once per waypoint outcome (not per tick), so
    # a slow, expensive model is affordable here.
    model: str = "claude-opus-5"

    # --- sampling --------------------------------------------------------
    # None = don't send `temperature` at all. Opus 5 (and other recent models)
    # DEPRECATE temperature and the API rejects it with a 400, so the default
    # must omit it; set XIAO_HEI_NAV_VLM_TEMPERATURE only for a model that still
    # accepts it.
    temperature: float | None = None
    max_output_tokens: int = 1024
    # Extended-thinking budget (tokens). 0 disables thinking. Note: forcing a
    # specific tool (tool_choice=propose_waypoint) is incompatible with
    # extended thinking on the Anthropic API, so keep this 0 unless you also
    # switch the engine to tool_choice="auto".
    thinking_budget: int = 0

    # --- image preprocessing --------------------------------------------
    image_long_edge: int = 1024  # downscale panorama long-edge before send
    occupancy_dpi: int = 100     # top-down map render resolution

    # --- network ---------------------------------------------------------
    request_timeout_s: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> NavVLMConfig:
        """Build a config from ``ANTHROPIC_API_KEY`` + ``XIAO_HEI_NAV_VLM_*``.

        Raises:
            ValueError: if no Anthropic API key is set.
        """
        env = os.environ.get
        api_key = (
            env("ANTHROPIC_API_KEY")
            or env("XIAO_HEI_ANTHROPIC_API_KEY")
            or ""
        )
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY (or XIAO_HEI_ANTHROPIC_API_KEY) is required "
                "for the nav_vlm exploration strategy",
            )
        temp_raw = env("XIAO_HEI_NAV_VLM_TEMPERATURE")
        return cls(
            api_key=api_key,
            # `or cls.model`, not a get-default: an env passthrough like
            # `XIAO_HEI_NAV_VLM_MODEL=${…:-}` sets the var to an EMPTY string,
            # which must fall back to the default rather than send model="".
            model=env("XIAO_HEI_NAV_VLM_MODEL") or cls.model,
            # Omitted unless explicitly set (Opus 5 rejects temperature).
            temperature=float(temp_raw) if temp_raw else None,
            max_output_tokens=int(
                env("XIAO_HEI_NAV_VLM_MAX_OUTPUT_TOKENS", str(cls.max_output_tokens)),
            ),
            thinking_budget=int(
                env("XIAO_HEI_NAV_VLM_THINKING_BUDGET", str(cls.thinking_budget)),
            ),
            image_long_edge=int(
                env("XIAO_HEI_NAV_VLM_IMAGE_LONG_EDGE", str(cls.image_long_edge)),
            ),
            occupancy_dpi=int(
                env("XIAO_HEI_NAV_VLM_OCCUPANCY_DPI", str(cls.occupancy_dpi)),
            ),
            request_timeout_s=float(
                env("XIAO_HEI_NAV_VLM_REQUEST_TIMEOUT_S", str(cls.request_timeout_s)),
            ),
            max_retries=int(
                env("XIAO_HEI_NAV_VLM_MAX_RETRIES", str(cls.max_retries)),
            ),
        )
