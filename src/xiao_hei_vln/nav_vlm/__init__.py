"""VLM-backed navigation: propose the next waypoint from an image + map.

This package is the *waypoint proposer* half of a VLM navigation stack. It
does not drive the robot — it decides *where to go next* and hands a single
``Waypoint`` to the existing local planner (published on
``/way_point_with_heading``), exactly where the geometric explorers
(``FrontierExplorer`` / ``NextBestViewExplorer``) sit today.

Layers
------
config
    :class:`NavVLMConfig` — env-var configuration (API key, model, image
    budget, timeouts). Backend-agnostic; ``ANTHROPIC_API_KEY`` is only
    required when the Anthropic engine is actually constructed.
engine
    :class:`NavVLMEngineProtocol` — the one method the explorer depends on,
    ``propose(...) -> WaypointProposal``. :class:`AnthropicNavEngine` is the
    Opus-5 implementation (lazy ``anthropic`` import; a fake client can be
    injected for tests). Any object satisfying the protocol works, so a
    Gemini-backed engine can drop in later without touching the explorer.
prompts
    The system prompt + tool schema that turn a free-form model into a
    structured ``propose_waypoint(x, y, heading, done, rationale)`` caller.
render
    Top-down PNG of the exploration occupancy grid — the spatial grounding
    the model needs so it stops proposing waypoints into walls.

The explorer itself lives at
:class:`xiao_hei_vln.exploration.NavVLMExplorer` so it sits beside the other
strategies and is selected with ``XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm``.
"""

from __future__ import annotations

from xiao_hei_vln.nav_vlm.config import NavVLMConfig
from xiao_hei_vln.nav_vlm.engine import (
    AnthropicNavEngine,
    NavVLMEngineProtocol,
    WaypointProposal,
)

__all__ = [
    "AnthropicNavEngine",
    "NavVLMConfig",
    "NavVLMEngineProtocol",
    "WaypointProposal",
]
