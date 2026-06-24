"""Phase-A responder: frontier-based exploration of an unknown room.

Mirrors the ``dummy/`` and ``qwen/`` package layouts so swapping
``XIAO_HEI_RESPONDER=perception`` in ``app/main.py`` is a one-line
change.

Phase A scope:
  * persistent ``GlobalMap`` stitched from ``terrain_ext`` + pose,
  * per-tick frontier scoring + selection,
  * emit ``WaypointPathResponse`` toward the chosen frontier,
  * no object detection, no localisation, no VLM calls (Phase 2+).
"""

from xiao_hei_vln.perception_responder.responder import PerceptionResponder

__all__ = ["PerceptionResponder"]
