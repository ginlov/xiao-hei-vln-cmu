"""Perception-responder package.

Talks HTTP to the perception sidecar (YOLOv8x-World v2 + SAM 2.1 Hiera
Tiny, see `perception/` at the repo root), projects each returned
equirect mask through the LiDAR scan to lift to 3D, and pushes the
detections into the shared :class:`SceneRepresentation` via
:meth:`add_object`.

The :class:`PerceptionResponder` runs a Phase A (walk coverage
trajectory) / Phase B (answer question) split, integrated with the
report tooling and logger wiring.
"""

from xiao_hei_vln.perception.responder import PerceptionResponder

__all__ = ["PerceptionResponder"]
