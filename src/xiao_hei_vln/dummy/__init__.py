"""Reference dummy VLM ported from `CMU-VLN-Challenge-2026/ai_module/src/dummy_vlm/`.

This package exists so we can run our end-to-end stack against the
challenge simulator before the real VLM is ready. Replace
`DummyResponder` with the real model and the rest of the system stays
identical.
"""

from xiao_hei_vln.dummy.fixtures import read_object_list, read_waypoints_ply
from xiao_hei_vln.dummy.responder import DummyResponder

__all__ = ["DummyResponder", "read_object_list", "read_waypoints_ply"]
