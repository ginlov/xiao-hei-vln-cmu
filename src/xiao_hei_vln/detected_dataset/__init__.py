"""Build an LLM-eval dataset from real perception instead of ground truth.

See :mod:`xiao_hei_vln.detected_dataset.builder`. CLI: ``python -m
xiao_hei_vln.detected_dataset``.
"""

from xiao_hei_vln.detected_dataset.builder import (
    build_dataset,
    extract_scene_graph_from_session,
    load_qa_for_scene,
    scene_graph_to_object_list,
    splice_detected,
    truth_records,
)

__all__ = [
    "build_dataset",
    "extract_scene_graph_from_session",
    "load_qa_for_scene",
    "scene_graph_to_object_list",
    "splice_detected",
    "truth_records",
]
