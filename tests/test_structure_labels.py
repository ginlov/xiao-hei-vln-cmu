"""Tests for the structure ("stuff") vs instance ("thing") split.

Walls, floors and ceilings are detected like anything else, but one mask
covers an entire surface — often spanning rooms — so lifting and fusing them
yields dozens of overlapping nodes that are all the same floor. They are
tagged rather than dropped (a counting question may ask about doors), and the
instance consumers filter on the tag.
"""

from __future__ import annotations

import numpy as np

from xiao_hei_vln.detected_dataset.builder import scene_graph_to_object_list
from xiao_hei_vln.perception.object_map import ObjectMap
from xiao_hei_vln.perception.vocab import STRUCTURE_LABELS, is_structure


class TestIsStructure:
    def test_architecture_is_structure(self) -> None:
        for label in ("wall", "floor", "ceiling", "door", "window", "carpet"):
            assert is_structure(label), label

    def test_real_objects_are_not(self) -> None:
        for label in ("chair", "vase", "potted plant", "sofa", "picture", "book"):
            assert not is_structure(label), label

    def test_matching_is_case_and_space_insensitive(self) -> None:
        assert is_structure("Door Frame")
        assert is_structure("  FLOOR ")
        assert is_structure("door_frame")

    def test_unknown_label_is_not_structure(self) -> None:
        # An open-vocabulary detector can return anything; default to
        # "instance" so a real object is never silently dropped.
        assert not is_structure("hookah")
        assert not is_structure("")


class TestObjectMapTagging:
    def _pts(self, x: float) -> np.ndarray:
        return np.random.default_rng(0).normal(loc=[x, 0, 0], scale=0.05, size=(40, 3))

    def test_nodes_carry_the_flag(self) -> None:
        om = ObjectMap()
        om.add("floor", 0.7, self._pts(0.0))
        om.add("vase", 0.7, self._pts(5.0))
        by_label = {n["label"]: n for n in om.export()}
        assert by_label["floor"]["is_structure"] is True
        assert by_label["vase"]["is_structure"] is False

    def test_flag_follows_a_relabel_on_merge(self) -> None:
        # merge() adopts the higher-scoring observation's label, so the
        # structure verdict has to move with it or the node keeps a stale tag.
        om = ObjectMap()
        pts = self._pts(0.0)
        om.add("wall", 0.4, pts)
        om.nodes[0].merge("picture", 0.9, pts)
        assert om.nodes[0].label == "picture"
        assert om.nodes[0].is_structure is False


class TestDatasetBuilderFiltering:
    def _scene(self) -> dict:
        return {"objects": [
            {"label": "floor", "position": [0, 0, 0], "is_structure": True},
            {"label": "vase", "position": [1, 1, 0.5], "is_structure": False},
            {"label": "chair", "position": [2, 2, 0.5]},        # no flag at all
        ]}

    def test_structure_is_dropped_by_default(self) -> None:
        lines = scene_graph_to_object_list(self._scene())
        assert len(lines) == 2
        assert all("floor" not in ln for ln in lines)
        # Ids are reassigned over what survives, with no gap where the floor was.
        assert [ln.split()[0] for ln in lines] == ["0", "1"]

    def test_structure_can_be_kept_explicitly(self) -> None:
        assert len(scene_graph_to_object_list(self._scene(), include_structure=True)) == 3

    def test_scene_graphs_without_the_flag_pass_through(self) -> None:
        # Recorded before the flag existed: nothing is tagged, so nothing is
        # dropped rather than everything.
        old = {"objects": [{"label": "wall", "position": [0, 0, 0]}]}
        assert len(scene_graph_to_object_list(old)) == 1


def test_structure_label_set_is_normalised() -> None:
    # is_structure() normalises its input; entries that are not already
    # lower-case single-spaced could never match.
    for label in STRUCTURE_LABELS:
        assert label == label.strip().lower()
        assert "  " not in label
