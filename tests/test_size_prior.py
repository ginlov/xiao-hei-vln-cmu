"""Tests for the per-class size prior and how the responder applies it."""

from __future__ import annotations

import numpy as np
import pytest

from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.perception.object_map import (
    CLUSTER_MAX_DIM_M,
    ObjectMap,
    _percentile_box,
    _voxel_largest_cluster,
)
from xiao_hei_vln.perception.responder import _size_from_obs
from xiao_hei_vln.perception.size_prior import SIZE_PRIOR, size_for
from xiao_hei_vln.scene import ObjectObservation


class TestSizePriorTable:
    def test_covers_the_common_indoor_classes(self) -> None:
        for label in ("chair", "table", "vase", "picture", "potted plant",
                      "pillow", "book", "lamp", "door", "window"):
            assert size_for(label) is not None, label

    def test_sizes_are_physically_plausible(self) -> None:
        for label, size in SIZE_PRIOR.items():
            assert all(s > 0 for s in size), label
            assert max(size) < 30.0, label      # a room-scale surface at worst

    def test_lookup_normalises_label_form(self) -> None:
        assert size_for("Potted Plant") == size_for("potted plant")
        assert size_for("potted_plant") == size_for("potted plant")

    def test_unknown_label_is_none(self) -> None:
        assert size_for("hookah wire zzz") is None


class TestSizeFromObs:
    def _obs(self, label: str, size: tuple[float, float, float] | None):
        kwargs = {}
        if size is not None:
            kwargs = dict(
                bbox_min=Vector3(x=0.0, y=0.0, z=0.0),
                bbox_max=Vector3(x=size[0], y=size[1], z=size[2]),
            )
        return ObjectObservation(label=label, position=Vector3(x=0, y=0, z=0), **kwargs)

    def test_measurement_wins_when_we_have_one(self) -> None:
        # Measured beats the class median once boxes are robust: per-instance
        # error is 1.27x against ground truth, and a class median cannot know
        # this particular chair. The prior is a fallback, not an override.
        measured = (0.4, 0.3, 0.9)
        got = _size_from_obs(self._obs("chair", measured))
        assert (got.x, got.y, got.z) == pytest.approx(measured)

    def test_measurement_wins_even_when_it_contradicts_the_prior(self) -> None:
        measured = (8.0, 8.0, 8.0)
        got = _size_from_obs(self._obs("vase", measured))
        assert (got.x, got.y, got.z) == pytest.approx(measured)

    def test_prior_used_when_there_is_no_box_at_all(self) -> None:
        # A boxless detection would otherwise answer with a zero-volume box,
        # which scores nothing at all.
        got = _size_from_obs(self._obs("vase", None))
        assert (got.x, got.y, got.z) == pytest.approx(size_for("vase"))

    def test_unknown_label_with_no_prior_uses_the_measurement(self) -> None:
        got = _size_from_obs(self._obs("zzz not a thing", (0.5, 0.5, 0.5)))
        assert (got.x, got.y, got.z) == pytest.approx((0.5, 0.5, 0.5))

    def test_unknown_label_and_no_box_is_zero(self) -> None:
        got = _size_from_obs(self._obs("zzz not a thing", None))
        assert (got.x, got.y, got.z) == (0.0, 0.0, 0.0)


class TestPercentileBox:
    def test_a_single_outlier_no_longer_sets_a_corner(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.normal(0.0, 0.05, size=(200, 3))
        pts[0] = [25.0, 25.0, 0.0]          # one return through a doorway
        lo, hi = _percentile_box(pts)
        assert float(np.max(hi - lo)) < 1.0

    def test_a_genuinely_large_object_keeps_its_extent(self) -> None:
        pts = np.column_stack([
            np.linspace(-1.4, 1.4, 300), np.zeros(300), np.zeros(300),
        ])
        lo, hi = _percentile_box(pts)
        assert float(hi[0] - lo[0]) > 2.5

    def test_tiny_clouds_use_raw_minmax(self) -> None:
        pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0, 0]])
        lo, hi = _percentile_box(pts)
        assert float(hi[0] - lo[0]) == pytest.approx(1.0)


class TestVoxelClustering:
    def test_keeps_the_larger_blob(self) -> None:
        rng = np.random.default_rng(1)
        near = rng.normal([0, 0, 0], 0.05, size=(200, 3))
        far = rng.normal([6, 0, 0], 0.05, size=(30, 3))
        kept = _voxel_largest_cluster(np.vstack([near, far]), 0.25)
        assert len(kept) == 200
        assert float(np.max(kept[:, 0])) < 1.0

    def test_one_blob_is_returned_unchanged(self) -> None:
        rng = np.random.default_rng(2)
        pts = rng.normal(0.0, 0.05, size=(100, 3))
        assert len(_voxel_largest_cluster(pts, 0.25)) == 100


class TestNodeGeometry:
    def test_bimodal_node_is_split_not_bridged(self) -> None:
        # Two rooms' worth of "floor" merged into one node: percentiles alone
        # would still span both, so the cluster fallback has to engage.
        rng = np.random.default_rng(3)
        a = rng.normal([0, 0, 0], 0.1, size=(300, 3))
        b = rng.normal([12, 0, 0], 0.1, size=(60, 3))
        om = ObjectMap()
        om.add("table", 0.9, np.vstack([a, b]))
        node = om.nodes[0]
        assert float(np.max(node.cmax - node.cmin)) < CLUSTER_MAX_DIM_M

    def test_centre_lies_inside_its_own_box(self) -> None:
        # The old code took the centre from a gated cloud and the box from raw
        # min/max, so a node could report a centre 1.3 m outside its own box.
        rng = np.random.default_rng(4)
        pts = np.vstack([
            rng.normal([0, 0, 0], 0.1, size=(200, 3)),
            np.array([[20.0, 20.0, 0.0]]),
        ])
        om = ObjectMap()
        om.add("vase", 0.9, pts)
        node = om.nodes[0]
        assert np.all(node.center >= node.cmin - 1e-6)
        assert np.all(node.center <= node.cmax + 1e-6)
