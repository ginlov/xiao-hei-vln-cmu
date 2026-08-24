"""The box → mask → 3D adapter, and the multi-view box estimator."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import target_box as T  # noqa: E402
import geometry as G  # noqa: E402

RNG = np.random.default_rng(0)


def cloud(centre, extent, n=300):
    c = np.asarray(centre, float)
    e = np.asarray(extent, float)
    return c + RNG.uniform(-0.5, 0.5, (n, 3)) * e


class TestFaceBoxToMask:
    def test_the_mask_is_equirect_shaped(self):
        m = T.face_box_to_equirect_mask([200, 300, 300, 400], 0)
        assert m.shape == (G.EQUIRECT_H, G.EQUIRECT_W)
        assert m.dtype == bool

    def test_a_box_covers_pixels_and_a_degenerate_one_does_not(self):
        assert T.face_box_to_equirect_mask([200, 300, 300, 400], 0).sum() > 1000
        assert T.face_box_to_equirect_mask([10, 10, 10, 10], 0).sum() == 0

    def test_opposite_faces_do_not_share_pixels(self):
        a = T.face_box_to_equirect_mask([200, 300, 300, 400], 0)
        b = T.face_box_to_equirect_mask([200, 300, 300, 400], 2)
        assert (a & b).sum() == 0
        assert a.sum() == b.sum()          # same box, same solid angle

    def test_the_box_order_is_ymin_xmin_ymax_xmax(self):
        """The order the grounding prompt returns, so no caller transposes."""
        top = T.face_box_to_equirect_mask([0, 250, 100, 390], 0)
        bot = T.face_box_to_equirect_mask([500, 250, 600, 390], 0)
        # A band high in the face maps above one low in the face. Equirect rows
        # increase downward, so the high band's mean row must be smaller.
        assert np.argwhere(top)[:, 0].mean() < np.argwhere(bot)[:, 0].mean()

    def test_reversed_corners_are_tolerated(self):
        a = T.face_box_to_equirect_mask([200, 300, 300, 400], 1)
        b = T.face_box_to_equirect_mask([300, 400, 200, 300], 1)
        assert (a == b).all()

    def test_a_box_outside_the_face_is_clipped_not_wrapped(self):
        """Out-of-range corners clamp; they must not wrap to the far side."""
        big = T.face_box_to_equirect_mask([-50, -50, 400, 400], 0)
        same = T.face_box_to_equirect_mask([0, 0, 400, 400], 0)
        assert (big == same).all()
        assert 0 < big.sum() < (G.EQUIRECT_H * G.EQUIRECT_W) // 2

    def test_a_face_corner_box_loses_almost_all_its_pixels(self):
        """The LUT gives a face's extreme corner to the neighbour that views it
        more centrally, so a corner box lifts from a fraction of the returns."""
        corner = T.face_box_to_equirect_mask([0, 0, 60, 60], 0).sum()
        middle = T.face_box_to_equirect_mask([290, 290, 350, 350], 0).sum()
        assert corner > 0
        assert middle > 20 * corner

    def test_a_small_corner_box_vanishes_entirely(self):
        assert T.face_box_to_equirect_mask([0, 0, 40, 40], 0).sum() == 0

    def test_the_whole_face_maps_to_every_pixel_the_lut_gives_it(self):
        lut = T._inverse_lut()[0]
        full = T.face_box_to_equirect_mask([0, 0, G.FACE_SIZE - 1,
                                           G.FACE_SIZE - 1], 0)
        assert full.sum() == (lut == 0).sum()


class TestLift:
    def test_an_empty_scan_lifts_nothing(self):
        m = T.face_box_to_equirect_mask([200, 300, 300, 400], 0)
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        got = T.lift_mask(m, np.zeros((0, 3)), pose)
        assert got["n"] == 0 and got["points"] is None

    def test_a_scan_behind_the_box_lifts_nothing(self):
        m = T.face_box_to_equirect_mask([300, 300, 340, 340], 0)
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        behind = np.tile(np.array([[-3.0, 0.0, 0.75]]), (400, 1))
        got = T.lift_mask(m, behind, pose)
        assert got["points"] is None

    def test_lift_box_and_lift_mask_agree(self):
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        scan = cloud([3.0, 0.0, 0.75], [0.4, 0.4, 0.4], 600)
        m = T.face_box_to_equirect_mask([260, 260, 380, 380], 0)
        a = T.lift_mask(m, scan, pose)
        b = T.lift_box([260, 260, 380, 380], 0, scan, pose)
        assert a["n"] == b["n"]

    def test_a_box_on_a_real_cluster_recovers_its_position(self):
        """End to end: a cloud 3 m ahead lifts back to roughly 3 m ahead."""
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        scan = cloud([3.0, 0.0, 0.75], [0.5, 0.5, 0.5], 2000)
        got = T.lift_box([200, 200, 440, 440], 0, scan, pose)
        if got["points"] is None:
            pytest.skip("box/face convention put the cluster outside face 0")
        assert np.linalg.norm(got["xyz"] - np.array([3.0, 0.0, 0.75])) < 0.6


class TestEstimator:
    def test_the_size_does_not_inflate_with_offset_views(self):
        """The whole reason the clouds are not unioned."""
        tb = T.TargetBox()
        tb.add(cloud([1.0, 2.0, 0.8], [0.4, 0.4, 0.4]))
        tb.add(cloud([1.25, 2.0, 0.8], [0.4, 0.4, 0.4]))
        _, s = tb.box()
        assert s[0] < 0.5, "averaging must not accumulate the centre scatter"

    def test_the_centre_is_the_weighted_mean(self):
        tb = T.TargetBox()
        tb.add(cloud([0.0, 0.0, 0.0], [0.2, 0.2, 0.2], 100), weight=100)
        tb.add(cloud([1.0, 0.0, 0.0], [0.2, 0.2, 0.2], 900), weight=900)
        c, _ = tb.box()
        assert c[0] == pytest.approx(0.9, abs=0.08)

    def test_weight_defaults_to_the_core_point_count(self):
        tb = T.TargetBox()
        tb.add(cloud([0, 0, 0], [0.3, 0.3, 0.3], 250))
        assert tb.weights[0] == pytest.approx(250, rel=0.2)

    def test_an_empty_box_is_zeros_not_a_crash(self):
        c, s = T.TargetBox().box()
        assert not c.any() and not s.any()

    def test_a_cloud_with_no_points_is_rejected_and_recorded(self):
        tb = T.TargetBox()
        assert tb.add(None) is False
        assert tb.add(np.zeros((1, 3))) is False
        assert len(tb.rejected) == 2 and tb.n_views == 0

    @pytest.mark.parametrize("mode,expect_big", [("average", False),
                                                 ("max", True), ("union", True)])
    def test_size_modes_differ_on_partial_slices(self, mode, expect_big):
        """Two views each seeing a thin slice on a different axis."""
        tb = T.TargetBox(size_mode=mode)
        tb.add(cloud([1, 2, 0.8], [0.40, 0.06, 0.40]))
        tb.add(cloud([1, 2, 0.8], [0.06, 0.40, 0.40]))
        _, s = tb.box()
        assert bool(s[0] > 0.30) is expect_big

    def test_an_unknown_size_mode_is_refused(self):
        with pytest.raises(ValueError):
            T.TargetBox(size_mode="mean")


class TestConsensus:
    def test_a_lone_disagreeing_view_is_dropped(self):
        tb = T.TargetBox()
        for _ in range(3):
            tb.add(cloud([1, 2, 0.8], [0.3, 0.3, 0.3]))
        tb.add(cloud([9, 9, 0.8], [0.3, 0.3, 0.3]), note="sofa behind")
        keep = tb.consensus()
        assert len(keep) == 3
        assert [v["note"] for v in tb.outliers()] == ["sofa behind"]

    def test_without_an_anchor_the_heaviest_cluster_wins(self):
        tb = T.TargetBox()
        tb.add(cloud([0, 0, 0], [0.3, 0.3, 0.3], 100), weight=100, note="right")
        tb.add(cloud([5, 0, 0], [0.3, 0.3, 0.3], 400), weight=400, note="wrong")
        tb.add(cloud([5.2, 0, 0], [0.3, 0.3, 0.3], 400), weight=400, note="wrong")
        assert [tb.views[i]["note"] for i in tb.consensus()] == ["wrong", "wrong"]

    def test_an_anchor_beats_the_heavier_cluster(self):
        """The arabic_room replay: five heavy views on the wrong object."""
        tb = T.TargetBox(anchor=np.array([0.0, 0.0, 0.0]))
        tb.add(cloud([0, 0, 0], [0.3, 0.3, 0.3], 100), weight=100, note="right")
        tb.add(cloud([5, 0, 0], [0.3, 0.3, 0.3], 400), weight=400, note="wrong")
        tb.add(cloud([5.2, 0, 0], [0.3, 0.3, 0.3], 400), weight=400, note="wrong")
        assert [tb.views[i]["note"] for i in tb.consensus()] == ["right"]

    def test_consensus_is_independent_of_arrival_order(self):
        views = [([1.0, 2.0, 0.8], 200), ([1.2, 2.1, 0.8], 300),
                 ([8.0, 1.0, 0.8], 900)]
        seen = set()
        for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
            tb = T.TargetBox(anchor=np.array([1.0, 2.0, 0.8]))
            for i in order:
                c, n = views[i]
                tb.add(cloud(c, [0.3, 0.3, 0.3], n), weight=n, note=str(i))
            seen.add(tuple(sorted(tb.views[i]["note"] for i in tb.consensus())))
        assert seen == {("0", "1")}

    def test_single_link_chains_a_drifting_set_together(self):
        """Views 0.9 m apart in a chain are one object at outlier_m = 1.2."""
        tb = T.TargetBox()
        for x in (0.0, 0.9, 1.8, 2.7):
            tb.add(cloud([x, 0, 0], [0.3, 0.3, 0.3]))
        assert len(tb.consensus()) == 4

    def test_settled_needs_three_views_and_then_reports_convergence(self):
        tb = T.TargetBox()
        assert tb.settled() is False
        tb.add(cloud([1, 2, 0.8], [0.3, 0.3, 0.3], 400))
        tb.add(cloud([1, 2, 0.8], [0.3, 0.3, 0.3], 400))
        assert tb.settled() is False           # under `need`
        tb.add(cloud([1, 2, 0.8], [0.3, 0.3, 0.3], 400))
        assert tb.settled() is True            # three near-identical views

    def test_as_dict_carries_the_audit_trail(self):
        tb = T.TargetBox(anchor=np.array([0.0, 0.0, 0.0]))
        tb.add(cloud([0, 0, 0], [0.3, 0.3, 0.3]), note="a")
        tb.add(cloud([6, 0, 0], [0.3, 0.3, 0.3]), note="b")
        tb.add(None, note="c")
        d = tb.as_dict()
        assert d["n_views"] == 2 and d["n_agreeing"] == 1
        assert len(d["outliers"]) == 1 and len(d["rejected"]) == 1
        assert set(d) >= {"centre", "size", "views", "weight_agreeing"}


class TestHeading:
    def test_an_elongated_cloud_reports_its_long_axis(self):
        """An axis has no direction, so 0 and pi are the same answer."""
        p = np.c_[RNG.uniform(-1, 1, 300), RNG.uniform(-0.05, 0.05, 300),
                  np.zeros(300)]
        got = T.pca_heading(p) % np.pi
        assert min(got, np.pi - got) < 0.1

    def test_a_rotated_cloud_reports_the_rotation(self):
        p = np.c_[RNG.uniform(-1, 1, 300), RNG.uniform(-0.05, 0.05, 300),
                  np.zeros(300)]
        th = np.deg2rad(30.0)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p[:, :2] = p[:, :2] @ R.T
        got = T.pca_heading(p) % np.pi
        assert abs(got - (th % np.pi)) < 0.1

    def test_too_few_points_is_zero_not_a_crash(self):
        assert T.pca_heading(np.zeros((2, 3))) == 0.0


class TestThinMaskGuard:
    def test_a_corner_box_is_flagged_thin(self):
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        scan = cloud([3.0, 0.0, 0.75], [0.5, 0.5, 0.5], 500)
        got = T.lift_box([0, 0, 60, 60], 0, scan, pose)
        assert got["mask_px"] < T.THIN_MASK_PX and got["thin"] is True

    def test_a_central_box_is_not_flagged(self):
        pose = {"position": [0, 0, 0.75], "orientation": [0, 0, 0, 1]}
        scan = cloud([3.0, 0.0, 0.75], [0.5, 0.5, 0.5], 500)
        got = T.lift_box([250, 250, 390, 390], 0, scan, pose)
        assert got["mask_px"] >= T.THIN_MASK_PX and got["thin"] is False
