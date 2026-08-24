"""How much of an orbit the furniture leaves — the geometry, not the key."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "scripts", REPO / "perception"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import orbit_arc as OA  # noqa: E402


def obj(id_, label, c, sz):
    c, sz = np.asarray(c, float), np.asarray(sz, float)
    return {"id": id_, "label": label, "c": c, "sz": sz,
            "lo": c - sz / 2, "hi": c + sz / 2}


class TestLongestArc:
    def test_an_empty_ring_is_zero_not_an_error(self):
        assert OA.longest_arc([]) == 0

    def test_a_whole_ring_is_360(self):
        assert OA.longest_arc(list(range(0, 360, 10))) == 360

    def test_scattered_bins_are_not_an_orbit(self):
        """200° free in four pieces is four drives, not one orbit."""
        free = [0, 10, 90, 100, 180, 190, 270, 280]
        assert OA.longest_arc(free) == 20

    def test_a_run_wrapping_past_zero_is_one_run(self):
        assert OA.longest_arc([340, 350, 0, 10]) == 40

    def test_the_longest_of_several_runs_wins(self):
        assert OA.longest_arc([0, 10, 100, 110, 120, 130]) == 40


class TestObstacles:
    def test_a_room_sized_unknown_is_dropped(self):
        """japanese_room's 8.6 x 10.7 m `unknown` blocked every ring."""
        objs = [obj("1", "unknown", [0, 0, 1.0], [8.6, 10.7, 2.0])]
        assert OA.obstacles(objs, "0") == []

    def test_a_straight_wall_segment_still_blocks(self):
        objs = [obj("1", "wall", [0, 2, 1.0], [4.0, 0.2, 2.4])]
        assert len(OA.obstacles(objs, "0")) == 1

    def test_an_l_shaped_wall_is_dropped_because_its_box_is_not_the_wall(self):
        objs = [obj("1", "wall", [0, 0, 1.0], [6.0, 6.0, 2.4])]
        assert OA.obstacles(objs, "0") == []

    def test_the_target_is_not_its_own_obstacle(self):
        objs = [obj("7", "pillow", [0, 0, 0.6], [0.2, 0.4, 0.35])]
        assert OA.obstacles(objs, "7") == []

    def test_a_box_below_the_wheels_is_not_in_the_way(self):
        objs = [obj("1", "book", [0, 0, 0.05], [0.3, 0.2, 0.05])]
        assert OA.obstacles(objs, "0") == []

    def test_a_box_above_the_mast_is_not_in_the_way(self):
        objs = [obj("1", "clock", [0, 0, 2.2], [0.3, 0.1, 0.3])]
        assert OA.obstacles(objs, "0") == []

    def test_furniture_in_the_band_blocks(self):
        objs = [obj("1", "chair", [1, 0, 0.5], [0.5, 0.5, 1.0])]
        assert len(OA.obstacles(objs, "0")) == 1


class TestFreeBins:
    def test_an_empty_room_leaves_the_whole_ring(self):
        got = OA.free_bins(np.zeros(2), [], 1.5, 0.35)
        assert len(got) == 36

    def test_a_wall_on_one_side_takes_that_side(self):
        wall = (np.array([-5.0, 1.2]), np.array([5.0, 1.4]))
        got = OA.free_bins(np.zeros(2), [wall], 1.5, 0.35)
        assert 90 not in got, "straight at the wall"
        assert 270 in got, "away from it"

    def test_a_bigger_robot_fits_in_fewer_places(self):
        wall = (np.array([-5.0, 1.2]), np.array([5.0, 1.4]))
        small = OA.free_bins(np.zeros(2), [wall], 1.5, 0.20)
        big = OA.free_bins(np.zeros(2), [wall], 1.5, 0.60)
        assert len(big) < len(small)

    def test_a_target_boxed_in_leaves_nothing(self):
        ring = [(np.array([-2.0, 1.0]), np.array([2.0, 1.2])),
                (np.array([-2.0, -1.2]), np.array([2.0, -1.0])),
                (np.array([1.0, -2.0]), np.array([1.2, 2.0])),
                (np.array([-1.2, -2.0]), np.array([-1.0, 2.0]))]
        assert OA.free_bins(np.zeros(2), ring, 1.5, 0.35) == []


class TestCensus:
    def test_a_question_with_no_single_target_is_skipped(self, monkeypatch):
        monkeypatch.setattr(OA, "load", lambda scene: [])
        rows = OA.census([{"scene": "x", "i": 0, "resolved": {"n": 2}}],
                         (1.5,), 0.35)
        assert rows == []

    def test_a_target_missing_from_the_annotation_is_skipped(self, monkeypatch):
        monkeypatch.setattr(OA, "load", lambda scene: [])
        rows = OA.census([{"scene": "x", "i": 0, "resolved": {"n": 1, "id": "9"}}],
                         (1.5,), 0.35)
        assert rows == []

    def test_a_resolved_target_is_measured_at_every_radius(self, monkeypatch):
        monkeypatch.setattr(OA, "load", lambda scene: [
            obj("9", "vase", [0, 0, 0.5], [0.2, 0.2, 0.4])])
        rows = OA.census([{"scene": "x", "i": 1, "resolved": {"n": 1, "id": "9"}}],
                         (1.2, 1.8), 0.35)
        assert len(rows) == 1
        assert set(rows[0]["arc"]) == {1.2, 1.8}
        assert rows[0]["arc"][1.2] == (360, 360), "an empty room"
        assert rows[0]["label"] == "vase"
