"""The numerical responder: the answer key, the split, the merge, the loop."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import count_view as CV                                          # noqa: E402
import numerical_plan as NP                                      # noqa: E402
import score_numerical as SN                                     # noqa: E402
from answer_numerical import (FRAME_DEG, VIEW_MAX_M,             # noqa: E402
                              VIEW_MIN_M, anchor_offset, frame_pose,
                              orbit_point)


def box(cx, cy, cz, sx=1.0, sy=1.0, sz=1.0, label="thing", colours=()):
    c = np.array([cx, cy, cz], float)
    s = np.array([sx, sy, sz], float)
    return {"id": "0", "label": label, "c": c, "lo": c - s / 2, "hi": c + s / 2,
            "sz": s, "colours": list(colours)}


class TestRelations:
    def test_on_needs_overlap_and_height(self):
        table = box(0, 0, 0.4, 1.2, 0.8, 0.8)
        cup = box(0, 0, 0.9, .1, .1, .1)
        assert SN.holds("on", cup, table)
        assert not SN.holds("on", box(5, 0, 0.9, .1, .1, .1), table)

    def test_below_needs_the_anchor_overhead(self):
        """A window on the far wall is taller than every sofa in the room."""
        sofa = box(0, 0, 0.35, 2.2, 0.8, 0.7)
        over = box(0, 0.3, 1.8, 1.5, 0.1, 0.9)
        assert SN.holds("below", sofa, over)
        # Same height, different wall: overlaps nothing, so it does not count.
        assert not SN.holds("below", sofa, box(9, 9, 1.8, 1.5, 0.1, 0.9))
        # Overhead in plan but not in height -- a picture at knee level.
        assert not SN.holds("below", sofa, box(0, 0.3, 0.4, 1.5, 0.1, 0.2))

    def test_pad_widens_the_footprint_test(self):
        sofa = box(0, 0, 0.35, 2.0, 0.8, 0.7)
        just_off = box(1.6, 0, 0.9, .3, .3, .3)
        assert not SN.holds("on", just_off, sofa, 0.0)
        assert SN.holds("on", just_off, sofa, 0.5)


class TestColour:
    def test_maroon_answers_to_red(self):
        p = box(0, 0, 0, colours=[{"name": "maroon", "pct": .78,
                                   "rgb": (178, 34, 34), "luma": 78.0}])
        assert SN.is_colour(p, "red")
        assert not SN.is_colour(p, "black")

    def test_only_the_dark_end_of_gray_is_black(self):
        """VLA-3D's `gray` spans darkslategray to darkgray; only one is black."""
        dark = box(0, 0, 0, colours=[{"name": "gray", "pct": 1.0,
                                      "rgb": (47, 79, 79), "luma": 69.4}])
        light = box(0, 0, 0, colours=[{"name": "gray", "pct": 1.0,
                                       "rgb": (112, 128, 144), "luma": 124.8}])
        assert SN.is_colour(dark, "black")
        assert not SN.is_colour(light, "black")

    def test_a_second_colour_does_not_count(self):
        """A grey pillow with a red pattern is a grey pillow."""
        p = box(0, 0, 0, colours=[
            {"name": "gray", "pct": .7, "rgb": (150, 150, 150), "luma": 150.0},
            {"name": "maroon", "pct": .3, "rgb": (178, 34, 34), "luma": 78.0}])
        assert not SN.is_colour(p, "red")

    def test_no_colour_recorded_is_not_a_match(self):
        assert not SN.is_colour(box(0, 0, 0), "red")


class TestKey:
    """The key is only as good as its agreement with the challenge's own text."""

    def test_every_spec_matches_the_released_question(self):
        assert SN.check_questions() == []

    def test_every_scene_has_a_spec(self):
        assert len(SN.SPECS) == 15

    @pytest.mark.skipif(not (SN.VLA3D / "hotel_room_1").is_dir(),
                        reason="VLA-3D annotation not present")
    def test_a_known_answer(self):
        objs = SN.load("hotel_room_1")
        assert SN.answer("hotel_room_1", SN.SPECS["hotel_room_1"], objs)["answer"] == 4

    @pytest.mark.skipif(not (SN.VLA3D / "chinese_room").is_dir(),
                        reason="VLA-3D annotation not present")
    def test_counting_anchors_not_targets(self):
        """'chairs with pillows on them' is a number of chairs."""
        spec = SN.SPECS["chinese_room"]
        assert spec["count"] == "anchors"
        objs = SN.load("chinese_room")
        got = SN.answer("chinese_room", spec, objs)
        assert got["answer"] <= len(SN.of(objs, "chair"))


class TestSplit:
    def test_regex_splits_at_the_first_preposition_not_the_first_listed(self):
        """'on the sofa under the pictures' splits at `on`, not at `under`."""
        p = NP._clean(NP._fallback(
            "How many pillows are on the sofa under the pictures?"))
        assert p["target"] == "pillow"
        assert p["relation"] == "on"
        assert p["anchor"] == "sofa"
        assert p["anchor_qualifier"] == "under the pictures"

    def test_colour_comes_off_the_target(self):
        p = NP._clean(NP._fallback("How many red pillows are on the sofa?"))
        assert (p["target"], p["attribute"]) == ("pillow", "red")

    def test_determiner_is_reported_verbatim(self):
        assert NP._clean(NP._fallback(
            "How many pillows are on a sofa?"))["anchor_determiner"] == "a"
        assert NP._clean(NP._fallback(
            "How many pillows are on the sofa?"))["anchor_determiner"] == "the"

    def test_anchor_scope_needs_an_anchor(self):
        p = NP._clean({"target": "chair", "scope": "anchor", "anchor": None})
        assert p["scope"] == "scene"

    def test_phrase_carries_the_qualifier(self):
        """'the sofa' in a room with four is not a referring expression."""
        p = NP._clean(NP._fallback(
            "How many pillows are on the sofa under the pictures?"))
        assert NP.phrase_for(p) == "the sofa under the pictures"


class TestBlind:
    def test_the_prompt_cannot_be_told_a_running_total(self):
        """The blinding rule is enforced by there being nowhere to put one."""
        import inspect
        for fn in (CV.build_prompt, CV.count_view):
            names = set(inspect.signature(fn).parameters)
            assert not names & {"seen", "total", "so_far", "previous", "tally",
                                "history", "count"}

    def test_no_view_count_reaches_the_prompt(self):
        p = CV.build_prompt("How many pillows are on the bed?", "pillow", "bed")
        assert "pillow" in p and "bed" in p
        for word in ("so far", "previously", "running total", "last time"):
            assert word not in p.lower()

    def test_the_prompt_says_not_to_total_the_room(self):
        p = CV.build_prompt("How many cups are on the table?", "cup", "table")
        assert "FROM HERE" in p


class TestMerge:
    def test_two_views_of_one_object_answer_one(self):
        a = [{"xy": np.array([1.0, 0.0]), "note": "left", "certain": True}]
        b = [{"xy": np.array([1.1, 0.0]), "note": "the near one", "certain": True}]
        assert CV.tally([a, b])["count"] == 1

    def test_distinct_objects_stay_distinct(self):
        a = [{"xy": np.array([0.0, 0.0]), "note": "a", "certain": True},
             {"xy": np.array([0.5, 0.0]), "note": "b", "certain": True}]
        assert CV.tally([a])["count"] == 2

    def test_the_answer_is_never_the_sum_of_the_views(self):
        v = [{"xy": np.array([0.0, 0.0]), "note": "x", "certain": True},
             {"xy": np.array([0.45, 0.0]), "note": "y", "certain": True}]
        t = CV.tally([v, list(v), list(v)])
        assert t["per_view"] == [2, 2, 2]
        assert t["count"] == 2

    def test_unplaced_instances_are_taken_at_the_best_single_view(self):
        """Summing them would count one object once per view that saw it."""
        v = [{"xy": None, "note": "a cup", "certain": True},
             {"xy": None, "note": "another", "certain": True}]
        t = CV.tally([v, list(v)])
        assert t["unplaced"] == 2 and t["count"] == 2

    def test_placed_and_unplaced_add(self):
        v = [{"xy": np.array([0.0, 0.0]), "note": "seen", "certain": True},
             {"xy": None, "note": "unmeasured", "certain": True}]
        t = CV.tally([v])
        assert (t["clusters"], t["unplaced"], t["count"]) == (1, 1, 2)

    def test_a_second_view_upgrades_an_uncertain_instance(self):
        a = [{"xy": np.array([1.0, 0.0]), "note": "half hidden", "certain": False}]
        b = [{"xy": np.array([1.05, 0.0]), "note": "plain", "certain": True}]
        out = CV.merge(CV.merge([], a), b)
        assert len(out) == 1 and out[0]["certain"] and out[0]["seen"] == 2


class TestReplyHandling:
    def test_the_list_wins_over_the_models_arithmetic(self, monkeypatch):
        monkeypatch.setattr(CV, "ask_claude", lambda *a, **k:
                            '{"instances": [{"image_index": 0}], "count_here": 7,'
                            ' "sufficient": true}')
        r = CV.count_view([b""], "q", "cup", "table")
        assert r["count_here"] == 1
        assert r["count_mismatch"] == [7, 1]

    def test_a_failed_call_costs_a_look_not_the_question(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("529")
        monkeypatch.setattr(CV, "ask_claude", boom)
        r = CV.count_view([b""], "q", "cup", "table")
        assert r["count_here"] == 0 and r["sufficient"] is False and r["error"]

    def test_an_unparseable_reply_is_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(CV, "ask_claude", lambda *a, **k: "sorry, no")
        r = CV.count_view([b""], "q", "cup", "table")
        assert r["error"] == "unparseable reply" and r["instances"] == []

    def test_sufficient_is_never_true_by_accident(self, monkeypatch):
        monkeypatch.setattr(CV, "ask_claude", lambda *a, **k: '{"instances": []}')
        assert CV.count_view([b""], "q", "cup", "table")["sufficient"] is False


class TestViewpoints:
    def test_backing_off_keeps_the_bearing(self):
        pose = {"position": [4.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]}
        got = anchor_offset(pose, np.array([0.0, 0.0]), 2.6)
        assert np.allclose(got, [2.6, 0.0])

    def test_orbit_keeps_the_distance(self):
        pose = {"position": [3.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]}
        got = orbit_point(pose, np.array([0.0, 0.0]), 55.0, 2.6)
        assert np.linalg.norm(got) == pytest.approx(2.6)

    def test_standing_on_the_anchor_still_yields_a_point(self):
        pose = {"position": [0.0, 0.0, 0.0], "orientation": [0, 0, 0, 1]}
        got = anchor_offset(pose, np.array([0.0, 0.0]), 2.6)
        assert np.isfinite(got).all()
        assert np.linalg.norm(got) == pytest.approx(2.6)


class TestFraming:
    """Where to stand is decided by how far apart the counted things are.

    The 2.6 m standoff that frames a bed does not frame a 2.9 m desk, and the
    live `office_1` run is the case: parked at the desk's far corner it counted
    3 of 6 and said `too_far` or `occluded` four times running.
    """

    def _set(self, pts):
        return [{"xy": np.array(p, float), "note": "", "certain": True}
                for p in pts]

    def test_it_stands_square_to_the_long_axis(self):
        row = self._set([(0, 0), (1, 0), (2, 0), (3, 0)])
        pose = {"position": [1.5, -3.0, 0.0], "orientation": [0, 0, 0, 1]}
        got = frame_pose(row, pose)
        # Centred on the set along its axis, offset along the normal.
        assert got[0] == pytest.approx(1.5, abs=1e-6)
        assert got[1] < 0

    def test_it_keeps_the_side_the_vehicle_is_on(self):
        row = self._set([(0, 0), (3, 0)])
        near = frame_pose(row, {"position": [1.5, -4.0, 0.0],
                                "orientation": [0, 0, 0, 1]})
        far = frame_pose(row, {"position": [1.5, 4.0, 0.0],
                               "orientation": [0, 0, 0, 1]})
        assert near[1] < 0 < far[1]

    def test_a_wider_set_is_viewed_from_further_back(self):
        pose = {"position": [0.0, -6.0, 0.0], "orientation": [0, 0, 0, 1]}
        tight = frame_pose(self._set([(-0.3, 0), (0.3, 0)]), pose)
        wide = frame_pose(self._set([(-2.0, 0), (2.0, 0)]), pose)
        assert abs(wide[1]) > abs(tight[1])

    def test_the_standoff_is_clamped_both_ways(self):
        pose = {"position": [0.0, -9.0, 0.0], "orientation": [0, 0, 0, 1]}
        tiny = frame_pose(self._set([(-0.01, 0), (0.01, 0)]), pose)
        huge = frame_pose(self._set([(-20.0, 0), (20.0, 0)]), pose)
        assert abs(tiny[1]) == pytest.approx(VIEW_MIN_M)
        assert abs(huge[1]) == pytest.approx(VIEW_MAX_M)

    def test_the_set_subtends_the_intended_angle(self):
        pose = {"position": [0.0, -6.0, 0.0], "orientation": [0, 0, 0, 1]}
        half = 2.0                      # wide enough that neither clamp binds
        got = frame_pose(self._set([(-half, 0), (half, 0)]), pose)
        deg = 2 * np.degrees(np.arctan2(half, abs(got[1])))
        assert deg == pytest.approx(FRAME_DEG, abs=0.5)

    def test_a_narrow_set_is_held_at_the_minimum_rather_than_crowded(self):
        """Below `VIEW_MIN_M` the angle is not the binding constraint.

        Two pillows 0.6 m apart would subtend 70 deg from 0.43 m, which is
        inside the platform's own obstacle clearance and would put the vehicle
        in the sofa. The clamp wins and the set simply fills less of the frame.
        """
        pose = {"position": [0.0, -6.0, 0.0], "orientation": [0, 0, 0, 1]}
        got = frame_pose(self._set([(-0.3, 0), (0.3, 0)]), pose)
        assert abs(got[1]) == pytest.approx(VIEW_MIN_M)
        assert 2 * np.degrees(np.arctan2(0.3, abs(got[1]))) < FRAME_DEG

    def test_one_instance_defines_no_axis(self):
        pose = {"position": [0.0, -3.0, 0.0], "orientation": [0, 0, 0, 1]}
        assert frame_pose(self._set([(0, 0)]), pose) is None
        assert frame_pose([], pose) is None

    def test_unplaced_instances_do_not_vote(self):
        pose = {"position": [0.0, -3.0, 0.0], "orientation": [0, 0, 0, 1]}
        row = self._set([(0, 0)]) + [{"xy": None, "note": "", "certain": True}]
        assert frame_pose(row, pose) is None

    def test_the_office_1_case(self):
        """The six monitors, and the corner the live run actually parked in."""
        monitors = [(1.89, -1.64), (2.69, -2.01), (2.72, -1.65),
                    (3.79, -1.65), (3.66, -2.00), (1.81, -2.00)]
        pose = {"position": [4.81, -3.54, 0.0], "orientation": [0, 0, 0, 1]}
        got = frame_pose(self._set(monitors), pose)
        # Within 10 cm of the pose the run drove through on its way past.
        assert float(np.linalg.norm(got - np.array([2.71, -3.77]))) < 0.10
        # And a real move from where it gave up.
        assert float(np.linalg.norm(
            got - np.array(pose["position"][:2]))) > 1.0
