"""The plan executor's geometry and its clause split.

Only the parts that decide where the robot goes are tested here: the drive
itself needs a simulator, but choosing a point inside a passage is arithmetic
and is exactly where a wrong answer is invisible in a log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from approach_loop import (JUMP_M, REVISIT_M, bind_target,  # noqa: E402
                           corroborated, revisited)
from execute_plan import (THROUGH_M, far_side_goal, gate_point,  # noqa: E402
                          same_thing, through_point, went_between, xy_of)
from vlm_probe import parse  # noqa: E402
from instruction_plan import (AVOID, GOTO, PASS, Clause,  # noqa: E402
                              keepouts, parse_instruction, steps)


def anchor(name, x, y):
    return {"name": name, "xy": np.array([float(x), float(y)])}


class TestGatePoint:
    def test_between_takes_the_midpoint(self):
        got = gate_point([anchor("tv", 0, 0), anchor("bed", 4, 0)],
                         "between", np.zeros(2))
        assert got is not None
        mid, why, sides = got
        assert np.allclose(mid, [2.0, 0.0])
        assert "4.00 m apart" in why
        assert sides is not None and len(sides) == 2

    def test_between_uses_the_widest_pair(self):
        """A third reported object must not shrink the gap it is not part of."""
        mid, why, sides = gate_point(
            [anchor("tv", 0, 0), anchor("lamp", 1, 0), anchor("bed", 6, 0)],
            "between", np.zeros(2))
        assert np.allclose(mid, [3.0, 0.0])
        assert "6.00 m apart" in why
        assert np.allclose(sorted(s[0] for s in sides), [0.0, 6.0])

    def test_two_anchors_on_the_same_spot_are_not_a_gap(self):
        """One object reported twice would put the waypoint inside it."""
        assert gate_point([anchor("tv", 1, 1), anchor("tv", 1.2, 1.1)],
                          "between", np.zeros(2)) is None

    def test_between_with_one_anchor_falls_back_to_alongside(self):
        mid, why, sides = gate_point([anchor("tv", 3, 0)], "between",
                                     np.zeros(2))
        assert np.allclose(mid, [3.0, 0.0])
        assert "alongside" in why
        assert sides is None, "a single landmark has no line to cross"

    def test_near_picks_the_closer_landmark(self):
        mid, why, sides = gate_point(
            [anchor("stairs", 8, 0), anchor("fireplace", 2, 0)], "near",
            np.zeros(2))
        assert np.allclose(mid, [2.0, 0.0])
        assert "fireplace" in why
        assert sides is None

    def test_nothing_lifted_is_no_gate(self):
        assert gate_point([], "between", np.zeros(2)) is None


class TestWentBetween:
    COUCH = np.array([4.47, -1.08])
    TABLE = np.array([2.60, -1.07])

    def test_the_studio_path_did_not_go_between(self):
        """The path that reported a passage it never made.

        Recorded from `runs/exec_studio3`: the vehicle came down to y=+0.30,
        1.4 m north of the couch-table line, then turned west and went round.
        """
        track = [[0.00, 0.00], [1.01, 1.05], [2.38, 0.61], [2.79, 0.30],
                 [1.60, -0.41], [1.53, -0.44], [0.29, -3.10], [0.30, -3.15]]
        assert not went_between(track, self.COUCH, self.TABLE)

    def test_a_path_through_the_gap_counts(self):
        track = [[3.5, 1.0], [3.5, 0.0], [3.5, -2.0], [3.5, -3.0]]
        assert went_between(track, self.COUCH, self.TABLE)

    def test_passing_outside_the_anchors_does_not_count(self):
        """Crossing the *line* beyond an anchor is going round, not through."""
        track = [[6.0, 1.0], [6.0, -3.0]]
        assert not went_between(track, self.COUCH, self.TABLE)

    def test_a_track_too_short_to_have_moved(self):
        assert not went_between([[3.5, 1.0]], self.COUCH, self.TABLE)
        assert not went_between([], self.COUCH, self.TABLE)


class TestSameThing:
    def test_the_studio_duplicate(self):
        """`couch` and `couch (left view)` were used as the two sides of a gap."""
        assert same_thing("couch", "couch (left view)")

    def test_a_real_pair_is_not_a_duplicate(self):
        assert not same_thing("couch", "table")
        assert not same_thing("the TV", "the bed")

    def test_articles_do_not_matter(self):
        assert same_thing("the couch", "couch")

    def test_a_qualified_name_matches_its_head(self):
        assert same_thing("coffee table", "table")

    def test_empty_names_never_match(self):
        assert not same_thing("", "couch")
        assert not same_thing("?", "")


class TestFarSideGoal:
    class FakeCM:
        """Just the two methods `far_side_goal` uses."""

        def __init__(self, pts):
            self.pts = np.asarray(pts, float)

        def legal_points(self):
            return self.pts

        def settle(self, waypoint, vehicle):
            return np.asarray(waypoint, float)

    SIDES = (np.array([0.0, 0.0]), np.array([0.0, 4.0]))   # gap along x=0

    def test_it_refuses_the_near_side_however_close(self):
        """The `studio` shape: many near points, few far ones, far must win."""
        near = [[-1.4, 2.0], [-1.5, 1.5], [-1.6, 2.5]]
        far = [[3.3, 2.0]]
        cm = self.FakeCM(near + far)
        got = far_side_goal(cm, self.SIDES, np.array([-3.0, 2.0]),
                            np.array([1.2, 2.0]))
        assert got is not None
        assert np.allclose(got[0], [3.3, 2.0])

    def test_it_flips_with_the_vehicle(self):
        cm = self.FakeCM([[-2.0, 2.0], [2.0, 2.0]])
        assert np.allclose(
            far_side_goal(cm, self.SIDES, np.array([5.0, 2.0]),
                          np.array([-1.2, 2.0]))[0], [-2.0, 2.0])

    def test_points_beyond_the_doorway_do_not_count(self):
        """Far side, but 9 m along the wall — that is not through the gap."""
        cm = self.FakeCM([[2.0, 12.0]])
        assert far_side_goal(cm, self.SIDES, np.array([-3.0, 2.0]),
                             np.array([1.2, 2.0])) is None

    def test_nothing_legal_beyond_the_gap(self):
        cm = self.FakeCM([[-1.0, 2.0], [-2.0, 2.0]])
        assert far_side_goal(cm, self.SIDES, np.array([-3.0, 2.0]),
                             np.array([1.2, 2.0])) is None

    def test_a_vehicle_on_the_line_has_no_far_side(self):
        cm = self.FakeCM([[2.0, 2.0], [-2.0, 2.0]])
        assert far_side_goal(cm, self.SIDES, np.array([0.0, 2.0]),
                             np.array([1.2, 2.0])) is None


class TestThroughPoint:
    def test_it_lands_on_the_far_side(self):
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        veh = np.array([-3.0, 2.0])
        p = through_point(a, b, veh)
        assert p[0] > 0, "the vehicle is west of the gap, so aim east"
        assert np.isclose(p[1], 2.0)
        assert np.isclose(np.linalg.norm(p - np.array([0.0, 2.0])), THROUGH_M)

    def test_it_flips_with_the_vehicle(self):
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        assert through_point(a, b, np.array([3.0, 2.0]))[0] < 0

    def test_driving_to_it_would_cross_the_gap(self):
        """The property the whole thing exists for."""
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        veh = np.array([-3.0, 2.0])
        assert went_between([veh.tolist(), through_point(a, b, veh).tolist()],
                            a, b)

    def test_coincident_anchors_degrade_to_the_midpoint(self):
        a = np.array([1.0, 1.0])
        assert np.allclose(through_point(a, a.copy(), np.zeros(2)), a)


class TestXyOf:
    def test_capture_shape(self):
        assert np.allclose(
            xy_of({"position": [1.0, 2.0, 3.0], "orientation": [0, 0, 0, 1]}),
            [1.0, 2.0])

    def test_drive_shape(self):
        """`robot_io`'s driver answers a bare list, not a dict."""
        assert np.allclose(xy_of([1.0, 2.0, 3.0]), [1.0, 2.0])

    def test_missing(self):
        assert xy_of(None) is None


class TestRevisited:
    def test_the_studio_cycle(self):
        """The three poses that burned a whole leg walking a ring."""
        stood = [np.array([2.81, 0.29]), np.array([1.53, -0.41]),
                 np.array([2.28, -3.51]), np.array([2.85, -3.55])]
        assert revisited(np.array([2.31, -3.57]), stood)

    def test_the_previous_pose_does_not_count(self):
        """A short legitimate move is not a cycle; the progress tests own it."""
        stood = [np.array([0.0, 0.0]), np.array([5.0, 0.0])]
        assert not revisited(np.array([5.1, 0.0]), stood)

    def test_a_straight_run_is_not_a_cycle(self):
        stood = [np.array([0.0, 0.0]), np.array([2.0, 0.0]),
                 np.array([4.0, 0.0])]
        assert not revisited(np.array([6.0, 0.0]), stood)

    def test_nothing_stood_in_yet(self):
        assert not revisited(np.array([0.0, 0.0]), [])
        assert not revisited(np.array([0.0, 0.0]), [np.array([0.0, 0.0])])

    def test_the_threshold_clears_the_waypoint_radius(self):
        """Landing on a waypoint already visited has to count as a return."""
        from waypoint_converter_model import WAYPOINT_XY_RADIUS
        assert REVISIT_M > WAYPOINT_XY_RADIUS


class FakeWp:
    """The parts of a waypoint `bind_target` reads."""

    def __init__(self, xy, range_m):
        self.xy = np.asarray(xy, float)
        self.range_m = range_m
        self.committed = True


def bind(seen, origin, bound, pending, *, conf=0.6, switched=None,
         measured=True):
    """Drive one reading through `bind_target` and return the new binding.

    `seen` is where the target should land; the waypoint direction and range
    are constructed to put it there, which is how the real caller works.
    """
    origin = np.asarray(origin, float)
    seen = np.asarray(seen, float)
    d = seen - origin
    r = float(np.linalg.norm(d))
    wp = FakeWp(origin + d / r * 2.0, r)
    reply = {"confidence": conf, "same_object_as_previous": switched}
    _, out = bind_target(wp, origin, reply, bound, {}, verified=True,
                         measured=measured, pending=pending)
    return out


class TestBindingArbitration:
    """The `studio` leg that reported arrival 3.2 m from the right window.

    Positions are the ones actually logged in `runs/exec_studio3`; the correct
    window sits at (+3.30, -4.75).
    """

    TRUTH = np.array([3.30, -4.75])
    WRONG = np.array([0.85, -2.65])
    R6 = np.array([3.57, -4.93])
    R7 = np.array([4.09, -4.97])

    def test_equal_confidence_no_longer_blocks_a_reported_switch(self):
        """`0.6 > 0.6` was false, so a correct reading was refused twice."""
        pending = []
        b = bind(self.WRONG, [0.0, 0.0], None, pending, conf=0.6)
        b = bind(self.R6, [1.0, -1.0], b, pending, conf=0.6, switched=False)
        assert np.allclose(b["xy"], self.R6)
        assert np.linalg.norm(b["xy"] - self.TRUTH) < 0.4

    def test_two_agreeing_readings_overrule_the_binding(self):
        """Independent of what the model says about identity."""
        pending = []
        b = bind(self.WRONG, [0.0, 0.0], None, pending, conf=0.6)
        b = bind(self.R6, [1.0, -1.0], b, pending, conf=0.6, switched=True)
        assert np.allclose(b["xy"], self.WRONG), "one reading must not move it"
        b = bind(self.R7, [2.0, -2.0], b, pending, conf=0.6, switched=True)
        assert np.allclose(b["xy"], self.R7), "two that agree must"

    def test_one_wild_reading_still_cannot_move_it(self):
        """`japanese_room`: a single 4.18 m jump followed the wrong lantern."""
        pending = []
        b = bind([1.0, 1.0], [0.0, 0.0], None, pending, conf=0.8)
        b = bind([5.0, 2.0], [0.5, 0.5], b, pending, conf=0.6, switched=True)
        assert np.allclose(b["xy"], [1.0, 1.0])

    def test_disagreeing_refusals_do_not_accumulate(self):
        """Two refusals that disagree with each other are two misreads."""
        pending = []
        b = bind([1.0, 1.0], [0.0, 0.0], None, pending, conf=0.8)
        b = bind([5.0, 2.0], [0.5, 0.5], b, pending, conf=0.6, switched=True)
        b = bind([-4.0, 3.0], [0.5, 0.5], b, pending, conf=0.6, switched=True)
        assert np.allclose(b["xy"], [1.0, 1.0])

    def test_an_accepted_reading_breaks_the_run(self):
        """A refusal, an agreement, then a refusal is not two in a row."""
        pending = []
        b = bind([1.0, 1.0], [0.0, 0.0], None, pending, conf=0.8)
        b = bind([5.0, 2.0], [0.5, 0.5], b, pending, conf=0.6, switched=True)
        b = bind([1.2, 1.1], [0.5, 0.5], b, pending, conf=0.6)   # refines
        assert pending == []
        b = bind([5.1, 2.1], [0.5, 0.5], b, pending, conf=0.6, switched=True)
        assert np.linalg.norm(b["xy"] - np.array([1.2, 1.1])) < 0.3


class TestCorroborated:
    def test_needs_a_previous_refusal(self):
        assert not corroborated(np.array([1.0, 1.0]), [])
        assert not corroborated(np.array([1.0, 1.0]), None)

    def test_agreement_is_within_the_jump_gate(self):
        assert corroborated(np.array([1.0, 1.0]),
                            [np.array([1.0 + JUMP_M * 0.9, 1.0])])
        assert not corroborated(np.array([1.0, 1.0]),
                                [np.array([1.0 + JUMP_M * 1.1, 1.0])])


class TestPlanSplit:
    def test_keepouts_leave_the_drive_order(self):
        plan = parse_instruction(
            "First, go to the chair near the window, then stop at the soccer "
            "ball near the couch, avoiding the path between the TV and the tea "
            "table.")
        assert [c.kind for c in steps(plan)] == [GOTO, GOTO]
        assert [c.kind for c in keepouts(plan)] == [AVOID]

    def test_a_keepout_with_one_destination_still_lifts_out(self):
        """The phrasing that broke hoisting it into the sequence."""
        plan = parse_instruction(
            "Go to the cup near the TV remote and avoid the path near the "
            "cabinet.")
        assert [c.kind for c in steps(plan)] == [GOTO]
        assert len(keepouts(plan)) == 1

    def test_passage_stays_in_order(self):
        plan = parse_instruction(
            "First, go near the bedside table closest to the bench, then take "
            "the path between the TV and the bed to the picture closest to the "
            "TV.")
        assert [c.kind for c in steps(plan)] == [GOTO, PASS, GOTO]
        assert keepouts(plan) == []

    @pytest.mark.parametrize("kind", [GOTO, PASS, AVOID])
    def test_every_clause_lands_in_exactly_one_bucket(self, kind):
        plan = [Clause(kind, "x")]
        assert len(steps(plan)) + len(keepouts(plan)) == 1


class TestReplyParsing:
    """`vlm_probe.parse` — the greedy regex that lost a whole passage leg."""

    def test_the_studio_split_reply(self):
        """The approach block says "appended to the JSON above"; the model
        read that as a second object, and the greedy span covered both."""
        raw = ('{"visible": true, "gate": [{"name": "couch"}], "here": null}\n'
               '{"here": "attic living room", "target_state": "far"}')
        r = parse(raw)
        assert r["visible"] is True
        assert [g["name"] for g in r["gate"]] == ["couch"]
        assert r["here"] == "attic living room"
        assert r["target_state"] == "far"

    def test_a_later_object_cannot_overwrite_the_answer(self):
        assert parse('{"a": 1}{"a": 9}')["a"] == 1

    def test_a_later_object_fills_a_null(self):
        assert parse('{"a": null}{"a": 9}')["a"] == 9

    def test_fenced(self):
        assert parse('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_with_a_stray_brace_first(self):
        assert parse('note {not json} then {"a": 1}') == {"a": 1}

    def test_no_json_at_all(self):
        assert parse("sorry, I cannot") is None
