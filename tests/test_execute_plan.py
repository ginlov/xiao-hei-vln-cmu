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

from approach_loop import (JUMP_M, MAX_LOOPS, MIN_EXPLORE_M,  # noqa: E402
                           REVISIT_M,
                           SPENT_CONE_DEG, SPENT_PENALTY, already_tried,
                           bind_target, corroborated, explore_direction,
                           gates_from, GATE_PAD_M, lift_way, WAY_MAX_M,
                           nearest_allowed_step, recrosses, revisited,
                           same_thing, side_of)
from execute_plan import (THROUGH_M, far_side_goal,  # noqa: E402
                          far_side_stalled, gate_point,
                          through_point, went_between, xy_of)
from vlm_probe import build_prompt, parse  # noqa: E402
from instruction_plan import (AVOID, GOTO, PASS, Clause,  # noqa: E402
                              keepouts, parse_instruction, steps)


def anchor(name, x, y):
    return {"name": name, "xy": np.array([float(x), float(y)])}


def fake_cm(points=None, *, gates=(), keepout=()):
    """A `ConverterModel` over a handful of legal points, without a terrain map.

    The constructor wants a (N, 4) cloud and derives legality from obstacle
    inflation; these tests are about the keep-out, so the legal set is set
    directly and everything upstream of it is left out.
    """
    from scipy.spatial import cKDTree
    from waypoint_converter_model import ConverterModel
    cm = ConverterModel.__new__(ConverterModel)
    P = np.zeros((0, 2)) if points is None else np.asarray(points, float)
    # The real `legal` is (N, 3) — `snap` queries it in 3-D as the C++ does —
    # while `legal_points()` is the planar view of it.
    cm.legal = cm.allowed = np.column_stack([P, np.zeros(len(P))])
    cm.kd_legal = cKDTree(cm.legal) if len(P) else None
    cm.keepout = [(np.asarray(xy, float)[:2], float(r)) for xy, r in keepout]
    cm.gates = [(np.asarray(a, float)[:2], np.asarray(b, float)[:2])
                for a, b in gates]
    return cm


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

    def test_a_frozen_entry_outranks_the_live_pose(self):
        """Past the line, "beyond" must not become "back where I came from"."""
        cm = self.FakeCM([[-2.0, 2.0], [2.0, 2.0]])
        past = np.array([0.4, 2.0])
        entry = np.array([-3.0, 2.0])
        assert np.allclose(
            far_side_goal(cm, self.SIDES, past, np.array([1.2, 2.0]))[0],
            [-2.0, 2.0]), "the live pose sends it back"
        assert np.allclose(
            far_side_goal(cm, self.SIDES, past, np.array([1.2, 2.0]),
                          entry=entry)[0], [2.0, 2.0])

    def test_settle_still_reads_the_live_pose(self):
        """Two positions with two jobs: `entry` picks the half-plane, the
        vehicle's own pose is what `settle` needs to answer where it lands."""
        seen = {}

        class Recording(self.FakeCM):
            def settle(self, waypoint, vehicle):
                seen["vehicle"] = np.asarray(vehicle, float)
                return np.asarray(waypoint, float)

        cm = Recording([[2.0, 2.0]])
        far_side_goal(cm, self.SIDES, np.array([-1.0, 2.0]),
                      np.array([1.2, 2.0]), entry=np.array([-3.0, 2.0]))
        assert np.allclose(seen["vehicle"], [-1.0, 2.0])


class TestFarSideStalled:
    """When a far-side waypoint stops being news.

    The `living_room_1` numbers throughout: the converter answered every step
    with the same corner about 2 m north-west of the gate, and the leg took it
    as progress each time.
    """

    HERE = np.array([-1.35, 0.12])

    def test_a_real_new_point_is_progress(self):
        far = (np.array([-1.83, 0.13]), np.array([-0.20, -1.60]))
        assert far_side_stalled(far, self.HERE, []) is None

    def test_no_point_at_all(self):
        assert far_side_stalled(None, self.HERE, []) == \
            "no legal point beyond the gap yet"

    def test_a_waypoint_the_vehicle_is_already_standing_on(self):
        """The measured step 5: 0.20 m of motion on offer."""
        far = (np.array([-1.83, 0.13]), np.array([-1.55, 0.12]))
        assert far_side_stalled(far, self.HERE, []) == \
            "the best far-side waypoint is where it stands"

    def test_a_waypoint_this_leg_has_already_driven_to(self):
        """The measured step 6: a 1.45 m move, back to step 4's resting pose.

        Distance alone calls this progress -- which is exactly how the leg
        talked itself into shuttling. Only the history refuses it.
        """
        here = np.array([-0.15, -0.69])          # where step 6 actually stood
        far = (np.array([-1.73, 0.02]), np.array([-1.47, -0.10]))
        assert np.linalg.norm(far[1] - here) > 1.4, "distance calls it progress"
        assert far_side_stalled(far, here, []) is None
        assert far_side_stalled(far, here, [np.array([-1.38, 0.14])]) == \
            "the far-side waypoint is one already driven to"

    def test_a_different_far_side_point_still_counts(self):
        """Refusing repeats must not refuse a genuinely new approach."""
        far = (np.array([1.90, -3.10]), np.array([1.60, -2.80]))
        assert far_side_stalled(far, self.HERE,
                                [np.array([-1.38, 0.14])]) is None


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

    def test_frozen_entry_keeps_forward_forward(self):
        """The oscillation this exists to stop.

        The vehicle has crossed the gap and `went_between` did not fire -- the
        planner rounded the end of the segment, or an anchor lifted half a
        metre out. Passing the live pose flips the aim back the way it came,
        and the leg drives through the gap and out again until its steps run
        out. Passing the pose the gap was bound at does not.
        """
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        entry = np.array([-3.0, 2.0])
        past = np.array([0.4, 2.0])                  # just over the line
        assert through_point(a, b, past)[0] < 0, "the live pose flips it"
        assert through_point(a, b, entry)[0] > 0, "the frozen one does not"


class TestRecrosses:
    """Bearings that would undo a passage the robot has already driven."""

    # The `home_building_1` gap, from the run logs: the dining table north of
    # the corridor and the picture on the wall south of it. The robot entered
    # from the east and came out west.
    TABLE = np.array([-1.68, -4.17])
    PICTURE = np.array([-0.32, -6.57])
    ENTRY = np.array([2.51, -4.43])
    OUT = np.array([-1.22, -5.48])
    CROSSED = [(TABLE, PICTURE, ENTRY)]

    def east(self):
        return np.array([1.0, 0.0])

    def west(self):
        return np.array([-1.0, 0.0])

    def test_going_back_east_is_caught(self):
        assert recrosses(self.OUT, self.east(), self.CROSSED, 3.0)

    def test_carrying_on_west_is_not(self):
        assert not recrosses(self.OUT, self.west(), self.CROSSED, 3.0)

    def test_it_only_bites_within_reach(self):
        """A bearing the robot cannot get far enough along is not a reversal.

        The margin here is thin and that is the measurement, not a rounding:
        the robot came out only 0.28 m past the line, so half a metre back east
        already re-crosses it. That is why the exit pose alone cannot carry
        this -- one step of any size leaves the neighbourhood `already_tried`
        recognises, while the gap stays exactly where it was.
        """
        assert recrosses(self.OUT, self.east(), self.CROSSED, 0.5)
        assert not recrosses(self.OUT, self.east(), self.CROSSED, 0.2)

    def test_driving_the_passage_the_right_way_is_never_penalised(self):
        """From the entry side, crossing is the constraint, not a reversal."""
        assert not recrosses(self.ENTRY, self.west(), self.CROSSED, 6.0)

    def test_nothing_crossed_yet_penalises_nothing(self):
        assert not recrosses(self.OUT, self.east(), [], 3.0)

    def test_a_bearing_that_misses_the_gap_is_not_a_crossing(self):
        """Past the end of the segment is not through it.

        Synthetic, because the real gap is diagonal and almost every bearing
        from the exit pose meets it somewhere. A gap along x=0 spanning y=0..4,
        entered from the west and left to the east; heading back west level
        with it is a reversal, and heading back west 9 m north of it is not --
        that is the line the anchors lie on, not the passage between them.
        """
        crossed = [(np.array([0.0, 0.0]), np.array([0.0, 4.0]),
                    np.array([-3.0, 2.0]))]
        assert recrosses(np.array([1.0, 2.0]), self.west(), crossed, 5.0)
        assert not recrosses(np.array([1.0, 9.0]), self.west(), crossed, 5.0)


class TestSideOf:
    def test_it_signs_the_two_halves_apart(self):
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        assert side_of(a, b, np.array([-1.0, 2.0])) != \
            side_of(a, b, np.array([1.0, 2.0]))

    def test_a_point_on_the_line_is_neither(self):
        a, b = np.array([0.0, 0.0]), np.array([0.0, 4.0])
        assert side_of(a, b, np.array([0.0, 2.0])) == 0.0


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


class TestAlreadyTried:
    """Departure memory: the model cannot hold it, so the loop does.

    Positions are the eight poses of `home_building_1` q5 leg 1, which searched
    one hall for 183 s of a 400 s slice and left it by the same two doors.
    """

    def test_the_same_door_from_the_same_spot(self):
        spent = [(np.array([0.0, 0.0]), np.array([-0.71, -0.70]))]
        assert already_tried(np.array([0.05, -0.02]),
                             np.array([-0.70, -0.71]), spent)

    def test_the_same_heading_from_another_room_is_a_different_move(self):
        spent = [(np.array([0.0, 0.0]), np.array([-0.71, -0.70]))]
        assert not already_tried(np.array([-5.13, 0.16]),
                                 np.array([-0.71, -0.70]), spent)

    def test_another_door_from_the_same_spot_is_the_whole_point(self):
        spent = [(np.array([0.0, 0.0]), np.array([-0.71, -0.70]))]
        assert not already_tried(np.array([0.05, -0.02]),
                                 np.array([0.0, 1.0]), spent)

    def test_nothing_spent_yet(self):
        assert not already_tried(np.array([0.0, 0.0]),
                                 np.array([1.0, 0.0]), [])

    def test_the_cone_is_wider_than_the_face_quantisation(self):
        """The model's heading comes off 90° faces; the cone must survive that.

        Otherwise the same corridor, nominated as 180° on one call and 135° on
        the next, reads as two different corridors and nothing is ever spent.
        """
        assert SPENT_CONE_DEG >= 22.5

    def test_a_repeat_is_discounted_not_forbidden(self):
        """A room with one exit must keep that exit reachable."""
        assert 0.0 < SPENT_PENALTY < 1.0


class FlatTerrain:
    """A converter model stand-in: everything drivable, nothing preferred."""

    def reach_along(self, origin, u):
        return 5.0


class Doorway:
    """A short opening straight ahead, a long corridor 30° off it.

    The shape that made leg 1 unable to enter a room: the door the model wants
    reaches 2 m, the hallway beside it reaches 6.
    """

    def reach_along(self, origin, u):
        ang = np.degrees(np.arctan2(float(u[1]), float(u[0])))
        if abs(ang) < 7.5:
            return 2.0
        if abs(ang - 30.0) < 7.5:
            return 6.0
        return 0.3


class TestExploreDirectionMemory:
    def test_the_door_beats_the_corridor(self):
        """The whole fix, in one assertion.

        `reach · cos(Δ)` scores the corridor 6·cos30 = 5.20 against the door's
        2·cos0 = 2.00 and takes the corridor. Reach as a gate takes the door.
        """
        u, reach, delta = explore_direction(Doorway(), np.array([0.0, 0.0]),
                                            0.0, [])
        assert delta == 0.0
        assert reach == 2.0
        assert u[0] == pytest.approx(1.0)

    def test_a_bearing_that_goes_nowhere_is_still_refused(self):
        """loft drove into a wall three times; that gate has to survive.

        Its numbers: the heading asked for reached 0.00 m and 30° off it
        reached 5.16. Preferring the asked-for bearing must not mean taking one
        the vehicle cannot move along at all.
        """

        class Loft:
            def reach_along(self, origin, u):
                ang = np.degrees(np.arctan2(float(u[1]), float(u[0])))
                return 5.16 if abs(ang - 30.0) < 7.5 else 0.0

        u, reach, delta = explore_direction(Loft(), np.array([0.0, 0.0]),
                                            0.0, [])
        assert reach == pytest.approx(5.16)
        assert reach >= MIN_EXPLORE_M
        assert delta == 30.0

    def test_an_untried_direction_beats_a_spent_one(self):
        """Equal reach both ways, so only the history can break the tie."""
        want = 0.0
        spent = [(np.array([0.0, 0.0]), np.array([1.0, 0.0]))]
        u, _, delta = explore_direction(FlatTerrain(), np.array([0.0, 0.0]),
                                        want, spent)
        assert abs(delta) > 0.0
        assert not already_tried(np.array([0.0, 0.0]), u, spent)

    def test_a_passage_already_driven_is_not_the_way_onward(self):
        """The `home_building_1` regression, in the terms the loop sees it.

        The robot has just come west through the dining-table gap and the next
        destination is not in sight. The model asks for due east -- back the
        way it came, which in three runs of four is what it got, the first
        exploration step of the leg driving 1.7 m back through the gap. Open
        floor in every direction, so nothing but the crossed passage can break
        the tie.
        """
        crossed = [(TestRecrosses.TABLE, TestRecrosses.PICTURE,
                    TestRecrosses.ENTRY)]
        out = TestRecrosses.OUT
        assert explore_direction(FlatTerrain(), out, 0.0, [])[2] == 0.0
        u, _, delta = explore_direction(FlatTerrain(), out, 0.0, [], crossed)
        assert abs(delta) > 0.0, "due east still chosen"
        assert not recrosses(out, u, crossed, 5.0)

    def test_it_may_still_go_back_if_that_is_the_only_way(self):
        """Discounted, never forbidden -- the room with one door again."""

        class OneCorridorEast:
            def reach_along(self, origin, u):
                return 5.0 if float(u[0]) > 0.9 else 0.0

        crossed = [(TestRecrosses.TABLE, TestRecrosses.PICTURE,
                    TestRecrosses.ENTRY)]
        u, reach, _ = explore_direction(OneCorridorEast(), TestRecrosses.OUT,
                                        0.0, [], crossed)
        assert reach == 5.0
        assert u[0] == pytest.approx(1.0)

    def test_with_no_history_it_takes_what_was_asked_for(self):
        u, reach, delta = explore_direction(FlatTerrain(),
                                            np.array([0.0, 0.0]), 0.0, [])
        assert delta == 0.0
        assert reach == 5.0
        assert u[0] == pytest.approx(1.0)

    def test_the_only_way_out_stays_reachable(self):
        """One drivable bearing, already taken: it must still be chosen."""

        class OneCorridor:
            def reach_along(self, origin, u):
                return 5.0 if float(u[0]) > 0.9 else 0.0

        spent = [(np.array([0.0, 0.0]), np.array([1.0, 0.0]))]
        u, reach, _ = explore_direction(OneCorridor(), np.array([0.0, 0.0]),
                                        0.0, spent)
        assert reach == 5.0
        assert u[0] == pytest.approx(1.0)

    def test_hemmed_in_on_all_sides_it_still_moves(self):
        """Nothing clears the gate, so the old rule picks the least bad."""

        class Boxed:
            def reach_along(self, origin, u):
                return 0.4 if float(u[0]) > 0.5 else 0.1

        u, reach, _ = explore_direction(Boxed(), np.array([0.0, 0.0]), 0.0, [])
        assert reach == pytest.approx(0.4)
        assert u[0] > 0.5

    def test_reach_no_longer_decides_between_viable_bearings(self):
        """Two ways out, both fine, the nearer one wins however short it is."""

        class TwoWays:
            def reach_along(self, origin, u):
                ang = np.degrees(np.arctan2(float(u[1]), float(u[0])))
                if abs(ang - 15.0) < 7.5:
                    return 0.6
                if abs(ang - 90.0) < 7.5:
                    return 40.0
                return 0.0

        _, reach, delta = explore_direction(TwoWays(), np.array([0.0, 0.0]),
                                            0.0, [])
        assert delta == 15.0 and reach == pytest.approx(0.6)


class TestPromptV6:
    """The way-out field, and that adding it left the older versions alone."""

    def test_v5_is_byte_for_byte_unchanged(self):
        """117 cached replies and every offline script are keyed to v5."""
        from vlm_probe import PROMPT_V3, PROMPT_V4, PROMPT_V5, PROMPTS
        assert PROMPTS["v5-constraints"] is PROMPT_V5
        assert PROMPTS["v4-relational"] is PROMPT_V4
        assert PROMPTS["v3-occlusion-distance"] is PROMPT_V3
        assert "POINT AT THE WAY OUT" not in PROMPT_V5

    def test_v6_asks_for_a_box_not_just_a_bearing(self):
        from vlm_probe import PROMPT_V6
        assert '"way": {{' in PROMPT_V6, "the schema must declare the field"
        schema = PROMPT_V6.split('"way": {{')[1][:200]
        assert "box_2d" in schema and "image_index" in schema

    def test_v6_keeps_everything_v5_asked_for(self):
        """Surgery, not a rewrite: the constraint and relational fields stay."""
        from vlm_probe import PROMPT_V6
        for field in ('"gate": [', '"avoid": [', '"candidates"', '"anchors"',
                      '"explore"'):
            assert field in PROMPT_V6, field

    def test_v6_is_the_default_and_v5_is_still_selectable(self):
        from vlm_probe import DEFAULT_PROMPT_VER, PROMPTS
        assert DEFAULT_PROMPT_VER == "v6-way-out"
        assert "v5-constraints" in PROMPTS

    def test_it_says_to_box_the_opening_not_the_door(self):
        """A door leaf is a flat panel; the scanner would return the panel."""
        from vlm_probe import PROMPT_V6
        assert "not the door leaf" in PROMPT_V6

    def test_it_allows_saying_there_is_no_way_out(self):
        """Otherwise the model boxes a wall to satisfy the schema."""
        from vlm_probe import PROMPT_V6
        assert "null" in PROMPT_V6.split('"way": {{')[1][:300]
        assert "leave \"way\" null" in PROMPT_V6


class TestDoneBlock:
    """What the executor tells the model about constraints already banked."""

    MISSION = {"question": "First, go to the nightstand with a clock on it, "
                           "then take the path between the dining table and "
                           "the picture, and stop at the trash can closest to "
                           "the refridgerator.",
               "plan": ["GOTO  the nightstand with a clock on it",
                        "PASS  between the dining table + the picture",
                        "GOTO  the trash can closest to the refridgerator"],
               "k": 3}

    def test_nothing_banked_renders_nothing(self):
        """The condition every cached reply depends on.

        The block is appended to whichever version is selected, so a prompt
        that renders it unconditionally would move v5's bytes -- and the 117
        cached replies are keyed to them.
        """
        assert build_prompt("x", mission=self.MISSION) == \
            build_prompt("x", mission={**self.MISSION, "done": []})

    def test_a_banked_passage_reaches_the_model(self):
        got = build_prompt("x", mission={
            **self.MISSION,
            "done": ["drove the passage 'between the dining table and the "
                     "picture' — between 'dining table' and 'picture'"]})
        assert "Already driven" in got
        assert "dining table" in got.split("Already driven")[1]

    def test_it_says_a_spent_constraint_is_not_the_way_onward(self):
        """The rule itself, which is the whole point of showing the list."""
        got = build_prompt("x", mission={**self.MISSION, "done": ["a passage"]})
        assert "not back the way it came" in got

    def test_it_leaves_a_way_back_open(self):
        """Discouraged, not forbidden -- the geometry says the same thing."""
        got = " ".join(build_prompt(
            "x", mission={**self.MISSION, "done": ["a passage"]}).split())
        assert "last resort" in got
        assert "Send it back through one only if" in got

    def test_it_survives_having_no_mission_at_all(self):
        """Single-object runs pass no mission, and must not grow a block."""
        assert "Already driven" not in build_prompt("x")


class TestLiftWay:
    """`lift_way` reads the field; the lift itself is `_lift_xy`'s test."""

    def test_no_way_field_is_no_way(self):
        assert lift_way({}, np.zeros((0, 3)), {}) is None
        assert lift_way({"way": None}, np.zeros((0, 3)), {}) is None

    def test_a_partial_box_is_refused(self):
        """Half an answer would lift to a wrong place rather than to nothing."""
        assert lift_way({"way": {"box_2d": [0, 0, 10, 10]}},
                        np.zeros((0, 3)), {}) is None
        assert lift_way({"way": {"image_index": 0}},
                        np.zeros((0, 3)), {}) is None

    def test_a_string_where_an_object_belongs(self):
        """The model sometimes answers a field with prose; that is not a box."""
        assert lift_way({"way": "the doorway on the left"},
                        np.zeros((0, 3)), {}) is None


class TestWayRange:
    """The lift's bearing is the model's; its range is the scanner's guess.

    Recorded from `runs/hm1_q2_v6`, where four of seven lifts came back beyond
    9 m because the ray passed through the opening and stopped in the room
    after next.
    """

    @staticmethod
    def clamp(here, seen):
        """What `lift_way` does to a lifted point, without the lift."""
        here, seen = np.asarray(here, float), np.asarray(seen, float)
        v = seen - here
        d = float(np.linalg.norm(v))
        return here + v / d * min(d, WAY_MAX_M)

    def test_the_first_step_no_longer_aims_outside_the_house(self):
        """13.69 m to (+8.57, -10.68) — beyond anything the GT path visits."""
        got = self.clamp([0.0, 0.0], [8.57, -10.68])
        assert np.linalg.norm(got) == pytest.approx(WAY_MAX_M)

    def test_the_bearing_survives_the_clamp(self):
        """Only the range is doubted, so the direction must be untouched."""
        here, seen = np.array([0.0, 0.0]), np.array([8.57, -10.68])
        got = self.clamp(here, seen)
        cos = float(np.dot(got, seen) / (np.linalg.norm(got)
                                         * np.linalg.norm(seen)))
        assert cos == pytest.approx(1.0)

    def test_the_lift_that_worked_is_left_alone(self):
        """4.47 m, and it is the one that drove the robot into the bedroom."""
        here, seen = np.array([3.00, -3.18]), np.array([5.08, -7.13])
        assert np.allclose(self.clamp(here, seen), seen)

    def test_the_cap_clears_a_room(self):
        """Under a room's width it would stop short of doorways worth taking."""
        assert WAY_MAX_M >= 5.0

    def test_lift_way_itself_clamps(self, monkeypatch):
        """The shipped path, not just the arithmetic beside it."""
        import approach_loop as al

        monkeypatch.setattr(al, "_lift_xy",
                            lambda *a, **k: np.array([8.57, -10.68, 0.0]))
        monkeypatch.setattr(al, "scan_to_camera", lambda scan, pose: scan)
        reply = {"way": {"box_2d": [0, 0, 10, 10], "image_index": 0}}
        got = al.lift_way(reply, np.zeros((0, 3)),
                          {"position": [0.0, 0.0, 0.0]})
        assert got is not None
        assert np.linalg.norm(got) == pytest.approx(WAY_MAX_M)

    def test_lift_way_passes_a_near_opening_through(self, monkeypatch):
        import approach_loop as al

        monkeypatch.setattr(al, "_lift_xy",
                            lambda *a, **k: np.array([5.08, -7.13, 0.0]))
        monkeypatch.setattr(al, "scan_to_camera", lambda scan, pose: scan)
        reply = {"way": {"box_2d": [0, 0, 10, 10], "image_index": 0}}
        got = al.lift_way(reply, np.zeros((0, 3)),
                          {"position": [3.00, -3.18, 0.0]})
        assert np.allclose(got, [5.08, -7.13])

    def test_lift_way_survives_a_lift_onto_the_vehicle(self, monkeypatch):
        """A zero-length vector would divide by zero on the way to the cap."""
        import approach_loop as al

        monkeypatch.setattr(al, "_lift_xy",
                            lambda *a, **k: np.array([1.0, 2.0, 0.0]))
        monkeypatch.setattr(al, "scan_to_camera", lambda scan, pose: scan)
        reply = {"way": {"box_2d": [0, 0, 10, 10], "image_index": 0}}
        assert al.lift_way(reply, np.zeros((0, 3)),
                           {"position": [1.0, 2.0, 0.0]}) is None


class TestGate:
    """A forbidden corridor, from `livingroom_2` q5.

    "avoiding the path between the TV and the tea table". The robot drove
    through the middle of it, 0.14 m from the midpoint, and every number here
    is from that run.
    """

    TV = np.array([2.41, -2.90])        # as lifted, not ground truth
    TEA = np.array([0.41, -2.31])
    VEH = np.array([0.45, -1.31])       # where it stood at the violating step
    WP = np.array([2.62, -6.23])        # what it published, straight through

    def anchors(self, *names):
        return [{"xy": xy, "name": nm} for nm, xy in names]

    def test_it_pairs_the_two_anchors(self):
        g = gates_from(self.anchors(("TV", self.TV), ("tea table", self.TEA)))
        assert len(g) == 1
        a, b = g[0]
        assert np.linalg.norm(a - b) > np.linalg.norm(self.TV - self.TEA)

    def test_both_ends_are_padded(self):
        """A lift lands on the face the scanner saw, so the segment joining two
        anchors is short of the furniture at both ends. Unpadded, a route that
        'clears' this gate misses the tea table's centre by 0.03 m."""
        (a, b), = gates_from(self.anchors(("TV", self.TV), ("tea table", self.TEA)))
        grew = np.linalg.norm(a - b) - np.linalg.norm(self.TV - self.TEA)
        assert grew == pytest.approx(2 * GATE_PAD_M, abs=1e-6)

    def test_one_anchor_is_not_a_gate(self):
        assert gates_from(self.anchors(("TV", self.TV))) == []

    def test_the_same_object_twice_is_not_a_gate(self):
        """Three of the five discs in that run were the same television."""
        assert gates_from(self.anchors(("TV", self.TV),
                                       ("tv (right view)", self.TV + 1.4))) == []

    def test_the_violating_route_is_refused(self):
        cm = fake_cm(gates=gates_from(
            self.anchors(("TV", self.TV), ("tea table", self.TEA))))
        assert cm.crosses_gate(self.VEH, self.WP)

    def test_a_route_that_stays_on_one_side_is_allowed(self):
        cm = fake_cm(gates=gates_from(
            self.anchors(("TV", self.TV), ("tea table", self.TEA))))
        assert not cm.crosses_gate(self.VEH, np.array([0.07, -1.54]))

    def test_a_gate_removes_nothing_from_the_legal_set(self):
        """It forbids crossing, not standing. Two discs wide enough to close a
        2 m gap closed every route the leg had; this is why."""
        pts = np.array([[0.0, 0.0], [1.0, -2.9], [2.0, -5.0]])
        assert len(fake_cm(pts, gates=[(self.TEA, self.TV)]).legal_points()) \
            == len(pts)


class TestSearchWindowWithAConstraint:
    """`search` ranks candidates by distance to the target and keeps the
    nearest. That is a pure optimisation until a keep-out exists — and then the
    candidates it keeps are exactly the ones the keep-out rejects, because the
    target lies beyond the thing being avoided.

    `livingroom_2` q5 reported `boxed in (no legal move)` from a frame with 721
    legal moves available, and the leg before it fell through to publishing its
    raw waypoint with the constraint dropped. One cause, two symptoms.
    """

    @staticmethod
    def room():
        """600 legal points beyond the gate, 300 on this side of it."""
        r = np.random.RandomState
        far = np.column_stack([r(0).uniform(-3, 3, 600),
                               r(1).uniform(-8, -4, 600)])
        near = np.column_stack([r(2).uniform(-3, 3, 300),
                                r(3).uniform(1, 5, 300)])
        return np.vstack([far, near])

    GATE = [(np.array([-6.0, 0.0]), np.array([6.0, 0.0]))]
    VEH = np.array([0.0, 2.0])
    AIM = np.array([0.0, -7.0])          # beyond the gate

    def test_every_nearest_candidate_is_refused_and_it_still_answers(self):
        cm = fake_cm(self.room(), gates=self.GATE)
        got = cm.best_waypoint_toward(self.AIM, self.VEH, search=400,
                                      min_move=0.5)
        assert got is not None, "721 legal moves existed and it said there were none"
        assert got[0][1] > 0.0, "and the one it picks must not be over the gate"

    def test_the_first_reachable_candidate_is_outside_the_window(self):
        """The premise: without the pre-filter, rank 600 of 900 is unreachable
        by a 400-wide window, so this is not a test that would pass anyway."""
        P = self.room()
        d = np.linalg.norm(P - self.AIM, axis=1)
        first = int(np.argsort(d)[np.searchsorted(
            np.sort(d), d[P[:, 1] > 0].min())])
        assert int((d < d[first]).sum()) > 400

    def test_an_unconstrained_frame_is_unchanged(self):
        """The window still does its job when there is nothing to avoid."""
        cm = fake_cm(self.room())
        got = cm.best_waypoint_toward(self.AIM, self.VEH, search=400,
                                      min_move=0.5)
        assert got is not None and got[0][1] < 0.0

    def test_a_truly_sealed_frame_still_says_so(self):
        cm = fake_cm(self.room()[:600], gates=self.GATE)     # far side only
        assert cm.best_waypoint_toward(self.AIM, self.VEH, min_move=0.5) is None


class TestNearestAllowedStep:
    """What to do when every waypoint toward the aim is forbidden.

    The old answer was to publish the raw waypoint with the constraint dropped
    in silence, which is how the violation happened.
    """

    GATE = [(np.array([-2.0, -1.0]), np.array([2.0, -1.0]))]

    def test_it_takes_the_nearest_point_it_may_reach(self):
        pts = np.array([[0.0, -3.0], [0.5, 0.5], [1.5, 0.4]])   # first is over
        cm = fake_cm(pts, gates=self.GATE)
        got = nearest_allowed_step(cm, np.array([0.0, 1.0]),
                                   np.array([0.0, -4.0]))
        assert got is not None and got[1] > -1.0, "it must not cross the gate"

    def test_none_when_everything_is_forbidden(self):
        cm = fake_cm(np.array([[0.0, -3.0], [1.0, -4.0]]), gates=self.GATE)
        assert nearest_allowed_step(cm, np.array([0.0, 1.0]),
                                    np.array([0.0, -4.0])) is None

    def test_no_legal_points_at_all(self):
        assert nearest_allowed_step(fake_cm(np.zeros((0, 2))),
                                    np.zeros(2), np.zeros(2)) is None


class TestKeepOutAnchorMerge:
    """One television, lifted three times, became three keep-out discs."""

    def lifted(self, name, xy):
        return {"avoid": [{"name": name, "box_2d": [0, 0, 1, 1],
                           "image_index": 0}]}, np.asarray(xy, float)

    def test_the_same_name_is_the_same_anchor(self):
        assert same_thing("TV", "tv")
        assert same_thing("TV", "TV (same set seen at right edge)")
        assert same_thing("tea table", "tea table (coffee table)")

    def test_different_names_stay_apart(self):
        assert not same_thing("TV", "tea table")


class TestLoopBudget:
    def test_a_leg_may_come_back_before_it_gives_up(self):
        """The fix in one line: one return is no longer fatal.

        Leg 1 stopped on its first return with 217 s of its slice unspent, and
        the corridor it wanted was found five steps later by another clause.
        """
        assert MAX_LOOPS > 1


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
