"""Integration-style tests for the Phase A perception responder.

Drives synthetic ``VLMInput`` snapshots through the responder and asserts
that emitted waypoints behave the way the algorithm description promises:
* with no inputs yet, no output;
* with terrain + pose, a waypoint moving toward an unexplored direction;
* after the room is fully explored, fall back to "stand still";
* a new question wipes the cross-tick state.
"""

from __future__ import annotations

import math

import numpy as np

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.sensors import TerrainMap
from xiao_hei_vln.perception.global_map import FREE, GlobalMap
from xiao_hei_vln.perception_responder import PerceptionResponder


def _stamp() -> Stamp:
    return Stamp.from_seconds(0.0)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _terrain(points: list[tuple[float, float, float, float]]) -> TerrainMap:
    arr = np.array(points, dtype=np.float32) if points else np.zeros((0, 4), dtype=np.float32)
    return TerrainMap(
        header=Header(stamp=_stamp(), frame_id="map"),
        points=arr,
        range="ext_20m",
    )


def _input(
    *,
    tick_id: int = 0,
    question: str | None = "How many cups",
    pose: OdomPose | None,
    terrain: TerrainMap | None,
) -> VLMInput:
    q = ChallengeQuestion.from_text(question, _stamp()) if question is not None else None
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(),
        question=q,
        pose=pose,
        terrain_ext=terrain,
    )


class TestColdStart:
    def test_no_output_without_question(self) -> None:
        r = PerceptionResponder()
        out = r.respond(_input(question=None, pose=_pose(0.0, 0.0), terrain=_terrain([])))
        assert out is None
        assert r.is_done() is False

    def test_no_output_without_terrain(self) -> None:
        r = PerceptionResponder()
        out = r.respond(_input(pose=_pose(0.0, 0.0), terrain=None))
        assert out is None

    def test_no_output_without_pose(self) -> None:
        r = PerceptionResponder()
        out = r.respond(_input(pose=None, terrain=_terrain([])))
        assert out is None


class TestExploration:
    def test_emits_waypoint_when_frontier_exists(self) -> None:
        r = PerceptionResponder(min_cluster_size=1)
        # Drop a small patch of FREE terrain in front of the robot so a
        # frontier exists on at least one side.
        points = [
            (1.0, 0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (2.5, 0.0, 0.0, 0.0),
        ]
        out = r.respond(_input(pose=_pose(0.0, 0.0), terrain=_terrain(points)))
        assert isinstance(out, WaypointPathResponse)
        wp = out.waypoints[0]
        # Frontier should be somewhere along the strip we observed.
        assert 0.0 < wp.x < 5.0
        # Heading should roughly point forward (atan2(0, +x) ≈ 0).
        assert abs(wp.heading) < math.pi / 2

    def test_falls_back_to_stand_still_after_exhaustion(self) -> None:
        # Force-build a GlobalMap that has NO frontier (everything FREE inside,
        # surrounded by OCCUPIED), then plug it into the responder.
        gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
        gm.update(np.zeros((0, 4), dtype=np.float32), _pose(0.0, 0.0))
        from xiao_hei_vln.perception.global_map import OCCUPIED
        gm.grid[:] = OCCUPIED
        gm.grid[4:8, 4:8] = FREE  # FREE block fully enclosed by OCCUPIED

        r = PerceptionResponder(global_map=gm, min_cluster_size=1)
        out = r.respond(
            _input(pose=_pose(0.0, -0.25), terrain=_terrain([])),
        )
        assert isinstance(out, WaypointPathResponse)
        wp = out.waypoints[0]
        # Stand-still emits the robot's own pose.
        assert wp.x == 0.0
        assert wp.y == -0.25
        assert out.rationale and "phase_a_exhausted" in out.rationale


class TestReset:
    def test_reset_clears_state(self) -> None:
        r = PerceptionResponder(min_cluster_size=1)
        points = [(1.0, 0.0, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0)]
        r.respond(_input(pose=_pose(0.0, 0.0), terrain=_terrain(points)))
        assert len(r._planner.history) >= 1  # noqa: SLF001 (testing internal state)

        r.reset()
        assert r._planner.history == ()  # noqa: SLF001
        # The global map should be reinitialised so a far-away pose works again.
        r.respond(_input(pose=_pose(1000.0, -2500.0), terrain=_terrain([])))
        # Origin should now be anchored on the new pose.
        assert r._map.origin_xy is not None  # noqa: SLF001
        ox, _ = r._map.origin_xy  # noqa: SLF001
        assert abs(ox - (1000.0 - 30.0)) < 1.0
