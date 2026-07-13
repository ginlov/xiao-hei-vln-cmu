"""Tests for ``xiao_hei_vln.perception.PerceptionResponder``."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    ImageFrame,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
)
from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.question import QuestionType
from xiao_hei_vln.messages.sensors import LidarScan
from xiao_hei_vln.perception.client import Detection
from xiao_hei_vln.perception.responder import PerceptionResponder
from xiao_hei_vln.perception.vocab import Vocabulary
from xiao_hei_vln.scene import ObjectObservation, SceneRepresentation

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _stamp() -> Stamp:
    return Stamp(sec=0, nanosec=0)


def _header(frame: str = "map") -> Header:
    return Header(stamp=_stamp(), frame_id=frame)


def _pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> OdomPose:
    return OdomPose(
        header=_header(),
        position=Vector3(x=x, y=y, z=0.0),
        orientation=Quaternion(
            x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2),
        ),
    )


def _image(width: int = 1920, height: int = 640) -> ImageFrame:
    step = width * 3
    data = b"\x80" * (step * height)        # mid-grey BGR8
    return ImageFrame(
        header=_header("camera"),
        width=width, height=height, encoding="bgr8",
        step=step, data=data,
    )


def _scan(points: np.ndarray) -> LidarScan:
    if points.ndim == 1:
        points = points.reshape(0, 4)
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    return LidarScan(header=_header(), points=points.astype(np.float32), source="registered")


def _snapshot(
    tick_id: int = 0,
    *,
    pose: OdomPose | None = None,
    image: ImageFrame | None = None,
    scan: LidarScan | None = None,
    question_text: str | None = None,
    qtype: QuestionType | None = None,
) -> VLMInput:
    question = None
    if question_text is not None:
        question = ChallengeQuestion(
            text=question_text,
            type=qtype if qtype is not None else QuestionType.NUMERICAL,
            received_at=_stamp(),
        )
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(),
        pose=pose,
        image=image,
        registered_scan=scan,
        question=question,
    )


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


@dataclass
class _FakeClient:
    """Mock perception sidecar — returns the queued detections list."""

    detections: list[Detection] = field(default_factory=list)
    set_classes_calls: list[tuple[str, ...]] = field(default_factory=list)
    detect_calls: int = 0
    closed: bool = False

    def set_classes(self, classes):  # noqa: ANN001
        self.set_classes_calls.append(tuple(classes))
        return True

    def detect(self, image_bgr, *, classes=None, **_):  # noqa: ANN001
        self.detect_calls += 1
        return list(self.detections)

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeLifter:
    """Mock 2D→3D lifter — returns canned positions in order."""

    next_positions: list[Vector3 | None] = field(default_factory=list)
    calls: int = 0

    def lift(self, mask, scan_points_map, pose_position, pose_orientation):  # noqa: ANN001
        from xiao_hei_vln.perception.lifter import LiftResult

        pos = (
            self.next_positions[self.calls]
            if self.calls < len(self.next_positions)
            else None
        )
        self.calls += 1
        return LiftResult(position=pos, n_inliers=42 if pos is not None else 0)


def _detection(label: str, score: float = 0.9) -> Detection:
    return Detection(
        label=label, score=score,
        bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
        mask=np.zeros((640, 1920), dtype=bool),
    )


def _responder(
    *,
    scene: SceneRepresentation | None = None,
    detections: list[Detection] | None = None,
    lifter_positions: list[Vector3 | None] | None = None,
    trajectory_path: Path | None = None,
    score_threshold: float = 0.25,
    take_waypoint_reached_signals=None,
) -> tuple[PerceptionResponder, _FakeClient, _FakeLifter, SceneRepresentation]:
    scene = scene or SceneRepresentation()
    client = _FakeClient(detections=list(detections or []))
    lifter = _FakeLifter(next_positions=list(lifter_positions or []))
    vocab = Vocabulary(prior=["chair", "table", "lamp"])
    r = PerceptionResponder(
        scene,
        client=client,                # type: ignore[arg-type]
        lifter=lifter,                # type: ignore[arg-type]
        vocabulary=vocab,
        trajectory_path=trajectory_path,
        score_threshold=score_threshold,
        take_waypoint_reached_signals=take_waypoint_reached_signals,
    )
    return r, client, lifter, scene


# ---------------------------------------------------------------------------
# 1. _inject_visible — the perception loop
# ---------------------------------------------------------------------------


class TestInjectVisible:
    def test_no_image_or_scan_skips_detect(self) -> None:
        r, client, _, _ = _responder()
        # Snapshot with pose but no image / no scan → no /detect call.
        r.respond(_snapshot(pose=_pose(0, 0)))
        assert client.detect_calls == 0

    def test_detections_lifted_and_added_to_scene(self) -> None:
        positions = [
            Vector3(x=1.0, y=0.0, z=0.0),
            Vector3(x=2.0, y=0.0, z=0.0),
        ]
        r, client, lifter, scene = _responder(
            detections=[_detection("chair"), _detection("table")],
            lifter_positions=positions,
        )
        r.respond(_snapshot(
            pose=_pose(0, 0),
            image=_image(),
            scan=_scan(np.zeros((5, 4))),
        ))
        assert client.detect_calls == 1
        assert lifter.calls == 2
        labels = {o.label for o in scene.objects}
        assert labels == {"chair", "table"}

    def test_detections_dropped_when_lifter_returns_none(self) -> None:
        r, _, lifter, scene = _responder(
            detections=[_detection("chair"), _detection("table")],
            lifter_positions=[None, Vector3(x=2.0, y=0.0, z=0.0)],
        )
        r.respond(_snapshot(
            pose=_pose(0, 0),
            image=_image(),
            scan=_scan(np.zeros((5, 4))),
        ))
        assert lifter.calls == 2
        labels = {o.label for o in scene.objects}
        assert labels == {"table"}     # chair was dropped (no LiDAR support)

    def test_set_classes_pushed_with_current_vocab(self) -> None:
        r, client, _, _ = _responder()
        r.respond(_snapshot(
            pose=_pose(0, 0),
            image=_image(),
            scan=_scan(np.zeros((5, 4))),
            question_text="Find the lamp",
        ))
        assert client.set_classes_calls       # at least one set_classes call
        last = client.set_classes_calls[-1]
        assert "lamp" in last                  # from prior
        assert "chair" in last                 # from prior
        # No question-derived nouns of interest beyond the prior here
        # because the prior already covers "lamp".


# ---------------------------------------------------------------------------
# 2. Phase A — coverage trajectory walking
# ---------------------------------------------------------------------------


def _write_traj(tmp_path: Path, waypoints: list[tuple[float, float]]) -> Path:
    p = tmp_path / "traj.json"
    p.write_text(json.dumps({
        "waypoints": [{"x": x, "y": y, "heading": 0.0} for x, y in waypoints],
    }))
    return p


class TestPhaseAWalk:
    def test_no_trajectory_skips_phase_a(self, tmp_path: Path) -> None:
        r, _, _, scene = _responder()
        # Add a chair so an answer is possible.
        scene.add_object(ObjectObservation(
            label="chair", position=Vector3(x=1.0, y=0.0, z=0.0),
        ))
        out = r.respond(_snapshot(
            pose=_pose(0, 0),
            question_text="How many chairs are there",
        ))
        assert isinstance(out, NumericalResponse)
        assert out.value == 1
        assert r.is_done()

    def test_trajectory_does_not_advance_without_signal(self, tmp_path: Path) -> None:
        """Without a signal source, the responder never advances past wp0 —
        proximity to our commanded waypoint is no longer a trigger. Matches
        the autonomy stack's contract: only /way_point_reached counts."""
        traj = _write_traj(tmp_path, [(1.0, 0.0), (5.0, 0.0)])
        r, _, _, _ = _responder(trajectory_path=traj)

        # Snapshot puts robot exactly at wp0 — but no signal source, so
        # responder keeps emitting wp0 indefinitely.
        for _ in range(3):
            out = r.respond(_snapshot(
                pose=_pose(1.0, 0.0), image=_image(), scan=_scan(np.zeros((5, 4))),
                question_text="How many chairs are there",
            ))
            assert isinstance(out, WaypointPathResponse)
            assert out.waypoints[0].x == 1.0

    def test_trajectory_advances_on_signal_even_when_far(self, tmp_path: Path) -> None:
        """The autonomy stack's /way_point_reached signal overrides the
        distance-based check. When the signal fires, advance the wp_idx
        even though the robot is nowhere near our commanded waypoint
        (the autonomy stack steered the robot to a safer adjusted
        point and considers that 'done')."""
        traj = _write_traj(tmp_path, [(10.0, 0.0), (20.0, 0.0)])
        signals = [0]

        def _take() -> int:
            n = signals[0]
            signals[0] = 0
            return n

        r, _, _, _ = _responder(trajectory_path=traj, take_waypoint_reached_signals=_take)

        # First tick — far from wp0, no signal yet → emit wp0.
        out = r.respond(_snapshot(
            pose=_pose(0, 0), image=_image(), scan=_scan(np.zeros((5, 4))),
            question_text="How many chairs are there",
        ))
        assert isinstance(out, WaypointPathResponse)
        assert out.waypoints[0].x == 10.0

        # Autonomy stack signals "I reached *my* adjusted waypoint" —
        # responder must advance wp_idx even though the robot is still at origin.
        signals[0] = 1
        out = r.respond(_snapshot(
            pose=_pose(0, 0), image=_image(), scan=_scan(np.zeros((5, 4))),
            question_text="How many chairs are there",
        ))
        assert isinstance(out, WaypointPathResponse)
        assert out.waypoints[0].x == 20.0     # advanced to wp1

    def test_signal_drained_per_tick(self, tmp_path: Path) -> None:
        """Signal counter must be drained on each tick — only the *first*
        signal advances; later ticks without new signals stay put."""
        traj = _write_traj(tmp_path, [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)])
        signals = [0]

        def _take() -> int:
            n = signals[0]
            signals[0] = 0
            return n

        r, _, _, _ = _responder(trajectory_path=traj, take_waypoint_reached_signals=_take)
        snap = _snapshot(
            pose=_pose(0, 0), image=_image(), scan=_scan(np.zeros((5, 4))),
            question_text="How many chairs are there",
        )
        # Drain once at wp0
        signals[0] = 1
        r.respond(snap)                        # advances to wp1 (no _within_reach)
        # Wait: prior call drained signal. Next call without new signal stays at wp1.
        out = r.respond(snap)
        assert isinstance(out, WaypointPathResponse)
        assert out.waypoints[0].x == 20.0      # still wp1
        # New signal → wp2
        signals[0] = 1
        out = r.respond(snap)
        assert out.waypoints[0].x == 30.0


# ---------------------------------------------------------------------------
# 3. Phase B — answering from the scene graph
# ---------------------------------------------------------------------------


class TestPhaseBAnswers:
    def test_numerical_counts_scene_objects(self) -> None:
        r, _, _, scene = _responder()
        for x in (1.0, 5.0, 10.0):     # outside merge_radius, so 3 distinct
            scene.add_object(ObjectObservation(
                label="chair", position=Vector3(x=x, y=0.0, z=0.0),
            ))
        out = r.respond(_snapshot(
            pose=_pose(0, 0),
            question_text="How many chairs are in the room",
        ))
        assert isinstance(out, NumericalResponse)
        assert out.value == 3
        assert r.is_done()

    def test_numerical_zero_when_label_not_in_scene(self) -> None:
        r, _, _, _ = _responder()
        out = r.respond(_snapshot(
            pose=_pose(0, 0),
            question_text="How many gargoyles are there",
        ))
        assert isinstance(out, NumericalResponse)
        assert out.value == 0

    def test_object_reference_picks_closest(self) -> None:
        r, _, _, scene = _responder()
        scene.add_object(ObjectObservation(
            label="chair", position=Vector3(x=5.0, y=0.0, z=0.0),
        ))
        scene.add_object(ObjectObservation(
            label="chair", position=Vector3(x=1.0, y=0.0, z=0.0),
        ))
        out = r.respond(_snapshot(
            pose=_pose(0, 0),
            question_text="Find the chair",
            qtype=QuestionType.OBJECT_REFERENCE,
        ))
        assert isinstance(out, ObjectReferenceResponse)
        assert out.label == "chair"
        assert out.center.x == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_closes_client(self) -> None:
        r, client, _, _ = _responder()
        r.close()
        assert client.closed

    def test_reset_clears_state(self) -> None:
        r, _, _, scene = _responder()
        scene.add_object(ObjectObservation(
            label="chair", position=Vector3(x=1, y=0, z=0),
        ))
        r.respond(_snapshot(
            pose=_pose(0, 0),
            question_text="How many chairs are there",
        ))
        assert r.is_done()
        r.reset()
        assert not r.is_done()
