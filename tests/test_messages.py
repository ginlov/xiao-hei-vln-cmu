"""Tests for the VLM I/O data classes."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    ImageFrame,
    LidarScan,
    NumericalResponse,
    ObjectReferenceResponse,
    OdomPose,
    Quaternion,
    QuestionType,
    Stamp,
    TerrainMap,
    Vector3,
    VLMInput,
    Waypoint,
    WaypointPathResponse,
    classify_question,
    parse_vlm_output,
)


def _stamp(t: float = 0.0) -> Stamp:
    return Stamp.from_seconds(t)


def _header(frame: str, t: float = 0.0) -> Header:
    return Header(stamp=_stamp(t), frame_id=frame)


# --- common -----------------------------------------------------------------------


class TestStamp:
    def test_roundtrip_seconds(self) -> None:
        s = Stamp.from_seconds(12.5)
        assert s.sec == 12 and s.nanosec == 500_000_000
        assert s.to_seconds() == pytest.approx(12.5)

    def test_nanosec_rollover(self) -> None:
        # 1.0 - 1e-12 rounds to 1.0 exactly; ensure we don't get 1_000_000_000 ns.
        s = Stamp.from_seconds(1.0)
        assert s.sec == 1 and s.nanosec == 0

    def test_nanosec_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Stamp(sec=0, nanosec=1_000_000_000)


# --- sensors ---------------------------------------------------------------------


class TestImageFrame:
    def test_constructs_with_matching_step(self) -> None:
        f = ImageFrame(
            header=_header("camera"),
            width=4,
            height=2,
            encoding="bgr8",
            step=12,  # 4 px * 3 channels
            data=b"\x00" * 24,
        )
        assert len(f.data) == 24

    def test_rejects_mismatched_data_length(self) -> None:
        with pytest.raises(ValidationError):
            ImageFrame(
                header=_header("camera"),
                width=4,
                height=2,
                encoding="bgr8",
                step=12,
                data=b"\x00" * 20,
            )


class TestLidarScan:
    def test_accepts_n_by_4_array(self) -> None:
        pts = np.zeros((10, 4), dtype=np.float32)
        scan = LidarScan(header=_header("map"), points=pts, source="registered")
        assert scan.points.shape == (10, 4)
        assert scan.points.dtype == np.float32

    def test_upcasts_dtype(self) -> None:
        pts = np.zeros((3, 4), dtype=np.float64)
        scan = LidarScan(header=_header("map"), points=pts, source="sensor")
        assert scan.points.dtype == np.float32

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValidationError):
            LidarScan(
                header=_header("map"),
                points=np.zeros((5, 3), dtype=np.float32),
                source="registered",
            )


class TestTerrainMap:
    @pytest.mark.parametrize("rng", ["local_5m", "ext_20m"])
    def test_valid_ranges(self, rng: str) -> None:
        TerrainMap(
            header=_header("map"),
            points=np.zeros((1, 4), dtype=np.float32),
            range=rng,  # type: ignore[arg-type]
        )

    def test_rejects_unknown_range(self) -> None:
        with pytest.raises(ValidationError):
            TerrainMap(
                header=_header("map"),
                points=np.zeros((1, 4), dtype=np.float32),
                range="wrong",  # type: ignore[arg-type]
            )


class TestOdomPose:
    def test_minimal_construction(self) -> None:
        p = OdomPose(
            header=_header("map"),
            position=Vector3(x=1.0, y=2.0, z=0.75),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        assert p.child_frame_id == "sensor"
        assert p.linear_velocity is None


# --- question --------------------------------------------------------------------


class TestQuestionClassifier:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("How many books are on the sofa", QuestionType.NUMERICAL),
            ("how many chairs", QuestionType.NUMERICAL),
            ("Find the orange chair", QuestionType.OBJECT_REFERENCE),
            ("find the potted plant", QuestionType.OBJECT_REFERENCE),
            ("Take the path near the window", QuestionType.INSTRUCTION_FOLLOWING),
            ("   How many lights", QuestionType.NUMERICAL),  # leading space
        ],
    )
    def test_classify(self, text: str, expected: QuestionType) -> None:
        assert classify_question(text) == expected

    def test_from_text(self) -> None:
        q = ChallengeQuestion.from_text("Find the red cup", _stamp())
        assert q.type == QuestionType.OBJECT_REFERENCE


# --- inputs ----------------------------------------------------------------------


class TestVLMInput:
    def test_cold_start_all_none(self) -> None:
        snap = VLMInput(tick_id=0, tick_time=_stamp())
        assert snap.image is None
        assert snap.is_ready is False

    def test_ready_when_image_pose_question_present(self) -> None:
        snap = VLMInput(
            tick_id=1,
            tick_time=_stamp(),
            image=ImageFrame(
                header=_header("camera"),
                width=2,
                height=1,
                encoding="bgr8",
                step=6,
                data=b"\x00" * 6,
            ),
            pose=OdomPose(
                header=_header("map"),
                position=Vector3(x=0, y=0, z=0.75),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            question=ChallengeQuestion.from_text("How many cups", _stamp()),
        )
        assert snap.is_ready is True


# --- outputs ---------------------------------------------------------------------


class TestVLMOutputDiscriminator:
    def test_parse_numerical(self) -> None:
        out = parse_vlm_output({"kind": "numerical", "value": 3})
        assert isinstance(out, NumericalResponse)
        assert out.value == 3

    def test_parse_object_reference(self) -> None:
        out = parse_vlm_output(
            {
                "kind": "object_reference",
                "label": "chair",
                "object_id": 42,
                "center": {"x": 1.0, "y": 2.0, "z": 0.5},
                "size": {"x": 0.5, "y": 0.5, "z": 1.0},
                "heading": 0.0,
            }
        )
        assert isinstance(out, ObjectReferenceResponse)
        assert out.object_id == 42

    def test_parse_waypoint_path(self) -> None:
        out = parse_vlm_output(
            {
                "kind": "waypoint_path",
                "waypoints": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 0.5, "heading": 1.57},
                ],
            }
        )
        assert isinstance(out, WaypointPathResponse)
        assert len(out.waypoints) == 2

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_vlm_output({"kind": "bogus", "value": 1})

    def test_empty_waypoints_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WaypointPathResponse(waypoints=[])

    def test_json_roundtrip(self) -> None:
        path = WaypointPathResponse(
            waypoints=[Waypoint(x=1.0, y=2.0), Waypoint(x=3.0, y=4.0, heading=0.5)],
        )
        payload = path.model_dump_json()
        restored = parse_vlm_output({"kind": "waypoint_path", **path.model_dump()})
        assert restored == path
        # Round-trip through string too.
        restored_from_json = WaypointPathResponse.model_validate_json(payload)
        assert restored_from_json == path
