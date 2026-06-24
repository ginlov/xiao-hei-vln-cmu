"""Tests for ``xiao_hei_vln.gemini.scene_rep``.

These verify the *serialization* layer on top of the cherry-picked
:class:`xiao_hei_vln.scene.SceneRepresentation`. We don't re-test the
graph construction itself — ``tests/test_scene_representation.py``
already covers that with 54 tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from xiao_hei_vln.gemini.scene_rep import (
    GeminiSceneBundle,
    build_bundle,
    render_occupancy_png,
    serialize_for_gemini,
)
from xiao_hei_vln.messages import (
    ChallengeQuestion,
    Header,
    OdomPose,
    Quaternion,
    Stamp,
    Vector3,
    VLMInput,
)
from xiao_hei_vln.perception.global_map import GlobalMap
from xiao_hei_vln.scene import ObjectObservation, SceneRepresentation


def _stamp() -> Stamp:
    return Stamp.from_seconds(0.0)


def _pose(x: float, y: float) -> OdomPose:
    return OdomPose(
        header=Header(stamp=_stamp(), frame_id="map"),
        position=Vector3(x=x, y=y, z=0.75),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _snapshot(tick_id: int, pose: OdomPose | None) -> VLMInput:
    return VLMInput(
        tick_id=tick_id,
        tick_time=_stamp(),
        pose=pose,
        question=ChallengeQuestion.from_text("How many cups", _stamp()),
    )


def _populated_map() -> GlobalMap:
    gm = GlobalMap(half_extent_m=3.0, resolution_m=0.5)
    pts = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [1.5, 0.5, 0.0, 0.0],
            [1.5, 1.5, 0.0, 0.9],  # one OCCUPIED point
        ],
        dtype=np.float32,
    )
    gm.update(pts, _pose(0.0, 0.0))
    return gm


class TestRenderOccupancyPng:
    def test_returns_png_bytes(self) -> None:
        gm = _populated_map()
        png = render_occupancy_png(gm, trajectory_xy=[(0.0, 0.0), (1.0, 0.5)])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png) > 1000

    def test_uninitialised_map(self) -> None:
        gm = GlobalMap()
        png = render_occupancy_png(gm, trajectory_xy=None)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_viewpoint_markers_change_output(self) -> None:
        gm = _populated_map()
        png_no_vp = render_occupancy_png(gm, viewpoint_xys=None)
        png_with_vp = render_occupancy_png(
            gm, viewpoint_xys=[(0.0, 0.0), (1.0, 0.5)],
        )
        assert png_no_vp != png_with_vp


class TestBuildBundle:
    def test_bundle_carries_scene_dict_and_viewpoints(self) -> None:
        gm = _populated_map()
        scene = SceneRepresentation(viewpoint_radius=0.5)
        # Drive two distinct viewpoints into the scene
        scene.update(_snapshot(0, _pose(0.0, 0.0)))
        scene.update(_snapshot(5, _pose(2.0, 0.0)))

        bundle = build_bundle(
            snapshot=_snapshot(5, _pose(2.0, 0.0)),
            scene=scene,
            global_map=gm,
            trajectory_xy=[(0.0, 0.0), (1.0, 0.5), (2.0, 0.0)],
        )
        assert isinstance(bundle, GeminiSceneBundle)
        assert bundle.scene_dict["tick_id"] == 5
        assert len(bundle.scene_dict["viewpoints"]) == 2
        assert bundle.occupancy_png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        # No image in snapshot → no panorama bytes
        assert bundle.panorama_jpg_bytes is None

    def test_scene_objects_appear_in_dict(self) -> None:
        gm = _populated_map()
        scene = SceneRepresentation(viewpoint_radius=0.5, merge_radius=0.5)
        scene.update(_snapshot(0, _pose(0.0, 0.0)))
        scene.add_object(
            ObjectObservation(
                label="cup",
                position=Vector3(x=1.0, y=0.5, z=0.7),
                confidence=0.9,
            ),
        )
        bundle = build_bundle(
            snapshot=_snapshot(0, _pose(0.0, 0.0)),
            scene=scene,
            global_map=gm,
            trajectory_xy=[(0.0, 0.0)],
        )
        assert any(o["label"] == "cup" for o in bundle.scene_dict["objects"])


class TestSerializeForGemini:
    def test_text_block_contains_scene_json(self) -> None:
        gm = _populated_map()
        scene = SceneRepresentation()
        scene.update(_snapshot(7, _pose(1.0, 2.0)))
        bundle = build_bundle(
            snapshot=_snapshot(7, _pose(1.0, 2.0)),
            scene=scene,
            global_map=gm,
            trajectory_xy=[(0.0, 0.0), (1.0, 2.0)],
        )
        text, images = serialize_for_gemini(bundle)
        # The JSON dump is embedded in the prompt
        assert '"tick_id": 7' in text
        assert "Recent trajectory" in text
        # No panorama in snapshot → only one image (occupancy map)
        assert len(images) == 1
        assert images[0] == bundle.occupancy_png_bytes

    def test_json_in_text_is_parseable(self) -> None:
        gm = _populated_map()
        scene = SceneRepresentation()
        scene.update(_snapshot(3, _pose(0.0, 0.0)))
        bundle = build_bundle(
            snapshot=_snapshot(3, _pose(0.0, 0.0)),
            scene=scene,
            global_map=gm,
            trajectory_xy=[(0.0, 0.0)],
        )
        text, _ = serialize_for_gemini(bundle)
        # Extract the JSON block and verify it parses
        body = text.split("```json\n", 1)[1].split("\n```", 1)[0]
        parsed = json.loads(body)
        assert parsed["tick_id"] == 3
        assert "viewpoints" in parsed
        assert "objects" in parsed
        assert "room" in parsed
