"""Tests for ``xiao_hei_vln.scene.render`` — graph/topdown PNGs + tables."""

from __future__ import annotations

import base64

import pytest

from xiao_hei_vln.scene.render import (
    SCENE_TABLE_CSS,
    render_graph_png,
    render_node_tables,
    render_topdown_png,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scene(tick_id: int = 0, *, vps=None, objs=None, bounds=None) -> dict:
    """Build a scene_dict mirroring SceneRepresentation.to_dict()."""
    return {
        "tick_id": tick_id,
        "room": {
            "label": "scene",
            "scene_bounds": bounds,
            "best_image_tick_id": None,
            "best_image_position": None,
            "viewpoint_tick_ids": [v["tick_id"] for v in (vps or [])],
        },
        "viewpoints": vps or [],
        "objects": objs or [],
    }


def _vp(tick_id: int, x: float, y: float, yaw: float = 0.0) -> dict:
    return {"tick_id": tick_id, "position": [x, y, 0.0], "yaw": yaw}


def _obj(label: str, x: float, y: float, *, observed_from=(0,), conf=1.0,
         relations=None) -> dict:
    return {
        "label": label,
        "position": [x, y, 0.0],
        "confidence": conf,
        "bbox_min": None, "bbox_max": None,
        "first_tick_id": observed_from[0] if observed_from else 0,
        "last_tick_id": observed_from[-1] if observed_from else 0,
        "observing_viewpoint_ids": list(observed_from),
        "spatial_relations": list(relations or []),
    }


def _is_valid_base64_png(s: str) -> bool:
    """Cheap sniff: decodes + starts with PNG signature."""
    raw = base64.b64decode(s)
    return raw[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# render_topdown_png
# ---------------------------------------------------------------------------


class TestTopdown:
    def test_empty_scene_renders(self) -> None:
        png = render_topdown_png(_scene())
        assert _is_valid_base64_png(png)
        assert len(base64.b64decode(png)) > 200  # not a stub

    def test_populated_scene_renders(self) -> None:
        scene = _scene(
            tick_id=3,
            vps=[_vp(1, 0, 0), _vp(3, 5, 0)],
            objs=[_obj("chair", 1, 1), _obj("table", 4, 0)],
            bounds=[[-1, -1, 0], [6, 6, 0]],
        )
        png = render_topdown_png(
            scene,
            current_pose={"x": 5.0, "y": 0.0, "yaw": 0.0},
            pose_history=[(0.0, 0.0), (2.5, 0.0), (5.0, 0.0)],
            planned_waypoints=[{"x": 0, "y": 0}, {"x": 5, "y": 0}],
        )
        assert _is_valid_base64_png(png)

    def test_renders_without_pose(self) -> None:
        # Local-near-edges + labels should silently skip when no pose given.
        scene = _scene(
            tick_id=1,
            vps=[_vp(0, 0, 0)],
            objs=[_obj("chair", 1, 0)],
        )
        png = render_topdown_png(scene)
        assert _is_valid_base64_png(png)


# ---------------------------------------------------------------------------
# render_graph_png — hierarchical topology
# ---------------------------------------------------------------------------


class TestGraph:
    def test_empty_scene_just_room_node(self) -> None:
        png = render_graph_png(_scene())
        assert _is_valid_base64_png(png)

    def test_three_levels_renders(self) -> None:
        scene = _scene(
            tick_id=5,
            vps=[_vp(0, 0, 0), _vp(2, 3, 0), _vp(4, 6, 0)],
            objs=[
                _obj("chair", 1, 1, observed_from=(0,)),
                _obj("table", 1, 2, observed_from=(0, 2),
                     relations=[{"target_label": "chair", "target_index": 0,
                                 "relation": "near"}]),
                _obj("lamp",  6, 0, observed_from=(4,)),
            ],
        )
        png = render_graph_png(scene)
        assert _is_valid_base64_png(png)

    def test_layout_is_stable_across_calls(self) -> None:
        # Same scene_dict twice should produce identical PNGs (deterministic).
        scene = _scene(
            tick_id=2,
            vps=[_vp(0, 0, 0), _vp(2, 3, 0)],
            objs=[_obj("chair", 1, 1, observed_from=(0,))],
        )
        assert render_graph_png(scene) == render_graph_png(scene)

    def test_object_observed_at_non_viewpoint_tick(self) -> None:
        """Regression: an object's observed_from_tick_ids is a raw tick_id;
        only some ticks create viewpoints. Edges and positions used to
        crash with KeyError when the tick wasn't a viewpoint.
        """
        scene = _scene(
            tick_id=5,
            vps=[_vp(0, 0, 0), _vp(3, 5, 0)],            # viewpoints at ticks 0, 3
            objs=[
                _obj("chair", 1, 1, observed_from=(1,)),   # observed at tick 1
                _obj("table", 4, 0, observed_from=(2, 4)), # observed at ticks 2, 4
                _obj("lamp",  6, 0, observed_from=(8,)),   # observed at tick 8 (future-ish)
            ],
        )
        png = render_graph_png(scene)
        assert _is_valid_base64_png(png)

    def test_object_observed_before_any_viewpoint(self) -> None:
        """Edge case: an object added before the first viewpoint exists."""
        scene = _scene(
            tick_id=2,
            vps=[],                                       # no viewpoints yet
            objs=[_obj("ghost", 1, 1, observed_from=(1,))],
        )
        png = render_graph_png(scene)
        assert _is_valid_base64_png(png)


# ---------------------------------------------------------------------------
# render_node_tables — HTML
# ---------------------------------------------------------------------------


class TestTables:
    def test_empty_scene_has_placeholder_rows(self) -> None:
        html = render_node_tables(_scene())
        assert "<table class='scene-table'>" in html
        assert "no viewpoints yet" in html
        assert "no objects yet" in html

    def test_room_row_present(self) -> None:
        html = render_node_tables(_scene(
            bounds=[[-1, -2, 0], [3, 4, 0]],
        ))
        assert "<h3>Room</h3>" in html
        assert "scene" in html  # the room label
        assert "[-1.0, -2.0]" in html and "[3.0, 4.0]" in html

    def test_viewpoint_rows_count(self) -> None:
        scene = _scene(vps=[_vp(0, 0, 0), _vp(2, 1, 0), _vp(5, 3, 0)])
        html = render_node_tables(scene)
        assert "Viewpoints (3)" in html
        # 3 data rows, not the placeholder.
        assert "no viewpoints yet" not in html

    def test_object_rows_and_near_summary(self) -> None:
        scene = _scene(
            objs=[
                _obj("chair", 0, 0, observed_from=(0, 2),
                     relations=[
                         {"target_label": "table", "target_index": 1, "relation": "near"},
                         {"target_label": "lamp",  "target_index": 2, "relation": "near"},
                     ]),
                _obj("table", 1, 0, observed_from=(2,)),
                _obj("lamp",  2, 0, observed_from=(2,)),
            ],
        )
        html = render_node_tables(scene)
        assert "Objects (3)" in html
        assert "chair" in html and "table" in html and "lamp" in html
        # near summary "table, lamp" listed for chair
        assert "table, lamp" in html

    def test_html_is_well_formed_ish(self) -> None:
        html = render_node_tables(_scene(
            vps=[_vp(0, 0, 0)],
            objs=[_obj("chair", 1, 0)],
        ))
        assert html.count("<table") == 3
        assert html.count("</table>") == 3
        assert html.count("<thead>") == 3
        assert html.count("<tbody>") == 3

    def test_label_is_html_escaped(self) -> None:
        html = render_node_tables(_scene(
            objs=[_obj("<script>alert(1)</script>", 0, 0)],
        ))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# CSS constant — must contain the class names the tables use
# ---------------------------------------------------------------------------


def test_css_constant_is_consistent_with_class_names() -> None:
    assert ".scene-table" in SCENE_TABLE_CSS
    assert ".scene-tables" in SCENE_TABLE_CSS


# ---------------------------------------------------------------------------
# Sanity — wiring into a logger-like JSONL fixture file
# ---------------------------------------------------------------------------


def test_renders_a_full_fixture_session(tmp_path) -> None:
    """Drive all three renderers off a tiny synthetic fixture."""
    scenes = [
        _scene(),
        _scene(tick_id=1, vps=[_vp(1, 0, 0)]),
        _scene(
            tick_id=2,
            vps=[_vp(1, 0, 0)],
            objs=[_obj("chair", 1, 0, observed_from=(1,))],
        ),
        _scene(
            tick_id=5,
            vps=[_vp(1, 0, 0), _vp(5, 4, 0)],
            objs=[
                _obj("chair", 1, 0, observed_from=(1,)),
                _obj("table", 4, 0, observed_from=(5,),
                     relations=[{"target_label": "chair", "target_index": 0,
                                 "relation": "near"}]),
            ],
            bounds=[[-1, -1, 0], [6, 5, 0]],
        ),
    ]
    for s in scenes:
        assert _is_valid_base64_png(render_topdown_png(s))
        assert _is_valid_base64_png(render_graph_png(s))
        assert "<table class='scene-table'>" in render_node_tables(s)


@pytest.mark.parametrize("scene_dict", [
    # Empty (no viewpoints / objects).
    {"tick_id": 0, "room": {"label": "scene", "scene_bounds": None,
                            "best_image_tick_id": None,
                            "best_image_position": None,
                            "viewpoint_tick_ids": []},
     "viewpoints": [], "objects": []},
    # Missing optional keys — renderers should not crash.
    {"tick_id": 1, "room": {"label": "scene"}, "viewpoints": [], "objects": []},
])
def test_robust_to_minimal_inputs(scene_dict) -> None:
    assert _is_valid_base64_png(render_topdown_png(scene_dict))
    assert _is_valid_base64_png(render_graph_png(scene_dict))
    assert render_node_tables(scene_dict)
