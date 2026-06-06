"""Tests for the coverage trajectory pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from xiao_hei_vln.trajectory._coverage import (
    _line_of_sight_clear,
    compute_floor_coverage_sets,
    compute_object_coverage_sets,
    greedy_set_cover,
    sample_candidates,
)
from xiao_hei_vln.trajectory._polygon import build_polygon, erode_polygon
from xiao_hei_vln.trajectory._tsp import compute_headings, path_length, solve_tsp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_polygon():
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def square_with_hole():
    outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
    return Polygon(outer, [hole])


class FakeObject:
    """Minimal stand-in for ObjectEntry."""

    def __init__(self, x: float, y: float):
        self.center = type("V", (), {"x": x, "y": y, "z": 0.0})()


# ---------------------------------------------------------------------------
# _polygon tests
# ---------------------------------------------------------------------------

class TestPolygon:
    def test_build_polygon_returns_polygon(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 10, size=(500, 2))
        poly = build_polygon(pts, ratio=0.3)
        assert poly.geom_type in ("Polygon", "MultiPolygon")
        assert poly.area > 0

    def test_erode_polygon_shrinks(self, square_polygon):
        eroded = erode_polygon(square_polygon, robot_radius=1.0)
        assert eroded.area < square_polygon.area

    def test_erode_fallback_on_tiny(self):
        tiny = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
        eroded = erode_polygon(tiny, robot_radius=1.0)
        assert eroded.area == tiny.area


# ---------------------------------------------------------------------------
# _coverage tests
# ---------------------------------------------------------------------------

class TestCandidateSampling:
    def test_candidates_inside_polygon(self, square_polygon):
        cands = sample_candidates(square_polygon, resolution=1.0)
        assert len(cands) > 0
        from shapely import contains_xy
        mask = contains_xy(square_polygon, cands[:, 0], cands[:, 1])
        assert mask.all()

    def test_resolution_affects_count(self, square_polygon):
        coarse = sample_candidates(square_polygon, resolution=2.0)
        fine = sample_candidates(square_polygon, resolution=0.5)
        assert len(fine) > len(coarse)


class TestLineOfSight:
    def test_clear_no_holes(self):
        assert _line_of_sight_clear(0, 0, 5, 5, None) is True
        assert _line_of_sight_clear(0, 0, 5, 5, []) is True

    def test_blocked_by_hole(self):
        hole = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        assert _line_of_sight_clear(0, 0, 5, 5, [hole]) is False

    def test_clear_around_hole(self):
        hole = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        assert _line_of_sight_clear(0, 0, 5, 0, [hole]) is True

    def test_skip_hole(self):
        hole = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        assert _line_of_sight_clear(0, 0, 5, 5, [hole], skip_hole=0) is True


class TestObjectCoverage:
    def test_objects_within_radius_visible(self, square_polygon):
        cands = np.array([[5.0, 5.0]])
        objs = {1: FakeObject(6.0, 5.0), 2: FakeObject(7.0, 5.0)}
        sets = compute_object_coverage_sets(cands, objs, [], radius=3.0)
        assert 1 in sets[0]
        assert 2 in sets[0]

    def test_objects_beyond_radius_not_visible(self, square_polygon):
        cands = np.array([[0.0, 0.0]])
        objs = {1: FakeObject(10.0, 10.0)}
        sets = compute_object_coverage_sets(cands, objs, [], radius=3.0)
        assert 1 not in sets[0]

    def test_hole_blocks_visibility(self):
        cands = np.array([[0.0, 0.0]])
        objs = {1: FakeObject(2.5, 2.5)}
        hole = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        sets = compute_object_coverage_sets(cands, objs, [hole], radius=5.0)
        assert 1 not in sets[0]

    def test_host_hole_exclusion(self):
        cands = np.array([[0.0, 0.0]])
        hole = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        objs = {1: FakeObject(2.5, 2.5)}
        sets = compute_object_coverage_sets(cands, objs, [hole], radius=5.0)
        assert 1 in sets[0]

    def test_empty_objects(self):
        cands = np.array([[0.0, 0.0]])
        sets = compute_object_coverage_sets(cands, {}, [], radius=3.0)
        assert sets == [set()]


class TestFloorCoverage:
    def test_floor_cells_within_radius(self):
        cands = np.array([[0.0, 0.0]])
        cells = np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        sets = compute_floor_coverage_sets(cands, cells, [], radius=3.0)
        assert 0 in sets[0]
        assert 1 in sets[0]
        assert 2 not in sets[0]

    def test_floor_blocked_by_hole(self):
        cands = np.array([[0.0, 0.0]])
        cells = np.array([[2.5, 0.0]])
        hole = Polygon([(1, -1), (2, -1), (2, 1), (1, 1)])
        sets = compute_floor_coverage_sets(cands, cells, [hole], radius=5.0)
        assert 0 not in sets[0]

    def test_empty_floor(self):
        cands = np.array([[0.0, 0.0]])
        sets = compute_floor_coverage_sets(cands, np.array([]).reshape(0, 2), [], radius=3.0)
        assert sets == [set()]


class TestGreedySetCover:
    def test_covers_all_objects(self):
        cands = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        obj_sets = [{1, 2}, {2, 3}, {3, 4}]
        floor_sets = [set(), set(), set()]
        result = greedy_set_cover(cands, obj_sets, floor_sets, {1, 2, 3, 4}, 0)
        assert result.object_coverage == 1.0
        assert len(result.uncovered_object_ids) == 0

    def test_zero_objects(self):
        cands = np.array([[0.0, 0.0], [3.0, 0.0]])
        obj_sets = [set(), set()]
        floor_sets = [{0, 1, 2}, {3, 4, 5}]
        result = greedy_set_cover(cands, obj_sets, floor_sets, set(), 6)
        assert result.object_coverage == 1.0
        assert result.floor_coverage >= 0.95


# ---------------------------------------------------------------------------
# _tsp tests
# ---------------------------------------------------------------------------

class TestTSP:
    def test_visits_all_points(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        order = solve_tsp(pts)
        assert sorted(order) == [0, 1, 2, 3]

    def test_single_point(self):
        assert solve_tsp(np.array([[0.0, 0.0]])) == [0]

    def test_two_points(self):
        order = solve_tsp(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert sorted(order) == [0, 1]

    def test_path_length_triangle(self):
        pts = np.array([[0, 0], [3, 0], [3, 4]], dtype=float)
        assert abs(path_length(pts) - 7.0) < 1e-10

    def test_headings_right(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        h = compute_headings(pts)
        assert all(abs(hi) < 1e-10 for hi in h)

    def test_headings_up(self):
        pts = np.array([[0, 0], [0, 1]], dtype=float)
        h = compute_headings(pts)
        assert abs(h[0] - math.pi / 2) < 1e-10


# ---------------------------------------------------------------------------
# Integration: plan_trajectory
# ---------------------------------------------------------------------------

class TestPlanTrajectory:
    def test_simple_square(self):
        from xiao_hei_vln.trajectory import plan_trajectory

        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 10, size=(2000, 2))
        objs = {i: FakeObject(rng.uniform(2, 8), rng.uniform(2, 8)) for i in range(10)}

        result = plan_trajectory(pts, objs, hull_ratio=0.3, grid_resolution=1.0)
        assert len(result.waypoints) > 0
        assert result.coverage.object_coverage >= 0.8
        assert result.path_length_m > 0

    @pytest.mark.skipif(
        not Path("/home/leo/Projects/CMU-VLN-Challenge-data/unity_env_models/studio.zip").exists(),
        reason="Scene data not available",
    )
    def test_studio_from_zip(self):
        from xiao_hei_vln.trajectory import plan_trajectory_from_zip

        zip_path = Path("/home/leo/Projects/CMU-VLN-Challenge-data/unity_env_models/studio.zip")
        result = plan_trajectory_from_zip(zip_path, "studio")
        assert result.coverage.object_coverage >= 0.95
        assert result.coverage.floor_coverage >= 0.90
        assert len(result.waypoints) > 0
        assert result.path_length_m > 0
