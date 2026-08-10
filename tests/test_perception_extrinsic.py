"""Tests for the sensor→camera extrinsic.

The sim publishes this transform as a ROS `static_transform_publisher`, whose
translation is expressed in the **parent (sensor) frame**. The constant this
module needs is in the **camera frame**, so the ROS numbers cannot be pasted
across — they have to be rotated first. They were, once, pasted across, which
put the camera 0.1 m forward instead of 0.1 m up and cost ~1° of systematic
elevation error on every lift (TASK 27).

Nothing downstream would notice: the error is small, constant, and entirely
self-consistent. These tests state the geometry in terms a reader can check
against the physical setup — where a point directly above the LiDAR lands, and
where the LiDAR itself lands — rather than restating the constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from xiao_hei_vln.perception.geometry import (
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)

CAMERA_ABOVE_LIDAR_M = 0.1


def _to_camera(p_sensor) -> np.ndarray:
    R, t = sensor_to_camera_transform()
    return np.asarray(p_sensor, dtype=float) @ R.T + t


class TestAxes:
    """+x forward, +y left, +z up  →  +z forward, +x right, +y down."""

    @pytest.mark.parametrize(("sensor", "camera"), [
        ((1, 0, 0), (0, 0, 1)),        # forward → forward
        ((0, 1, 0), (-1, 0, 0)),       # left    → -x (i.e. left of centre)
        ((0, 0, 1), (0, -1, 0)),       # up      → -y (y is down)
    ])
    def test_axis_mapping(self, sensor, camera) -> None:
        R, _ = sensor_to_camera_transform()
        assert np.allclose(np.asarray(sensor, dtype=float) @ R.T, camera)

    def test_rotation_is_orthonormal(self) -> None:
        R, _ = sensor_to_camera_transform()
        assert np.allclose(R @ R.T, np.eye(3))
        assert np.isclose(np.linalg.det(R), 1.0)      # right-handed, no mirror


class TestTranslation:
    def test_lidar_origin_sits_below_the_camera(self) -> None:
        """The whole content of the constant, stated physically.

        The LiDAR is 0.1 m below the camera, so in camera coords (+y down) it
        is at y = +0.1 — not at z = +0.1, which would put it 0.1 m *in front*.
        """
        assert np.allclose(_to_camera([(0.0, 0.0, 0.0)]),
                           [[0.0, CAMERA_ABOVE_LIDAR_M, 0.0]])

    def test_point_level_with_the_camera_projects_to_the_horizon(self) -> None:
        """A point 0.1 m above the LiDAR is level with the camera, so it must
        land on the equirect horizon — the row test that the forward-shifted
        constant fails."""
        p = _to_camera([(3.0, 0.0, CAMERA_ABOVE_LIDAR_M)])
        _, v, valid = project_camera_points_to_equirect(p)
        assert valid[0]
        assert v[0] == pytest.approx(320.0, abs=1e-6)     # EQUIRECT_H / 2

    def test_translation_is_purely_vertical_in_camera_coords(self) -> None:
        # A forward or lateral component would skew azimuth as well, which is
        # what made the original error show up in the bearing residual too.
        _, t = sensor_to_camera_transform()
        assert t[0] == 0.0 and t[2] == 0.0
        assert t[1] == pytest.approx(CAMERA_ABOVE_LIDAR_M)

    def test_accessors_return_copies(self) -> None:
        # Callers cache these (PointLifter keeps them for the process
        # lifetime); handing out the module globals would let one mutation
        # silently re-aim every lift.
        _, t = sensor_to_camera_transform()
        t[1] = 99.0
        _, again = sensor_to_camera_transform()
        assert again[1] == pytest.approx(CAMERA_ABOVE_LIDAR_M)
