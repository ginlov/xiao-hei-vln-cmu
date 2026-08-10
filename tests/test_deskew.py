"""Tests for the image/scan time-skew pose correction.

The direction of the correction is the whole point. Applying it backwards
doubles the error instead of removing it (measured: yaw-rate correlation
−0.63 → −0.85 at the wrong sign), and no smoke test would notice — the boxes
would simply be wrong in the other direction. So these tests assert the *sign*
against the stack's actual bearing convention, not just the magnitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xiao_hei_vln.messages.common import Quaternion
from xiao_hei_vln.perception.deskew import (
    MAX_YAW_RATE_RAD_S,
    PoseDeskew,
    rotate_yaw,
    yaw_of,
)
from xiao_hei_vln.perception.geometry import sensor_to_camera_transform
from xiao_hei_vln.perception.lifter import _rotation_from_quaternion


def _q(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def _bearing(yaw: float, point=(3.0, 0.0, 0.0)) -> float:
    """Camera-frame longitude of a fixed map point, at this robot yaw.

    Uses the production transform chain — the same one the lifter runs.
    """
    R = _rotation_from_quaternion(_q(yaw))
    R_sc, t_sc = sensor_to_camera_transform()
    cam = (np.array([point]) @ R) @ R_sc.T + t_sc
    return float(np.arctan2(cam[0, 0], cam[0, 2]))


class TestYawHelpers:
    def test_yaw_of_round_trips(self):
        for y in (-3.0, -1.0, 0.0, 0.5, 3.0):
            assert yaw_of(_q(y)) == pytest.approx(y, abs=1e-9)

    def test_rotate_yaw_adds(self):
        assert yaw_of(rotate_yaw(_q(0.3), 0.2)) == pytest.approx(0.5, abs=1e-9)

    def test_rotate_yaw_zero_is_identity(self):
        q = _q(0.7)
        assert rotate_yaw(q, 0.0) is q

    def test_rotate_yaw_wraps(self):
        out = yaw_of(rotate_yaw(_q(3.0), 0.3))
        assert out == pytest.approx(3.3 - 2 * math.pi, abs=1e-9)


class TestBearingConvention:
    def test_bearing_increases_with_yaw(self):
        """The premise the correction's sign rests on: d(bearing)/d(yaw) = +1.

        If this ever flips, `PoseDeskew.update` is applying its correction the
        wrong way and this test is the only thing that will say so.
        """
        d = (_bearing(0.1) - _bearing(-0.1)) / 0.2
        assert d == pytest.approx(1.0, abs=0.05)


class TestPoseDeskew:
    def test_zero_lag_is_a_no_op(self):
        d = PoseDeskew(0.0)
        d.update(_q(0.0), 0.0)
        assert yaw_of(d.update(_q(1.0), 0.5)) == pytest.approx(1.0)

    def test_first_sample_is_never_corrected(self):
        # No previous pose means no rate; the pose as given is the only
        # honest answer.
        d = PoseDeskew(0.368)
        assert yaw_of(d.update(_q(0.4), 10.0)) == pytest.approx(0.4)

    def test_correction_lags_the_pose_when_turning(self):
        """Turning left (+rate), the shutter opened at a SMALLER yaw.

        This is the direction the offline fit demanded: the image is older
        than the pose, so the camera had not yet turned as far.
        """
        d = PoseDeskew(0.4)
        d.update(_q(0.0), 0.0)
        out = yaw_of(d.update(_q(1.0), 1.0))          # rate = +1 rad/s
        assert out == pytest.approx(1.0 - 0.4, abs=1e-9)
        assert out < 1.0

    def test_correction_reverses_with_the_turn(self):
        d = PoseDeskew(0.4)
        d.update(_q(0.0), 0.0)
        out = yaw_of(d.update(_q(-1.0), 1.0))         # rate = -1 rad/s
        assert out == pytest.approx(-1.0 + 0.4, abs=1e-9)

    def test_correction_removes_the_bearing_error(self):
        """End to end: a bearing computed from the corrected pose matches the
        bearing the camera actually saw."""
        lag, rate = 0.368, 0.8
        yaw_shutter = 2.0                       # where the camera really was
        yaw_pose = yaw_shutter + lag * rate     # where the pose says it is
        d = PoseDeskew(lag)
        d.update(_q(yaw_pose - rate * 1.0), 0.0)
        corrected = yaw_of(d.update(_q(yaw_pose), 1.0))
        assert _bearing(corrected) == pytest.approx(_bearing(yaw_shutter),
                                                    abs=1e-6)

    def test_stationary_robot_is_untouched(self):
        d = PoseDeskew(0.368)
        d.update(_q(0.7), 0.0)
        assert yaw_of(d.update(_q(0.7), 0.5)) == pytest.approx(0.7)

    def test_gap_in_the_stream_skips_the_correction(self):
        d = PoseDeskew(0.4)
        d.update(_q(0.0), 0.0)
        assert yaw_of(d.update(_q(1.0), 60.0)) == pytest.approx(1.0)

    def test_implausible_rate_skips_the_correction(self):
        # A relocalisation jump is not motion; correcting for it would throw
        # the pose somewhere arbitrary.
        d = PoseDeskew(0.4)
        d.update(_q(0.0), 0.0)
        jump = (MAX_YAW_RATE_RAD_S + 1.0) * 0.1
        assert yaw_of(d.update(_q(jump), 0.1)) == pytest.approx(jump)

    def test_rate_is_measured_across_the_wrap(self):
        d = PoseDeskew(0.1)
        d.update(_q(3.1), 0.0)
        # 3.1 → -3.1 is a +0.083 rad step forwards, not a -6.2 rad lurch.
        out = yaw_of(d.update(_q(-3.1), 1.0))
        expected = -3.1 - 0.1 * (2 * math.pi - 6.2)
        assert out == pytest.approx(expected, abs=1e-6)
        assert d.yaw_rate == pytest.approx(2 * math.pi - 6.2, abs=1e-6)

    def test_reset_clears_history(self):
        d = PoseDeskew(0.4)
        d.update(_q(0.0), 0.0)
        d.reset()
        assert yaw_of(d.update(_q(1.0), 1.0)) == pytest.approx(1.0)
