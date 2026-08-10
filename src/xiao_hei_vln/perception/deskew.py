"""Compensate the lift pose for image/scan time skew — OFFLINE only.

The live stack fixes this at the source: ``LatestCache`` keeps a pose history
and matches it to the image's stamp (TASK 27), so the responder never needs
this. It survives for the **offline** benchmark, where the captures store one
already-mispaired pose per viewpoint and no stream history to match against —
so ``replay_score --image-lag`` uses this to *simulate* the timestamp-matched
stack on frozen data, which is the only way to estimate the fix's effect
without re-capturing from the sim.

The skew: a plain latest-value cache pairs whichever message arrived most
recently on each topic. The image stream is the slower one, so the frame handed
to the detector is older than the pose. Standing still that costs nothing;
turning, every mask is lifted against a pose the robot had already rotated past.

Measured on `captures_nav/arabic_room` (TASK 27): **0.368 s**, i.e. up to ~17°
of azimuth at the turn rates in that capture. The measurement is a regression
of per-frame mask-vs-GT bearing error on yaw rate, falsified by showing the
correlation collapses at the fitted value (−0.63 → −0.004) and doubles at the
wrong sign.

The correction is a pure yaw rotation of the pose used for lifting:

    yaw_at_image_time = yaw_of_pose - lag * yaw_rate

Sign, derived and then locked by a test: ``d(bearing)/d(yaw) = +1`` with this
stack's conventions, and the offline fit needed the *GT* bearings rotated by
``-lag * rate``, so the camera's yaw when the shutter opened is *behind* the
pose. Getting this backwards makes the error twice as large, which is why
``test_deskew.py`` asserts the direction rather than only the magnitude.

Roll and pitch are left alone: the vehicle is planar, and a rate estimated from
two ticks 0.5 s apart is not good enough to correct axes that barely move.
"""

from __future__ import annotations

import math

from xiao_hei_vln.messages.common import Quaternion

# Beyond this, a "rate" is a pose glitch or a relocalisation jump, not motion.
MAX_YAW_RATE_RAD_S: float = 3.0
# Two samples further apart than this are not consecutive; rate is meaningless.
MAX_GAP_S: float = 2.0


def yaw_of(q: Quaternion) -> float:
    """Yaw about map +z, in (-π, π]."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rotate_yaw(q: Quaternion, d_yaw: float) -> Quaternion:
    """Pre-multiply ``q`` by a rotation of ``d_yaw`` about the map +z axis."""
    if d_yaw == 0.0:
        return q
    c, s = math.cos(d_yaw / 2.0), math.sin(d_yaw / 2.0)
    # (0, 0, s, c) ⊗ (x, y, z, w)
    return Quaternion(
        x=c * q.x - s * q.y,
        y=c * q.y + s * q.x,
        z=c * q.z + s * q.w,
        w=c * q.w - s * q.z,
    )


class PoseDeskew:
    """Rolling yaw-rate estimate, and the pose correction it implies.

    Stateful because the rate has to come from consecutive poses — nothing in
    the message stream carries angular velocity. Feed it every tick, in order.
    """

    def __init__(self, lag_s: float = 0.0) -> None:
        self.lag_s = float(lag_s)
        self._prev_yaw: float | None = None
        self._prev_t: float | None = None
        self.yaw_rate: float = 0.0          # last estimate, rad/s, for logging

    def reset(self) -> None:
        self._prev_yaw = self._prev_t = None
        self.yaw_rate = 0.0

    def update(self, orientation: Quaternion, t: float) -> Quaternion:
        """Record this sample and return the orientation to lift with.

        A no-op when ``lag_s`` is 0, on the first sample, across a gap, or on
        an implausible rate — in every one of those cases the honest answer is
        the pose as given.
        """
        yaw = yaw_of(orientation)
        prev_yaw, prev_t = self._prev_yaw, self._prev_t
        self._prev_yaw, self._prev_t = yaw, float(t)

        if not self.lag_s or prev_t is None:
            self.yaw_rate = 0.0
            return orientation
        dt = float(t) - prev_t
        if not (1e-3 < dt <= MAX_GAP_S):
            self.yaw_rate = 0.0
            return orientation
        rate = _wrap(yaw - prev_yaw) / dt
        if abs(rate) > MAX_YAW_RATE_RAD_S:
            self.yaw_rate = 0.0
            return orientation
        self.yaw_rate = rate
        return rotate_yaw(orientation, -self.lag_s * rate)
