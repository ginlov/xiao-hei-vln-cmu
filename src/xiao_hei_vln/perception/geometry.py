"""Responder-side equirect geometry — only what the lifter needs.

The sidecar's ``perception/geometry.py`` (at the repo root) handles
both forward (equirect → face) and inverse projections plus LUT
construction. The responder side only needs **forward projection**:
camera-frame XYZ → equirect pixel. Keeping this module tiny means
the import path stays clear and there's no temptation to pull in
``cv2`` or LUT machinery the responder never touches.

The numbers must match the sidecar's exactly. If you change one,
change the other and re-run both test suites.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Camera model constants — must match perception/geometry.py
# ---------------------------------------------------------------------------

EQUIRECT_W: int = 1920
EQUIRECT_H: int = 640
H_FOV: float = 2.0 * np.pi          # full 360°
V_FOV: float = 2.0 * np.pi / 3.0    # 120° vertical crop


# ---------------------------------------------------------------------------
# Sensor → camera static extrinsic
# ---------------------------------------------------------------------------
#
# From the challenge sim's static_transform_publisher logs:
#
#   translation: (0, 0, 0.1)            # camera 10 cm above the LiDAR
#   rotation:    (-0.5, 0.5, -0.5, 0.5) # axis swap from robotics → camera
#                                        # (x forward, z up) → (z forward,
#                                        #                       y down, x right)
#
# Encoded as a 4×4 homogeneous transform applied to *points in the sensor
# frame* to get points in the *camera frame*: ``p_cam = T @ p_sensor``.

_SENSOR_TO_CAMERA_ROTATION: np.ndarray = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)

_SENSOR_TO_CAMERA_TRANSLATION: np.ndarray = np.array([0.0, 0.0, 0.1], dtype=np.float64)


def sensor_to_camera_transform() -> tuple[np.ndarray, np.ndarray]:
    """Return the static ``(R, t)`` taking points from the sensor frame
    (robotics convention: +x forward, +y left, +z up) to the camera frame
    (+z forward, +x right, +y down)."""
    return _SENSOR_TO_CAMERA_ROTATION.copy(), _SENSOR_TO_CAMERA_TRANSLATION.copy()


# ---------------------------------------------------------------------------
# Forward projection — camera-frame point → equirect pixel
# ---------------------------------------------------------------------------


def project_camera_points_to_equirect(
    points_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a batch of camera-frame points into equirect pixel coords.

    Parameters
    ----------
    points_cam : ndarray of shape ``(N, 3)``
        Each row is ``(x, y, z)`` in the camera frame (+z forward,
        +x right, +y down).

    Returns
    -------
    u, v : ndarray of shape ``(N,)``
        Equirect pixel coordinates (floats; the caller rounds to int).
    valid : ndarray of bool, shape ``(N,)``
        ``True`` where the point lies inside the equirect vertical FOV
        (``|phi| <= V_FOV/2``) **and** isn't at the camera origin.
        Points outside the vertical crop have undefined ``(u, v)``.

    The projection uses the standard equirect formula — no intrinsic
    matrix, no per-pixel calibration. Wrap-around in longitude is
    handled by the modulo in :func:`_lam_to_u`.
    """
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(
            f"points_cam must be (N, 3); got {points_cam.shape}",
        )
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    # Defensive — points at the optical centre have undefined direction.
    r_xz = np.sqrt(x * x + z * z)
    on_origin = (r_xz == 0.0) & (y == 0.0)
    lam = np.arctan2(x, z)                       # longitude
    phi = np.arctan2(-y, np.where(r_xz == 0.0, 1.0, r_xz))  # latitude
    valid = (~on_origin) & (np.abs(phi) <= V_FOV / 2.0)
    u = _lam_to_u(lam)
    v = (0.5 - phi / V_FOV) * EQUIRECT_H
    return u, v, valid


def _lam_to_u(lam: np.ndarray) -> np.ndarray:
    """Wrap longitude into ``(-π, π]`` then map to ``[0, EQUIRECT_W)``."""
    lam_wrapped = ((lam + np.pi) % (2.0 * np.pi)) - np.pi
    return (lam_wrapped / H_FOV + 0.5) * EQUIRECT_W
