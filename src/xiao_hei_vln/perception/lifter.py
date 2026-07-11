"""2D mask + LiDAR → 3D position.

For each detection mask coming back from the perception sidecar, the
lifter:

1. Transforms the registered scan from the map frame into the sensor
   frame using the robot's current pose (``snapshot.pose``).
2. Applies the static sensor → camera extrinsic (challenge sim).
3. Projects each camera-frame point into equirect pixel coords (the
   same formula the sidecar uses to *create* the mask).
4. Keeps the scan returns whose projected pixel falls inside the mask,
   then drops the ones **occluded** along their bearing (a z-buffer gate:
   only the nearest surface ±``ZBUF_TOL_M`` at each pixel survives, so a
   far wall visible through a doorway inside the mask is not pulled into
   the object's cloud).
5. Returns the median XYZ of the surviving inliers **in the map frame**
   (so downstream consumers see object positions in the same coordinate
   system as everything else in ``SceneRepresentation``), and — for
   consumers that fuse across frames (``ObjectMap``) — the inlier points
   themselves.

The median is robust to mask-edge noise and to the occasional outlier
scan return that snaps through a window or doorway. ``min_inliers``
guards against masks with too few LiDAR supports — typical floors of
~10 keep small / faraway / glassy detections from emitting bogus 3D
points.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)

log = logging.getLogger(__name__)

DEFAULT_MIN_INLIERS: int = 10
# A scan return is "front" (not occluded) if its range is within this margin of
# the nearest range seen at its equirect pixel. Matches the z-buffer tolerance
# from the offline lifter prototype the gate was ported from.
ZBUF_TOL_M: float = 0.2


@dataclass
class LiftResult:
    """Outcome of a single mask lift. ``position`` is ``None`` when the
    mask lacked enough LiDAR support to commit a 3D point.

    ``inlier_points`` is the (M, 3) array of surviving map-frame scan
    returns (post mask + z-buffer), or ``None`` when no position was
    committed — consumers that fuse point clouds across frames
    (``ObjectMap``) use it to grow a converged 3D box."""

    position: Vector3 | None
    n_inliers: int
    inlier_points: np.ndarray | None = None


class PointLifter:
    """Pure-numpy 2D mask → 3D map-frame point projection."""

    def __init__(
        self,
        *,
        min_inliers: int = DEFAULT_MIN_INLIERS,
        max_depth_m: float | None = None,
        enable_zbuffer: bool = True,
    ) -> None:
        """
        Args:
            min_inliers: Minimum number of registered-scan returns that
                must project inside the mask (after the z-buffer gate)
                before a position is returned. Below this, :meth:`lift`
                returns ``LiftResult(position=None, n_inliers=...)`` and
                the caller drops the detection.
            max_depth_m: Optional cap on scan return distance from the
                sensor. Useful when a sparse return through a doorway
                snaps onto a far wall and biases the median; ``None``
                disables the cap.
            enable_zbuffer: Drop mask inliers occluded along their
                bearing — at each equirect pixel only the nearest surface
                ±``ZBUF_TOL_M`` survives. Prevents a far wall/background
                seen through the mask from contaminating the object's
                cloud. **Defaults to ``True``**: a camera cannot see
                through a foreground object, so background returns that
                fall inside a small object's mask must be rejected or the
                lifted position is pulled onto the wall behind it. Kept as
                a flag only so tests can exercise the raw (no-occlusion)
                projection with ``False``.
        """
        self._min_inliers = int(min_inliers)
        self._max_depth_m = max_depth_m
        self._enable_zbuffer = bool(enable_zbuffer)
        self._R_sc, self._t_sc = sensor_to_camera_transform()

    def lift(
        self,
        mask: np.ndarray,
        scan_points_map: np.ndarray,
        pose_position: Vector3,
        pose_orientation: Quaternion,
    ) -> LiftResult:
        """Return the median XYZ (in the *map* frame) of scan returns
        that project inside ``mask``.

        Parameters
        ----------
        mask : (H, W) bool
            Equirectangular pixel mask returned by the perception
            sidecar; ``H = EQUIRECT_H``, ``W = EQUIRECT_W``.
        scan_points_map : (N, 3) or (N, 4) float
            ``registered_scan.points``. The first three columns are the
            point XYZ in the map frame. A trailing intensity column is
            tolerated and ignored.
        pose_position, pose_orientation : robot pose (map frame).
        """
        if mask.shape != (EQUIRECT_H, EQUIRECT_W):
            raise ValueError(
                f"mask must be ({EQUIRECT_H}, {EQUIRECT_W}); got {mask.shape}",
            )
        if scan_points_map.ndim != 2 or scan_points_map.shape[1] < 3:
            raise ValueError(
                f"scan_points_map must be (N, >=3); got {scan_points_map.shape}",
            )
        if scan_points_map.shape[0] == 0:
            return LiftResult(position=None, n_inliers=0)

        xyz_map = scan_points_map[:, :3].astype(np.float64, copy=False)

        # 1. map → sensor frame: subtract robot position, then rotate
        # by the inverse of the robot's orientation (R_sensor_to_map).
        # The sensor sits at the vehicle origin (confirmed by the sim's
        # static_transform_publisher: /sensor → /vehicle is identity).
        R_ms = _rotation_from_quaternion(pose_orientation)         # map ← sensor
        t_ms = np.array([pose_position.x, pose_position.y, pose_position.z])
        xyz_sensor = (xyz_map - t_ms) @ R_ms                       # R_ms.T @ p
        # 2. sensor → camera frame
        xyz_cam = xyz_sensor @ self._R_sc.T + self._t_sc

        # Optional depth cap (use camera-frame z as cheap proxy).
        depth = np.linalg.norm(xyz_cam, axis=1)
        if self._max_depth_m is not None:
            depth_ok = depth <= self._max_depth_m
        else:
            depth_ok = np.ones(depth.shape, dtype=bool)

        # 3. project to equirect pixels
        u, v, in_fov = project_camera_points_to_equirect(xyz_cam)

        # 4. filter by mask. Round to int and clip pixel indices that
        # land exactly on the right edge (u == EQUIRECT_W) to W-1.
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        np.mod(ui, EQUIRECT_W, out=ui)              # wrap-around in u
        in_bounds = (vi >= 0) & (vi < EQUIRECT_H)
        keep = in_fov & in_bounds & depth_ok

        if not keep.any():
            return LiftResult(position=None, n_inliers=0)

        # 4b. z-buffer occlusion gate: per equirect pixel keep only the
        # nearest surface (± ZBUF_TOL_M). ``depth`` is the camera-frame
        # range. fidx uses vi clipped into range so the flat index is safe
        # for every point (out-of-bounds points are excluded via ``keep``).
        if self._enable_zbuffer:
            vclip = np.clip(vi, 0, EQUIRECT_H - 1)
            fidx = vclip * EQUIRECT_W + ui
            nearest = np.full(EQUIRECT_H * EQUIRECT_W, np.inf)
            np.minimum.at(nearest, fidx[keep], depth[keep])
            front = keep & (depth <= nearest[fidx] + ZBUF_TOL_M)
        else:
            front = keep

        in_mask = np.zeros_like(keep)
        in_mask[front] = mask[vi[front], ui[front]]
        n_inliers = int(in_mask.sum())
        if n_inliers < self._min_inliers:
            return LiftResult(position=None, n_inliers=n_inliers)

        inliers = xyz_map[in_mask]
        med = np.median(inliers, axis=0)
        return LiftResult(
            position=Vector3(x=float(med[0]), y=float(med[1]), z=float(med[2])),
            n_inliers=n_inliers,
            inlier_points=inliers,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_from_quaternion(q: Quaternion) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix (XYZW input, returns R such that
    ``world_point = R @ local_point``).

    Standard quaternion-to-matrix formula. We re-implement instead of
    pulling in scipy to keep the perception dependency footprint small.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    x /= norm; y /= norm; z /= norm; w /= norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz),     2.0 * (xy - wz),      2.0 * (xz + wy)],
        [    2.0 * (xy + wz),   1.0 - 2.0 * (xx + zz),    2.0 * (yz - wx)],
        [    2.0 * (xz - wy),       2.0 * (yz + wx),  1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)
