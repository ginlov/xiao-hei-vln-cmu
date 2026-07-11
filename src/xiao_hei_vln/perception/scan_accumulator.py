"""Multi-frame LiDAR accumulation for the 2D→3D lift.

A single ``/registered_scan`` sweep is too sparse to support small
objects: their masks intersect only a handful of returns, so the lifter's
``min_inliers`` gate drops them — and the few points that *do* land are
often background bleeding through the mask edge, which mislocalises the
object by metres. This accumulator densifies the cloud the lifter sees.

Because ``/registered_scan`` is already in the **map frame**, sweeps taken
from different robot poses concatenate directly. We keep a rolling window
of **keyframes** — a sweep is stored only once the robot has moved or
turned enough to add genuinely new coverage, so standing still doesn't
stack near-identical clouds. The merged cloud is voxel-downsampled to cap
the point count (projection cost) and to collapse the overlap between
sweeps of the same surface.

The lifter still transforms the cloud with the *current* pose and applies
the mask + z-buffer, so points from earlier viewpoints that no longer sit
along a current bearing are filtered out naturally — accumulation only
adds density where the current view and a past view overlap, which is
exactly where a small object needs more support.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from xiao_hei_vln.messages.common import Quaternion, Vector3

DEFAULT_MAX_KEYFRAMES: int = 10
DEFAULT_MIN_MOVE_M: float = 0.25
DEFAULT_MIN_ROT_DEG: float = 15.0
DEFAULT_VOXEL_M: float = 0.05


def _yaw(q: Quaternion) -> float:
    """Planar yaw (rad) from an XYZW quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """Keep one representative point per ``voxel_m`` cube.

    Collapses the overlap between sweeps of the same surface and caps the
    total point count. ``voxel_m <= 0`` disables downsampling.
    """
    if voxel_m <= 0.0 or points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


class ScanAccumulator:
    """Rolling keyframe buffer of map-frame LiDAR sweeps.

    Call :meth:`update` once per tick with the current registered scan and
    pose; it returns the densified, voxel-downsampled cloud to lift against.
    """

    def __init__(
        self,
        *,
        max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
        min_move_m: float = DEFAULT_MIN_MOVE_M,
        min_rot_deg: float = DEFAULT_MIN_ROT_DEG,
        voxel_m: float = DEFAULT_VOXEL_M,
    ) -> None:
        self._max_keyframes = int(max_keyframes)
        self._min_move_m = float(min_move_m)
        self._min_rot_rad = math.radians(float(min_rot_deg))
        self._voxel_m = float(voxel_m)
        self._keyframes: deque[np.ndarray] = deque(maxlen=self._max_keyframes)
        self._last_pos: tuple[float, float, float] | None = None
        self._last_yaw: float = 0.0
        self._cache: np.ndarray = np.empty((0, 3), dtype=np.float64)

    @property
    def n_keyframes(self) -> int:
        return len(self._keyframes)

    def reset(self) -> None:
        """Drop all accumulated sweeps (e.g. on a new scene/session)."""
        self._keyframes.clear()
        self._last_pos = None
        self._cache = np.empty((0, 3), dtype=np.float64)

    def _should_keyframe(self, pose_position: Vector3, yaw: float) -> bool:
        if self._last_pos is None:
            return True
        dx = pose_position.x - self._last_pos[0]
        dy = pose_position.y - self._last_pos[1]
        moved = math.hypot(dx, dy)
        turned = abs(_wrap(yaw - self._last_yaw))
        return moved >= self._min_move_m or turned >= self._min_rot_rad

    def update(
        self,
        scan_points_map: np.ndarray,
        pose_position: Vector3,
        pose_orientation: Quaternion,
    ) -> np.ndarray:
        """Ingest the current sweep; return the densified map-frame cloud.

        A new keyframe (and a recompute of the merged cloud) is committed
        only when the robot has moved ``>= min_move_m`` or turned
        ``>= min_rot_deg`` since the last keyframe; otherwise the previously
        merged cloud is returned unchanged (cheap on stationary ticks).
        """
        if scan_points_map.ndim != 2 or scan_points_map.shape[1] < 3:
            raise ValueError(
                f"scan_points_map must be (N, >=3); got {scan_points_map.shape}",
            )
        pts = scan_points_map[:, :3].astype(np.float64, copy=False)
        yaw = _yaw(pose_orientation)
        if self._should_keyframe(pose_position, yaw):
            self._keyframes.append(pts)
            self._last_pos = (pose_position.x, pose_position.y, pose_position.z)
            self._last_yaw = yaw
            merged = (
                np.vstack(self._keyframes) if self._keyframes else pts
            )
            self._cache = voxel_downsample(merged, self._voxel_m)
        return self._cache
