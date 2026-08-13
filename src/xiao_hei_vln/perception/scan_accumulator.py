"""Multi-frame LiDAR accumulation for the 2D→3D lift.

A single ``/registered_scan`` sweep is too sparse to support small
objects: their masks intersect only a handful of returns, so the lifter's
``min_inliers`` gate drops them — and the few points that *do* land are
often background bleeding through the mask edge, which mislocalises the
object by metres. This accumulator densifies the cloud the lifter sees.

Because ``/registered_scan`` is already in the **map frame**, sweeps taken
from different robot poses concatenate directly. We keep a rolling window
of the last ``max_keyframes`` sweeps — every tick contributes one, so the
window spans a fixed number of *ticks*, not a fixed amount of travel. The
merged cloud is voxel-downsampled to cap the point count (projection cost)
and to collapse the overlap between sweeps of the same surface.

Note the consequence of a tick-keyed window: a robot that stops moving
refills the buffer with copies of one sweep, so ``max_keyframes`` ticks of
standing still discard the accumulated coverage. At the 2 Hz live tick that
is 5 s. See TASK 24 for the measurement.

The lifter still transforms the cloud with the *current* pose and applies
the mask + z-buffer, so points from earlier viewpoints that no longer sit
along a current bearing are filtered out naturally — accumulation only
adds density where the current view and a past view overlap, which is
exactly where a small object needs more support.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from xiao_hei_vln.messages.common import Quaternion, Vector3

DEFAULT_MAX_KEYFRAMES: int = 10
DEFAULT_VOXEL_M: float = 0.05


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
    """Rolling buffer of the last ``max_keyframes`` map-frame LiDAR sweeps.

    Call :meth:`update` once per tick with the current registered scan; it
    returns the densified, voxel-downsampled cloud to lift against.
    """

    def __init__(
        self,
        *,
        max_keyframes: int = DEFAULT_MAX_KEYFRAMES,
        voxel_m: float = DEFAULT_VOXEL_M,
    ) -> None:
        self._max_keyframes = int(max_keyframes)
        self._voxel_m = float(voxel_m)
        self._keyframes: deque[np.ndarray] = deque(maxlen=self._max_keyframes)
        self._cache: np.ndarray = np.empty((0, 3), dtype=np.float64)

    @property
    def n_keyframes(self) -> int:
        return len(self._keyframes)

    def reset(self) -> None:
        """Drop all accumulated sweeps (e.g. on a new scene/session)."""
        self._keyframes.clear()
        self._cache = np.empty((0, 3), dtype=np.float64)

    def update(
        self,
        scan_points_map: np.ndarray,
        pose_position: Vector3 | None = None,      # noqa: ARG002 — see below
        pose_orientation: Quaternion | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """Ingest the current sweep; return the densified map-frame cloud.

        Every call commits a keyframe and rebuilds the merged cloud, so the
        cost is paid on every tick — including ones where the robot has not
        moved and the sweep is a near-duplicate.

        The pose arguments are accepted but unused: the buffer is keyed on
        ticks, not on where the robot was. They are kept in the signature
        because ``/registered_scan`` is only concatenable *because* it is
        already map-frame, and because keying eviction on travel instead
        would need exactly this data back.
        """
        if scan_points_map.ndim != 2 or scan_points_map.shape[1] < 3:
            raise ValueError(
                f"scan_points_map must be (N, >=3); got {scan_points_map.shape}",
            )
        pts = scan_points_map[:, :3].astype(np.float64, copy=False)
        self._keyframes.append(pts)
        self._cache = voxel_downsample(np.vstack(self._keyframes), self._voxel_m)
        return self._cache
