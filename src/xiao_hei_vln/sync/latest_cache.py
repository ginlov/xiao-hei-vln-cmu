"""Thread-safe latest-value cache that feeds `VLMInput` snapshots."""

from __future__ import annotations

import math
import threading
from collections import deque

from xiao_hei_vln.messages.common import Quaternion, Stamp, Vector3
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap

# /state_estimation runs at ~200 Hz, the camera at ~4 Hz. Two seconds of pose
# history is therefore ~400 samples and covers any plausible image lag, with
# room to spare for a stalled tick.
DEFAULT_POSE_HISTORY: int = 512


class LatestCache:
    """One slot per logical channel; writers overwrite, readers snapshot atomically.

    Designed for the latest-cache + VLM tick pattern documented in
    `docs/task1_io_spec.md`. Callbacks from many ROS subscribers can
    write concurrently; the VLM main loop calls `snapshot()` at its
    own cadence to produce a `VLMInput`.

    The question channel is *sticky* — once set it is returned in every
    subsequent snapshot until `clear_question()` is called.
    """

    def __init__(self, *, pose_history: int = DEFAULT_POSE_HISTORY) -> None:
        self._lock = threading.Lock()
        # Rolling pose history, so a snapshot can hand out the pose as it was
        # *when the camera frame was taken* rather than the newest one. The
        # camera lags the odometry by a frame period or more, and lifting a
        # mask with a pose from a different instant translates directly into
        # 3D centre error.
        self._pose_history: deque[OdomPose] = deque(maxlen=max(2, int(pose_history)))
        self._image: ImageFrame | None = None
        self._registered_scan: LidarScan | None = None
        self._sensor_scan: LidarScan | None = None
        self._terrain_local: TerrainMap | None = None
        self._terrain_ext: TerrainMap | None = None
        self._pose: OdomPose | None = None
        self._question: ChallengeQuestion | None = None

    # --- writers -----------------------------------------------------------------

    def put_image(self, msg: ImageFrame) -> None:
        with self._lock:
            self._image = msg

    def put_registered_scan(self, msg: LidarScan) -> None:
        if msg.source != "registered":
            raise ValueError("registered_scan slot requires LidarScan(source='registered')")
        with self._lock:
            self._registered_scan = msg

    def put_sensor_scan(self, msg: LidarScan) -> None:
        if msg.source != "sensor":
            raise ValueError("sensor_scan slot requires LidarScan(source='sensor')")
        with self._lock:
            self._sensor_scan = msg

    def put_terrain_local(self, msg: TerrainMap) -> None:
        if msg.range != "local_5m":
            raise ValueError("terrain_local slot requires TerrainMap(range='local_5m')")
        with self._lock:
            self._terrain_local = msg

    def put_terrain_ext(self, msg: TerrainMap) -> None:
        if msg.range != "ext_20m":
            raise ValueError("terrain_ext slot requires TerrainMap(range='ext_20m')")
        with self._lock:
            self._terrain_ext = msg

    def put_pose(self, msg: OdomPose) -> None:
        with self._lock:
            self._pose = msg
            self._pose_history.append(msg)

    def put_question(self, msg: ChallengeQuestion) -> None:
        with self._lock:
            self._question = msg

    def clear_question(self) -> None:
        with self._lock:
            self._question = None

    # --- reader ------------------------------------------------------------------

    def snapshot(self, tick_id: int, tick_time: Stamp) -> VLMInput:
        """Atomically read every slot into a single `VLMInput`.

        ``pose`` is the newest odometry (what the explorer wants: where the
        robot is *now*). ``image_pose`` is that same odometry rewound to the
        camera frame's own stamp — what the 2D→3D lift wants, since the mask
        was drawn from the world as it looked at that instant.
        """
        with self._lock:
            image = self._image
            return VLMInput(
                tick_id=tick_id,
                tick_time=tick_time,
                image=image,
                registered_scan=self._registered_scan,
                sensor_scan=self._sensor_scan,
                terrain_local=self._terrain_local,
                terrain_ext=self._terrain_ext,
                pose=self._pose,
                image_pose=(
                    self._pose_at(image.header.stamp.to_seconds())
                    if image is not None else None
                ),
                question=self._question,
            )

    def pose_at(self, t: float) -> OdomPose | None:
        """Pose interpolated at time ``t`` (seconds), or ``None`` if no history."""
        with self._lock:
            return self._pose_at(t)

    # --- internals ----------------------------------------------------------------

    def _pose_at(self, t: float) -> OdomPose | None:
        """Interpolate the pose history at ``t``. Caller holds the lock.

        Outside the buffered window the nearest end is returned rather than
        extrapolated — a pose older than our history is a stall, and inventing
        motion for it would be worse than admitting we only know the oldest
        sample.
        """
        hist = self._pose_history
        if not hist:
            return None
        if len(hist) == 1:
            return hist[0]

        stamps = [p.header.stamp.to_seconds() for p in hist]
        if t <= stamps[0]:
            return hist[0]
        if t >= stamps[-1]:
            return hist[-1]

        # History is append-ordered by arrival; a plain scan from the newest
        # end finds the bracketing pair in a handful of steps for a fresh
        # image, which is the common case.
        hi = len(hist) - 1
        while hi > 0 and stamps[hi - 1] > t:
            hi -= 1
        lo = hi - 1
        span = stamps[hi] - stamps[lo]
        if span <= 0.0:
            return hist[hi]
        alpha = (t - stamps[lo]) / span
        return _lerp_pose(hist[lo], hist[hi], alpha)


# ---------------------------------------------------------------------------
# Pose interpolation
# ---------------------------------------------------------------------------


def _lerp_pose(a: OdomPose, b: OdomPose, alpha: float) -> OdomPose:
    """Blend two odometry samples. Position lerps, orientation nlerps.

    At 200 Hz the two samples are ~5 ms apart, so the rotation between them is
    tiny and normalised-lerp is indistinguishable from slerp while staying
    dependency-free. Velocities are taken from the nearer sample rather than
    blended — they are consumed as "how fast are we moving right now" gates,
    not integrated.
    """
    t = min(1.0, max(0.0, alpha))
    near = b if t >= 0.5 else a
    return OdomPose(
        header=near.header,
        child_frame_id=near.child_frame_id,
        position=Vector3(
            x=a.position.x + (b.position.x - a.position.x) * t,
            y=a.position.y + (b.position.y - a.position.y) * t,
            z=a.position.z + (b.position.z - a.position.z) * t,
        ),
        orientation=_nlerp(a.orientation, b.orientation, t),
        linear_velocity=near.linear_velocity,
        angular_velocity=near.angular_velocity,
    )


def _nlerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Normalised linear quaternion blend, taking the shorter arc."""
    # q and -q are the same rotation; without this the blend can swing the
    # long way round and produce a wildly wrong heading.
    dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    bx, by, bz, bw = (b.x, b.y, b.z, b.w) if dot >= 0.0 else (-b.x, -b.y, -b.z, -b.w)
    x = a.x + (bx - a.x) * t
    y = a.y + (by - a.y) * t
    z = a.z + (bz - a.z) * t
    w = a.w + (bw - a.w) * t
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return a
    return Quaternion(x=x / norm, y=y / norm, z=z / norm, w=w / norm)
