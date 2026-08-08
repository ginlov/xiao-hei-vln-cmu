"""Thread-safe latest-value cache that feeds `VLMInput` snapshots."""

from __future__ import annotations

import threading
from collections import deque

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap

# How many past pose/scan messages to keep for timestamp matching. At the sim's
# ~5-10 Hz odometry this is a few seconds of history — far more than the
# sub-second skew we align across, with headroom for a slow tick.
_HISTORY = 128

# If the nearest history entry is further than this from the image stamp, the
# streams are not merely skewed — there is a gap (a dropped topic, a stall). In
# that case matching would pair the image with an unrelated pose, so fall back
# to the latest value instead, which is the old behaviour.
_MATCH_WINDOW_S = 1.0


def _nearest(history: deque, t: float | None):
    """The history entry whose stamp is closest to ``t``.

    ``None`` when empty. With no anchor time, or when the closest entry is
    still outside :data:`_MATCH_WINDOW_S`, returns the most recent entry — the
    pre-matching behaviour, which is the safe default when alignment is
    impossible.
    """
    if not history:
        return None
    if t is None:
        return history[-1]
    best = min(history, key=lambda m: abs(m.header.stamp.to_seconds() - t))
    if abs(best.header.stamp.to_seconds() - t) > _MATCH_WINDOW_S:
        return history[-1]
    return best


class LatestCache:
    """One slot per logical channel; writers overwrite, readers snapshot atomically.

    Designed for the latest-cache + VLM tick pattern documented in
    `docs/task1_io_spec.md`. Callbacks from many ROS subscribers can
    write concurrently; the VLM main loop calls `snapshot()` at its
    own cadence to produce a `VLMInput`.

    **Timestamp matching.** The camera stream is the slow one, so at tick time
    the newest image is older than the newest pose. Pairing the two — which a
    plain latest-value cache does — lifts every mask against a pose the robot
    has already turned past, misplacing detections by up to ~17° while turning
    (measured in TASK 27). To avoid that, ``pose`` and ``registered_scan`` are
    kept as short histories and, at snapshot time, the entries whose stamps are
    **nearest the image's stamp** are returned. The image is the anchor because
    it is the laggard; when there is no image, or the streams have a gap wider
    than :data:`_MATCH_WINDOW_S`, the cache falls back to latest-value.

    The question channel is *sticky* — once set it is returned in every
    subsequent snapshot until `clear_question()` is called.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image: ImageFrame | None = None
        # Histories, newest last — matched to the image stamp in snapshot().
        # These are the two inputs the 3D lift consumes, and the only ones
        # whose misalignment with the image is known to hurt (TASK 27).
        self._pose_hist: deque[OdomPose] = deque(maxlen=_HISTORY)
        self._rscan_hist: deque[LidarScan] = deque(maxlen=_HISTORY)
        # Not lift inputs; kept latest-only. Sensor scan and terrain feed
        # navigation, which is coarse against a sub-second skew.
        self._sensor_scan: LidarScan | None = None
        self._terrain_local: TerrainMap | None = None
        self._terrain_ext: TerrainMap | None = None
        self._question: ChallengeQuestion | None = None

    # --- writers -----------------------------------------------------------------

    def put_image(self, msg: ImageFrame) -> None:
        with self._lock:
            self._image = msg

    def put_registered_scan(self, msg: LidarScan) -> None:
        if msg.source != "registered":
            raise ValueError("registered_scan slot requires LidarScan(source='registered')")
        with self._lock:
            self._rscan_hist.append(msg)

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
            self._pose_hist.append(msg)

    def put_question(self, msg: ChallengeQuestion) -> None:
        with self._lock:
            self._question = msg

    def clear_question(self) -> None:
        with self._lock:
            self._question = None

    # --- reader ------------------------------------------------------------------

    def snapshot(self, tick_id: int, tick_time: Stamp) -> VLMInput:
        """Atomically read every slot into a single `VLMInput`.

        ``pose`` and ``registered_scan`` are matched to the image's stamp (see
        the class docstring); the rest are latest-value.
        """
        with self._lock:
            image = self._image
            t_img = image.header.stamp.to_seconds() if image is not None else None
            return VLMInput(
                tick_id=tick_id,
                tick_time=tick_time,
                image=image,
                registered_scan=_nearest(self._rscan_hist, t_img),
                sensor_scan=self._sensor_scan,
                terrain_local=self._terrain_local,
                terrain_ext=self._terrain_ext,
                pose=_nearest(self._pose_hist, t_img),
                question=self._question,
            )
