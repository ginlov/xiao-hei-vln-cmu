"""Thread-safe latest-value cache that feeds `VLMInput` snapshots."""

from __future__ import annotations

import threading

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap


class LatestCache:
    """One slot per logical channel; writers overwrite, readers snapshot atomically.

    Designed for the latest-cache + VLM tick pattern documented in
    `docs/task1_io_spec.md`. Callbacks from many ROS subscribers can
    write concurrently; the VLM main loop calls `snapshot()` at its
    own cadence to produce a `VLMInput`.

    The question channel is *sticky* — once set it is returned in every
    subsequent snapshot until `clear_question()` is called.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
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

    def put_question(self, msg: ChallengeQuestion) -> None:
        with self._lock:
            self._question = msg

    def clear_question(self) -> None:
        with self._lock:
            self._question = None

    # --- reader ------------------------------------------------------------------

    def snapshot(self, tick_id: int, tick_time: Stamp) -> VLMInput:
        """Atomically read every slot into a single `VLMInput`."""
        with self._lock:
            return VLMInput(
                tick_id=tick_id,
                tick_time=tick_time,
                image=self._image,
                registered_scan=self._registered_scan,
                sensor_scan=self._sensor_scan,
                terrain_local=self._terrain_local,
                terrain_ext=self._terrain_ext,
                pose=self._pose,
                question=self._question,
            )
