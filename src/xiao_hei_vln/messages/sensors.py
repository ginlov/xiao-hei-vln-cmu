"""Sensor messages exposed to the VLM by the challenge platform.

Field rates and conventions mirror what we observed from the official
simulator (see `docs/task1_phase1_measurements.md`). The contract is
plain Python: anything that accepts a numpy array can be tested without
ROS.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from xiao_hei_vln.messages.common import Header, Quaternion, Vector3


class ImageFrame(BaseModel):
    """`sensor_msgs/Image` consumed from `/camera/image` (frame_id="camera").

    Raw bytes are stored as-is; conversion to an ndarray is left to the
    consumer to avoid forcing a numpy dependency on every read.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    header: Header
    width: int = Field(gt=0, default=1920)
    height: int = Field(gt=0, default=640)
    encoding: str = "bgr8"
    step: int = Field(gt=0)
    data: bytes

    @field_validator("data")
    @classmethod
    def _data_matches_step(cls, v: bytes, info) -> bytes:
        # Only validate when both step and height are concrete; pydantic
        # passes already-validated fields via info.data.
        step = info.data.get("step")
        height = info.data.get("height")
        if step is not None and height is not None and len(v) != step * height:
            raise ValueError(
                f"data length {len(v)} does not match step*height {step * height}",
            )
        return v


class LidarScan(BaseModel):
    """`sensor_msgs/PointCloud2` from `/registered_scan` or `/sensor_scan`.

    Points are flattened into an (N, 4) ndarray of (x, y, z, intensity).
    `/sensor_scan` has no intensity field on the wire, so the adapter
    fills the trailing column with zeros.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    header: Header
    points: np.ndarray
    source: Literal["registered", "sensor"]

    @field_validator("points")
    @classmethod
    def _points_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 4:
            raise ValueError(f"points must be (N, 4); got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32, copy=False)
        return v


class TerrainMap(BaseModel):
    """`sensor_msgs/PointCloud2` from `/terrain_map` (5 m) or `/terrain_map_ext` (20 m).

    Points are (x, y, z, cost) — the `intensity` PointField encodes the
    traversability cost emitted by the terrain analysis module.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    header: Header
    points: np.ndarray
    range: Literal["local_5m", "ext_20m"]

    @field_validator("points")
    @classmethod
    def _points_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 4:
            raise ValueError(f"points must be (N, 4); got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32, copy=False)
        return v


class OdomPose(BaseModel):
    """`nav_msgs/Odometry` from `/state_estimation` (map → sensor)."""

    model_config = ConfigDict(frozen=True)

    header: Header
    child_frame_id: str = "sensor"
    position: Vector3
    orientation: Quaternion
    linear_velocity: Vector3 | None = None
    angular_velocity: Vector3 | None = None
