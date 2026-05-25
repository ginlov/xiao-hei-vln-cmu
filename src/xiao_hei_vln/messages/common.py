"""Primitive geometry types mirroring the ROS std/geometry messages we consume."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Stamp(BaseModel):
    """ROS time stamp, split into seconds and nanoseconds."""

    model_config = ConfigDict(frozen=True)

    sec: int = Field(ge=0)
    nanosec: int = Field(ge=0, lt=1_000_000_000)

    def to_seconds(self) -> float:
        return self.sec + self.nanosec * 1e-9

    @classmethod
    def from_seconds(cls, t: float) -> Stamp:
        sec = int(t)
        nanosec = int(round((t - sec) * 1e9))
        if nanosec == 1_000_000_000:
            sec += 1
            nanosec = 0
        return cls(sec=sec, nanosec=nanosec)


class Header(BaseModel):
    """Mirror of std_msgs/Header (stamp + frame_id)."""

    model_config = ConfigDict(frozen=True)

    stamp: Stamp
    frame_id: str


class Vector3(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float


class Quaternion(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    z: float
    w: float
