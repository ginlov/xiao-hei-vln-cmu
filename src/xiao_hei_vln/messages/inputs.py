"""`VLMInput` — the snapshot the VLM consumes on each tick."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap


class VLMInput(BaseModel):
    """Bundle of latest sensor values at the moment of the VLM tick.

    Any of the optional fields may be `None` if the corresponding topic
    has not yet produced a message at snapshot time (cold start).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tick_id: int = Field(ge=0)
    tick_time: Stamp

    image: ImageFrame | None = None
    registered_scan: LidarScan | None = None
    sensor_scan: LidarScan | None = None
    terrain_local: TerrainMap | None = None
    terrain_ext: TerrainMap | None = None
    pose: OdomPose | None = None
    question: ChallengeQuestion | None = None

    @property
    def is_ready(self) -> bool:
        """True iff the inputs sufficient for any kind of inference are present."""
        return (
            self.image is not None
            and self.pose is not None
            and self.question is not None
        )
