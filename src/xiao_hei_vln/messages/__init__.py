"""Pydantic data classes that define the VLM I/O contract."""

from xiao_hei_vln.messages.common import Header, Quaternion, Stamp, Vector3
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    VLMOutput,
    Waypoint,
    WaypointPathResponse,
    parse_vlm_output,
)
from xiao_hei_vln.messages.question import ChallengeQuestion, QuestionType, classify_question
from xiao_hei_vln.messages.sensors import ImageFrame, LidarScan, OdomPose, TerrainMap

__all__ = [
    "ChallengeQuestion",
    "Header",
    "ImageFrame",
    "LidarScan",
    "NumericalResponse",
    "ObjectReferenceResponse",
    "OdomPose",
    "QuestionType",
    "Quaternion",
    "Stamp",
    "TerrainMap",
    "VLMInput",
    "VLMOutput",
    "Vector3",
    "Waypoint",
    "WaypointPathResponse",
    "classify_question",
    "parse_vlm_output",
]
