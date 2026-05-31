"""`VLMOutput` — discriminated union of the three valid response shapes."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from xiao_hei_vln.messages.common import Vector3


class NumericalResponse(BaseModel):
    """Answer to a `how many ...` question, published on `/numerical_response`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["numerical"] = "numerical"
    value: int
    # Free-text "why this answer" / running tally. Optional, not published
    # to ROS — used by the responder to thread context across ticks (see
    # `docs/task3_phase3_prompt.md`).
    rationale: str | None = None


class ObjectReferenceResponse(BaseModel):
    """Selected-object bounding box, published on `/selected_object_marker`.

    `center`/`size` are in the `map` frame; `heading` is the yaw (rad)
    of the object's local frame around +Z.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["object_reference"] = "object_reference"
    label: str
    object_id: int
    center: Vector3
    size: Vector3
    heading: float = 0.0
    rationale: str | None = None


class Waypoint(BaseModel):
    """A single 2D waypoint in the `map` frame."""

    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    heading: float = 0.0  # theta in geometry_msgs/Pose2D; currently ignored by the system


class WaypointPathResponse(BaseModel):
    """Sequence of waypoints, published one-by-one on `/way_point_with_heading`."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["waypoint_path"] = "waypoint_path"
    waypoints: list[Waypoint] = Field(min_length=1)
    rationale: str | None = None


VLMOutput = Annotated[
    NumericalResponse | ObjectReferenceResponse | WaypointPathResponse,
    Field(discriminator="kind"),
]

_OUTPUT_ADAPTER: TypeAdapter[VLMOutput] = TypeAdapter(VLMOutput)


def parse_vlm_output(data: Any) -> VLMOutput:
    """Validate arbitrary input (dict, JSON-decoded payload, ...) as a `VLMOutput`."""
    return _OUTPUT_ADAPTER.validate_python(data)
