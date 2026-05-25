"""Parsers for the two dummy-VLM fixture files shipped with the challenge.

Both files are ASCII and tiny. We deliberately avoid pulling in
`plyfile` or `pandas` — the formats are simple and stable.
"""

from __future__ import annotations

from pathlib import Path

from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.messages.outputs import ObjectReferenceResponse, Waypoint


def read_waypoints_ply(path: str | Path) -> list[Waypoint]:
    """Parse the dummy waypoint PLY (header + `x y heading` per vertex).

    Matches the format produced by the challenge `dummy_vlm/data/waypoints.ply`.
    Properties beyond the first three columns are ignored.
    """
    lines = Path(path).read_text().splitlines()
    vertex_count: int | None = None
    header_end = 0
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line == "end_header":
            header_end = i + 1
            break
    if vertex_count is None:
        raise ValueError(f"{path}: missing 'element vertex' line")

    body = lines[header_end : header_end + vertex_count]
    if len(body) != vertex_count:
        raise ValueError(
            f"{path}: header promised {vertex_count} vertices, body has {len(body)}",
        )

    waypoints: list[Waypoint] = []
    for raw in body:
        parts = raw.split()
        if len(parts) < 3:
            raise ValueError(f"{path}: malformed vertex line {raw!r}")
        waypoints.append(
            Waypoint(x=float(parts[0]), y=float(parts[1]), heading=float(parts[2])),
        )
    return waypoints


def read_object_list(path: str | Path) -> ObjectReferenceResponse:
    """Parse the dummy object list (single-line, label may contain spaces).

    Layout: `id x y z L W H heading "label with possible spaces"`.
    """
    text = Path(path).read_text().strip()
    # Split label out first so its internal spaces don't confuse the rest.
    if text.count('"') < 2:
        raise ValueError(f"{path}: expected a quoted label")
    pre, label, _ = text.split('"', 2)
    head = pre.split()
    if len(head) != 8:
        raise ValueError(
            f"{path}: expected 8 numeric fields before label, got {len(head)}",
        )
    obj_id, mid_x, mid_y, mid_z, length, width, height, heading = head
    return ObjectReferenceResponse(
        label=label,
        object_id=int(obj_id),
        center=Vector3(x=float(mid_x), y=float(mid_y), z=float(mid_z)),
        size=Vector3(x=float(length), y=float(width), z=float(height)),
        heading=float(heading),
    )
