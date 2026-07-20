"""Parse the `object_list` field in VLA-3D JSONL entries.

Each line has the format produced by `vla3d_loader.render_object`:
    id cx cy cz lx ly lz heading "label"

We build a dict keyed by object_id so downstream code can look up
center/size for any object in O(1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from xiao_hei_vln.messages.common import Vector3


@dataclass(frozen=True)
class ObjectEntry:
    object_id: int
    label: str
    center: Vector3
    size: Vector3
    heading: float
    # Optional natural-language colour (e.g. "red"), carried as a second
    # quoted token after the label:  ... heading "label" "color". None when
    # the line has no colour token (all pre-colour object_lists).
    color: str | None = None


def parse_object_list(lines: list[str]) -> dict[int, ObjectEntry]:
    """Parse a list of object_list strings into a dict keyed by object_id."""
    result: dict[int, ObjectEntry] = {}
    for line in lines:
        if not isinstance(line, str):
            continue
        entry = _parse_line(line.strip())
        if entry is not None:
            result[entry.object_id] = entry
    return result


def _parse_line(line: str) -> ObjectEntry | None:
    if not line:
        return None
    # Split out the quoted tokens first so their spaces don't confuse the
    # numeric split. The 1st quoted group is the label; an optional 2nd is
    # the colour (`... heading "label" "color"`).
    quoted = list(re.finditer(r'"([^"]*)"', line))
    label = quoted[0].group(1) if quoted else ""
    color = quoted[1].group(1) if len(quoted) > 1 else None
    color = color or None  # treat an empty "" colour token as absent
    numeric_part = line[: quoted[0].start()].strip() if quoted else line

    parts = numeric_part.split()
    if len(parts) < 8:
        return None
    try:
        obj_id = int(parts[0])
        cx, cy, cz = float(parts[1]), float(parts[2]), float(parts[3])
        lx, ly, lz = float(parts[4]), float(parts[5]), float(parts[6])
        heading = float(parts[7])
    except ValueError:
        return None

    return ObjectEntry(
        object_id=obj_id,
        label=label,
        center=Vector3(x=cx, y=cy, z=cz),
        size=Vector3(x=lx, y=ly, z=lz),
        heading=heading,
        color=color,
    )
