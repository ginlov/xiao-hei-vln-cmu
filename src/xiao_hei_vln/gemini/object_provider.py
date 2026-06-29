"""File-backed access to scene object entries for Gemini prompting."""

from __future__ import annotations

from pathlib import Path

from xiao_hei_vln.eval_sampler.object_list import ObjectEntry, parse_object_list


class ObjectEntryProvider:
    """Loads ``object_list.txt`` records and refreshes when the file changes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._mtime_ns: int | None = None
        self._objects: dict[int, ObjectEntry] = {}

    @property
    def path(self) -> Path:
        return self._path

    def get(self) -> dict[int, ObjectEntry]:
        """Return the latest objects, re-reading the file if it changed."""
        stat = self._path.stat()
        if self._mtime_ns != stat.st_mtime_ns:
            lines = self._path.read_text().splitlines()
            self._objects = parse_object_list(lines)
            self._mtime_ns = stat.st_mtime_ns
        return dict(self._objects)


def object_entries_to_markers(objects: dict[int, ObjectEntry]) -> list[dict]:
    """Serialize ObjectEntry records into the JSON marker shape Gemini sees."""
    markers: list[dict] = []
    for object_id in sorted(objects):
        obj = objects[object_id]
        markers.append(
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "center": {
                    "x": obj.center.x,
                    "y": obj.center.y,
                    "z": obj.center.z,
                },
                "size": {
                    "x": obj.size.x,
                    "y": obj.size.y,
                    "z": obj.size.z,
                },
                "heading": obj.heading,
                # Raw ObjectEntry records do not carry color. A future detector
                # can fill this after 3D-to-2D projection.
                "color": None,
            },
        )
    return markers
