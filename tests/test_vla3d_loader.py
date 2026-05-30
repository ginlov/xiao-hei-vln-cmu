"""Tests for the VLA-3D loader's text renderers.

The key guarantee here is that ``render_object`` produces a line that the
runtime parser ``xiao_hei_vln.dummy.fixtures.read_object_list`` accepts
and that round-trips field-for-field. Without this the generated training
corpus is misaligned with the runtime contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vla3d_loader import (
    VLAObject,
    VLARegion,
    VLAScene,
    render_object,
    render_object_list,
)

from xiao_hei_vln.dummy.fixtures import read_object_list


def _obj(
    *,
    id: int = 7,
    region_id: int = 1,
    raw_label: str = "chair",
    x: float = 1.0,
    y: float = 2.0,
    z: float = 0.5,
    lx: float = 0.6,
    ly: float = 0.6,
    lz: float = 1.0,
    heading: float = 0.0,
) -> VLAObject:
    return VLAObject(
        id=id,
        region_id=region_id,
        raw_label=raw_label,
        nyu40_label="chair",
        nyu40_id=5,
        x=x, y=y, z=z,
        lx=lx, ly=ly, lz=lz,
        heading=heading,
        colors=("brown", "N/A", "N/A"),
        color_percentages=(1.0, 0.0, 0.0),
    )


def _write_and_read(line: str, tmp_path: Path):
    f = tmp_path / "obj.txt"
    f.write_text(line + "\n")
    return read_object_list(f)


class TestRenderObject:
    def test_quotes_label(self) -> None:
        line = render_object(_obj(raw_label="chair"))
        assert line.count('"') == 2
        assert line.endswith('"chair"')

    def test_round_trip_single_word_label(self, tmp_path: Path) -> None:
        o = _obj(id=7, raw_label="chair", x=1.0, y=2.0, z=0.5,
                 lx=0.6, ly=0.6, lz=1.0, heading=0.0)
        parsed = _write_and_read(render_object(o), tmp_path)
        assert parsed.object_id == 7
        assert parsed.label == "chair"
        assert parsed.center.x == pytest.approx(1.0)
        assert parsed.center.y == pytest.approx(2.0)
        assert parsed.center.z == pytest.approx(0.5)
        assert parsed.size.x == pytest.approx(0.6)
        assert parsed.size.y == pytest.approx(0.6)
        assert parsed.size.z == pytest.approx(1.0)
        assert parsed.heading == pytest.approx(0.0)

    def test_round_trip_multi_word_label(self, tmp_path: Path) -> None:
        # The bare-label bug this test guards against:
        # ``read_object_list`` calls ``pre.split()`` on everything before the
        # first ``"`` and demands exactly 8 numeric tokens. A bare
        # ``potted plant`` label would push the count to 9 and raise.
        o = _obj(id=42, raw_label="potted plant")
        parsed = _write_and_read(render_object(o), tmp_path)
        assert parsed.object_id == 42
        assert parsed.label == "potted plant"

    def test_round_trip_negative_coordinates(self, tmp_path: Path) -> None:
        o = _obj(id=3, raw_label="coffee table",
                 x=-1.25, y=-3.5, z=0.0, heading=-1.5708)
        parsed = _write_and_read(render_object(o), tmp_path)
        assert parsed.object_id == 3
        assert parsed.label == "coffee table"
        assert parsed.center.x == pytest.approx(-1.25)
        assert parsed.center.y == pytest.approx(-3.5)
        assert parsed.heading == pytest.approx(-1.57, abs=1e-2)


class TestRenderObjectList:
    def _scene(self, objs: list[VLAObject]) -> VLAScene:
        return VLAScene(
            name="test_scene",
            objects=objs,
            regions={1: VLARegion(region_id=1, region_name="living_1"),
                     2: VLARegion(region_id=2, region_name="kitchen_1")},
        )

    def test_drops_unassigned_objects(self) -> None:
        sc = self._scene([
            _obj(id=1, region_id=1, raw_label="chair"),
            _obj(id=2, region_id=-1, raw_label="ghost"),
            _obj(id=3, region_id=2, raw_label="table"),
        ])
        lines = render_object_list(sc)
        assert len(lines) == 2
        assert all('"ghost"' not in ln for ln in lines)

    def test_region_filter(self) -> None:
        sc = self._scene([
            _obj(id=1, region_id=1, raw_label="chair"),
            _obj(id=2, region_id=2, raw_label="table"),
            _obj(id=3, region_id=2, raw_label="lamp"),
        ])
        lines = render_object_list(sc, region_ids={2})
        assert len(lines) == 2
        labels = {ln.rsplit('"', 2)[-2] for ln in lines}
        assert labels == {"table", "lamp"}

    def test_every_line_round_trips(self, tmp_path: Path) -> None:
        sc = self._scene([
            _obj(id=1, region_id=1, raw_label="chair"),
            _obj(id=2, region_id=1, raw_label="potted plant"),
            _obj(id=3, region_id=2, raw_label="coffee table"),
        ])
        for ln in render_object_list(sc):
            (tmp_path / "obj.txt").write_text(ln + "\n")
            read_object_list(tmp_path / "obj.txt")  # raises on failure
