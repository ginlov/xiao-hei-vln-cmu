"""Tests for the dummy-VLM fixture parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from xiao_hei_vln.dummy.fixtures import read_object_list, read_waypoints_ply

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "src" / "xiao_hei_vln" / "dummy" / "data"


class TestWaypointsPLY:
    def test_parses_bundled_fixture(self) -> None:
        wps = read_waypoints_ply(DATA_DIR / "waypoints.ply")
        assert len(wps) == 2
        assert wps[0].x == pytest.approx(7.5)
        assert wps[0].y == pytest.approx(-1.0)
        assert wps[0].heading == pytest.approx(0.0)
        assert wps[1].x == pytest.approx(5.0)
        assert wps[1].y == pytest.approx(-4.0)

    def test_rejects_count_mismatch(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.ply"
        bad.write_text(
            "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float heading\n"
            "end_header\n1 2 0\n3 4 0\n",
        )
        with pytest.raises(ValueError, match="header promised 3"):
            read_waypoints_ply(bad)

    def test_rejects_missing_header(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.ply"
        bad.write_text("ply\nformat ascii 1.0\nend_header\n")
        with pytest.raises(ValueError, match="missing 'element vertex'"):
            read_waypoints_ply(bad)


class TestObjectList:
    def test_parses_bundled_fixture(self) -> None:
        obj = read_object_list(DATA_DIR / "object_list.txt")
        assert obj.object_id == 0
        assert obj.label == "sofa"
        assert obj.center.x == pytest.approx(3.3707286769194647)
        assert obj.size.x == pytest.approx(2.8618994)
        assert obj.size.z == pytest.approx(1.0225521)

    def test_multi_word_label(self, tmp_path: Path) -> None:
        f = tmp_path / "obj.txt"
        f.write_text('7 1.0 2.0 0.5 0.5 0.5 1.0 0.0 "potted plant on counter"\n')
        obj = read_object_list(f)
        assert obj.label == "potted plant on counter"
        assert obj.object_id == 7

    def test_rejects_missing_quotes(self, tmp_path: Path) -> None:
        f = tmp_path / "obj.txt"
        f.write_text("0 1 2 3 4 5 6 0 nolabel\n")
        with pytest.raises(ValueError, match="quoted label"):
            read_object_list(f)
