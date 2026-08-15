"""Periodic exploration-artefact snapshots.

The end-of-sweep save only fires in tick()'s DONE branch, so a run that is
torn down early used to leave no PNG at all. These cover the pieces that make
a mid-run save safe: the write-then-rename publish, and the fact that both
shipped strategies expose what the plot needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xiao_hei_vln.app import main as app_main


class _FakeLogger:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.debug_lines: list[str] = []
        self.warn_lines: list[str] = []

    def info(self, msg: str) -> None:
        self.info_lines.append(msg)

    def debug(self, msg: str) -> None:
        self.debug_lines.append(msg)

    def warn(self, msg: str) -> None:
        self.warn_lines.append(msg)


class _FakeNode:
    def __init__(self) -> None:
        self._logger = _FakeLogger()

    def get_logger(self) -> _FakeLogger:
        return self._logger


def test_exploration_dir_separates_scene_and_strategy(monkeypatch, tmp_path) -> None:
    """Two strategies on one scene must not share a directory: exploration.log
    opens with mode "w", so a collision deletes the earlier run's evidence."""
    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(app_main, "_EXPLORATION_SCENE", "chinese_room")

    monkeypatch.setattr(app_main, "_EXPLORATION_RUN_LABEL", "frontier")
    frontier = app_main._exploration_dir()
    monkeypatch.setattr(app_main, "_EXPLORATION_RUN_LABEL", "nbv")
    nbv = app_main._exploration_dir()

    assert frontier == tmp_path / "chinese_room" / "frontier"
    assert nbv == tmp_path / "chinese_room" / "nbv"
    assert frontier != nbv


def test_exploration_dir_falls_back_to_default_scene(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(app_main, "_EXPLORATION_SCENE", "default_scene")
    monkeypatch.setattr(app_main, "_EXPLORATION_RUN_LABEL", "frontier")

    assert app_main._exploration_dir() == tmp_path / "default_scene" / "frontier"


def test_disabled_exploration_does_not_claim_a_strategy_dir() -> None:
    """MAX_WAYPOINTS=0 still leaves STRATEGY at its default, so without the
    guard a no-exploration run would truncate frontier/exploration.log."""
    import importlib

    import xiao_hei_vln.app.main as fresh

    for waypoints, expected in (("0", "no_exploration"), ("100", "nbv")):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("XIAO_HEI_EXPLORATION_MAX_WAYPOINTS", waypoints)
            mp.setenv("XIAO_HEI_EXPLORATION_STRATEGY", "nbv")
            importlib.reload(fresh)
            label = fresh._EXPLORATION_RUN_LABEL
            assert label == expected

    importlib.reload(fresh)  # restore module state for the rest of the session


def test_default_time_budget_is_eight_minutes() -> None:
    assert app_main._EXPLORATION_MAX_SECONDS == 480.0


def test_budget_not_expired_before_the_deadline() -> None:
    assert not app_main._budget_expired(1000.0, 1000.0 + 479.9, 480.0)


def test_budget_expires_at_the_deadline() -> None:
    """`>=`, not `>`: at a 1 Hz tick the sample can land exactly on it."""
    assert app_main._budget_expired(1000.0, 1000.0 + 480.0, 480.0)
    assert app_main._budget_expired(1000.0, 1000.0 + 600.0, 480.0)


def test_budget_never_expires_before_the_clock_starts() -> None:
    """/state_estimation takes 90-190 s to arrive; that dead time is not
    exploration, and charging it to the budget would silently shorten it."""
    assert not app_main._budget_expired(None, 1_000_000.0, 480.0)


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_zero_budget_disables_the_cutoff(budget: float) -> None:
    assert not app_main._budget_expired(1000.0, 1000.0 + 99_999.0, budget)


def test_time_budget_is_env_tunable_and_disablable() -> None:
    import importlib

    import xiao_hei_vln.app.main as fresh

    for value, expected in (("120", 120.0), ("0", 0.0)):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("XIAO_HEI_EXPLORATION_MAX_SECONDS", value)
            importlib.reload(fresh)
            budget = fresh._EXPLORATION_MAX_SECONDS
            assert budget == expected

    importlib.reload(fresh)


def test_publish_atomically_renames_over_target(tmp_path: Path) -> None:
    out = tmp_path / "exploration.png"
    out.write_bytes(b"stale")

    app_main._publish_atomically(out, lambda tmp: tmp.write_bytes(b"fresh"))

    assert out.read_bytes() == b"fresh"
    # No temp file left behind for the next reader to trip over.
    assert list(tmp_path.iterdir()) == [out]


def test_publish_atomically_keeps_png_suffix(tmp_path: Path) -> None:
    """matplotlib and pillow both pick the format off the extension, so the
    temp file must stay a .png or the save raises instead of writing."""
    seen: list[Path] = []

    app_main._publish_atomically(
        tmp_path / "rviz.png", lambda tmp: (seen.append(tmp), tmp.write_bytes(b"x"))[1]
    )

    assert seen[0].suffix == ".png"


def test_publish_atomically_leaves_target_intact_when_write_fails(tmp_path: Path) -> None:
    out = tmp_path / "exploration.png"
    out.write_bytes(b"last good snapshot")

    def _boom(tmp: Path) -> None:
        raise RuntimeError("grid vanished mid-plot")

    with pytest.raises(RuntimeError):
        app_main._publish_atomically(out, _boom)

    assert out.read_bytes() == b"last good snapshot"


def test_save_png_skips_strategy_without_grid_accessors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    node = _FakeNode()

    app_main._maybe_save_png(object(), node)

    assert any("does not support" in line for line in node.get_logger().info_lines)


def test_save_png_quiet_does_not_log_at_info(monkeypatch, tmp_path) -> None:
    """A 30 s timer must not narrate itself into the run log."""
    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    node = _FakeNode()

    app_main._maybe_save_png(object(), node, quiet=True)

    assert node.get_logger().info_lines == []


def test_save_rviz_quiet_does_not_log_when_headless(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("DISPLAY", raising=False)
    node = _FakeNode()

    app_main._maybe_save_rviz(node, quiet=True)

    assert node.get_logger().info_lines == []


@pytest.mark.parametrize("strategy", ["frontier", "nbv"])
def test_both_strategies_expose_what_the_plot_needs(strategy: str) -> None:
    """_maybe_save_png silently no-ops on a strategy missing these, which
    would make periodic snapshots vanish for nbv without any error."""
    from xiao_hei_vln.exploration import FrontierExplorer, NextBestViewExplorer

    explorer = FrontierExplorer() if strategy == "frontier" else NextBestViewExplorer()

    assert hasattr(explorer, "get_visited_waypoints")
    assert hasattr(explorer, "get_grid")
    assert explorer.get_visited_waypoints() == []
    assert explorer.get_grid() is not None


@pytest.mark.parametrize("strategy", ["frontier", "nbv"])
def test_plot_renders_mid_sweep_for_both_strategies(tmp_path: Path, strategy: str) -> None:
    """A snapshot is taken while the sweep is still running, so the plot has
    to cope with a partially-filled grid and an in-progress target."""
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib  # imported for the skip condition only

    from xiao_hei_vln.exploration import (
        FrontierExplorer,
        NextBestViewExplorer,
        save_exploration_plot,
    )
    from xiao_hei_vln.messages.outputs import Waypoint

    explorer = FrontierExplorer() if strategy == "frontier" else NextBestViewExplorer()
    explorer._visited = [Waypoint(x=0.0, y=0.0, heading=0.0), Waypoint(x=1.0, y=0.5, heading=0.3)]
    explorer._current_target = Waypoint(x=2.0, y=1.0, heading=0.1)
    explorer._grid._free.update({(0, 0), (1, 0), (1, 1), (2, 1)})

    out = tmp_path / "exploration.png"
    save_exploration_plot(explorer.get_visited_waypoints(), explorer.get_grid(), out)

    assert out.stat().st_size > 0


@pytest.mark.parametrize("strategy", ["frontier", "nbv"])
def test_maybe_save_png_writes_through_the_real_path(monkeypatch, tmp_path, strategy) -> None:
    """End-to-end over the helper the timer actually calls: the PNG lands at
    the published name, and no `.partial` file is left behind."""
    pytest.importorskip("matplotlib")
    from xiao_hei_vln.exploration import FrontierExplorer, NextBestViewExplorer
    from xiao_hei_vln.messages.outputs import Waypoint

    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(app_main, "_EXPLORATION_SCENE", "chinese_room")
    monkeypatch.setattr(app_main, "_EXPLORATION_RUN_LABEL", strategy)

    explorer = FrontierExplorer() if strategy == "frontier" else NextBestViewExplorer()
    explorer._visited = [Waypoint(x=0.0, y=0.0, heading=0.0)]
    explorer._grid._free.update({(0, 0), (1, 0)})
    node = _FakeNode()

    app_main._maybe_save_png(explorer, node, quiet=True)

    out = tmp_path / "chinese_room" / strategy / "exploration.png"
    assert out.stat().st_size > 0
    assert [p.name for p in out.parent.iterdir()] == ["exploration.png"]
    assert node.get_logger().warn_lines == []


def test_repeated_snapshots_overwrite_in_place(monkeypatch, tmp_path) -> None:
    """Successive snapshots must land on the same path — the point is that the
    file on disk is always the newest view, not a growing pile of images."""
    pytest.importorskip("matplotlib")
    from xiao_hei_vln.exploration import FrontierExplorer
    from xiao_hei_vln.messages.outputs import Waypoint

    monkeypatch.setattr(app_main, "_EXPLORATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(app_main, "_EXPLORATION_SCENE", "chinese_room")
    monkeypatch.setattr(app_main, "_EXPLORATION_RUN_LABEL", "frontier")

    explorer = FrontierExplorer()
    explorer._grid._free.update({(0, 0)})
    node = _FakeNode()
    app_main._maybe_save_png(explorer, node, quiet=True)
    first = (tmp_path / "chinese_room" / "frontier" / "exploration.png").read_bytes()

    # The sweep progresses: more map, more waypoints.
    explorer._visited = [Waypoint(x=float(i), y=0.0, heading=0.0) for i in range(4)]
    explorer._grid._free.update({(i, 0) for i in range(20)})
    app_main._maybe_save_png(explorer, node, quiet=True)

    out_dir = tmp_path / "chinese_room" / "frontier"
    assert [p.name for p in out_dir.iterdir()] == ["exploration.png"]
    assert (out_dir / "exploration.png").read_bytes() != first


def test_empty_grid_snapshot_does_not_raise(tmp_path: Path) -> None:
    """The first timer firing can land before any terrain has arrived."""
    pytest.importorskip("matplotlib")
    from xiao_hei_vln.exploration import FrontierExplorer, save_exploration_plot

    explorer = FrontierExplorer()
    out = tmp_path / "exploration.png"

    save_exploration_plot(explorer.get_visited_waypoints(), explorer.get_grid(), out)

    assert out.stat().st_size > 0
