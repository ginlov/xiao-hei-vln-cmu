"""Unit tests for NavVLMExplorer — the VLM waypoint-proposer strategy.

Rendering (matplotlib) is monkeypatched to dummy bytes so these exercise the
trigger state machine + snapping logic without the plotting cost. A fake
engine returns scripted proposals and records the prompts it was asked, so no
Anthropic key or network is touched.
"""

from __future__ import annotations

import time

import pytest

from xiao_hei_vln.exploration import _nav_vlm
from xiao_hei_vln.exploration._nav_vlm import NavVLMExplorer
from xiao_hei_vln.messages import Header, OdomPose, Quaternion, Stamp, Vector3
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.nav_vlm.engine import WaypointProposal


@pytest.fixture(autouse=True)
def _no_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the matplotlib render / JPEG encode in every test here."""
    monkeypatch.setattr(_nav_vlm, "render_grid_png", lambda *a, **k: b"png")
    monkeypatch.setattr(_nav_vlm, "panorama_to_jpeg", lambda *a, **k: b"jpg")


class FakeEngine:
    """Returns scripted proposals; records each call's prompt/images."""

    def __init__(self, proposals: list[WaypointProposal]) -> None:
        self._proposals = list(proposals)
        self.calls: list[dict] = []

    def propose(self, *, user_text, panorama_jpg, occupancy_png):
        self.calls.append(
            {"user_text": user_text, "has_pano": panorama_jpg is not None}
        )
        if not self._proposals:
            return WaypointProposal(done=True, rationale="exhausted")
        return self._proposals.pop(0)


def _snapshot(t: float, *, x: float = 0.2, y: float = 0.2) -> VLMInput:
    stamp = Stamp.from_seconds(t)
    pose = OdomPose(
        header=Header(stamp=stamp, frame_id="map"),
        position=Vector3(x=x, y=y, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return VLMInput(tick_id=int(t * 10), tick_time=stamp, pose=pose)


def _reseed(exp: NavVLMExplorer) -> None:
    """Clean, fully-reachable free block around the origin (resolution 1.0)."""
    exp._grid._blacklisted.clear()
    exp._grid._occupied.clear()
    exp._grid._free = {(ix, iy) for ix in range(-1, 7) for iy in range(-3, 4)}


def _make(engine: FakeEngine, **kw) -> NavVLMExplorer:
    return NavVLMExplorer(
        engine, grid_resolution=1.0, async_calls=False, **kw
    )


# --- cold start / basic proposal ------------------------------------------


def test_cold_start_triggers_and_sets_snapped_target() -> None:
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="right")])
    exp = _make(eng, max_waypoints=5)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0))

    assert eng.calls, "engine should be asked at cold start"
    assert wp is not None
    # (2.5,0.5) is the world centre of free cell (2,0) → snapped exactly there.
    assert (round(wp.x, 1), round(wp.y, 1)) == (2.5, 0.5)


def test_proposal_is_snapped_into_reachable_free_space() -> None:
    # Model proposes a far-away point; snapping must land on a reachable free
    # cell within the hop horizon, never the raw pick.
    eng = FakeEngine([WaypointProposal(done=False, x=50.0, y=50.0, rationale="far")])
    exp = _make(eng, max_waypoints=5, max_hop_m=3.0)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0))

    assert wp is not None
    cell = exp._grid.world_to_grid(wp.x, wp.y)
    assert cell in exp._grid.free_cells
    # Within the hop horizon of the robot (path cost ≤ 3 m).
    costs = exp._grid.reachable_path_costs(0.2, 0.2)
    assert costs.get(cell, 1e9) <= 3.0 + 1e-6


# --- reached → next -------------------------------------------------------


def test_reached_advances_and_requests_next() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=1.5, y=1.5, rationale="b"),
    ])
    exp = _make(eng, max_waypoints=5)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    assert exp._current_target is not None
    assert len(exp._visited) == 0

    exp.advance()  # nav stack reached it
    assert len(exp._visited) == 1
    assert exp._current_target is None

    _reseed(exp)
    exp.update(_snapshot(1.0))
    assert len(eng.calls) == 2, "reached event should trigger a fresh proposal"
    assert exp._current_target is not None


# --- cannot reach → failure feedback --------------------------------------


def test_force_skip_counts_and_feeds_failure_into_next_prompt() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=-0.5, y=-1.5, rationale="b"),
    ])
    exp = _make(eng, max_waypoints=5, max_consecutive_skips=9)
    _reseed(exp)
    exp.update(_snapshot(0.0))

    exp.force_skip()  # local planner blocked
    assert exp.skipped_count == 1
    assert exp._consecutive_skip_count == 1

    _reseed(exp)
    exp.update(_snapshot(1.0))
    assert "FAILED" in eng.calls[-1]["user_text"], "failure must reach the model"


def test_watchdog_skips_a_stuck_target() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=1.5, y=1.5, rationale="b"),
    ])
    exp = _make(eng, max_waypoints=5, stuck_timeout_s=10.0, max_consecutive_skips=9)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    assert exp._current_target is not None

    _reseed(exp)
    exp.update(_snapshot(11.0))  # overstayed the stuck horizon
    assert exp.skipped_count == 1, "watchdog should register a cannot-reach"


# --- termination conditions -----------------------------------------------


def test_model_done_completes_exploration() -> None:
    eng = FakeEngine([WaypointProposal(done=True, rationale="all explored")])
    exp = _make(eng, max_waypoints=5)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0))
    assert wp is None
    assert exp.is_complete()


def test_consecutive_skips_terminate() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=1.5, y=1.5, rationale="b"),
    ])
    exp = _make(eng, max_waypoints=9, max_consecutive_skips=2)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    exp.force_skip()            # skip 1
    _reseed(exp)
    exp.update(_snapshot(1.0))  # sets a new target
    exp.force_skip()            # skip 2 == limit
    assert exp.is_complete()


def test_budget_exhausted_completes() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=1.5, y=1.5, rationale="b"),
    ])
    exp = _make(eng, max_waypoints=2)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    exp.advance()
    _reseed(exp)
    exp.update(_snapshot(1.0))
    exp.advance()
    _reseed(exp)
    assert exp.update(_snapshot(2.0)) is None
    assert exp.is_complete()


# --- async path smoke test ------------------------------------------------


def test_async_call_holds_then_delivers() -> None:
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="a")])
    exp = NavVLMExplorer(eng, grid_resolution=1.0, async_calls=True, max_waypoints=5)
    _reseed(exp)

    first = exp.update(_snapshot(0.0))
    assert first is None, "must not block; returns None while the call is in flight"

    target = None
    for i in range(200):  # spin the tick loop until the future resolves
        target = exp.update(_snapshot(0.1 * (i + 1)))
        if target is not None:
            break
        time.sleep(0.01)
    assert target is not None, "the resolved proposal should surface as a target"
    exp.reset()  # tears down the executor / cancels pending
