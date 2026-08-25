"""Unit tests for NavTask1Explorer — question-directed VLM navigation.

Rendering is monkeypatched to dummy bytes; a fake engine returns scripted
proposals and records each prompt, so no key or network is touched. The
explorer inherits its state machine from NavVLMExplorer, so these focus on the
task-1 differences: holding before a question, feeding the question + scene
graph into the prompt, and completing on model-declared arrival.
"""

from __future__ import annotations

import pytest

from xiao_hei_vln.exploration import _nav_vlm
from xiao_hei_vln.exploration._nav_task1 import NavTask1Explorer
from xiao_hei_vln.messages import Header, OdomPose, Quaternion, Stamp, Vector3
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.question import ChallengeQuestion
from xiao_hei_vln.nav_vlm.engine import WaypointProposal


@pytest.fixture(autouse=True)
def _no_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_nav_vlm, "render_grid_png", lambda *a, **k: b"png")
    monkeypatch.setattr(_nav_vlm, "panorama_to_jpeg", lambda *a, **k: b"jpg")


class FakeEngine:
    def __init__(self, proposals: list[WaypointProposal]) -> None:
        self._proposals = list(proposals)
        self.calls: list[dict] = []

    def propose(self, *, user_text, panorama_jpg, occupancy_png):
        self.calls.append({"user_text": user_text})
        if not self._proposals:
            return WaypointProposal(done=True, rationale="exhausted")
        return self._proposals.pop(0)


class FakeScene:
    """Minimal scene: only to_dict() is read by the explorer's prompt builder."""

    def __init__(self, objects: list[dict]) -> None:
        self._objects = objects

    def to_dict(self) -> dict:
        return {"objects": self._objects}


def _snapshot(t: float, *, question: str | None = "the red chair") -> VLMInput:
    stamp = Stamp.from_seconds(t)
    pose = OdomPose(
        header=Header(stamp=stamp, frame_id="map"),
        position=Vector3(x=0.2, y=0.2, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    q = ChallengeQuestion.from_text(question, stamp) if question is not None else None
    return VLMInput(tick_id=int(t * 10), tick_time=stamp, pose=pose, question=q)


def _reseed(exp: NavTask1Explorer) -> None:
    exp._grid._blacklisted.clear()
    exp._grid._occupied.clear()
    exp._grid._free = {(ix, iy) for ix in range(-1, 7) for iy in range(-3, 4)}


def _make(engine: FakeEngine, scene: FakeScene, **kw) -> NavTask1Explorer:
    return NavTask1Explorer(
        engine, scene=scene, grid_resolution=1.0, async_calls=False, **kw
    )


def test_holds_until_a_question_arrives() -> None:
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="go")])
    exp = _make(eng, FakeScene([]), max_waypoints=5)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0, question=None))
    assert wp is None, "no target object yet → hold"
    assert not eng.calls, "the model must not be called before a question"


def test_question_triggers_nav_and_prompt_carries_question_and_scene() -> None:
    # Position is a [x, y, z] list — the real SceneRepresentation.to_dict shape.
    scene = FakeScene([
        {"object_id": 7, "label": "red chair", "position": [3.0, 0.0, 0.0]},
    ])
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="toward chair")])
    exp = _make(eng, scene, max_waypoints=5)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0, question="the red chair"))

    assert wp is not None, "a question should start goal-directed nav"
    text = eng.calls[-1]["user_text"]
    assert "the red chair" in text, "the question must reach the model"
    assert "#7" in text and "red chair" in text, "the scene graph must reach the model"


def test_model_arrival_completes() -> None:
    eng = FakeEngine([WaypointProposal(done=True, rationale="reached the chair")])
    exp = _make(eng, FakeScene([]), max_waypoints=5)
    _reseed(exp)

    wp = exp.update(_snapshot(0.0))
    assert wp is None
    assert exp.is_complete(), "done=true means arrived → explorer completes"


def test_reached_advances_and_requests_next() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=1.5, y=1.5, rationale="b"),
    ])
    exp = _make(eng, FakeScene([]), max_waypoints=5)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    assert exp._current_target is not None

    exp.advance()  # nav stack reached it
    assert exp._current_target is None
    _reseed(exp)
    exp.update(_snapshot(1.0))
    assert len(eng.calls) == 2, "a reach event should trigger a fresh nav call"


def test_reach_switches_to_explore_after_repeated_skips() -> None:
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="a")] * 10)
    exp = _make(eng, FakeScene([]), max_waypoints=20, explore_after_skips=2)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    assert exp._mode == "reach"
    exp.force_skip()                       # consecutive=1
    _reseed(exp)
    exp.update(_snapshot(1.0))
    exp.force_skip()                       # consecutive=2
    _reseed(exp)
    exp.update(_snapshot(2.0))             # mode switch runs at start of update
    assert exp._mode == "explore"
    assert not exp.is_complete(), "reach stalling must NOT terminate — it explores"


def test_explore_switches_back_on_novel_view() -> None:
    scene = FakeScene([
        {"object_id": 1, "label": "couch", "position": [3.0, 1.0, 0.0],
         "observing_viewpoint_ids": [1]},
    ])
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="a")] * 10)
    exp = _make(eng, scene, max_waypoints=20, explore_after_skips=1)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    exp.force_skip()
    _reseed(exp)
    exp.update(_snapshot(1.0))
    assert exp._mode == "explore"
    # A new viewpoint of the object grows scene novelty → back to reach.
    scene._objects[0]["observing_viewpoint_ids"] = [1, 2]
    _reseed(exp)
    exp.update(_snapshot(2.0))
    assert exp._mode == "reach"


def test_question_time_cap_completes() -> None:
    eng = FakeEngine([WaypointProposal(done=False, x=2.5, y=0.5, rationale="a")] * 5)
    exp = _make(eng, FakeScene([]), max_waypoints=20, max_question_seconds=5.0)
    _reseed(exp)
    exp.update(_snapshot(0.0))
    assert not exp.is_complete()
    assert exp.update(_snapshot(6.0)) is None  # past the cap
    assert exp.is_complete(), "the per-question time cap must force completion → answer"


def test_coverage_plateau_waits_for_the_named_target() -> None:
    """A local novelty plateau must NOT end the search while the object the
    question names is still missing — only once it is detected (bounded
    elsewhere by the nav time cap)."""
    scene = FakeScene([{"label": "table"}])  # graph populated, but no "chair"
    exp = _make(
        FakeEngine([]), scene,
        coverage_plateau_s=45.0, min_visited_before_plateau=4,
    )
    exp._question_classes = {"chair"}
    exp._visited = [1, 2, 3, 4]  # only len is read here

    assert exp._check_coverage_plateau(0.0) is False   # seeds the novelty timer
    assert exp._check_coverage_plateau(100.0) is False  # target absent → keep going
    assert not exp._done

    # Target appears: the same flat-novelty window now ends the search.
    scene._objects = [{"label": "chair"}]
    exp._last_novelty = exp._scene_novelty()
    exp._last_novelty_time = 100.0
    assert exp._check_coverage_plateau(160.0) is True
    assert exp._done


def test_target_nouns_and_detection() -> None:
    from xiao_hei_vln.exploration._nav_task1 import _target_nouns

    assert _target_nouns("Find the vase closest to the hookah.") == {"vase", "hookah"}
    exp = _make(FakeEngine([]), FakeScene([{"label": "flower vase"}]))
    exp._question_classes = {"vase"}
    assert exp._target_detected() is True          # substring match
    exp._question_classes = {"lamp"}
    assert exp._target_detected() is False
    exp._question_classes = set()
    assert exp._target_detected() is True           # unknown target → satisfied


def test_force_skip_feeds_failure_into_next_prompt() -> None:
    eng = FakeEngine([
        WaypointProposal(done=False, x=2.5, y=0.5, rationale="a"),
        WaypointProposal(done=False, x=-0.5, y=-1.5, rationale="b"),
    ])
    exp = _make(eng, FakeScene([]), max_waypoints=5, max_consecutive_skips=9)
    _reseed(exp)
    exp.update(_snapshot(0.0))

    exp.force_skip()
    assert exp.skipped_count == 1
    _reseed(exp)
    exp.update(_snapshot(1.0))
    assert "FAILED" in eng.calls[-1]["user_text"]
