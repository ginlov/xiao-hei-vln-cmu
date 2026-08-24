"""The semantic second opinion: where it fires, and what it is allowed to do.

The verifier is the one component that can throw away a measured position on a
model's say-so, so the tests here are mostly about the things it must *not* do:
fire where the geometry still has leverage, act on a hedge, or let a failed
network call disturb the binding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verify_binding as V  # noqa: E402
from vlm_approach import cam_dir_to_map  # noqa: E402

FLAT = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}


class TestMapDirToCam:
    def test_it_inverts_cam_dir_to_map_exactly(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            pose = {"orientation": q.tolist(), "position": [0, 0, 0]}
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            back = V.map_dir_to_cam(cam_dir_to_map(d, pose), pose)
            assert np.allclose(back, d, atol=1e-9)


class TestWhereIs:
    """The four cardinal bearings, which is what a wrong sign would break."""

    @pytest.mark.parametrize("xy,face", [([3, 0], 0), ([0, -3], 1),
                                         ([-3, 0], 2), ([0, 3], 3)])
    def test_each_cardinal_lands_centred_on_its_own_face(self, xy, face):
        at = V.where_is(xy, FLAT)
        assert at["face"] == face
        assert at["across"] == pytest.approx(0.5, abs=0.01)
        assert at["range_m"] == pytest.approx(3.0)

    def test_a_binding_on_the_vehicle_has_no_bearing(self):
        assert V.where_is([0.0, 0.0], FLAT) is None

    def test_the_side_words_follow_the_column(self):
        assert V._side(0.1) == "the left third"
        assert V._side(0.5) == "the middle"
        assert V._side(0.9) == "the right third"


class TestShouldVerify:
    BOUND = {"xy": np.array([3.0, 0.0]), "conf": 0.8, "verified": True}

    def test_nothing_bound_is_nothing_to_check(self):
        assert V.should_verify(carried_run=9, rejected=True, arriving=True,
                               bound=None) is None

    def test_one_refusal_is_ordinary(self):
        # The blind cone is body-fixed and heals by driving, so a single
        # carried step is the system working, not a symptom.
        assert V.should_verify(carried_run=1, rejected=False, arriving=False,
                               bound=self.BOUND) is None

    def test_two_in_a_row_is_the_case_geometry_cannot_reach(self):
        assert V.should_verify(carried_run=2, rejected=False, arriving=False,
                               bound=self.BOUND) == "carried"

    def test_an_undecided_disagreement_also_fires(self):
        assert V.should_verify(carried_run=0, rejected=True, arriving=False,
                               bound=self.BOUND) == "rejected"

    def test_carried_outranks_rejected(self):
        assert V.should_verify(carried_run=3, rejected=True, arriving=False,
                               bound=self.BOUND) == "carried"

    def test_arrival_on_a_verified_binding_is_not_questioned(self):
        assert V.should_verify(carried_run=0, rejected=False, arriving=True,
                               bound=self.BOUND) is None

    def test_arrival_on_a_binding_nothing_measured_is(self):
        guess = dict(self.BOUND, verified=False)
        assert V.should_verify(carried_run=0, rejected=False, arriving=True,
                               bound=guess) == "arriving"

    def test_a_quiet_step_is_left_alone(self):
        assert V.should_verify(carried_run=0, rejected=False, arriving=False,
                               bound=self.BOUND) is None


class TestVerifyBinding:
    """The single-call `anchored` prosecutor.

    Kept and tested because the audit compares the two styles, and a style that
    is only reachable through an environment variable still has to hold its
    contract: only a confident refutation acts, and nothing else may disturb the
    binding.
    """

    FACES = [b"", b"", b"", b""]

    @pytest.fixture(autouse=True)
    def _anchored(self, monkeypatch):
        monkeypatch.setattr(V, "VERIFY_STYLE", "anchored")

    def _ask(self, monkeypatch, reply: str, calls: list | None = None):
        def fake(prompt, images, model, previous=None):
            if calls is not None:
                calls.append(prompt)
            return reply
        monkeypatch.setattr(V, "ask_claude", fake)

    def test_a_binding_within_arm_s_reach_is_not_worth_a_call(self, monkeypatch):
        calls: list = []
        self._ask(monkeypatch, '{"verdict": "refuted", "confidence": 1.0}', calls)
        v = V.verify_binding(self.FACES, "the lamp", [1.0, 0.0], FLAT,
                             why="carried", model="m")
        assert calls == []
        assert v["called"] is False
        assert "acted" not in v
        assert "skipped" in v

    def test_a_confident_refutation_acts(self, monkeypatch):
        self._ask(monkeypatch, '{"verdict": "refuted", "what_is_there": "a wall",'
                               ' "why": "no lamp there", "confidence": 0.9}')
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m", carried_steps=2)
        assert v["acted"] is True
        assert v["what_is_there"] == "a wall"

    def test_a_hedged_refutation_does_not(self, monkeypatch):
        self._ask(monkeypatch, '{"verdict": "refuted", "confidence": 0.4}')
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert v["acted"] is False

    @pytest.mark.parametrize("verdict", ["holds", "cannot_tell"])
    def test_only_a_refutation_can_act(self, monkeypatch, verdict):
        self._ask(monkeypatch, '{"verdict": "%s", "confidence": 1.0}' % verdict)
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert v["acted"] is False

    def test_a_failed_call_leaves_the_binding_alone(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("529")
        monkeypatch.setattr(V, "ask_claude", boom)
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert v.get("acted") is None and "skipped" in v
        # It still cost wall clock, and wall clock is the budget.
        assert v["called"] is True

    def test_an_unparseable_reply_is_not_a_refutation(self, monkeypatch):
        self._ask(monkeypatch, "I'm not sure what you mean")
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert "skipped" in v and v["called"] is True

    def test_a_non_numeric_confidence_is_not_confident(self, monkeypatch):
        self._ask(monkeypatch, '{"verdict": "refuted", "confidence": "high"}')
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert v["acted"] is False


class TestThePrompt:
    AT = {"face": 1, "across": 0.8, "range_m": 4.2}

    def test_it_names_the_place_and_the_phrase(self):
        p = V.build_verify_prompt("the lamp near the chair", self.AT,
                                  "carried", carried_steps=3)
        assert "image 1 (right)" in p
        assert "the right third" in p
        assert "4.2 m" in p
        assert "the lamp near the chair" in p

    def test_each_trigger_says_why_it_is_being_asked(self):
        carried = V.build_verify_prompt("x", self.AT, "carried", carried_steps=2)
        rejected = V.build_verify_prompt("x", self.AT, "rejected")
        arriving = V.build_verify_prompt("x", self.AT, "arriving")
        assert "2 steps running" in carried
        assert "same object" in rejected
        assert "call this leg done" in arriving

    def test_declining_is_offered_as_a_normal_answer(self):
        # If the prompt does not say this, the model guesses, and a guessing
        # prosecutor is worse than none: it can only destroy bindings.
        p = V.build_verify_prompt("x", self.AT, "carried")
        assert "cannot_tell" in p
        assert "preferred to a guess" in p

    def test_it_refuses_to_take_nominations(self):
        p = V.build_verify_prompt("x", self.AT, "carried")
        assert "Do not suggest a better target" in p


class TestBlindVerdict:
    """The two-call prosecutor: describe the place, then match the description.

    The split exists because the anchored version was measured agreeing with
    whatever it was told -- "holds" on 83% of bindings beyond 5 m against 35%
    within 5 m, which is the shape of a model confirming a hypothesis it was
    handed rather than checking one. The eye that looks must not be told what it
    is looking for.
    """

    FACES = [b"a", b"b", b"c", b"d"]
    AT = {"face": 1, "across": 0.5, "range_m": 4.0}

    def _replies(self, monkeypatch, *replies):
        seen = []

        def fake(prompt, images, model, previous=None):
            seen.append((prompt, len(images)))
            return replies[len(seen) - 1]
        monkeypatch.setattr(V, "ask_claude", fake)
        monkeypatch.setattr(V, "strip_of", lambda face, across, frac=0.34: face)
        return seen

    def test_the_looking_call_never_sees_the_phrase(self, monkeypatch):
        seen = self._replies(monkeypatch,
                             '{"objects": ["a white wall"], "unsure": false}',
                             '{"verdict": "refuted", "confidence": 0.9}')
        V.blind_verdict(self.FACES, "the potted plant", self.AT, "m")
        look, match = seen
        assert "potted plant" not in look[0]
        assert look[1] == 1                 # one strip, not four faces
        assert "potted plant" in match[0]
        assert match[1] == 0                # the match call carries no image

    def test_it_reports_what_was_seen_alongside_the_verdict(self, monkeypatch):
        self._replies(monkeypatch,
                      '{"objects": ["a white wall", "a radiator"]}',
                      '{"verdict": "refuted", "confidence": 0.9}')
        out, calls = V.blind_verdict(self.FACES, "the lamp", self.AT, "m")
        assert calls == 2
        assert out["what_is_there"] == "a white wall; a radiator"
        assert out["verdict"] == "refuted"

    def test_nothing_seen_is_not_a_refutation(self, monkeypatch):
        # An empty description is the camera declining, not evidence that the
        # target is absent — and it costs one call, not two.
        self._replies(monkeypatch, '{"objects": [], "unsure": true}')
        out, calls = V.blind_verdict(self.FACES, "the lamp", self.AT, "m")
        assert out["verdict"] == "cannot_tell"
        assert calls == 1

    def test_an_unreadable_description_is_not_a_refutation(self, monkeypatch):
        self._replies(monkeypatch, "the camera is broken")
        out, _ = V.blind_verdict(self.FACES, "the lamp", self.AT, "m")
        assert out["verdict"] == "cannot_tell"

    def test_an_unreadable_match_is_not_a_refutation(self, monkeypatch):
        self._replies(monkeypatch,
                      '{"objects": ["a white wall"]}', "no idea")
        out, calls = V.blind_verdict(self.FACES, "the lamp", self.AT, "m")
        assert out["verdict"] == "cannot_tell"
        assert calls == 2
        assert out["what_is_there"] == "a white wall"

    def test_it_says_when_the_camera_hedged(self, monkeypatch):
        seen = self._replies(monkeypatch,
                             '{"objects": ["maybe a lamp"], "unsure": true}',
                             '{"verdict": "cannot_tell", "confidence": 0.2}')
        V.blind_verdict(self.FACES, "the lamp", self.AT, "m")
        assert "unsure" in seen[1][0]

    def test_a_blind_refutation_still_needs_confidence_to_act(self, monkeypatch):
        self._replies(monkeypatch,
                      '{"objects": ["a white wall"]}',
                      '{"verdict": "refuted", "confidence": 0.4}')
        v = V.verify_binding(self.FACES, "the lamp", [5.0, 0.0], FLAT,
                             why="carried", model="m")
        assert v["style"] == "blind" and v["calls"] == 2
        assert v["acted"] is False


class TestStripOf:
    def _jpeg(self, w=640, h=640):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (30, 30, 30)).save(buf, format="JPEG")
        return buf.getvalue()

    def _size(self, raw):
        import io
        from PIL import Image
        return Image.open(io.BytesIO(raw)).size

    def test_it_keeps_full_height(self):
        # The binding is a map xy with no elevation, so cropping vertically
        # would be inventing the one coordinate we do not have.
        assert self._size(V.strip_of(self._jpeg(), 0.5))[1] == 640

    def test_the_band_is_the_same_width_wherever_it_sits(self):
        widths = {self._size(V.strip_of(self._jpeg(), a))[0]
                  for a in (0.0, 0.2, 0.5, 0.8, 1.0)}
        assert len(widths) == 1
