"""The object-reference responder, driven against a fake robot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "scripts", REPO / "perception"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import answer_reference as AR  # noqa: E402
import target_box as T  # noqa: E402

RNG = np.random.default_rng(7)


class TestPhraseOf:
    @pytest.mark.parametrize("q,want", [
        ("Find the pillow closest to the book on the stool.",
         "the pillow closest to the book on the stool"),
        ("The red pillow closest to the sushi.",
         "The red pillow closest to the sushi"),
        ("The blue chair that is closest to the cup of coffee.",
         "The blue chair that is closest to the cup of coffee"),
        ("  Find the vase closest to the guitar  ",
         "the vase closest to the guitar"),
    ])
    def test_the_imperative_comes_off_and_nothing_else_does(self, q, want):
        assert AR.phrase_of(q) == want

    def test_a_question_that_never_says_find_is_untouched(self):
        """Two of the thirty released questions have no imperative."""
        q = "The lantern between the vase and the stone decoration."
        assert AR.phrase_of(q) == q.rstrip(".")

    def test_every_released_question_keeps_its_relation_words(self):
        f = (REPO.parent / "CMU-VLN-Challenge-2026" / "questions"
             / "questions.json")
        if not f.is_file():
            pytest.skip("challenge repo not available")
        for scene in json.loads(f.read_text()):
            for q in scene["questions"]["object_reference"]:
                p = AR.phrase_of(q)
                assert p and not p.lower().startswith("find")
                for w in ("closest", "between", "farthest", "furthest", "near"):
                    assert (w in q.lower()) == (w in p.lower())


# --------------------------------------------------------------------------
# A fake robot: four black faces, a scan holding one cube, and a pose that
# moves when driven. Enough to run the whole loop without a simulator.


def cube(centre, side=0.4, n=900):
    return np.asarray(centre, float) + RNG.uniform(-side / 2, side / 2, (n, 3))


class FakeRobot:
    def __init__(self, target=(3.0, 0.0, 0.75), extra=None):
        self.pos = np.array([0.0, 0.0, 0.75])
        self.target = np.asarray(target, float)
        self.extra = extra
        self.drives: list[tuple[float, float]] = []
        self.stopped = False

    def push(self):
        pass

    def capture(self):
        pts = cube(self.target)
        if self.extra is not None:
            pts = np.vstack([pts, cube(self.extra)])
        pose = {"position": self.pos.tolist(), "orientation": [0, 0, 0, 1]}
        # A 1920x640 JPEG is not needed: `faces_of` is monkeypatched in the
        # tests that call this, so the equirect bytes are never decoded.
        terrain = np.c_[RNG.uniform(-6, 6, (500, 2)), np.zeros(500)]
        return b"EQUIRECT", pts, terrain, pose

    def drive_to(self, x, y, timeout=None):
        self.drives.append((x, y))
        self.pos = np.array([x, y, 0.75])
        return {"ok": True, "pose": self.pos.tolist()}

    def stop(self):
        self.stopped = True
        return {"ok": True, "why": "parked"}


def make_ctx(tmp_path, robot):
    from approach_loop import Ctx
    return Ctx(robot=robot, out=tmp_path,
               log=(tmp_path / "steps.jsonl").open("w"),
               model="test-model", deadline=None)


@pytest.fixture
def patched(monkeypatch):
    """Stub the model call, the face unwrap, and the approach."""
    monkeypatch.setattr(AR, "faces_of", lambda eq: [b"f0", b"f1", b"f2", b"f3"])
    monkeypatch.setattr(AR, "crop_face", lambda face, box, **k: b"CROP")

    from approach_loop import Outcome
    monkeypatch.setattr(
        AR, "run_goto",
        lambda ctx, phrase, **k: Outcome(arrived=True, why="reached",
                                         xy=np.array([3.0, 0.0]),
                                         prev_crop=b"FIRSTCROP"))
    # A converter that accepts whatever it is asked for, so the orbit moves.
    monkeypatch.setattr(AR, "go",
                        lambda ctx, cm, pose, aim, why, **k:
                        ctx.robot.drive_to(float(aim[0]), float(aim[1])))
    monkeypatch.setattr(AR, "ConverterModel", lambda terrain: object())
    return monkeypatch


def box_on_target(pose, target, half_px=110):
    """Which face holds `target` from `pose`, and a box around it.

    The fake robot's heading never changes, so after one orbit step the target
    is no longer in front of it -- exactly as on the real vehicle, whose camera
    is a 360 rig and whose target simply moves to another face. Computing the
    face here rather than hard-coding 0 is what makes the orbit tests exercise
    the same conversion the live path uses.

    Goes through `scan_to_camera` rather than rotating by the pose alone: the
    faces are defined in the CAMERA frame, and skipping the sensor-to-camera
    extrinsic puts the box on the right face pointing at nothing.
    """
    import geometry as G
    from vlm_locate import scan_to_camera

    p_cam = scan_to_camera(np.asarray(target, float)[None, :3], pose)[0]
    d = p_cam / np.linalg.norm(p_cam)
    best = None
    for face in range(G.N_FACES):
        u, v, ok = G.world_dir_to_face_pixel(d[None, :], face)
        if not bool(ok[0]):
            continue
        cu, cv = float(u[0]), float(v[0])
        if not (0 <= cu < G.FACE_SIZE and 0 <= cv < G.FACE_SIZE):
            continue
        # Prefer the face that views it most centrally, the same rule the
        # inverse LUT uses when two faces overlap.
        off = max(abs(cu - G.FACE_SIZE / 2), abs(cv - G.FACE_SIZE / 2))
        if best is None or off < best[0]:
            best = (off, face, cu, cv)
    if best is None:
        return None, None
    _, face, cu, cv = best
    lo_v = max(0.0, cv - half_px)
    lo_u = max(0.0, cu - half_px)
    hi_v = min(G.FACE_SIZE - 1.0, cv + half_px)
    hi_u = min(G.FACE_SIZE - 1.0, cu + half_px)
    return face, [lo_v, lo_u, hi_v, hi_u]


def reply_found(**kw):
    out = {"found": True, "same_object": True, "image_index": 0,
           "box_2d": [200, 200, 440, 440], "box_px": [200, 200, 440, 440],
           "extent": "whole", "confidence": 0.9, "evidence": "the cube"}
    out.update(kw)
    return out


def tracking_reply(bot, **kw):
    """A stub that boxes the target wherever it actually is from here."""
    def stub(faces, phrase, crop=None, model=None, **k):
        pose = {"position": bot.pos.tolist(), "orientation": [0, 0, 0, 1]}
        face, box = box_on_target(pose, bot.target)
        if face is None:
            return {"found": False, "why_not": "not_in_view"}
        return reply_found(image_index=face, box_2d=box, box_px=box, **kw)
    return stub


class TestTheLoop:
    def test_a_clean_run_produces_a_box_from_several_views(self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=4)
        ctx.close()
        assert got["ok"] is True
        assert got["n_views"] >= 1
        assert got["anchored"] is True
        assert bot.stopped is True, "the vehicle must be parked before publishing"

    def test_the_box_lands_on_the_object(self, tmp_path, patched):
        bot = FakeRobot(target=(3.0, 0.0, 0.75))
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert np.linalg.norm(np.array(got["centre"])
                              - np.array([3.0, 0.0, 0.75])) < 0.6

    def test_a_view_the_model_refuses_never_reaches_the_estimator(
            self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: {
            "found": False, "why_not": "cannot_tell_which_instance"})
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert got["ok"] is False
        assert got["why"] == "no view produced a liftable box"
        assert got["n_views"] == 0

    def test_a_failing_model_call_does_not_end_the_question(
            self, tmp_path, patched):
        calls = {"n": 0}

        bot = FakeRobot()
        good = tracking_reply(bot)

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"found": False, "error": "call failed",
                        "why_not": "call_failed"}
            return good(*a, **k)
        patched.setattr(AR, "re_look", flaky)
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert calls["n"] >= 2 and got["ok"] is True

    def test_the_orbit_moves_the_vehicle_between_views(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        bot = FakeRobot()
        ctx = make_ctx(tmp_path, bot)
        AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert len(bot.drives) >= 2
        # every orbit waypoint sits at the standoff from the target
        for x, y in bot.drives[1:]:
            r = float(np.linalg.norm(np.array([x, y]) - np.array([3.0, 0.0])))
            assert r == pytest.approx(AR.VIEW_M, abs=0.2)

    def test_it_stops_once_the_box_settles(self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=8)
        ctx.close()
        assert got["n_views"] < 8, "identical views should settle early"

    def test_without_a_binding_it_still_looks_but_says_it_is_unanchored(
            self, tmp_path, patched, monkeypatch):
        from approach_loop import Outcome
        monkeypatch.setattr(AR, "run_goto",
                            lambda ctx, phrase, **k: Outcome(
                                arrived=False, why="boxed in", xy=None))
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert got["anchored"] is False
        assert got["n_views"] == 1, "nothing to orbit around"

    def test_every_view_is_recorded_with_its_reason(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found(
            extent="occluded", confidence=0.55))
        ctx = make_ctx(tmp_path, FakeRobot())
        AR.answer_reference(ctx, "Find the cube.", max_views=2)
        ctx.close()
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        views = [r for r in rows if r.get("kind") == "reference"]
        assert views
        for r in views:
            assert set(r) >= {"found", "extent", "n_returns", "mask_px",
                              "thin", "took", "confidence", "pose"}
            assert r["extent"] == "occluded"

    def test_faces_and_scans_are_written_for_every_view(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        AR.answer_reference(ctx, "Find the cube.", max_views=2)
        ctx.close()
        assert list(tmp_path.glob("step*_face0.jpg"))
        assert list(tmp_path.glob("step*_scan.npy"))


class TestCommit:
    def test_an_oversized_box_is_refused_rather_than_published(self, tmp_path):
        from approach_loop import Ctx
        box = T.TargetBox()
        box.centres = [np.zeros(3)]
        box.extents = [np.array([5.0, 0.3, 0.3])]
        box.weights = [100.0]
        box.views = [{"n": 100, "note": "wall"}]
        ctx = Ctx(robot=None, out=tmp_path,
                  log=(tmp_path / "s.jsonl").open("w"), deadline=None)
        got = AR.commit(box, "q", "p", np.array([0.0, 0.0]), ctx)
        ctx.close()
        assert got["ok"] is False and "mask spill" in got["why"]

    def test_a_heading_needs_three_agreeing_views(self, tmp_path):
        from approach_loop import Ctx
        ctx = Ctx(robot=None, out=tmp_path,
                  log=(tmp_path / "s.jsonl").open("w"), deadline=None)
        box = T.TargetBox()
        for x in (0.0, 0.02):
            box.add(cube([x, 0, 0.5], 0.3, 200), note="v")
        got = AR.commit(box, "q", "p", None, ctx)
        assert got["heading"] == 0.0            # two views is a line, not an axis
        box.add(cube([0.04, 0, 0.5], 0.3, 200), note="v")
        got = AR.commit(box, "q", "p", None, ctx)
        ctx.close()
        assert isinstance(got["heading"], float)

    def test_the_answer_carries_its_audit_trail(self, tmp_path):
        from approach_loop import Ctx
        ctx = Ctx(robot=None, out=tmp_path,
                  log=(tmp_path / "s.jsonl").open("w"), deadline=None)
        box = T.TargetBox(anchor=np.array([0.0, 0.0]))
        box.add(cube([0, 0, 0.5], 0.4, 400), note="a")
        box.add(cube([9, 0, 0.5], 0.4, 400), note="b")
        got = AR.commit(box, "the q", "the p", np.array([0.0, 0.0]), ctx)
        ctx.close()
        assert got["question"] == "the q" and got["phrase"] == "the p"
        assert got["n_views"] == 2 and got["n_agreeing"] == 1
        assert len(got["outliers"]) == 1
        assert got["binding"] == [0.0, 0.0]
