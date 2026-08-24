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
        self.thetas: list[float | None] = []
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

    def drive_to(self, x, y, timeout=None, theta=None):
        self.drives.append((x, y))
        self.thetas.append(theta)
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
        # Each view is two hops: out to the orbit point at VIEW_M + NUDGE_M,
        # then straight at the target to VIEW_M, which is what turns the
        # vehicle to face it. So every waypoint sits at one radius or the other.
        for x, y in bot.drives[1:]:
            r = float(np.linalg.norm(np.array([x, y]) - np.array([3.0, 0.0])))
            assert (r == pytest.approx(AR.VIEW_M, abs=0.2)
                    or r == pytest.approx(AR.VIEW_M + AR.NUDGE_M, abs=0.2)), \
                f"waypoint at {r:.2f} m is on neither ring"

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

    def test_a_drive_that_stopped_short_keeps_its_binding(
            self, tmp_path, patched, monkeypatch):
        """chinese_room: 1.06 m short, binding 0.155 m from the true object.

        `xy` is None on every failure path on purpose -- the numerical
        responder reads it as "this leg framed the anchor". Object reference
        needs the belief, not the arrival, so it falls back to `bound_xy`.
        Without this the run went unanchored, had nothing to orbit around,
        and committed a box from its one remaining view.
        """
        from approach_loop import Outcome
        monkeypatch.setattr(AR, "run_goto",
                            lambda ctx, phrase, **k: Outcome(
                                arrived=False, why="stack clamped short",
                                xy=None, bound_xy=np.array([3.0, 0.0])))
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=3)
        ctx.close()
        assert got["anchored"] is True
        assert got["binding"] == pytest.approx([3.0, 0.0])
        assert got["reached"] is False, "anchored, but it never got there"
        looks = [json.loads(l) for l in
                 (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        assert len([r for r in looks if r.get("kind") == "reference"]) > 1, \
            "a binding is something to orbit around"

    def test_an_arrival_is_recorded_as_one(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=2)
        ctx.close()
        assert got["reached"] is True

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


class TestStandOff:
    """Measured: two drives parked 1.06 m and 2.60 m short of their binding."""

    def place(self, tmp_path, patched, at):
        r = FakeRobot()
        r.pos = np.array([at[0], at[1], 0.75])
        return r, make_ctx(tmp_path, r)

    def test_it_closes_in_when_the_approach_stopped_short(self, tmp_path, patched):
        """loft: 2.60 m out, sharing the frame with an identical plant."""
        r, ctx = self.place(tmp_path, patched, (0.4, 0.0))
        AR.stand_off(ctx, np.array([3.0, 0.0]))
        ctx.close()
        assert r.drives, "it must drive"
        gap = float(np.linalg.norm(np.asarray(r.drives[-1]) - np.array([3.0, 0.0])))
        assert gap == pytest.approx(AR.VIEW_M, abs=0.05)

    def test_it_still_backs_off_when_parked_on_top_of_the_target(
            self, tmp_path, patched):
        r, ctx = self.place(tmp_path, patched, (2.4, 0.0))
        AR.stand_off(ctx, np.array([3.0, 0.0]))
        ctx.close()
        assert r.drives
        gap = float(np.linalg.norm(np.asarray(r.drives[-1]) - np.array([3.0, 0.0])))
        assert gap == pytest.approx(AR.VIEW_M, abs=0.05)

    def test_a_good_range_is_left_alone(self, tmp_path, patched):
        """Inside the band, a second drive buys centimetres and costs a park."""
        r, ctx = self.place(tmp_path, patched, (3.0 - AR.VIEW_M, 0.0))
        AR.stand_off(ctx, np.array([3.0, 0.0]))
        ctx.close()
        assert r.drives == []

    def test_the_tolerance_band_is_not_a_cliff(self, tmp_path, patched):
        just_inside = 3.0 - (AR.VIEW_M + AR.CLOSE_TOL_M - 0.05)
        r, ctx = self.place(tmp_path, patched, (just_inside, 0.0))
        AR.stand_off(ctx, np.array([3.0, 0.0]))
        ctx.close()
        assert r.drives == [], "inside the band, no drive"

        just_outside = 3.0 - (AR.VIEW_M + AR.CLOSE_TOL_M + 0.05)
        r2, ctx2 = self.place(tmp_path, patched, (just_outside, 0.0))
        AR.stand_off(ctx2, np.array([3.0, 0.0]))
        ctx2.close()
        assert r2.drives, "past it, drive"


class TestFaceGuard:
    """The model names the face it found the target in; the binding knows.

    Measured over the three driven runs: with a good binding the guard
    rejected 7 views and every one of them was bad -- five that had drifted
    onto an identical potted plant 5.3 m away, and the two that named the
    back face while the target sat 24 and 38 degrees off the front and so
    lifted nothing at all. With a binding that was itself wrong it cost one
    good view.
    """

    def pose(self, at, yaw_deg):
        h = np.radians(yaw_deg) / 2
        return {"position": [at[0], at[1], 0.75],
                "orientation": [0.0, 0.0, float(np.sin(h)), float(np.cos(h))]}

    @pytest.mark.parametrize("bearing,face", [
        (0, 0), (-90, 1), (180, 2), (90, 3),
        (-8, 0), (96, 3), (-58, 1), (160, 2),
    ])
    def test_the_convention_matches_the_faces(self, bearing, face):
        """front/right/back/left at 0/90/180/270, azimuth clockwise."""
        b = np.radians(bearing)
        target = np.array([3.0 * np.cos(b), 3.0 * np.sin(b)])
        assert AR.expected_face(self.pose((0, 0), 0), target) == face

    def test_heading_rotates_the_faces_with_the_vehicle(self):
        target = np.array([3.0, 0.0])
        assert AR.expected_face(self.pose((0, 0), 0), target) == 0
        assert AR.expected_face(self.pose((0, 0), 90), target) == 1
        assert AR.expected_face(self.pose((0, 0), 180), target) == 2

    def test_a_neighbouring_face_is_not_an_error(self):
        """The faces overlap ~10 deg; an object on a seam is in either."""
        assert AR.face_off(self.pose((0, 0), 0), np.array([3.0, 0.0]), 1) == 1
        assert 1 <= AR.MAX_FACE_OFF

    def test_the_opposite_face_is_two_off(self):
        assert AR.face_off(self.pose((0, 0), 0), np.array([3.0, 0.0]), 2) == 2

    def test_without_a_binding_there_is_nothing_to_check_against(self):
        assert AR.face_off(self.pose((0, 0), 0), None, 2) is None

    def test_a_view_naming_the_wrong_face_is_refused(self, tmp_path, patched):
        """loft steps 8 and 11: `whole`, confident, and zero returns."""
        patched.setattr(AR, "re_look",
                        lambda *a, **k: reply_found(image_index=2))
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=2)
        ctx.close()
        assert got["n_views"] == 0, "the target is in front, not behind"
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        v = [r for r in rows if r.get("kind") == "reference"]
        # Only the first view is checked: the fake robot turns as it orbits,
        # so by the second the named face is no longer the opposite one and
        # the view is refused by the sparse gate instead.
        assert v and v[0]["refused"] == "wrong_face"
        assert v[0]["face_off"] == 2 and v[0]["expected_face"] == 0


class TestSparseViewGate:
    def test_a_view_with_too_few_returns_is_not_taken(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=2,
                                  min_returns=100_000)
        ctx.close()
        assert got["n_views"] == 0
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        v = [r for r in rows if r.get("kind") == "reference"]
        assert v and v[0]["refused"] == "too_sparse"

    def test_the_gate_can_be_turned_off(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=2,
                                  min_returns=0)
        ctx.close()
        assert got["n_views"] >= 1

    def test_the_policy_records_the_gate_that_was_driven(self, tmp_path, patched):
        patched.setattr(AR, "re_look", lambda *a, **k: reply_found())
        ctx = make_ctx(tmp_path, FakeRobot())
        got = AR.answer_reference(ctx, "Find the cube.", max_views=1,
                                  min_returns=7)
        ctx.close()
        assert got["policy"]["min_returns"] == 7
        assert got["policy"]["max_face_off"] == AR.MAX_FACE_OFF


class TestHiddenRun:
    """livingroom_3 spent four calls on an arc where a cabinet was in the way."""

    def test_a_run_of_misses_stops_the_orbit(self, tmp_path, patched):
        patched.setattr(AR, "re_look",
                        lambda *a, **k: {"found": False, "why_not": "hidden"})
        ctx = make_ctx(tmp_path, FakeRobot())
        AR.answer_reference(ctx, "Find the cube.", max_views=8)
        ctx.close()
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        n = len([r for r in rows if r.get("kind") == "reference"])
        assert n == AR.HIDDEN_RUN, f"stopped after {n}, wanted {AR.HIDDEN_RUN}"

    def test_a_corpus_drive_keeps_looking_so_the_stop_is_replayable(
            self, tmp_path, patched):
        patched.setattr(AR, "re_look",
                        lambda *a, **k: {"found": False, "why_not": "hidden"})
        ctx = make_ctx(tmp_path, FakeRobot())
        AR.answer_reference(ctx, "Find the cube.", max_views=6, settle=False)
        ctx.close()
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        assert len([r for r in rows if r.get("kind") == "reference"]) == 6

    def test_a_hit_resets_the_run(self, tmp_path, patched):
        seen = {"i": 0}

        def alternate(*a, **k):
            seen["i"] += 1
            return (reply_found() if seen["i"] % 3 == 1
                    else {"found": False, "why_not": "hidden"})
        patched.setattr(AR, "re_look", alternate)
        ctx = make_ctx(tmp_path, FakeRobot())
        AR.answer_reference(ctx, "Find the cube.", max_views=5)
        ctx.close()
        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        assert len([r for r in rows if r.get("kind") == "reference"]) > AR.HIDDEN_RUN


class TestAimAt:
    """Heading follows travel, so the last hop has to point at the object.

    Asked to hold -121 deg the vehicle arrived at +37 -- the direction it had
    been travelling. `Pose2D.theta` is published and ignored.
    """

    def test_the_last_hop_points_at_the_target(self, tmp_path, patched):
        r = FakeRobot()
        r.pos = np.array([0.0, 0.0, 0.75])
        ctx = make_ctx(tmp_path, r)
        AR.aim_at(ctx, np.array([5.0, 0.0]))
        ctx.close()
        assert r.drives, "it must drive"
        x, y = r.drives[-1]
        assert x == pytest.approx(AR.NUDGE_M, abs=1e-6) and y == pytest.approx(0.0)

    def test_it_stops_at_the_viewing_range_not_on_the_object(self, tmp_path, patched):
        r = FakeRobot()
        r.pos = np.array([5.0 - AR.VIEW_M - 0.2, 0.0, 0.75])
        ctx = make_ctx(tmp_path, r)
        AR.aim_at(ctx, np.array([5.0, 0.0]))
        ctx.close()
        gap = 5.0 - r.drives[-1][0]
        assert gap == pytest.approx(AR.VIEW_M, abs=1e-6), "never closer than VIEW_M"

    def test_standing_inside_the_range_is_left_alone(self, tmp_path, patched):
        r = FakeRobot()
        r.pos = np.array([5.0 - 1.0, 0.0, 0.75])
        ctx = make_ctx(tmp_path, r)
        AR.aim_at(ctx, np.array([5.0, 0.0]))
        ctx.close()
        assert r.drives == [], "a nudge here would drive into the object"

    def test_the_orbit_parks_a_nudge_further_out(self, tmp_path, patched):
        """So the hop that turns the vehicle still lands on the view ring."""
        assert AR.NUDGE_M > 0
        src = (REPO / "scripts" / "answer_reference.py").read_text()
        assert "orbit_point(pose, target_xy, orbit_deg, VIEW_M + NUDGE_M)" in src
        assert "stand_off(ctx, target_xy, want=VIEW_M + NUDGE_M)" in src


class TestWaypointHeading:
    """`Pose2D.theta` was published as a hard 0.0 on every waypoint ever sent.

    The vehicle therefore parked facing map-east however the target lay, and
    a target left behind and below it falls in the wedge its own body
    occludes. Measured across four drives: every one of the four views taken
    on the back face lifted zero returns, against seventeen of seventeen on
    the other three faces.
    """

    def setup_go(self, monkeypatch, robot):
        """The real `go`, with only the converter stubbed out."""
        import answer_numerical as AN
        monkeypatch.setattr(AN, "nearest_allowed_step",
                            lambda cm, here, aim: np.asarray(aim, float)[:2])

        class CM:
            def best_waypoint_toward(self, aim, here, min_move=0.0):
                return (np.asarray(aim, float)[:2], 0.0)
        from approach_loop import Ctx
        return AN, CM()

    def ctx_for(self, tmp_path, robot):
        from approach_loop import Ctx
        return Ctx(robot=robot, out=tmp_path,
                   log=(tmp_path / "steps.jsonl").open("w"),
                   model="test-model", deadline=None)

    def test_the_heading_points_from_the_goal_at_the_object(
            self, tmp_path, monkeypatch):
        r = FakeRobot()
        AN, cm = self.setup_go(monkeypatch, r)
        ctx = self.ctx_for(tmp_path, r)
        pose = {"position": [0.0, 0.0, 0.75], "orientation": [0, 0, 0, 1]}
        # stand at (0, -2), look at (0, 0): due north, +90 deg
        AN.go(ctx, cm, pose, np.array([0.0, -2.0]), why="t",
              face=np.array([0.0, 0.0]))
        ctx.close()
        assert r.thetas[-1] == pytest.approx(np.pi / 2, abs=1e-6)

    def test_without_a_face_the_drive_is_unchanged(self, tmp_path, monkeypatch):
        r = FakeRobot()
        AN, cm = self.setup_go(monkeypatch, r)
        ctx = self.ctx_for(tmp_path, r)
        pose = {"position": [0.0, 0.0, 0.75], "orientation": [0, 0, 0, 1]}
        AN.go(ctx, cm, pose, np.array([1.0, 1.0]), why="t")
        ctx.close()
        assert r.thetas[-1] is None, "None, not 0.0 -- the bridge owns the default"

    @pytest.mark.parametrize("goal,look,deg", [
        ((-2.0, 0.0), (0.0, 0.0), 0.0),      # east
        ((2.0, 0.0), (0.0, 0.0), 180.0),     # west
        ((0.0, 2.0), (0.0, 0.0), -90.0),     # south
    ])
    def test_every_quadrant(self, tmp_path, monkeypatch, goal, look, deg):
        r = FakeRobot()
        AN, cm = self.setup_go(monkeypatch, r)
        ctx = self.ctx_for(tmp_path, r)
        pose = {"position": [0.0, 0.0, 0.75], "orientation": [0, 0, 0, 1]}
        AN.go(ctx, cm, pose, np.asarray(goal), why="t", face=np.asarray(look))
        ctx.close()
        assert np.degrees(r.thetas[-1]) == pytest.approx(deg, abs=1e-6)


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


class TestCorpusDrive:
    """A drive is the one thing replay cannot redo, so it must be a superset."""

    def test_corpus_mode_keeps_looking_past_the_settle_point(
            self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=6,
                                  settle=False)
        ctx.close()
        assert got["n_views"] == 6, "settle=False must not stop early"

    def test_the_default_still_settles(self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=6)
        ctx.close()
        assert got["n_views"] < 6

    def test_the_orbit_step_is_a_parameter(self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        AR.answer_reference(ctx, "Find the cube.", max_views=3, settle=False,
                            orbit_deg=30.0)
        ctx.close()
        # Consecutive orbit waypoints subtend the requested angle at the target.
        t = np.array([3.0, 0.0])
        angs = [np.degrees(np.arctan2(*(np.array(d) - t)[::-1]))
                for d in bot.drives]
        # The aim hop drives straight at the target, so it keeps the bearing
        # and only changes the range: collapse those before measuring the step.
        turns = [a for i, a in enumerate(angs)
                 if i == 0 or abs(((a - angs[i - 1] + 180) % 360) - 180) > 1e-6]
        steps = [abs(((b - a + 180) % 360) - 180) for a, b in zip(turns, turns[1:])]
        assert steps and all(s == pytest.approx(30.0, abs=1.0) for s in steps)

    def test_the_policy_travels_with_the_answer(self, tmp_path, patched):
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        got = AR.answer_reference(ctx, "Find the cube.", max_views=4,
                                  settle=False, orbit_deg=40.0,
                                  size_mode="max", use_extent=False)
        ctx.close()
        p = got["policy"]
        assert p["orbit_deg"] == 40.0 and p["settle"] is False
        assert p["size_mode"] == "max" and p["extent_weights"] is False
        assert p["standoff_m"] == AR.VIEW_M

    def test_a_recorded_view_can_be_re_lifted_without_the_model(
            self, tmp_path, patched):
        """The point of recording `box_px`: replay must not need a new call."""
        bot = FakeRobot()
        patched.setattr(AR, "re_look", tracking_reply(bot))
        ctx = make_ctx(tmp_path, bot)
        AR.answer_reference(ctx, "Find the cube.", max_views=3, settle=False)
        ctx.close()

        rows = [json.loads(l) for l in
                (tmp_path / "steps.jsonl").read_text().splitlines() if l.strip()]
        views = [r for r in rows if r.get("kind") == "reference" and r["took"]]
        assert views, "no view was taken"
        for r in views:
            assert r["box_px"] is not None and r["image_index"] is not None
            assert r["orbit_deg"] is not None and r["range_m"] is not None
            scan = np.load(tmp_path / f"step{r['step']}_scan.npy")
            again = T.lift_box(r["box_px"], r["image_index"], scan, r["pose"])
            assert again["n"] == r["n_returns"], \
                "a replayed lift must reproduce the driven one exactly"
