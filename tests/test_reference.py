"""The object-reference answer key, and the geometry it is built from."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import score_reference as R  # noqa: E402
from score_numerical import is_colour  # noqa: E402


def box(cx, cy, cz, sx, sy, sz, label="thing", oid="1", colours=None):
    c = np.array([cx, cy, cz], float)
    s = np.array([sx, sy, sz], float)
    return {"id": oid, "label": label, "c": c, "sz": s,
            "lo": c - s / 2, "hi": c + s / 2, "colours": colours or []}


class TestGap:
    def test_overlapping_boxes_have_zero_gap(self):
        a, b = box(0, 0, 0, 1, 1, 1), box(0.5, 0, 0, 1, 1, 1)
        assert R._gap(a, b) == pytest.approx(0.0)

    def test_gap_is_surface_to_surface_not_centre_to_centre(self):
        # Two 1 m cubes 3 m apart centre to centre: 2 m of air between faces.
        a, b = box(0, 0, 0, 1, 1, 1), box(3, 0, 0, 1, 1, 1)
        assert R._gap(a, b) == pytest.approx(2.0)
        assert R.dist(a, b, "centre") == pytest.approx(3.0)

    def test_a_wide_anchor_is_near_by_surface_and_far_by_centre(self):
        """The reason `surface` is the default reading of "closest to"."""
        cabinet = box(0, 0, 0, 4.0, 0.5, 0.8)
        cup = box(2.1, 0, 0, 0.1, 0.1, 0.1)
        assert R.dist(cup, cabinet, "surface") < 0.2
        assert R.dist(cup, cabinet, "centre") > 2.0


class TestResolve:
    def test_a_label_may_be_a_list(self):
        objs = [box(0, 0, 0, 1, 1, 1, "tv stand", "1"),
                box(5, 0, 0, 1, 1, 1, "sofa", "2")]
        assert not R.resolve({"label": "tv cabinet"}, objs)
        got = R.resolve({"label": ["tv cabinet", "tv stand"]}, objs)
        assert [o["id"] for o in got] == ["1"]

    def test_label_matching_is_exact_not_substring(self):
        """`cabinet` must not sweep in `kitchen cabinet` and `tv remote`."""
        objs = [box(0, 0, 0, 1, 1, 1, "cabinet", "1"),
                box(2, 0, 0, 1, 1, 1, "kitchen cabinet", "2")]
        assert [o["id"] for o in R.resolve({"label": "cabinet"}, objs)] == ["1"]

    def test_rel_filters_to_objects_standing_in_it(self):
        table = box(0, 0, 0.35, 1.0, 1.0, 0.7, "table", "t")
        on = box(0, 0, 0.75, 0.1, 0.1, 0.1, "cup", "on")
        off = box(3, 0, 0.75, 0.1, 0.1, 0.1, "cup", "off")
        objs = [table, on, off]
        got = R.resolve({"label": "cup", "rel": ("on", {"label": "table"})}, objs)
        assert [o["id"] for o in got] == ["on"]

    def test_inverse_rel_swaps_the_arguments(self):
        """*"the cabinet with a TV on it"* — the cabinet is the target."""
        cab = box(0, 0, 0.3, 1.5, 0.5, 0.6, "cabinet", "cab")
        bare = box(5, 0, 0.3, 1.5, 0.5, 0.6, "cabinet", "bare")
        tv = box(0, 0, 0.9, 1.0, 0.1, 0.6, "tv", "tv")
        objs = [cab, bare, tv]
        got = R.resolve({"label": "cabinet", "rel": ("on_inv", {"label": "tv"})}, objs)
        assert [o["id"] for o in got] == ["cab"]

    def test_a_terms_own_pick_narrows_it_to_one(self):
        objs = [box(0, 0, 0, 1, 1, 1, "table", "near"),
                box(9, 0, 0, 1, 1, 1, "table", "far"),
                box(0.6, 0, 0, 0.2, 0.2, 0.2, "screen", "s")]
        got = R.resolve({"label": "table",
                         "pick": ("closest_to", {"label": "screen"})}, objs)
        assert [o["id"] for o in got] == ["near"]


class TestPick:
    def test_closest_and_farthest_are_opposite_ends(self):
        objs = [box(1, 0, 0, 0.2, 0.2, 0.2, "vase", "a"),
                box(5, 0, 0, 0.2, 0.2, 0.2, "vase", "b"),
                box(0, 0, 0, 0.2, 0.2, 0.2, "guitar", "g")]
        cands = R.resolve({"label": "vase"}, objs)
        assert R.pick(cands, ("closest_to", {"label": "guitar"}), objs)["id"] == "a"
        assert R.pick(cands, ("farthest_from", {"label": "guitar"}), objs)["id"] == "b"

    def test_between_prefers_the_one_on_the_segment(self):
        a = box(0, 0, 0, 0.2, 0.2, 0.2, "vase", "a")
        b = box(4, 0, 0, 0.2, 0.2, 0.2, "stool", "b")
        mid = box(2, 0, 0, 0.2, 0.2, 0.2, "lantern", "mid")
        off = box(2, 3, 0, 0.2, 0.2, 0.2, "lantern", "off")
        objs = [a, b, mid, off]
        cands = R.resolve({"label": "lantern"}, objs)
        got = R.pick(cands, ("between", {"label": "vase"}, {"label": "stool"}), objs)
        assert got["id"] == "mid"

    def test_pick_returns_none_when_the_reference_is_missing(self):
        objs = [box(0, 0, 0, 1, 1, 1, "vase", "a")]
        cands = R.resolve({"label": "vase"}, objs)
        assert R.pick(cands, ("closest_to", {"label": "guitar"}), objs) is None

    def test_unique_refuses_a_tie(self):
        objs = [box(0, 0, 0, 1, 1, 1, "vase", "a"),
                box(2, 0, 0, 1, 1, 1, "vase", "b")]
        cands = R.resolve({"label": "vase"}, objs)
        assert R.pick(cands, ("unique",), objs) is None


class TestColourFloor:
    def test_a_colour_covering_one_percent_is_not_a_colour(self):
        """`loft`'s eleven chairs, which is why the floor exists."""
        o = box(0, 0, 0, 1, 1, 1, "chair",
                colours=[{"name": "gray", "pct": 0.01, "rgb": (128, 128, 128),
                          "luma": 128.0}])
        assert is_colour(o, "black", min_pct=0.0) is False      # gray, too light
        o["colours"][0].update(name="red")
        assert is_colour(o, "red", min_pct=0.0) is True         # old behaviour
        assert is_colour(o, "red", min_pct=0.5) is False        # not with the floor

    def test_a_dominant_colour_survives_the_floor(self):
        o = box(0, 0, 0, 1, 1, 1, "pillow",
                colours=[{"name": "maroon", "pct": 0.996, "rgb": (178, 34, 34),
                          "luma": 70.0}])
        assert is_colour(o, "red", min_pct=R.COLOUR_MIN_PCT) is True

    def test_the_default_floor_is_zero_so_nothing_else_moves(self):
        """`score_numerical`'s published key must not shift under this change."""
        o = box(0, 0, 0, 1, 1, 1, "x",
                colours=[{"name": "red", "pct": 0.02, "rgb": (255, 0, 0),
                          "luma": 76.0}])
        assert is_colour(o, "red") is True


class TestScoreBox:
    def test_a_perfect_box_scores_two(self):
        gt = box(1, 2, 3, 0.4, 0.4, 0.4)
        got = R.score_box([1, 2, 3], [0.4, 0.4, 0.4], gt)
        assert got["iou"] == pytest.approx(1.0)
        assert got["points"] == 2

    def test_the_tiers_are_the_challenge_readmes(self):
        gt = box(0, 0, 0, 1, 1, 1)
        # Two identical unit cubes offset by d on one axis have
        # IoU = (1-d)/(1+d): d=1/3 -> 0.5 exactly, d=0.6 -> 0.25 exactly.
        assert R.score_box([1 / 3, 0, 0], [1, 1, 1], gt)["points"] == 2
        assert R.score_box([0.6, 0, 0], [1, 1, 1], gt)["points"] == 1
        assert R.score_box([0.7, 0, 0], [1, 1, 1], gt)["points"] == 0

    def test_a_disjoint_box_scores_zero_and_does_not_divide_by_zero(self):
        gt = box(0, 0, 0, 0.3, 0.3, 0.3)
        got = R.score_box([9, 9, 9], [0.3, 0.3, 0.3], gt)
        assert got["iou"] == 0.0 and got["points"] == 0

    def test_centre_error_and_size_ratio_are_reported(self):
        gt = box(0, 0, 0, 1, 1, 1)
        got = R.score_box([0.3, 0, 0], [0.5, 0.5, 0.5], gt)
        assert got["centre_error_m"] == pytest.approx(0.3)
        assert got["size_ratio"] == pytest.approx(0.125)


class TestTheKeyItself:
    """These run against the real VLA-3D annotation."""

    @pytest.fixture(scope="class")
    def rows(self):
        rows = R.key()
        if not any(r["resolved"]["n"] for r in rows):
            pytest.skip("VLA-3D annotation not available")
        return rows

    def test_every_spec_matches_a_question_the_challenge_asks(self):
        bad = R.check_questions()
        if bad and any("not found" in b for b in bad):
            pytest.skip("challenge repo not available")
        assert bad == []

    def test_there_are_thirty_specs_two_per_scene(self):
        assert len(R.SPECS) == 30
        scenes = {s["scene"] for s in R.SPECS}
        assert len(scenes) == 15
        for sc in scenes:
            assert sorted(s["i"] for s in R.SPECS if s["scene"] == sc) == [0, 1]

    def test_every_question_resolves_to_exactly_one_object(self, rows):
        """The challenge guarantees the referred object is unique, so a spec
        that leaves 0 or 2 candidates is a misreading, not a hard question."""
        bad = [(r["scene"], r["i"], r["resolved"]["why"])
               for r in rows if r["resolved"]["n"] != 1]
        assert bad == []

    def test_no_answer_flips_to_a_different_object_under_the_free_parameters(self, rows):
        base = {(r["scene"], r["i"]): r["resolved"]["id"] for r in rows}
        for how, pad in (("centre", 0.10), ("surface", 0.30)):
            for r in R.key(how=how, pad=pad):
                got = r["resolved"].get("id")
                if got is not None:
                    assert got == base[(r["scene"], r["i"])], \
                        f"{r['scene']} q{r['i']} flips under {how}/{pad}"

    def test_every_reading_is_one_of_the_three_declared_words(self):
        assert {s["reading"] for s in R.SPECS} <= {"solid", "shaky", "open"}

    def test_a_shaky_or_open_reading_carries_a_written_note(self):
        for s in R.SPECS:
            if s["reading"] in ("shaky", "open"):
                assert s.get("note"), f"{s['scene']} q{s['i']} has no note"
