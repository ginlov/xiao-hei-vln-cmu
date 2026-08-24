"""The orbit re-look prompt, and the coercion of what comes back."""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
# `perception` before `scripts`, and both before pytest's own `pythonpath`
# (`src`, `dataset_generator`) -- `dataset_generator/geometry.py` shadows
# `perception/geometry.py` otherwise, and only the latter has FACE_SIZE.
for _p in (REPO / "scripts", REPO / "perception"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geometry as G  # noqa: E402
import reference_view as RV  # noqa: E402


class TestPromptIsBlind:
    """The re-look must not be told what it is confirming — TASK 55."""

    def test_build_prompt_has_nowhere_to_put_the_running_estimate(self):
        params = set(inspect.signature(RV.build_prompt).parameters)
        assert params == {"phrase", "has_crop", "size"}

    def test_look_has_nowhere_to_put_the_running_estimate(self):
        params = set(inspect.signature(RV.look).parameters)
        assert params == {"faces", "phrase", "crop", "model"}

    @pytest.mark.parametrize("word", [
        "estimate", "so far", "running total", "previous box", "current box",
        "accumulated", "consensus", "we think it is at",
    ])
    def test_the_prompt_never_leaks_the_running_estimate(self, word):
        """Words that would only appear if the 3D state had leaked in.

        "centre" is deliberately NOT on this list: the prompt says "the centre
        of your box", which describes the scanner, not our estimate.
        """
        pat = re.compile(rf"\b{re.escape(word)}\b")
        for has_crop in (True, False):
            assert not pat.search(RV.build_prompt("the vase",
                                                  has_crop=has_crop).lower())

    def test_the_prompt_asks_for_no_distance(self):
        """The grounding call asks for `distance_m`; this one must not.

        Range is the channel TASK 26 measured as worse than a constant, and
        here the lidar supplies it. Asking would invite the model to reason
        about where the object is in metres, which is the state we are hiding.
        """
        for has_crop in (True, False):
            p = RV.build_prompt("the vase", has_crop=has_crop).lower()
            assert "distance_m" not in p and "metres" not in p


class TestPromptText:
    @pytest.mark.parametrize("has_crop", [True, False])
    def test_it_renders_with_no_leftover_placeholders(self, has_crop):
        """The JSON schema legitimately contains braces; a `{name}` does not."""
        p = RV.build_prompt("the pillow on the sofa", has_crop=has_crop)
        left = re.findall(r"\{[a-z_]+\}", p)
        assert left == []

    @pytest.mark.parametrize("has_crop", [True, False])
    def test_an_empty_placeholder_leaves_no_double_space(self, has_crop):
        """`{which}` is empty without a crop; the sentence must still read.

        Checked on the one line the substitution touches, not the whole prompt
        -- the heading block is aligned on purpose.
        """
        p = RV.build_prompt("the pillow on the sofa", has_crop=has_crop)
        line = next(x for x in p.splitlines() if "perspective views" in x)
        assert "  " not in line
        assert line.startswith("The ") and "four images are" in line

    def test_it_names_the_object_and_its_plural(self):
        p = RV.build_prompt("the red pillow closest to the sushi", has_crop=True)
        assert "the red pillow closest to the sushi" in p
        assert "several pillows" in p

    def test_the_crop_variant_refers_to_a_first_image(self):
        p = RV.build_prompt("the vase", has_crop=True)
        assert "FIRST image" in p and "last four images" in p

    def test_the_no_crop_variant_refers_to_no_earlier_view(self):
        p = RV.build_prompt("the vase", has_crop=False)
        assert "FIRST image" not in p
        assert "no earlier view" in p
        # and must not tell the model to compare against something absent
        assert "The earlier view is the reference" not in p

    def test_it_asks_for_the_coordinate_space_explicitly(self):
        """TASK 26: the pixels-vs-thousandths bug that inverted a conclusion."""
        p = RV.build_prompt("the vase", has_crop=True)
        assert "coord_space" in p
        assert "normalized_1000" in p and "pixels" in p

    def test_it_offers_not_found_as_a_correct_answer(self):
        p = RV.build_prompt("the vase", has_crop=True)
        assert "found: false" in p
        assert "more damaging than not answering" in p

    def test_it_asks_how_much_of_the_object_is_in_frame(self):
        p = RV.build_prompt("the vase", has_crop=True)
        for word in RV.EXTENTS:
            assert f'"{word}"' in p

    def test_the_face_size_is_stated(self):
        p = RV.build_prompt("the vase", has_crop=True, size=640)
        assert "640 x 640" in p


class TestNormalise:
    def base(self, **kw):
        out = {"coord_space": "pixels", "found": True, "image_index": 1,
               "box_2d": [100, 200, 300, 400], "same_object": True,
               "extent": "whole", "confidence": 0.8}
        out.update(kw)
        return out

    def test_a_good_reply_survives_and_gains_pixel_coords(self):
        got = RV.normalise(self.base(), has_crop=True)
        assert got["found"] is True
        assert got["box_px"] == [100, 200, 300, 400]
        assert got["image_index"] == 1
        assert got["had_crop"] is True

    def test_a_missing_box_is_not_found(self):
        got = RV.normalise(self.base(box_2d=None), has_crop=True)
        assert got["found"] is False and got["box_px"] is None
        assert got["why_not"] == "not_in_view"

    def test_a_missing_image_index_is_not_found(self):
        got = RV.normalise(self.base(image_index=None), has_crop=True)
        assert got["found"] is False

    def test_a_different_instance_is_not_found(self):
        """The arabic_room failure: a right-type box on the wrong object."""
        got = RV.normalise(self.base(same_object=False), has_crop=True)
        assert got["found"] is False
        assert got["why_not"] == "cannot_tell_which_instance"

    def test_a_thousandths_box_is_converted_to_pixels(self):
        got = RV.normalise(self.base(coord_space="normalized_1000",
                                     box_2d=[500, 500, 600, 600]), has_crop=True)
        assert got["found"] is True
        assert max(got["box_px"]) <= G.FACE_SIZE
        assert got["box_px"][0] == pytest.approx(320, abs=1)

    def test_a_declaration_the_arithmetic_refutes_is_overridden(self):
        """A coordinate larger than the image is not a pixel, whatever it says."""
        got = RV.normalise(self.base(coord_space="pixels",
                                     box_2d=[100, 200, 900, 990]), has_crop=True)
        assert got["found"] is True
        assert max(got["box_px"]) <= G.FACE_SIZE

    def test_an_unknown_extent_falls_back_to_whole(self):
        assert RV.normalise(self.base(extent="mostly"),
                            has_crop=True)["extent"] == "whole"

    @pytest.mark.parametrize("bad", [None, "high", float("nan")])
    def test_a_bad_confidence_becomes_a_number_in_range(self, bad):
        c = RV.normalise(self.base(confidence=bad), has_crop=True)["confidence"]
        assert 0.0 <= c <= 1.0

    def test_confidence_is_clamped_not_rejected(self):
        assert RV.normalise(self.base(confidence=3.0),
                            has_crop=True)["confidence"] == 1.0

    def test_found_defaults_false_when_absent(self):
        got = RV.normalise({"box_2d": [1, 2, 3, 4], "image_index": 0},
                           has_crop=False)
        assert got["found"] is False


class TestViewWeight:
    def test_weight_is_the_point_count_when_the_view_is_whole_and_certain(self):
        r = {"extent": "whole", "confidence": 1.0}
        assert RV.view_weight(r, 400) == pytest.approx(400)

    def test_a_slice_counts_for_less(self):
        whole = RV.view_weight({"extent": "whole", "confidence": 1.0}, 400)
        cut = RV.view_weight({"extent": "cut_off", "confidence": 1.0}, 400)
        occl = RV.view_weight({"extent": "occluded", "confidence": 1.0}, 400)
        assert cut < occl < whole

    def test_a_cautious_view_still_counts(self):
        w = RV.view_weight({"extent": "whole", "confidence": 0.0}, 400)
        assert w >= 0.4 * 400

    def test_the_discounts_can_be_turned_off_for_the_ab(self):
        r = {"extent": "cut_off", "confidence": 0.3}
        assert RV.view_weight(r, 400, use_extent=False) == pytest.approx(400)

    def test_zero_points_is_zero_weight_not_negative(self):
        assert RV.view_weight({"extent": "whole", "confidence": 1.0}, 0) == 0.0
        assert RV.view_weight({"extent": "whole", "confidence": 1.0}, -5) == 0.0

    def test_every_extent_has_a_weight(self):
        assert set(RV.EXTENT_WEIGHT) == set(RV.EXTENTS)


class TestLookNeverRaises:
    def test_a_failing_call_returns_not_found(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no key")
        monkeypatch.setattr(RV, "ask_claude", boom)
        got = RV.look([b"", b"", b"", b""], "the vase")
        assert got["found"] is False and got["why_not"] == "call_failed"

    def test_an_unparseable_reply_returns_not_found(self, monkeypatch):
        monkeypatch.setattr(RV, "ask_claude", lambda *a, **k: "not json at all")
        got = RV.look([b"", b"", b"", b""], "the vase")
        assert got["found"] is False and got["why_not"] == "unparseable"

    def test_the_crop_goes_through_previous_not_as_a_fifth_image(self, monkeypatch):
        """`ask_claude` labels image i with NAMES[i], and NAMES is four long.

        The first loft drive died here: the crop was prepended to `images`, so
        the call raised IndexError on the fifth and every crop-carrying re-look
        came back `call_failed`. Sending it as `previous` keeps the faces
        numbered 0-3, which is what `image_index` means.
        """
        seen = {}

        def fake(prompt, images, model, previous=None, **k):
            seen["n"] = len(images)
            seen["previous"] = previous
            return '{"found": false}'
        monkeypatch.setattr(RV, "ask_claude", fake)
        RV.look([b"f0", b"f1", b"f2", b"f3"], "the vase", crop=b"CROP")
        assert seen["n"] == 4, "the four faces, and only the four faces"
        assert seen["previous"] == b"CROP"


class TestTheRealCallAcceptsWhatLookSends:
    """The mock above cannot catch a signature the real callee rejects.

    `look` is tested against a fake `ask_claude` everywhere else, which is why
    a five-image call passed 44 tests and then failed on the robot. This runs
    the real `ask_claude` against a fake SDK instead, so the contract between
    the two is exercised even though no request leaves the machine.
    """

    @pytest.fixture
    def sdk(self, monkeypatch):
        """A stand-in `anthropic` module that records the content block."""
        sent = {}

        class Msg:
            stop_reason = "end_turn"
            usage = type("U", (), {"output_tokens": 10})()
            content = [type("B", (), {"type": "text",
                                      "text": '{"found": false}'})()]

        class Messages:
            def create(self, *, model, max_tokens, messages):
                sent["content"] = messages[0]["content"]
                return Msg()

        class Anthropic:
            def __init__(self, **kw):
                self.messages = Messages()

        mod = type(sys)("anthropic")
        mod.Anthropic = Anthropic
        monkeypatch.setitem(sys.modules, "anthropic", mod)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
        return sent

    def faces(self):
        return [b"f0", b"f1", b"f2", b"f3"]

    def test_a_re_look_with_a_crop_does_not_raise(self, sdk):
        got = RV.look(self.faces(), "the vase", crop=b"CROP")
        assert got.get("why_not") != "call_failed", got.get("error")

    def test_the_faces_keep_their_headings_when_a_crop_is_sent(self, sdk):
        RV.look(self.faces(), "the vase", crop=b"CROP")
        labels = [b["text"] for b in sdk["content"] if b["type"] == "text"]
        for i, name in enumerate(["front", "right", "back", "left"]):
            assert any(t.startswith(f"image {i} ({name}") for t in labels), \
                f"face {i} lost its heading label: {labels}"

    def test_the_crop_arrives_before_the_faces(self, sdk):
        RV.look(self.faces(), "the vase", crop=b"CROP")
        kinds = [b["type"] for b in sdk["content"]]
        first_image = kinds.index("image")
        # the caption for the earlier view, then the earlier view itself,
        # and only then "image 0 (front...)"
        assert sdk["content"][first_image - 1]["text"].startswith("previous view")
        assert kinds.count("image") == 5, "the crop plus the four faces"

    def test_without_a_crop_only_the_faces_are_sent(self, sdk):
        RV.look(self.faces(), "the vase")
        assert [b["type"] for b in sdk["content"]].count("image") == 4

    def test_without_a_crop_only_the_faces_are_sent(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(RV, "ask_claude",
                            lambda p, images, m, **k: seen.setdefault("n", len(images))
                            and '{"found": false}' or '{"found": false}')
        RV.look([b"f0", b"f1", b"f2", b"f3"], "the vase")
        assert seen["n"] == 4
