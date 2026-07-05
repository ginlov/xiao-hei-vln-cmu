"""Tests for ``xiao_hei_vln.perception.vocab``."""

from __future__ import annotations

import pytest

from xiao_hei_vln.perception.vocab import (
    DEFAULT_PRIOR,
    Vocabulary,
)


class TestPrior:
    def test_prior_returned_when_no_question(self) -> None:
        v = Vocabulary()
        classes = v.current_classes(None)
        assert classes == DEFAULT_PRIOR

    def test_empty_string_acts_like_none(self) -> None:
        v = Vocabulary()
        assert v.current_classes("") == DEFAULT_PRIOR

    def test_custom_prior_is_normalised_and_returned(self) -> None:
        v = Vocabulary(prior=["  Sushi  ", "RAMEN"])
        assert v.current_classes(None) == ("sushi", "ramen")


class TestQuestionMerge:
    def test_question_nouns_are_appended(self) -> None:
        v = Vocabulary(prior=["chair", "table"])
        classes = v.current_classes("Find the lamp near the sushi")
        assert classes[:2] == ("chair", "table")
        assert "lamp" in classes
        assert "sushi" in classes

    def test_stopwords_dropped(self) -> None:
        v = Vocabulary(prior=[])
        classes = v.current_classes(
            "How many chairs are in the room and where is the bowl",
        )
        for stopword in ("the", "a", "in", "is", "are", "where", "how", "many"):
            assert stopword not in classes

    def test_dedup_against_prior(self) -> None:
        # "chair" already in prior; the question shouldn't duplicate it.
        v = Vocabulary(prior=["chair", "table"])
        classes = v.current_classes("How many chairs are in the room")
        assert classes.count("chair") == 1

    def test_plural_depluralised_to_match_prior(self) -> None:
        v = Vocabulary(prior=["chair", "table"])
        # "chairs" → "chair" → already in prior; "tables" → "table"
        # → already in prior. Any extra non-stopword tokens
        # (e.g. "count") are tolerated — YOLO-World handles them with
        # low embeddings — what matters is no duplicate plural form.
        classes = v.current_classes("Count the chairs and tables")
        assert classes[:2] == ("chair", "table")
        assert "chairs" not in classes
        assert "tables" not in classes

    def test_short_words_dropped(self) -> None:
        v = Vocabulary(prior=[])
        # 1-char tokens never become labels (they're noise from
        # tokenising single letters like "a" or contractions).
        classes = v.current_classes("a b X y z")
        assert classes == ()

    def test_returns_stable_tuple_across_calls(self) -> None:
        v = Vocabulary()
        q = "Find the lamp on the table"
        assert v.current_classes(q) == v.current_classes(q)


class TestEdgeCases:
    def test_word_with_punctuation_around_it(self) -> None:
        v = Vocabulary(prior=[])
        classes = v.current_classes('Where is the "soft-pillow"?')
        # Punctuation is stripped; "soft" and "pillow" both come through.
        assert "soft" in classes
        assert "pillow" in classes

    def test_double_pluralisation_safe(self) -> None:
        # "boxes" is a tricky plural; the depluraliser refuses to
        # over-strip and leaves it alone (YOLO-World handles either).
        v = Vocabulary(prior=[])
        classes = v.current_classes("Count the boxes")
        assert "boxes" in classes

    def test_mixed_case_normalised(self) -> None:
        v = Vocabulary(prior=["chair"])
        classes = v.current_classes("Find the LAMP")
        assert "lamp" in classes
        # Prior stays lowercase too.
        assert "chair" in classes
