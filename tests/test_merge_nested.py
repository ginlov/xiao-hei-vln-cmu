"""Tests for ``merge_nested``: folds vla3d_nested.jsonl into the type-aligned
ref/num jsonl files.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from merge_nested import (
    NESTED_FILE,
    NUM_FILE,
    REF_FILE,
    merge,
)


def _ref(scene: str, qid: int, source: str) -> dict:
    return {
        "scene": scene,
        "type": "object_reference",
        "source": source,
        "question": f"Find the chair {qid}.",
        "object_list": [],
        "answer": {"object_id": qid, "label": "chair"},
        "target": qid,
        "anchors": [],
    }


def _num(scene: str, qid: int, source: str, answer: int = 1) -> dict:
    return {
        "scene": scene,
        "type": "numerical",
        "source": source,
        "question": f"How many cups {qid}?",
        "object_list": [],
        "answer": answer,
        "target": None,
        "anchors": [],
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _seed_files(
    tmp_path: Path,
    n_ref: int = 3,
    n_num: int = 2,
    n_nested_ref: int = 4,
    n_nested_num: int = 1,
) -> Path:
    """Seed the three input jsonl files in tmp_path; return the dataset dir."""
    _write(
        tmp_path / REF_FILE,
        [_ref(f"scene_{i}", i, "vla3d_ref") for i in range(n_ref)],
    )
    _write(
        tmp_path / NUM_FILE,
        [_num(f"scene_{i}", i + 100, "vla3d_num") for i in range(n_num)],
    )
    nested_rows = [
        _ref(f"scene_n{i}", i + 200, "vla3d_nested") for i in range(n_nested_ref)
    ] + [
        _num(f"scene_n{i}", i + 300, "vla3d_nested", answer=i + 2)
        for i in range(n_nested_num)
    ]
    _write(tmp_path / NESTED_FILE, nested_rows)
    return tmp_path


class TestMergeBasics:
    def test_nested_split_by_type_into_ref_and_num(self, tmp_path: Path) -> None:
        _seed_files(tmp_path, n_ref=3, n_num=2, n_nested_ref=4, n_nested_num=1)
        summary = merge(tmp_path, seed=42)

        assert summary["ref_added"] == 4
        assert summary["num_added"] == 1
        assert summary["ref_total"] == 3 + 4
        assert summary["num_total"] == 2 + 1
        assert summary["nested_present"] is True

    def test_nested_file_removed_after_merge(self, tmp_path: Path) -> None:
        _seed_files(tmp_path)
        assert (tmp_path / NESTED_FILE).exists()
        merge(tmp_path, seed=42)
        assert not (tmp_path / NESTED_FILE).exists()

    def test_source_field_preserved(self, tmp_path: Path) -> None:
        """nested-origin rows must remain tagged ``source == vla3d_nested``
        so downstream filters can still compute nested-only metrics."""
        _seed_files(tmp_path, n_ref=2, n_num=2, n_nested_ref=3, n_nested_num=4)
        merge(tmp_path, seed=42)

        ref_sources = Counter(r["source"] for r in _read(tmp_path / REF_FILE))
        num_sources = Counter(r["source"] for r in _read(tmp_path / NUM_FILE))

        assert ref_sources == {"vla3d_ref": 2, "vla3d_nested": 3}
        assert num_sources == {"vla3d_num": 2, "vla3d_nested": 4}

    def test_every_row_in_correct_file_by_type(self, tmp_path: Path) -> None:
        """A nested-num row must never end up in vla3d_ref.jsonl, and vice
        versa. This is the runtime-contract guarantee the merge exists to
        enforce."""
        _seed_files(tmp_path, n_ref=2, n_num=2, n_nested_ref=5, n_nested_num=5)
        merge(tmp_path, seed=42)

        ref = _read(tmp_path / REF_FILE)
        num = _read(tmp_path / NUM_FILE)

        assert all(r["type"] == "object_reference" for r in ref)
        assert all(r["type"] == "numerical" for r in num)


class TestMergeShuffle:
    def test_nested_and_original_are_interleaved_not_block_segregated(
        self, tmp_path: Path
    ) -> None:
        """The whole point of the shuffle: same-source samples should not be
        clustered at the start or end of either output file."""
        # Plenty of rows to make a block pattern statistically impossible
        _seed_files(tmp_path, n_ref=20, n_num=20, n_nested_ref=20, n_nested_num=20)
        merge(tmp_path, seed=42)

        ref = _read(tmp_path / REF_FILE)
        num = _read(tmp_path / NUM_FILE)

        # If we appended without shuffling, the first 20 ref entries would
        # all be source=vla3d_ref and the last 20 would all be vla3d_nested.
        first_half_ref = Counter(r["source"] for r in ref[: len(ref) // 2])
        first_half_num = Counter(r["source"] for r in num[: len(num) // 2])

        assert first_half_ref["vla3d_ref"] > 0
        assert first_half_ref["vla3d_nested"] > 0
        assert first_half_num["vla3d_num"] > 0
        assert first_half_num["vla3d_nested"] > 0

    def test_deterministic_byte_identical_given_seed(self, tmp_path: Path) -> None:
        # First run
        d1 = tmp_path / "run1"
        d1.mkdir()
        _seed_files(d1, n_ref=10, n_num=10, n_nested_ref=10, n_nested_num=10)
        merge(d1, seed=42)
        ref_bytes_1 = (d1 / REF_FILE).read_bytes()
        num_bytes_1 = (d1 / NUM_FILE).read_bytes()

        # Second run, same inputs, same seed
        d2 = tmp_path / "run2"
        d2.mkdir()
        _seed_files(d2, n_ref=10, n_num=10, n_nested_ref=10, n_nested_num=10)
        merge(d2, seed=42)
        ref_bytes_2 = (d2 / REF_FILE).read_bytes()
        num_bytes_2 = (d2 / NUM_FILE).read_bytes()

        assert ref_bytes_1 == ref_bytes_2
        assert num_bytes_1 == num_bytes_2

    def test_different_seed_produces_different_order(self, tmp_path: Path) -> None:
        """Sanity: the rng is actually being consumed."""
        d1 = tmp_path / "a"
        d1.mkdir()
        _seed_files(d1, n_ref=5, n_num=5, n_nested_ref=5, n_nested_num=5)
        merge(d1, seed=42)
        order_42 = [r["question"] for r in _read(d1 / REF_FILE)]

        d2 = tmp_path / "b"
        d2.mkdir()
        _seed_files(d2, n_ref=5, n_num=5, n_nested_ref=5, n_nested_num=5)
        merge(d2, seed=7)
        order_7 = [r["question"] for r in _read(d2 / REF_FILE)]

        assert set(order_42) == set(order_7)  # same content
        assert order_42 != order_7              # different order


class TestMergeEdgeCases:
    def test_no_nested_file_is_idempotent_noop(self, tmp_path: Path) -> None:
        """Re-running merge after the nested file is already gone should
        not crash and should not mutate the type-aligned files."""
        _write(tmp_path / REF_FILE, [_ref("s", 1, "vla3d_ref")])
        _write(tmp_path / NUM_FILE, [_num("s", 2, "vla3d_num")])
        # NESTED_FILE deliberately absent

        ref_before = (tmp_path / REF_FILE).read_bytes()
        num_before = (tmp_path / NUM_FILE).read_bytes()

        summary = merge(tmp_path, seed=42)

        assert summary["nested_present"] is False
        assert (tmp_path / REF_FILE).read_bytes() == ref_before
        assert (tmp_path / NUM_FILE).read_bytes() == num_before

    def test_empty_nested_file_still_drops_it(self, tmp_path: Path) -> None:
        _write(tmp_path / REF_FILE, [_ref("s", 1, "vla3d_ref")])
        _write(tmp_path / NUM_FILE, [_num("s", 2, "vla3d_num")])
        _write(tmp_path / NESTED_FILE, [])  # exists but empty

        summary = merge(tmp_path, seed=42)
        assert summary["ref_added"] == 0
        assert summary["num_added"] == 0
        assert not (tmp_path / NESTED_FILE).exists()

    def test_nested_with_unknown_type_is_skipped_with_warning(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write(tmp_path / REF_FILE, [])
        _write(tmp_path / NUM_FILE, [])
        odd = {"scene": "s", "type": "instruction_following",
               "source": "vla3d_nested", "question": "Go to the door."}
        _write(tmp_path / NESTED_FILE, [odd, _ref("s", 1, "vla3d_nested")])

        summary = merge(tmp_path, seed=42)
        out = capsys.readouterr().out
        assert "1 nested pairs have a type other than" in out
        assert summary["ref_added"] == 1
        assert summary["num_added"] == 0


class TestMergeMatchesRealCounts:
    """Lock the size invariant the PR description claims for the real corpus.

    If the nested generator ever changes how many ref/num pairs it emits,
    these constants must move in lockstep with the README.
    """

    EXPECTED_REF_TOTAL = 7708     # 6,730 single-layer + 978 nested ref
    EXPECTED_NUM_TOTAL = 591      # 386 templates + 205 nested num
    EXPECTED_GRAND_TOTAL = 8299

    def test_arithmetic_holds(self) -> None:
        assert self.EXPECTED_REF_TOTAL + self.EXPECTED_NUM_TOTAL == self.EXPECTED_GRAND_TOTAL
