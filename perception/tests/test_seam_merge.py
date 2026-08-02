"""Tests for face-seam duplicate merging.

No GPU, no torch, no model weights — ``merge_seam_duplicates`` is a pure
function over already-projected equirect masks, which is the whole reason it
was factored out of ``detect()``.

The behaviour under test is a trade: merging fragments of one object that the
seams split apart, *without* collapsing two genuinely adjacent objects of the
same label. Both directions are asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.geometry import EQUIRECT_H, EQUIRECT_W, FACE_SIZE
from perception.pipeline import (
    _FaceDetection,
    _touches_face_border,
    merge_seam_duplicates,
)


def _mask(u0: int, u1: int, v0: int = 200, v1: int = 400) -> np.ndarray:
    """Half-open column range, so ``_mask(a, b)`` and ``_mask(b, c)`` abut
    exactly the way real cross-face fragments do — the inverse LUT gives each
    equirect pixel one owning face, so they never share a pixel."""
    m = np.zeros((EQUIRECT_H, EQUIRECT_W), dtype=bool)
    m[v0:v1, u0:u1] = True
    return m


def _det(
    face_idx: int,
    label: str,
    u0: int,
    u1: int,
    *,
    score: float = 0.9,
    face_bbox: tuple[float, float, float, float] = (100.0, 100.0, 300.0, 300.0),
) -> _FaceDetection:
    """A detection whose face bbox is interior unless told otherwise."""
    return _FaceDetection(
        face_idx=face_idx,
        face_bbox=face_bbox,
        label=label,
        score=score,
        eq_bbox=(float(u0), 200.0, float(u1), 400.0),
        eq_mask=_mask(u0, u1),
    )


# A bbox running into the right-hand face edge — i.e. clipped.
_CLIPPED_RIGHT = (400.0, 100.0, float(FACE_SIZE - 1), 300.0)
_CLIPPED_LEFT = (0.0, 100.0, 200.0, 300.0)


# --- the border test --------------------------------------------------------


class TestTouchesFaceBorder:
    def test_interior_box_does_not_touch(self) -> None:
        assert not _touches_face_border((100.0, 100.0, 300.0, 300.0), FACE_SIZE)

    @pytest.mark.parametrize("bbox", [
        (0.0, 100.0, 200.0, 300.0),                              # left edge
        (400.0, 100.0, float(FACE_SIZE - 1), 300.0),             # right edge
    ])
    def test_vertical_edges_count_as_clipped(self, bbox) -> None:
        assert _touches_face_border(bbox, FACE_SIZE)

    @pytest.mark.parametrize("bbox", [
        (100.0, 0.0, 300.0, 200.0),                              # top edge
        (100.0, 400.0, 300.0, float(FACE_SIZE - 1)),             # bottom edge
    ])
    def test_horizontal_edges_do_not_count(self, bbox) -> None:
        """Seams are vertical. A top/bottom clip is the panorama's ±60°
        vertical crop, which every floor-level detection hits — treating it as
        seam evidence would make `carpet` mergeable with anything adjacent."""
        assert not _touches_face_border(bbox, FACE_SIZE)

    def test_tolerance_is_applied(self) -> None:
        # 2 px short of the edge still counts — SAM masks rarely land exactly
        # on the last pixel column.
        assert _touches_face_border((100.0, 100.0, 637.0, 300.0), FACE_SIZE)


# --- merging ----------------------------------------------------------------


class TestMergeSeamDuplicates:
    def test_empty_and_single_pass_through(self) -> None:
        assert merge_seam_duplicates([]) == []
        one = [_det(0, "chair", 100, 200)]
        assert len(merge_seam_duplicates(one)) == 1

    def test_clipped_pair_across_faces_merges(self) -> None:
        """The case this exists for: one carpet, clipped in both faces.

        Column ranges taken from a real split observed at arabic_room vp_024 —
        face 3 ends at column 719, face 0 starts at 720, zero shared pixels.
        """
        dets = [
            _det(0, "carpet", 100, 260, score=0.94, face_bbox=_CLIPPED_RIGHT),
            _det(1, "carpet", 260, 400, score=0.95, face_bbox=_CLIPPED_LEFT),
        ]
        out = merge_seam_duplicates(dets)
        assert len(out) == 1
        # Union of the fragments, not either one alone.
        assert out[0].eq_mask.sum() == sum(d.eq_mask.sum() for d in dets)
        assert out[0].eq_bbox[0] == 100.0 and out[0].eq_bbox[2] == 400.0
        assert out[0].score == pytest.approx(0.95)      # best of the group

    def test_adjacent_objects_inside_one_face_are_left_alone(self) -> None:
        """Two real pictures side by side, both whole — must stay two.

        This is the failure mode a distance-only rule would cause, and the
        reason the merge requires a clipped fragment.
        """
        dets = [
            _det(0, "picture", 100, 260),
            _det(1, "picture", 260, 400),      # abutting, but neither clipped
        ]
        assert len(merge_seam_duplicates(dets)) == 2

    def test_same_face_duplicates_are_left_alone(self) -> None:
        # Two detections within one face are a detector duplicate, not a seam
        # split — YOLO's own NMS owns that case.
        dets = [
            _det(0, "chair", 100, 260, face_bbox=_CLIPPED_RIGHT),
            _det(0, "chair", 260, 400, face_bbox=_CLIPPED_RIGHT),
        ]
        assert len(merge_seam_duplicates(dets)) == 2

    def test_different_labels_never_merge(self) -> None:
        dets = [
            _det(0, "carpet", 100, 260, face_bbox=_CLIPPED_RIGHT),
            _det(1, "sofa", 260, 400, face_bbox=_CLIPPED_LEFT),
        ]
        assert len(merge_seam_duplicates(dets)) == 2

    def test_disjoint_masks_never_merge(self) -> None:
        # Clipped and same-label, but nowhere near each other: two objects that
        # happen to sit at different seams.
        dets = [
            _det(0, "carpet", 100, 200, face_bbox=_CLIPPED_RIGHT),
            _det(1, "carpet", 900, 1000, face_bbox=_CLIPPED_LEFT),
        ]
        assert len(merge_seam_duplicates(dets)) == 2

    def test_three_way_split_collapses_to_one(self) -> None:
        """A wide object can cross two seams; pairwise merging must be
        transitive or it would leave a fragment behind."""
        dets = [
            _det(0, "carpet", 100, 300, score=0.8, face_bbox=_CLIPPED_RIGHT),
            _det(1, "carpet", 300, 500, score=0.9, face_bbox=_CLIPPED_LEFT),
            _det(2, "carpet", 500, 700, score=0.7, face_bbox=_CLIPPED_LEFT),
        ]
        out = merge_seam_duplicates(dets)
        assert len(out) == 1
        assert out[0].score == pytest.approx(0.9)
        assert out[0].eq_bbox[0] == 100.0 and out[0].eq_bbox[2] == 700.0

    def test_unrelated_detections_survive_a_merge(self) -> None:
        dets = [
            _det(0, "carpet", 100, 260, face_bbox=_CLIPPED_RIGHT),
            _det(1, "carpet", 260, 400, face_bbox=_CLIPPED_LEFT),
            _det(2, "lamp", 900, 1000),
        ]
        out = merge_seam_duplicates(dets)
        assert sorted(d.label for d in out) == ["carpet", "lamp"]

    def test_masks_are_not_mutated_in_place(self) -> None:
        # The caller still holds these arrays for the debug dump.
        a = _det(0, "carpet", 100, 260, face_bbox=_CLIPPED_RIGHT)
        b = _det(1, "carpet", 260, 400, face_bbox=_CLIPPED_LEFT)
        before = a.eq_mask.sum()
        merge_seam_duplicates([a, b])
        assert a.eq_mask.sum() == before


class TestSeamAdjacency:
    def test_vertically_stacked_fragments_do_not_merge(self) -> None:
        """Sharing a column boundary is not enough — the masks must meet.

        Two same-label objects one above the other can abut in columns while
        being unrelated; the junction-row check is what separates them.
        """
        a = _det(0, "picture", 100, 260, face_bbox=_CLIPPED_RIGHT)
        b = _det(1, "picture", 260, 400, face_bbox=_CLIPPED_LEFT)
        b.eq_mask = _mask(260, 400, v0=500, v1=600)   # different rows entirely
        assert len(merge_seam_duplicates([a, b])) == 2

    def test_small_column_gap_still_merges(self) -> None:
        # SAM mask edges are ragged; a couple of empty columns at the junction
        # must not defeat the merge.
        a = _det(0, "carpet", 100, 258, face_bbox=_CLIPPED_RIGHT)
        b = _det(1, "carpet", 260, 400, face_bbox=_CLIPPED_LEFT)
        assert len(merge_seam_duplicates([a, b])) == 1

    def test_fragments_across_the_wrap_seam_merge(self) -> None:
        """The λ=±π seam puts one fragment at column 0 and the other at the
        last column; the gap must be measured around the cylinder."""
        a = _det(0, "carpet", 0, 120, face_bbox=_CLIPPED_LEFT)
        b = _det(3, "carpet", EQUIRECT_W - 120, EQUIRECT_W,
                 face_bbox=_CLIPPED_RIGHT)
        assert len(merge_seam_duplicates([a, b])) == 1
