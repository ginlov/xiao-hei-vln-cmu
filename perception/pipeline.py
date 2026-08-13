"""Perception pipeline — equirect frame → list of (label, bbox, mask).

Runs inside the perception sidecar (Phase 2). Loads YOLO-World and
SAM 2.1 Hiera Large on construction, then handles each ``/detect`` request
by:

1. Decoding the equirect JPEG to BGR ndarray.
2. Unwrapping into 4 perspective faces (precomputed LUTs from
   :mod:`perception.geometry`, applied with ``cv2.remap``).
3. Running YOLO-World open-vocab detection on the 4-face batch.
4. Running SAM 2.1 per detected bbox.
5. Reprojecting each face mask back to equirectangular coordinates.
6. Encoding the equirect masks as base64 COCO RLE for the wire format.

7. Merging detections that the face seams split in two
   (:func:`merge_seam_duplicates`).

Step 7 replaces the original v1 plan, which had no cross-face NMS at all
and leaned on the responder-side ``SceneRepresentation.merge_radius``
(1.5 m) to absorb duplicates. That path no longer exists — fusion is now
``ObjectMap``, whose ``MERGE_DIST`` is 0.4 m — and measurement showed the
seams are a real source of fragmentation: over one 316-frame scene, 88% of
same-label detection pairs lying <1 m apart *within a single frame* fell in
a 30° band around a seam bearing (uniform would be 11% per 10° bin). A
carpet split at a seam surfaced as two nodes 0.48 m apart, just wide enough
to escape ``MERGE_DIST``.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    FACE_SIZE,
    build_forward_luts,
    build_inverse_lut,
    face_mask_to_equirect_mask,
    project_face_bbox_to_equirect_aabb,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger("perception.pipeline")

# --- face-seam duplicate merging --------------------------------------------
#
# The 4 faces are 100° wide with ~10° of overlap, so an object narrower than
# 10° of arc always lands whole inside at least one face. A *wider* one does
# not: it is clipped by the border in both faces, and neither detection covers
# it. At 3 m, 10° is only 0.5 m, so this catches ordinary furniture.
#
# Merging is deliberately conservative. It fires only when the two fragments
# come from DIFFERENT faces, carry the SAME label, have overlapping equirect
# masks, and at least one of them is clipped by its face border. That last
# condition is what keeps two genuinely adjacent same-label objects — both
# sitting whole inside one face — from being collapsed into one.

# Distance (px) from the face edge within which a bbox counts as clipped.
# Only the LEFT and RIGHT edges are evidence of a seam split: the seams are
# vertical, so a top/bottom clip is just the panorama's ±60° vertical crop —
# which every floor-level detection hits and which says nothing about faces.
SEAM_BORDER_PX: float = 2.0
# Equirect column gap (px) below which two fragments count as adjacent.
#
# They are adjacent rather than overlapping: the inverse LUT assigns each
# equirect pixel to exactly one owning face, so cross-face masks partition the
# sphere and their intersection is always empty — measured on a real
# seam-split carpet, the two halves ended at column 719 and began at 720.
# Testing for overlap therefore never fires; adjacency is the real signal.
SEAM_ADJACENCY_PX: int = 3


@dataclass
class DetectionRecord:
    """A single detection in equirectangular coordinates, ready for the
    wire format. ``server.py`` serialises this into the response Pydantic
    model."""

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_rle: str
    sam_score: float = 1.0        # SAM's predicted mask IoU (mask quality)


@dataclass
class _FaceDetection:
    """One detection, still carrying the face it came from.

    ``merge_seam_duplicates`` needs the face index and the face-space bbox to
    tell a clipped fragment from a complete object; both are dropped once the
    merge is done and the record goes on the wire.
    """

    face_idx: int
    face_bbox: tuple[float, float, float, float]
    label: str
    score: float
    eq_bbox: tuple[float, float, float, float]
    eq_mask: NDArray[np.bool_]
    sam_score: float = 1.0        # SAM's predicted IoU for this mask


def _touches_face_border(
    bbox: tuple[float, float, float, float],
    face_size: int,
    tol: float = SEAM_BORDER_PX,
) -> bool:
    """True when a face-space bbox runs into the LEFT or RIGHT face edge.

    Such a box is evidence the object continues past the edge into the
    neighbouring face, rather than genuinely ending there. Vertical edges only:
    the seams are vertical, so a top/bottom clip comes from the panorama's
    vertical crop and is not evidence of anything.
    """
    x0, _, x1, _ = bbox
    return x0 <= tol or x1 >= face_size - 1 - tol


def _mask_span(mask: NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    """``(col_min, col_max, row_min, row_max)`` of a mask, or None if empty."""
    cols = np.flatnonzero(mask.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def _seam_adjacent(
    a: NDArray[np.bool_], span_a: tuple[int, int, int, int],
    b: NDArray[np.bool_], span_b: tuple[int, int, int, int],
    *, tol: int, width: int,
) -> bool:
    """True when two equirect masks meet along a shared vertical boundary.

    The column axis is a cylinder — column ``width-1`` neighbours column ``0``
    at the λ=±π seam — so the gap is measured in both directions modulo the
    width, and the smaller one wins. The masks must then actually occupy
    overlapping *rows* in the columns around that junction: two same-label
    objects stacked one above the other share a column boundary without being
    the same thing.
    """
    a_c0, a_c1, _, _ = span_a
    b_c0, b_c1, _, _ = span_b

    # Distance travelling right from a's end to b's start, and vice versa.
    gap_ab = (b_c0 - a_c1) % width
    gap_ba = (a_c0 - b_c1) % width
    if gap_ab <= gap_ba:
        gap, junction = gap_ab, a_c1
    else:
        gap, junction = gap_ba, b_c1
    if gap > tol:
        return False

    band = (np.arange(junction - tol, junction + gap + tol + 1)) % width
    return bool((a[:, band].any(axis=1) & b[:, band].any(axis=1)).any())


def merge_seam_duplicates(
    dets: list[_FaceDetection],
    *,
    face_size: int = FACE_SIZE,
    adjacency_px: int = SEAM_ADJACENCY_PX,
) -> list[_FaceDetection]:
    """Union fragments of one object that adjacent faces split apart.

    Pure function over already-projected equirect masks — no models, no CUDA —
    so it is unit-testable on its own. Grouping is transitive (union-find): a
    very wide object can be clipped across three faces, and merging only
    neighbouring pairs would leave a fragment behind.

    The surviving record keeps the highest score of the group and the union of
    the masks, so the lifter sees the whole object and produces one node.
    """
    n = len(dets)
    if n < 2:
        return list(dets)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    clipped = [_touches_face_border(d.face_bbox, face_size) for d in dets]
    spans = [_mask_span(d.eq_mask) for d in dets]
    width = dets[0].eq_mask.shape[1]
    for i in range(n):
        if spans[i] is None:
            continue
        for j in range(i + 1, n):
            a, b = dets[i], dets[j]
            if a.face_idx == b.face_idx or a.label != b.label:
                continue
            # Only a clipped fragment can be half of a seam-split object.
            if not (clipped[i] or clipped[j]) or spans[j] is None:
                continue
            if not _seam_adjacent(a.eq_mask, spans[i], b.eq_mask, spans[j],
                                  tol=adjacency_px, width=width):
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: list[_FaceDetection] = []
    for root in sorted(groups):
        members = groups[root]
        if len(members) == 1:
            merged.append(dets[members[0]])
            continue
        best = max(members, key=lambda i: dets[i].score)
        mask = dets[members[0]].eq_mask.copy()
        for i in members[1:]:
            mask |= dets[i].eq_mask
        boxes = np.array([dets[i].eq_bbox for i in members], dtype=float)
        merged.append(_FaceDetection(
            face_idx=dets[best].face_idx,
            face_bbox=dets[best].face_bbox,
            label=dets[best].label,
            score=dets[best].score,
            eq_bbox=(
                float(boxes[:, 0].min()), float(boxes[:, 1].min()),
                float(boxes[:, 2].max()), float(boxes[:, 3].max()),
            ),
            eq_mask=mask,
            sam_score=dets[best].sam_score,
        ))
    return merged


class PerceptionPipeline:
    """Model orchestrator. Construct once at server startup."""

    # YOLO-World v2 (YOLOv8x) — open-vocab detector. Fast enough that SAM
    # dominates the per-frame cost; its box score feeds the lift directly and
    # SAM refines localisation afterwards. Weights are baked into the image
    # (see the Dockerfile).
    YOLO_WEIGHTS = "/opt/perception/models/yolov8x-worldv2.pt"
    # SAM 2.1 Hiera Large — best mask quality. VRAM is not the constraint on the
    # A10G (measured: sidecar 2.2 GB tiny → 3.3 GB large under load, on a 22 GB
    # card), and Task 1/2 answer within a 10-min budget, not real time, so the
    # extra latency (~1.0 → ~1.5 s/frame) is affordable. Tiny remains in
    # models/ for a quick revert.
    SAM_WEIGHTS = "/opt/perception/models/sam2.1_hiera_large.pt"
    SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    def __init__(
        self, *, device: str | None = None, merge_seams: bool | None = None,
    ) -> None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLOWorld

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading YOLOWorld from %s", self.YOLO_WEIGHTS)
        self._yolo = YOLOWorld(self.YOLO_WEIGHTS)
        self._yolo.to(self._device)

        log.info("loading SAM2 (%s) from %s", self.SAM_CONFIG, self.SAM_WEIGHTS)
        sam_model = build_sam2(self.SAM_CONFIG, self.SAM_WEIGHTS, device=self._device)
        self._sam = SAM2ImagePredictor(sam_model)

        log.info("building equirect ⇄ face LUTs (one-time)")
        self._fwd_luts = build_forward_luts()
        self._inv_lut = build_inverse_lut()

        self._current_classes: list[str] = []

        # Face-seam merging is on by default; PERCEPTION_MERGE_SEAMS=0 turns it
        # off so the A/B can be measured without rebuilding the image.
        self._merge_seams = (
            merge_seams if merge_seams is not None
            else os.environ.get("PERCEPTION_MERGE_SEAMS", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )

        # Debug mode: when PERCEPTION_DEBUG is truthy, every /detect call
        # dumps per-step visualisations (equirect → faces → bboxes →
        # masks → reprojected equirect) under PERCEPTION_DEBUG_DIR.
        self._debug = os.environ.get("PERCEPTION_DEBUG", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        self._debug_dir = os.environ.get("PERCEPTION_DEBUG_DIR", "/opt/perception/debug")
        self._debug_counter = 0
        if self._debug:
            log.info("PERCEPTION_DEBUG on → step images to %s", self._debug_dir)

        log.info("pipeline ready (device=%s)", self._device)

    # ------------------------------------------------------------------
    # Class-list management
    # ------------------------------------------------------------------

    def set_classes(self, classes: list[str]) -> int:
        """Push a new open-vocab class list into YOLO-World.

        Re-encoding the text prompt is the expensive part (~50 ms on a
        4090 for ~50 classes); subsequent calls with the same list are
        a no-op. Returns the cached class count.
        """
        if classes == self._current_classes:
            return len(self._current_classes)
        self._yolo.set_classes(classes)
        self._current_classes = list(classes)
        log.info("YOLO-World classes set: %d", len(classes))
        return len(classes)

    @property
    def current_classes(self) -> list[str]:
        return list(self._current_classes)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        equirect_bgr: NDArray[np.uint8],
        *,
        classes: list[str] | None = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> list[DetectionRecord]:
        """Detect + segment objects in an equirectangular frame.

        ``classes`` overrides the cached class list for this call. If
        omitted, uses whatever was last set via :meth:`set_classes`.
        Returns detections in equirectangular pixel coordinates.
        """
        if equirect_bgr.shape[:2] != (EQUIRECT_H, EQUIRECT_W):
            raise ValueError(
                f"equirect frame must be ({EQUIRECT_H}, {EQUIRECT_W}); "
                f"got {equirect_bgr.shape[:2]}",
            )
        if classes is not None:
            self.set_classes(classes)
        if not self._current_classes:
            log.warning("detect() with empty class list — returning []")
            return []

        # 1. Unwrap to 4 faces.
        faces = self._unwrap_to_faces(equirect_bgr)

        # 2. YOLO-World on the 4-face batch.
        results = self._yolo(
            faces,
            conf=score_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        # 3 + 4. For each face: SAM per bbox → mask, project back.
        # Collected with their face of origin so step 7 can rejoin fragments.
        found: list[_FaceDetection] = []
        # Per-step artefacts collected only when debugging (cheap no-ops otherwise).
        dbg_face_boxes: list[list[tuple]] = [[] for _ in faces]
        dbg_face_masks: list[list] = [[] for _ in faces]
        dbg_eq_items: list[tuple] = []
        for face_idx, (face_img, face_result) in enumerate(
            zip(faces, results, strict=True),
        ):
            boxes = face_result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu().numpy()          # (N, 4)
            scores = boxes.conf.detach().cpu().numpy()        # (N,)
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

            # Set the image once per face; SAM caches its image features.
            # `.copy()` is required because the [..., ::-1] view has a
            # negative stride, which torch.from_numpy refuses.
            self._sam.set_image(face_img[..., ::-1].copy())     # BGR → RGB
            for bbox, score, cls_id in zip(xyxy, scores, class_ids, strict=True):
                masks, sam_iou, _ = self._sam.predict(
                    box=bbox[None, :],
                    multimask_output=False,
                )
                # masks: (1, H, W) in float; threshold at 0.5. sam_iou: (1,)
                # SAM's own estimate of this mask's IoU with the true object —
                # a mask-quality signal the YOLO box score does not carry.
                face_mask = masks[0] > 0.5
                sam_score = float(sam_iou[0])
                eq_mask = face_mask_to_equirect_mask(
                    face_mask, face_idx, self._inv_lut,
                )
                label = self._current_classes[cls_id]
                if self._debug:
                    dbg_face_boxes[face_idx].append(
                        (tuple(map(float, bbox)), label, float(score)),
                    )
                    dbg_face_masks[face_idx].append(face_mask)
                if not eq_mask.any():
                    continue
                eq_bbox = project_face_bbox_to_equirect_aabb(
                    tuple(map(float, bbox)), face_idx,
                )
                if self._debug:
                    dbg_eq_items.append((eq_bbox, eq_mask, label))
                found.append(_FaceDetection(
                    face_idx=face_idx,
                    face_bbox=tuple(map(float, bbox)),
                    label=label,
                    score=float(score),
                    eq_bbox=eq_bbox,
                    eq_mask=eq_mask,
                    sam_score=sam_score,
                ))

        # 7. Rejoin objects the face seams split (no-op when nothing straddles).
        if self._merge_seams:
            before = len(found)
            found = merge_seam_duplicates(found)
            if len(found) != before:
                log.debug(
                    "seam merge: %d detections -> %d", before, len(found),
                )

        out = [
            DetectionRecord(
                label=d.label,
                score=d.score,
                bbox_xyxy=d.eq_bbox,
                mask_rle=_encode_mask_rle(d.eq_mask),
                sam_score=d.sam_score,
            )
            for d in found
        ]

        if self._debug:
            self._dump_debug(
                equirect_bgr, faces, dbg_face_boxes, dbg_face_masks, dbg_eq_items,
            )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dump_debug(
        self,
        equirect_bgr: NDArray[np.uint8],
        faces: list[NDArray[np.uint8]],
        face_boxes: list[list[tuple]],
        face_masks: list[list[NDArray[np.bool_]]],
        eq_items: list[tuple],
    ) -> None:
        """Save per-step visualisations for one ``/detect`` call.

        Only called when ``PERCEPTION_DEBUG`` is on. One folder per call,
        ``<debug_dir>/detect_NNNNNN/``::

            00_equirect.png          original equirect input
            01_faceK.png             the 4 perspective faces (K = 0..3)
            02_faceK_bboxes.png      detector boxes drawn on each face
            03_faceK_masks.png       SAM masks overlaid on each face
            04_equirect_overlay.png  masks + boxes reprojected onto equirect
        """
        self._debug_counter += 1
        d = os.path.join(self._debug_dir, f"detect_{self._debug_counter:06d}")
        green, red = (0, 255, 0), (0, 0, 255)
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "00_equirect.png"), equirect_bgr)
            for i, face in enumerate(faces):
                cv2.imwrite(os.path.join(d, f"01_face{i}.png"), face)

                bx = face.copy()
                for (x0, y0, x1, y1), label, score in face_boxes[i]:
                    cv2.rectangle(bx, (int(x0), int(y0)), (int(x1), int(y1)), green, 2)
                    cv2.putText(bx, f"{label} {score:.2f}", (int(x0), max(12, int(y0) - 5)),
                                font, 0.5, green, 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(d, f"02_face{i}_bboxes.png"), bx)

                seg = face.copy()
                ov = seg.copy()
                for m in face_masks[i]:
                    ov[m] = red
                cv2.imwrite(
                    os.path.join(d, f"03_face{i}_masks.png"),
                    cv2.addWeighted(ov, 0.5, seg, 0.5, 0),
                )

            eq = equirect_bgr.copy()
            ov = eq.copy()
            for _bbox, mask, _label in eq_items:
                ov[mask] = red
            eq = cv2.addWeighted(ov, 0.5, eq, 0.5, 0)
            for bbox, _mask, label in eq_items:
                x0, y0, x1, y1 = (int(v) for v in bbox)
                cv2.rectangle(eq, (x0, y0), (x1, y1), green, 2)
                cv2.putText(eq, label, (x0, max(12, y0 - 5)), font, 0.5, green, 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(d, "04_equirect_overlay.png"), eq)

            log.info("PERCEPTION_DEBUG: wrote %s (%d detections)", d, len(eq_items))
        except Exception:  # noqa: BLE001 — debug output must never break /detect
            log.exception("PERCEPTION_DEBUG: failed to write debug images to %s", d)

    def _unwrap_to_faces(
        self, equirect_bgr: NDArray[np.uint8],
    ) -> list[NDArray[np.uint8]]:
        """Apply each face's forward LUT with cv2.remap. Returns a list
        of 4 ``(FACE_SIZE, FACE_SIZE, 3)`` BGR images.

        ``BORDER_WRAP`` handles the λ=±π seam transparently — any face
        pixel sampling from across the wrap finds the right column on
        the other side.
        """
        import cv2

        out: list[NDArray[np.uint8]] = []
        for map_x, map_y in self._fwd_luts:
            face = cv2.remap(
                equirect_bgr, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            out.append(face)
        return out


# ---------------------------------------------------------------------------
# Mask encoding — COCO RLE wrapped in base64 for transport
# ---------------------------------------------------------------------------


def _encode_mask_rle(mask: NDArray[np.bool_]) -> str:
    """Encode a 2D boolean mask as base64-wrapped COCO RLE.

    Wire format: a JSON-serialisable string. The receiver decodes it
    with :func:`decode_mask_rle` (which mirrors the
    :mod:`pycocotools.mask` API used here). Compression is typically
    50–200× for our sparse object masks.
    """
    from pycocotools import mask as coco_mask

    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask.encode(fortran)
    return base64.b64encode(rle["counts"]).decode("ascii")


def decode_mask_rle(rle_str: str, height: int, width: int) -> NDArray[np.bool_]:
    """Inverse of :func:`_encode_mask_rle`. Used by the responder client."""
    from pycocotools import mask as coco_mask

    rle = {
        "counts": base64.b64decode(rle_str.encode("ascii")),
        "size": [height, width],
    }
    return coco_mask.decode(rle).astype(bool)
