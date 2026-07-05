"""Perception pipeline — equirect frame → list of (label, bbox, mask).

Runs inside the perception sidecar (Phase 2). Loads YOLOv8x-World v2 and
SAM 2.1 Hiera Tiny on construction, then handles each ``/detect`` request
by:

1. Decoding the equirect JPEG to BGR ndarray.
2. Unwrapping into 4 perspective faces (precomputed LUTs from
   :mod:`perception.geometry`, applied with ``cv2.remap``).
3. Running YOLO-World on the 4 faces as a single batch.
4. Running SAM 2.1 per detected bbox.
5. Reprojecting each face mask back to equirectangular coordinates.
6. Encoding the equirect masks as base64 COCO RLE for the wire format.

No cross-face NMS for v1 — the responder-side ``SceneRepresentation``
merges duplicates by 3D distance (``merge_radius``), which handles the
~10° face-overlap region naturally. We can revisit if duplicate
detections become a quality problem in Phase 4.
"""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    build_forward_luts,
    build_inverse_lut,
    face_mask_to_equirect_mask,
    project_face_bbox_to_equirect_aabb,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger("perception.pipeline")


@dataclass
class DetectionRecord:
    """A single detection in equirectangular coordinates, ready for the
    wire format. ``server.py`` serialises this into the response Pydantic
    model."""

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask_rle: str


class PerceptionPipeline:
    """Model orchestrator. Construct once at server startup."""

    YOLO_WEIGHTS = "/opt/perception/models/yolov8x-worldv2.pt"
    SAM_WEIGHTS = "/opt/perception/models/sam2.1_hiera_tiny.pt"
    SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

    def __init__(self, *, device: str | None = None) -> None:
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
        out: list[DetectionRecord] = []
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
                masks, _, _ = self._sam.predict(
                    box=bbox[None, :],
                    multimask_output=False,
                )
                # masks: (1, H, W) in float; threshold at 0.5.
                face_mask = masks[0] > 0.5
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
                out.append(DetectionRecord(
                    label=label,
                    score=float(score),
                    bbox_xyxy=eq_bbox,
                    mask_rle=_encode_mask_rle(eq_mask),
                ))

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
            02_faceK_bboxes.png      YOLO boxes drawn on each face
            03_faceK_masks.png       SAM masks overlaid on each face
            04_equirect_overlay.png  masks + boxes reprojected onto equirect
        """
        self._debug_counter += 1
        d = os.path.join(self._debug_dir, f"detect_{self._debug_counter:06d}")
        green, red = (0, 255, 0), (0, 0, 255)
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
