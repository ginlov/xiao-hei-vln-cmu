"""HTTP client for the perception sidecar.

Talks to the FastAPI service from ``perception/server.py``. Two methods
matter at runtime:

* :meth:`HTTPPerceptionClient.set_classes` — pushes the open-vocab
  class list. Dedups against the last list so a no-op call is free.
* :meth:`HTTPPerceptionClient.detect` — JPEG-encodes a BGR image,
  POSTs ``/detect``, decodes the response (bbox + RLE → numpy bool
  masks).

Error model is intentionally permissive: any HTTP / network / decode
failure logs and returns an empty detection list. The responder treats
a tick with no detections as "we didn't see anything new" — same
behaviour as a real perception system on a glare frame or motion blur.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from xiao_hei_vln.perception.geometry import EQUIRECT_H, EQUIRECT_W

log = logging.getLogger(__name__)

DEFAULT_BASE_URL: str = "http://localhost:8001"
DEFAULT_REQUEST_TIMEOUT_S: float = 2.0
DEFAULT_HEALTH_TIMEOUT_S: float = 60.0


@dataclass
class Detection:
    """One detection lifted from a sidecar ``/detect`` response.

    ``mask`` is decoded eagerly to a ``(EQUIRECT_H, EQUIRECT_W)`` bool
    ndarray so the lifter doesn't need pycocotools at runtime.
    """

    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray


class HTTPPerceptionClient:
    """Thin wrapper around the sidecar's HTTP routes."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=request_timeout_s,
        )
        self._cached_classes: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def wait_until_ready(
        self, timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    ) -> bool:
        """Block until ``/healthz`` reports ``model_loaded: true`` or
        the timeout elapses. Returns the final state."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                r = self._client.get("/healthz")
                if r.status_code == 200 and r.json().get("model_loaded"):
                    log.info("perception sidecar ready: %s", r.json())
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        log.warning(
            "perception sidecar did not become ready within %.1f s", timeout_s,
        )
        return False

    # ------------------------------------------------------------------
    # Class list
    # ------------------------------------------------------------------

    def set_classes(self, classes: tuple[str, ...]) -> bool:
        """Push a new class list to the sidecar if it changed.

        Returns ``True`` if a network call was actually made (and
        succeeded). ``False`` means either the cache was already
        warm or the call failed — in either case the caller can
        proceed; ``detect`` will pass ``classes`` explicitly anyway.
        """
        if tuple(classes) == self._cached_classes:
            return False
        try:
            r = self._client.post(
                "/reload_classes",
                json={"classes": list(classes)},
            )
            r.raise_for_status()
            body = r.json()
            if body.get("ok"):
                self._cached_classes = tuple(classes)
                log.info("reload_classes ok: %d classes", body.get("n_classes", 0))
                return True
            log.warning("reload_classes rejected: %s", body)
        except httpx.HTTPError as e:
            log.warning("reload_classes failed: %s", e)
        return False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        classes: tuple[str, ...] | None = None,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> list[Detection]:
        """JPEG-encode and POST the image, parse the response, decode
        masks to ndarrays. Empty list on any failure (logged).
        """
        if image_bgr.shape[:2] != (EQUIRECT_H, EQUIRECT_W):
            log.warning(
                "detect: image shape %s != (%d, %d); skipping",
                image_bgr.shape, EQUIRECT_H, EQUIRECT_W,
            )
            return []

        if classes is not None:
            self.set_classes(classes)

        try:
            jpeg = _encode_jpeg(image_bgr)
        except Exception:
            log.exception("detect: JPEG encode failed")
            return []

        files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
        form: dict[str, str] = {
            "score_threshold": f"{score_threshold:g}",
            "iou_threshold": f"{iou_threshold:g}",
        }
        if classes is not None:
            form["classes"] = ",".join(classes)

        try:
            r = self._client.post("/detect", files=files, data=form)
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPError as e:
            log.warning("detect HTTP failed: %s", e)
            return []
        except ValueError:
            log.exception("detect: response was not JSON")
            return []

        return [_parse_detection(d) for d in body.get("detections", [])]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_jpeg(image_bgr: np.ndarray, quality: int = 90) -> bytes:
    """BGR ndarray → JPEG bytes via Pillow (RGB ordering inside)."""
    rgb = image_bgr[..., ::-1]               # BGR → RGB; ascontiguousarray-safe
    pil = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _parse_detection(payload: dict) -> Detection:
    bbox = tuple(float(v) for v in payload["bbox_xyxy"])
    return Detection(
        label=str(payload["label"]),
        score=float(payload["score"]),
        bbox_xyxy=bbox,        # type: ignore[arg-type]
        mask=_decode_mask_rle(payload["mask_rle"], EQUIRECT_H, EQUIRECT_W),
    )


def _decode_mask_rle(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode a base64-wrapped COCO RLE into a ``(H, W)`` bool ndarray.

    Uses ``pycocotools.mask.decode`` when available — the C-accelerated
    path matches the sidecar's encoder exactly. Falls back to a tiny
    pure-Python decoder when pycocotools isn't installed (CI without
    native deps).
    """
    counts = base64.b64decode(rle_str.encode("ascii"))
    try:
        from pycocotools import mask as coco_mask

        rle = {"counts": counts, "size": [height, width]}
        return coco_mask.decode(rle).astype(bool)
    except ImportError:
        return _decode_coco_rle_py(counts, height, width)


def _decode_coco_rle_py(counts: bytes, height: int, width: int) -> np.ndarray:
    """Pure-Python decoder for COCO's variable-length integer RLE.

    Used only when ``pycocotools`` isn't installed. Matches the
    reference implementation byte-for-byte but is ~10× slower; fine for
    tests but the production responder should pull in pycocotools.

    See ``cocoapi/PythonAPI/pycocotools/mask.py`` for the encoding
    spec — runs alternate between 0 and 1 (starting with 0), each run
    length is a 6-bit-per-byte signed varint.
    """
    runs: list[int] = []
    i = 0
    n = len(counts)
    while i < n:
        x = 0
        k = 0
        more = True
        while more:
            c = counts[i] - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if runs:
            x += runs[-2] if len(runs) >= 2 else 0
        runs.append(x)
    out = np.zeros(height * width, dtype=bool)
    pos = 0
    val = False
    for r in runs:
        if val:
            out[pos:pos + r] = True
        pos += r
        val = not val
    return out.reshape((width, height)).T          # Fortran order in encoder
