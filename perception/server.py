"""Perception sidecar — Phase 2 server.

FastAPI app wrapping :class:`perception.pipeline.PerceptionPipeline`.
Models load on startup (blocking — uvicorn won't accept traffic until
the lifespan event finishes), then ``/detect`` posts run YOLO-World +
SAM 2.1 + mask reprojection over the wire.

Endpoints
---------
GET  /healthz         → { model_loaded, gpu_available, schema_version, notes }
POST /reload_classes  → cache the open-vocab class list (and refresh the
                        YOLO-World prompt embeddings)
POST /detect          → multipart image + form fields → list of
                        DetectionRecord (label, score, bbox_xyxy, mask_rle)
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0.0"

log = logging.getLogger("perception")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


# ---------------------------------------------------------------------------
# Response models — wire contract for the PerceptionResponder (Phase 3).
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    mask_rle: str = Field(
        description=(
            "base64-encoded COCO-style RLE of a (EQUIRECT_H, EQUIRECT_W) "
            "binary mask in equirectangular pixel coordinates."
        ),
    )


class DetectResponse(BaseModel):
    detections: list[Detection] = Field(default_factory=list)
    inference_ms: float
    schema_version: str = SCHEMA_VERSION


class HealthResponse(BaseModel):
    model_loaded: bool
    gpu_available: bool
    schema_version: str = SCHEMA_VERSION
    notes: str = ""


class ReloadResponse(BaseModel):
    ok: bool
    n_classes: int


# ---------------------------------------------------------------------------
# Lifespan — load the pipeline once.
# ---------------------------------------------------------------------------


_pipeline = None      # populated in lifespan
_started_at = time.monotonic()


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Block uvicorn startup until models are ready; tear them down
    cleanly on shutdown."""
    global _pipeline
    from perception.pipeline import PerceptionPipeline

    log.info("loading perception pipeline (cuda=%s)", _gpu_available())
    _pipeline = PerceptionPipeline()
    log.info("pipeline ready; classes=%d", len(_pipeline.current_classes))
    yield
    log.info("shutting down")
    _pipeline = None


app = FastAPI(
    title="xiao-hei perception sidecar",
    version=SCHEMA_VERSION,
    description=(
        "YOLOv8x-World v2 + SAM 2.1 Hiera Tiny over an equirectangular "
        "360°×120° camera. Detection runs on 4 perspective faces; masks "
        "are reprojected back into equirectangular pixel coordinates "
        "before they leave the sidecar."
    ),
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    n = len(_pipeline.current_classes) if _pipeline is not None else 0
    return HealthResponse(
        model_loaded=_pipeline is not None,
        gpu_available=_gpu_available(),
        notes=(
            f"up {time.monotonic() - _started_at:.0f}s; "
            f"{n} classes cached"
        ),
    )


@app.post("/reload_classes", response_model=ReloadResponse)
def reload_classes(payload: dict[str, Any]) -> ReloadResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not yet loaded")
    classes = payload.get("classes", [])
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        return ReloadResponse(ok=False, n_classes=len(_pipeline.current_classes))
    n = _pipeline.set_classes(classes)
    return ReloadResponse(ok=True, n_classes=n)


@app.post("/detect", response_model=DetectResponse)
async def detect(
    image: UploadFile = File(..., description="JPEG/PNG of the equirectangular frame"),
    classes: str = Form(
        "",
        description=(
            "Comma-separated class names. Empty falls back to the cached "
            "list from the last /reload_classes call."
        ),
    ),
    score_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.5),
) -> DetectResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not yet loaded")

    payload = await image.read()
    pil = Image.open(io.BytesIO(payload)).convert("RGB")
    # PIL gives RGB; convert to BGR ndarray to match the responder's
    # snapshot.image convention (ros2 sensor_msgs/Image is BGR8).
    rgb = np.asarray(pil, dtype=np.uint8)
    bgr = rgb[..., ::-1].copy()

    class_list = [c.strip() for c in classes.split(",") if c.strip()] or None

    t0 = time.perf_counter()
    detections = _pipeline.detect(
        bgr,
        classes=class_list,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    log.info(
        "detect: %d bytes, %d classes → %d detections, %.1f ms",
        len(payload), len(_pipeline.current_classes),
        len(detections), inference_ms,
    )

    return DetectResponse(
        detections=[
            Detection(
                label=d.label,
                score=d.score,
                bbox_xyxy=list(d.bbox_xyxy),
                mask_rle=d.mask_rle,
            )
            for d in detections
        ],
        inference_ms=inference_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.environ.get("PERCEPTION_HOST", "0.0.0.0"),
        port=int(os.environ.get("PERCEPTION_PORT", "8001")),
    )
