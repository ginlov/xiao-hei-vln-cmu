#!/usr/bin/env python3
"""The 360 image, cut into the four faces the VLM is actually shown.

Every grounding call sends four perspective crops rather than the equirect
frame, because a model asked to point at something in an equirect image points
at the wrong place: the distortion is severe enough near the poles that a box
drawn on it does not correspond to a bearing.

This lived in `vlm_sweep.py`, one of the offline measurement scripts, so the
live drive loop imported a sweep harness to take a picture — and that harness
in turn imported the fusion sweep, which existed only to tune the retired
object map. Pulling twenty lines out here let the old stack be removed without
the loop noticing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402
from grab_faces import build_luts  # noqa: E402

# The prompt numbers them clockwise from the vehicle's forward axis. Anything
# that reports an `image_index` is answering in this order.
NAMES = ("front", "right", "back", "left")


def faces_of(eq: np.ndarray, quality: int = 95) -> list[bytes]:
    """Four JPEG-encoded perspective views, in `NAMES` order."""
    out = []
    for map_x, map_y in build_luts(G.FACE_SIZE):
        f = cv2.remap(eq, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_WRAP)
        out.append(cv2.imencode(".jpg", f,
                                [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())
    return out
