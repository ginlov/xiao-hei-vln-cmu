"""Shared image helpers (engine + logger)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xiao_hei_vln.messages.sensors import ImageFrame

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def image_frame_to_pil(frame: ImageFrame) -> PILImage:
    """Convert an ``ImageFrame`` (BGR8 raw bytes) to a PIL RGB image."""
    import numpy as np
    from PIL import Image  # type: ignore[import-not-found]

    if frame.encoding != "bgr8":
        raise ValueError(f"unsupported image encoding {frame.encoding!r}; expected 'bgr8'")

    arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    rgb = arr[:, :, ::-1]
    return Image.fromarray(rgb, mode="RGB")


def resize_pil(img: PILImage, *, long_edge: int) -> PILImage:
    """Downscale *img* so its longest side is at most *long_edge* pixels."""
    from PIL import Image  # type: ignore[import-not-found]

    longest = max(img.width, img.height)
    if longest <= long_edge:
        return img
    scale = long_edge / longest
    return img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.Resampling.BILINEAR,
    )
