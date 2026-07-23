"""Class-typical AABB priors for repairing inflated LiDAR lifts.

Used by ObjectMap export and Gemini packaging. Not GT remaps — only size.
"""

from __future__ import annotations

# Metres [sx, sy, sz] — clamp detected extent when absurdly large.
SIZE_PRIOR: dict[str, tuple[float, float, float]] = {
    "bottle": (0.12, 0.12, 0.35),
    "beer bottle": (0.12, 0.12, 0.35),
    "wine bottle": (0.12, 0.12, 0.40),
    "box": (0.40, 0.35, 0.30),
    "round box": (0.30, 0.30, 0.25),
    "phone": (0.08, 0.15, 0.02),
    "cup": (0.10, 0.10, 0.12),
    "bowl": (0.20, 0.20, 0.10),
    "book": (0.35, 0.25, 0.08),
    # Window-sill / stacked books often form a long low run.
    "books": (3.0, 0.45, 0.50),
    "shelf": (2.5, 0.45, 0.50),
    "newspaper": (0.30, 0.40, 0.02),
    "remote": (0.05, 0.18, 0.02),
    "candle": (0.08, 0.08, 0.20),
    "vase": (0.25, 0.25, 0.45),
    "picture": (0.70, 0.05, 0.80),
    "painting": (0.70, 0.05, 0.80),
    "poster": (0.60, 0.05, 0.80),
    "photo": (0.40, 0.05, 0.50),
    "framed record": (0.60, 0.05, 0.60),
    "lamp": (0.35, 0.35, 0.70),
    "wall lamp": (0.25, 0.25, 0.35),
    "ceiling lamp": (0.40, 0.40, 0.25),
    "ceiling light": (0.40, 0.40, 0.20),
    "focus light": (0.20, 0.20, 0.08),
    "spot light": (0.20, 0.20, 0.08),
    "pillow": (0.50, 0.40, 0.20),
    "guitar": (0.40, 0.15, 1.00),
    "potted plant": (0.50, 0.50, 0.80),
    "chair": (1.05, 1.05, 1.20),
    "light switch": (0.08, 0.08, 0.08),
    "window": (2.20, 1.80, 2.20),
    "windows": (2.20, 1.80, 2.20),
}

_LARGE_OK = {
    "wall", "floor", "ceiling", "room", "column", "door", "door frame",
    "window", "windows", "window frame", "exterior walls",
    "couch", "sofa", "table", "coffee table", "dining table", "desk",
    "bookcase", "cabinet", "tv cabinet", "bed", "rug", "carpet",
    "bench", "counter", "kitchen island",
}


def size_prior_for(label: str) -> tuple[float, float, float] | None:
    lab = (label or "").lower().strip()
    if lab in SIZE_PRIOR:
        return SIZE_PRIOR[lab]
    for key, prior in SIZE_PRIOR.items():
        if key in lab or lab.endswith(key):
            return prior
    return None


def clamp_aabb_size(
    label: str,
    size: tuple[float, float, float] | list[float],
) -> list[float]:
    """Clamp absurd extents; keep large furniture mostly intact."""
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    lab = (label or "").lower().strip()
    prior = size_prior_for(lab)
    if lab in {"books", "book", "shelf"}:
        # One horizontal axis is the run (<=3.2 m), the other is depth (<=0.5).
        # If one side is already shallow, treat it as depth; if both are
        # inflated (common LiDAR sheet), assume run along the larger prior X.
        if min(sx, sy) <= 1.2:
            if sx <= sy:
                return [min(sx, 0.50), min(sy, 3.2), min(sz, 0.60)]
            return [min(sx, 3.2), min(sy, 0.50), min(sz, 0.60)]
        return [min(max(sx, sy), 3.2), 0.40, min(sz, 0.60)]
    if lab in {"chair", "armchair", "office chair"}:
        # Detector often emits 2×2 m chair cubes; clamp to seat-like extent.
        if min(sx, sy) > 1.2:
            return [1.05, 1.05, min(sz, 1.3)]
        return [min(sx, 1.2), min(sy, 1.2), min(sz, 1.3)]
    if lab in _LARGE_OK:
        if lab in {"window", "windows", "door", "balcony door", "kitchen door"}:
            return [min(sx, 2.5), min(sy, 2.5), min(sz, 2.8)]
        if lab in {"couch", "sofa"}:
            # Square 4×4 phantoms: keep the longer run, clamp depth.
            if min(sx, sy) > 2.0:
                long = min(max(sx, sy), 3.5)
                return [long, 1.25, min(sz, 1.2)] if sx >= sy else [1.25, long, min(sz, 1.2)]
            return [min(sx, 3.5), min(sy, 2.0), min(sz, 1.2)]
        if lab in {"table", "coffee table", "dining table", "desk"}:
            # Square/inflated table lifts: keep elongated run, clamp depth.
            if min(sx, sy) > 1.8:
                long = min(max(sx, sy), 3.5)
                return [long, 1.2, min(sz, 0.9)] if sx >= sy else [1.2, long, min(sz, 0.9)]
            return [min(sx, 3.5), min(sy, 2.2), min(sz, 1.0)]
        return [min(sx, 4.0), min(sy, 4.0), min(sz, 3.5)]
    if prior is None:
        return [min(sx, 2.0), min(sy, 2.0), min(sz, 2.5)]
    px, py, pz = prior
    return [
        sx if sx <= px * 2.5 else px,
        sy if sy <= py * 2.5 else py,
        sz if sz <= pz * 2.5 else pz,
    ]


def aabb_from_center_size(
    center: tuple[float, float, float] | list[float],
    size: list[float],
) -> tuple[list[float], list[float]]:
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    return (
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz + hz],
    )
