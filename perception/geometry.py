"""Equirectangular ⇄ perspective face geometry for the perception sidecar.

The CMU challenge `/camera/image` topic publishes a 360° equirectangular
panorama at 1920×640 BGR8, cropped to ±60° vertical FOV. YOLO-World and
SAM 2.1 are trained on perspective imagery and degrade on raw equirect
input, so the sidecar unwraps the panorama into four 640×640 perspective
"faces" (front / right / back / left, 100° square FOV with ~10° overlap
between adjacent faces). YOLO detects per face; SAM segments per face;
the resulting masks are reprojected back into equirectangular pixel
coordinates so the caller (PerceptionResponder) never sees per-face
geometry.

This module is **pure numpy** — no torch, no opencv, no model deps. It's
safe to import from anywhere and the unit tests cover it without a GPU.

Coordinate conventions
----------------------

- Equirectangular pixel coords: ``(u_eq, v_eq)`` with ``u_eq`` along the
  full 360° horizontal axis (left = -180° = back-left wrap, center =
  0° = forward, right = +180°) and ``v_eq`` along the 120° vertical
  axis (top = +60° latitude, bottom = -60°).
- World direction vector ``d = (d_x, d_y, d_z)`` lives in a right-handed
  frame: +z forward, +x right, +y down. This matches the camera
  convention the sensor→camera extrinsic transforms into.
- Per-face local frame: same convention, with the face's optical axis
  along +z; ``FACE_YAWS`` rotates each face around the world +y axis
  (so yaw=0 looks along +z, yaw=π/2 looks along +x, etc.).

Public API
----------

- ``build_forward_luts()`` → per-face ``(map_x, map_y)`` for
  ``cv2.remap(equirect, map_x, map_y, ...)`` → face image. Computed
  once at server startup.
- ``build_inverse_lut()`` → ``(face_idx, u_face, v_face)`` per equirect
  pixel, for mask reprojection.
- ``project_world_to_equirect(d)`` and the inverse — vectorised.
- Constants (``EQUIRECT_W``, ``H_FOV``, ``FACE_SIZE``, …) are the single
  source of truth; ``pipeline.py`` consumes them.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants — match the challenge sensor + our 4-face unwrap design.
# ---------------------------------------------------------------------------

EQUIRECT_W: int = 1920
EQUIRECT_H: int = 640
H_FOV: float = 2.0 * np.pi          # full 360°
V_FOV: float = 2.0 * np.pi / 3.0    # 120° (vertical crop in the sim's panorama)

# Per-face: 100° FOV square gives ~10° overlap between adjacent faces
# (4 × 100° = 400° vs the 360° we need), which means seam-spanning
# objects fall fully inside at least one face. 640×640 keeps the pixel
# density similar to the equirect equator.
FACE_SIZE: int = 640
FACE_FOV: float = np.deg2rad(100.0)
FACE_F: float = (FACE_SIZE / 2.0) / np.tan(FACE_FOV / 2.0)

# Yaw center for each face. Index 0 = front (+z), 1 = right (+x),
# 2 = back (-z), 3 = left (-x).
FACE_YAWS: np.ndarray = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]))
N_FACES: int = len(FACE_YAWS)


# ---------------------------------------------------------------------------
# Pixel ↔ angle conversion (vectorised)
# ---------------------------------------------------------------------------


def equirect_pixel_to_angles(
    u_eq: np.ndarray | float, v_eq: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular pixel coords → ``(lambda, phi)`` in radians.

    ``lambda`` (longitude) ∈ ``(-π, π]``;
    ``phi`` (latitude)    ∈ ``(-V_FOV/2, V_FOV/2)``.
    """
    u = np.asarray(u_eq, dtype=np.float64)
    v = np.asarray(v_eq, dtype=np.float64)
    lam = (u / EQUIRECT_W - 0.5) * H_FOV          # u=0 → -π, u=W → +π
    phi = (0.5 - v / EQUIRECT_H) * V_FOV          # v=0 → +V_FOV/2 (top), v=H → -V_FOV/2
    return lam, phi


def angles_to_equirect_pixel(
    lam: np.ndarray | float, phi: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(lambda, phi)`` → equirectangular pixel coords. Inverse of
    :func:`equirect_pixel_to_angles`."""
    lam_arr = np.asarray(lam, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    # Wrap lambda into (-π, π] so the mapping is single-valued.
    lam_wrapped = ((lam_arr + np.pi) % (2.0 * np.pi)) - np.pi
    u = (lam_wrapped / H_FOV + 0.5) * EQUIRECT_W
    v = (0.5 - phi_arr / V_FOV) * EQUIRECT_H
    return u, v


def angles_to_world_dir(
    lam: np.ndarray | float, phi: np.ndarray | float,
) -> np.ndarray:
    """``(lambda, phi)`` → unit direction in the world camera frame.

    Convention: +z forward (lambda=0), +x right (lambda=π/2),
    +y down (phi=-π/2). The output shape is the broadcast of the
    inputs plus a trailing axis of length 3.
    """
    lam_arr = np.asarray(lam, dtype=np.float64)
    phi_arr = np.asarray(phi, dtype=np.float64)
    cos_phi = np.cos(phi_arr)
    x = np.sin(lam_arr) * cos_phi
    y = -np.sin(phi_arr)
    z = np.cos(lam_arr) * cos_phi
    return np.stack([x, y, z], axis=-1)


def world_dir_to_angles(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit world direction → ``(lambda, phi)``. Vectorised over the
    leading axes of ``d``."""
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    lam = np.arctan2(dx, dz)
    phi = np.arctan2(-dy, np.sqrt(dx * dx + dz * dz))
    return lam, phi


# ---------------------------------------------------------------------------
# Face geometry
# ---------------------------------------------------------------------------


def _yaw_rotation(yaw: float) -> np.ndarray:
    """3×3 rotation around the world +y axis (down) by ``yaw`` rad."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float64)


def face_pixel_to_world_dir(
    u_f: np.ndarray, v_f: np.ndarray, face_idx: int,
) -> np.ndarray:
    """Face pixel coords → unit direction in the **world** camera frame.

    Used by :func:`build_forward_luts` to compute, for each face pixel,
    which equirect pixel it should sample from.
    """
    cx = FACE_SIZE / 2.0 - 0.5    # -0.5 so pixel centres straddle the optical axis
    cy = FACE_SIZE / 2.0 - 0.5
    x_im = np.asarray(u_f, dtype=np.float64) - cx
    y_im = np.asarray(v_f, dtype=np.float64) - cy
    # Face-local direction (z forward).
    local = np.stack([
        x_im,
        y_im,
        np.full_like(x_im, FACE_F),
    ], axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    # Rotate into world: world = R_yaw @ local^T.
    R = _yaw_rotation(float(FACE_YAWS[face_idx]))
    world = local @ R.T
    return world


def world_dir_to_face_pixel(
    d: np.ndarray, face_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World direction → ``(u_f, v_f, in_front)`` for the given face.

    ``in_front`` is a boolean mask: True where the direction projects in
    front of that face's optical axis (z_local > 0).
    """
    R = _yaw_rotation(float(FACE_YAWS[face_idx]))
    local = d @ R       # equivalent to R.T @ d.T then transpose
    zx = local[..., 2]
    in_front = zx > 1e-9
    # Avoid division-by-zero warnings; mask handles the unsafe entries.
    z_safe = np.where(in_front, zx, 1.0)
    x_im = local[..., 0] / z_safe * FACE_F
    y_im = local[..., 1] / z_safe * FACE_F
    cx = FACE_SIZE / 2.0 - 0.5
    cy = FACE_SIZE / 2.0 - 0.5
    return x_im + cx, y_im + cy, in_front


# ---------------------------------------------------------------------------
# LUT builders
# ---------------------------------------------------------------------------


def build_forward_luts() -> list[tuple[np.ndarray, np.ndarray]]:
    """Build ``(map_x, map_y)`` for each face, suitable for
    ``cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR)``.

    Returns a list of length :data:`N_FACES`; each entry is a pair of
    ``float32`` arrays of shape ``(FACE_SIZE, FACE_SIZE)``.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    grid_u, grid_v = np.meshgrid(
        np.arange(FACE_SIZE, dtype=np.float64),
        np.arange(FACE_SIZE, dtype=np.float64),
        indexing="xy",
    )
    for face_idx in range(N_FACES):
        d_world = face_pixel_to_world_dir(grid_u, grid_v, face_idx)
        lam, phi = world_dir_to_angles(d_world)
        u_eq, v_eq = angles_to_equirect_pixel(lam, phi)
        out.append((u_eq.astype(np.float32), v_eq.astype(np.float32)))
    return out


def build_inverse_lut() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every equirect pixel, store ``(face_idx, u_face, v_face)``
    of the face that pixel comes from.

    When multiple faces cover the same equirect pixel (the ~10° overlap
    bands), pick the face with the largest local-z, i.e. the most
    centred view. Pixels outside every face's FOV get ``face_idx = -1``
    and ``u/v = NaN`` (top/bottom strips that lie within the equirect
    crop but are clipped by the per-face 100° vertical FOV at the seams).

    Returns
    -------
    face_idx_lut : int8  ``(EQUIRECT_H, EQUIRECT_W)``
        Source face per equirect pixel; ``-1`` for unmapped.
    face_u_lut, face_v_lut : float32 ``(EQUIRECT_H, EQUIRECT_W)``
        Source face pixel coords; ``NaN`` for unmapped pixels.
    """
    u_grid, v_grid = np.meshgrid(
        np.arange(EQUIRECT_W, dtype=np.float64),
        np.arange(EQUIRECT_H, dtype=np.float64),
        indexing="xy",
    )
    lam, phi = equirect_pixel_to_angles(u_grid, v_grid)
    d_world = angles_to_world_dir(lam, phi)

    scores = np.full((N_FACES, EQUIRECT_H, EQUIRECT_W), -np.inf)
    u_face = np.zeros((N_FACES, EQUIRECT_H, EQUIRECT_W), dtype=np.float32)
    v_face = np.zeros((N_FACES, EQUIRECT_H, EQUIRECT_W), dtype=np.float32)

    for face_idx in range(N_FACES):
        R = _yaw_rotation(float(FACE_YAWS[face_idx]))
        local = d_world @ R
        z_local = local[..., 2]
        x_im = local[..., 0] / np.where(z_local > 1e-9, z_local, 1.0) * FACE_F
        y_im = local[..., 1] / np.where(z_local > 1e-9, z_local, 1.0) * FACE_F
        cx = FACE_SIZE / 2.0 - 0.5
        u_pix = x_im + cx
        v_pix = y_im + cx       # cx == cy
        in_face = (
            (z_local > 1e-9)
            & (u_pix >= 0) & (u_pix <= FACE_SIZE - 1)
            & (v_pix >= 0) & (v_pix <= FACE_SIZE - 1)
        )
        scores[face_idx] = np.where(in_face, z_local, -np.inf)
        u_face[face_idx] = u_pix.astype(np.float32)
        v_face[face_idx] = v_pix.astype(np.float32)

    best = np.argmax(scores, axis=0).astype(np.int8)
    unmapped = ~np.isfinite(np.max(scores, axis=0))
    rows = np.arange(EQUIRECT_H)[:, None]
    cols = np.arange(EQUIRECT_W)[None, :]
    u_out = u_face[best, rows, cols].astype(np.float32)
    v_out = v_face[best, rows, cols].astype(np.float32)
    u_out[unmapped] = np.nan
    v_out[unmapped] = np.nan
    face_out = best.copy()
    face_out[unmapped] = -1
    return face_out, u_out, v_out


# ---------------------------------------------------------------------------
# Bbox / mask reprojection helpers
# ---------------------------------------------------------------------------


def project_face_bbox_to_equirect_aabb(
    bbox_xyxy: tuple[float, float, float, float],
    face_idx: int,
    samples_per_edge: int = 16,
) -> tuple[float, float, float, float]:
    """Convert a per-face axis-aligned bbox into an equirectangular
    axis-aligned bbox.

    Faces are perspective views; their straight bbox edges curve when
    projected onto the equirectangular surface, so we sample points
    along the four edges and take the AABB of the projections. The
    returned AABB is **clamped to the equirect canvas** but is **not**
    wrap-aware — a bbox spanning the λ = ±π seam will produce a
    full-width AABB. Callers that need wrap-aware geometry should use
    the per-edge samples directly.

    Returned format: ``(u_min, v_min, u_max, v_max)`` in equirect pixels.
    """
    x1, y1, x2, y2 = bbox_xyxy
    n = max(int(samples_per_edge), 2)
    top_u    = np.linspace(x1, x2, n);  top_v    = np.full(n, y1)
    bot_u    = np.linspace(x1, x2, n);  bot_v    = np.full(n, y2)
    left_u   = np.full(n, x1);          left_v   = np.linspace(y1, y2, n)
    right_u  = np.full(n, x2);          right_v  = np.linspace(y1, y2, n)
    us = np.concatenate([top_u, bot_u, left_u, right_u])
    vs = np.concatenate([top_v, bot_v, left_v, right_v])

    d_world = face_pixel_to_world_dir(us, vs, face_idx)
    lam, phi = world_dir_to_angles(d_world)
    u_eq, v_eq = angles_to_equirect_pixel(lam, phi)
    return (
        float(np.clip(u_eq.min(), 0, EQUIRECT_W - 1)),
        float(np.clip(v_eq.min(), 0, EQUIRECT_H - 1)),
        float(np.clip(u_eq.max(), 0, EQUIRECT_W - 1)),
        float(np.clip(v_eq.max(), 0, EQUIRECT_H - 1)),
    )


def face_mask_to_equirect_mask(
    face_mask: np.ndarray,
    face_idx: int,
    inverse_lut: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Reproject a (FACE_SIZE, FACE_SIZE) boolean face mask into a
    (EQUIRECT_H, EQUIRECT_W) boolean equirectangular mask.

    Uses nearest-neighbour sampling from the precomputed inverse LUT.
    """
    if face_mask.shape != (FACE_SIZE, FACE_SIZE):
        raise ValueError(
            f"face_mask must be ({FACE_SIZE}, {FACE_SIZE}); got {face_mask.shape}",
        )
    face_idx_lut, u_lut, v_lut = inverse_lut
    out = np.zeros((EQUIRECT_H, EQUIRECT_W), dtype=bool)
    sel = face_idx_lut == face_idx
    u = np.rint(u_lut[sel]).astype(np.int32)
    v = np.rint(v_lut[sel]).astype(np.int32)
    np.clip(u, 0, FACE_SIZE - 1, out=u)
    np.clip(v, 0, FACE_SIZE - 1, out=v)
    out[sel] = face_mask[v, u]
    return out
