"""Pure-numpy tests for the equirect ⇄ perspective face geometry.

No GPU, no torch, no model weights. These cover the math that pipeline.py
relies on; if a yaw sign or pixel-centre offset is wrong, the per-face
detections silently drift relative to the equirectangular mask. We catch
that here before it leaks into the sidecar.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.geometry import (
    EQUIRECT_H,
    EQUIRECT_W,
    FACE_F,
    FACE_FOV,
    FACE_SIZE,
    FACE_YAWS,
    H_FOV,
    N_FACES,
    V_FOV,
    angles_to_equirect_pixel,
    angles_to_world_dir,
    build_forward_luts,
    build_inverse_lut,
    equirect_pixel_to_angles,
    face_mask_to_equirect_mask,
    face_pixel_to_world_dir,
    project_face_bbox_to_equirect_aabb,
    world_dir_to_angles,
    world_dir_to_face_pixel,
)


# ---------------------------------------------------------------------------
# 1. Pixel ↔ angle round-trip
# ---------------------------------------------------------------------------


class TestEquirectAngles:
    def test_center_pixel_is_origin(self) -> None:
        lam, phi = equirect_pixel_to_angles(EQUIRECT_W / 2, EQUIRECT_H / 2)
        assert lam == pytest.approx(0.0, abs=1e-9)
        assert phi == pytest.approx(0.0, abs=1e-9)

    def test_horizontal_extents(self) -> None:
        lam_left, _ = equirect_pixel_to_angles(0, EQUIRECT_H / 2)
        lam_right, _ = equirect_pixel_to_angles(EQUIRECT_W, EQUIRECT_H / 2)
        assert lam_left == pytest.approx(-np.pi, abs=1e-9)
        assert lam_right == pytest.approx(+np.pi, abs=1e-9)

    def test_vertical_extents(self) -> None:
        _, phi_top = equirect_pixel_to_angles(EQUIRECT_W / 2, 0)
        _, phi_bot = equirect_pixel_to_angles(EQUIRECT_W / 2, EQUIRECT_H)
        assert phi_top == pytest.approx(+V_FOV / 2, abs=1e-9)
        assert phi_bot == pytest.approx(-V_FOV / 2, abs=1e-9)

    def test_roundtrip_on_grid(self) -> None:
        u = np.linspace(0, EQUIRECT_W, 17)
        v = np.linspace(0, EQUIRECT_H, 9)
        uu, vv = np.meshgrid(u, v, indexing="xy")
        lam, phi = equirect_pixel_to_angles(uu, vv)
        u2, v2 = angles_to_equirect_pixel(lam, phi)
        # The right-edge wraps: u=W should round-trip to u=0 (the same
        # column, since 0° and 360° alias). Compare modulo width.
        diff = np.minimum(np.abs(u2 - uu), np.abs(u2 - uu + EQUIRECT_W))
        diff = np.minimum(diff, np.abs(u2 - uu - EQUIRECT_W))
        assert np.max(diff) < 1e-6
        assert np.max(np.abs(v2 - vv)) < 1e-6


# ---------------------------------------------------------------------------
# 2. Direction conversions
# ---------------------------------------------------------------------------


class TestAngleDirRoundtrip:
    @pytest.mark.parametrize("lam, phi, expected_axis", [
        (0.0,         0.0, np.array([0.0, 0.0, +1.0])),    # forward
        (+np.pi / 2,  0.0, np.array([+1.0, 0.0, 0.0])),    # right
        (-np.pi / 2,  0.0, np.array([-1.0, 0.0, 0.0])),    # left
        (np.pi,       0.0, np.array([0.0, 0.0, -1.0])),    # backward
        (0.0,        +np.pi / 6, np.array([0.0, -0.5, np.sqrt(3) / 2])),  # up 30°
        (0.0,        -np.pi / 6, np.array([0.0, +0.5, np.sqrt(3) / 2])),  # down 30°
    ])
    def test_unit_directions(self, lam: float, phi: float, expected_axis: np.ndarray) -> None:
        d = angles_to_world_dir(lam, phi)
        np.testing.assert_allclose(d, expected_axis, atol=1e-9)

    def test_dir_to_angles_inverts_angles_to_dir(self) -> None:
        # Avoid the south/north pole singularities by staying inside the
        # equirect crop's vertical range (±60°).
        rng = np.random.default_rng(0)
        lam = rng.uniform(-np.pi, np.pi, size=200)
        phi = rng.uniform(-V_FOV / 2 * 0.95, V_FOV / 2 * 0.95, size=200)
        d = angles_to_world_dir(lam, phi)
        lam2, phi2 = world_dir_to_angles(d)
        np.testing.assert_allclose(lam2, lam, atol=1e-9)
        np.testing.assert_allclose(phi2, phi, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Face geometry — yaw conventions
# ---------------------------------------------------------------------------


class TestFaceGeometry:
    @pytest.mark.parametrize("face_idx, expected_world_z_axis", [
        (0, np.array([0.0, 0.0, +1.0])),    # front face looks at +z
        (1, np.array([+1.0, 0.0, 0.0])),    # right face looks at +x
        (2, np.array([0.0, 0.0, -1.0])),    # back face looks at -z
        (3, np.array([-1.0, 0.0, 0.0])),    # left face looks at -x
    ])
    def test_face_center_pixel_points_in_yaw_direction(
        self, face_idx: int, expected_world_z_axis: np.ndarray,
    ) -> None:
        # Centre pixel of the face (~ (320, 320)) should look along that
        # face's yaw axis.
        cx = FACE_SIZE / 2.0 - 0.5
        d = face_pixel_to_world_dir(np.array(cx), np.array(cx), face_idx)
        np.testing.assert_allclose(d, expected_world_z_axis, atol=1e-9)

    def test_face_extreme_pixel_within_fov(self) -> None:
        # The face's diagonal corner is the worst case for FOV coverage.
        # For 100° square FOV with 640×640 pixels, the corner should still
        # be a valid direction inside the hemisphere.
        d = face_pixel_to_world_dir(np.array(0.0), np.array(0.0), face_idx=0)
        assert d[..., 2] > 0  # still in front

    def test_world_to_face_inverts_face_to_world(self) -> None:
        # Pick face 0 (front), grab a grid of pixels, round-trip through
        # the world frame.
        rng = np.random.default_rng(1)
        u = rng.uniform(0, FACE_SIZE - 1, size=50)
        v = rng.uniform(0, FACE_SIZE - 1, size=50)
        d = face_pixel_to_world_dir(u, v, face_idx=0)
        u2, v2, in_front = world_dir_to_face_pixel(d, face_idx=0)
        assert np.all(in_front)
        np.testing.assert_allclose(u2, u, atol=1e-6)
        np.testing.assert_allclose(v2, v, atol=1e-6)

    def test_face_size_intrinsics_consistent(self) -> None:
        # The intrinsic FOV constant must match what FACE_F implements.
        recomputed_fov = 2.0 * np.arctan(FACE_SIZE / 2.0 / FACE_F)
        assert recomputed_fov == pytest.approx(FACE_FOV, abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Forward LUT — used by cv2.remap(equirect, …) → face image
# ---------------------------------------------------------------------------


class TestForwardLut:
    def test_shape_and_dtype(self) -> None:
        luts = build_forward_luts()
        assert len(luts) == N_FACES
        for map_x, map_y in luts:
            assert map_x.shape == (FACE_SIZE, FACE_SIZE)
            assert map_y.shape == (FACE_SIZE, FACE_SIZE)
            assert map_x.dtype == np.float32
            assert map_y.dtype == np.float32

    def test_face0_center_maps_to_equirect_center(self) -> None:
        luts = build_forward_luts()
        map_x, map_y = luts[0]  # front face
        cx = FACE_SIZE // 2
        # Front center pixel should sample from equirect (W/2, H/2)
        # (lambda=0, phi=0).
        assert map_x[cx, cx] == pytest.approx(EQUIRECT_W / 2.0, abs=1.0)
        assert map_y[cx, cx] == pytest.approx(EQUIRECT_H / 2.0, abs=1.0)

    def test_face1_center_maps_to_right_quarter_of_equirect(self) -> None:
        luts = build_forward_luts()
        map_x, map_y = luts[1]   # right face
        cx = FACE_SIZE // 2
        # Right face centre points at lambda = +π/2 → equirect column = 3/4 * W.
        assert map_x[cx, cx] == pytest.approx(EQUIRECT_W * 0.75, abs=1.0)
        assert map_y[cx, cx] == pytest.approx(EQUIRECT_H / 2.0, abs=1.0)

    def test_face3_center_maps_to_left_quarter_of_equirect(self) -> None:
        luts = build_forward_luts()
        map_x, map_y = luts[3]   # left face
        cx = FACE_SIZE // 2
        # Left face centre points at lambda = -π/2 → equirect column = W/4.
        # (Or W*1.25 due to wrap; both are valid columns of the same point.)
        assert map_x[cx, cx] == pytest.approx(EQUIRECT_W * 0.25, abs=1.0)
        assert map_y[cx, cx] == pytest.approx(EQUIRECT_H / 2.0, abs=1.0)


# ---------------------------------------------------------------------------
# 5. Inverse LUT — used by mask reprojection
# ---------------------------------------------------------------------------


class TestInverseLut:
    @pytest.fixture(scope="class")
    def lut(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return build_inverse_lut()

    def test_shape_and_dtype(self, lut: tuple) -> None:
        face_idx_lut, u_lut, v_lut = lut
        assert face_idx_lut.shape == (EQUIRECT_H, EQUIRECT_W)
        assert u_lut.shape == (EQUIRECT_H, EQUIRECT_W)
        assert v_lut.shape == (EQUIRECT_H, EQUIRECT_W)
        assert face_idx_lut.dtype == np.int8
        assert u_lut.dtype == np.float32
        assert v_lut.dtype == np.float32

    def test_equirect_center_is_front_face(self, lut: tuple) -> None:
        face_idx_lut, u_lut, v_lut = lut
        cy, cx = EQUIRECT_H // 2, EQUIRECT_W // 2
        assert face_idx_lut[cy, cx] == 0
        # And it should sample from the front face's centre pixel.
        assert u_lut[cy, cx] == pytest.approx(FACE_SIZE / 2.0 - 0.5, abs=1.0)
        assert v_lut[cy, cx] == pytest.approx(FACE_SIZE / 2.0 - 0.5, abs=1.0)

    def test_right_quarter_is_right_face(self, lut: tuple) -> None:
        face_idx_lut, _, _ = lut
        cy = EQUIRECT_H // 2
        # Column at λ = +π/2 → equirect column = 3/4 * W.
        cx = int(EQUIRECT_W * 0.75)
        assert face_idx_lut[cy, cx] == 1

    def test_left_quarter_is_left_face(self, lut: tuple) -> None:
        face_idx_lut, _, _ = lut
        cy = EQUIRECT_H // 2
        cx = int(EQUIRECT_W * 0.25)
        assert face_idx_lut[cy, cx] == 3

    def test_seam_band_covered_by_either_face(self, lut: tuple) -> None:
        # In the ~10° overlap band between front (face 0) and right
        # (face 1), the LUT should pick one of them — never -1.
        face_idx_lut, _, _ = lut
        cy = EQUIRECT_H // 2
        # 45° to the right of forward → exactly between front/right.
        cx = int((np.pi / 4 / H_FOV + 0.5) * EQUIRECT_W)
        assert face_idx_lut[cy, cx] in (0, 1)

    def test_all_equator_pixels_mapped(self, lut: tuple) -> None:
        face_idx_lut, _, _ = lut
        # Every pixel on the equator (φ=0) lies inside some face's FOV.
        row = face_idx_lut[EQUIRECT_H // 2, :]
        assert np.all(row >= 0)


# ---------------------------------------------------------------------------
# 6. Mask reprojection — face mask → equirect mask
# ---------------------------------------------------------------------------


class TestFaceMaskReprojection:
    def test_full_face_mask_round_trips_to_equator_band(self) -> None:
        # An all-True front-face mask should populate the equirect mask
        # over the equirect region that face 0 "owns" — and only that
        # region.
        lut = build_inverse_lut()
        face_mask = np.ones((FACE_SIZE, FACE_SIZE), dtype=bool)
        eq_mask = face_mask_to_equirect_mask(face_mask, 0, lut)
        face_idx_lut, _, _ = lut
        np.testing.assert_array_equal(eq_mask, face_idx_lut == 0)

    def test_single_face_pixel_lands_at_one_equirect_pixel(self) -> None:
        # Plant a single True at the front face's centre. After
        # reprojection we expect a True near equirect (W/2, H/2).
        lut = build_inverse_lut()
        face_mask = np.zeros((FACE_SIZE, FACE_SIZE), dtype=bool)
        face_mask[FACE_SIZE // 2, FACE_SIZE // 2] = True
        eq_mask = face_mask_to_equirect_mask(face_mask, 0, lut)
        # Inverse LUT round-trip means many equirect pixels map to the
        # same face pixel; with NN sampling, the projected True forms a
        # small blob around (W/2, H/2). Check the centre is True and
        # the total active count is reasonable (not zero, not full).
        assert eq_mask[EQUIRECT_H // 2, EQUIRECT_W // 2]
        assert 1 < eq_mask.sum() < EQUIRECT_W * EQUIRECT_H // 2

    def test_wrong_shape_raises(self) -> None:
        lut = build_inverse_lut()
        with pytest.raises(ValueError):
            face_mask_to_equirect_mask(np.zeros((100, 100), dtype=bool), 0, lut)


# ---------------------------------------------------------------------------
# 7. Bbox AABB projection
# ---------------------------------------------------------------------------


class TestBboxProjection:
    def test_front_center_bbox_lands_at_equirect_center(self) -> None:
        # 40-px bbox centred on the front face's centre pixel.
        cx = FACE_SIZE / 2.0
        bbox = (cx - 20, cx - 20, cx + 20, cx + 20)
        u_min, v_min, u_max, v_max = project_face_bbox_to_equirect_aabb(bbox, 0)
        # The projected AABB should span both sides of equirect center.
        assert u_min < EQUIRECT_W / 2 < u_max
        assert v_min < EQUIRECT_H / 2 < v_max
        # And be small (subtended by ~6° on a 100° face → ~32 px on equirect).
        assert (u_max - u_min) < 100
        assert (v_max - v_min) < 100

    def test_face_corner_bbox_projects_to_face_extreme(self) -> None:
        # A bbox at the right-edge column of face 1 (right face) should
        # land near the right-back extreme of equirect.
        bbox = (FACE_SIZE - 30, FACE_SIZE / 2 - 15, FACE_SIZE - 1, FACE_SIZE / 2 + 15)
        u_min, _, u_max, _ = project_face_bbox_to_equirect_aabb(bbox, 1)
        # Should land somewhere right of equirect centre (3W/4 region or beyond).
        assert u_min > EQUIRECT_W * 0.7
