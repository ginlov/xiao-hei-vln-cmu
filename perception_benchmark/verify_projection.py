"""Verify the 2D-detection → 3D-lift angular convention.

The lift decides which LiDAR returns belong to a mask by projecting each
return into equirect pixels and testing membership. If the mask's angular
convention and the projection's disagree — a yaw offset, a sign flip, a
frame confusion — the lift silently attaches the wrong surface to the object
and *nothing downstream complains*: the result is self-consistent, just wrong.

So the projection cannot be checked against the lift. It has to be checked
against something independent, which is ground truth in the map frame.

Three checks:

  1. round-trip  — the geometry functions against themselves and against each
     other (sidecar `perception/geometry.py` vs responder
     `xiao_hei_vln/perception/geometry.py`, two implementations of one
     convention). Pure, no data.
  2. azimuth     — mask centroid bearing vs GT object bearing, as an offset
     scan. Reports the offset that best aligns them; 0 means the convention
     is right. Also regressed against robot yaw, which separates a constant
     offset from a frame error.
  3. elevation   — the same scan on latitude, run separately because a
     vertical bug (flipped v, wrong crop) would otherwise blur check 2.

    uv run --extra perception python perception_benchmark/verify_projection.py \
        --scene arabic_room
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perception import geometry as sg  # noqa: E402
from xiao_hei_vln.messages.common import Quaternion  # noqa: E402
from xiao_hei_vln.perception.deskew import PoseDeskew  # noqa: E402
from xiao_hei_vln.perception.geometry import (  # noqa: E402
    EQUIRECT_H,
    EQUIRECT_W,
    V_FOV,
    project_camera_points_to_equirect,
    sensor_to_camera_transform,
)
from xiao_hei_vln.perception.lifter import (  # noqa: E402
    _rotation_from_quaternion,
)

RNG = np.random.default_rng(0)


def _wrap(a):
    """Wrap radians into (-π, π]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class _Q:
    """Minimal quaternion stand-in — the lifter only reads .x/.y/.z/.w."""

    def __init__(self, xyzw):
        self.x, self.y, self.z, self.w = (float(c) for c in xyzw)


# ---------------------------------------------------------------------------
# 1. round trip
# ---------------------------------------------------------------------------


def check_round_trip() -> list[tuple[str, float, str]]:
    """Each row: (what, max error, unit)."""
    out = []

    # a) pixel ↔ angle, sidecar
    u = RNG.uniform(0, EQUIRECT_W, 20000)
    v = RNG.uniform(0, EQUIRECT_H, 20000)
    lam, phi = sg.equirect_pixel_to_angles(u, v)
    u2, v2 = sg.angles_to_equirect_pixel(lam, phi)
    out.append(("equirect pixel → angle → pixel",
                float(max(np.abs(u2 - u).max(), np.abs(v2 - v).max())), "px"))

    # b) angle ↔ direction, sidecar
    d = sg.angles_to_world_dir(lam, phi)
    lam2, phi2 = sg.world_dir_to_angles(d)
    out.append(("angle → world dir → angle",
                float(np.degrees(max(np.abs(_wrap(lam2 - lam)).max(),
                                     np.abs(phi2 - phi).max()))), "deg"))

    # c) face pixel ↔ world dir, per face
    worst = 0.0
    for f in range(sg.N_FACES):
        uf = RNG.uniform(0, sg.FACE_SIZE - 1, 5000)
        vf = RNG.uniform(0, sg.FACE_SIZE - 1, 5000)
        dw = sg.face_pixel_to_world_dir(uf, vf, f)
        uf2, vf2, ok = sg.world_dir_to_face_pixel(dw, f)
        assert ok.all(), f"face {f}: a face pixel projected behind its own axis"
        worst = max(worst, float(max(np.abs(uf2 - uf).max(),
                                     np.abs(vf2 - vf).max())))
    out.append(("face pixel → world dir → face pixel", worst, "px"))

    # d) THE cross-check: responder projection vs sidecar projection.
    # Two independent implementations of the same convention; camera-frame
    # points in, equirect pixels out. Any disagreement here is the bug the
    # whole exercise is looking for.
    pts = RNG.normal(size=(20000, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 1e-3]
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    ur, vr, valid = project_camera_points_to_equirect(pts)
    lam_s, phi_s = sg.world_dir_to_angles(pts)
    us, vs = sg.angles_to_equirect_pixel(lam_s, phi_s)
    # Compare u on the cylinder: 0 and W are the same column.
    du = np.abs(_wrap((ur[valid] - us[valid]) / EQUIRECT_W * 2 * np.pi))
    du = du / (2 * np.pi) * EQUIRECT_W
    out.append(("responder projection vs sidecar projection",
                float(max(du.max(), np.abs(vr[valid] - vs[valid]).max())), "px"))

    # e) inverse LUT: the face pixel each equirect pixel is assigned to must
    # point back at that equirect pixel. This is what mask reprojection rides
    # on, so an error here moves every mask.
    fi, uf, vf = sg.build_inverse_lut()
    ok = fi >= 0
    rows, cols = np.nonzero(ok)
    sub = RNG.choice(len(rows), 20000, replace=False)
    rows, cols = rows[sub], cols[sub]
    worst = 0.0
    for f in range(sg.N_FACES):
        sel = fi[rows, cols] == f
        if not sel.any():
            continue
        dw = sg.face_pixel_to_world_dir(uf[rows[sel], cols[sel]],
                                        vf[rows[sel], cols[sel]], f)
        lam_b, phi_b = sg.world_dir_to_angles(dw)
        ub, vb = sg.angles_to_equirect_pixel(lam_b, phi_b)
        d_u = np.abs(_wrap((ub - cols[sel]) / EQUIRECT_W * 2 * np.pi))
        d_u = d_u / (2 * np.pi) * EQUIRECT_W
        worst = max(worst, float(max(d_u.max(),
                                     np.abs(vb - rows[sel]).max())))
    out.append(("inverse LUT: equirect px → face px → equirect px",
                worst, "px"))
    return out


# ---------------------------------------------------------------------------
# 2 + 3. angular offset against ground truth
# ---------------------------------------------------------------------------


def _mask_bearings(mask: np.ndarray) -> tuple[float, float]:
    """Circular-mean longitude and mean latitude of a boolean equirect mask.

    Longitude has to be a circular mean — a mask straddling λ=±π would
    otherwise average to 0, i.e. dead ahead, which is the worst possible
    wrong answer.
    """
    col = mask.sum(axis=0).astype(np.float64)
    row = mask.sum(axis=1).astype(np.float64)
    n = col.sum()
    if n == 0:
        return float("nan"), float("nan")
    u = np.arange(EQUIRECT_W) + 0.5
    lam_u = (u / EQUIRECT_W - 0.5) * 2 * np.pi
    lam = float(np.arctan2((col * np.sin(lam_u)).sum(),
                           (col * np.cos(lam_u)).sum()))
    v = np.arange(EQUIRECT_H) + 0.5
    phi_v = (0.5 - v / EQUIRECT_H) * V_FOV
    phi = float((row * phi_v).sum() / n)
    return lam, phi


def _gt_camera_bearings(gt_xyz: np.ndarray, pose: dict):
    """GT centres (map frame) → (lambda, phi) in the camera frame.

    Uses the production transform chain, not a re-derivation: the point is to
    test what the lifter actually does.
    """
    R_ms = _rotation_from_quaternion(_Q(pose["orientation_xyzw"]))
    t_ms = np.asarray(pose["position"], dtype=np.float64)
    R_sc, t_sc = sensor_to_camera_transform()
    xyz_sensor = (gt_xyz - t_ms) @ R_ms
    xyz_cam = xyz_sensor @ R_sc.T + t_sc
    lam = np.arctan2(xyz_cam[:, 0], xyz_cam[:, 2])
    phi = np.arctan2(-xyz_cam[:, 1], np.hypot(xyz_cam[:, 0], xyz_cam[:, 2]))
    rng = np.linalg.norm(xyz_cam, axis=1)
    return lam, phi, rng


def _best_offset(deltas: np.ndarray, groups: list[np.ndarray],
                 grid: np.ndarray) -> tuple[float, np.ndarray]:
    """Offset that minimises Σ min-over-candidates |wrap(Δ − θ)|.

    A plain mean of nearest-neighbour residuals cannot find a large offset:
    the nearest GT is chosen *after* the offset is applied, so a wrong pairing
    always looks small. Scanning θ over the whole circle and re-pairing at
    each step is what makes a 90° or 180° error visible.
    """
    cost = np.empty(len(grid))
    for i, th in enumerate(grid):
        c = 0.0
        for g in groups:
            c += float(np.abs(_wrap(g - th)).min())
        cost[i] = c / max(len(groups), 1)
    return float(grid[int(np.argmin(cost))]), cost


def check_against_gt(scene: str, cap_dir: Path, debug_dir: Path,
                     max_range_m: float, max_frames: int | None,
                     skew_s: float = 0.0):
    viz = json.loads((debug_dir / scene / "viz.json").read_text())
    gt = viz["gt"]
    gt_xyz = np.array([g["center"] for g in gt], dtype=np.float64)
    gt_lab = np.array([g["label"] for g in gt])

    vp_dirs = sorted((cap_dir / scene).glob("vp_*"))
    if max_frames:
        vp_dirs = vp_dirs[:max_frames]

    az_groups: list[np.ndarray] = []     # per detection: Δλ to each same-label GT
    el_groups: list[np.ndarray] = []
    # Detections whose label has exactly ONE GT instance in range: the pairing
    # is forced, so the residual measures the projection rather than the
    # ambiguity of choosing among 20 identical wall lamps.
    az_unique: list[float] = []
    el_unique: list[float] = []
    rng_unique: list[float] = []
    per_frame: list[tuple[float, float]] = []    # (robot yaw, frame az offset)
    n_det = n_used = 0

    grid = np.deg2rad(np.arange(-180, 180, 1.0))

    deskew = PoseDeskew(skew_s)
    for vpd in vp_dirs:
        npz = vpd / "detections.npz"
        if not npz.is_file():
            continue
        pose = json.loads((vpd / "pose.json").read_text())
        meta_t = float(json.loads((vpd / "meta.json").read_text()).get("t", 0.0))

        # Run the pose through the SHIPPING correction, not a re-derivation of
        # it here: this is what validates PoseDeskew's sign end to end. With
        # skew 0 it is a no-op and the measurement is the raw one.
        qx, qy, qz, qw = pose["orientation_xyzw"]
        q = deskew.update(Quaternion(x=qx, y=qy, z=qz, w=qw), meta_t)
        pose = {"position": pose["position"],
                "orientation_xyzw": [q.x, q.y, q.z, q.w]}

        lam_gt, phi_gt, rng_gt = _gt_camera_bearings(gt_xyz, pose)
        near = rng_gt <= max_range_m

        D = np.load(npz, allow_pickle=True)
        masks, labels = D["masks"], D["labels"]
        n_det += len(labels)
        frame_groups = []
        for m, lab in zip(masks, labels, strict=True):
            sel = near & (gt_lab == lab)
            if not sel.any():
                continue                    # no same-label GT in range to test
            lam_m, phi_m = _mask_bearings(np.asarray(m, dtype=bool))
            if not np.isfinite(lam_m):
                continue
            g_az = _wrap(lam_m - lam_gt[sel])
            az_groups.append(g_az)
            frame_groups.append(g_az)
            el_groups.append(phi_m - phi_gt[sel])
            n_used += 1
            if sel.sum() == 1:
                az_unique.append(float(g_az[0]))
                el_unique.append(float(phi_m - phi_gt[sel][0]))
                rng_unique.append(float(rng_gt[sel][0]))
        if frame_groups:
            th, _ = _best_offset(None, frame_groups, grid)
            yaw = 2.0 * np.arctan2(pose["orientation_xyzw"][2],
                                   pose["orientation_xyzw"][3])
            per_frame.append((_wrap(yaw), th, meta_t))

    az_off, az_cost = _best_offset(None, az_groups, grid)
    el_grid = np.deg2rad(np.arange(-60, 60, 0.5))
    el_off, el_cost = _best_offset(None, el_groups, el_grid)

    # Residual at zero offset vs at the best offset — how much the scan
    # actually buys. If they are equal, there is nothing to correct.
    res0 = float(np.mean([np.abs(g).min() for g in az_groups]))
    resb = float(np.mean([np.abs(_wrap(g - az_off)).min() for g in az_groups]))
    el0 = float(np.mean([np.abs(g).min() for g in el_groups]))
    elb = float(np.mean([np.abs(g - el_off).min() for g in el_groups]))

    # Slope of the per-frame offset against robot yaw. A frame confusion
    # (yaw applied twice, or not at all) shows up here and nowhere else.
    slope = corr = float("nan")
    skew_s = skew_corr = float("nan")
    if len(per_frame) > 2:
        y = np.array([p[0] for p in per_frame])
        th = np.array([p[1] for p in per_frame])
        ts = np.array([p[2] for p in per_frame])
        if np.std(y) > 1e-6:
            slope = float(np.polyfit(y, np.unwrap(th), 1)[0])
            corr = float(np.corrcoef(y, np.unwrap(th))[0, 1])

        # A time skew between the image and the scan is invisible to every
        # test above: it is zero while the robot is still (two thirds of this
        # capture) and grows with turn rate. Regressing the per-frame offset
        # on yaw rate estimates that skew directly — the slope is in seconds.
        dt = np.diff(ts)
        rate = np.where(dt > 1e-3, _wrap(np.diff(y)) / np.where(dt > 1e-3, dt, 1), 0.0)
        ok = (dt > 1e-3) & (np.abs(rate) < 3.0)         # drop pose glitches
        if ok.sum() > 10 and np.std(rate[ok]) > 1e-3:
            skew_s = float(np.polyfit(rate[ok], th[1:][ok], 1)[0])
            skew_corr = float(np.corrcoef(rate[ok], th[1:][ok])[0, 1])

    def _robust(v, wrap=True):
        """Median residual and the share landing near zero, in degrees.

        A mean is useless on this sample: a "unique GT" pairing is *forced*,
        so every false-positive detection of that label contributes an
        essentially random residual and drags the mean tens of degrees. The
        median and the concentration near zero describe the real population —
        a peak at 0 plus a tail of things that were never that object.
        """
        if not v:
            return dict(median=float("nan"), p5=0.0, p10=0.0, p20=0.0, n=0)
        a = np.abs(_wrap(np.asarray(v)) if wrap else np.asarray(v))
        d = np.degrees(a)
        return dict(median=float(np.median(d)),
                    p5=float((d < 5).mean() * 100),
                    p10=float((d < 10).mean() * 100),
                    p20=float((d < 20).mean() * 100), n=len(d))

    az_u = _robust(az_unique)
    el_u = _robust(el_unique, wrap=False)

    # A camera-height error puts a systematic tilt on elevation that falls off
    # as 1/range: Δφ ≈ Δh / r. Fitting that slope turns a vague "a couple of
    # degrees off" into metres, where 0.1 m would mean the camera-above-LiDAR
    # offset is mis-signed and ~0 means the residual is just mask centroid vs
    # box centre. Restricted to the near-zero core so false pairs stay out.
    dh = float("nan")
    if rng_unique:
        rr = np.asarray(rng_unique)
        ee = np.asarray(el_unique)
        core = np.abs(np.degrees(ee)) < 15
        if core.sum() > 10:
            dh = float(np.polyfit(1.0 / rr[core], ee[core], 1)[0])

    return {
        "az_u": az_u, "el_u": el_u, "height_err_m": dh,
        "frames": len(per_frame), "detections": n_det, "tested": n_used,
        "az_offset_deg": np.degrees(az_off),
        "az_resid0_deg": np.degrees(res0), "az_residbest_deg": np.degrees(resb),
        "el_offset_deg": np.degrees(el_off),
        "el_resid0_deg": np.degrees(el0), "el_residbest_deg": np.degrees(elb),
        "yaw_slope": slope, "yaw_corr": corr,
        "skew_s": skew_s, "skew_corr": skew_corr,
        "az_cost": az_cost, "az_grid": grid,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="arabic_room")
    p.add_argument("--cap-dir", default="perception_benchmark/captures_nav")
    p.add_argument("--debug-dir", default="perception_benchmark/debug_k10")
    p.add_argument("--max-range", type=float, default=8.0,
                   help="ignore GT further than this from the camera")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--old-extrinsic", action="store_true",
                   help="restore the pre-fix t_sc = (0, 0, 0.1) to reproduce "
                        "the extrinsic bug — see the note in main(). The shipped "
                        "default is now the corrected (0, 0.1, 0).")
    p.add_argument("--skew", type=float, default=0.0,
                   help="compensate GT bearings for an image/scan time skew "
                        "of this many seconds, then re-measure. The residual "
                        "must DROP if the skew is real. Simulates the live "
                        "LatestCache timestamp match on frozen captures.")
    a = p.parse_args()

    if a.old_extrinsic:
        # The sim's static_transform_publisher gives the camera origin 0.1 m up
        # in the SENSOR frame. p_cam = R (p - c), so the camera-frame constant
        # is -R@c = (0, +0.1, 0) — 0.1 m along camera +y (down), because the
        # sensor sits below the camera. It shipped as (0, 0, 0.1) — the ROS
        # translation pasted in unrotated, 0.1 m FORWARD — and is now fixed;
        # this flag puts the bug back to reproduce the before/after.
        import xiao_hei_vln.perception.geometry as _rg
        _rg._SENSOR_TO_CAMERA_TRANSLATION = np.array([0.0, 0.0, 0.1])

    print("=" * 72)
    print("1. ROUND TRIP  (pure geometry, no data)")
    print("=" * 72)
    worst = 0.0
    for what, err, unit in check_round_trip():
        flag = "ok " if err < (1.0 if unit == "px" else 1e-6) else "FAIL"
        worst = max(worst, err if unit == "px" else err)
        print(f"  [{flag}] {what:<48} {err:.3e} {unit}")

    print()
    print("=" * 72)
    print(f"2+3. AGAINST GROUND TRUTH  ({a.scene})")
    print("=" * 72)
    r = check_against_gt(a.scene, Path(a.cap_dir), Path(a.debug_dir),
                         a.max_range, a.max_frames, a.skew)
    if a.skew:
        print(f"  (GT bearings compensated for a {a.skew:+.4f} s skew)")
    print(f"  frames {r['frames']}  ·  detections {r['detections']}  ·  "
          f"testable (same-label GT in range) {r['tested']}")
    print()
    print("  AZIMUTH")
    print(f"    best global offset      {r['az_offset_deg']:+8.2f}°")
    print(f"    mean |residual| @ 0°    {r['az_resid0_deg']:8.2f}°")
    print(f"    mean |residual| @ best  {r['az_residbest_deg']:8.2f}°"
          f"   (gain {r['az_resid0_deg'] - r['az_residbest_deg']:+.2f}°)")
    print(f"    offset vs robot yaw     slope {r['yaw_slope']:+.3f}  "
          f"corr {r['yaw_corr']:+.3f}")
    u = r["az_u"]
    print(f"    forced-pair subset      median |Δ| {u['median']:.2f}°  ·  "
          f"<5° {u['p5']:.0f}%  <10° {u['p10']:.0f}%  <20° {u['p20']:.0f}%  "
          f"(n={u['n']})")
    print(f"    offset vs YAW RATE      skew {r['skew_s']:+.4f} s  "
          f"corr {r['skew_corr']:+.3f}"
          "   (image/scan time offset; 0 = synchronised)")
    print()
    print("  ELEVATION")
    print(f"    best global offset      {r['el_offset_deg']:+8.2f}°")
    print(f"    mean |residual| @ 0°    {r['el_resid0_deg']:8.2f}°")
    print(f"    mean |residual| @ best  {r['el_residbest_deg']:8.2f}°"
          f"   (gain {r['el_resid0_deg'] - r['el_residbest_deg']:+.2f}°)")
    e = r["el_u"]
    print(f"    forced-pair subset      median |Δ| {e['median']:.2f}°  ·  "
          f"<5° {e['p5']:.0f}%  <10° {e['p10']:.0f}%  <20° {e['p20']:.0f}%  "
          f"(n={e['n']})")
    print(f"    residual ∝ 1/range fit  Δheight {r['height_err_m']:+.3f} m"
          "   (bug detector; ~0.1 flagged the extrinsic. Noisy once the")
    print("                            offset is gone — trust the 'gain' and "
          "forced-pair rows instead.)")
    print()
    c = r["az_cost"]
    g = np.degrees(r["az_grid"])
    print("  azimuth cost curve (mean nearest |Δ| in degrees):")
    for th in (-180, -135, -90, -45, -10, 0, 10, 45, 90, 135):
        i = int(np.argmin(np.abs(g - th)))
        print(f"    θ={th:+5d}°  {np.degrees(c[i]):7.2f}")


if __name__ == "__main__":
    main()
