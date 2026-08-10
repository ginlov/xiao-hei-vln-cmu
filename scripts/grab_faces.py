#!/usr/bin/env python3
"""Unwrap one 360° frame into the four perspective faces, as images to look at.

These are exactly the images the VLM grounding test will send: the sidecar
already unwraps `/camera/image` this way before YOLO sees it, so looking at the
faces is looking at the detector's actual input rather than at the panorama a
human would rather read.

Uses `perception/geometry.py` (pure numpy) so the faces are pixel-identical to
the sidecar's, not a lookalike built from different constants.

    uv run python scripts/grab_faces.py shot.jpg -o faces/
    uv run python scripts/grab_faces.py shot.jpg -o faces/ --face-size 1024

On resolution: the equirect is 1920 px over 360°, so 5.33 px/deg. A 100° face
at the default 640 px is 6.4 px/deg — already slightly *above* the source
density. Asking for more upsamples; it looks smoother without carrying more
information. Worth knowing before concluding the VLM failed for lack of pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perception"))
import geometry as G  # noqa: E402

FACE_NAMES = ["0_front", "1_right", "2_back", "3_left"]


def build_luts(face_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward LUTs at an arbitrary face size.

    `geometry.build_forward_luts` is hard-wired to `FACE_SIZE`, which is the
    right call for the sidecar — the inverse LUT and the mask reprojection must
    agree with it. Here the faces are only being looked at, so the module
    constants are patched for the render and put straight back.
    """
    old_size, old_f = G.FACE_SIZE, G.FACE_F
    G.FACE_SIZE = face_size
    G.FACE_F = (face_size / 2.0) / np.tan(G.FACE_FOV / 2.0)
    try:
        return G.build_forward_luts()
    finally:
        G.FACE_SIZE, G.FACE_F = old_size, old_f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("equirect", help="1920x640 panorama (jpg/png)")
    ap.add_argument("-o", "--out", default="faces", help="output directory")
    ap.add_argument("--face-size", type=int, default=G.FACE_SIZE,
                    help=f"pixels per face (default {G.FACE_SIZE}; "
                         f"beyond ~700 this only upsamples)")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also write all four side by side, labelled")
    args = ap.parse_args()

    eq = cv2.imread(args.equirect, cv2.IMREAD_COLOR)
    if eq is None:
        print(f"cannot read {args.equirect}", file=sys.stderr)
        return 2
    h, w = eq.shape[:2]
    if (w, h) != (G.EQUIRECT_W, G.EQUIRECT_H):
        # The LUTs are built for the challenge sensor's exact geometry, so a
        # differently sized panorama would be silently mis-sampled.
        print(f"warning: expected {G.EQUIRECT_W}x{G.EQUIRECT_H}, got {w}x{h} — "
              f"resizing, angles may be off", file=sys.stderr)
        eq = cv2.resize(eq, (G.EQUIRECT_W, G.EQUIRECT_H), interpolation=cv2.INTER_AREA)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    faces = []
    for (map_x, map_y), name in zip(build_luts(args.face_size), FACE_NAMES):
        face = cv2.remap(eq, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)
        p = out / f"face_{name}.jpg"
        cv2.imwrite(str(p), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        faces.append(face)
        print(f"  {p}  {face.shape[1]}x{face.shape[0]}")

    cv2.imwrite(str(out / "equirect.jpg"), eq, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {out / 'equirect.jpg'}  {G.EQUIRECT_W}x{G.EQUIRECT_H}")

    if args.contact_sheet:
        pad = 8
        s = faces[0].shape[0]
        sheet = np.full((s + 34, 4 * s + 5 * pad, 3), 24, np.uint8)
        for i, (face, name) in enumerate(zip(faces, FACE_NAMES)):
            x = pad + i * (s + pad)
            sheet[30:30 + s, x:x + s] = face
            cv2.putText(sheet, name.split("_")[1], (x, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        p = out / "contact_sheet.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {p}  {sheet.shape[1]}x{sheet.shape[0]}")

    px_per_deg = args.face_size / np.rad2deg(G.FACE_FOV)
    print(f"\nface FOV {np.rad2deg(G.FACE_FOV):.0f}°, {px_per_deg:.1f} px/deg "
          f"(equirect source: {G.EQUIRECT_W / 360.0:.2f} px/deg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
