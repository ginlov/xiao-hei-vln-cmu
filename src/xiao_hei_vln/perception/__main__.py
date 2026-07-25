"""CLI for the offline perception replay — see :mod:`xiao_hei_vln.perception.replay`.

    # stage A: cache the detector's masks (needs the sidecar on --base-url)
    python -m xiao_hei_vln.perception replay detect --frames frames/run1

    # stage B: lift + fuse the cached masks into a scorable scene graph
    python -m xiao_hei_vln.perception replay lift --frames frames/run1 \\
        --out scene.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import sys
from pathlib import Path


def _add_detect(sub) -> None:
    p = sub.add_parser("detect", help="stage A — cache sidecar detections")
    p.add_argument("--frames", type=Path, required=True, help="recorded corpus dir")
    p.add_argument("--base-url", default="http://localhost:8001")
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="only the first N frames")
    p.add_argument(
        "--classes",
        default=None,
        help="comma-separated class list; default is the responder's prior",
    )


def _add_lift(sub) -> None:
    p = sub.add_parser("lift", help="stage B — lift + fuse cached detections")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="scene graph JSON to write")
    p.add_argument("--min-inliers", type=int, default=10)
    p.add_argument("--max-depth-m", type=float, default=None)
    p.add_argument("--no-scan-accumulator", action="store_true")
    p.add_argument(
        "--no-image-pose",
        action="store_true",
        help="lift with the newest pose instead of the one time-synced to the "
             "camera stamp (reproduces the pre-sync behaviour)",
    )
    p.add_argument("--max-speed", type=float, default=None,
                   help="skip frames captured above this speed (m/s)")
    p.add_argument("--max-yaw-rate", type=float, default=None,
                   help="skip frames captured above this yaw rate (rad/s)")
    p.add_argument("--min-move-m", type=float, default=0.0,
                   help="thin to viewpoint keyframes: require this much "
                        "translation since the last kept frame")
    p.add_argument("--min-rot-deg", type=float, default=0.0,
                   help="...or this much rotation (pairs with --min-move-m)")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="python -m xiao_hei_vln.perception",
                                 description=__doc__)
    top = ap.add_subparsers(dest="group", required=True)
    replay_p = top.add_parser("replay", help="offline replay of a recorded run")
    sub = replay_p.add_subparsers(dest="stage", required=True)
    _add_detect(sub)
    _add_lift(sub)
    args = ap.parse_args(argv)

    from xiao_hei_vln.perception import replay

    if args.stage == "detect":
        from xiao_hei_vln.perception.vocab import Vocabulary

        classes = (
            tuple(c.strip() for c in args.classes.split(",") if c.strip())
            if args.classes else Vocabulary().current_classes(None)
        )
        print(f"stage A: {len(classes)} classes → {args.base_url}")
        out = replay.run_detect(
            args.frames,
            base_url=args.base_url,
            classes=classes,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            limit=args.limit,
        )
        print(f"wrote {out}")
        return 0

    scene, stats = replay.replay_lift(
        args.frames,
        min_inliers=args.min_inliers,
        max_depth_m=args.max_depth_m,
        use_scan_accumulator=not args.no_scan_accumulator,
        use_image_pose=not args.no_image_pose,
        min_move_m=args.min_move_m,
        min_rot_deg=args.min_rot_deg,
        max_speed=args.max_speed,
        max_yaw_rate=args.max_yaw_rate,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scene))

    print(f"detections replayed : {stats.n_detections}")
    print(f"  lifted            : {stats.n_lifted}")
    print(f"  dropped (inliers) : {stats.n_below_min_inliers}")
    print(f"fused objects       : {len(scene['objects'])}")
    if stats.spreads:
        # The single-frame answer to "is the lift or the fusion at fault":
        # a compact object lifted cleanly sits at a few cm here.
        sp = sorted(stats.spreads)
        print("single-frame inlier spread p95 (m): "
              f"median {st.median(sp):.2f}  p90 {sp[int(len(sp) * 0.9)]:.2f}  "
              f"max {sp[-1]:.2f}")
        print(f"  observations spanning >1 m: "
              f"{sum(1 for s in sp if s > 1.0)}/{len(sp)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
