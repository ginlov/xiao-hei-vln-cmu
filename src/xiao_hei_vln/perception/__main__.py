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
    p.add_argument("--min-move-m", type=float, default=0.0,
                   help="skip near-duplicate viewpoints before detecting them; "
                       "must be no larger than the value stage B will use")
    p.add_argument("--min-rot-deg", type=float, default=0.0,
                   help="...or this much rotation (pairs with --min-move-m)")


def _add_lift(sub) -> None:
    p = sub.add_parser("lift", help="stage B — lift + fuse cached detections")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="scene graph JSON to write")
    p.add_argument("--min-inliers", type=int, default=10)
    p.add_argument("--max-depth-m", type=float, default=None)
    p.add_argument("--scan-accumulator", action="store_true",
                   help="stack lidar keyframes before lifting; off by default "
                        "because production runs without it")
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
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop cached detections below this confidence; sweeping "
                        "the detector threshold upwards needs no stage-A re-run")


def _add_benchmark(top) -> None:
    p = top.add_parser("benchmark", help="score every recorded scene at once")
    p.add_argument("scenes", nargs="*", help="default: every corpus under --frames-root")
    p.add_argument("--frames-root", type=Path, default=Path("frames"))
    p.add_argument("--gt-root", type=Path,
                   default=Path.home() / "workspace/dataset/unity-scene")
    p.add_argument("--run", default="run1", help="corpus suffix, frames/<scene>_<run>")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="also write per-scene scene graphs and a summary json")
    p.add_argument("--include-structure", action="store_true",
                   help="score walls/floors/doors too (excluded by default, as "
                        "they are excluded from the object list given to the LLM)")
    p.add_argument("--min-inliers", type=int, default=10)
    p.add_argument("--scan-accumulator", action="store_true",
                   help="stack lidar keyframes before lifting; off by default "
                        "because production runs without it")
    p.add_argument("--min-obs", type=int, default=1,
                   help="drop nodes seen fewer than this many times")
    p.add_argument("--min-move-m", type=float, default=0.15)
    p.add_argument("--min-rot-deg", type=float, default=10.0)
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop cached detections below this confidence")


def _run_benchmark(args) -> int:
    from dataclasses import asdict

    from xiao_hei_vln.perception import benchmark

    scenes = args.scenes or sorted(
        d.name[: -len(f"_{args.run}")]
        for d in args.frames_root.glob(f"*_{args.run}")
        if (d / "frames.jsonl").is_file()
    )
    if not scenes:
        print(f"no corpora under {args.frames_root} matching *_{args.run}")
        return 1

    results = benchmark.run(
        scenes,
        frames_root=args.frames_root,
        gt_root=args.gt_root,
        run_name=args.run,
        out_dir=args.out_dir,
        include_structure=args.include_structure,
        min_inliers=args.min_inliers,
        use_scan_accumulator=args.scan_accumulator,
        min_move_m=args.min_move_m,
        min_rot_deg=args.min_rot_deg,
        min_score=args.min_score,
    )
    print()
    print(benchmark.format_table(results))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary = args.out_dir / "benchmark.json"
        summary.write_text(json.dumps([asdict(r) for r in results], indent=1))
        print(f"\nwrote {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="python -m xiao_hei_vln.perception",
                                 description=__doc__)
    top = ap.add_subparsers(dest="group", required=True)
    replay_p = top.add_parser("replay", help="offline replay of a recorded run")
    sub = replay_p.add_subparsers(dest="stage", required=True)
    _add_detect(sub)
    _add_lift(sub)
    _add_benchmark(top)
    args = ap.parse_args(argv)

    if args.group == "benchmark":
        return _run_benchmark(args)

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
            min_move_m=args.min_move_m,
            min_rot_deg=args.min_rot_deg,
        )
        print(f"wrote {out}")
        return 0

    scene, stats = replay.replay_lift(
        args.frames,
        min_inliers=args.min_inliers,
        max_depth_m=args.max_depth_m,
        use_scan_accumulator=args.scan_accumulator,
        use_image_pose=not args.no_image_pose,
        min_move_m=args.min_move_m,
        min_rot_deg=args.min_rot_deg,
        max_speed=args.max_speed,
        max_yaw_rate=args.max_yaw_rate,
        min_score=args.min_score,
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
