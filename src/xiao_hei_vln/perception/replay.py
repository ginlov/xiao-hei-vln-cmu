"""Offline replay of a recorded run through the perception pipeline.

Tuning detect → lift → fuse against the live simulator costs ~15 minutes per
experiment and is not reproducible: the explorer takes a different path every
run, so a metric delta may be the route rather than the change under test.
This module replays a corpus captured by
:class:`~xiao_hei_vln.perception.recorder.FrameRecorder` so every variant sees
byte-identical input.

Two stages, split at the GPU boundary:

**Stage A — detect** (needs the sidecar; minutes)
    Each recorded frame is POSTed to ``/detect`` and the returned masks are
    cached as COCO RLE in ``detections.jsonl``. Only a change to the detector
    itself — the class list, the score threshold, the sidecar code — invalidates
    this cache.

**Stage B — lift + fuse** (pure numpy; seconds)
    The cached masks are re-lifted against the recorded scans and poses, fused
    through :class:`~xiao_hei_vln.perception.object_map.ObjectMap`, and written
    out as a scene graph that ``xiao_hei_vln.perception.eval`` can score
    directly. This is the loop that geometry work iterates in.

Usage::

    # stage A (once per detector config)
    python -m xiao_hei_vln.perception.replay detect --frames frames/run1

    # stage B (once per geometry config)
    python -m xiao_hei_vln.perception.replay lift --frames frames/run1 \\
        --out scene.json

    python -m xiao_hei_vln.perception.eval --scene scene.json \\
        --gt-zip scenes/livingroom_3.zip --scene-name livingroom_3
"""

from __future__ import annotations

import base64
import json
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xiao_hei_vln.messages.common import Quaternion, Vector3
from xiao_hei_vln.perception.geometry import EQUIRECT_H, EQUIRECT_W

log = logging.getLogger(__name__)

DETECTIONS_FILE = "detections.jsonl"


# ---------------------------------------------------------------------------
# Corpus reading
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """One recorded tick, as the pipeline sees it."""

    tick_id: int
    image_path: Path
    scan_path: Path
    position: Vector3
    orientation: Quaternion
    speed: float
    yaw_rate: float
    raw: dict[str, Any]

    def scan(self) -> np.ndarray:
        return np.load(self.scan_path)

    def image_bgr(self) -> np.ndarray:
        from PIL import Image

        rgb = np.asarray(Image.open(self.image_path).convert("RGB"))
        return rgb[:, :, ::-1].copy()          # RGB → BGR, as the sidecar wants


def _vec(seq) -> Vector3:
    return Vector3(x=float(seq[0]), y=float(seq[1]), z=float(seq[2]))


def _quat(seq) -> Quaternion:
    return Quaternion(x=float(seq[0]), y=float(seq[1]), z=float(seq[2]), w=float(seq[3]))


def load_frames(
    frames_dir: str | Path,
    *,
    use_image_pose: bool = True,
    min_move_m: float = 0.0,
    min_rot_deg: float = 0.0,
) -> list[Frame]:
    """Read ``frames.jsonl`` into :class:`Frame` objects.

    ``use_image_pose`` picks the time-synced pose (the one interpolated at the
    camera stamp) when the recording has it — that is what the online lift
    uses, so the replay must too, or the two disagree by however far the robot
    moved during the camera lag. Set ``False`` to reproduce the pre-sync
    behaviour for comparison.

    ``min_move_m`` / ``min_rot_deg`` thin the corpus to viewpoint keyframes.
    A run that sits still at the end of exploration records hundreds of
    near-identical frames, and letting them through would weight the metrics
    towards whichever corner the robot happened to stop in.
    """
    frames_dir = Path(frames_dir)
    out: list[Frame] = []
    last_kept: Frame | None = None
    for line in (frames_dir / "frames.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pos, quat = r["position"], r["orientation"]
        if use_image_pose and r.get("image_pose_position") and r.get("image_pose_orientation"):
            pos, quat = r["image_pose_position"], r["image_pose_orientation"]
        lin = r.get("linear_velocity") or [0.0, 0.0, 0.0]
        ang = r.get("angular_velocity") or [0.0, 0.0, 0.0]
        frame = Frame(
            tick_id=int(r["tick_id"]),
            image_path=frames_dir / r["image"],
            scan_path=frames_dir / r["scan"],
            position=_vec(pos),
            orientation=_quat(quat),
            speed=float(np.hypot(lin[0], lin[1])),
            yaw_rate=abs(float(ang[2])),
            raw=r,
        )
        if last_kept is not None and (min_move_m > 0.0 or min_rot_deg > 0.0):
            moved = math.dist(
                (frame.position.x, frame.position.y, frame.position.z),
                (last_kept.position.x, last_kept.position.y, last_kept.position.z),
            )
            turned = abs(_wrap(_yaw(frame.orientation) - _yaw(last_kept.orientation)))
            if moved < min_move_m and turned < math.radians(min_rot_deg):
                continue
        out.append(frame)
        last_kept = frame
    return out


def _yaw(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Stage A — cache the detector's output
# ---------------------------------------------------------------------------


def _encode_mask(mask: np.ndarray) -> str:
    from pycocotools import mask as coco_mask

    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return base64.b64encode(rle["counts"]).decode("ascii")


def decode_mask(rle_str: str) -> np.ndarray:
    from pycocotools import mask as coco_mask

    rle = {"counts": base64.b64decode(rle_str.encode("ascii")),
           "size": [EQUIRECT_H, EQUIRECT_W]}
    return coco_mask.decode(rle).astype(bool)


def run_detect(
    frames_dir: str | Path,
    *,
    base_url: str,
    classes: tuple[str, ...],
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    limit: int | None = None,
    min_move_m: float = 0.0,
    min_rot_deg: float = 0.0,
) -> Path:
    """Stage A: POST every frame to the sidecar, cache detections to disk.

    Returns the path of the written ``detections.jsonl``. Each line is
    ``{"tick_id": int, "detections": [{"label", "score", "mask_rle"}, ...]}``.

    ``min_move_m`` / ``min_rot_deg`` thin to viewpoint keyframes before paying
    for detection. A tick where the robot has not moved yields the same masks
    as the one before it at a second of GPU each, and stage B discards them
    anyway — but a cache built this way is only valid for a stage B thinned at
    least as aggressively, so the thresholds are recorded in the metadata.
    """
    from xiao_hei_vln.perception.client import HTTPPerceptionClient

    frames_dir = Path(frames_dir)
    frames = load_frames(frames_dir, min_move_m=min_move_m, min_rot_deg=min_rot_deg)
    if limit is not None:
        frames = frames[:limit]

    out_path = frames_dir / DETECTIONS_FILE
    client = HTTPPerceptionClient(base_url=base_url)
    try:
        client.set_classes(classes)
        with out_path.open("w", buffering=1) as fh:
            for i, frame in enumerate(frames):
                dets = client.detect(
                    frame.image_bgr(),
                    score_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                )
                fh.write(json.dumps({
                    "tick_id": frame.tick_id,
                    "detections": [
                        {"label": d.label, "score": round(float(d.score), 4),
                         "mask_rle": _encode_mask(d.mask)}
                        for d in dets
                    ],
                }, separators=(",", ":")) + "\n")
                if (i + 1) % 25 == 0:
                    log.info("detect: %d/%d frames", i + 1, len(frames))
    finally:
        client.close()

    # The class list is part of the cache's identity — a later lift run has no
    # other way to tell which vocabulary produced these masks.
    (frames_dir / "detections_meta.json").write_text(json.dumps({
        "classes": list(classes),
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "n_frames": len(frames),
        "min_move_m": min_move_m,
        "min_rot_deg": min_rot_deg,
    }, indent=2) + "\n")
    return out_path


def load_detections(frames_dir: str | Path) -> dict[int, list[dict]]:
    """``tick_id → [{label, score, mask_rle}, ...]`` from the stage-A cache."""
    path = Path(frames_dir) / DETECTIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `replay detect` first (stage A needs the sidecar)",
        )
    out: dict[int, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[int(rec["tick_id"])] = rec["detections"]
    return out


# ---------------------------------------------------------------------------
# Stage B — lift + fuse
# ---------------------------------------------------------------------------


@dataclass
class LiftStats:
    """Per-observation diagnostics — enough to tell a bad lift from bad fusion.

    ``spreads`` is the 95th-percentile radius of each *single frame's* inlier
    cloud around its own median. A compact object lifted correctly measures a
    few centimetres here; metres means the mask pulled in background and the
    problem is upstream of any cross-frame merging.
    """

    n_detections: int = 0
    n_lifted: int = 0
    n_below_min_inliers: int = 0
    spreads: list[float] = None            # type: ignore[assignment]
    labels: list[str] = None               # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.spreads is None:
            self.spreads = []
        if self.labels is None:
            self.labels = []


def replay_lift(
    frames_dir: str | Path,
    *,
    min_inliers: int = 10,
    max_depth_m: float | None = None,
    use_object_map: bool = True,
    # Off, which no longer matches production: app/main.py now defaults
    # XIAO_HEI_SCAN_KEYFRAMES to 10 (main's value — see the note there).
    # Left off here deliberately, so replay keeps measuring what this branch
    # measured; flip it to compare — and flatters nothing: stacking
    # keyframes stretches an object's cloud, inflates its box, and splits it
    # into more nodes.
    use_scan_accumulator: bool = False,
    use_image_pose: bool = True,
    min_move_m: float = 0.0,
    min_rot_deg: float = 0.0,
    max_speed: float | None = None,
    max_yaw_rate: float | None = None,
    min_score: float = 0.0,
    frames: list[Frame] | None = None,
) -> tuple[dict, LiftStats]:
    """Stage B: cached masks + recorded scans/poses → fused scene graph.

    Returns ``(scene_dict, stats)`` where ``scene_dict`` is in the same shape
    ``SceneRepresentation.to_dict()`` produces, so
    :mod:`xiao_hei_vln.perception.eval` scores it unchanged.

    ``max_speed`` / ``max_yaw_rate`` drop frames captured while the robot was
    moving faster than the given gate — the motion-gating experiment, run
    offline instead of by re-driving the robot.

    ``min_score`` discards weak detections from the cache. Raising the
    detector's threshold is therefore a 40-second experiment rather than a
    stage-A re-run — but only upwards: the cache cannot produce a detection
    the sidecar already filtered out at capture time.
    """
    from xiao_hei_vln.perception.lifter import PointLifter
    from xiao_hei_vln.perception.object_map import ObjectMap
    from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator

    frames = frames if frames is not None else load_frames(
        frames_dir,
        use_image_pose=use_image_pose,
        min_move_m=min_move_m,
        min_rot_deg=min_rot_deg,
    )
    detections = load_detections(frames_dir)

    lifter = PointLifter(min_inliers=min_inliers, max_depth_m=max_depth_m)
    accum = ScanAccumulator() if use_scan_accumulator else None
    omap = ObjectMap() if use_object_map else None
    stats = LiftStats()

    for frame in frames:
        dets = detections.get(frame.tick_id)
        if not dets:
            continue
        if max_speed is not None and frame.speed > max_speed:
            continue
        if max_yaw_rate is not None and frame.yaw_rate > max_yaw_rate:
            continue

        scan = frame.scan()
        points = (
            accum.update(scan, frame.position, frame.orientation)
            if accum is not None else scan
        )
        for det in dets:
            if float(det["score"]) < min_score:
                continue
            stats.n_detections += 1
            result = lifter.lift(
                mask=decode_mask(det["mask_rle"]),
                scan_points_map=points,
                pose_position=frame.position,
                pose_orientation=frame.orientation,
            )
            if result.position is None:
                stats.n_below_min_inliers += 1
                continue
            stats.n_lifted += 1
            pts = result.inlier_points
            assert pts is not None
            med = np.median(pts, axis=0)
            stats.spreads.append(
                float(np.percentile(np.linalg.norm(pts - med, axis=1), 95)),
            )
            stats.labels.append(det["label"])
            if omap is not None:
                omap.add(det["label"], float(det["score"]), pts)

    nodes = omap.export() if omap is not None else []
    return _scene_dict(nodes), stats


def _scene_dict(nodes: list[dict]) -> dict:
    """ObjectMap export → the scene-graph shape ``perception.eval`` reads."""
    objects = []
    for i, nd in enumerate(nodes):
        box = nd["bbox_aabb"]
        objects.append({
            "object_id": i,
            "label": nd["label"],
            "position": nd["center_3d"],
            "confidence": nd["score"],
            "bbox_min": box["min"],
            "bbox_max": box["max"],
            "is_structure": nd.get("is_structure", False),
            "n_obs": nd.get("n_obs"),
            "n_pts": nd.get("n_pts"),
            "spatial_relations": [],
        })
    return {"tick_id": -1, "room": None, "viewpoints": [], "objects": objects}


def iter_frames_with_detections(
    frames_dir: str | Path,
) -> Iterator[tuple[Frame, list[dict]]]:
    """Convenience for ad-hoc analysis: pair each frame with its cached masks."""
    detections = load_detections(frames_dir)
    for frame in load_frames(frames_dir):
        yield frame, detections.get(frame.tick_id, [])
