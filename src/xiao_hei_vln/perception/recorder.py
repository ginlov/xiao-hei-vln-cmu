"""Raw-frame recorder — the input side of the offline perception replay.

Tuning the perception pipeline against the live simulator is slow and not
reproducible: every run explores differently, so a metric delta can just as
easily be the explorer taking another turn as the change under test. This
recorder freezes one run's *sensor input* to disk so the whole detect → lift →
fuse chain can be re-run offline, deterministically, as many times as we like
(see :mod:`xiao_hei_vln.perception.replay`).

Activated by ``XIAO_HEI_FRAME_RECORD_DIR``. Layout::

    frames/
      frames.jsonl                  one record per captured tick
      tick_000004.jpg               equirect camera frame
      tick_000004_registered.npy    /registered_scan points, map frame (N, 4)

``frames.jsonl`` deliberately keeps **all three message stamps** (image, pose,
scan) next to the tick time, plus the odometry velocities. The lift pairs the
latest image with the latest pose, and those two arrive at different rates —
the recorded stamps are what let us measure that skew offline instead of
guessing at it.

Unlike :class:`~xiao_hei_vln.logger.VLMLogger`, this records during the
exploration phase too: the logger only writes from ``respond()`` (which needs
an active question), so a pure exploration run leaves no frames behind at all.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import IO, Any

from xiao_hei_vln.messages.inputs import VLMInput

log = logging.getLogger(__name__)

DEFAULT_STRIDE: int = 1


class FrameRecorder:
    """Append-only recorder of (image, registered scan, pose, stamps) per tick.

    Args:
        out_dir: directory to write into; created if missing.
        stride: record every Nth *recordable* tick (a tick missing an image,
            a pose or a scan is not recordable and never counts). ``1``
            records everything the run produced.
        meta: optional run metadata (scene, env config) written to
            ``session.json`` — replay reports are worthless without knowing
            which scene and which detector settings produced the frames.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        stride: int = DEFAULT_STRIDE,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._dir = Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._stride = max(1, int(stride))
        self._seen = 0
        self._written = 0
        self._fh: IO[str] = (self._dir / "frames.jsonl").open("a", buffering=1)
        if meta is not None:
            try:
                (self._dir / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
            except (TypeError, ValueError):
                log.warning("frame recorder: session meta is not JSON-serializable")
        log.info("FrameRecorder writing to %s (stride=%d)", self._dir, self._stride)

    @property
    def out_dir(self) -> Path:
        return self._dir

    @property
    def n_written(self) -> int:
        return self._written

    def record(self, snapshot: VLMInput) -> bool:
        """Persist this tick's frame. Returns True when a frame was written.

        A tick without image / pose / registered scan cannot be replayed, so
        it is skipped silently (cold start, dropped frame). Recording is
        best-effort: an I/O failure is logged and swallowed rather than
        allowed to take down the ROS node mid-run.
        """
        if (
            snapshot.image is None
            or snapshot.pose is None
            or snapshot.registered_scan is None
        ):
            return False

        self._seen += 1
        if (self._seen - 1) % self._stride != 0:
            return False

        tid = snapshot.tick_id
        image_rel = f"tick_{tid:06d}.jpg"
        scan_rel = f"tick_{tid:06d}_registered.npy"
        try:
            self._save_image(snapshot, self._dir / image_rel)
            self._save_scan(snapshot, self._dir / scan_rel)
            self._fh.write(
                json.dumps(self._record(snapshot, image_rel, scan_rel),
                           separators=(",", ":")) + "\n",
            )
        except Exception:  # noqa: BLE001 — recording must never break the run
            log.exception("frame recorder: failed to write tick %d", tid)
            return False

        self._written += 1
        return True

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        log.info("FrameRecorder closed: %d frames in %s", self._written, self._dir)

    # ------------------------------------------------------------------

    @staticmethod
    def _record(snapshot: VLMInput, image_rel: str, scan_rel: str) -> dict[str, Any]:
        pose = snapshot.pose
        assert pose is not None and snapshot.image is not None
        assert snapshot.registered_scan is not None
        p, q = pose.position, pose.orientation
        return {
            "tick_id": snapshot.tick_id,
            "tick_time": snapshot.tick_time.to_seconds(),
            # The three stamps that make the lift's pose↔image pairing
            # measurable offline. Do not drop them.
            "image_stamp": snapshot.image.header.stamp.to_seconds(),
            "pose_stamp": pose.header.stamp.to_seconds(),
            "scan_stamp": snapshot.registered_scan.header.stamp.to_seconds(),
            "position": [p.x, p.y, p.z],
            "orientation": [q.x, q.y, q.z, q.w],
            # The pose rewound to the image stamp — what the lift actually
            # projects with. Recorded separately from the newest pose so the
            # replay reproduces the online lift, and so the two can be
            # compared to see how much the time sync was worth.
            "image_pose_position": _xyz(
                snapshot.image_pose.position if snapshot.image_pose else None
            ),
            "image_pose_orientation": (
                [
                    snapshot.image_pose.orientation.x,
                    snapshot.image_pose.orientation.y,
                    snapshot.image_pose.orientation.z,
                    snapshot.image_pose.orientation.w,
                ]
                if snapshot.image_pose else None
            ),
            "linear_velocity": _xyz(pose.linear_velocity),
            "angular_velocity": _xyz(pose.angular_velocity),
            "question": snapshot.question.text if snapshot.question else None,
            "image": image_rel,
            "scan": scan_rel,
        }

    @staticmethod
    def _save_image(snapshot: VLMInput, path: Path) -> None:
        from xiao_hei_vln.image_utils import image_frame_to_pil

        assert snapshot.image is not None
        image_frame_to_pil(snapshot.image).save(
            os.fspath(path), format="JPEG", quality=95,
        )

    @staticmethod
    def _save_scan(snapshot: VLMInput, path: Path) -> None:
        import numpy as np

        assert snapshot.registered_scan is not None
        np.save(path, snapshot.registered_scan.points, allow_pickle=False)


def _xyz(v) -> list[float] | None:
    """``Vector3 | None`` → ``[x, y, z] | None`` (odometry velocities are optional)."""
    return None if v is None else [v.x, v.y, v.z]


def from_env(meta: dict[str, Any] | None = None) -> FrameRecorder | None:
    """Build a recorder from ``XIAO_HEI_FRAME_RECORD_DIR``, or ``None`` when unset.

    ``XIAO_HEI_FRAME_RECORD_STRIDE`` (default 1) thins the capture — a long run
    at 2 Hz is a lot of frames, and consecutive ones are near-duplicates.
    """
    out_dir = os.environ.get("XIAO_HEI_FRAME_RECORD_DIR", "").strip()
    if not out_dir:
        return None
    try:
        stride = int(os.environ.get("XIAO_HEI_FRAME_RECORD_STRIDE", str(DEFAULT_STRIDE)))
    except ValueError:
        stride = DEFAULT_STRIDE
    try:
        return FrameRecorder(out_dir, stride=stride, meta=meta)
    except OSError:
        log.exception("frame recorder: cannot open %s; recording disabled", out_dir)
        return None
