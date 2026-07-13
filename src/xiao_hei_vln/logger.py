"""File-based VLM tick logger for post-run debugging and visualization.

Activated by setting ``XIAO_HEI_VLM_LOG_DIR``. When active, every tick
writes one JSON line to a per-question ``ticks.jsonl``, saves the
camera frame as a JPEG, and saves any available point-cloud arrays as
``.npy`` files. A ``session.json`` is written once at startup.

Typical on-disk layout::

    vlm_logs/
      session_20260530_143022/
        session.json
        q_001_how_many_chairs/
          ticks.jsonl
          images/
            tick_000003.jpg
          pointclouds/
            tick_000003_registered.npy
            tick_000003_terrain_local.npy
        q_002_find_the_red_cup/
          ticks.jsonl
          images/
            tick_000007.jpg
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from xiao_hei_vln.messages import VLMInput, VLMOutput

log = logging.getLogger(__name__)


class VLMLogger:
    """Append-only per-session tick logger with per-question subdirectories."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        config: dict[str, Any],
        responder_name: str,
        tick_hz: float,
    ) -> None:
        start = datetime.now(tz=UTC)
        session_name = f"session_{start.strftime('%Y%m%d_%H%M%S')}"
        self._session_dir = Path(log_dir) / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        session_meta: dict[str, Any] = {
            "start_time": start.isoformat(),
            "responder": responder_name,
            "tick_hz": tick_hz,
            "config": config,
        }
        try:
            (self._session_dir / "session.json").write_text(
                json.dumps(session_meta, indent=2) + "\n",
            )
        except (TypeError, ValueError):
            log.warning(
                "session.json: config is not JSON-serializable; "
                "callers must pass a plain-dict config (e.g. dataclasses.asdict())",
            )

        self._question_count = 0
        self._question_dir: Path | None = None
        self._images_dir: Path | None = None
        self._pointclouds_dir: Path | None = None
        self._jsonl_fh: IO[str] | None = None
        self._attached_scene: Any = None

        log.info("VLMLogger session started: %s", self._session_dir)

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def attach_scene(self, scene: Any) -> None:
        """Bind a scene object whose ``.to_dict()`` is logged per tick.

        Once attached, callers can omit the ``scene`` kwarg from
        :meth:`log_tick`; the logger pulls a fresh snapshot via
        ``scene.to_dict()`` for every record. Pass an explicit
        ``scene`` to ``log_tick`` to override on a per-call basis.
        Duck-typed — any object exposing a ``to_dict() -> dict``
        method works (no import of :class:`SceneRepresentation` here).
        """
        self._attached_scene = scene

    def new_question(self, question_text: str) -> None:
        """Open a new per-question subdirectory and JSONL file."""
        self._close_question()
        self._question_count += 1
        slug = _slugify(question_text)
        dir_name = f"q_{self._question_count:03d}_{slug}"
        self._question_dir = self._session_dir / dir_name
        self._images_dir = self._question_dir / "images"
        self._pointclouds_dir = self._question_dir / "pointclouds"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._pointclouds_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_fh = (self._question_dir / "ticks.jsonl").open(
            "a",
            buffering=1,
        )
        log.info("VLMLogger new question: %s", self._question_dir.name)

    def log_tick(
        self,
        snapshot: VLMInput,
        system_prompt: str,
        user_text: str,
        output: VLMOutput | None,
        inference_ms: float,
        evidence: list[str],
        scene: dict[str, Any] | None = None,
    ) -> None:
        if self._jsonl_fh is None or self._question_dir is None:
            return

        image_path: str | None = None
        if snapshot.image is not None and self._images_dir is not None:
            image_path = f"images/tick_{snapshot.tick_id:06d}.jpg"
            self._save_image(snapshot, self._question_dir / image_path)

        pc_paths = self._save_pointclouds(snapshot)

        record: dict[str, Any] = {
            "tick_id": snapshot.tick_id,
            "tick_time": snapshot.tick_time.to_seconds(),
            "question_text": (
                snapshot.question.text if snapshot.question else None
            ),
            "question_type": (
                snapshot.question.type.value if snapshot.question else None
            ),
            "pose": _serialize_pose(snapshot),
            "system_prompt": system_prompt,
            "user_text": user_text,
            "output": output.model_dump() if output is not None else None,
            "inference_ms": round(inference_ms, 2),
            "evidence": list(evidence),
            "image_path": image_path,
            "pointclouds": pc_paths,
        }
        if scene is None and self._attached_scene is not None:
            scene = self._attached_scene.to_dict()
        if scene is not None:
            record["scene"] = scene
        self._jsonl_fh.write(
            json.dumps(record, separators=(",", ":")) + "\n",
        )

    def write_prediction(self, question_text: str, output: VLMOutput) -> None:
        """Append a final prediction for offline evaluation."""
        record = {
            "question": question_text,
            "prediction": output.model_dump(),
        }
        with (self._session_dir / "predictions.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._close_question()
        log.info("VLMLogger session closed: %s", self._session_dir)

    # ------------------------------------------------------------------

    def _close_question(self) -> None:
        if self._jsonl_fh is not None and not self._jsonl_fh.closed:
            self._jsonl_fh.close()
        self._jsonl_fh = None
        self._question_dir = None
        self._images_dir = None
        self._pointclouds_dir = None

    @staticmethod
    def _save_image(snapshot: VLMInput, path: Path) -> None:
        from xiao_hei_vln.image_utils import image_frame_to_pil

        assert snapshot.image is not None
        img = image_frame_to_pil(snapshot.image)
        img.save(os.fspath(path), format="JPEG", quality=90)

    def _save_pointclouds(self, snapshot: VLMInput) -> dict[str, str]:
        import numpy as np

        if self._pointclouds_dir is None:
            return {}

        pc_paths: dict[str, str] = {}
        tid = snapshot.tick_id

        sensors: list[tuple[str, Any]] = [
            ("registered", snapshot.registered_scan),
            ("sensor", snapshot.sensor_scan),
            ("terrain_local", snapshot.terrain_local),
            ("terrain_ext", snapshot.terrain_ext),
        ]
        for name, sensor in sensors:
            if sensor is None:
                continue
            rel = f"pointclouds/tick_{tid:06d}_{name}.npy"
            np.save(
                self._question_dir / rel,  # type: ignore[arg-type]
                sensor.points,
                allow_pickle=False,
            )
            pc_paths[name] = rel

        return pc_paths


def _slugify(text: str, max_len: int = 40) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    slug = slug[:max_len].rstrip("_")
    return slug or "untitled"


def _serialize_pose(snapshot: VLMInput) -> dict[str, Any] | None:
    if snapshot.pose is None:
        return None
    p = snapshot.pose.position
    q = snapshot.pose.orientation
    return {
        "position": {"x": p.x, "y": p.y, "z": p.z},
        "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
    }
