"""File-based VLM tick logger for post-run debugging and visualization.

Activated by setting ``XIAO_HEI_VLM_LOG_DIR``. When active, every tick
writes one JSON line to ``ticks.jsonl`` and saves the camera frame as a
JPEG in ``images/``. A ``session.json`` is written once at startup.

Typical on-disk layout::

    vlm_logs/
      session_20260530_143022/
        session.json
        ticks.jsonl
        images/
          tick_000001.jpg
          tick_000004.jpg
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from xiao_hei_vln.messages import VLMInput, VLMOutput
from xiao_hei_vln.qwen.config import QwenConfig

log = logging.getLogger(__name__)


class VLMLogger:
    """Append-only per-session tick logger."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        config: QwenConfig,
        responder_name: str,
        tick_hz: float,
    ) -> None:
        start = datetime.now(tz=UTC)
        session_name = f"session_{start.strftime('%Y%m%d_%H%M%S')}"
        self._session_dir = Path(log_dir) / session_name
        self._images_dir = self._session_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        session_meta: dict[str, Any] = {
            "start_time": start.isoformat(),
            "responder": responder_name,
            "tick_hz": tick_hz,
            "config": _config_to_dict(config),
        }
        (self._session_dir / "session.json").write_text(
            json.dumps(session_meta, indent=2) + "\n",
        )

        self._jsonl_fh: IO[str] = (self._session_dir / "ticks.jsonl").open(
            "a",
            buffering=1,
        )

        log.info("VLMLogger session started: %s", self._session_dir)

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def log_tick(
        self,
        snapshot: VLMInput,
        system_prompt: str,
        user_text: str,
        output: VLMOutput | None,
        inference_ms: float,
        evidence: list[str],
    ) -> None:
        image_path: str | None = None
        if snapshot.image is not None:
            image_path = f"images/tick_{snapshot.tick_id:06d}.jpg"
            self._save_image(snapshot, self._session_dir / image_path)

        record: dict[str, Any] = {
            "tick_id": snapshot.tick_id,
            "tick_time": snapshot.tick_time.to_seconds(),
            "question_text": snapshot.question.text if snapshot.question else None,
            "question_type": snapshot.question.type.value if snapshot.question else None,
            "pose": _serialize_pose(snapshot),
            "system_prompt": system_prompt,
            "user_text": user_text,
            "output": output.model_dump() if output is not None else None,
            "inference_ms": round(inference_ms, 2),
            "evidence": list(evidence),
            "image_path": image_path,
        }
        self._jsonl_fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._jsonl_fh.close()
        log.info("VLMLogger session closed: %s", self._session_dir)

    # ------------------------------------------------------------------

    @staticmethod
    def _save_image(snapshot: VLMInput, path: Path) -> None:
        from xiao_hei_vln.qwen.image_utils import image_frame_to_pil

        assert snapshot.image is not None
        img = image_frame_to_pil(snapshot.image)
        img.save(os.fspath(path), format="JPEG", quality=90)


def _config_to_dict(config: QwenConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


def _serialize_pose(snapshot: VLMInput) -> dict[str, Any] | None:
    if snapshot.pose is None:
        return None
    p = snapshot.pose.position
    q = snapshot.pose.orientation
    return {
        "position": {"x": p.x, "y": p.y, "z": p.z},
        "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
    }
