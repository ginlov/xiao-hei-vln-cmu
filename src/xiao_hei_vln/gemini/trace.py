"""Full-fidelity trace log of every Gemini API call, for debugging.

:class:`GeminiTracer` records — as one JSON line per call — the complete
request and response of each :meth:`GeminiEngine.infer_multimodal` call:

  - **request**: model, system prompt, user text (which embeds the scene
    graph JSON), image count + byte sizes, sampling knobs
  - **response**: the *raw* text Gemini returned (before JSON parsing),
    finish reason, and token usage
  - **parsed**: the ``VLMOutput`` we parsed out (or ``null`` on failure)
  - **latency_ms** and **error** (the exception repr when the call raised)

Because it hooks the engine — the single chokepoint every call flows
through — it captures both the offline batch harness and the live ROS
responder without either needing to know about it.

The trace is append-only JSONL, so tail it live or post-process it::

    jq -r 'select(.error != null)' gemini_trace.jsonl        # failed calls
    jq -r '.response.usage.total_tokens' gemini_trace.jsonl  # token cost

Optionally, ``save_images=True`` writes each inline image to a sibling
``<stem>_images/`` directory and records the paths (useful for the live
panorama/occupancy path; the offline harness sends no images).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class GeminiTracer:
    """Append-only JSONL recorder for Gemini calls."""

    def __init__(self, path: str | Path, *, save_images: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._save_images = save_images
        self._images_dir = self._path.with_name(self._path.stem + "_images")
        self._n = 0

    @property
    def path(self) -> Path:
        return self._path

    def save_images(self, images: list[bytes]) -> list[str]:
        """Persist inline images; return their relative paths (or []).

        No-op returning byte-sizes-only markers unless ``save_images`` was
        enabled at construction.
        """
        if not images or not self._save_images:
            return []
        self._images_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for blob in images:
            ext = "png" if blob[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
            digest = hashlib.sha1(blob).hexdigest()[:16]
            fp = self._images_dir / f"{digest}.{ext}"
            if not fp.exists():
                fp.write_bytes(blob)
            paths.append(str(fp.relative_to(self._path.parent)))
        return paths

    def record(
        self,
        *,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        parsed: dict[str, Any] | None,
        latency_ms: float,
        error: str | None,
    ) -> None:
        """Append one call record. Never raises — logging must not break a run."""
        self._n += 1
        rec = {
            "ts": datetime.now(UTC).isoformat(),
            "call_index": self._n,
            "latency_ms": round(latency_ms, 1),
            "request": request,
            "response": response,
            "parsed": parsed,
            "error": error,
        }
        try:
            with self._path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — tracing must never break inference
            log.exception("GeminiTracer failed to write record; continuing")
