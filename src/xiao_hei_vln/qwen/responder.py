"""Qwen3.5 VLM responder.

Same shape as `DummyResponder` so `app.main` can swap them via env
var. Takes an `EngineProtocol` so tests can inject a fake engine and
exercise the loop without CUDA / vLLM.

Loop semantics:

- The engine returns a `VLMOutput`. Numerical / object-reference are
  treated as terminal (sets `is_done = True`); a waypoint_path emitted
  in response to a numerical/object question is treated as a
  navigation step toward a better viewpoint, and the responder stays
  not-done.
- For numerical questions the loop uses the Phase 3 prompt
  (`build_numerical_user_message`) and a dedicated system prompt that
  asks the model to thread its running tally through the `rationale`
  field on every reply. The rationale is captured verbatim into the
  evidence log so the next tick sees it.
- For instruction-following questions, a waypoint_path is published
  one waypoint at a time (matching `VLMOutputPublisher`), and we
  consider the question answered once the model has emitted at least
  one path (the underlying ROS publisher + `/way_point_reached` drive
  the per-waypoint advancement on the system side).
- A safety cap (`max_ticks_per_question`) prevents infinite loops on
  numerical questions: once hit, we emit a `NumericalResponse(value=0)`
  so the evaluation node always sees an answer.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from xiao_hei_vln.messages import (
    NumericalResponse,
    ObjectReferenceResponse,
    QuestionType,
    VLMInput,
    VLMOutput,
    WaypointPathResponse,
)
from xiao_hei_vln.qwen.config import QwenConfig
from xiao_hei_vln.qwen.prompts import (
    build_numerical_system_prompt,
    build_numerical_user_message,
    build_system_prompt,
    build_user_message,
)

if TYPE_CHECKING:
    from xiao_hei_vln.qwen.engine import EngineProtocol
    from xiao_hei_vln.logger import VLMLogger

log = logging.getLogger(__name__)


class QwenResponder:
    def __init__(
        self,
        engine: EngineProtocol,
        config: QwenConfig | None = None,
        *,
        logger: VLMLogger | None = None,
    ) -> None:
        self._engine = engine
        self._config = config or QwenConfig()
        self._logger = logger
        self._generic_system_prompt = build_system_prompt()
        self._numerical_system_prompt = build_numerical_system_prompt()

        self._done = False
        self._tick_count = 0
        self._evidence: list[str] = []

    # --- main entry --------------------------------------------------------------

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.question is None or self._done:
            return None

        if self._tick_count >= self._config.max_ticks_per_question:
            return self._timeout_response(snapshot.question.type)

        self._tick_count += 1

        if self._tick_count == 1 and self._logger is not None:
            self._logger.new_question(snapshot.question.text)

        system_prompt, user_text = self._build_prompts(snapshot, snapshot.question)

        output: VLMOutput | None = None
        t0 = time.perf_counter()
        try:
            output = self._engine.infer(
                system=system_prompt,
                user_text=user_text,
                image=snapshot.image,
            )
        except Exception:
            log.exception("Qwen engine inference failed; skipping tick")
        finally:
            inference_ms = (time.perf_counter() - t0) * 1000.0
            if self._logger is not None:
                try:
                    self._logger.log_tick(
                        snapshot,
                        system_prompt,
                        user_text,
                        output,
                        inference_ms,
                        list(self._evidence),
                    )
                except Exception:
                    log.exception("VLMLogger.log_tick failed; continuing")

        if output is None:
            return None

        self._record_evidence(snapshot.question.type, output)
        self._update_done_flag(snapshot.question.type, output)
        return output

    # --- lifecycle ---------------------------------------------------------------

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._done = False
        self._tick_count = 0
        self._evidence = []

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()

    # --- internals ---------------------------------------------------------------

    def _build_prompts(
        self,
        snapshot: VLMInput,
        question,
    ) -> tuple[str, str]:
        if question.type is QuestionType.NUMERICAL:
            user_text = build_numerical_user_message(
                snapshot,
                question,
                self._evidence,
                tick_index=self._tick_count,
                max_ticks=self._config.max_ticks_per_question,
            )
            return self._numerical_system_prompt, user_text
        user_text = build_user_message(snapshot, question, self._evidence)
        return self._generic_system_prompt, user_text

    def _update_done_flag(self, qtype: QuestionType, output: VLMOutput) -> None:
        if isinstance(output, (NumericalResponse, ObjectReferenceResponse)):
            self._done = True
            return
        # WaypointPathResponse: terminal for instruction-following, otherwise a
        # navigation step toward a better viewpoint.
        if qtype is QuestionType.INSTRUCTION_FOLLOWING:
            self._done = True

    def _record_evidence(self, qtype: QuestionType, output: VLMOutput) -> None:
        # For numerical questions we trust the prompt protocol: the model's
        # rationale is the only state that crosses ticks, so we capture it
        # verbatim and skip our own coarse summary.
        if qtype is QuestionType.NUMERICAL and output.rationale:
            self._evidence.append(f"tick {self._tick_count}: {output.rationale.strip()}")
            return

        if isinstance(output, NumericalResponse):
            entry = f"answered NUMERICAL={output.value}"
        elif isinstance(output, ObjectReferenceResponse):
            c = output.center
            entry = (
                f"answered OBJECT label={output.label!r} "
                f"center=({c.x:.2f},{c.y:.2f},{c.z:.2f})"
            )
        elif isinstance(output, WaypointPathResponse):
            wp = output.waypoints[0]
            entry = (
                f"navigated toward waypoint ({wp.x:.2f},{wp.y:.2f}) "
                f"[{len(output.waypoints)} in path]"
            )
        else:  # pragma: no cover - exhaustiveness check
            return

        if output.rationale:
            entry = f"{entry} :: {output.rationale.strip()}"
        self._evidence.append(entry)

    def _timeout_response(self, qtype: QuestionType) -> VLMOutput | None:
        self._done = True
        if qtype is QuestionType.NUMERICAL:
            log.warning(
                "Qwen responder timed out after %d ticks; emitting NumericalResponse(value=0)",
                self._config.max_ticks_per_question,
            )
            return NumericalResponse(
                value=0,
                rationale=f"timeout after {self._config.max_ticks_per_question} ticks",
            )
        log.warning(
            "Qwen responder timed out after %d ticks on a %s question; no fallback emit",
            self._config.max_ticks_per_question,
            qtype.value,
        )
        return None
