"""Question-directed VLM navigation, as an ExplorationStrategy.

:class:`NavTask1Explorer` reuses the whole :class:`NavVLMExplorer` machinery
(occupancy grid, reachability snapping, async off-thread calls, the reach /
skip / stuck-watchdog trigger policy) but changes the *goal*: instead of
undirected exploration, it drives the robot toward the object named in an
OBJECT_REFERENCE question, and completes when the model declares it has
ARRIVED at that object.

Differences from the base explorer:

  * It **holds** (returns no waypoint) until a question is present — there is no
    target object to head for before then. The scene graph is still built every
    tick by the responder's ``ingest`` (2 Hz), so the map fills in while it
    waits.
  * Each model call is fed the QUESTION plus the objects detected so far (the
    shared :class:`~xiao_hei_vln.scene.SceneRepresentation`), so the model can
    head for the target once it appears in the scene graph.
  * ``done`` from the model means "arrived at the target object", which flips
    :meth:`is_complete` — the app tick loop then hands over to the answering
    responder (``scene_claude``), which dumps the scene graph to Claude.

Because it slots into the exact same explorer surface, the app's existing
reach / skip / odometry supervisor drives ``advance()`` / ``force_skip()`` for
it unchanged — that is the "call the model only on reach or skip" trigger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from xiao_hei_vln.exploration._nav_vlm import NavVLMExplorer
from xiao_hei_vln.messages.inputs import VLMInput
from xiao_hei_vln.messages.outputs import Waypoint
from xiao_hei_vln.nav_vlm.render import render_grid_png, trajectory_summary
from xiao_hei_vln.nav_vlm.task1_prompts import (
    OBJECT_VOCAB_TOOL,
    VOCAB_SYSTEM_PROMPT,
    build_nav_user_text,
    build_vocab_user_text,
    scene_objects_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    from xiao_hei_vln.nav_vlm.config import NavVLMConfig
    from xiao_hei_vln.nav_vlm.engine import NavVLMEngineProtocol, WaypointProposal
    from xiao_hei_vln.scene import SceneRepresentation

log = logging.getLogger(__name__)

# Non-object words to strip from a question before treating the rest as the
# named target/anchor nouns. Kept tiny and generic (relations + articles +
# imperatives) so it never accidentally drops a real object word.
_QUESTION_STOPWORDS = frozenset({
    "find", "locate", "identify", "go", "move", "navigate", "get", "the", "a",
    "an", "of", "to", "on", "in", "at", "by", "near", "nearest", "closest",
    "close", "next", "beside", "between", "is", "are", "that", "this", "which",
    "what", "where", "and", "with", "from", "please", "robot", "toward",
    "towards", "closest", "furthest", "farthest", "left", "right", "front",
    "back", "behind", "above", "below", "over", "under",
})


def _target_nouns(text: str) -> set[str]:
    """Best-effort object nouns from a question (the named target + anchor).

    e.g. "Find the vase closest to the hookah." -> {"vase", "hookah"}. Purely
    lexical and generic — used only to decide whether the object the question
    is about has actually been detected yet, never to pick the answer.
    """
    words = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text).split()
    return {w for w in words if len(w) >= 3 and w not in _QUESTION_STOPWORDS}


class NavTask1Explorer(NavVLMExplorer):
    """Navigate to the object named in the question; complete on arrival."""

    def __init__(
        self,
        engine: NavVLMEngineProtocol,
        *,
        scene: SceneRepresentation | None = None,
        config: NavVLMConfig | None = None,
        explore_after_skips: int = 4,
        max_question_seconds: float = 600.0,
        coverage_plateau_s: float = 45.0,
        min_visited_before_plateau: int = 4,
        dynamic_vocab: set[str] | None = None,
        max_dynamic_vocab: int = 30,
        class_thresholds: dict[str, float] | None = None,
        default_score_threshold: float = 0.4,
        verify_threshold_step: float = 0.1,
        verify_threshold_floor: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(engine, config=config, **kwargs)
        self._scene = scene
        self._question_text: str | None = None
        # Shared with the perception Vocabulary: the objects the model reports
        # seeing (+ synonyms) get added here so the open-vocab detector is
        # primed to find rare items its default prompt would miss (#1).
        self._dynamic_vocab = dynamic_vocab
        self._max_dynamic_vocab = max_dynamic_vocab
        # Shared with the perception responder: per-class score-threshold
        # overrides. When the model reports seeing an object that perception
        # missed (verify_objects), we lower that class's threshold a step so a
        # low-confidence detection is not wrongly excluded — for that class only.
        self._class_thresholds = class_thresholds
        self._default_score_threshold = default_score_threshold
        self._verify_step = verify_threshold_step
        self._verify_floor = verify_threshold_floor
        # Deterministic arrival backstop: once the scene graph stops gaining new
        # views for this long (it has seen every reachable angle), stop and
        # answer instead of wandering until the time cap. 0 disables it.
        self._coverage_plateau_s = coverage_plateau_s
        self._min_visited_before_plateau = min_visited_before_plateau
        self._last_novelty = 0
        self._last_novelty_time: float | None = None
        # reach ↔ explore state machine. In "reach" the robot drives to the
        # target; after this many consecutive failed approaches it flips to
        # "explore" (reveal the room to find a reachable side), and flips back
        # to "reach" as soon as exploring yields a novel view (the scene graph
        # grows). A hard per-question wall-clock cap bounds the whole thing.
        self._mode = "reach"
        self._explore_after_skips = explore_after_skips
        self._max_question_seconds = max_question_seconds
        self._question_start: float | None = None
        self._novelty_at_explore = 0
        # The object nouns the question is about (target + anchor). Coverage
        # cannot count as "complete" until at least one of these is actually in
        # the scene graph — otherwise the plateau backstop arrives on a graph
        # that lacks the very object being asked about (fresh-graph runs stopped
        # at ~10 waypoints before reaching a target across the room).
        self._question_classes: set[str] = set()

    # ------------------------------------------------------------------
    # Strategy interface

    def update(self, snapshot: VLMInput) -> Waypoint | None:
        if self._done:
            return None
        # Keep the map fresh every tick, even before a question arrives.
        self._ingest_observation(snapshot)

        if snapshot.question is None:
            return None  # nothing to head for yet — hold

        now = snapshot.tick_time.to_seconds()
        if snapshot.question.text != self._question_text:
            self._on_new_question(snapshot.question.text, now)

        # Navigation deadline: stop driving in time for the answer to still land
        # inside the total per-question budget (this cap is total - answer
        # reserve). Completing hands over to the responder, which answers.
        if (
            self._question_start is not None
            and now - self._question_start > self._max_question_seconds
        ):
            if not self._done:
                log.info(
                    "nav_task1: nav budget (%.0fs) reached — stopping to answer "
                    "within the remaining reserve",
                    self._max_question_seconds,
                )
            self._done = True
            return None

        self._maybe_switch_mode()
        if self._check_coverage_plateau(now):
            return None
        return self._step(now)

    def reset(self) -> None:
        super().reset()
        self._question_text = None
        self._mode = "reach"
        self._question_start = None
        self._novelty_at_explore = 0
        self._last_novelty = 0
        self._last_novelty_time = None
        self._question_classes = set()

    # ------------------------------------------------------------------
    # reach ↔ explore state machine

    def _scene_novelty(self) -> int:
        """A monotone signature of scene-graph richness: #objects + total views.

        Grows whenever a new object is detected or an existing one is seen from
        a new viewpoint — i.e. whenever exploring has revealed something new.
        """
        if self._scene is None:
            return 0
        objs = self._scene.to_dict().get("objects") or []
        return len(objs) + sum(len(o.get("observing_viewpoint_ids") or []) for o in objs)

    def _maybe_switch_mode(self) -> None:
        # Reach keeps failing → explore for a reachable vantage instead of
        # giving up (the base class's skip-cap termination is disabled for
        # nav_task1; the time budget is the only hard stop).
        if self._mode == "reach" and self._consecutive_skip_count >= self._explore_after_skips:
            self._mode = "explore"
            self._novelty_at_explore = self._scene_novelty()
            self._reset_skip_state()
            log.info("nav_task1: target unreachable here — switching to EXPLORE")
        # A novel view (scene graph grew while exploring) → try reaching again.
        elif self._mode == "explore" and self._scene_novelty() > self._novelty_at_explore:
            self._mode = "reach"
            self._reset_skip_state()
            log.info("nav_task1: novel view acquired — switching back to REACH")

    def _expand_question_vocab(self, question_text: str) -> None:
        """One Claude call at question start: add the question's objects +
        synonyms to the shared detector vocabulary. Best-effort — a failure or
        an engine without ``call_tool`` (test fakes) just skips it."""
        if self._dynamic_vocab is None:
            return
        call_tool = getattr(self._engine, "call_tool", None)
        if call_tool is None:
            return
        try:
            data = call_tool(
                system=VOCAB_SYSTEM_PROMPT,
                tool=OBJECT_VOCAB_TOOL,
                user_text=build_vocab_user_text(question_text),
                images=[],
            )
        except Exception as exc:  # noqa: BLE001 — vocab priming must not break nav
            log.warning("nav_task1: question vocab expansion failed: %s", exc)
            return
        added = 0
        for name in data.get("classes") or []:
            clean = str(name).strip().lower()
            if (
                clean and len(clean) < 40
                and clean not in self._dynamic_vocab
                and len(self._dynamic_vocab) < self._max_dynamic_vocab
            ):
                self._dynamic_vocab.add(clean)
                added += 1
        log.info(
            "nav_task1: primed detector vocab with %d question class(es): %s",
            added, sorted(self._dynamic_vocab),
        )

    def _absorb_verify_objects(self, proposal: WaypointProposal) -> None:
        """The model sees these objects but perception missed them: lower each
        class's detection threshold a step (per-class only), and keep priming
        the vocabulary. Stop relaxing a class once it is in the scene graph."""
        if self._class_thresholds is None or not proposal.verify_objects:
            return
        detected = self._detected_labels()
        for name in proposal.verify_objects:
            clean = name.strip().lower()
            if not clean or len(clean) >= 40 or clean in detected:
                continue
            # ensure the detector is also looking for it by name
            dv = self._dynamic_vocab
            if dv is not None and len(dv) < self._max_dynamic_vocab:
                dv.add(clean)
            cur = self._class_thresholds.get(clean, self._default_score_threshold)
            new = max(self._verify_floor, round(cur - self._verify_step, 3))
            if new < cur:
                self._class_thresholds[clean] = new
                log.info("nav_task1: lowering detect threshold for '%s' -> %.2f", clean, new)

    def _target_detected(self) -> bool:
        """Has the object the question names actually shown up in the graph yet?

        Fuzzy (substring either way) so "vase" matches "flower vase" and the
        detector's own wording matches the question's. Unknown target (no nouns
        parsed) counts as satisfied, so behaviour degrades to the old backstop.
        """
        if not self._question_classes:
            return True
        labels = self._detected_labels()
        return any(
            q in lbl or lbl in q
            for q in self._question_classes
            for lbl in labels
            if lbl
        )

    def _detected_labels(self) -> set[str]:
        if self._scene is None:
            return set()
        return {
            str(o.get("label", "")).strip().lower()
            for o in (self._scene.to_dict().get("objects") or [])
        }

    def _absorb_visible_objects(self, proposal: WaypointProposal) -> None:
        """Feed the model's panorama observations into the shared detector vocab
        so rare objects it names get looked for. Bounded so the open-vocab
        prompt (and its runtime) does not grow without limit."""
        if self._dynamic_vocab is None or not proposal.visible_objects:
            return
        for name in proposal.visible_objects:
            clean = name.strip().lower()
            if clean and len(clean) < 40 and len(self._dynamic_vocab) < self._max_dynamic_vocab:
                self._dynamic_vocab.add(clean)

    def _check_coverage_plateau(self, now: float) -> bool:
        """Arrive once the scene graph stops revealing anything new.

        Novelty (objects + total views) growing resets the timer; if it has been
        flat for ``coverage_plateau_s`` after the robot has moved a bit, every
        reachable view has been collected, so stop and answer. Returns True when
        it triggered completion.
        """
        if self._coverage_plateau_s <= 0 or self._done:
            return False
        nov = self._scene_novelty()
        if self._last_novelty_time is None or nov > self._last_novelty:
            self._last_novelty = nov
            self._last_novelty_time = now
            return False
        if (
            # An EMPTY scene graph (novelty 0) is not "coverage complete" — it
            # looks identical to a fully-collected one under a flat signal, and
            # arriving on it hands the answerer nothing. Only plateau once we
            # have actually detected something to reason about.
            self._last_novelty > 0
            # …and specifically the object the question is ABOUT. A local
            # novelty plateau elsewhere in the room must not end the search
            # before the named target has been found (bounded by the nav
            # budget, which still stops a genuinely-undetectable target).
            and self._target_detected()
            and len(self._visited) >= self._min_visited_before_plateau
            and now - self._last_novelty_time > self._coverage_plateau_s
        ):
            log.info(
                "nav_task1: scene coverage plateaued (%.0fs no new views) — "
                "arriving to answer",
                self._coverage_plateau_s,
            )
            self._last_rationale = (
                f"coverage plateau: no new views for {self._coverage_plateau_s:.0f}s"
            )
            self._done = True
            return True
        return False

    def _reset_skip_state(self) -> None:
        self._consecutive_skip_count = 0
        self._failed_history.clear()
        self._last_failure = None
        self._failed_xy = None

    # ------------------------------------------------------------------
    # Internals

    def _on_new_question(self, text: str, now: float) -> None:
        """Restart the navigation state for a freshly received target object.

        The occupancy grid is intentionally kept — the physical room is the
        same — but the current target, pending call, skip/visit counters, mode,
        and per-question timer are reset so the search begins cleanly.
        """
        had_prior = self._question_text is not None
        self._question_text = text
        self._question_start = now
        self._mode = "reach"
        self._last_novelty = 0
        self._last_novelty_time = None
        self._question_classes = _target_nouns(text)
        # Prime the detector vocabulary with the question's objects + synonyms
        # up front, so rare words (e.g. "hookah") are searched for from the
        # start rather than depending on the model seeing them in the panorama.
        self._expand_question_vocab(text)
        if not had_prior:
            return
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._current_target = None
        self._target_set_time = None
        self._visited = []
        self.skipped_count = 0
        self._consecutive_skip_count = 0
        self._last_failure = None
        self._failed_xy = None
        self._failed_history.clear()
        self._propose_failures = 0

    def _render_occupancy(self) -> bytes:
        """Occupancy map + the detected objects drawn on it, so the model can
        route toward a target it can SEE rather than a bare text coordinate."""
        return render_grid_png(
            self._grid,
            robot_xy=self._robot_xy,
            failed_xy=self._failed_xy,
            failed_points=self._failed_history,
            trajectory_xy=self._trajectory,
            objects=self._object_markers(),
            dpi=self._occupancy_dpi,
        )

    def _object_markers(self, *, limit: int = 40) -> list[tuple[float, float, str]]:
        """(x, y, label) for detected objects, tolerating either position shape."""
        if self._scene is None:
            return []
        markers: list[tuple[float, float, str]] = []
        for o in (self._scene.to_dict().get("objects") or [])[:limit]:
            pos = o.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                x, y = pos[0], pos[1]
            elif isinstance(pos, dict):
                x, y = pos.get("x"), pos.get("y")
            else:
                continue
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                markers.append((x, y, str(o.get("label", "?"))))
        return markers

    def _build_user_text(self) -> str:
        scene_dict = self._scene.to_dict() if self._scene is not None else {}
        return build_nav_user_text(
            question=self._question_text or "",
            robot_xy=self._robot_xy,
            robot_yaw=self._robot_yaw,
            scene_summary=scene_objects_summary(scene_dict),
            failure_reason=self._last_failure,
            failed_points=self._failed_history,
            waypoints_taken=len(self._visited),
            trajectory_summary=trajectory_summary(self._trajectory),
            mode=self._mode,
        )

    def _apply_proposal(self, proposal: WaypointProposal, now: float) -> None:
        self._absorb_visible_objects(proposal)
        self._absorb_verify_objects(proposal)
        if proposal.done:
            log.info(
                "nav_task1: model signalled ARRIVED at target object (%s)",
                proposal.rationale,
            )
            self._last_rationale = proposal.rationale
            self._done = True
            return
        super()._apply_proposal(proposal, now)
