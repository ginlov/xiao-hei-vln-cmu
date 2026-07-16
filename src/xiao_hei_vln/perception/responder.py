"""Perception responder — pipeline-grade VLN responder.

Two-phase shape: Phase A walks the coverage trajectory; Phase B answers
the question. Object information comes from the perception sidecar:
``PerceptionResponder`` calls the sidecar's ``/detect`` route on the
current camera frame and projects each returned mask through the LiDAR
scan to lift to 3D before pushing into :class:`SceneRepresentation`.

Questions are answered against ``scene.objects`` — which contains only
what the perception path actually observed during this run (no
see-through-walls).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.messages.outputs import (
    NumericalResponse,
    ObjectReferenceResponse,
    Waypoint,
    WaypointPathResponse,
)
from xiao_hei_vln.messages.question import QuestionType
from xiao_hei_vln.perception.client import HTTPPerceptionClient
from xiao_hei_vln.perception.lifter import PointLifter
from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator
from xiao_hei_vln.perception.vocab import Vocabulary
from xiao_hei_vln.scene import ObjectObservation

if TYPE_CHECKING:
    from xiao_hei_vln.logger import VLMLogger
    from xiao_hei_vln.messages.inputs import VLMInput
    from xiao_hei_vln.messages.outputs import VLMOutput
    from xiao_hei_vln.perception.object_map import ObjectMap
    from xiao_hei_vln.scene import SceneRepresentation


log = logging.getLogger(__name__)


DEFAULT_NEAR_THRESHOLD = 2.0       # m — "near" relation threshold
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5


class PerceptionResponder:
    """Responder that wires the perception sidecar into the scene graph."""

    def __init__(
        self,
        scene: SceneRepresentation,
        *,
        client: HTTPPerceptionClient,
        lifter: PointLifter,
        vocabulary: Vocabulary,
        near_threshold: float = DEFAULT_NEAR_THRESHOLD,
        trajectory_path: Path | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        take_waypoint_reached_signals: Callable[[], int] | None = None,
        logger: VLMLogger | None = None,
        object_map: ObjectMap | None = None,
        scan_accumulator: ScanAccumulator | None = None,
    ) -> None:
        """
        Args:
            scene: the live SceneRepresentation maintained by the app.
            client: HTTP client to the perception sidecar (``/detect``).
            lifter: 2D mask → 3D point projector.
            vocabulary: open-vocab class list manager. The responder
                pushes the current question's vocabulary to the sidecar
                via :meth:`HTTPPerceptionClient.set_classes` whenever
                the set changes.
            near_threshold: XY radius (m) for the on-demand
                ``derive_near_relations`` call at answer time.
            trajectory_path: optional Task 7 coverage-trajectory JSON.
                Walked during Phase A. Without it, the responder
                answers from tick 0 with no movement.
            score_threshold, iou_threshold: forwarded to the sidecar's
                YOLO-World call per tick.
            take_waypoint_reached_signals: callable returning the count
                of ``/way_point_reached`` ROS messages since the last
                call. This is the **sole** Phase A advance trigger —
                Phase A advances on each new signal, regardless of
                where the robot physically is. The autonomy stack
                publishes this when *its* internally-adjusted waypoint
                (a nearby safe point, not necessarily our commanded
                one) is reached; we trust that as the definition of
                "done with this waypoint". ``None`` (test default)
                means the responder never advances — fine for Phase B
                tests that bypass the trajectory walk via an empty
                ``trajectory_path``, but Phase A unit tests must
                inject a stateful counter.
            logger: optional VLMLogger; receives per-tick records.
            object_map: optional :class:`ObjectMap`. When supplied, each
                detection's lifted LiDAR cloud is fused here (cross-frame
                point-cloud union → converged 3D box, NMS, wall-sheet
                rejection) and the fused snapshot is synced into ``scene``
                every tick via ``sync_from_object_map`` — instead of the
                per-detection ``scene.add_object`` path. ``None`` keeps the
                historical add_object behaviour.
            scan_accumulator: optional :class:`ScanAccumulator`. Densifies
                the per-tick registered scan with a rolling window of
                keyframes before lifting, so small objects clear the
                lifter's ``min_inliers`` gate with genuine on-surface
                returns. Defaults to a standard accumulator; the buffer
                persists across questions (the physical scene is the same
                for the whole session).
        """
        self._scene = scene
        self._client = client
        self._lifter = lifter
        self._object_map = object_map
        self._scan_accum = scan_accumulator or ScanAccumulator()
        self._vocab = vocabulary
        self._near_threshold = float(near_threshold)
        self._score_threshold = float(score_threshold)
        self._iou_threshold = float(iou_threshold)

        self._waypoints: list[Waypoint] = (
            _load_waypoints(trajectory_path) if trajectory_path is not None else []
        )
        self._wp_idx = 0
        self._trajectory_done = not self._waypoints
        self._done = False

        self._take_waypoint_reached_signals = (
            take_waypoint_reached_signals or (lambda: 0)
        )

        self._logger = logger
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Responder protocol
    # ------------------------------------------------------------------

    def respond(self, snapshot: VLMInput) -> VLMOutput | None:
        # Scene maintenance runs every tick (even after the answer)
        # so the logged scene graph keeps evolving through the rest
        # of the question window. Object↔object ``near`` edges are
        # only derived on demand at answer time — see _compute_output.
        self._inject_visible(snapshot)

        if self._done or snapshot.question is None:
            return None

        self._tick_count += 1
        if self._tick_count == 1 and self._logger is not None:
            self._logger.new_question(snapshot.question.text)

        t0 = time.perf_counter()
        output = self._compute_output(snapshot)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        if self._logger is not None:
            try:
                self._logger.log_tick(
                    snapshot,
                    system_prompt="PerceptionResponder",
                    user_text=snapshot.question.text,
                    output=output,
                    inference_ms=inference_ms,
                    evidence=[],
                )
            except Exception:  # noqa: BLE001 — never let logging break the loop
                log.exception("VLMLogger.log_tick failed; continuing")

        return output

    def ingest(self, snapshot: VLMInput) -> None:
        """Build the scene from this frame *without* answering.

        Runs only the detect → lift → add_object cycle. The exploration
        phase calls this every tick so the scene graph keeps growing while
        the explorer drives movement — importantly, it does **not** emit an
        answer even when a question is already active, so a question that
        arrives mid-exploration is deferred until exploration completes.

        When a question is present its text still flows into the detector
        vocabulary (via :meth:`_inject_visible`), so the queried object is
        actively looked for during the remaining sweep.
        """
        self._inject_visible(snapshot)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._wp_idx = 0
        self._trajectory_done = not self._waypoints
        self._done = False
        self._tick_count = 0

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            if self._logger is not None:
                self._logger.close()

    # ------------------------------------------------------------------
    # Phase split
    # ------------------------------------------------------------------

    def _compute_output(self, snapshot: VLMInput) -> VLMOutput | None:
        qtype = snapshot.question.type

        # Phase A — walk the coverage trajectory before answering, so
        # the robot moves and the scene graph grows. Advance solely on
        # ``/way_point_reached`` signals: the autonomy stack adjusts
        # our commanded waypoint to a nearby safe point internally and
        # publishes the signal when *that* (not our literal commanded
        # XY) has been reached. A distance-based check based on our
        # commanded waypoint can deadlock when the safe point and our
        # commanded point are >0 m apart — the autonomy stack will
        # never let the robot reach our literal waypoint, so we'd
        # never advance.
        if not self._trajectory_done:
            current = self._waypoints[self._wp_idx]
            reached_signals = self._take_waypoint_reached_signals()
            if reached_signals > 0:
                if self._wp_idx + 1 >= len(self._waypoints):
                    self._trajectory_done = True
                else:
                    self._wp_idx += 1
                    current = self._waypoints[self._wp_idx]
            if qtype is QuestionType.INSTRUCTION_FOLLOWING and self._trajectory_done:
                self._done = True
            return WaypointPathResponse(
                waypoints=[current],
                rationale=(
                    f"PerceptionResponder: coverage wp "
                    f"{self._wp_idx + 1}/{len(self._waypoints)}"
                ),
            )

        # Phase B — answer from the live scene graph. Derive ``near``
        # edges now so the logged snapshot has them at answer time.
        self._scene.derive_near_relations(self._near_threshold)
        if qtype is QuestionType.NUMERICAL:
            ans = self._answer_numerical(snapshot.question.text)
        elif qtype is QuestionType.OBJECT_REFERENCE:
            ans = self._answer_object_reference(snapshot)
        else:
            return self._stub_instruction(snapshot)

        self._done = True
        return ans

    # ------------------------------------------------------------------
    # Scene maintenance — detect → lift → add_object
    # ------------------------------------------------------------------

    def _inject_visible(self, snapshot: VLMInput) -> None:
        """Run a detect → lift → add_object cycle for this tick.

        Silently skipped when the snapshot lacks the inputs the
        pipeline needs (no image, no pose, no scan). Empties from the
        sidecar (e.g. a motion-blurred frame) just leave the scene
        unchanged this tick.
        """
        if snapshot.image is None or snapshot.pose is None or snapshot.registered_scan is None:
            return

        classes = self._vocab.current_classes(
            snapshot.question.text if snapshot.question is not None else None,
        )
        if not classes:
            return
        # set_classes() is dedup-cached; only fires a network call when
        # the list actually changes.
        self._client.set_classes(classes)

        try:
            bgr = _image_frame_to_bgr(snapshot.image)
        except Exception:
            log.exception("failed to decode image frame; skipping tick")
            return

        detections = self._client.detect(
            bgr,
            classes=None,                # rely on the cached set
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold,
        )
        if not detections:
            return

        # Densify the sparse single sweep with a rolling window of
        # keyframes (map-frame, so directly concatenable) before lifting —
        # a lone sweep leaves small objects below the lifter's min_inliers
        # gate. Returns the same cloud on near-stationary ticks.
        scan_points = self._scan_accum.update(
            snapshot.registered_scan.points,
            snapshot.pose.position,
            snapshot.pose.orientation,
        )
        for det in detections:
            result = self._lifter.lift(
                mask=det.mask,
                scan_points_map=scan_points,
                pose_position=snapshot.pose.position,
                pose_orientation=snapshot.pose.orientation,
            )
            if result.position is None:
                continue
            color_rgb, color_name = _mask_color(bgr, det.mask)
            if self._object_map is not None:
                # Fuse this detection's lifted cloud across frames; the scene
                # object layer is rebuilt from the fused snapshot below.
                self._object_map.add(
                    det.label, det.score, result.inlier_points,
                    color_rgb, color_name,
                )
                continue
            self._scene.add_object(ObjectObservation(
                label=det.label,
                position=result.position,
                confidence=det.score,
                color_rgb=color_rgb,      # median RGB of the masked pixels
                color_name=color_name,    # nearest basic-colour label
                bbox_min=None,            # AABB from a 2D mask isn't a 3D bbox;
                bbox_max=None,            # leave None until we estimate it.
            ))

        if self._object_map is not None:
            # One fused snapshot → scene objects (with real 3D boxes) per tick.
            self._scene.sync_from_object_map(self._object_map.export())

    # ------------------------------------------------------------------
    # Question handlers — read from the live scene graph
    # ------------------------------------------------------------------

    def _answer_numerical(self, text: str) -> NumericalResponse:
        target = self._find_label_in_text(text)
        if target is None:
            return NumericalResponse(
                value=0,
                rationale="PerceptionResponder: no scene label matched the question",
            )
        count = sum(1 for o in self._scene.objects if _labels_match(o.label, target))
        return NumericalResponse(
            value=count,
            rationale=(
                f"PerceptionResponder: counted {count} '{target}' from scene graph"
            ),
        )

    def _answer_object_reference(
        self, snapshot: VLMInput,
    ) -> ObjectReferenceResponse | None:
        target = self._find_label_in_text(snapshot.question.text)
        if target is None:
            return None
        candidates = [o for o in self._scene.objects if _labels_match(o.label, target)]
        if not candidates:
            return None
        origin = snapshot.pose.position if snapshot.pose is not None else None
        if origin is not None:
            candidates.sort(key=lambda o: _xy_dist(o.position, origin))
        chosen = candidates[0]
        return ObjectReferenceResponse(
            label=chosen.label,
            object_id=chosen.object_id,
            center=chosen.position,
            size=_size_from_obs(chosen),
            heading=0.0,                 # we don't estimate orientation
            rationale=(
                f"PerceptionResponder: closest '{target}' to current pose"
                if origin is not None else
                f"PerceptionResponder: first '{target}' in scene graph"
            ),
        )

    def _stub_instruction(self, snapshot: VLMInput) -> VLMOutput | None:
        if snapshot.pose is None:
            return None
        pos = snapshot.pose.position
        return WaypointPathResponse(
            waypoints=[Waypoint(x=pos.x, y=pos.y, heading=0.0)],
            rationale=(
                "PerceptionResponder: instruction following is not implemented; "
                "holding position"
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_label_in_text(self, text: str) -> str | None:
        """Look for any scene-graph label (or its plural) in the question.

        Multi-word labels (``"dining chair"``) take precedence over their
        single-word substring (``"chair"``) because we sort by descending
        length. We match against labels currently *in the scene graph* —
        i.e. things the perception path has actually detected — so a
        question for a label we never observed cleanly returns ``None``
        and the numeric handler answers 0.
        """
        labels = sorted(
            {o.label for o in self._scene.objects},
            key=lambda s: -len(s),
        )
        normalised = _norm_label(text)
        for label in labels:
            lab = _norm_label(label)
            if _word_in(lab, normalised) or _word_in(lab + "s", normalised):
                return label
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_label(s: str) -> str:
    """Lower-case and collapse underscores/whitespace runs to single spaces."""
    return re.sub(r"[_\s]+", " ", s.lower())


def _labels_match(scene_label: str, target: str) -> bool:
    """Loose label compare — case-insensitive, whitespace-collapsed."""
    return _norm_label(scene_label) == _norm_label(target)


def _xy_dist(a: Vector3, b: Vector3) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _size_from_obs(obs: ObjectObservation) -> Vector3:
    """Derive a (size_x, size_y, size_z) from the observation's 3D AABB.

    Returns a zero-vector when no bbox is available — that's the
    typical case for the perception path today (we'd need to also
    project the mask through the LiDAR scan and fit a 3D extent,
    which is a Phase 4 tuning item).
    """
    if obs.bbox_min is None or obs.bbox_max is None:
        return Vector3(x=0.0, y=0.0, z=0.0)
    return Vector3(
        x=abs(obs.bbox_max.x - obs.bbox_min.x),
        y=abs(obs.bbox_max.y - obs.bbox_min.y),
        z=abs(obs.bbox_max.z - obs.bbox_min.z),
    )


def _image_frame_to_bgr(image_frame) -> "np.ndarray":  # type: ignore[name-defined]
    """Convert an :class:`ImageFrame` into a ``(H, W, 3)`` BGR ndarray.

    The image arrives as raw bytes in ``bgr8`` encoding (per the
    challenge sim); the layout is ``height * step``. We trust the
    metadata and reshape directly.
    """
    import numpy as np

    if image_frame.encoding != "bgr8":
        raise ValueError(
            f"expected bgr8 image encoding, got {image_frame.encoding!r}",
        )
    arr = np.frombuffer(image_frame.data, dtype=np.uint8)
    return arr.reshape(image_frame.height, image_frame.step // 3, 3)[
        :, : image_frame.width, :
    ]


# Basic-colour anchors in RGB (0-255). A detection's median colour is
# labelled by nearest Euclidean anchor. Kept small and perceptually spread
# so the label is stable — questions ask for coarse colours ("the red
# samovar"), not exact shades.
_COLOR_ANCHORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("gray", (128, 128, 128)),
    ("red", (200, 30, 30)),
    ("orange", (230, 140, 30)),
    ("yellow", (230, 220, 50)),
    ("green", (40, 160, 60)),
    ("blue", (40, 80, 200)),
    ("purple", (130, 60, 170)),
    ("pink", (235, 150, 190)),
    ("brown", (120, 75, 45)),
)


def _mask_color(
    bgr: "np.ndarray",  # type: ignore[name-defined]
    mask: "np.ndarray",  # type: ignore[name-defined]
) -> tuple[tuple[int, int, int] | None, str | None]:
    """Return ``((r, g, b), name)`` for the pixels under ``mask``.

    The median (per channel) is used — robust to specular highlights and
    mask-edge bleed, the same reasoning the point-lifter uses for XYZ.
    Returns ``(None, None)`` when the mask is empty or its shape doesn't
    line up with the frame, so a colourless detection just carries no
    colour rather than a bogus one.
    """
    import numpy as np

    if bgr.ndim != 3 or bgr.shape[:2] != mask.shape:
        return None, None
    pixels = bgr[mask.astype(bool)]           # (K, 3) in BGR order
    if pixels.shape[0] == 0:
        return None, None
    b, g, r = np.median(pixels, axis=0)
    rgb = (int(r), int(g), int(b))
    return rgb, _nearest_color_name(rgb)


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    """Nearest basic-colour label to ``rgb`` by squared Euclidean distance."""
    r, g, b = rgb
    best_name = _COLOR_ANCHORS[0][0]
    best_d = float("inf")
    for name, (ar, ag, ab) in _COLOR_ANCHORS:
        d = (r - ar) ** 2 + (g - ag) ** 2 + (b - ab) ** 2
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def _word_in(needle: str, haystack: str) -> bool:
    """Whole-word substring check that respects multi-word phrases."""
    pattern = r"(?<![a-z])" + re.escape(needle) + r"(?![a-z])"
    return re.search(pattern, haystack) is not None


def _load_waypoints(path: Path) -> list[Waypoint]:
    """Parse the ``waypoints`` array out of a Task 7 trajectory JSON."""
    data = json.loads(path.read_text())
    raw = data.get("waypoints", [])
    return [
        Waypoint(x=float(w["x"]), y=float(w["y"]),
                 heading=float(w.get("heading", 0.0)))
        for w in raw
    ]
