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
from xiao_hei_vln.perception.size_prior import size_for
from xiao_hei_vln.perception.vocab import Vocabulary
from xiao_hei_vln.scene import ObjectObservation

if TYPE_CHECKING:
    from xiao_hei_vln.logger import VLMLogger
    from xiao_hei_vln.messages.inputs import VLMInput
    from xiao_hei_vln.messages.outputs import VLMOutput
    from xiao_hei_vln.perception.object_map import ObjectMap
    from xiao_hei_vln.scene import SceneRepresentation


log = logging.getLogger(__name__)


# 0.25 was set when the class prior held 28 labels. It now holds 110, so the
# detector scores four times as many phrases against every region and the weak
# tail is mostly labels that never fire confidently at all — `photo` lands
# below 0.35 in 98% of its detections. Swept offline over 3 recorded scenes,
# 0.35 cut the object count by a third and the counting error by 26% while
# recall and mAP held (0.463 -> 0.459, 0.316 -> 0.325).
#
# Was raised to 0.6 to keep low-confidence fragments out of the sidecar's
# face-seam merge (a 0.9 detection could otherwise be unioned with a 0.3
# fragment) — but a per-class confidence audit (TASK 34) showed 0.6 silently
# drops real objects whose YOLO score peaks below it (door 0.52, table 0.60,
# door frame 0.59), while noise-floor classes never clear it anyway. Lowered to
# 0.4 and paired with the SAM mask-quality gate below, which removes the weak
# masks the lower floor lets in. Net effect measured on arabic_room: see
# DEFAULT_SAM_THRESHOLD. Override with XIAO_HEI_PERCEPTION_SCORE_THRESHOLD.
DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.5
# SAM mask-quality gate (B5). Detections whose SAM confidence is below this are
# dropped before lifting. Pairs with the lowered 0.4 score floor: 0.4 alone adds
# recall but also weak/bleeding masks, and the SAM gate keeps the extra
# detections clean. Measured on arabic_room (gate on 0.3 m, live sidecar):
# 0.4 + SAM 0.8 vs the old 0.6 raised mAP@1.0 0.235 -> 0.337 (+43%), recall
# 0.235 -> 0.383, IoU@0.25 0.091 -> 0.134, at a precision cost (0.79 -> 0.62).
# Override with XIAO_HEI_PERCEPTION_SAM_THRESHOLD (0.0 = off).
DEFAULT_SAM_THRESHOLD = 0.8
# Capture-time viewpoint-novelty gate (TASK 33). Perception (detect -> lift ->
# fuse) runs only when the robot has moved farther than this from every
# previously perceived viewpoint; the scan accumulator still ingests every tick.
# A 360 panorama makes frame content position-only, so novelty is pure
# translation. Stops a dwelling robot from flooding a node with redundant
# partial views (which shrinks + biases its box).
#
# DEFAULT 0.3 m (ON). Measured offline on arabic_room (TASK 33, flat estimator
# on): a 0.3 m gate cuts perceived frames 397 -> 36 (~11x less compute) and
# raises box IoU@0.25 (0.051 -> 0.091). It costs some centre-distance mAP@0.5
# (0.221 -> 0.199) because a few objects only cleared the score/inlier gates from
# a now-skipped closer pose, but on the live robot the accumulator still ingests
# LiDAR every tick (so the cloud stays dense) and the compute headroom + box
# quality win the tradeoff. Override with XIAO_HEI_NOVEL_VIEWPOINT_M (0.0 = off).
DEFAULT_NOVEL_VIEWPOINT_M = 0.3


class PerceptionResponder:
    """Responder that wires the perception sidecar into the scene graph."""

    def __init__(
        self,
        scene: SceneRepresentation,
        *,
        client: HTTPPerceptionClient,
        lifter: PointLifter,
        vocabulary: Vocabulary,
        trajectory_path: Path | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        sam_threshold: float = DEFAULT_SAM_THRESHOLD,
        novel_viewpoint_m: float = DEFAULT_NOVEL_VIEWPOINT_M,
        take_waypoint_reached_signals: Callable[[], int] | None = None,
        logger: VLMLogger | None = None,
        object_map: ObjectMap,
        scan_accumulator: ScanAccumulator,
        class_thresholds: dict[str, float] | None = None,
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
            object_map: the :class:`ObjectMap` every detection's lifted
                LiDAR cloud is fused into (cross-frame point-cloud union →
                converged 3D box, NMS, wall-sheet rejection). The fused
                snapshot is synced into ``scene`` every tick via
                ``sync_from_object_map``.
            scan_accumulator: the :class:`ScanAccumulator` that densifies
                the per-tick registered scan with a rolling window of
                keyframes before lifting, so small objects clear the
                lifter's ``min_inliers`` gate with genuine on-surface
                returns. The buffer persists across questions (the physical
                scene is the same for the whole session).
        """
        self._scene = scene
        self._client = client
        self._lifter = lifter
        self._object_map = object_map
        self._scan_accum = scan_accumulator
        self._vocab = vocabulary
        self._score_threshold = float(score_threshold)
        self._iou_threshold = float(iou_threshold)
        self._sam_threshold = float(sam_threshold)
        # Shared per-class threshold overrides (VLM verify-recall). None/empty =
        # the normal single-threshold behaviour.
        self._class_thresholds = class_thresholds if class_thresholds is not None else {}
        # Viewpoint-novelty gate state: xy of every viewpoint we have actually
        # perceived from. A new tick perceives only if it is farther than
        # `_novel_viewpoint_m` from ALL of these (nearest-neighbour over all
        # kept, so a revisit/loop does not re-admit an already-covered pose).
        self._novel_viewpoint_m = float(novel_viewpoint_m)
        self._kept_xy: list[tuple[float, float]] = []

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

        Runs only the detect → lift → fuse cycle. The exploration
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

        # Phase B — answer from the live scene graph.
        if qtype is QuestionType.NUMERICAL:
            ans = self._answer_numerical(snapshot.question.text)
        elif qtype is QuestionType.OBJECT_REFERENCE:
            ans = self._answer_object_reference(snapshot)
        else:
            return self._stub_instruction(snapshot)

        self._done = True
        return ans

    # ------------------------------------------------------------------
    # Scene maintenance — detect → lift → fuse
    # ------------------------------------------------------------------

    def _viewpoint_is_novel(self, position) -> bool:
        """True when ``position`` is farther than ``_novel_viewpoint_m`` from
        every viewpoint we have already perceived from (nearest-neighbour over
        all kept). ``<= 0`` disables the gate — always novel."""
        if self._novel_viewpoint_m <= 0.0 or not self._kept_xy:
            return True
        px, py = float(position.x), float(position.y)
        nearest = min(
            math.hypot(px - kx, py - ky) for kx, ky in self._kept_xy
        )
        return nearest > self._novel_viewpoint_m

    def _inject_visible(self, snapshot: VLMInput) -> None:
        """Run a detect → lift → fuse cycle for this tick.

        Silently skipped when the snapshot lacks the inputs the
        pipeline needs (no image, no pose, no scan). Empties from the
        sidecar (e.g. a motion-blurred frame) just leave the scene
        unchanged this tick.
        """
        if snapshot.image is None or snapshot.pose is None or snapshot.registered_scan is None:
            return

        # Project with the pose as it was when this *image* was captured, not
        # the newest one. The camera lags the 200 Hz odometry, so the two
        # differ by however far the robot moved in between — and that offset
        # lands directly on every lifted point. Falls back to the current pose
        # when no interpolated one is available (no pose history yet, or a
        # caller that builds VLMInput by hand).
        lift_pose = snapshot.image_pose or snapshot.pose

        # Optionally densify the sweep with a rolling window of keyframes.
        # Off by default: measured on two scenes, accumulation costs more than
        # it gives once the lifter clusters by depth — it mixes returns from
        # viewpoints metres apart into one cloud, which inflates every box
        # (livingroom_3 size error 2.28x accumulated vs 1.28x from the single
        # sweep; chinese_room 1.52x vs 0.87x) and drops precision.
        #
        # This runs EVERY tick — including ticks the novelty gate skips
        # perception on — so the cloud stays dense/registered regardless of
        # whether we detect this frame (TASK 33).
        scan_points = (
            self._scan_accum.update(
                snapshot.registered_scan.points,
                lift_pose.position,
                lift_pose.orientation,
            )
            if self._scan_accum is not None
            else snapshot.registered_scan.points
        )

        classes = self._vocab.current_classes(
            snapshot.question.text if snapshot.question is not None else None,
        )
        if not classes:
            return

        # Viewpoint-novelty gate: skip the expensive detect → lift → fuse when
        # the robot has not moved far enough from every viewpoint already
        # perceived. A dwelling robot otherwise floods a node with redundant
        # near-identical partial views, shrinking and biasing its box (TASK 33).
        if not self._viewpoint_is_novel(lift_pose.position):
            return
        self._kept_xy.append((float(lift_pose.position.x), float(lift_pose.position.y)))

        # set_classes() is dedup-cached; only fires a network call when
        # the list actually changes.
        self._client.set_classes(classes)

        try:
            bgr = _image_frame_to_bgr(snapshot.image)
        except Exception:
            log.exception("failed to decode image frame; skipping tick")
            return

        # Per-class threshold overrides (VLM-requested "verify" recall boost):
        # detect at the LOWEST active floor so low-confidence candidates for a
        # relaxed class come back, then re-apply the threshold per class — every
        # class the model has not relaxed keeps the normal floor, so global
        # precision is unchanged. With no overrides this is exactly the old path.
        detect_floor = self._score_threshold
        if self._class_thresholds:
            detect_floor = min(detect_floor, min(self._class_thresholds.values()))
        detections = self._client.detect(
            bgr,
            classes=None,                # rely on the cached set
            score_threshold=detect_floor,
            iou_threshold=self._iou_threshold,
        )
        if detect_floor < self._score_threshold:
            detections = [
                d for d in detections
                if d.score >= self._class_thresholds.get(
                    d.label.strip().lower(), self._score_threshold
                )
            ]
        if self._sam_threshold > 0.0:                # B5 mask-quality gate
            detections = [d for d in detections
                          if d.sam_score >= self._sam_threshold]
        if not detections:
            return
        # The pose here is already matched to the image's stamp by LatestCache
        # (TASK 27), so it is the pose the camera had when the frame was taken —
        # no per-lift time-skew correction is needed.
        for det in detections:
            result = self._lifter.lift(
                mask=det.mask,
                scan_points_map=scan_points,
                pose_position=lift_pose.position,
                pose_orientation=lift_pose.orientation,
            )
            if result.position is None:
                continue
            color_rgb, color_name = _mask_color(bgr, det.mask)
            # Fuse this detection's lifted cloud across frames; the scene
            # object layer is rebuilt from the fused snapshot below.
            self._object_map.add(
                det.label, det.score, result.inlier_points,
                color_rgb, color_name,
            )

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
    """Best available (size_x, size_y, size_z) for a detected object.

    The measured box wins whenever we have one. That is not obvious — a
    partial LiDAR view only covers the faces we saw — but it is what the
    numbers say: once the box comes from gated percentiles rather than raw
    min/max, its median error against ground truth is 1.27x, and per-instance
    measurement beats a class median (mean IoU 0.150 measured vs 0.109 with
    the prior, on the livingroom_3 corpus). The prior was a good idea when
    boxes were 6.8x oversized; it is a regression now.

    The class prior (:mod:`xiao_hei_vln.perception.size_prior`) stays as the
    fallback for detections that never got a box at all — the alternative
    there is a zero-volume box, which scores exactly nothing.
    """
    if obs.bbox_min is not None and obs.bbox_max is not None:
        return Vector3(
            x=abs(obs.bbox_max.x - obs.bbox_min.x),
            y=abs(obs.bbox_max.y - obs.bbox_min.y),
            z=abs(obs.bbox_max.z - obs.bbox_min.z),
        )

    prior = size_for(obs.label)
    if prior is not None:
        return Vector3(x=prior[0], y=prior[1], z=prior[2])
    return Vector3(x=0.0, y=0.0, z=0.0)


def _image_frame_to_bgr(image_frame) -> np.ndarray:  # type: ignore[name-defined]
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
    bgr: np.ndarray,  # type: ignore[name-defined]
    mask: np.ndarray,  # type: ignore[name-defined]
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
