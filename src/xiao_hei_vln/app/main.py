"""rclpy entry point: drives the Task-1 stack at 1 Hz with the configured responder.

Pick the responder with `XIAO_HEI_RESPONDER`:

  - `dummy` (default) — the deterministic port of `dummyVLM.cpp`. No GPU.
  - `perception`      — YOLOv8x-World v2 + SAM 2.1 Hiera Tiny via the
                        perception sidecar (`XIAO_HEI_PERCEPTION_BASE_URL`).
                        Detects + segments objects per tick, projects
                        masks through the LiDAR scan to lift to 3D,
                        and answers from the live scene graph.
  - `scene_gemini`    — the submission pipeline. The shared frontier
                        explorer sweeps the scene while the perception
                        sidecar builds the object scene graph
                        (`ingest()`); once exploration completes, the
                        populated graph + panorama + occupancy map are
                        handed to Gemini for the final answer (Task 1) or
                        route plan (Task 2). Requires
                        `XIAO_HEI_GEMINI_API_KEY` and the perception
                        sidecar. Build with
                        `XIAO_HEI_EXTRA=perception,gemini,exploration`.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.scene import SceneRepresentation
from xiao_hei_vln.sync import LatestCache

# 2 Hz default: the viewpoint-novelty gate (TASK 36) now runs detect→lift→fuse
# only on position-novel ticks, so a faster tick no longer floods the pipeline —
# it just samples poses more finely without extra perception work when the robot
# dwells. Override with XIAO_HEI_VLM_TICK_HZ.
TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0"))
RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy").lower()

# Exploration phase — set XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0 to disable.
# Exploration is NOT interrupted when a question arrives: it runs until the
# strategy completes (budget exhausted, consecutive-skip hatch, or no frontiers
# remain), and only then does the responder answer — from the fully-built scene.
_EXPLORATION_MAX_WAYPOINTS = int(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINTS", "500"))
_EXPLORATION_STRATEGY = os.environ.get("XIAO_HEI_EXPLORATION_STRATEGY", "frontier").lower()
_EXPLORATION_MAX_WAYPOINT_DIST = float(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST", "1.5"))
_EXPLORATION_LOG_DIR = os.environ.get("XIAO_HEI_EXPLORATION_LOG_DIR", "")
# Wall-clock budget for the sweep (8 min; 0 disables). None of the strategy's
# own stop conditions is bounded in time, so a big scene with reachable
# frontiers left never hands over to the responder.
_EXPLORATION_MAX_SECONDS = float(os.environ.get("XIAO_HEI_EXPLORATION_MAX_SECONDS", "480"))
# How often to re-write exploration.png / rviz.png mid-sweep (0 = on DONE only).
# A run killed before DONE would otherwise leave no image at all.
_EXPLORATION_SNAPSHOT_S = float(os.environ.get("XIAO_HEI_EXPLORATION_SNAPSHOT_S", "30"))
# How often to log a MAP heartbeat (0 = off). WP_* events alone say nothing
# about coverage — the objective had to be reconstructed from pose bounding
# boxes in TASK 50 — and nothing at all about the gaps between waypoints.
_EXPLORATION_MAP_LOG_S = float(os.environ.get("XIAO_HEI_EXPLORATION_MAP_LOG_S", "10"))
# Scene the sim is running. Only the basename is meaningful here — the value is
# a *host* path (compose bind-mounts it into the sim, not into this container),
# so we never open it, we only name the log dir after it.
_EXPLORATION_SCENE = (
    os.path.basename(os.environ.get("XIAO_HEI_SCENE_DIR_HOST", "").rstrip("/"))
    or "default_scene"
)
# The strategy is part of the artefact path too: comparing two algorithms on one
# scene is the common case, and a shared directory means the second run destroys
# the first's evidence. "no_exploration" keeps a disabled run from claiming — and
# truncating — a real strategy's directory.
_EXPLORATION_RUN_LABEL = (
    _EXPLORATION_STRATEGY if _EXPLORATION_MAX_WAYPOINTS > 0 else "no_exploration"
)


def _exploration_dir() -> Path:
    """Where this run's log and images go: <log dir>/<scene>/<strategy>/.

    Keying on both means consecutive runs never clobber each other —
    exploration.log opens with mode "w", so a collision is a deletion.
    """
    return (
        Path(_EXPLORATION_LOG_DIR or "/exploration_logs")
        / _EXPLORATION_SCENE
        / _EXPLORATION_RUN_LABEL
    )


@dataclass(frozen=True)
class _PerceptionSettings:
    """The `XIAO_HEI_PERCEPTION_*` / `XIAO_HEI_SCAN_*` knobs shared by every
    perception-backed responder.

    Split out from `_build_responder` so the `perception` and `scene_gemini`
    branches read the environment through one code path — they used to parse
    the same variables independently, which meant a default could drift
    between the two responders without anything failing.
    """

    base_url: str
    score_threshold: float
    sam_threshold: float
    min_inliers: int
    scan_keyframes: int
    scan_voxel_m: float
    novel_viewpoint_m: float
    request_timeout_s: float

    @classmethod
    def from_env(cls) -> _PerceptionSettings:
        # Imported lazily: these modules pull in httpx/pycocotools (the
        # `perception` extra), and the package must stay importable without it.
        from xiao_hei_vln.perception.client import (
            DEFAULT_BASE_URL,
            DEFAULT_REQUEST_TIMEOUT_S,
        )
        from xiao_hei_vln.perception.lifter import DEFAULT_MIN_INLIERS
        from xiao_hei_vln.perception.responder import (
            DEFAULT_NOVEL_VIEWPOINT_M,
            DEFAULT_SAM_THRESHOLD,
            DEFAULT_SCORE_THRESHOLD,
        )

        return cls(
            base_url=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", DEFAULT_BASE_URL),
            score_threshold=float(os.environ.get(
                "XIAO_HEI_PERCEPTION_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD),
            )),
            sam_threshold=float(os.environ.get(
                "XIAO_HEI_PERCEPTION_SAM_THRESHOLD", str(DEFAULT_SAM_THRESHOLD),
            )),
            min_inliers=int(os.environ.get(
                "XIAO_HEI_PERCEPTION_MIN_INLIERS", str(DEFAULT_MIN_INLIERS),
            )),
            # 2, not the accumulator's own 10: the offline perception sweeps
            # (score/SAM/novelty-gate/box-estimator tuning) were all measured at
            # 2 keyframes, so the live default matches what those numbers reflect.
            scan_keyframes=int(os.environ.get("XIAO_HEI_SCAN_KEYFRAMES", "2")),
            scan_voxel_m=float(os.environ.get("XIAO_HEI_SCAN_VOXEL_M", "0.05")),
            novel_viewpoint_m=float(os.environ.get(
                "XIAO_HEI_NOVEL_VIEWPOINT_M", str(DEFAULT_NOVEL_VIEWPOINT_M),
            )),
            # Open-vocab /detect scales with the class count; with the question
            # nouns + Claude's observed-object vocab it runs ~2-3s (and grows as
            # the vocab does). The old 2.0s client default timed out on every
            # such call, so NO detections reached the scene graph. Give it real
            # headroom (env-overridable).
            request_timeout_s=float(os.environ.get(
                "XIAO_HEI_PERCEPTION_REQUEST_TIMEOUT_S", str(DEFAULT_REQUEST_TIMEOUT_S),
            )),
        )

    def as_log_config(self) -> dict[str, object]:
        """The knobs worth recording in `session.json` for *any* responder
        that builds the perception stack.

        """
        return {
            "perception_base_url": self.base_url,
            "score_threshold": self.score_threshold,
            "sam_threshold": self.sam_threshold,
            "min_inliers": self.min_inliers,
            "novel_viewpoint_m": self.novel_viewpoint_m,
            "request_timeout_s": self.request_timeout_s,
        }


def _build_perception_responder(
    scene: SceneRepresentation,
    settings: _PerceptionSettings,
    *,
    trajectory_path: Path | None = None,
    take_waypoint_reached_signals: Callable[[], int] | None = None,
    logger=None,
    dynamic_vocab: set[str] | None = None,
    class_thresholds: dict[str, float] | None = None,
):
    """Build the sidecar-backed `PerceptionResponder`.

    Blocks until the sidecar's `/healthz` is green, so this must not be called
    before the sidecar container is up.
    """
    from xiao_hei_vln.perception import PerceptionResponder
    from xiao_hei_vln.perception.client import HTTPPerceptionClient
    from xiao_hei_vln.perception.lifter import PointLifter
    from xiao_hei_vln.perception.object_map import ObjectMap
    from xiao_hei_vln.perception.scan_accumulator import ScanAccumulator
    from xiao_hei_vln.perception.vocab import Vocabulary

    client = HTTPPerceptionClient(
        base_url=settings.base_url,
        request_timeout_s=settings.request_timeout_s,
    )
    client.wait_until_ready()       # blocks until /healthz is green

    # The z-buffer occlusion gate is always on: a camera can't see through a
    # foreground object, so background returns falling inside a mask must be
    # rejected (default in PointLifter). Independent of ObjectMap fusion.
    lifter = PointLifter(min_inliers=settings.min_inliers)

    # Densify the sparse single sweep before lifting so small objects clear
    # min_inliers with genuine on-surface returns (env-tunable).
    scan_accum = ScanAccumulator(
        max_keyframes=settings.scan_keyframes,
        voxel_m=settings.scan_voxel_m,
    )

    return PerceptionResponder(
        scene,
        client=client,
        lifter=lifter,
        vocabulary=Vocabulary(dynamic=dynamic_vocab),
        score_threshold=settings.score_threshold,
        sam_threshold=settings.sam_threshold,
        novel_viewpoint_m=settings.novel_viewpoint_m,
        trajectory_path=trajectory_path,
        take_waypoint_reached_signals=take_waypoint_reached_signals,
        logger=logger,
        object_map=ObjectMap(),
        scan_accumulator=scan_accum,
        class_thresholds=class_thresholds,
    )


def _build_responder(
    name: str,
    scene: SceneRepresentation,
    *,
    take_waypoint_reached_signals: Callable[[], int] | None = None,
    dynamic_vocab: set[str] | None = None,
    class_thresholds: dict[str, float] | None = None,
):
    if name == "dummy":
        from xiao_hei_vln.dummy import DummyResponder

        return DummyResponder(), None
    if name == "perception":
        from xiao_hei_vln.logger import VLMLogger

        settings = _PerceptionSettings.from_env()
        traj_str = os.environ.get("XIAO_HEI_TRAJECTORY_JSON", "")
        traj_path = Path(traj_str) if traj_str else None

        logger = None
        log_dir = os.environ.get("XIAO_HEI_VLM_LOG_DIR", "")
        if log_dir:
            logger = VLMLogger(
                log_dir,
                config={
                    **settings.as_log_config(),
                    # Phase-B-only: only this responder walks a trajectory.
                    "trajectory_json": traj_str or None,
                },
                responder_name="perception",
                tick_hz=TICK_HZ,
            )
        responder = _build_perception_responder(
            scene,
            settings,
            trajectory_path=traj_path,
            take_waypoint_reached_signals=take_waypoint_reached_signals,
            logger=logger,
        )
        return responder, logger
    if name == "scene_gemini":
        from dataclasses import asdict

        from xiao_hei_vln.gemini import GeminiConfig, GeminiEngine
        from xiao_hei_vln.logger import VLMLogger
        from xiao_hei_vln.scene_gemini import SceneGeminiResponder

        # --- Perception sidecar → scene-graph building (used via ingest()) ---
        # No trajectory walk / no logger on the perception responder: it is
        # driven purely via ingest() during the shared exploration sweep, and
        # scene_gemini owns all logging.
        settings = _PerceptionSettings.from_env()
        perception = _build_perception_responder(scene, settings)

        # --- Gemini reasoning ------------------------------------------------
        config = GeminiConfig.from_env()
        engine = GeminiEngine(config)
        engine.warmup()

        logger = None
        log_dir = os.environ.get("XIAO_HEI_VLM_LOG_DIR", "")
        if log_dir:
            # Never persist the API key into session.json.
            safe_cfg = {k: v for k, v in asdict(config).items() if k != "api_key"}
            logger = VLMLogger(
                log_dir,
                config={**safe_cfg, **settings.as_log_config()},
                responder_name="scene_gemini",
                tick_hz=TICK_HZ,
            )
        responder = SceneGeminiResponder(
            engine, config, scene, perception=perception, logger=logger,
        )
        return responder, logger
    if name == "scene_claude":
        from dataclasses import asdict

        from xiao_hei_vln.logger import VLMLogger
        from xiao_hei_vln.nav_vlm.config import NavVLMConfig
        from xiao_hei_vln.nav_vlm.engine import AnthropicNavEngine
        from xiao_hei_vln.scene_claude import SceneClaudeResponder

        # --- Perception sidecar → scene-graph building (via ingest()) --------
        # Driven by the app tick loop's exploration branch while the
        # question-directed nav_task1 explorer approaches the target object.
        settings = _PerceptionSettings.from_env()
        perception = _build_perception_responder(
            scene, settings, dynamic_vocab=dynamic_vocab,
            class_thresholds=class_thresholds,
        )

        # --- Claude answering (needs the Anthropic key) ----------------------
        cfg = NavVLMConfig.from_env()   # raises if no key — the answerer needs it
        engine = AnthropicNavEngine(cfg)  # generic call_tool used for answering

        logger = None
        log_dir = os.environ.get("XIAO_HEI_VLM_LOG_DIR", "")
        if log_dir:
            # Never persist the API key into session.json.
            safe_cfg = {k: v for k, v in asdict(cfg).items() if k != "api_key"}
            logger = VLMLogger(
                log_dir,
                config={**safe_cfg, **settings.as_log_config()},
                responder_name="scene_claude",
                tick_hz=TICK_HZ,
            )
        responder = SceneClaudeResponder(
            engine, cfg, scene, perception=perception, logger=logger,
        )
        return responder, logger
    raise ValueError(
        f"Unknown XIAO_HEI_RESPONDER={name!r}; "
        "expected one of: dummy, perception, scene_gemini, scene_claude",
    )


def _build_explorer(node, scene: SceneRepresentation, dynamic_vocab: set[str] | None = None,
                    class_thresholds: dict[str, float] | None = None):
    """Instantiate the configured exploration strategy, or None if disabled.

    Select the strategy with XIAO_HEI_EXPLORATION_STRATEGY (default: frontier).
    Add new strategies here as additional elif branches. ``scene`` is the shared
    scene graph, needed by question-directed strategies (nav_task1) that reason
    over the objects detected so far.
    """
    if _EXPLORATION_MAX_WAYPOINTS <= 0:
        return None

    if _EXPLORATION_STRATEGY == "frontier":
        from xiao_hei_vln.exploration import FrontierExplorer
        explorer = FrontierExplorer(
            max_waypoints=_EXPLORATION_MAX_WAYPOINTS,
            waypoint_reach_dist=0.3,
            max_waypoint_dist=_EXPLORATION_MAX_WAYPOINT_DIST,
            stuck_timeout_s=12.0,
            max_consecutive_skips=20,
        )
    elif _EXPLORATION_STRATEGY == "nbv":
        from xiao_hei_vln.exploration import NextBestViewExplorer
        explorer = NextBestViewExplorer(
            max_waypoints=_EXPLORATION_MAX_WAYPOINTS,
            waypoint_reach_dist=0.3,
        )
    elif _EXPLORATION_STRATEGY == "nav_vlm":
        # VLM waypoint proposer (Opus 5). Fails soft: a missing Anthropic key
        # disables exploration rather than crashing the whole node at boot.
        from xiao_hei_vln.exploration import NavVLMExplorer
        from xiao_hei_vln.nav_vlm import AnthropicNavEngine, NavVLMConfig

        try:
            cfg = NavVLMConfig.from_env()
        except ValueError as exc:
            node.get_logger().error(
                f"nav_vlm strategy selected but {exc} — disabling exploration."
            )
            return None
        explorer = NavVLMExplorer(
            AnthropicNavEngine(cfg),
            config=cfg,
            max_waypoints=_EXPLORATION_MAX_WAYPOINTS,
            waypoint_reach_dist=0.3,
            max_hop_m=_EXPLORATION_MAX_WAYPOINT_DIST,
        )
    elif _EXPLORATION_STRATEGY == "nav_task1":
        # Question-directed VLM navigation: drive to the object named in an
        # OBJECT_REFERENCE question, then hand over to the scene_claude
        # responder to answer. Fails soft on a missing key like nav_vlm.
        from xiao_hei_vln.exploration import NavTask1Explorer
        from xiao_hei_vln.nav_vlm import AnthropicNavEngine, NavVLMConfig
        from xiao_hei_vln.nav_vlm.task1_prompts import (
            NAV_SYSTEM_PROMPT,
            PROPOSE_OR_ARRIVE_TOOL,
        )

        try:
            cfg = NavVLMConfig.from_env()
        except ValueError as exc:
            node.get_logger().error(
                f"nav_task1 strategy selected but {exc} — disabling exploration."
            )
            return None
        engine = AnthropicNavEngine(
            cfg, system_prompt=NAV_SYSTEM_PROMPT, tool=PROPOSE_OR_ARRIVE_TOOL,
        )
        # The 10-min budget is TOTAL (navigation + answering). Navigation must
        # therefore stop early enough that the final Claude answer still lands
        # inside the budget, so the nav deadline = total - answer reserve.
        _t1_total_s = float(os.environ.get("XIAO_HEI_NAV_TASK1_MAX_QUESTION_S") or "600")
        _t1_answer_reserve_s = float(
            os.environ.get("XIAO_HEI_NAV_TASK1_ANSWER_RESERVE_S") or "60"
        )
        explorer = NavTask1Explorer(
            engine,
            scene=scene,
            config=cfg,
            max_waypoints=_EXPLORATION_MAX_WAYPOINTS,
            waypoint_reach_dist=0.3,
            max_hop_m=_EXPLORATION_MAX_WAYPOINT_DIST,
            # Skip-cap termination is disabled: the reach/explore mode machine
            # falls back to exploring when reach stalls, and the per-question
            # time budget is the only hard stop.
            max_consecutive_skips=1_000_000,
            max_question_seconds=max(30.0, _t1_total_s - _t1_answer_reserve_s),
            # Arrive once the scene graph stops gaining new views for this long
            # (it has seen every reachable angle) instead of running to the cap.
            coverage_plateau_s=float(
                os.environ.get("XIAO_HEI_NAV_TASK1_COVERAGE_PLATEAU_S") or "45"
            ),
            dynamic_vocab=dynamic_vocab,
            class_thresholds=class_thresholds,
            default_score_threshold=float(
                os.environ.get("XIAO_HEI_PERCEPTION_SCORE_THRESHOLD") or "0.4"
            ),
            # Ablation B: "0"/"false" disables the navigator's in-loop control
            # of perception (verify_objects + visible-object vocab priming).
            perception_steer=(
                os.environ.get("XIAO_HEI_NAV_PERCEPTION_STEER", "1").strip().lower()
                not in ("0", "false", "no")
            ),
            # Ablation A ("truly naive") re-plan cadence, seconds. Only used when
            # the engine's raw_reasoning flag is on; ignored otherwise.
            naive_replan_s=float(
                os.environ.get("XIAO_HEI_NAV_NAIVE_REPLAN_S") or "5.0"
            ),
        )
    else:
        node.get_logger().error(
            f"Unknown exploration strategy {_EXPLORATION_STRATEGY!r} — disabling exploration."
        )
        return None

    node.get_logger().info(
        f"Exploration enabled: {type(explorer).__name__} "
        f"(strategy={_EXPLORATION_STRATEGY}, max_waypoints={_EXPLORATION_MAX_WAYPOINTS}, "
        f"reach_dist=0.3m, max_waypoint_dist={_EXPLORATION_MAX_WAYPOINT_DIST}m)"
    )
    return explorer


def _final_map_stats(explorer, pose) -> dict:
    """Grid counters for a DONE line, best-effort.

    Every caller is on a termination path, so a failure here must never be what
    stops the run from recording why it ended.
    """
    try:
        if pose is None:
            return explorer.get_grid().stats()
        return explorer.get_grid().stats(pose.position.x, pose.position.y)
    except Exception as exc:
        return {"map_stats_error": type(exc).__name__}


def _budget_expired(clock_start: float | None, now_s: float, budget_s: float) -> bool:
    """Has the sweep burned its wall-clock budget?

    ``clock_start`` is None until the first tick with a pose, so the 90-190 s
    wait for /state_estimation costs nothing. ``budget_s <= 0`` disables it.
    """
    if budget_s <= 0 or clock_start is None:
        return False
    return now_s - clock_start >= budget_s


def _publish_atomically(out_path: Path, write) -> None:
    """Run ``write(tmp)`` then rename tmp over ``out_path``.

    Snapshots are read mid-run (an `scp`, an auto-reloading viewer), so the
    file must never be observed half-written. The temp name keeps the ``.png``
    suffix — matplotlib and pillow both infer the format from it.
    """
    tmp = out_path.with_name(f".{out_path.stem}.partial{out_path.suffix}")
    write(tmp)
    tmp.replace(out_path)


def _maybe_save_png(explorer, node, *, quiet: bool = False) -> None:
    """Save the debug PNG if XIAO_HEI_EXPLORATION_LOG_DIR is configured.

    Skipped silently when the active strategy does not expose get_grid() /
    get_visited_waypoints() (not all algorithms maintain an OccupancyGrid).

    ``quiet`` demotes the routine log lines to debug — set it for the periodic
    snapshots, which would otherwise narrate every interval.
    """
    if not _EXPLORATION_LOG_DIR:
        return
    if not (hasattr(explorer, "get_visited_waypoints") and hasattr(explorer, "get_grid")):
        if not quiet:
            node.get_logger().info("Exploration plot skipped: strategy does not support it.")
        return
    try:
        from xiao_hei_vln.exploration import save_exploration_plot

        out_dir = _exploration_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "exploration.png"
        _publish_atomically(
            out_path,
            lambda tmp: save_exploration_plot(
                explorer.get_visited_waypoints(),
                explorer.get_grid(),
                tmp,
            ),
        )
        log = node.get_logger().debug if quiet else node.get_logger().info
        log(f"Exploration plot saved to {out_path}")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().warn(f"Could not save exploration plot: {exc}")


def _maybe_save_rviz(node, *, quiet: bool = False) -> None:
    """Screenshot the sim's RViz window if a display is available.

    Gated on DISPLAY, so a headless run (the challenge submission, CI) skips
    it without complaint rather than failing.  Everything is best-effort: a
    missing X server, a missing python-xlib, or an RViz that never opened
    must not take the node down — a lost debug image is not worth a crash.

    ``quiet`` demotes the routine log lines to debug (see `_maybe_save_png`).
    A *failed* grab still warns either way — that is how you notice mid-run
    that snapshots are not landing.
    """
    if not _EXPLORATION_LOG_DIR:
        return
    if not os.environ.get("DISPLAY"):
        if not quiet:
            node.get_logger().info("RViz screenshot skipped: no DISPLAY set.")
        return
    try:
        from xiao_hei_vln.exploration import save_rviz_screenshot

        out_path = _exploration_dir() / "rviz.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _publish_atomically(out_path, save_rviz_screenshot)
        log = node.get_logger().debug if quiet else node.get_logger().info
        log(f"RViz screenshot saved to {out_path}")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().warn(f"Could not save RViz screenshot: {exc}")


def main() -> None:
    # Local imports so the rest of the package stays importable without rclpy.
    import rclpy
    from rclpy.node import Node

    from xiao_hei_vln.adapters.ros.publishers import VLMOutputPublisher
    from xiao_hei_vln.adapters.ros.subscribers import bind_subscribers

    rclpy.init()
    node_name = {
        "perception": "xiao_hei_perception_vlm",
        "scene_gemini": "xiao_hei_scene_gemini_vlm",
        "scene_claude": "xiao_hei_scene_claude_vlm",
    }.get(RESPONDER_NAME, "xiao_hei_dummy_vlm")
    node: Node = rclpy.create_node(node_name)

    cache = LatestCache()
    subs = bind_subscribers(node, cache)
    publisher = VLMOutputPublisher(node)

    # /way_point_reached is the autonomy stack's signal that the
    # *adjusted* waypoint (its safe approximation of our commanded
    # waypoint) has been reached. We accumulate signals into a
    # shared mutable counter; the responder polls it once per tick
    # via `take_waypoint_reached_signals` and advances Phase A on
    # each new signal. This is the sole advance trigger — distance
    # to our commanded waypoint can't be used as a fallback because
    # the autonomy stack often steers to a safer nearby point and
    # never actually reaches our literal commanded XY.
    from std_msgs.msg import Float32

    waypoint_reached_count = [0]

    def _on_waypoint_reached(_msg) -> None:
        waypoint_reached_count[0] += 1

    node.create_subscription(
        Float32, "/way_point_reached", _on_waypoint_reached, 10,
    )

    def take_waypoint_reached_signals() -> int:
        n = waypoint_reached_count[0]
        waypoint_reached_count[0] = 0
        return n

    # Scene memory persists across responder.reset() (which fires per question)
    # because the underlying Unity scene is the same for every question in a
    # session. Built before the responder so a scene-aware responder (e.g.
    # PerceptionResponder) can take the same reference at construction time.
    scene = SceneRepresentation()
    # Shared VLM-observed detector vocabulary: the nav_task1 navigator adds the
    # objects Claude sees in the panorama (+ synonyms) and the perception
    # Vocabulary reads them, so rare objects the fixed prior misses get detected.
    dynamic_vocab: set[str] = set()
    # Shared per-class detector thresholds: the navigator lowers a class's floor
    # when it sees an object perception missed (verify_objects), for that class
    # only. Empty = normal single-threshold behaviour.
    class_thresholds: dict[str, float] = {}
    responder, logger = _build_responder(
        RESPONDER_NAME, scene,
        take_waypoint_reached_signals=take_waypoint_reached_signals,
        dynamic_vocab=dynamic_vocab,
        class_thresholds=class_thresholds,
    )
    if logger is not None:
        logger.attach_scene(scene)

    # Live rviz view of the fused scene graph (3D boxes + labels on
    # /perception/objects), so the perception map can be watched while
    # driving. Perception-backed responders only — the scene is empty
    # otherwise. Opt out with XIAO_HEI_PUBLISH_MARKERS=0.
    if RESPONDER_NAME in ("perception", "scene_gemini", "scene_claude") and os.environ.get(
        "XIAO_HEI_PUBLISH_MARKERS", "1",
    ).lower() not in ("0", "false", "no", "off"):
        from xiao_hei_vln.app.scene_markers import ScenePublisher, Scoreboard

        _scene_pub = ScenePublisher(node)
        # Optional dev scoreboard: live metrics vs the scene's object_list.txt
        # (GT is not available at test time). Enabled by mounting the GT file
        # and pointing XIAO_HEI_GT_OBJECT_LIST at it.
        _scoreboard = None
        _gt_path = os.environ.get("XIAO_HEI_GT_OBJECT_LIST", "")
        if _gt_path and os.path.exists(_gt_path):
            try:
                _scoreboard = Scoreboard(_gt_path)
                node.get_logger().info(f"Scoreboard enabled (GT: {_gt_path})")
            except Exception:  # noqa: BLE001 — never let the scoreboard break bringup
                node.get_logger().exception("Scoreboard init failed; disabling")

        def _publish_markers() -> None:
            snap = scene.to_dict()
            text = None
            if _scoreboard is not None:
                try:
                    text = _scoreboard.text(snap)
                except Exception:  # noqa: BLE001
                    text = None
            _scene_pub.publish(snap, scoreboard_text=text)

        node.create_timer(0.5, _publish_markers)

    # Periodic full scene-graph dump to disk. A pure exploration run has no
    # per-question VLM session log, so without this it would leave no scene
    # graph for offline dataset building (xiao_hei_vln.detected_dataset).
    # Opt in with XIAO_HEI_SCENE_DUMP_PATH; atomic write, last dump = fullest map.
    _scene_dump_path = os.environ.get("XIAO_HEI_SCENE_DUMP_PATH", "")
    if _scene_dump_path:
        _dump_target = Path(_scene_dump_path)
        _dump_target.parent.mkdir(parents=True, exist_ok=True)

        def _dump_scene() -> None:
            try:
                tmp = _dump_target.with_suffix(_dump_target.suffix + ".tmp")
                tmp.write_text(json.dumps(scene.to_dict()))
                tmp.replace(_dump_target)
            except Exception:  # noqa: BLE001 — never let the dump break the run
                node.get_logger().exception("scene dump failed")

        node.create_timer(2.0, _dump_scene)

    explorer = _build_explorer(node, scene, dynamic_vocab=dynamic_vocab,
                               class_thresholds=class_thresholds)

    # Structured exploration log — survives the container via the mounted volume.
    # Create the dir if it doesn't exist so a run without a bind-mounted
    # /exploration_logs (e.g. the non-GPU compose.yml) doesn't crash on startup.
    _log_dir = _exploration_dir()
    _log_dir.mkdir(parents=True, exist_ok=True)
    _exp_log_file = open(_log_dir / "exploration.log", "w", buffering=1)

    def _exp_log(event: str, **fields) -> None:
        now_s = node.get_clock().now().nanoseconds / 1e9
        parts = "  ".join(f"{k}={v}" for k, v in fields.items())
        _exp_log_file.write(f"[{now_s:.3f}] {event}  {parts}\n")

    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32

    _nav_qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )

    # Track nav stack's distance to current waypoint; best = closest approach this target.
    # best_odom / last_progress_time mirror that in odometry, so a silent
    # /way_point_reached cannot masquerade as "the robot has stopped".
    _wp_reached_state = {"value": float("inf"), "close_ticks": 0, "best": float("inf"),
                         "settled_ticks": 0, "prev_best": float("inf"),
                         "best_odom": float("inf"), "last_progress_time": None}
    _WP_REACHED_THRESHOLD = 0.92  # nav stack settles between 0.25-0.90m depending on obstacles
    _WP_ODOM_PROGRESS_M = 0.10    # closing this much on the target counts as progress
    _WP_ODOM_STALL_S = 8.0        # no progress for this long before an early skip is allowed

    def _on_wp_reached(msg) -> None:
        v = float(msg.data)
        _wp_reached_state["value"] = v
        if v < _wp_reached_state["best"]:
            _wp_reached_state["best"] = v

    node.create_subscription(Float32, "/way_point_reached", _on_wp_reached, _nav_qos)

    state = {
        "tick_id": 0,
        "last_question_text": None,
        "exploration_started": False,
        "last_exploration_wp": None,
        "wp_start_time": None,
        # Wall-clock cutoff. The strategy is never told it expired —
        # is_complete() stays False — so the flag is what ends the sweep.
        "exploration_clock_start": None,
        "exploration_timed_out": False,
        # True path length, accumulated every tick. Sampling pose only at
        # WP_SET/WP_SKIP undercounts badly on the scenes that drive the most.
        "path_len_m": 0.0,
        "last_pose": None,
        "last_map_log": None,
        "last_no_target_log": None,
        "last_no_target_reason": None,
        # Set when the supervisor forces the skip, so WP_SKIP can name which of
        # the two skip paths fired instead of us inferring it from `elapsed`.
        "forced_skip": False,
    }

    def _exploration_over() -> bool:
        """True once the sweep has ended, for either reason."""
        return explorer is None or explorer.is_complete() or state["exploration_timed_out"]

    if _EXPLORATION_SNAPSHOT_S > 0 and explorer is not None:

        def _snapshot_exploration() -> None:
            # rclpy's default executor is single-threaded, so this never
            # interleaves with tick() — the grid is read between ticks.
            if _exploration_over():
                return  # the end-of-sweep save already wrote the fullest pair
            _maybe_save_png(explorer, node, quiet=True)
            _maybe_save_rviz(node, quiet=True)

        node.create_timer(_EXPLORATION_SNAPSHOT_S, _snapshot_exploration)
        node.get_logger().info(
            f"Exploration snapshots every {_EXPLORATION_SNAPSHOT_S:.0f}s → {_exploration_dir()}"
        )

    def tick() -> None:
        from xiao_hei_vln.messages.outputs import WaypointPathResponse

        now = node.get_clock().now().to_msg()
        snapshot = cache.snapshot(state["tick_id"], Stamp(sec=now.sec, nanosec=now.nanosec))
        state["tick_id"] += 1

        # Exploration phase: runs until the strategy completes or the wall-clock
        # budget expires. A question arriving mid-exploration does NOT interrupt
        # it — exploration keeps going (still building the scene) and the answer
        # is deferred to the responder block below once the sweep is over.
        if not _exploration_over():
            # Build the scene graph on the fly *while* exploring. scene.update()
            # maintains viewpoint/bounds nodes; responder.ingest() runs the
            # perception detect→lift→fuse cycle without ever emitting an
            # answer (so a pending question stays deferred). Responders without
            # a scene path (dummy) simply don't expose ingest().
            scene.update(snapshot)
            if hasattr(responder, "ingest"):
                responder.ingest(snapshot)

            if not state["exploration_started"]:
                node.get_logger().info("Exploration started.")
                state["exploration_started"] = True
                _exp_log("START",
                         max_waypoints=explorer._max_waypoints,
                         max_seconds=(_EXPLORATION_MAX_SECONDS or "off"),
                         threshold=_WP_REACHED_THRESHOLD,
                         stuck_timeout=f"{explorer._stuck_timeout_s}s",
                         max_skips=explorer._max_consecutive_skips)

            now_s = node.get_clock().now().nanoseconds / 1e9
            pose = snapshot.pose
            robot_pos = (
                f"({pose.position.x:.2f},{pose.position.y:.2f})" if pose is not None else "unknown"
            )

            # Clock starts at the first pose, not at START: the node ticks for
            # 90-190 s before /state_estimation arrives and cannot explore yet.
            if (
                _EXPLORATION_MAX_SECONDS > 0
                and state["exploration_clock_start"] is None
                and pose is not None
            ):
                state["exploration_clock_start"] = now_s
                _exp_log("CLOCK_START",
                         budget=f"{_EXPLORATION_MAX_SECONDS:.0f}s",
                         robot=robot_pos)

            if pose is not None:
                here = (pose.position.x, pose.position.y)
                if state["last_pose"] is not None:
                    step = math.hypot(here[0] - state["last_pose"][0],
                                      here[1] - state["last_pose"][1])
                    # Ignore localisation jumps — a re-lookup can teleport the
                    # pose metres in one tick and inflate the total.
                    if step < 1.0:
                        state["path_len_m"] += step
                state["last_pose"] = here

            # MAP heartbeat: the coverage curve, plus the map-health counters
            # that show a strategy walling itself in (reachable stops tracking
            # free) before it manifests as a run of unexplained skips.
            if (
                _EXPLORATION_MAP_LOG_S > 0
                and pose is not None
                and (
                    state["last_map_log"] is None
                    or now_s - state["last_map_log"] >= _EXPLORATION_MAP_LOG_S
                )
            ):
                state["last_map_log"] = now_s
                try:
                    grid_stats = explorer.get_grid().stats(pose.position.x, pose.position.y)
                except Exception as exc:  # a logging path must never kill the sweep
                    grid_stats = {"error": type(exc).__name__}
                started = state["exploration_clock_start"]
                _exp_log("MAP",
                         elapsed=f"{now_s - started:.0f}s" if started else "?",
                         robot=robot_pos,
                         path_m=f"{state['path_len_m']:.1f}",
                         visited=len(explorer._visited),
                         skipped=explorer.skipped_count,
                         consecutive=explorer._consecutive_skip_count,
                         hatch_resets=getattr(explorer, "skip_hatch_resets", 0),
                         **grid_stats)

            clock_start = state["exploration_clock_start"]
            if _budget_expired(clock_start, now_s, _EXPLORATION_MAX_SECONDS):
                # Same exit as the strategy's own stop conditions — DONE, then
                # the final images — so log watchers (vla3d_eval_sim.sh greps
                # " DONE ") see one termination event whatever the reason.
                state["exploration_timed_out"] = True
                elapsed = now_s - clock_start
                _exp_log("DONE",
                         visited=len(explorer._visited),
                         skipped=explorer.skipped_count,
                         reason="time_limit",
                         elapsed=f"{elapsed:.1f}s",
                         budget=f"{_EXPLORATION_MAX_SECONDS:.0f}s",
                         path_m=f"{state['path_len_m']:.1f}",
                         hatch_resets=getattr(explorer, "skip_hatch_resets", 0),
                         robot=robot_pos,
                         **_final_map_stats(explorer, pose))
                node.get_logger().info(
                    f"Exploration time limit reached ({elapsed:.1f}s of "
                    f"{_EXPLORATION_MAX_SECONDS:.0f}s): visited={len(explorer._visited)} "
                    f"skipped={explorer.skipped_count} — handing over to the responder"
                )
                _maybe_save_png(explorer, node)
                _maybe_save_rviz(node)
                return

            # Advance when the nav stack signals it reached the waypoint:
            # /way_point_reached fires (a small distance) only on an actual
            # reach, so value < threshold for 3 consecutive ticks IS the reach.
            # When it never fires the waypoint was not reached — the early-skip
            # below handles that, so the model re-routes around the obstacle.
            if explorer._current_target is not None:
                if _wp_reached_state["value"] < _WP_REACHED_THRESHOLD:
                    _wp_reached_state["close_ticks"] += 1
                    if _wp_reached_state["close_ticks"] >= 3:
                        best = _wp_reached_state["best"]
                        _exp_log("WP_ADVANCE",
                                 target=f"({explorer._current_target.x:.2f},{explorer._current_target.y:.2f})",
                                 nav_dist=f"{best:.2f}",
                                 visited=len(explorer._visited) + 1)
                        node.get_logger().info(
                            f"Nav stack settled at {best:.2f}m "
                            f"— advancing waypoint (visited={len(explorer._visited) + 1})"
                        )
                        explorer.advance()
                        _wp_reached_state["close_ticks"] = 0
                        _wp_reached_state["value"] = float("inf")
                        _wp_reached_state["best"] = float("inf")
                        _wp_reached_state["best_odom"] = float("inf")
                        _wp_reached_state["last_progress_time"] = None
                        state["last_exploration_wp"] = None  # force WP_SET for next target
                else:
                    _wp_reached_state["close_ticks"] = 0

            prev_skipped = explorer.skipped_count
            prev_visited = len(explorer._visited)

            # Odometry progress towards the target — the second opinion the
            # early-skip needs. best_nav_dist stays inf whenever /way_point_reached
            # is silent, and inf >= inf - 0.02 held, so a silent nav stack used to
            # look "settled" and the waypoint was dropped ~6 s in. 65% of frontier's
            # skips fired while the robot was still driving (median 2.4 m covered).
            if pose is not None and explorer._current_target is not None:
                odom_dist = math.hypot(
                    explorer._current_target.x - pose.position.x,
                    explorer._current_target.y - pose.position.y,
                )
                if odom_dist < _wp_reached_state["best_odom"] - _WP_ODOM_PROGRESS_M:
                    _wp_reached_state["best_odom"] = odom_dist
                    _wp_reached_state["last_progress_time"] = now_s
                elif _wp_reached_state["last_progress_time"] is None:
                    _wp_reached_state["best_odom"] = min(
                        odom_dist, _wp_reached_state["best_odom"]
                    )
                    _wp_reached_state["last_progress_time"] = now_s

            # Early skip: nav stack settled above threshold with no improvement for 16 ticks (8s).
            # 8s minimum delay gives the nav stack time to respond before we start counting.
            since_progress = (
                now_s - _wp_reached_state["last_progress_time"]
                if _wp_reached_state["last_progress_time"] is not None
                else 0.0
            )
            if (
                explorer._current_target is not None
                and _wp_reached_state["best"] > _WP_REACHED_THRESHOLD
                and state["wp_start_time"] is not None
                and now_s - state["wp_start_time"] > 8.0
                # Never abandon a waypoint the robot is visibly closing on.
                and since_progress > _WP_ODOM_STALL_S
            ):
                if _wp_reached_state["best"] >= _wp_reached_state["prev_best"] - 0.02:
                    _wp_reached_state["settled_ticks"] += 1
                else:
                    _wp_reached_state["settled_ticks"] = 0
                _wp_reached_state["prev_best"] = _wp_reached_state["best"]
                if _wp_reached_state["settled_ticks"] >= 16:
                    state["forced_skip"] = True
                    explorer.force_skip()
                    _wp_reached_state["settled_ticks"] = 0
                    _wp_reached_state["prev_best"] = float("inf")

            wp = explorer.update(snapshot)

            # Log odometry-reach advances (update() clears the target internally — no nav event fired).
            if len(explorer._visited) > prev_visited and explorer.skipped_count == prev_skipped:
                _exp_log("WP_ADVANCE",
                         target=f"({explorer._visited[-1].x:.2f},{explorer._visited[-1].y:.2f})",
                         nav_dist="odom",
                         visited=len(explorer._visited))

            if explorer.skipped_count > prev_skipped:
                elapsed = round(now_s - state["wp_start_time"], 1) if state["wp_start_time"] else "?"
                last_wp = state["last_exploration_wp"]
                skip_target = f"({last_wp[0]:.2f},{last_wp[1]:.2f})" if last_wp else "unknown"
                _exp_log("WP_SKIP",
                         target=skip_target,
                         robot=robot_pos,
                         # no_progress = the supervisor's early skip;
                         # stuck_timeout = the strategy's own clock ran out.
                         kind="no_progress" if state["forced_skip"] else "stuck_timeout",
                         elapsed=f"{elapsed}s",
                         best_nav_dist=f"{_wp_reached_state['best']:.2f}",
                         last_nav_dist=f"{_wp_reached_state['value']:.2f}",
                         best_odom_dist=f"{_wp_reached_state['best_odom']:.2f}",
                         path_m=f"{state['path_len_m']:.1f}",
                         consecutive=explorer._consecutive_skip_count)
                node.get_logger().info(
                    f"Exploration SKIP: target={skip_target}  "
                    f"best_nav_dist={_wp_reached_state['best']:.2f}m  "
                    f"consecutive={explorer._consecutive_skip_count}"
                )
                state["last_exploration_wp"] = None
                state["forced_skip"] = False
                _wp_reached_state["best"] = float("inf")
                _wp_reached_state["best_odom"] = float("inf")
                _wp_reached_state["last_progress_time"] = None

            # A tick that wanted a target and got none. Logged every time, not
            # only at DONE: `no_frontiers` used to appear with no trace of what
            # the selector had actually been looking at.
            # Throttled to one line per reason-change or 5 s, so a long wait for
            # terrain at startup doesn't bury the run in identical lines.
            if wp is None and not explorer.is_complete() and pose is not None:
                diag = (
                    explorer.selection_diagnostics()
                    if hasattr(explorer, "selection_diagnostics")
                    else {}
                )
                why = diag.get("why")
                last_t = state["last_no_target_log"]
                if why != state["last_no_target_reason"] or last_t is None or now_s - last_t >= 5.0:
                    state["last_no_target_reason"] = why
                    state["last_no_target_log"] = now_s
                    _exp_log("NO_TARGET",
                             robot=robot_pos,
                             empty_ticks=getattr(explorer, "_empty_selection_ticks", "n/a"),
                             **diag)

            if wp is not None:
                wp_key = (round(wp.x, 2), round(wp.y, 2))
                if wp_key != state["last_exploration_wp"]:
                    dist_to_wp = math.hypot(
                        wp.x - (pose.position.x if pose else 0.0),
                        wp.y - (pose.position.y if pose else 0.0),
                    )
                    diag = (
                        explorer.selection_diagnostics()
                        if hasattr(explorer, "selection_diagnostics")
                        else {}
                    )
                    _exp_log("WP_SET",
                             target=f"({wp.x:.2f},{wp.y:.2f})",
                             robot=robot_pos,
                             dist=f"{dist_to_wp:.2f}",
                             why=(getattr(explorer, "_last_rationale", "") or "")[:160],
                             **diag)
                    state["last_exploration_wp"] = wp_key
                    state["wp_start_time"] = now_s
                    _wp_reached_state["best"] = float("inf")
                    _wp_reached_state["value"] = float("inf")
                    _wp_reached_state["close_ticks"] = 0
                    _wp_reached_state["settled_ticks"] = 0
                    _wp_reached_state["prev_best"] = float("inf")
                    _wp_reached_state["best_odom"] = float("inf")
                    _wp_reached_state["last_progress_time"] = None
                publisher.publish(WaypointPathResponse(waypoints=[wp]))

            if explorer.is_complete():
                if len(explorer._visited) >= explorer._max_waypoints:
                    reason = "budget_exhausted"
                elif explorer._consecutive_skip_count >= explorer._max_consecutive_skips:
                    reason = "max_consecutive_skips"
                else:
                    reason = "no_frontiers"
                started = state["exploration_clock_start"]
                _exp_log("DONE",
                         visited=len(explorer._visited),
                         skipped=explorer.skipped_count,
                         reason=reason,
                         elapsed=f"{now_s - started:.1f}s" if started else "?",
                         path_m=f"{state['path_len_m']:.1f}",
                         hatch_resets=getattr(explorer, "skip_hatch_resets", 0),
                         # `no_frontiers` covers several very different states;
                         # the selector's own last word disambiguates them.
                         select_why=(
                             explorer.selection_diagnostics().get("why")
                             if hasattr(explorer, "selection_diagnostics")
                             else "n/a"
                         ),
                         **_final_map_stats(explorer, pose))
                node.get_logger().info(
                    f"Exploration complete: visited={len(explorer._visited)} "
                    f"skipped={explorer.skipped_count} reason={reason}"
                )
                _maybe_save_png(explorer, node)
                _maybe_save_rviz(node)
            return

        # New question → reset the responder so it handles it from scratch.
        if (
            snapshot.question is not None
            and snapshot.question.text != state["last_question_text"]
        ):
            responder.reset()
            state["last_question_text"] = snapshot.question.text
            node.get_logger().info(f"Received question: {snapshot.question.text!r}")

        scene.update(snapshot)
        out = responder.respond(snapshot)
        if out is not None:
            publisher.publish(out)

        if responder.is_done():
            # The responder must always produce a non-None final answer,
            # even on timeout — the guard here is defensive, not expected.
            if logger is not None and out is not None and snapshot.question is not None:
                logger.write_prediction(snapshot.question.text, out)
            node.get_logger().info("Response complete; awaiting next question.")
            cache.clear_question()
            state["last_question_text"] = None
            responder.reset()

    period_s = 1.0 / TICK_HZ
    node.create_timer(period_s, tick)
    node.get_logger().info(
        f"{node_name} ready (responder={RESPONDER_NAME}, "
        f"tick = {TICK_HZ:.2f} Hz, {len(subs)} subscribers)",
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        responder.close()
        _exp_log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
