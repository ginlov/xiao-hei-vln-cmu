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

# 1 Hz default (was 2 Hz): the OWLv2 + SAM-large stack needs ~0.5–0.8 s per
# tick, so a 1 s period keeps ticks from overlapping. Override with
# XIAO_HEI_VLM_TICK_HZ.
TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "1.0"))
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
    min_inliers: int
    scan_keyframes: int
    scan_voxel_m: float

    @classmethod
    def from_env(cls) -> _PerceptionSettings:
        # Imported lazily: these modules pull in httpx/pycocotools (the
        # `perception` extra), and the package must stay importable without it.
        from xiao_hei_vln.perception.client import DEFAULT_BASE_URL
        from xiao_hei_vln.perception.lifter import DEFAULT_MIN_INLIERS
        from xiao_hei_vln.perception.responder import DEFAULT_SCORE_THRESHOLD

        return cls(
            base_url=os.environ.get("XIAO_HEI_PERCEPTION_BASE_URL", DEFAULT_BASE_URL),
            score_threshold=float(os.environ.get(
                "XIAO_HEI_PERCEPTION_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD),
            )),
            min_inliers=int(os.environ.get(
                "XIAO_HEI_PERCEPTION_MIN_INLIERS", str(DEFAULT_MIN_INLIERS),
            )),
            scan_keyframes=int(os.environ.get("XIAO_HEI_SCAN_KEYFRAMES", "10")),
            scan_voxel_m=float(os.environ.get("XIAO_HEI_SCAN_VOXEL_M", "0.05")),
        )

    def as_log_config(self) -> dict[str, object]:
        """The knobs worth recording in `session.json` for *any* responder
        that builds the perception stack.

        """
        return {
            "perception_base_url": self.base_url,
            "score_threshold": self.score_threshold,
            "min_inliers": self.min_inliers,
        }


def _build_perception_responder(
    scene: SceneRepresentation,
    settings: _PerceptionSettings,
    *,
    trajectory_path: Path | None = None,
    take_waypoint_reached_signals: Callable[[], int] | None = None,
    logger=None,
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

    client = HTTPPerceptionClient(base_url=settings.base_url)
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
        vocabulary=Vocabulary(),
        score_threshold=settings.score_threshold,
        trajectory_path=trajectory_path,
        take_waypoint_reached_signals=take_waypoint_reached_signals,
        logger=logger,
        object_map=ObjectMap(),
        scan_accumulator=scan_accum,
    )


def _build_responder(
    name: str,
    scene: SceneRepresentation,
    *,
    take_waypoint_reached_signals: Callable[[], int] | None = None,
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
    raise ValueError(
        f"Unknown XIAO_HEI_RESPONDER={name!r}; "
        "expected one of: dummy, perception, scene_gemini",
    )


def _build_explorer(node):
    """Instantiate the configured exploration strategy, or None if disabled.

    Select the strategy with XIAO_HEI_EXPLORATION_STRATEGY (default: frontier).
    Add new strategies here as additional elif branches.
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
    responder, logger = _build_responder(
        RESPONDER_NAME, scene,
        take_waypoint_reached_signals=take_waypoint_reached_signals,
    )
    if logger is not None:
        logger.attach_scene(scene)

    # Live rviz view of the fused scene graph (3D boxes + labels on
    # /perception/objects), so the perception map can be watched while
    # driving. Perception-backed responders only — the scene is empty
    # otherwise. Opt out with XIAO_HEI_PUBLISH_MARKERS=0.
    if RESPONDER_NAME in ("perception", "scene_gemini") and os.environ.get(
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

    explorer = _build_explorer(node)

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
    _wp_reached_state = {"value": float("inf"), "close_ticks": 0, "best": float("inf"),
                         "settled_ticks": 0, "prev_best": float("inf")}
    _WP_REACHED_THRESHOLD = 0.92  # nav stack settles between 0.25-0.90m depending on obstacles

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
                         robot=robot_pos)
                node.get_logger().info(
                    f"Exploration time limit reached ({elapsed:.1f}s of "
                    f"{_EXPLORATION_MAX_SECONDS:.0f}s): visited={len(explorer._visited)} "
                    f"skipped={explorer.skipped_count} — handing over to the responder"
                )
                _maybe_save_png(explorer, node)
                _maybe_save_rviz(node)
                return

            # Advance when nav stack has settled within threshold for 3 consecutive ticks.
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
                        state["last_exploration_wp"] = None  # force WP_SET for next target
                else:
                    _wp_reached_state["close_ticks"] = 0

            prev_skipped = explorer.skipped_count
            prev_visited = len(explorer._visited)

            # Early skip: nav stack settled above threshold with no improvement for 5 ticks (2.5s).
            # 4s minimum delay gives the nav stack time to respond before we start counting.
            if (
                explorer._current_target is not None
                and _wp_reached_state["best"] > _WP_REACHED_THRESHOLD
                and state["wp_start_time"] is not None
                and now_s - state["wp_start_time"] > 4.0
            ):
                if _wp_reached_state["best"] >= _wp_reached_state["prev_best"] - 0.02:
                    _wp_reached_state["settled_ticks"] += 1
                else:
                    _wp_reached_state["settled_ticks"] = 0
                _wp_reached_state["prev_best"] = _wp_reached_state["best"]
                if _wp_reached_state["settled_ticks"] >= 5:
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
                         elapsed=f"{elapsed}s",
                         best_nav_dist=f"{_wp_reached_state['best']:.2f}",
                         last_nav_dist=f"{_wp_reached_state['value']:.2f}",
                         consecutive=explorer._consecutive_skip_count)
                node.get_logger().info(
                    f"Exploration SKIP: target={skip_target}  "
                    f"best_nav_dist={_wp_reached_state['best']:.2f}m  "
                    f"consecutive={explorer._consecutive_skip_count}"
                )
                state["last_exploration_wp"] = None
                _wp_reached_state["best"] = float("inf")

            if wp is not None:
                wp_key = (round(wp.x, 2), round(wp.y, 2))
                if wp_key != state["last_exploration_wp"]:
                    dist_to_wp = math.hypot(
                        wp.x - (pose.position.x if pose else 0.0),
                        wp.y - (pose.position.y if pose else 0.0),
                    )
                    _exp_log("WP_SET",
                             target=f"({wp.x:.2f},{wp.y:.2f})",
                             robot=robot_pos,
                             dist=f"{dist_to_wp:.2f}")
                    state["last_exploration_wp"] = wp_key
                    state["wp_start_time"] = now_s
                    _wp_reached_state["best"] = float("inf")
                    _wp_reached_state["value"] = float("inf")
                    _wp_reached_state["close_ticks"] = 0
                    _wp_reached_state["settled_ticks"] = 0
                    _wp_reached_state["prev_best"] = float("inf")
                publisher.publish(WaypointPathResponse(waypoints=[wp]))

            if explorer.is_complete():
                if len(explorer._visited) >= explorer._max_waypoints:
                    reason = "budget_exhausted"
                elif explorer._consecutive_skip_count >= explorer._max_consecutive_skips:
                    reason = "max_consecutive_skips"
                else:
                    reason = "no_frontiers"
                _exp_log("DONE",
                         visited=len(explorer._visited),
                         skipped=explorer.skipped_count,
                         reason=reason)
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
