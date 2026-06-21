"""rclpy entry point: drives the Task-1 stack at 2 Hz with the configured responder.

Pick the responder with `XIAO_HEI_RESPONDER`:

  - `dummy` (default) — the deterministic port of `dummyVLM.cpp`. No GPU.
  - `qwen`            — Qwen3.5 via vLLM. By default talks to a vLLM
                        HTTP sidecar (`XIAO_HEI_QWEN_VLLM_BASE_URL`).
                        Falls back to in-process vLLM when the URL is
                        unset (`pip install .[qwen-local]` + CUDA GPU).
"""

from __future__ import annotations

import math
import os

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.sync import LatestCache

TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0"))
RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy").lower()

# Exploration phase — set XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0 to disable.
_EXPLORATION_MAX_WAYPOINTS = int(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINTS", "100"))
_EXPLORATION_STRATEGY = os.environ.get("XIAO_HEI_EXPLORATION_STRATEGY", "frontier").lower()
_EXPLORATION_MAX_WAYPOINT_DIST = float(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST", "1.5"))
_EXPLORATION_LOG_DIR = os.environ.get("XIAO_HEI_EXPLORATION_PLOT_DIR", "")


def _build_responder(name: str):
    if name == "dummy":
        from xiao_hei_vln.dummy import DummyResponder

        return DummyResponder(), None
    if name == "qwen":
        from dataclasses import asdict

        from xiao_hei_vln.logger import VLMLogger
        from xiao_hei_vln.qwen import HTTPQwenEngine, QwenConfig, QwenEngine, QwenResponder

        config = QwenConfig.from_env()
        engine = HTTPQwenEngine(config) if config.vllm_base_url else QwenEngine(config)
        engine.warmup()

        logger = None
        log_dir = os.environ.get("XIAO_HEI_VLM_LOG_DIR", "")
        if log_dir:
            logger = VLMLogger(
                log_dir,
                config=asdict(config),
                responder_name="qwen",
                tick_hz=TICK_HZ,
            )
        return QwenResponder(engine, config, logger=logger), logger
    raise ValueError(
        f"Unknown XIAO_HEI_RESPONDER={name!r}; expected one of: dummy, qwen",
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


def _maybe_save_plot(explorer, node) -> None:
    """Save the debug PNG if XIAO_HEI_EXPLORATION_PLOT_DIR is configured.

    Skipped silently when the active strategy does not expose get_grid() /
    get_visited_waypoints() (not all algorithms maintain an OccupancyGrid).
    """
    if not _EXPLORATION_LOG_DIR:
        return
    if not (hasattr(explorer, "get_visited_waypoints") and hasattr(explorer, "get_grid")):
        node.get_logger().info("Exploration plot skipped: strategy does not support it.")
        return
    try:
        from pathlib import Path

        from xiao_hei_vln.exploration import save_exploration_plot

        out_dir = Path(_EXPLORATION_LOG_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "exploration.png"
        save_exploration_plot(
            explorer.get_visited_waypoints(),
            explorer.get_grid(),
            out_path,
        )
        node.get_logger().info(f"Exploration plot saved to {out_path}")
    except Exception as exc:  # noqa: BLE001
        node.get_logger().warn(f"Could not save exploration plot: {exc}")


def main() -> None:
    # Local imports so the rest of the package stays importable without rclpy.
    import rclpy
    from rclpy.node import Node

    from xiao_hei_vln.adapters.ros.publishers import VLMOutputPublisher
    from xiao_hei_vln.adapters.ros.subscribers import bind_subscribers

    rclpy.init()
    node_name = "xiao_hei_qwen_vlm" if RESPONDER_NAME == "qwen" else "xiao_hei_dummy_vlm"
    node: Node = rclpy.create_node(node_name)

    cache = LatestCache()
    subs = bind_subscribers(node, cache)
    publisher = VLMOutputPublisher(node)
    responder, logger = _build_responder(RESPONDER_NAME)

    explorer = _build_explorer(node)

    # Structured exploration log — survives the container via the mounted volume.
    _log_dir = _EXPLORATION_LOG_DIR or "/exploration_logs"
    _exp_log_file = open(os.path.join(_log_dir, "exploration.log"), "w", buffering=1)

    def _exp_log(event: str, **fields) -> None:
        now_s = node.get_clock().now().nanoseconds / 1e9
        parts = "  ".join(f"{k}={v}" for k, v in fields.items())
        _exp_log_file.write(f"[{now_s:.3f}] {event}  {parts}\n")

    from std_msgs.msg import Float32
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    _nav_qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )

    # Track nav stack's distance to current waypoint; best = closest approach this target.
    _wp_reached_state = {"value": float("inf"), "close_ticks": 0, "best": float("inf")}
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
    }

    def tick() -> None:
        from xiao_hei_vln.messages.outputs import WaypointPathResponse

        now = node.get_clock().now().to_msg()
        snapshot = cache.snapshot(state["tick_id"], Stamp(sec=now.sec, nanosec=now.nanosec))
        state["tick_id"] += 1

        # Exploration phase: runs only when no question is active.
        if explorer is not None and not explorer.is_complete() and snapshot.question is None:
            if not state["exploration_started"]:
                node.get_logger().info("Exploration started.")
                state["exploration_started"] = True
                _exp_log("START",
                         max_waypoints=explorer._max_waypoints,
                         threshold=_WP_REACHED_THRESHOLD,
                         stuck_timeout=f"{explorer._stuck_timeout_s}s",
                         max_skips=explorer._max_consecutive_skips)

            pose = snapshot.pose
            robot_pos = (
                f"({pose.position.x:.2f},{pose.position.y:.2f})" if pose is not None else "unknown"
            )

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
            wp = explorer.update(snapshot)

            if explorer.skipped_count > prev_skipped:
                now_s = node.get_clock().now().nanoseconds / 1e9
                elapsed = round(now_s - state["wp_start_time"], 1) if state["wp_start_time"] else "?"
                _exp_log("WP_SKIP",
                         target=state["last_exploration_wp"],
                         robot=robot_pos,
                         elapsed=f"{elapsed}s",
                         best_nav_dist=f"{_wp_reached_state['best']:.2f}",
                         last_nav_dist=f"{_wp_reached_state['value']:.2f}",
                         consecutive=explorer._consecutive_skip_count)
                node.get_logger().info(
                    f"Exploration SKIP: target={state['last_exploration_wp']}  "
                    f"best_nav_dist={_wp_reached_state['best']:.2f}m  "
                    f"consecutive={explorer._consecutive_skip_count}"
                )
                state["last_exploration_wp"] = None
                _wp_reached_state["best"] = float("inf")

            if wp is not None:
                wp_key = (round(wp.x, 2), round(wp.y, 2))
                if wp_key != state["last_exploration_wp"]:
                    now_s = node.get_clock().now().nanoseconds / 1e9
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
                _maybe_save_plot(explorer, node)
            return

        # New question → reset the responder so it handles it from scratch.
        if (
            snapshot.question is not None
            and snapshot.question.text != state["last_question_text"]
        ):
            responder.reset()
            state["last_question_text"] = snapshot.question.text
            node.get_logger().info(f"Received question: {snapshot.question.text!r}")

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
