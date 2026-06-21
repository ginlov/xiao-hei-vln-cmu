"""rclpy entry point: drives the Task-1 stack at 2 Hz with the configured responder.

Pick the responder with `XIAO_HEI_RESPONDER`:

  - `dummy` (default) — the deterministic port of `dummyVLM.cpp`. No GPU.
  - `qwen`            — Qwen3.5 via vLLM. By default talks to a vLLM
                        HTTP sidecar (`XIAO_HEI_QWEN_VLLM_BASE_URL`).
                        Falls back to in-process vLLM when the URL is
                        unset (`pip install .[qwen-local]` + CUDA GPU).
"""

from __future__ import annotations

import logging
import os

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.sync import LatestCache

TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0"))
RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy").lower()

# Exploration phase — set XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0 to disable.
_EXPLORATION_MAX_WAYPOINTS = int(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINTS", "30"))
_EXPLORATION_STRATEGY = os.environ.get("XIAO_HEI_EXPLORATION_STRATEGY", "frontier").lower()
_EXPLORATION_MAX_WAYPOINT_DIST = float(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINT_DIST", "1.5"))
# Optional: directory to save the debug PNG after exploration completes.
_EXPLORATION_PLOT_DIR = os.environ.get("XIAO_HEI_EXPLORATION_PLOT_DIR", "")


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
    if not _EXPLORATION_PLOT_DIR:
        return
    if not (hasattr(explorer, "get_visited_waypoints") and hasattr(explorer, "get_grid")):
        node.get_logger().info("Exploration plot skipped: strategy does not support it.")
        return
    try:
        from pathlib import Path

        from xiao_hei_vln.exploration import save_exploration_plot

        out_dir = Path(_EXPLORATION_PLOT_DIR)
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

    # Build the exploration strategy (None when disabled via env var).
    explorer = _build_explorer(node)

    # Route the frontier module's debug logs through rclpy so they appear in ROS output.
    # Set XIAO_HEI_EXPLORATION_DEBUG=1 to enable.
    if os.environ.get("XIAO_HEI_EXPLORATION_DEBUG", "0") == "1":
        _py_logger = logging.getLogger("xiao_hei_vln.exploration")
        _py_logger.setLevel(logging.DEBUG)
        _py_logger.addHandler(logging.StreamHandler())

    # Nav debug log — written to the mounted volume so it survives the container.
    _nav_log_path = os.path.join(_EXPLORATION_PLOT_DIR or "/exploration_logs", "nav_debug.log")
    _nav_log_file = open(_nav_log_path, "w", buffering=1)  # line-buffered

    def _nav_log(msg: str) -> None:
        now_s = node.get_clock().now().nanoseconds / 1e9
        line = f"[{now_s:.3f}] {msg}\n"
        _nav_log_file.write(line)
        node.get_logger().info(f"[nav_debug] {msg}")

    # Subscribe to /way_point_reached — nav stack signals when it finishes a goal.
    from std_msgs.msg import Float32
    from sensor_msgs.msg import PointCloud2
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    import struct

    _debug_qos = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
    )

    _wp_reached_state = {"value": float("inf"), "close_ticks": 0}
    _WP_REACHED_THRESHOLD = 0.92  # nav stack settles between 0.25-0.90m depending on obstacles

    def _on_wp_reached(msg) -> None:
        v = float(msg.data)
        _wp_reached_state["value"] = v
        _nav_log(f"way_point_reached value={v:.4f}")

    node.create_subscription(Float32, "/way_point_reached", _on_wp_reached, _debug_qos)

    # Subscribe to /traversable_area — log bounds and point count periodically.
    _traversable_logged = {"count": 0}

    def _on_traversable(msg) -> None:
        _traversable_logged["count"] += 1
        # Only log every 10th message to avoid spam.
        if _traversable_logged["count"] % 10 != 1:
            return
        n = int(msg.width) * int(msg.height)
        step = int(msg.point_step)
        raw = bytes(msg.data)
        field_offsets = {f.name: int(f.offset) for f in msg.fields}
        xs, ys = [], []
        for i in range(n):
            base = i * step
            ox = field_offsets.get("x")
            oy = field_offsets.get("y")
            if ox is not None and oy is not None:
                x = struct.unpack_from("<f", raw, base + ox)[0]
                y = struct.unpack_from("<f", raw, base + oy)[0]
                xs.append(x)
                ys.append(y)
        if xs:
            _nav_log(
                f"traversable_area points={n} "
                f"x=[{min(xs):.2f},{max(xs):.2f}] "
                f"y=[{min(ys):.2f},{max(ys):.2f}]"
            )
        else:
            _nav_log(f"traversable_area points={n} (no x/y fields found)")

    node.create_subscription(PointCloud2, "/traversable_area", _on_traversable, _debug_qos)

    state = {"tick_id": 0, "last_question_text": None, "exploration_started": False,
             "last_exploration_wp": None, "exploration_tick": 0}

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

            pose = snapshot.pose
            robot_pos = (
                f"({pose.position.x:.2f}, {pose.position.y:.2f})" if pose is not None else "unknown"
            )

            # Use /way_point_reached (nav stack's own distance) to detect arrival.
            # When it stays below threshold for 3 consecutive ticks, the nav stack
            # has done its best — advance regardless of odometry distance.
            if explorer._current_target is not None:
                if _wp_reached_state["value"] < _WP_REACHED_THRESHOLD:
                    _wp_reached_state["close_ticks"] += 1
                    if _wp_reached_state["close_ticks"] >= 3:
                        node.get_logger().info(
                            f"Nav stack settled at {_wp_reached_state['value']:.3f}m "
                            f"— advancing waypoint"
                        )
                        explorer.advance()
                        _wp_reached_state["close_ticks"] = 0
                        _wp_reached_state["value"] = float("inf")  # prevent carry-over to next target
                else:
                    _wp_reached_state["close_ticks"] = 0

            prev_skipped = explorer.skipped_count
            wp = explorer.update(snapshot)
            state["exploration_tick"] += 1
            etick = state["exploration_tick"]

            # Surface stuck-skip events immediately as INFO.
            if explorer.skipped_count > prev_skipped:
                node.get_logger().info(
                    f"Exploration waypoint SKIPPED (stuck): "
                    f"target={state['last_exploration_wp']}  robot={robot_pos}  "
                    f"total_skipped={explorer.skipped_count}"
                )
                state["last_exploration_wp"] = None

            if wp is not None:
                # Log every time the target waypoint changes.
                wp_key = (round(wp.x, 2), round(wp.y, 2))
                if wp_key != state["last_exploration_wp"]:
                    visited = len(explorer._visited)
                    node.get_logger().info(
                        f"Exploration waypoint {visited + 1}/{explorer._max_waypoints}: "
                        f"({wp.x:.2f}, {wp.y:.2f})  robot={robot_pos}  visited_so_far={visited}"
                    )
                    state["last_exploration_wp"] = wp_key
                publisher.publish(WaypointPathResponse(waypoints=[wp]))
            else:
                if not explorer.is_complete():
                    node.get_logger().debug(
                        f"Explorer returned None (no frontier yet)  robot={robot_pos}  "
                        f"etick={etick}"
                    )

            # Periodic heartbeat every 20 exploration ticks to catch silent-spin.
            if etick % 20 == 0:
                node.get_logger().info(
                    f"Exploration heartbeat: etick={etick}  "
                    f"visited={len(explorer._visited)}/{explorer._max_waypoints}  "
                    f"skipped={explorer.skipped_count}  "
                    f"current_target={state['last_exploration_wp']}  robot={robot_pos}"
                )
            if explorer.is_complete():
                node.get_logger().info(
                    f"Exploration complete. "
                    f"Visited {len(explorer._visited)} waypoints, "
                    f"skipped {explorer.skipped_count} "
                    f"(consecutive_skips={explorer._consecutive_skip_count})."
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
        _nav_log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
