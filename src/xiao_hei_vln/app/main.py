"""rclpy entry point: drives the Task-1 stack at 2 Hz with the configured responder.

Pick the responder with `XIAO_HEI_RESPONDER`:

  - `dummy` (default) — the deterministic port of `dummyVLM.cpp`. No GPU.
  - `qwen`            — Qwen3.5 via vLLM. By default talks to a vLLM
                        HTTP sidecar (`XIAO_HEI_QWEN_VLLM_BASE_URL`).
                        Falls back to in-process vLLM when the URL is
                        unset (`pip install .[qwen-local]` + CUDA GPU).
"""

from __future__ import annotations

import os

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.sync import LatestCache

TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0"))
RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy").lower()

# Exploration phase — set XIAO_HEI_EXPLORATION_MAX_WAYPOINTS=0 to disable.
_EXPLORATION_MAX_WAYPOINTS = int(os.environ.get("XIAO_HEI_EXPLORATION_MAX_WAYPOINTS", "30"))
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
    """Return a FrontierExplorer, or None when exploration is disabled."""
    if _EXPLORATION_MAX_WAYPOINTS <= 0:
        return None
    from xiao_hei_vln.exploration import FrontierExplorer

    explorer = FrontierExplorer(max_waypoints=_EXPLORATION_MAX_WAYPOINTS)
    node.get_logger().info(
        f"Exploration enabled: FrontierExplorer(max_waypoints={_EXPLORATION_MAX_WAYPOINTS})"
    )
    return explorer


def _maybe_save_plot(explorer, node) -> None:
    """Save the debug PNG if XIAO_HEI_EXPLORATION_PLOT_DIR is configured."""
    if not _EXPLORATION_PLOT_DIR:
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

    state = {"tick_id": 0, "last_question_text": None, "exploration_started": False,
             "last_exploration_wp": None}

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

            wp = explorer.update(snapshot)
            if wp is not None:
                # Log only when the target waypoint changes.
                wp_key = (round(wp.x, 2), round(wp.y, 2))
                if wp_key != state["last_exploration_wp"]:
                    visited = len(explorer.get_visited_waypoints())
                    node.get_logger().info(
                        f"Exploration waypoint {visited + 1}/{explorer._max_waypoints}: "
                        f"({wp.x:.2f}, {wp.y:.2f})"
                    )
                    state["last_exploration_wp"] = wp_key
                publisher.publish(WaypointPathResponse(waypoints=[wp]))
            if explorer.is_complete():
                node.get_logger().info(
                    f"Exploration complete. "
                    f"Visited {len(explorer.get_visited_waypoints())} waypoints."
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
