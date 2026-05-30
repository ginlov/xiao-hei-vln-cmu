"""rclpy entry point: drives the Task-1 stack at 2 Hz with the configured responder.

Pick the responder with `XIAO_HEI_RESPONDER`:

  - `dummy` (default) — the deterministic port of `dummyVLM.cpp`. No GPU.
  - `qwen`            — Qwen3.5 via an in-process vLLM engine (see
                        `docs/task3_phase1_framework.md`). Requires the
                        `[qwen]` optional install + a CUDA GPU.
"""

from __future__ import annotations

import os

from xiao_hei_vln.messages.common import Stamp
from xiao_hei_vln.sync import LatestCache

TICK_HZ = float(os.environ.get("XIAO_HEI_VLM_TICK_HZ", "2.0"))
RESPONDER_NAME = os.environ.get("XIAO_HEI_RESPONDER", "dummy").lower()


def _build_responder(name: str):
    if name == "dummy":
        from xiao_hei_vln.dummy import DummyResponder

        return DummyResponder()
    if name == "qwen":
        from xiao_hei_vln.qwen import QwenConfig, QwenEngine, QwenResponder

        config = QwenConfig.from_env()
        engine = QwenEngine(config)
        engine.warmup()
        return QwenResponder(engine, config)
    raise ValueError(
        f"Unknown XIAO_HEI_RESPONDER={name!r}; expected one of: dummy, qwen",
    )


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
    responder = _build_responder(RESPONDER_NAME)

    state = {"tick_id": 0, "last_question_text": None}

    def tick() -> None:
        now = node.get_clock().now().to_msg()
        snapshot = cache.snapshot(state["tick_id"], Stamp(sec=now.sec, nanosec=now.nanosec))
        state["tick_id"] += 1

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
