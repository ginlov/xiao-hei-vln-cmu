"""Qwen3.5 responder backed by an in-process vLLM engine.

The `QwenEngine` (in `engine.py`) lazily imports `vllm` only when
instantiated, so this package is safe to import in dev / CI / dummy-only
environments where vLLM and CUDA are not installed. Plug-and-play
inside `xiao_hei_vln.app.main` via `XIAO_HEI_RESPONDER=qwen`.

See `docs/task3_phase1_framework.md` for the design rationale.
"""

from xiao_hei_vln.qwen.config import QwenConfig
from xiao_hei_vln.qwen.engine import EngineProtocol, HTTPQwenEngine, QwenEngine
from xiao_hei_vln.qwen.responder import QwenResponder

__all__ = [
    "EngineProtocol",
    "HTTPQwenEngine",
    "QwenConfig",
    "QwenEngine",
    "QwenResponder",
]
