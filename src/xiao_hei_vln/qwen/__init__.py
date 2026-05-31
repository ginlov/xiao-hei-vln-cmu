"""Qwen3.5 responder with pluggable inference engine.

Two engine implementations are available (see ``engine.py``):

- ``HTTPQwenEngine`` (default) — calls a vLLM OpenAI-compatible HTTP
  sidecar. The ai_module image only needs ``openai`` + ``pillow``.
- ``QwenEngine`` (legacy) — loads vLLM in-process. Requires CUDA and
  ``pip install .[qwen-local]``.

Heavy imports are lazy, so this package is safe to import in dev / CI /
dummy-only environments. Select via ``XIAO_HEI_RESPONDER=qwen`` in
``xiao_hei_vln.app.main``.
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
