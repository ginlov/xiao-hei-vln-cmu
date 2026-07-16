"""Gemini API responder for the CMU VLN Challenge.

Architecture:

  - ``GeminiConfig`` — env-var configuration (API key, model, budgets).
  - ``GeminiEngine`` — thin wrapper around the ``google-genai`` SDK that
    issues JSON-schema-locked multimodal calls.
  - ``GeminiResponder`` — exploration-then-answer responder that wraps a
    ``PerceptionResponder`` for the explore phase, then on a trigger
    feeds the accumulated scene representation to Gemini and emits the
    parsed response.

Selected at runtime via ``XIAO_HEI_RESPONDER=gemini``. Requires the
``XIAO_HEI_GEMINI_API_KEY`` environment variable. Install the
optional-dependency group with ``pip install .[gemini]``.
"""

# Submodules import their dependencies lazily so the package stays
# importable even in environments that don't have the gemini extra
# installed.
from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.engine import GeminiEngine, GeminiEngineProtocol
from xiao_hei_vln.gemini.responder import GeminiResponder

__all__ = [
    "GeminiConfig",
    "GeminiEngine",
    "GeminiEngineProtocol",
    "GeminiResponder",
]
