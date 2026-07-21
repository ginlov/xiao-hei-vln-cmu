"""Gemini API engine + offline evaluator for the CMU VLN Challenge.

Architecture:

  - ``GeminiConfig`` — env-var configuration (API key, model, budgets).
  - ``GeminiEngine`` — thin wrapper around the ``google-genai`` SDK that
    issues JSON-schema-locked multimodal calls.
  - ``xiao_hei_vln.gemini.batch`` — offline batch evaluator (scores Gemini
    over reconstructed scene graphs, no simulator needed).

The **live** responder that drives Gemini during a run now lives in
:mod:`xiao_hei_vln.scene_gemini` (``XIAO_HEI_RESPONDER=scene_gemini``): it
reuses ``GeminiEngine`` here for reasoning while the perception sidecar
builds the object scene graph. The old ``GeminiResponder`` — which
explored internally and fed Gemini an object-less scene — has been retired.

Install the optional-dependency group with ``pip install .[gemini]``.
"""

# Submodules import their dependencies lazily so the package stays
# importable even in environments that don't have the gemini extra
# installed.
from xiao_hei_vln.gemini.config import GeminiConfig
from xiao_hei_vln.gemini.engine import GeminiEngine, GeminiEngineProtocol

__all__ = [
    "GeminiConfig",
    "GeminiEngine",
    "GeminiEngineProtocol",
]
