"""Scene-graph + Gemini submission responder for the CMU VLN Challenge.

The ``scene_gemini`` responder is the team's end-to-end pipeline:

  - **Frontier exploration** (shared app-level ``FrontierExplorer``) drives
    the robot around the scene.
  - **Scene graph building** — the perception sidecar (YOLO-World + SAM)
    lifts detections into a :class:`~xiao_hei_vln.scene.SceneRepresentation`
    on every exploration tick.
  - **Gemini answering** — once the sweep completes, the populated scene
    graph plus a panorama and occupancy map are handed to Gemini for the
    final numerical / object-reference answer or the Task-2 route.

Selected at runtime via ``XIAO_HEI_RESPONDER=scene_gemini``. Needs the
``XIAO_HEI_GEMINI_API_KEY`` env var and the perception sidecar; build the
image with ``XIAO_HEI_EXTRA=perception,gemini,exploration``.

This replaces the retired ``gemini`` responder, which explored internally
(duplicating the app explorer) and fed Gemini an object-less scene.
"""

from xiao_hei_vln.scene_gemini.responder import SceneGeminiResponder

__all__ = ["SceneGeminiResponder"]
