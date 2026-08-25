"""Claude-backed object-reference answering over the perception scene graph.

Pairs with the ``nav_task1`` explorer: the explorer drives the robot to the
object named in an OBJECT_REFERENCE question, then :class:`SceneClaudeResponder`
dumps the built scene graph to Claude to pick the answer. Select the responder
with ``XIAO_HEI_RESPONDER=scene_claude``.
"""

from __future__ import annotations

from xiao_hei_vln.scene_claude.responder import SceneClaudeResponder

__all__ = ["SceneClaudeResponder"]
