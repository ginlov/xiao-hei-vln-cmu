"""Online exploration strategies for the CMU VLN challenge.

The module is organised around a single protocol (``ExplorationStrategy``):

    update(snapshot: VLMInput) -> Waypoint | None
    is_complete() -> bool
    reset() -> None

Any class implementing these three methods is a valid exploration strategy
and can be dropped into the main tick loop.  Register new strategies in
``_build_explorer`` (``app/main.py``) and select them at runtime with the
``XIAO_HEI_EXPLORATION_STRATEGY`` environment variable.

Currently implemented
---------------------
frontier (default)
    ``FrontierExplorer`` — builds a 2-D occupancy grid from
    ``terrain_map_ext`` snapshots and navigates toward the largest nearby
    frontier cluster.  Stops when ``max_waypoints`` have been visited.

nbv
    ``NextBestViewExplorer`` — samples reachable FREE poses on the same
    belief map and picks the pose with highest unknown-gain / path-cost.
    Select with ``XIAO_HEI_EXPLORATION_STRATEGY=nbv``.

nav_vlm
    ``NavVLMExplorer`` — asks a vision-language model (Opus 5) for the next
    waypoint on each reach / cannot-reach event, snapping the pick to a
    grid-reachable free cell. Select with
    ``XIAO_HEI_EXPLORATION_STRATEGY=nav_vlm`` (needs an Anthropic key).

nav_task1
    ``NavTask1Explorer`` — a question-directed variant of ``nav_vlm``: it
    drives the robot toward the object named in an OBJECT_REFERENCE question
    (feeding the model the question + scene graph each call) and completes when
    the model declares arrival, at which point the ``scene_claude`` responder
    answers. Select with ``XIAO_HEI_EXPLORATION_STRATEGY=nav_task1``.

Visualisation
-------------
save_exploration_plot(visited_waypoints, grid, output_path)
    Saves a PNG debug plot showing the explored map and the robot path.
    Only supported by strategies that expose ``get_visited_waypoints()``
    and ``get_grid()``.  Requires matplotlib (install the ``[exploration]``
    extra).

save_rviz_screenshot(output_path, display_name=None)
    Saves a PNG of the simulator's RViz window — the traversed path drawn
    over the scene mesh, as the sim rendered it.  Needs an X server to grab
    from; raises ``CaptureError`` otherwise.  Requires python-xlib + pillow
    (the same ``[exploration]`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from xiao_hei_vln.messages.inputs import VLMInput
    from xiao_hei_vln.messages.outputs import Waypoint


@runtime_checkable
class ExplorationStrategy(Protocol):
    """Duck-type protocol every exploration algorithm must satisfy."""

    def update(self, snapshot: "VLMInput") -> "Waypoint | None": ...
    def is_complete(self) -> bool: ...
    def reset(self) -> None: ...


from xiao_hei_vln.exploration._capture import (
    CaptureError,
    list_windows,
    save_rviz_screenshot,
)
from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._nav_task1 import NavTask1Explorer
from xiao_hei_vln.exploration._nav_vlm import NavVLMExplorer
from xiao_hei_vln.exploration._nbv import NextBestViewExplorer
from xiao_hei_vln.exploration._visualize import save_exploration_plot

__all__ = [
    "CaptureError",
    "ExplorationStrategy",
    "FrontierExplorer",
    "NavTask1Explorer",
    "NavVLMExplorer",
    "NextBestViewExplorer",
    "list_windows",
    "save_exploration_plot",
    "save_rviz_screenshot",
]
