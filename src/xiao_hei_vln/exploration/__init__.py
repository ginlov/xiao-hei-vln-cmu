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

Visualisation
-------------
save_exploration_plot(visited_waypoints, grid, output_path)
    Saves a PNG debug plot showing the explored map and the robot path.
    Only supported by strategies that expose ``get_visited_waypoints()``
    and ``get_grid()``.  Requires matplotlib (install the ``[exploration]``
    extra).
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


from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._visualize import save_exploration_plot

__all__ = ["ExplorationStrategy", "FrontierExplorer", "save_exploration_plot"]
