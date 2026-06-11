"""Online exploration strategies for the CMU VLN challenge.

The module is organised around a single protocol:

    update(snapshot: VLMInput) -> Waypoint | None
    is_complete() -> bool
    reset() -> None

Any class implementing these three methods is a valid exploration strategy
and can be dropped into the main tick loop.

Currently implemented
---------------------
FrontierExplorer
    Frontier-based exploration.  Builds a 2-D occupancy grid from
    ``terrain_map_ext`` snapshots and navigates toward the largest nearby
    frontier cluster.  Stops when ``max_waypoints`` have been visited.

Visualisation
-------------
save_exploration_plot(visited_waypoints, grid, output_path)
    Saves a PNG debug plot showing the explored map and the robot path.
    Call after exploration is complete.  Requires matplotlib (install the
    ``[exploration]`` extra).
"""

from xiao_hei_vln.exploration._frontier import FrontierExplorer
from xiao_hei_vln.exploration._visualize import save_exploration_plot

__all__ = ["FrontierExplorer", "save_exploration_plot"]
