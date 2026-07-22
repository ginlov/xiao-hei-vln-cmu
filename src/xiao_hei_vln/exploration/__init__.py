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
    ``FrontierExplorer``
nearest / random
    ``NearestFrontierExplorer`` / ``RandomFrontierExplorer``
lawnmower
    ``LawnmowerExplorer``
wall_follow / nbv / rrt
    Map-free strategies on the live belief occupancy grid
    (``WallFollowExplorer``, ``NextBestViewExplorer``, ``RRTExplorer``).
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
from xiao_hei_vln.exploration._lawnmower import LawnmowerExplorer
from xiao_hei_vln.exploration._mapfree import (
    NextBestViewExplorer,
    RRTExplorer,
    WallFollowExplorer,
)
from xiao_hei_vln.exploration._metric import ExplorationScore, score_exploration
from xiao_hei_vln.exploration._nearest import NearestFrontierExplorer, RandomFrontierExplorer
from xiao_hei_vln.exploration._visualize import save_exploration_plot

__all__ = [
    "CaptureError",
    "ExplorationStrategy",
    "FrontierExplorer",
    "NearestFrontierExplorer",
    "RandomFrontierExplorer",
    "LawnmowerExplorer",
    "WallFollowExplorer",
    "NextBestViewExplorer",
    "RRTExplorer",
    "ExplorationScore",
    "score_exploration",
    "list_windows",
    "save_exploration_plot",
    "save_rviz_screenshot",
]
