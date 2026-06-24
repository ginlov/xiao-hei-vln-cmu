"""Persistent perception state shared across VLM ticks.

Currently exposes ``GlobalMap`` — a 2D occupancy grid that stitches
``terrain_ext`` snapshots together using the provided pose. Future
modules (object detector, localizer, scene memory) will live here as
well; see the plan in ``docs/task1_io_spec.md`` and the related
architecture notes.
"""

from xiao_hei_vln.perception.global_map import FrontierCluster, GlobalMap

__all__ = ["FrontierCluster", "GlobalMap"]
