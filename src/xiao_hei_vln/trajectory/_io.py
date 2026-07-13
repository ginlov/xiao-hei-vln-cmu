"""Back-compat shim — canonical home is :mod:`xiao_hei_vln.scene.io`.

The scene-zip readers used to live here, but pulling them in via
``xiao_hei_vln.trajectory._io`` also pulled the trajectory package's
top-level shapely import, breaking responders that don't ship shapely.
They now live in :mod:`xiao_hei_vln.scene.io`; this module re-exports
them so the trajectory CLI (``python -m xiao_hei_vln.trajectory``)
keeps working unchanged.
"""

from xiao_hei_vln.scene.io import (
    parse_traversable_ply,
    parse_traversable_ply_from_zip,
    read_objects_from_zip,
)

__all__ = [
    "parse_traversable_ply",
    "parse_traversable_ply_from_zip",
    "read_objects_from_zip",
]
