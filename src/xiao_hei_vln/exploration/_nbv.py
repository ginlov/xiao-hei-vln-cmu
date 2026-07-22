"""Next-Best-View explorer.

Canonical implementation lives in ``_mapfree.NextBestViewExplorer``; this
module re-exports it so ``XIAO_HEI_EXPLORATION_STRATEGY=nbv`` and imports of
``xiao_hei_vln.exploration._nbv`` keep working.
"""

from xiao_hei_vln.exploration._mapfree import NextBestViewExplorer

__all__ = ["NextBestViewExplorer"]
