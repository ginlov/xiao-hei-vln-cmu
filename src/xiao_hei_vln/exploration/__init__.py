"""Exploration policies that decide where the robot should go next.

Phase A ships ``FrontierPlanner`` — a reactive, per-tick scorer over
the ``GlobalMap``'s frontier clusters. Future phases will add TSP-based
coverage sweep and VLM-scored frontier ranking.
"""

from xiao_hei_vln.exploration.frontier import FrontierPlanner

__all__ = ["FrontierPlanner"]
