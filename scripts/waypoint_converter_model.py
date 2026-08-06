#!/usr/bin/env python3
"""Where will the robot actually stop if we publish this waypoint?

`/way_point_with_heading` is not a position command. `waypointConverter`
replaces our waypoint with a point of its own choosing, and TASK 28 read that
displacement as the platform clamping down on an approach. It is not: the
converter *discards* any waypoint whose neighbourhood is inside the obstacle
inflation and re-minimises globally, which on `japanese_room` sent the vehicle
1.08 m to the opposite side of where it was asked to go.

This reimplements that decision, so the loop can know the answer before it
spends a drive and a grounding call finding out. Every constant is read off
`waypoint_converter.launch`; the algorithm is `waypointConverter.cpp` lines
196-245:

    travArea      = /terrain_map points with intensity <  obstacleHeightThre
    obstacleArea  = the rest                                (both voxel 0.05)
    candidates    = travArea within searchDisThre of the *vehicle*
    legal(p)      = no obstacleArea point within obstacleDisThre of p
    pick          = argmin over legal candidates of
                        |p - waypoint| + vehicleDisWeight * |p - vehicle|

`/terrain_map` is on the README's list of topics an AI module may use at test
time, so nothing here needs a topic we are not allowed to read.

Validated against two drives on `japanese_room` from the same start pose,
publishing (+1.10, -0.44):

    predicted settle (+0.93, +0.59)
    actual           (+1.02, +0.64)   TASK 28's run   error 0.104 m
    actual           (+1.01, +0.61)   replayed live   error 0.081 m
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree

# waypoint_converter.launch, verbatim.
TERRAIN_VOXEL = 0.05
OBSTACLE_HEIGHT_THRE = 0.05
OBSTACLE_DIS_THRE = 0.75
SEARCH_DIS_THRE = 5.0
VEHICLE_DIS_WEIGHT = 0.5
ADJ_DIS_THRE = 5.0
WAYPOINT_XY_RADIUS = 0.3


def voxel_downsample(pts: np.ndarray, leaf: float = TERRAIN_VOXEL) -> np.ndarray:
    """PCL's VoxelGrid: one centroid per occupied voxel."""
    if len(pts) == 0:
        return pts
    _, inv = np.unique(np.floor(pts / leaf).astype(np.int64), axis=0,
                       return_inverse=True)
    out = np.zeros((int(inv.max()) + 1, pts.shape[1]))
    np.add.at(out, inv, pts)
    return out / np.bincount(inv).reshape(-1, 1)


class ConverterModel:
    """One `/terrain_map` frame, and what the converter would do with it."""

    def __init__(self, terrain: np.ndarray, *,
                 keepout: Sequence[tuple[np.ndarray, float]] = ()) -> None:
        """`terrain` is (N, 4): x, y, z, intensity, where intensity is height
        above the local ground — see `robot_io.py` on reading that column.

        `keepout` is a list of `(xy, radius)` the instruction forbids. It is
        deliberately *not* applied to `snap`/`settle`: those predict what the
        organisers' node will do, and it has never heard of our constraint. It
        applies to `allowed`, the set we are willing to choose from — and
        because every motion decision reads that one set, waypoint choice,
        exploration and the arrival test inherit the constraint at once.
        """
        if terrain.ndim != 2 or terrain.shape[1] != 4:
            raise ValueError(f"terrain must be (N, 4); got {terrain.shape}")
        xyz, h = terrain[:, :3].astype(float), terrain[:, 3]
        self.trav = voxel_downsample(xyz[h < OBSTACLE_HEIGHT_THRE])
        self.obst = voxel_downsample(xyz[h >= OBSTACLE_HEIGHT_THRE])
        if len(self.trav) == 0:
            raise ValueError("no traversable points — is column 3 really the "
                             "intensity field, or PCL padding?")
        self.kd_trav = cKDTree(self.trav)               # 3-D, as in the C++
        self.kd_obst = (cKDTree(self.obst[:, :2]) if len(self.obst)
                        else None)                      # obstacle test is 2-D
        # Legality is a property of the frame, not of the query, so resolve it
        # once. The C++ re-tests it inside its candidate loop, which is fine at
        # 10 Hz for one waypoint and ruinous when simulating hundreds.
        if self.kd_obst is None:
            self.legal = self.trav
        else:
            clear = self.kd_obst.query(self.trav[:, :2])[0]
            self.legal = self.trav[clear >= OBSTACLE_DIS_THRE]
        self.kd_legal = cKDTree(self.legal) if len(self.legal) else None

        self.keepout = [(np.asarray(xy, float)[:2], float(r)) for xy, r in keepout]
        self.allowed = self.legal
        for xy, r in self.keepout:
            if len(self.allowed) == 0:
                break
            self.allowed = self.allowed[
                np.linalg.norm(self.allowed[:, :2] - xy, axis=1) >= r]

    def forbidden(self, points) -> np.ndarray:
        """Which of `points` fall inside a keep-out disc."""
        p = np.atleast_2d(np.asarray(points, float))[:, :2]
        bad = np.zeros(len(p), dtype=bool)
        for xy, r in self.keepout:
            bad |= np.linalg.norm(p - xy, axis=1) < r
        return bad

    def legal_points(self) -> np.ndarray:
        """Every point we are willing to stand on: the converter's set, minus
        anything the instruction forbids."""
        return self.allowed[:, :2]

    def snap(self, waypoint, vehicle) -> np.ndarray:
        """One `poseHandler` callback: the point the converter republishes."""
        wp = np.asarray(waypoint, float)[:2]
        veh3 = np.array([vehicle[0], vehicle[1], 0.0], float)
        if np.linalg.norm(veh3[:2] - wp) >= ADJ_DIS_THRE:
            return wp                                   # waypointAdj stays false
        if self.kd_legal is None:
            return wp
        idx = self.kd_legal.query_ball_point(veh3, SEARCH_DIS_THRE)
        if not idx:
            return wp                                   # minInd < 0: unchanged
        cand = self.legal[idx]
        score = (np.linalg.norm(cand[:, :2] - wp, axis=1)
                 + VEHICLE_DIS_WEIGHT * np.linalg.norm(cand - veh3, axis=1))
        return cand[score.argmin()][:2]

    def reach_along(self, origin, direction, *, half_width: float = 1.0,
                    min_advance: float = 0.2) -> float:
        """How far the legal set extends along `direction`, within a corridor.

        Answers "can the vehicle actually go that way, and how far" from the
        terrain alone — no drive, no grounding call. On loft the model asked
        three times for a heading with 0.00 m of reach while 5.16 m was
        available 30 deg away, and each refusal cost a call.
        """
        if len(self.allowed) == 0:
            return 0.0
        u = np.asarray(direction, float)[:2]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        d = self.allowed[:, :2] - np.asarray(origin, float)[:2]
        # numpy's vectorised matmul path raises divide-by-zero and overflow on
        # the larger clouds while producing the right answer — checked against
        # the scalar form, max abs difference 0.0, on inputs verified finite.
        # `vlm_approach.free_range_along` carries the same note.
        with np.errstate(all="ignore"):
            along = d @ u
        perp = np.abs(d[:, 0] * u[1] - d[:, 1] * u[0])
        ok = (along > min_advance) & (perp < half_width)
        if not ok.any():
            return 0.0
        far = float(along[ok].max())
        # Removing forbidden points from `allowed` is not enough here, because
        # this takes a maximum: a keep-out sitting in the middle of the corridor
        # leaves the points beyond it and the reach reads straight through the
        # zone. Stop at the near edge instead.
        o = np.asarray(origin, float)[:2]
        for xy, r in self.keepout:
            t = float((xy - o) @ u)
            if 0 < t < far and float(np.linalg.norm(xy - (o + t * u))) < r:
                far = min(far, max(t - r, 0.0))
        return far

    def best_waypoint_toward(self, target, vehicle, *, search: int = 400
                             ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The waypoint whose *resting place* lands nearest `target`.

        Returns `(goal, where_it_settles, distance_from_there_to_target)`.

        Publishing a legal point instead of "target minus a fixed standoff" is
        the whole point of modelling the converter: a legal point is a local
        minimum of its score — stepping back toward the vehicle adds `d` and
        removes only `0.5 d` — so it is republished as itself, where a point
        inside the inflation is discarded and re-minimised somewhere we did not
        mean.

        Scoring those legal points by `|goal - target|` is the obvious rule and
        it is wrong. The vehicle stops `waypointXYRadius` short of its goal
        along the approach, so of two goals equally far from the target, the one
        nearer the vehicle settles *further* from it. On office_1 two candidates
        sat 1.088 m and 1.085 m from the target; picking the first because it
        was 0.37 m nearer the vehicle ended the run 0.89 m from the plant
        instead of 0.69 m. Score where the vehicle ends up, not where it aims.
        """
        legal = self.legal_points()
        if len(legal) == 0:
            return None
        tgt = np.asarray(target, float)[:2]
        veh = np.asarray(vehicle, float)[:2]
        # Walk candidates nearest-the-target first and prune with an exact
        # bound: settling stops within waypointXYRadius of the goal, so no goal
        # can settle nearer the target than `|goal - target| - waypointXYRadius`.
        # Once that lower bound exceeds the best distance found, nothing further
        # out can win. Without it this loop is seconds; with it, milliseconds.
        d = np.linalg.norm(legal - tgt, axis=1)
        order = np.argsort(d)[:search]
        best = None
        for i in order:
            if best is not None and d[i] - WAYPOINT_XY_RADIUS >= best[2]:
                break
            s = self.settle(legal[i], veh, step=0.1)
            # The converter does not know about the keep-out, so a goal we are
            # allowed to want can still be snapped into a zone we are not
            # allowed to enter — and the score is on the trajectory driven, so
            # the way there counts too.
            if self.keepout and (self.forbidden(s)[0]
                                 or self._crosses_keepout(veh, s)):
                continue
            v = float(np.linalg.norm(s - tgt))
            if best is None or v < best[2]:
                best = (legal[i], s, v)
        return best

    def _crosses_keepout(self, a, b, *, step: float = 0.1) -> bool:
        """Does the straight run from `a` to `b` clip a forbidden disc?"""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        n = max(int(np.linalg.norm(b - a) / step), 1)
        return bool(self.forbidden(a + np.outer(np.linspace(0, 1, n + 1),
                                                b - a)).any())

    def settle(self, waypoint, vehicle, *, step: float = 0.05,
               max_iter: int = 2000) -> np.ndarray:
        """Where the vehicle ends up.

        The C++ re-snaps on every pose callback at 10 Hz, and the score's
        `vehicleDisWeight` term re-centres as the vehicle moves — so the
        vehicle chases a target that moves with it, and the resting place is
        the fixed point of that iteration rather than the first snap. Motion is
        modelled as a straight line; the local planner's path differs, but the
        fixed point does not depend on how it is approached.
        """
        veh = np.asarray(vehicle, float)[:2].copy()
        for _ in range(max_iter):
            tgt = self.snap(waypoint, veh)
            gap = float(np.linalg.norm(tgt - veh))
            if gap < WAYPOINT_XY_RADIUS:
                return veh
            veh += (tgt - veh) / gap * min(step, gap)
        return veh
