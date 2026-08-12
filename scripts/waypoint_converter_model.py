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
# How near a planned run may come to a forbidden gate, as a fraction of its own
# length plus a floor. Both measured over the 121 recorded drives that carry a
# track: the sideways stray from the straight line we plan is 0.18 of the move
# at the median, 0.51 at p90 and 0.60 at p95, and for moves under a metre the
# worst seen was 0.51 m however short they were. So the margin is 0.60 of the
# move, never less than 0.5 m.
GATE_MARGIN_FRAC = 0.60
GATE_MARGIN_MIN = 0.5


def _seg_seg_dist(p1, p2, q1, q2) -> float:
    """Shortest distance between two 2-D segments; 0 if they intersect.

    Clamped parametric solve, with the parallel case falling through to the
    endpoint distances — which is also the answer when either segment is a
    point, so no special case is needed for a degenerate gate.
    """
    p1, p2, q1, q2 = (np.asarray(v, float)[:2] for v in (p1, p2, q1, q2))
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den > 1e-12:
        s = float(np.clip((b * e - c * d) / den, 0.0, 1.0))
        t = float(np.clip((a * e - b * d) / den, 0.0, 1.0))
        # One clamp can invalidate the other, so re-solve each against the
        # other's clamped value. Two passes are exact for segments.
        s = float(np.clip((b * t - d) / a, 0.0, 1.0)) if a > 1e-12 else 0.0
        t = float(np.clip((b * s + e) / c, 0.0, 1.0)) if c > 1e-12 else 0.0
        return float(np.linalg.norm((p1 + s * u) - (q1 + t * v)))
    best = float("inf")
    for pt, (r1, r2) in ((p1, (q1, q2)), (p2, (q1, q2)),
                         (q1, (p1, p2)), (q2, (p1, p2))):
        seg = r2 - r1
        L = float(seg @ seg)
        t = 0.0 if L < 1e-12 else float(np.clip(((pt - r1) @ seg) / L, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(pt - (r1 + t * seg))))
    return best


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
                 keepout: Sequence[tuple[np.ndarray, float]] = (),
                 gates: Sequence[tuple[np.ndarray, np.ndarray]] = ()) -> None:
        """`terrain` is (N, 4): x, y, z, intensity, where intensity is height
        above the local ground — see `robot_io.py` on reading that column.

        `keepout` is a list of `(xy, radius)` the instruction forbids. It is
        deliberately *not* applied to `snap`/`settle`: those predict what the
        organisers' node will do, and it has never heard of our constraint. It
        applies to `allowed`, the set we are willing to choose from — and
        because every motion decision reads that one set, waypoint choice,
        exploration and the arrival test inherit the constraint at once.

        `gates` is the other shape a keep-out comes in, and for "avoid the path
        between X and Y" it is the right one. Two discs big enough to close a
        2 m gap are big enough to close the room: on `livingroom_2` q5, five
        1.2 m discs (three of them the same TV, lifted three times and kept
        apart by `JUMP_M`) left `best_waypoint_toward` with no answer at all,
        and the caller then published its raw waypoint with the constraint
        silently dropped — which is how the vehicle came to drive through the
        middle of the forbidden gap, 0.14 m from its midpoint. A gate is the
        segment joining the two anchors: it closes the corridor between them
        and nothing else. On the same frame it rejects 19 of 928 legal points
        where the discs rejected every usable one.

        A gate forbids *crossing*, not standing: it is a line, so it removes
        nothing from `allowed`, and is enforced on the run from the vehicle to
        wherever a waypoint would settle.
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
        self.gates = [(np.asarray(a, float)[:2], np.asarray(b, float)[:2])
                      for a, b in gates]
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
        # A gate stops the ray for the same reason: exploration must not read
        # straight through a corridor it is forbidden to drive down.
        for g0, g1 in self.gates:
            e = g1 - g0
            den = float(u[0] * e[1] - u[1] * e[0])
            if abs(den) < 1e-9:
                continue                          # parallel: never crossed
            w = g0 - o
            t = float(w[0] * e[1] - w[1] * e[0]) / den      # along the ray
            s = float(w[0] * u[1] - w[1] * u[0]) / den      # along the gate
            if 0.0 < t < far and 0.0 <= s <= 1.0:
                far = min(far, max(t - min_advance, 0.0))
        return far

    def best_waypoint_toward(self, target, vehicle, *, search: int = 400,
                             min_move: float = 0.0
                             ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The waypoint whose *resting place* lands nearest `target`.

        Returns `(goal, where_it_settles, distance_from_there_to_target)`.

        `min_move` rejects goals the vehicle would settle less than that far
        from where it stands. Pass it whenever the *point* of the move is to
        change viewpoint rather than to close distance: the vehicle stops once
        it is within `waypointXYRadius` of its waypoint, so anything nearer
        than that is not a small move, it is no move at all. Leave it at zero
        for an approach, where settling close by is the correct answer.

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
        # Drop the candidates a constraint will reject *before* ranking, not
        # inside the loop. `search` keeps only the nearest candidates to the
        # target, which is a pure optimisation until a keep-out is added — and
        # then it is a bug, because the candidates a keep-out rejects are
        # exactly the nearest ones when the target lies beyond it. Every one of
        # the first 400 is refused, the loop ends with nothing, and the caller
        # reads "no legal move" from a frame with hundreds of them. That is how
        # `livingroom_2` q5 reported `boxed in` with 721 legal moves available,
        # and how the same leg on the run before it fell through to publishing
        # its raw waypoint with the keep-out dropped.
        #
        # The test on the published point is not the one that decides — the run
        # to where it *settles* is, below — but it is cheap and it culls the
        # doomed half, so the window covers plausible candidates instead.
        if self.gates or self.keepout:
            def survives(margin):
                return np.array(
                    [not (self.gates and self.crosses_gate(veh, p, margin))
                     and not (self.keepout and self._crosses_keepout(veh, p))
                     for p in legal])
            keep = survives(None)               # clearance, scaled by length
            # A margin must never be able to seal the only way through. It is
            # there because we cannot predict the driven path, and when nothing
            # clears it the honest fallback is the run that at least does not
            # cross — worse, but still not a violation of the constraint as
            # written. `livingroom_2`'s only legal route south is a strip the
            # reference trajectory threads 0.8 m from the tea table, which a
            # 1.2 m margin would otherwise close.
            if not keep.any():
                keep = survives(0.0)
            if not keep.any():
                return None
            legal = legal[keep]
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
            if self.gates and self.crosses_gate(veh, s):
                continue
            if min_move and float(np.linalg.norm(s - veh)) < min_move:
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

    def gate_clearance(self, a, b) -> float:
        """How near the straight run from `a` to `b` comes to a forbidden gate.

        Zero when it crosses. `inf` when there are no gates.
        """
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        return min((_seg_seg_dist(a, b, g0, g1) for g0, g1 in self.gates),
                   default=float("inf"))

    def crosses_gate(self, a, b, margin: float | None = None) -> bool:
        """Is this run too near a forbidden gate to publish?

        Not "does it cross": the vehicle does not drive the line we plan. It
        drives whatever `local_planner` chooses, and measured over 121 recorded
        drives that path strays sideways from the straight line by 0.18 of its
        length at the median and 0.60 at the 95th percentile. On
        `livingroom_2` a 2.42 m move planned straight down x = 0 ended 1.49 m
        east and took the vehicle through the middle of the forbidden gap; the
        crossing test had passed the plan, and the plan was not what was driven.

        So the test is on clearance, and the margin scales with the length of
        the move, because the deviation does. A short hop earns a small margin
        and a long one cannot be checked at all — which is the other half of
        the answer, and why the caller caps a step while a keep-out is in force.
        """
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        if margin is None:
            margin = max(GATE_MARGIN_MIN,
                         GATE_MARGIN_FRAC * float(np.linalg.norm(b - a)))
        return self.gate_clearance(a, b) <= margin

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
