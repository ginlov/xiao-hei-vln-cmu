#!/usr/bin/env python3
"""From a VLM's 2D box to one 3D box, over several views of the same object.

The object-reference answer is a 3D box scored by IoU, so this task needs two
things the grounding loop never needed: an *extent*, and a way to combine what
several viewpoints each saw. Both come from the perception package -- this
module is the adapter, not a second lifter.

    box_2d (one face)  ->  face mask  ->  equirect mask
                       ->  PointLifter.lift(mask, scan, pose)   # z-buffer,
                                                                # cluster gate
                       ->  TargetBox.add(...)                   # per-view box
                       ->  TargetBox.box()                      # the answer

Two things here are the opposite of the obvious design, and both are measured
rather than argued:

**The clouds are not unioned.** `ObjectMap._observe` records why: a single
observation's box is already close to right -- volume ratio 0.95 against ground
truth -- and what ruins it is pooling the points. Each view's cloud is offset
from the true centre by ~0.23 m in a direction that depends on where the robot
stood, so the union spans the object *plus* that scatter and comes out about 6x
too big. With an oracle association, one AABB over the pooled points scores
**0.138 of 2** against **0.443** for averaging the per-view boxes. So each view
contributes a box, and the boxes are averaged.

**The average is weighted by inlier count.** A view that caught 900 returns of
the object knows more about where it is than one that scraped 12 off its edge,
and unweighted the two count the same. Worth mIoU 0.203 -> 0.217 and 37.9% ->
44.2% at IoU >= 0.25 over seven scenes, with recall and precision unchanged.

`ObjectMap` itself is deliberately not used. Its association is keyed on the
detector's labels -- `add(label, ...)` merges into a same-label node and
`merge` follows "the stronger label" -- so every one of its entry points
inherits YOLO's naming. Here the target is already known and there is exactly
one of it, so association is not a problem that needs solving. The *estimator*
is imported from it, because that part was measured.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "perception", REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geometry as G  # noqa: E402
from xiao_hei_vln.messages.common import Quaternion, Vector3  # noqa: E402
from xiao_hei_vln.perception.lifter import PointLifter  # noqa: E402
from xiao_hei_vln.perception.object_map import (  # noqa: E402
    _core_points, _percentile_box, iou_3d, robust_center,
)

# Inlier floor for a view to count at all. The perception default is 10; a
# reference target is small and often seen from 2-3 m, where a 0.1 m cup
# subtends few returns, so this is deliberately lower than the map's. Views
# below it are recorded and skipped, not silently dropped -- `TargetBox.skipped`
# is how a run explains an answer built from two views instead of five.
MIN_INLIERS = 6

# A view whose lifted centre is this far from the running estimate is a
# different object, not a new look at this one. Generous, because early views
# are exactly the ones that are wrong by up to half a metre and still worth
# having; the point is to reject a box that grabbed the sofa behind the pillow.
OUTLIER_M = 1.2

# How `TargetBox.box` turns several views' extents into one. See its docstring:
# the choice is not settled and the default is the one with evidence.
SIZE_MODES = ("average", "max", "union")


@lru_cache(maxsize=1)
def _inverse_lut():
    """The equirect -> face lookup, built once. ~1.2 M pixels, so not per view.

    `errstate` because the builder projects every equirect pixel through every
    face, and the pixels outside a face's 100 deg FOV divide by zero on the
    way to being marked `-1`. The warnings are correct and the result is not
    affected; they are silenced here rather than in `geometry` so nothing else
    that calls it changes behaviour.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return G.build_inverse_lut()


def face_box_to_equirect_mask(box_px, face_idx: int) -> np.ndarray:
    """A rectangle in face pixels as a boolean equirect mask.

    `box_px` is `[ymin, xmin, ymax, xmax]` in face pixels -- the order the
    grounding prompt already returns and `vlm_locate` already consumes, kept
    the same so a box can be handed to either path without transposing.

    The rectangle is drawn in the *face* and reprojected, not drawn in the
    equirect: a straight-edged face box becomes a curved region on the sphere,
    and the equirect AABB of it is up to a third larger near the seams. The
    reprojection is what `face_mask_to_equirect_mask` does with the inverse
    LUT, and it is exactly what the sidecar does with SAM's masks.

    **A box near a face corner loses most of its pixels, and a small one loses
    all of them.** The inverse LUT gives each equirect pixel to whichever face
    views it most centrally, so a face's extreme corner -- at the edge of its
    100 deg FOV, inside a neighbour's overlap band -- is claimed by the
    neighbour. Measured: one 60x60 face box maps to 4814 equirect pixels near
    the centre and **186** in the corner, and a 41x41 box in the corner maps to
    **none**. The lift then sees 4% of the returns it would have, or nothing.
    `lift_box` reports `mask_px` so a caller can re-view instead of committing
    to a box built from a handful of returns; it is not this function's job to
    refuse, because the same box is fine one face over.
    """
    ymin, xmin, ymax, xmax = (float(v) for v in box_px)
    face = np.zeros((G.FACE_SIZE, G.FACE_SIZE), dtype=bool)
    y0, y1 = sorted((int(round(ymin)), int(round(ymax))))
    x0, x1 = sorted((int(round(xmin)), int(round(xmax))))
    y0, x0 = max(0, y0), max(0, x0)
    y1 = min(G.FACE_SIZE - 1, y1)
    x1 = min(G.FACE_SIZE - 1, x1)
    if y1 < y0 or x1 < x0:
        return np.zeros((G.EQUIRECT_H, G.EQUIRECT_W), dtype=bool)
    face[y0:y1 + 1, x0:x1 + 1] = True
    return G.face_mask_to_equirect_mask(face, int(face_idx), _inverse_lut())


def lift_mask(mask: np.ndarray, scan_map: np.ndarray, pose: dict, *,
              min_inliers: int = MIN_INLIERS,
              max_depth_m: float | None = None) -> dict:
    """One view's mask -> `{xyz, points, n}` in the map frame, or `n=0`.

    `pose` is the loop's plain dict (`position`, `orientation`), converted here
    so callers never have to know the perception package's message types. It
    must be the pose **at the image's timestamp**, not the newest one: pairing
    the newest pose with an older frame misplaces the lift by up to 17 deg of
    azimuth while turning, which at 3 m is 0.9 m and is more than a small
    target's whole error budget. `LatestCache` does that matching online; a
    recorded corpus carries it as `image_pose_*`.
    """
    lifter = PointLifter(min_inliers=min_inliers, max_depth_m=max_depth_m)
    p = np.asarray(pose["position"], float)
    q = np.asarray(pose["orientation"], float)
    # `errstate` because numpy raises divide/overflow/invalid from inside `@`
    # on the scan-sized matmuls even when every input is finite and the result
    # is exact -- checked against `einsum` on a recorded scan: identical, all
    # finite, and the flags still fire. Suppressed here rather than in
    # `lifter.py` so nothing else that calls it changes.
    with np.errstate(divide="ignore", over="ignore", under="ignore",
                     invalid="ignore"):
        res = lifter.lift(
            mask, np.asarray(scan_map, float),
            Vector3(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        )
    if res.position is None or res.inlier_points is None:
        return {"n": int(res.n_inliers), "xyz": None, "points": None}
    return {"n": int(res.n_inliers),
            "xyz": np.array([res.position.x, res.position.y, res.position.z]),
            "points": np.asarray(res.inlier_points, float)}


# Below this many equirect pixels a mask is too compressed to lift from -- see
# `face_box_to_equirect_mask` on the face-corner blind zone. Not a refusal:
# `lift_box` reports it and the caller decides whether to re-view.
THIN_MASK_PX = 400


def lift_box(box_px, face_idx: int, scan_map: np.ndarray, pose: dict, **kw) -> dict:
    """`face_box_to_equirect_mask` then `lift_mask`, for the no-SAM path.

    Adds `mask_px` and `thin` to the result. `thin` means the box reprojected
    to very few equirect pixels, which near a face corner happens to boxes that
    look perfectly reasonable in the face -- so a `thin` lift is a reason to
    look again from somewhere else, not a reason to trust the box it produced.
    """
    mask = face_box_to_equirect_mask(box_px, face_idx)
    px = int(mask.sum())
    got = lift_mask(mask, scan_map, pose, **kw)
    got["mask_px"] = px
    got["thin"] = px < THIN_MASK_PX
    return got


class TargetBox:
    """One object's box, averaged over the views that saw it.

    Not an `ObjectMap` with one node: there is no association to do, and the
    map's entry points all take a detector label. What is kept from it is the
    estimator -- `_core_points` to cut a mask that spans two surfaces,
    `_percentile_box` so one stray return cannot set a corner, and the
    inlier-weighted average over per-view boxes.

    **Outliers are settled at read time, not at add time.** The first version
    gated each incoming view against the running estimate, and a replay of
    `cv_0818_1745_arabic_room_arabicq4_p1` showed why that is exactly backwards:
    step 3 lifted 25 returns from 3.7 m into a 0.06 m-thick sliver, became the
    anchor, and then refused step 12's **242 returns from 0.8 m** for being
    1.2 m away from it. The first view is the one most likely to be wrong --
    returns on a target grow as 1/r^2, so the near views are both the last to
    arrive and the ones that know most. Clustering at read time makes the
    result independent of arrival order and lets weight decide.
    """

    def __init__(self, *, outlier_m: float = OUTLIER_M,
                 anchor: np.ndarray | None = None,
                 size_mode: str = "average"):
        if size_mode not in SIZE_MODES:
            raise ValueError(f"size_mode must be one of {SIZE_MODES}")
        self.outlier_m = float(outlier_m)
        self.anchor = None if anchor is None else np.asarray(anchor, float)
        self.size_mode = size_mode
        self.centres: list[np.ndarray] = []
        self.extents: list[np.ndarray] = []
        self.weights: list[float] = []
        self.views: list[dict] = []
        self.rejected: list[dict] = []

    # -- building ----------------------------------------------------------

    def add(self, points: np.ndarray | None, *, weight: float | None = None,
            note: str = "") -> bool:
        """Fold one view's inlier cloud in. `False` only when it had no cloud.

        `weight` defaults to the number of core points, which is the measured
        choice. A caller with a mask-quality signal -- SAM returns its own
        predicted IoU -- should pass `n_core * sam_iou` instead, so a view
        whose mask SAM itself doubts counts for less.
        """
        if points is None or len(points) < 3:
            self.rejected.append({"why": "no points", "n": 0, "note": note})
            return False
        core = _core_points(np.asarray(points, float))
        if len(core) < 3:
            self.rejected.append({"why": "no core points", "n": len(points),
                                  "note": note})
            return False
        c, _ = robust_center(core)
        if c is None:
            # Below `robust_center`'s own inlier floor the cloud is real but
            # too thin to place; the median is what `ObjectMap._observe` uses
            # in the same spot.
            c = np.median(core, axis=0)
        lo, hi = _percentile_box(core)
        self.centres.append(np.asarray(c, float))
        self.extents.append(np.asarray(hi, float) - np.asarray(lo, float))
        self.weights.append(float(weight if weight is not None else len(core)))
        self.views.append({"n": len(core), "xyz": np.asarray(c, float).tolist(),
                           "extent": (np.asarray(hi) - np.asarray(lo)).tolist(),
                           "weight": self.weights[-1], "note": note})
        return True

    # -- reading -----------------------------------------------------------

    @property
    def n_views(self) -> int:
        return len(self.centres)

    def consensus(self) -> list[int]:
        """Indices of the views that agree.

        Single-link clustering of the view centres at `outlier_m`, then:

        * with an `anchor` -- the cluster containing the view nearest it. The
          anchor is the binding the vehicle **committed to and drove at**, and
          that is the object being asked about. Election by weight is not a
          substitute for it.
        * without one -- the heaviest cluster, which is a guess.

        The anchor exists because electing a winner was measured and is wrong.
        Replaying `cv_0818_1745_arabic_room_arabicq4_p1` -- *"the stool under
        the picture"* -- the loop's grounding drifted mid-leg onto a table
        carrying a coffee pot, and five of eight views landed there. By total
        weight they won 511 to 203, so the heaviest cluster put the box 0.09 m
        from the coffee pot, while the two views it discarded sat 0.23 m and
        0.20 m from the two real stools. Weight measures where the looks went,
        not what was asked for.

        Single-link rather than a fixed radius around the anchor because a
        genuine set of views drifts -- each is offset from the truth in the
        direction the robot stood -- so a chain of overlapping views is one
        object, while a box that grabbed the sofa behind the pillow is a
        separate component however close it starts.
        """
        n = self.n_views
        if n == 0:
            return []
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        C = np.array(self.centres)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(C[i] - C[j]) <= self.outlier_m:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        if self.anchor is not None:
            nearest = int(np.argmin(np.linalg.norm(C - self.anchor, axis=1)))
            return sorted(groups[find(nearest)])
        return sorted(max(groups.values(),
                          key=lambda g: (sum(self.weights[i] for i in g), len(g))))

    def outliers(self) -> list[dict]:
        keep = set(self.consensus())
        return [v for i, v in enumerate(self.views) if i not in keep]

    def box(self, idx: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """`(centre, size)` over the agreeing views.

        The centre is always the inlier-weighted mean. The *extent* is the
        unresolved half, and `size_mode` names the choice:

        ``average``  the weighted mean of the per-view extents. This is the
                     measured default: `ObjectMap` reports a single
                     observation's volume ratio at 0.95 against ground truth,
                     so averaging protects the size from each view's centre
                     scatter rather than accumulating it.
        ``max``      the per-axis maximum over views. Right if each view sees a
                     different *slice* of the object rather than all of it --
                     replaying one recorded leg, a 25-return view from 3.7 m
                     gave a 0.06 m extent on an axis whose truth is 0.40 m, and
                     no amount of averaging recovers that.
        ``union``    the AABB of the agreeing views' boxes. The most generous,
                     and the one `ObjectMap` measured as worst when applied to
                     pooled *points* (0.138 of 2 against 0.443).

        Which is right depends on whether an orbit's views are complete looks
        or partial slices, and that has not been measured on orbit data. The
        default is the one with evidence behind it; the others exist so the
        replay can settle it instead of this docstring.
        """
        idx = self.consensus() if idx is None else idx
        if not idx:
            return np.zeros(3), np.zeros(3)
        w = np.array([self.weights[i] for i in idx], float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        C = np.array([self.centres[i] for i in idx])
        E = np.array([self.extents[i] for i in idx])
        centre = (C * w[:, None]).sum(axis=0)
        if self.size_mode == "average":
            size = (E * w[:, None]).sum(axis=0)
        elif self.size_mode == "max":
            size = E.max(axis=0)
        else:                                             # union
            lo = (C - E / 2).min(axis=0)
            hi = (C + E / 2).max(axis=0)
            centre, size = (lo + hi) / 2, hi - lo
        return centre, size

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = self.box()
        return c - s / 2, c + s / 2

    def settled(self, *, iou: float = 0.85, need: int = 3) -> bool:
        """Has the box stopped moving?

        Compares the box built from every view against the box built without
        the newest one. `need` views minimum, because the comparison is
        meaningless before there is something to compare. This is the stopping
        rule an orbit uses in place of a fixed view count -- and it can be
        wrong in the direction of stopping early, so a caller should also cap
        the views rather than trust it alone.
        """
        if self.n_views < need:
            return False
        prev = TargetBox(outlier_m=self.outlier_m, anchor=self.anchor,
                         size_mode=self.size_mode)
        prev.centres = self.centres[:-1]
        prev.extents = self.extents[:-1]
        prev.weights = self.weights[:-1]
        a_lo, a_hi = prev.bounds()
        b_lo, b_hi = self.bounds()
        return iou_3d(a_lo, a_hi, b_lo, b_hi) >= iou

    def as_dict(self) -> dict:
        c, s = self.box()
        keep = self.consensus()
        return {"centre": c.tolist(), "size": s.tolist(),
                "n_views": self.n_views, "n_agreeing": len(keep),
                "views": self.views, "outliers": self.outliers(),
                "rejected": self.rejected,
                "weight_agreeing": float(sum(self.weights[i] for i in keep))}


def pca_heading(points: np.ndarray) -> float:
    """Yaw of the cloud's long horizontal axis, radians.

    For `visualization_msgs/Marker`, which carries an orientation. Whether
    publishing one helps is unknown: our own scorer builds the ground-truth
    AABB with the heading ignored (`perception/eval.py`), which caps a perfect
    axis-aligned prediction at mean IoU 0.783, and the organisers' scorer is
    not public. So this exists to be A/B'd, not switched on.
    """
    p = np.asarray(points, float)[:, :2]
    if len(p) < 3:
        return 0.0
    p = p - p.mean(axis=0)
    axis = np.linalg.svd(p, full_matrices=False)[2][0]
    return float(np.arctan2(axis[1], axis[0]))
