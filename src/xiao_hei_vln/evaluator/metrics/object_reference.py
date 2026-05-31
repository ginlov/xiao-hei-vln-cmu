"""Metrics for Question Type 2: Object Reference Questions.

Challenge scoring: 0–2 points based on 3D bounding box overlap with ground truth.
  2 pts : IoU >= 0.5
  1 pt  : 0.25 <= IoU < 0.5
  0 pts : IoU < 0.25

Primary metric : Mean IoU
Diagnostics    : SR@IoU≥0.5, SR@IoU≥0.25, mean center distance, mean challenge score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from xiao_hei_vln.messages.common import Vector3
from xiao_hei_vln.messages.outputs import ObjectReferenceResponse


@dataclass
class ObjectReferenceMetrics:
    n: int
    mean_iou: float
    sr_at_iou_50: float     # fraction with IoU >= 0.5
    sr_at_iou_25: float     # fraction with IoU >= 0.25
    mean_center_distance: float
    mean_challenge_score: float     # average of 0/1/2 raw scores
    mean_challenge_score_norm: float  # normalized to [0, 1]
    ious: list[float] = field(default_factory=list)
    center_distances: list[float] = field(default_factory=list)
    challenge_scores: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_iou": self.mean_iou,
            "sr_at_iou_50": self.sr_at_iou_50,
            "sr_at_iou_25": self.sr_at_iou_25,
            "mean_center_distance": self.mean_center_distance,
            "mean_challenge_score": self.mean_challenge_score,
            "mean_challenge_score_norm": self.mean_challenge_score_norm,
        }


def _iou_3d_aabb(
    center_a: Vector3, size_a: Vector3,
    center_b: Vector3, size_b: Vector3,
) -> float:
    """Intersection over Union for two axis-aligned 3D bounding boxes."""

    def overlap_1d(ca: float, sa: float, cb: float, sb: float) -> float:
        lo_a, hi_a = ca - sa / 2, ca + sa / 2
        lo_b, hi_b = cb - sb / 2, cb + sb / 2
        return max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))

    ix = overlap_1d(center_a.x, size_a.x, center_b.x, size_b.x)
    iy = overlap_1d(center_a.y, size_a.y, center_b.y, size_b.y)
    iz = overlap_1d(center_a.z, size_a.z, center_b.z, size_b.z)

    intersection = ix * iy * iz
    vol_a = size_a.x * size_a.y * size_a.z
    vol_b = size_b.x * size_b.y * size_b.z
    union = vol_a + vol_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _center_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _challenge_score(iou: float) -> int:
    if iou >= 0.5:
        return 2
    if iou >= 0.25:
        return 1
    return 0


def compute_object_reference_metrics(
    pairs: list[tuple[ObjectReferenceResponse, ObjectReferenceResponse]],
) -> ObjectReferenceMetrics:
    """Compute object-reference metrics from (ground_truth, prediction) pairs."""
    if not pairs:
        return ObjectReferenceMetrics(
            n=0,
            mean_iou=0.0,
            sr_at_iou_50=0.0,
            sr_at_iou_25=0.0,
            mean_center_distance=0.0,
            mean_challenge_score=0.0,
            mean_challenge_score_norm=0.0,
        )

    ious: list[float] = []
    distances: list[float] = []
    scores: list[int] = []

    for gt, pred in pairs:
        iou = _iou_3d_aabb(pred.center, pred.size, gt.center, gt.size)
        dist = _center_distance(pred.center, gt.center)
        ious.append(iou)
        distances.append(dist)
        scores.append(_challenge_score(iou))

    n = len(pairs)
    mean_iou = sum(ious) / n
    mean_dist = sum(distances) / n
    mean_score = sum(scores) / n

    return ObjectReferenceMetrics(
        n=n,
        mean_iou=mean_iou,
        sr_at_iou_50=sum(1 for v in ious if v >= 0.5) / n,
        sr_at_iou_25=sum(1 for v in ious if v >= 0.25) / n,
        mean_center_distance=mean_dist,
        mean_challenge_score=mean_score,
        mean_challenge_score_norm=mean_score / 2.0,
        ious=ious,
        center_distances=distances,
        challenge_scores=scores,
    )
