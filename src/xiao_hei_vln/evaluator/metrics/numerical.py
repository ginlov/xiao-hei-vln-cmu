"""Metrics for Question Type 1: Numerical Questions.

Challenge scoring: 0 or 1 point (exact match).

Primary metric : Accuracy (exact match rate)
Diagnostics    : Mean Error, MAE, signed error list (for histogram)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xiao_hei_vln.messages.outputs import NumericalResponse


@dataclass
class NumericalMetrics:
    n: int
    accuracy: float
    mean_error: float       # avg(pred - gt); positive = overcounting
    mae: float              # avg |pred - gt|
    errors: list[int] = field(default_factory=list)  # signed errors for histogram

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "mean_error": self.mean_error,
            "mae": self.mae,
        }


def compute_numerical_metrics(
    pairs: list[tuple[NumericalResponse, NumericalResponse]],
) -> NumericalMetrics:
    """Compute numerical metrics from (ground_truth, prediction) pairs."""
    if not pairs:
        return NumericalMetrics(n=0, accuracy=0.0, mean_error=0.0, mae=0.0, errors=[])

    errors = [pred.value - gt.value for gt, pred in pairs]
    n = len(pairs)
    exact_matches = sum(1 for e in errors if e == 0)

    return NumericalMetrics(
        n=n,
        accuracy=exact_matches / n,
        mean_error=sum(errors) / n,
        mae=sum(abs(e) for e in errors) / n,
        errors=errors,
    )
