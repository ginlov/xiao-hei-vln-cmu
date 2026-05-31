"""EvalReport: aggregates metrics across all question types and formats output."""

from __future__ import annotations

import json
from dataclasses import dataclass

from xiao_hei_vln.evaluator.metrics.numerical import NumericalMetrics
from xiao_hei_vln.evaluator.metrics.object_reference import ObjectReferenceMetrics


@dataclass
class EvalReport:
    numerical: NumericalMetrics | None = None
    object_reference: ObjectReferenceMetrics | None = None

    def print_summary(self) -> None:
        print("=" * 60)
        print("CMU VLN Challenge 2026 — Evaluation Report")
        print("=" * 60)

        if self.numerical is not None:
            m = self.numerical
            print(f"\n[Task 1] Numerical Questions  (n={m.n})")
            print(f"  Accuracy (exact match) : {m.accuracy:.1%}")
            print(f"  Mean Error (pred - gt) : {m.mean_error:+.3f}")
            print(f"  MAE                    : {m.mae:.3f}")
            if m.errors:
                _print_error_hist(m.errors)

        if self.object_reference is not None:
            m = self.object_reference
            print(f"\n[Task 2] Object Reference Questions  (n={m.n})")
            print(f"  Mean IoU               : {m.mean_iou:.3f}")
            print(f"  SR @ IoU ≥ 0.50        : {m.sr_at_iou_50:.1%}")
            print(f"  SR @ IoU ≥ 0.25        : {m.sr_at_iou_25:.1%}")
            print(f"  Mean center distance   : {m.mean_center_distance:.3f} m")
            print(f"  Mean challenge score   : {m.mean_challenge_score:.3f} / 2")
            print(f"  Mean score (norm)      : {m.mean_challenge_score_norm:.3f}")

        print("\n" + "=" * 60)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.numerical is not None:
            result["numerical"] = self.numerical.as_dict()
        if self.object_reference is not None:
            result["object_reference"] = self.object_reference.as_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _print_error_hist(errors: list[int], width: int = 40) -> None:
    """Print a compact ASCII histogram of signed errors."""
    if not errors:
        return
    lo, hi = min(errors), max(errors)
    if lo == hi:
        print(f"  Error distribution     : all errors = {lo}")
        return

    buckets: dict[int, int] = {}
    for e in errors:
        buckets[e] = buckets.get(e, 0) + 1

    max_count = max(buckets.values())
    print("  Error distribution (signed pred - gt):")
    for val in range(lo, hi + 1):
        count = buckets.get(val, 0)
        bar_len = int(count / max_count * width) if max_count else 0
        bar = "#" * bar_len
        print(f"    {val:+4d} | {bar:<{width}} {count}")
