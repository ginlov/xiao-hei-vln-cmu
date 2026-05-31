"""Evaluator: groups EvalSamples by type and dispatches to metric computers."""

from __future__ import annotations

from xiao_hei_vln.evaluator.metrics.numerical import compute_numerical_metrics
from xiao_hei_vln.evaluator.metrics.object_reference import compute_object_reference_metrics
from xiao_hei_vln.evaluator.report import EvalReport
from xiao_hei_vln.evaluator.types import EvalSample
from xiao_hei_vln.messages.outputs import NumericalResponse, ObjectReferenceResponse


class Evaluator:
    """Run evaluation on a list of EvalSamples and produce an EvalReport."""

    def evaluate(self, samples: list[EvalSample]) -> EvalReport:
        numerical_pairs: list[tuple[NumericalResponse, NumericalResponse]] = []
        obj_ref_pairs: list[tuple[ObjectReferenceResponse, ObjectReferenceResponse]] = []

        for s in samples:
            gt, pred = s.ground_truth, s.prediction
            if isinstance(gt, NumericalResponse) and isinstance(pred, NumericalResponse):
                numerical_pairs.append((gt, pred))
            elif (
                isinstance(gt, ObjectReferenceResponse)
                and isinstance(pred, ObjectReferenceResponse)
            ):
                obj_ref_pairs.append((gt, pred))
            # Instruction-following (Task 3) requires the official evaluator — skip.

        return EvalReport(
            numerical=compute_numerical_metrics(numerical_pairs) if numerical_pairs else None,
            object_reference=(
                compute_object_reference_metrics(obj_ref_pairs) if obj_ref_pairs else None
            ),
        )
