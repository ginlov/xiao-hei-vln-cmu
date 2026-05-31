"""Shared data types for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from xiao_hei_vln.messages.outputs import VLMOutput


@dataclass(frozen=True)
class EvalSample:
    """One evaluation example: a question paired with ground truth and prediction."""

    question: str
    ground_truth: VLMOutput
    prediction: VLMOutput
