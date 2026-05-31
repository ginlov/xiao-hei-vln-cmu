"""Tests for numerical evaluation metrics."""

import pytest

from xiao_hei_vln.evaluator.metrics.numerical import NumericalMetrics, compute_numerical_metrics
from xiao_hei_vln.messages.outputs import NumericalResponse


def _pair(gt: int, pred: int) -> tuple[NumericalResponse, NumericalResponse]:
    return NumericalResponse(value=gt), NumericalResponse(value=pred)


def test_empty():
    m = compute_numerical_metrics([])
    assert m.n == 0
    assert m.accuracy == 0.0
    assert m.mean_error == 0.0
    assert m.mae == 0.0


def test_all_correct():
    pairs = [_pair(3, 3), _pair(5, 5), _pair(1, 1)]
    m = compute_numerical_metrics(pairs)
    assert m.n == 3
    assert m.accuracy == pytest.approx(1.0)
    assert m.mean_error == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.errors == [0, 0, 0]


def test_all_wrong():
    pairs = [_pair(3, 5), _pair(2, 1)]  # errors: +2, -1
    m = compute_numerical_metrics(pairs)
    assert m.n == 2
    assert m.accuracy == pytest.approx(0.0)
    assert m.mean_error == pytest.approx(0.5)   # (2 + -1) / 2
    assert m.mae == pytest.approx(1.5)            # (2 + 1) / 2


def test_mixed():
    pairs = [_pair(4, 4), _pair(2, 5), _pair(7, 6)]  # errors: 0, +3, -1
    m = compute_numerical_metrics(pairs)
    assert m.n == 3
    assert m.accuracy == pytest.approx(1 / 3)
    assert m.mean_error == pytest.approx((0 + 3 - 1) / 3)
    assert m.mae == pytest.approx((0 + 3 + 1) / 3)


def test_systematic_overcounting():
    pairs = [_pair(1, 3), _pair(2, 4), _pair(3, 5)]  # all +2
    m = compute_numerical_metrics(pairs)
    assert m.mean_error == pytest.approx(2.0)
    assert m.mae == pytest.approx(2.0)
    assert all(e == 2 for e in m.errors)


def test_as_dict_keys():
    m = compute_numerical_metrics([_pair(1, 1)])
    d = m.as_dict()
    assert set(d.keys()) == {"n", "accuracy", "mean_error", "mae"}
