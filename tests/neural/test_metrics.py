"""Offline unit tests for benchmarks/neural/metrics.py."""
import numpy as np

from benchmarks.neural.metrics import mae, smape


def test_mae_hand_computed():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 4.0, 3.0])
    assert mae(a, b) == 2.0 / 3.0


def test_smape_hand_computed():
    # a=[2], b=[4]: |2-4| / ((2+4)/2) = 2/3; *100 => 66.6667
    assert abs(smape(np.array([2.0]), np.array([4.0])) - (2.0 / 3.0 * 100)) < 1e-9


def test_smape_zero_denominator_is_zero():
    # both zero -> denom 0 -> contributes 0, not NaN
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_metrics_return_native_float():
    assert isinstance(mae([1.0], [2.0]), float)
    assert isinstance(smape([1.0], [2.0]), float)
