"""Chronax-side forecast metrics — one source of truth for the neural harness.

Same formulas as the retired bespoke scripts (gru_benchmark.py etc.), computed
in float64 accumulation. (The retired scripts accumulated in the array's native
dtype; values differ only in the last few ulps — harmless and arguably more
correct.)
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error as a native float."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%) as a native float; zero-denominator entries contribute 0."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(a) + np.abs(b)) / 2.0
    ratio = np.divide(
        np.abs(a - b),
        denom,
        out=np.zeros_like(denom),
        where=denom != 0.0,
    )
    return float(np.mean(ratio) * 100.0)
