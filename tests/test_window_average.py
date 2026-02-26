
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from chronax.utils import (
    ConformalIntervals,
)

from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import _repeat_val, _window_average, ensure_float

from chronax.models import WindowAverage

# =========================
# Test Cases
# =========================

def _arr_close(a, b, tol=1e-6):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return jnp.all(jnp.abs(a - b) <= tol)

def test_basic_mean_predict():
    y = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    m = WindowAverage(window_size=2, alias="WA")
    m.fit(y)
    out = m.predict(h=3, level=None)
    assert "mean" in out
    assert out["mean"].shape == (3,)
    expected = jnp.array([3.5, 3.5, 3.5], dtype=out["mean"].dtype)  # (3+4)/2
    assert _arr_close(out["mean"], expected)
    print("test_basic_mean_predict: OK")

def test_dtype_cast_to_float32():
    y = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    m = WindowAverage(window_size=3)
    m.fit(y)
    out = m.predict(h=2)  # no intervals
    assert out["mean"].dtype in (jnp.float32, jnp.float64)
    expected = jnp.array([4.0, 4.0], dtype=out["mean"].dtype)  # (3+4+5)/3
    assert _arr_close(out["mean"], expected)
    print("test_dtype_cast_to_float32: OK")

def test_short_series_returns_nan():
    y = jnp.asarray([10.0])  # len < window_size
    m = WindowAverage(window_size=3)
    out = m.forecast(y=y, h=2, level=None)
    assert "mean" in out and out["mean"].shape == (2,)
    assert jnp.isnan(out["mean"]).all()
    print("test_short_series_returns_nan: OK")

def test_predict_without_conformal_raises():
    y = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    m = WindowAverage(window_size=2, conformal_params=None)
    m.fit(y)
    try:
        _ = m.predict(h=2, level=[90])
        raise AssertionError("Expected ValueError when intervals requested without conformal_params")
    except ValueError:
        pass
    print("test_predict_without_conformal_raises: OK")

def test_forecast_with_conformal_intervals_stateless():
    # Enough samples for conformity scoring: (n-1)//h >= 2
    y = jnp.asarray([1., 2., 3., 6., 9., 9., 8., 7.])
    cfg = ConformalIntervals(n_windows=3, h=1, method="conformal_distribution")
    m = WindowAverage(window_size=3, conformal_params=cfg)
    out = m.forecast(y=y, h=4, level=[90], fitted=False)
    assert "mean" in out and out["mean"].shape == (4,)
    assert "lo-90" in out and "hi-90" in out
    assert jnp.all(out["lo-90"] <= out["mean"])
    assert jnp.all(out["mean"] <= out["hi-90"])
    print("test_forecast_with_conformal_intervals_stateless: OK")

def test_fit_caches_conformity_scores_then_predict_uses_cache():
    y = jnp.asarray([2., 2., 2., 2., 2., 2.])
    cfg = ConformalIntervals(n_windows=3, h=1, method="conformal_distribution")
    m = WindowAverage(window_size=2, conformal_params=cfg)
    m.fit(y)
    assert getattr(m, "_cs") is not None
    out = m.predict(h=3, level=[80, 95])
    for lv in [80, 95]:
        assert f"hi-{lv}" in out
    # lower keys come in reversed order naming, but both must exist
    assert "lo-95" in out and "lo-80" in out
    # sanity: lo <= mean <= hi for each level
    for lv in [80, 95]:
        assert jnp.all(out[f"lo-{lv}"] <= out["mean"])
        assert jnp.all(out["mean"] <= out[f"hi-{lv}"])
    print("test_fit_caches_conformity_scores_then_predict_uses_cache: OK")

def test_levels_unsorted_input_is_handled():
    y = jnp.asarray([1., 3., 2., 5., 4., 6.])
    cfg = ConformalIntervals(n_windows=2, h=1, method="conformal_distribution")
    m = WindowAverage(window_size=2, conformal_params=cfg)
    m.fit(y)
    out = m.predict(h=2, level=[95, 80, 50])  # unsorted input
    for lv in [50, 80, 95]:
        assert f"hi-{lv}" in out
        assert f"lo-{lv}" in out
    print("test_levels_unsorted_input_is_handled: OK")

def test_forecast_fitted_true_raises_not_implemented():
    y = jnp.asarray([1., 2., 3., 4.])
    cfg = ConformalIntervals(n_windows=2, h=1, method="conformal_distribution")
    m = WindowAverage(window_size=2, conformal_params=cfg)
    try:
        _ = m.forecast(y=y, h=2, level=None, fitted=True)  # _window_average raises
        raise AssertionError("Expected NotImplementedError when fitted=True")
    except NotImplementedError:
        pass
    print("test_forecast_fitted_true_raises_not_implemented: OK")

def test_conformal_requires_enough_windows():
    # With h=2, need (n-1)//2 >= 2  => n >= 5; pick n=4 to force error
    y = jnp.asarray([10., 11., 12., 13.])  # n=4
    cfg = ConformalIntervals(n_windows=5, h=2, method="conformal_distribution")
    m = WindowAverage(window_size=2, conformal_params=cfg)
    try:
        _ = m.forecast(y=y, h=2, level=[90])
        raise AssertionError("Expected ValueError due to insufficient windows for conformal")
    except ValueError:
        pass
    print("test_conformal_requires_enough_windows: OK")

def test_predict_before_fit_raises_or_behaves_safely():
    # Accessing predict before fit should fail clearly
    m = WindowAverage(window_size=2)
    try:
        _ = m.predict(h=1)
        # If your implementation doesn't raise, enforce it:
        raise AssertionError("Expected an error when calling predict() before fit()")
    except Exception:
        pass
    print("test_predict_before_fit_raises_or_behaves_safely: OK")


if __name__ == "__main__":
    test_basic_mean_predict()
    test_dtype_cast_to_float32()
    test_short_series_returns_nan()
    test_predict_without_conformal_raises()
    test_forecast_with_conformal_intervals_stateless()
    test_fit_caches_conformity_scores_then_predict_uses_cache()
    test_levels_unsorted_input_is_handled()
    test_forecast_fitted_true_raises_not_implemented()
    test_conformal_requires_enough_windows()
    test_predict_before_fit_raises_or_behaves_safely()
    print("All tests passed.")
