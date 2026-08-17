"""Tests for MSTL (multiple-seasonal-trend decomposition forecaster).

These pin the decomposition (reconstruction), forecast shape/sanity,
trend-tracking, multi-period, the vmap/conformal contract, and pickle.

MSTL's accuracy rests on the STL decomposition in `chronax/models/stl.py`, where
the seasonal is the canonical ``C - lowpass(C)``: a phase-mean centering instead
absorbs the trend into the seasonal and leaves the trend undershooting at the
boundary. A residual boundary trend-lag remains — chronax's final trend value sits
slightly below statsmodels' on a clean series — so the bounds below are regression
guards rather than targets.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronax.models import MSTL
from chronax.utils.conformal_intervals import ConformalIntervals


def _seasonal_trend(n=240, m=12, slope=0.3, amp=8.0, noise=0.5, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    y = 10.0 + slope * t + amp * np.sin(2 * np.pi * t / m) + noise * rng.randn(n)
    return jnp.asarray(y)


def _mase(actual, fcst, train, m):
    actual = np.asarray(actual); fcst = np.asarray(fcst); train = np.asarray(train)
    denom = np.mean(np.abs(train[m:] - train[:-m]))
    return float(np.mean(np.abs(actual - fcst)) / denom)


def test_decomposition_reconstructs_series():
    """trend + sum(seasonals) + remainder must reconstruct the input series."""
    y = _seasonal_trend()
    m = MSTL(period=12).fit(y)
    trend = np.asarray(m.model_["trend"])
    remainder = np.asarray(m.model_["remainder"])
    seas = np.asarray(m.model_["seasonals"])
    recon = trend + seas.sum(axis=0) + remainder
    assert np.allclose(recon, np.asarray(y), atol=1e-6), "decomposition does not reconstruct y"


def test_forecast_shape_and_finite():
    y = _seasonal_trend()
    out = MSTL(period=12).fit(y).predict(h=24)
    assert out["mean"].shape == (24,)
    assert np.all(np.isfinite(np.asarray(out["mean"])))


def test_beats_seasonal_naive_on_trend_plus_seasonal():
    """On a clean trend+seasonal series MSTL must beat a seasonal-naive baseline.

    Guards the canonical-STL seasonal ``C - lowpass(C)``: a seasonal that absorbs
    the trend, or a trend that undershoots at the boundary, loses to seasonal-naive
    on exactly this shape of series. The residual boundary trend-lag keeps MSTL above
    the ~0.2 a clean series would otherwise allow, so 1.45 is a loose regression
    guard rather than a target.
    """
    y = _seasonal_trend(n=240, m=12, slope=0.3, amp=8.0)
    tr, te = y[:-24], y[-24:]
    fc = np.asarray(MSTL(period=12).forecast(y=tr, h=24)["mean"])
    tr_np = np.asarray(tr)
    snaive = np.resize(tr_np[-12:], 24)
    mase_mstl = _mase(te, fc, tr, 12)
    mase_snaive = _mase(te, snaive, tr, 12)
    assert mase_mstl < mase_snaive, f"MSTL {mase_mstl:.3f} !< seasonal-naive {mase_snaive:.3f}"
    assert mase_mstl < 1.45, f"MSTL MASE {mase_mstl:.3f} regressed above the decomposition-fix bound"


def test_forecast_tracks_trend():
    """The seasonally-adjusted forecast must follow the trend, not stay flat.

    Guards the Phase-4 fix: a strong upward trend under seasonality should make
    the 24-step-ahead mean climb above the last observed level.
    """
    y = _seasonal_trend(n=240, m=12, slope=0.5, amp=5.0, noise=0.1)
    fc = np.asarray(MSTL(period=12).fit(y).predict(h=24)["mean"])
    last_level = float(np.mean(np.asarray(y)[-12:]))
    assert np.mean(fc[-12:]) > last_level, "forecast did not continue the upward trend"


def test_multi_period():
    """MSTL supports multiple seasonal periods."""
    rng = np.random.RandomState(1)
    t = np.arange(24 * 30, dtype=np.float64)
    y = jnp.asarray(
        50.0 + 0.01 * t
        + 5.0 * np.sin(2 * np.pi * t / 24)      # daily
        + 3.0 * np.sin(2 * np.pi * t / (24 * 7))  # weekly
        + 0.5 * rng.randn(t.size)
    )
    m = MSTL(period=[24, 24 * 7]).fit(y)
    assert m.model_["seasonals"].shape[0] == 2
    out = m.predict(h=48)
    assert out["mean"].shape == (48,) and np.all(np.isfinite(np.asarray(out["mean"])))


def test_conformity_scores_under_vmap():
    """The base conformity_scores vmaps forecast over CV windows — must trace
    (the sub-forecaster on the seasonally-adjusted series is vmap-native)."""
    y = _seasonal_trend()
    m = MSTL(period=12, conformal_params=ConformalIntervals(n_windows=2, h=6, method="conformal_distribution"))
    cs = m.conformity_scores(y)
    assert cs.shape == (2, 6)
    assert np.all(np.isfinite(np.asarray(cs)))
    # per-window scores must differ (forecast honestly re-fits each window)
    assert not bool(jnp.allclose(cs[0], cs[1]))


def test_predict_level_twice_then_pickle():
    """Interval branch twice + pickle round-trip (tracer-pollution guard)."""
    y = _seasonal_trend()
    m = MSTL(period=12, conformal_params=ConformalIntervals(n_windows=2, h=6, method="conformal_distribution"))
    m.fit(y)
    p1 = m.predict(h=6, level=[80])
    p2 = m.predict(h=6, level=[80])
    for p in (p1, p2):
        assert "lo-80" in p and "hi-80" in p
        assert bool(jnp.all(p["lo-80"] <= p["hi-80"]))
    m2 = pickle.loads(pickle.dumps(m))
    p3 = m2.predict(h=6, level=[80])
    assert "lo-80" in p3 and "hi-80" in p3


def test_no_seasonality_smooth_trend():
    """period=1 (or a flat series) still produces a finite forecast."""
    y = jnp.asarray(10.0 + 0.2 * np.arange(120, dtype=np.float64))
    out = MSTL(period=1).fit(y).predict(h=12)
    assert out["mean"].shape == (12,) and np.all(np.isfinite(np.asarray(out["mean"])))


def test_predict_in_sample_level_no_broadcast_crash():
    # Regression: predict_in_sample(level) previously applied (n_windows, h)
    # CV conformity scores to the (n,) fitted mean and crashed. Native fitted
    # intervals come from the remainder scale and match the series length.
    rng = np.random.default_rng(0)
    n = 120
    t = np.arange(n)
    y = jnp.asarray(50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, n))
    m = MSTL(period=12).fit(y)
    out = m.predict_in_sample(level=[80, 95])
    for k in ("fitted-lo-80", "fitted-hi-80", "fitted-lo-95", "fitted-hi-95"):
        assert k in out
        assert np.asarray(out[k]).shape == (n,)
    # Ordering: wider level fully contains the narrower one.
    assert np.all(np.asarray(out["fitted-lo-95"]) <= np.asarray(out["fitted-lo-80"]))
    assert np.all(np.asarray(out["fitted-hi-80"]) <= np.asarray(out["fitted-hi-95"]))
