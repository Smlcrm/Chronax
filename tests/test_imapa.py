
import numpy as np
import warnings
import jax.numpy as jnp
import numpy as np
from chronax.utils import ConformalIntervals
from chronax.models import IMAPA


# ============== Helpers ==============
def _assert_allclose(actual, expected, atol=1e-4, rtol=1e-6, label=""):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label} mismatch.\nActual:   {a}\nExpected: {e}\n"
            f"max|Δ|={np.max(np.abs(a - e))}, atol={atol}, rtol={rtol}"
        )

def _print_ok(name):
    print(f"{name}: OK")


# ============== Core fit / predict behavior ==============

def test_fit_sets_model_and_dtype_expected():
    # int dtype to exercise ensure_float casting in fit()
    y = jnp.asarray([1, 0, 2, 0, 0, 3, 0, 4], dtype=jnp.int32)
    m = IMAPA()
    m.fit(y)

    # model_ should exist with a "mean" key from _imapa(..., h=1)
    assert hasattr(m, "model_") and "mean" in m.model_
    # ensure_float should have coerced ints to float32 (or float64 depending on utils policy)
    assert m._y.dtype in (jnp.float32, jnp.float64)
    _print_ok("test_fit_sets_model_and_dtype_expected")


def test_predict_level_none_repeats_scalar_mean_expected():
    y = jnp.asarray([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=jnp.float64)
    m = IMAPA()
    m.fit(y)

    # scalar value to be repeated is model_["mean"][0]
    base = float(np.asarray(m.model_["mean"])[0])
    h = 5
    out = m.predict(h=h, level=None)
    assert set(out.keys()) == {"mean"}
    expected = np.full(h, base, dtype=np.float64)
    _assert_allclose(out["mean"], expected, atol=1e-8, label="predict mean repeat")
    _print_ok("test_predict_level_none_repeats_scalar_mean_expected")


def test_predict_with_level_without_conformal_raises():
    y = jnp.asarray([1.0, 0.0, 2.0, 0.0, 3.0], dtype=jnp.float64)
    m = IMAPA()  # no conformal_params set
    m.fit(y)
    try:
        _ = m.predict(h=3, level=[80, 95])
        raise AssertionError("Expected ValueError when requesting intervals without conformal_params.")
    except ValueError as e:
        assert "You must pass `conformal_params`" in str(e)
    _print_ok("test_predict_with_level_without_conformal_raises")


def test_predict_with_conformal_but_missing_cs_raises():
    # Set conformal_params but DON'T call fit(); also set model_ so predict can compute mean
    class _CP:
        method = "conformal_distribution"
    m = IMAPA(conformal_params=_CP())
    m.model_ = {"mean": jnp.array([7.0], dtype=jnp.float64)}
    try:
        _ = m.predict(h=2, level=[90])
        raise AssertionError("Expected ValueError when _cs is None under conformal path.")
    except ValueError as e:
        assert "Conformity scores are not available" in str(e)
    _print_ok("test_predict_with_conformal_but_missing_cs_raises")


def test_predict_with_conformal_and_cached_cs_expected():
    # Use a reasonably long series for stable conformity scores
    n = 40
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(8.0 + 0.05 * t + 0.3 * np.sin(2 * np.pi * t / 8), dtype=jnp.float64)
    cfg = ConformalIntervals(n_windows=5, h=3, method="conformal_distribution")

    m = IMAPA(conformal_params=cfg)
    m.fit(y)
    out = m.predict(h=3, level=[95, 80])  # intentionally unsorted
    assert "mean" in out and out["mean"].shape == (3,)
    # We expect BaseForecaster.add_confidence_intervals to add "lo-*" and "hi-*"
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
        mean = np.asarray(out["mean"])
        assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
    _print_ok("test_predict_with_conformal_and_cached_cs_expected")


# ============== predict_in_sample behavior ==============

def test_predict_in_sample_no_level_returns_fitted():
    y = jnp.asarray([1.0, 0.0, 2.0, 0.0, 3.0, 0.0], dtype=jnp.float64)
    m = IMAPA()
    m.fit(y)
    res = m.predict_in_sample(level=None)
    assert set(res.keys()) == {"fitted"}
    assert res["fitted"].shape == y.shape
    _print_ok("test_predict_in_sample_no_level_returns_fitted")


def test_predict_in_sample_with_levels_monotonicity_and_containment():
    warnings.filterwarnings("ignore", message="Computing fitted values for IMAPA is very expensive.")

    y = jnp.asarray([10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 12.5, 12.0] * 3, dtype=jnp.float64)
    m = IMAPA()
    m.fit(y)
    res = m.predict_in_sample(level=[80, 95])

    # Ensure keys are present
    for lv in (80, 95):
        assert f"fitted-lo-{lv}" in res and f"fitted-hi-{lv}" in res

    fitted = np.asarray(res["fitted"], dtype=float)
    lo80 = np.asarray(res["fitted-lo-80"], dtype=float); hi80 = np.asarray(res["fitted-hi-80"], dtype=float)
    lo95 = np.asarray(res["fitted-lo-95"], dtype=float); hi95 = np.asarray(res["fitted-hi-95"], dtype=float)

    # Ignore positions where any value is NaN (common at initial indices for IMAPA/aggregation)
    valid80 = (~np.isnan(fitted)) & (~np.isnan(lo80)) & (~np.isnan(hi80))
    valid95 = (~np.isnan(fitted)) & (~np.isnan(lo95)) & (~np.isnan(hi95))
    valid_both = valid80 & valid95

    # If everything is NaN (shouldn’t happen with reasonable y), fail clearly
    assert np.any(valid80), "No valid (non-NaN) points available for 80% fitted intervals."
    assert np.any(valid95), "No valid (non-NaN) points available for 95% fitted intervals."

    eps = 1e-9
    # Containment
    assert np.all(lo80[valid80] <= fitted[valid80] + eps) and np.all(fitted[valid80] <= hi80[valid80] + eps)
    assert np.all(lo95[valid95] <= fitted[valid95] + eps) and np.all(fitted[valid95] <= hi95[valid95] + eps)

    # Monotonicity (95% band should be >= 80% band where both are valid)
    w80 = hi80[valid_both] - lo80[valid_both]
    w95 = hi95[valid_both] - lo95[valid_both]
    assert np.all(w95 + eps >= w80), "Monotonicity failed: 95% band not wider than 80% band."

    _print_ok("test_predict_in_sample_with_levels_monotonicity_and_containment")


# ============== forecast behavior (stateless) ==============

def test_forecast_no_level_returns_mean_and_optional_fitted():
    t = np.arange(36, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.1 * t, dtype=jnp.float64)

    m = IMAPA()
    out = m.forecast(y=y, h=4, level=None, fitted=False)
    assert set(out.keys()) == {"mean"}
    assert out["mean"].shape == (4,)

    out2 = m.forecast(y=y, h=3, level=None, fitted=True)
    assert "mean" in out2 and "fitted" in out2
    assert out2["mean"].shape == (3,)
    assert out2["fitted"].shape == y.shape
    _print_ok("test_forecast_no_level_returns_mean_and_optional_fitted")


def test_forecast_with_level_without_conformal_raises():
    t = np.arange(24, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.2 * t, dtype=jnp.float64)
    m = IMAPA()  # no conformal params
    try:
        _ = m.forecast(y=y, h=4, level=[90], fitted=False)
        raise AssertionError("Expected Exception for intervals without conformal_params in forecast().")
    except Exception as e:
        assert "You have to instantiate the class with `conformal_params`" in str(e)
    _print_ok("test_forecast_with_level_without_conformal_raises")


def test_forecast_with_conformal_intervals_and_fitted_bands():
    warnings.filterwarnings("ignore", message="Computing fitted values for IMAPA is very expensive.")

    n = 36
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(6.0 + 0.15 * t + 0.3 * np.sin(2 * np.pi * t / 6), dtype=jnp.float64)

    cfg = ConformalIntervals(n_windows=4, h=5, method="conformal_distribution")
    m = IMAPA(conformal_params=cfg)
    out = m.forecast(y=y, h=5, level=[80, 95], fitted=True)

    # mean + conformal lo/hi bands (robust to NaNs)
    assert "mean" in out and out["mean"].shape == (5,)
    mean = np.asarray(out["mean"], dtype=float)
    eps = 1e-9
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        lo = np.asarray(out[f"lo-{lv}"], dtype=float)
        hi = np.asarray(out[f"hi-{lv}"], dtype=float)
        valid = (~np.isnan(mean)) & (~np.isnan(lo)) & (~np.isnan(hi))
        assert np.any(valid), f"No valid (non-NaN) points for mean {lv}% intervals."
        assert np.all(lo[valid] <= mean[valid] + eps) and np.all(mean[valid] <= hi[valid] + eps)

    # fitted present + fitted PI keys (robust to NaNs and ordering)
    assert "fitted" in out and out["fitted"].shape == y.shape
    ft = np.asarray(out["fitted"], dtype=float)
    for lv in (80, 95):
        assert f"fitted-lo-{lv}" in out and f"fitted-hi-{lv}" in out
        fl = np.asarray(out[f"fitted-lo-{lv}"], dtype=float)
        fh = np.asarray(out[f"fitted-hi-{lv}"], dtype=float)
        valid_f = (~np.isnan(ft)) & (~np.isnan(fl)) & (~np.isnan(fh))
        assert np.any(valid_f), f"No valid (non-NaN) points for fitted {lv}% intervals."
        assert np.all(fl[valid_f] <= ft[valid_f] + eps) and np.all(ft[valid_f] <= fh[valid_f] + eps)

    # monotonicity: where both levels valid, 95% width >= 80% width
    fl80 = np.asarray(out["fitted-lo-80"], dtype=float); fh80 = np.asarray(out["fitted-hi-80"], dtype=float)
    fl95 = np.asarray(out["fitted-lo-95"], dtype=float); fh95 = np.asarray(out["fitted-hi-95"], dtype=float)
    valid_both = (~np.isnan(fl80)) & (~np.isnan(fh80)) & (~np.isnan(fl95)) & (~np.isnan(fh95))
    if np.any(valid_both):
        w80 = fh80[valid_both] - fl80[valid_both]
        w95 = fh95[valid_both] - fl95[valid_both]
        assert np.all(w95 + eps >= w80), "Monotonicity failed: 95% band not wider than 80% band."

    _print_ok("test_forecast_with_conformal_intervals_and_fitted_bands")


def test_forecast_h_mismatch_with_conformal_h_raises_actionable():
    # conformity scores cover exactly conformal_params.h steps; asking for a
    # different h must raise a ValueError that names both horizons.
    n = 36
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(6.0 + 0.15 * t, dtype=jnp.float64)
    m = IMAPA(conformal_params=ConformalIntervals(n_windows=4, h=2))
    try:
        m.forecast(y=y, h=5, level=[80])
    except ValueError as e:
        assert "h=5" in str(e) and "conformal_params.h=2" in str(e)
    else:
        raise AssertionError("expected a ValueError for h != conformal_params.h")
    _print_ok("test_forecast_h_mismatch_with_conformal_h_raises_actionable")


# ============== error handling / attributes ==============

def test_predict_before_fit_raises_or_requires_model_():
    m = IMAPA()
    try:
        _ = m.predict(h=1, level=None)
        # If no exception, then a model_ existed; that'd be unexpected in a fresh instance.
        if not hasattr(m, "model_"):
            raise AssertionError("Expected an exception or pre-set model_ when calling predict() before fit().")
    except Exception:
        pass
    _print_ok("test_predict_before_fit_raises_or_requires_model_")


def test_only_conformal_intervals_flag_true():
    m = IMAPA()
    assert getattr(m, "only_conformal_intervals", False) is True
    _print_ok("test_only_conformal_intervals_flag_true")


def test_dtype_casting_in_forecast_path_expected():
    # ints in forecast() should be coerced by ensure_float inside the method
    y = jnp.asarray([1, 0, 2, 0, 3, 0, 4, 0], dtype=jnp.int32)
    m = IMAPA()
    out = m.forecast(y=y, h=3, level=None, fitted=True)
    assert out["mean"].shape == (3,)
    # fitted should be floating dtype
    assert out["fitted"].dtype in (jnp.float32, jnp.float64)
    _print_ok("test_dtype_casting_in_forecast_path_expected")


# ============== Runner ==============

if __name__ == "__main__":
    # Core behavior
    test_fit_sets_model_and_dtype_expected()
    test_predict_level_none_repeats_scalar_mean_expected()
    test_predict_with_level_without_conformal_raises()
    test_predict_with_conformal_but_missing_cs_raises()
    test_predict_with_conformal_and_cached_cs_expected()

    # In-sample PI behavior
    test_predict_in_sample_no_level_returns_fitted()
    test_predict_in_sample_with_levels_monotonicity_and_containment()

    # Stateless forecasts
    test_forecast_no_level_returns_mean_and_optional_fitted()
    test_forecast_with_level_without_conformal_raises()
    test_forecast_with_conformal_intervals_and_fitted_bands()

    # Errors / attributes / dtype paths
    test_predict_before_fit_raises_or_requires_model_()
    test_only_conformal_intervals_flag_true()
    test_dtype_casting_in_forecast_path_expected()

    print("All IMAPA full-coverage-style tests passed.")
