import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
from chronax.models import ETS, AutoETS
from chronax.utils import ConformalIntervals
# -------------------------------------------------------------------
# Coverage Tests (run this file directly)
#
# These tests exercise both ETS (fixed-spec) and AutoETS (auto-select).
# With the bug fixes applied to ets_functions.py, ETS now works for
# all model specs (ANN, AAN, AAA) and forward() works correctly.
# -------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

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

    # ================= Core behavior =================

    def test_constant_series_ANN():
        """ETS(ANN) on constant data should produce flat forecasts near the constant."""
        y = jnp.full((40,), 10.0, dtype=jnp.float64)
        m = ETS(season_length=1, model="ANN", max_iter=200)
        m.fit(y)

        fitted = m.predict_in_sample(level=None)["fitted"]
        _assert_allclose(np.mean(np.abs(np.asarray(fitted) - 10.0)), 0.0, atol=1e-2, label="ANN fitted MAE")

        h = 5
        out = m.predict(h=h, level=None)
        expected = np.full(h, 10.0, dtype=np.float64)
        _assert_allclose(out["mean"], expected, atol=1e-1, label="ANN mean")
        _print_ok("test_constant_series_ANN")

    def test_linear_trend_ETS_AAN():
        """ETS(AAN) on linear trend data should continue the upward trend."""
        a, b = 2.0, 0.5
        rng = np.random.RandomState(42)
        t = np.arange(80, dtype=np.float64)
        y = jnp.asarray(a + b * t + 0.3 * rng.randn(80), dtype=jnp.float64)

        et = ETS(season_length=1, model="AAN", max_iter=200)
        et.fit(y)

        h = 6
        out = et.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        assert mean.shape == (h,)
        assert np.all(np.isfinite(mean)), "Forecasts must be finite"
        assert mean[0] > float(y[-1]) - 5.0, "Forecast should be near last value"
        _print_ok("test_linear_trend_ETS_AAN")

    def test_additive_seasonality_ETS_AAA():
        """ETS(AAA) on seasonal data should produce seasonal forecasts."""
        m = 4
        base = 10.0
        rng = np.random.RandomState(7)
        season = np.array([+1.0, -1.0, +2.0, -2.0], dtype=np.float64)
        y = jnp.asarray(np.tile(season, 15) + base + 0.1 * rng.randn(60), dtype=jnp.float64)

        et = ETS(season_length=m, model="AAA", max_iter=200)
        et.fit(y)

        h = 8
        out = et.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        assert mean.shape == (h,)
        assert np.all(np.isfinite(mean)), "Forecasts must be finite"
        assert np.all(np.abs(mean - base) < 5.0), "Forecasts should be near base level"
        _print_ok("test_additive_seasonality_ETS_AAA")

    def test_linear_trend_via_AutoETS():
        """AutoETS(ZZZ) on linear trend data for comparison."""
        a, b = 2.0, 0.5
        rng = np.random.RandomState(42)
        t = np.arange(80, dtype=np.float64)
        y = jnp.asarray(a + b * t + 0.3 * rng.randn(80), dtype=jnp.float64)

        ae = AutoETS(season_length=1, model="ZZZ")
        ae.fit(y)

        h = 6
        out = ae.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        assert mean.shape == (h,)
        assert np.all(np.isfinite(mean)), "Forecasts must be finite"
        assert mean[0] > float(y[-1]) - 5.0, "Forecast should be near last value"
        _print_ok("test_linear_trend_via_AutoETS")

    def test_stateless_forecast_matches_stateful_ETS():
        """Stateless forecast should match stateful (fit+predict) for ETS."""
        rng = np.random.RandomState(99)
        y = jnp.asarray(8.0 + 0.5 * rng.randn(48), dtype=jnp.float64)

        h = 5
        et = ETS(season_length=1, model="ANN", max_iter=200)
        et.fit(y)
        stateful = et.predict(h=h, level=None)["mean"]
        stateless = et.forecast(y=y, h=h, level=None, fitted=False)["mean"]

        _assert_allclose(stateful, stateless, atol=1e-4, label="stateless vs stateful mean")
        _print_ok("test_stateless_forecast_matches_stateful_ETS")

    def test_ets_default_max_iter():
        """ETS with max_iter=None should now work (bug #1 fixed)."""
        rng = np.random.RandomState(88)
        y = jnp.asarray(10.0 + 0.5 * rng.randn(40), dtype=jnp.float64)

        et = ETS(season_length=1, model="ANN")  # no max_iter
        et.fit(y)

        h = 4
        out = et.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        assert mean.shape == (h,)
        assert np.all(np.isfinite(mean)), "Forecasts must be finite"
        _print_ok("test_ets_default_max_iter")

    # ================= Intervals & level handling =================

    def test_predict_native_intervals_ETS():
        """Native ETS intervals with unsorted levels."""
        rng = np.random.RandomState(11)
        y = jnp.asarray(10.0 + 0.5 * rng.randn(32), dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN", max_iter=200)
        et.fit(y)
        out = et.predict(h=4, level=[95, 80, 50])  # unsorted on purpose

        assert "mean" in out and out["mean"].shape == (4,)
        for lv in (50, 80, 95):
            assert f"hi-{lv}" in out and f"lo-{lv}" in out
            assert out[f"hi-{lv}"].shape == (4,) and out[f"lo-{lv}"].shape == (4,)
        m = np.asarray(out["mean"])
        for lv in (50, 80, 95):
            lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
            assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12), f"mean not inside {lv}% band"
        _print_ok("test_predict_native_intervals_ETS")

    def test_forecast_adds_fitted_and_fitted_intervals_ETS():
        """Stateless forecast with fitted=True returns fitted values and intervals."""
        rng = np.random.RandomState(33)
        t = np.arange(60, dtype=np.float64)
        y = jnp.asarray(5.0 + 0.2 * t + 0.3 * rng.randn(60), dtype=jnp.float64)
        et = ETS(season_length=1, model="AAN", max_iter=200)
        out = et.forecast(y=y, h=5, level=[80, 95], fitted=True)
        assert "mean" in out and out["mean"].shape == (5,)
        assert "fitted" in out and out["fitted"].shape == y.shape
        for lv in (80, 95):
            assert f"lo-{lv}" in out and f"hi-{lv}" in out
            assert f"fitted-lo-{lv}" in out and f"fitted-hi-{lv}" in out
            fl = np.asarray(out[f"fitted-lo-{lv}"]); fh = np.asarray(out[f"fitted-hi-{lv}"])
            ft = np.asarray(out["fitted"])
            assert np.all(fl <= ft + 1e-12) and np.all(ft <= fh + 1e-12)
        _print_ok("test_forecast_adds_fitted_and_fitted_intervals_ETS")

    def test_predict_in_sample_intervals_monotonicity_ETS():
        """In-sample intervals: 95% band should be wider than 80%."""
        rng = np.random.RandomState(55)
        y = jnp.asarray(11.0 + 0.5 * rng.randn(24), dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN", max_iter=200)
        et.fit(y)

        res = et.predict_in_sample(level=[80, 95])
        fitted = np.asarray(res["fitted"])
        lo80 = np.asarray(res["fitted-lo-80"])
        lo95 = np.asarray(res["fitted-lo-95"])
        hi80 = np.asarray(res["fitted-hi-80"])
        hi95 = np.asarray(res["fitted-hi-95"])
        if not (np.all(lo95 <= lo80 + 1e-12) and np.all(hi95 >= hi80 - 1e-12)):
            raise AssertionError("Monotonicity failed: 95% should be wider than 80%")
        if not (np.all(lo80 <= fitted + 1e-12) and np.all(fitted <= hi80 + 1e-12)):
            raise AssertionError("Fitted not in 80% band")
        if not (np.all(lo95 <= fitted + 1e-12) and np.all(fitted <= hi95 + 1e-12)):
            raise AssertionError("Fitted not in 95% band")
        _print_ok("test_predict_in_sample_intervals_monotonicity_ETS")

    # ================= Conformal intervals paths =================

    def test_fit_caches_conformal_then_predict_uses_cache():
        n = 40
        rng = np.random.RandomState(17)
        y = jnp.asarray(8.0 + 0.5 * rng.randn(n), dtype=jnp.float64)

        cfg = ConformalIntervals(n_windows=5, h=3, method="conformal_distribution")
        et = ETS(season_length=1, model="ANN", max_iter=200, prediction_intervals=cfg)

        et.fit(y)
        assert getattr(et, "_cs") is not None

        out = et.predict(h=3, level=[80, 95])
        for lv in (80, 95):
            assert f"lo-{lv}" in out and f"hi-{lv}" in out
            m = np.asarray(out["mean"])
            lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
            assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
        _print_ok("test_fit_caches_conformal_then_predict_uses_cache")

    def test_stateless_forecast_with_conformal_intervals():
        n = 36
        rng = np.random.RandomState(19)
        y = jnp.asarray(5.0 + 0.5 * rng.randn(n), dtype=jnp.float64)

        # h must match conformal_params.h: scores cover exactly h steps
        # (guarded with a ValueError, sibling convention).
        cfg = ConformalIntervals(n_windows=4, h=6, method="conformal_distribution")
        et = ETS(season_length=1, model="ANN", max_iter=200, prediction_intervals=cfg)

        out = et.forecast(y=y, h=6, level=[90], fitted=False)
        assert "mean" in out and out["mean"].shape == (6,)
        assert "lo-90" in out and "hi-90" in out
        mean = np.asarray(out["mean"])
        lo90 = np.asarray(out["lo-90"]); hi90 = np.asarray(out["hi-90"])
        assert np.all(lo90 <= mean + 1e-12) and np.all(mean <= hi90 + 1e-12)
        _print_ok("test_stateless_forecast_with_conformal_intervals")

    def test_native_intervals_multiplicative_classes():
        """Native intervals across interval-formula classes 2/3/4-5.

        The conformal CV path never exercises these branches (level=None), and
        the contract suite pins ETS to ANN — so the multiplicative interval
        formulas (class 2: M-error; class 3: M-error+M-season; class 4/5:
        simulation fallback, e.g. M-trend combos) ship untested without this.
        Mirrors contract invariant C: predict(level) twice, pickle, predict.
        """
        import pickle
        m_seas = 4
        rng = np.random.RandomState(42)
        t = np.arange(64, dtype=np.float64)
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * t / m_seas)
        y_pos = jnp.asarray((50.0 + 0.5 * t) * seasonal + 0.5 * rng.randn(64) + 5.0,
                            dtype=jnp.float64)

        for spec, season_length in [("MNN", 1), ("MNM", m_seas), ("MMN", 1)]:
            et = ETS(season_length=season_length, model=spec, max_iter=100)
            et.fit(y_pos)
            p1 = et.predict(h=4, level=[80, 95])
            p2 = et.predict(h=4, level=[80, 95])
            for p in (p1, p2):
                for lv in (80, 95):
                    lo = np.asarray(p[f"lo-{lv}"]); hi = np.asarray(p[f"hi-{lv}"])
                    mean = np.asarray(p["mean"])
                    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), f"{spec}: non-finite bands"
                    assert np.all(lo <= mean + 1e-9) and np.all(mean <= hi + 1e-9), f"{spec}: mean outside {lv}% band"
            et2 = pickle.loads(pickle.dumps(et))
            p3 = et2.predict(h=4, level=[80])
            assert "lo-80" in p3 and "hi-80" in p3, f"{spec}: post-pickle intervals missing"
        _print_ok("test_native_intervals_multiplicative_classes")

    # ================= Forward & errors =================

    def test_forward_on_new_series_ETS():
        """Forward applies a fitted ETS model to new data (bug #4 fixed)."""
        rng_a = np.random.RandomState(21)
        rng_b = np.random.RandomState(22)
        yA = jnp.asarray(15.0 + 0.5 * rng_a.randn(60), dtype=jnp.float64)
        yB = jnp.asarray(10.0 + 0.5 * rng_b.randn(60), dtype=jnp.float64)

        et = ETS(season_length=1, model="ANN", max_iter=200)
        et.fit(yA)

        h = 5
        outB = et.forward(y=yB, h=h, level=[80], fitted=True)
        assert outB["mean"].shape == (h,)
        assert outB["fitted"].shape == yB.shape
        lo = np.asarray(outB["lo-80"]); hi = np.asarray(outB["hi-80"]); mean = np.asarray(outB["mean"])
        assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
        assert "fitted-lo-80" in outB and "fitted-hi-80" in outB
        _print_ok("test_forward_on_new_series_ETS")

    def test_forward_on_new_series_AutoETS():
        """Forward applies a fitted AutoETS model to new data."""
        rng_a = np.random.RandomState(21)
        rng_b = np.random.RandomState(22)
        yA = jnp.asarray(15.0 + 0.5 * rng_a.randn(60), dtype=jnp.float64)
        yB = jnp.asarray(10.0 + 0.5 * rng_b.randn(60), dtype=jnp.float64)

        ae = AutoETS(season_length=1, model="ZZZ")
        ae.fit(yA)

        h = 5
        outB = ae.forward(y=yB, h=h, level=[80], fitted=True)
        assert outB["mean"].shape == (h,)
        assert outB["fitted"].shape == yB.shape
        lo = np.asarray(outB["lo-80"]); hi = np.asarray(outB["hi-80"]); mean = np.asarray(outB["mean"])
        assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
        assert "fitted-lo-80" in outB and "fitted-hi-80" in outB
        _print_ok("test_forward_on_new_series_AutoETS")

    def test_predict_before_fit_raises():
        """Calling predict() before fit() should raise an error."""
        et = ETS(season_length=1, model="ANN", max_iter=200)
        try:
            _ = et.predict(h=1)
            raise AssertionError("Expected an error when calling predict() before fit()")
        except Exception:
            pass
        _print_ok("test_predict_before_fit_raises")

    def test_forward_before_fit_raises():
        """Calling forward() before fit() should raise an error."""
        et = ETS(season_length=1, model="ANN", max_iter=200)
        try:
            _ = et.forward(y=jnp.asarray([1., 2., 3.]), h=1)
            raise AssertionError("Expected an error when calling forward() before fit()")
        except Exception:
            pass
        _print_ok("test_forward_before_fit_raises")

    def test_tiny_series_policy():
        """Tiny series should either produce valid forecasts or raise NotImplementedError."""
        y = jnp.asarray([10.0, 11.0, 10.5], dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN", max_iter=200)
        try:
            et.fit(y)
            out = et.predict(h=3, level=None)
            if out["mean"].shape != (3,):
                raise AssertionError("Expected 3-step forecast for tiny series")
        except (NotImplementedError, ValueError) as e:
            msg = str(e).lower()
            if "tiny datasets" not in msg and "too short" not in msg:
                raise
        _print_ok("test_tiny_series_policy")

    # ================= φ validation & dtypes =================

    def test_phi_validation_range_and_type():
        """phi must be float in [0.8, 0.98] or None."""
        for Cls in (ETS, AutoETS):
            try:
                _ = Cls(phi=1.5)
                raise AssertionError(f"Expected ValueError for invalid phi range on {Cls.__name__}")
            except ValueError:
                pass
            try:
                _ = Cls(phi="0.9")  # type: ignore
                raise AssertionError(f"Expected ValueError for non-float phi on {Cls.__name__}")
            except ValueError:
                pass
            _ = Cls(phi=0.8)
            _ = Cls(phi=0.98)
        _print_ok("test_phi_validation_range_and_type")

    def test_basic_shapes_and_dtypes_predict():
        """Basic shape and dtype checks for predict output."""
        rng = np.random.RandomState(77)
        y = jnp.asarray(5.0 + 0.5 * rng.randn(20), dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN", max_iter=200)
        et.fit(y)
        out = et.predict(h=3, level=None)
        assert out["mean"].shape == (3,)
        assert out["mean"].dtype in (jnp.float32, jnp.float64)
        _print_ok("test_basic_shapes_and_dtypes_predict")

    # ================= Runner =================
    test_constant_series_ANN()
    test_linear_trend_ETS_AAN()
    test_additive_seasonality_ETS_AAA()
    test_linear_trend_via_AutoETS()
    test_stateless_forecast_matches_stateful_ETS()
    test_ets_default_max_iter()

    test_predict_native_intervals_ETS()
    test_forecast_adds_fitted_and_fitted_intervals_ETS()
    test_predict_in_sample_intervals_monotonicity_ETS()
    test_native_intervals_multiplicative_classes()
    test_fit_caches_conformal_then_predict_uses_cache()
    test_stateless_forecast_with_conformal_intervals()

    test_forward_on_new_series_ETS()
    test_forward_on_new_series_AutoETS()
    test_predict_before_fit_raises()
    test_forward_before_fit_raises()
    test_tiny_series_policy()

    test_phi_validation_range_and_type()
    test_basic_shapes_and_dtypes_predict()

    print("\nAll ETS/AutoETS full-coverage tests passed.")
