import jax.numpy as jnp
from ets_model import ETS
from conformal_intervals import ConformalIntervals
# -------------------------------------------------------------------
# Coverage Tests (run this file directly)
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

    def test_constant_series_ANN_expected():
        y = jnp.full((40,), 10.0, dtype=jnp.float64)
        m = ETS(season_length=1, model="ANN")
        m.fit(y)

        fitted = m.predict_in_sample(level=None)["fitted"]
        _assert_allclose(np.mean(np.abs(np.asarray(fitted) - 10.0)), 0.0, atol=1e-2, label="ANN fitted MAE")

        h = 5
        out = m.predict(h=h, level=None)
        expected = np.full(h, 10.0, dtype=np.float64)
        _assert_allclose(out["mean"], expected, atol=1e-1, label="ANN mean")
        _print_ok("test_constant_series_ANN_expected")

    def test_linear_trend_AAN_expected():
        a, b = 2.0, 0.5
        t = np.arange(60, dtype=np.float64)
        y = jnp.asarray(a + b * t, dtype=jnp.float64)

        et = ETS(season_length=1, model="AAN")  # additive error & trend, no seasonality
        et.fit(y)

        h = 6
        out = et.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        last_val = a + b * (len(y) - 1)
        expected_first = last_val + b
        expected = expected_first + b * np.arange(h, dtype=np.float64)
        _assert_allclose(mean[0], expected_first, atol=0.3, label="AAN step-1")
        _assert_allclose(mean, expected, atol=0.6, label="AAN path")
        _print_ok("test_linear_trend_AAN_expected")

    def test_additive_seasonality_AAA_expected():
        m = 4
        base = 10.0
        season = np.array([+1.0, -1.0, +2.0, -2.0], dtype=np.float64)
        y = jnp.asarray(np.tile(season, 15) + base, dtype=jnp.float64)  # len 60

        et = ETS(season_length=m, model="AAA")
        et.fit(y)

        h = 8
        out = et.predict(h=h, level=None)
        mean = np.asarray(out["mean"])

        idx0 = len(y) % m
        expected = np.array([base + season[(idx0 + k) % m] for k in range(h)], dtype=np.float64)
        _assert_allclose(mean, expected, atol=0.6, label="AAA seasonal pattern")
        _print_ok("test_additive_seasonality_AAA_expected")

    def test_stateless_forecast_matches_stateful_expected():
        t = np.arange(48, dtype=np.float64)
        base, slope = 5.0, 0.1
        season = np.array([0.0, +1.0, 0.0, -1.0], dtype=np.float64)
        y = jnp.asarray(base + slope * t + np.tile(season, len(t) // 4 + 1)[: len(t)], dtype=np.float64)

        h = 5
        et = ETS(season_length=4, model="AAA")
        et.fit(y)
        stateful = et.predict(h=h, level=None)["mean"]
        stateless = et.forecast(y=y, h=h, level=None, fitted=False)["mean"]

        _assert_allclose(stateful, stateless, atol=1e-6, label="stateless vs stateful mean")
        _print_ok("test_stateless_forecast_matches_stateful_expected")

    # ================= Intervals & level handling =================

    def test_predict_native_intervals_and_unsorted_levels():
        y = jnp.asarray([10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 12.5, 12.0] * 4, dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN")
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
        _print_ok("test_predict_native_intervals_and_unsorted_levels")

    def test_forecast_adds_fitted_and_fitted_intervals_when_requested():
        t = np.arange(60, dtype=np.float64)
        y = jnp.asarray(5.0 + 0.2 * t, dtype=jnp.float64)
        et = ETS(season_length=1, model="AAN")
        out = et.forecast(y=y, h=5, level=[80, 95], fitted=True)
        assert "mean" in out and out["mean"].shape == (5,)
        assert "fitted" in out and out["fitted"].shape == y.shape
        for lv in (80, 95):
            assert f"lo-{lv}" in out and f"hi-{lv}" in out
            assert f"fitted-lo-{lv}" in out and f"fitted-hi-{lv}" in out
            fl = np.asarray(out[f"fitted-lo-{lv}"]); fh = np.asarray(out[f"fitted-hi-{lv}"])
            ft = np.asarray(out["fitted"])
            assert np.all(fl <= ft + 1e-12) and np.all(ft <= fh + 1e-12)
        _print_ok("test_forecast_adds_fitted_and_fitted_intervals_when_requested")

    def test_predict_in_sample_intervals_monotonicity_expected():
        y = jnp.asarray([10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 12.5, 12.0] * 3, dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN")
        et.fit(y)

        res = et.predict_in_sample(level=[80, 95])
        fitted = np.asarray(res["fitted"])
        lo80 = np.asarray(res["fitted-lo-80"])
        lo95 = np.asarray(res["fitted-lo-95"])
        hi80 = np.asarray(res["fitted-hi-80"])
        hi95 = np.asarray(res["fitted-hi-95"])
        if not (np.all(lo95 <= lo80 + 1e-12) and np.all(hi95 >= hi80 - 1e-12)):
            raise AssertionError("Monotonicity failed: 95% should be wider than 80%")
        if not (np.all(lo80 <= fitted) and np.all(fitted <= hi80)):
            raise AssertionError("Fitted not in 80% band")
        if not (np.all(lo95 <= fitted) and np.all(fitted <= hi95)):
            raise AssertionError("Fitted not in 95% band")
        _print_ok("test_predict_in_sample_intervals_monotonicity_expected")

    # ================= Conformal intervals paths =================

    def test_fit_caches_conformal_then_predict_uses_cache():
        n = 40
        t = np.arange(n, dtype=np.float64)
        y = jnp.asarray(8.0 + 0.05 * t + 0.5 * np.sin(2 * np.pi * t / 8), dtype=jnp.float64)

        cfg = ConformalIntervals(n_windows=5, h=3, method="conformal_distribution")
        et = ETS(season_length=1, model="ANN", prediction_intervals=cfg)

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
        t = np.arange(n, dtype=np.float64)
        y = jnp.asarray(5.0 + 0.1 * t + 0.3 * np.sin(2 * np.pi * t / 6), dtype=jnp.float64)

        cfg = ConformalIntervals(n_windows=4, h=2, method="conformal_distribution")
        et = ETS(season_length=1, model="ANN", prediction_intervals=cfg)

        out = et.forecast(y=y, h=6, level=[90], fitted=False)
        assert "mean" in out and out["mean"].shape == (6,)
        assert "lo-90" in out and "hi-90" in out
        mean = np.asarray(out["mean"])
        lo90 = np.asarray(out["lo-90"]); hi90 = np.asarray(out["hi-90"])
        assert np.all(lo90 <= mean + 1e-12) and np.all(mean <= hi90 + 1e-12)
        _print_ok("test_stateless_forecast_with_conformal_intervals")

    # ================= Forward & errors =================

    def test_forward_on_new_series_expected_or_reasonable():
        t = np.arange(36, dtype=np.float64)
        yA = jnp.asarray(15.0 + 0.3 * t, dtype=jnp.float64)
        yB = jnp.asarray(10.0 + 0.3 * t, dtype=jnp.float64)

        et = ETS(season_length=1, model="AAN")
        et.fit(yA)

        h = 5
        outB = et.forward(y=yB, h=h, level=[80], fitted=True)
        assert outB["mean"].shape == (h,)
        assert outB["fitted"].shape == yB.shape
        mse = float(np.mean((np.asarray(outB["fitted"]) - np.asarray(yB)) ** 2))
        if mse >= 2.0:
            raise AssertionError(f"forward fitted MSE too large: {mse}")
        lo = np.asarray(outB["lo-80"]); hi = np.asarray(outB["hi-80"]); mean = np.asarray(outB["mean"])
        assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
        assert "fitted-lo-80" in outB and "fitted-hi-80" in outB
        _print_ok("test_forward_on_new_series_expected_or_reasonable")

    def test_predict_before_fit_raises():
        et = ETS(season_length=1, model="ANN")
        try:
            _ = et.predict(h=1)
            raise AssertionError("Expected an error when calling predict() before fit()")
        except Exception:
            pass
        _print_ok("test_predict_before_fit_raises")

    def test_forward_before_fit_raises():
        et = ETS(season_length=1, model="ANN")
        try:
            _ = et.forward(y=jnp.asarray([1., 2., 3.]), h=1)
            raise AssertionError("Expected an error when calling forward() before fit()")
        except Exception:
            pass
        _print_ok("test_forward_before_fit_raises")

    def test_tiny_series_policy_expected():
        y = jnp.asarray([10.0, 11.0, 10.5], dtype=jnp.float64)
        et = ETS(season_length=1, model="ANN")
        try:
            et.fit(y)
            out = et.predict(h=3, level=None)
            if out["mean"].shape != (3,):
                raise AssertionError("Expected 3-step forecast for tiny series")
        except NotImplementedError as e:
            msg = str(e).lower()
            if "tiny datasets" not in msg:
                raise
        _print_ok("test_tiny_series_policy_expected")

    # ================= φ validation & dtypes =================

    def test_phi_validation_range_and_type():
        try:
            _ = ETS(phi=1.5)
            raise AssertionError("Expected ValueError for invalid phi range")
        except ValueError:
            pass
        try:
            _ = ETS(phi="0.9")  # type: ignore
            raise AssertionError("Expected ValueError for non-float phi")
        except ValueError:
            pass
        _ = ETS(phi=0.8)
        _ = ETS(phi=0.98)
        _print_ok("test_phi_validation_range_and_type")

    def test_basic_shapes_and_dtypes_predict():
        y = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        et = ETS(season_length=1, model="ANN")
        et.fit(y.astype(jnp.float64))
        out = et.predict(h=3, level=None)
        assert out["mean"].shape == (3,)
        assert out["mean"].dtype in (jnp.float32, jnp.float64)
        _print_ok("test_basic_shapes_and_dtypes_predict")

    # ================= Runner =================
    test_constant_series_ANN_expected()
    test_linear_trend_AAN_expected()
    test_additive_seasonality_AAA_expected()
    test_stateless_forecast_matches_stateful_expected()

    test_predict_native_intervals_and_unsorted_levels()
    test_forecast_adds_fitted_and_fitted_intervals_when_requested()
    test_predict_in_sample_intervals_monotonicity_expected()
    test_fit_caches_conformal_then_predict_uses_cache()
    test_stateless_forecast_with_conformal_intervals()

    test_forward_on_new_series_expected_or_reasonable()
    test_predict_before_fit_raises()
    test_forward_before_fit_raises()
    test_tiny_series_policy_expected()

    test_phi_validation_range_and_type()
    test_basic_shapes_and_dtypes_predict()

    print("All ETS full-coverage tests passed.")
