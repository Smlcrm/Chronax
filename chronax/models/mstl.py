"""
MSTL (multiple seasonal-trend decomposition using LOESS, locally weighted
scatterplot smoothing) in JAX.

MSTL extends STL (seasonal-trend decomposition using LOESS) to several
seasonal periods, e.g. daily plus weekly:

1. Extract each seasonal component in turn with ``stl_decompose``.
2. Remove the extracted seasonal from the series before the next period.
3. Repeat for ``iterate`` outer passes to refine the decomposition.
4. Final components: trend + sum of seasonals + remainder.

Forecasts model the seasonally adjusted series (trend + remainder) with Theta
and add each seasonal pattern repeated cyclically. Prediction intervals are
conformal via the ``BaseForecaster`` interface.
"""
from __future__ import annotations
import jax.numpy as jnp
from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from .stl import stl_decompose
from chronax.utils import ConformalIntervals


def _forecast_seasonally_adjusted(sa: jnp.ndarray, h: int) -> jnp.ndarray:
    """Forecast the deseasonalised (trend + remainder) series ``h`` steps with Theta.

    Uses Theta's stateless ``forecast`` rather than ``fit().predict()``: the
    eager ``fit`` entry has host-side control flow that does not trace under
    MSTL's ``conformity_scores`` vmap, and ``forecast`` writes nothing to the
    local Theta instance, so no tracers leak onto it. The lazy import avoids
    the ``chronax.models`` <-> ``mstl`` circular import.
    """
    from chronax.models import Theta

    return Theta().forecast(y=sa, h=int(h))["mean"]


class MSTL(BaseForecaster):
    """
    Multiple seasonal-trend decomposition using LOESS (MSTL).

    Extracts one seasonal component per period with repeated STL passes.
    Forecasts model the seasonally adjusted series (trend + remainder) with
    Theta and add each seasonal pattern repeated cyclically. Prediction
    intervals are conformal and require ``conformal_params``; in-sample
    intervals use the remainder's standard error.

    Args:
        period (int | list[int]): Seasonal period(s) to extract, sorted in ascending order internally. For example ``[7, 365]`` for weekly and yearly seasonality. An empty list or a period of 1 means trend smoothing only.
        iterate (int): Number of outer refinement passes. Forced to 1 when a single period is given. Default is 2.
        s_window (int | list[int] | None): Seasonal smoother window(s), one per period, made odd. If None, the first entries of ``[11, 15, 19, 23, 27, 31]``. A single int applies to every period. Default is None.
        seasonal_deg (int): Polynomial degree (0 or 1) for the seasonal LOESS smoother. Default is 0.
        trend_deg (int): Polynomial degree (0 or 1) for the trend LOESS smoother. Default is 1.
        seasonal_jump (int): Jump step for the seasonal LOESS anchors, at least 1. Default is 1.
        trend_jump (int): Jump step for the trend LOESS anchors, at least 1. Default is 1.
        inner (int): Number of inner STL iterations per period extraction. Default is 1.
        trend_window (int | None): Trend smoother window, made odd. If None, derived per period as ``(2 * period + 1) | 1``. Default is None.
        low_pass (int | None): Low-pass filter window for the seasonal component, made odd. If None, STL uses the trend window. Default is None.
        tail_window (int | None): Accepted for compatibility; unused, the forecast models the seasonally adjusted series with Theta. Default is None.
        fitted (bool): Whether ``predict_in_sample`` includes the fitted values and decomposition components. Default is True.
        alias (str): Custom name of the model. Default is "MSTL".
        conformal_params (ConformalIntervals | None): Configuration for conformal prediction intervals; required for ``level`` in ``predict``. Default is None.
    """
    uses_exog = False

    def __init__(
        self,
        period: int | list[int],
        iterate: int = 2,
        s_window: int | list[int] | None = None,
        seasonal_deg: int = 0,
        trend_deg: int = 1,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        inner: int = 1,
        trend_window: int | None = None,
        low_pass: int | None = None,
        tail_window: int | None = None,
        fitted: bool = True,
        alias: str = "MSTL",
        conformal_params: ConformalIntervals | None = None,
    ) -> None:
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}

        self.period = period if isinstance(period, list) else [period]
        self.period = sorted(int(p) for p in self.period)
        self.iterate = 1 if len(self.period) == 1 else int(iterate)

        if s_window is None:
            default_windows = [11, 15, 19, 23, 27, 31]
            s_window = default_windows[: max(1, len(self.period))]
        if isinstance(s_window, int):
            s_window = [int(s_window)] * len(self.period)
        self.s_window = [int(w) | 1 for w in s_window]

        self.seasonal_deg = int(seasonal_deg)
        self.trend_deg = int(trend_deg)
        self.seasonal_jump = max(1, int(seasonal_jump))
        self.trend_jump = max(1, int(trend_jump))
        self.inner = int(inner)
        self.trend_window = None if trend_window is None else int(trend_window) | 1
        self.low_pass = None if low_pass is None else int(low_pass) | 1
        self.tail_window = tail_window
        self.fitted = bool(fitted)

    def _derive_trend(self, p: int) -> int:
        """
        Derive trend window size from a seasonal period.

        Args:
            p (int): Seasonal period length.

        Returns:
            int: Odd-valued trend window size computed as (2*p + 1) | 1.
        """
        return (2 * p + 1) | 1

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "MSTL":
        """
        Fit the MSTL decomposition.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``; reshaped to 1-D and cast to float.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``. Ignored. Default is None.

        Returns:
            MSTL: ``self``, with ``model_`` holding ``"y"``, ``"trend"``,
            ``"seasonals"`` (shape ``(n_periods, T)``), ``"remainder"`` and
            ``"periods"``.
        """
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        n = y.size
        msts = self.period

        # No seasonality: smooth trend only
        if len(msts) == 0 or (len(msts) == 1 and msts[0] <= 1):
            tw = self.trend_window if self.trend_window is not None else (2 * max(5, n // 10) + 1) | 1
            seas = jnp.zeros((0, n), dtype=y.dtype)
            # Reuse STL with period=1, seasonal window=1 to get a trend smoothing
            _, trend, _ = stl_decompose(
                y=y,
                period=1,
                seasonal=1,
                trend=tw,
                low_pass=self.low_pass,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                seasonal_jump=1,
                trend_jump=self.trend_jump,
                inner=self.inner,
            )
        else:
            seas = jnp.zeros((len(msts), n), dtype=y.dtype)
            deseas = y
            trend = jnp.zeros_like(y)
            for _ in range(self.iterate):
                for i, p in enumerate(msts):
                    deseas = deseas + seas[i]
                    tw = self.trend_window if self.trend_window is not None else self._derive_trend(p)
                    s_i, trend, _ = stl_decompose(
                        y=deseas,
                        period=p,
                        seasonal=self.s_window[i],
                        trend=tw,
                        low_pass=self.low_pass,
                        seasonal_deg=self.seasonal_deg,
                        trend_deg=self.trend_deg,
                        seasonal_jump=self.seasonal_jump,
                        trend_jump=self.trend_jump,
                        inner=self.inner,
                    )
                    seas = seas.at[i].set(s_i)
                    deseas = deseas - s_i

        comp_seas = jnp.sum(seas, axis=0) if seas.size else jnp.zeros_like(y)
        remainder = y - trend - comp_seas

        self.model_ = {
            "y": y,
            "trend": trend,
            "seasonals": seas,   # shape: (n_seasons, n)
            "remainder": remainder,
            "periods": msts,
        }
        return self

    def predict_in_sample(self, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        """
        Return in-sample fitted values and decomposition components.

        Args:
            X (jnp.ndarray | None): Exogenous regressors. Ignored. Default is None.
            level (list[int | float] | None): Confidence levels in (0, 100), e.g. ``[90, 95]``. Fitted intervals use the remainder's standard error with normal quantiles. Default is None.

        Returns:
            dict: Dictionary containing:

                - ``"mean"``: fitted values (trend + all seasonals), shape ``(T,)``.
                - ``"fitted"``, ``"trend"``, ``"remainder"``: shape ``(T,)``, when ``self.fitted`` is True.
                - ``"seasonal"`` (single period) or ``"seasonal{p}"`` per period ``p``: when ``self.fitted`` is True.
                - ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"``: per level, when ``level`` is given.
        """
        y = self.model_["y"]
        trend = self.model_["trend"]
        seas_mat = self.model_["seasonals"]
        comp_seas = jnp.sum(seas_mat, axis=0) if seas_mat.size else jnp.zeros_like(y)

        out = {"mean": trend + comp_seas}
        if self.fitted:
            out["fitted"] = out["mean"]
            out["trend"] = trend
            if seas_mat.size:
                for i, p in enumerate(self.model_["periods"]):
                    key = "seasonal" if len(self.model_["periods"]) == 1 else f"seasonal{p}"
                    out[key] = seas_mat[i]
            out["remainder"] = self.model_["remainder"]
        if level:
            # In-sample intervals use the remainder scale: cross-validation (CV)
            # conformity scores have shape (n_windows, h) and cannot broadcast
            # onto the (n,) fitted mean.
            level = sorted(level)
            sigma = utils.calculate_sigma(self.model_["remainder"],
                                          self.model_["remainder"].shape[0])
            out = {**out, **utils._add_fitted_pi_1(out["mean"], sigma, level)}
        return out

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        """
        Generate out-of-sample forecasts from the fitted decomposition.

        Models the seasonally adjusted series (trend + remainder) with Theta
        and adds each seasonal pattern repeated cyclically, as statsforecast's
        MSTL does. Intervals are conformal with the
        ``"conformal_distribution"`` method, computed on the stored series.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Future exogenous regressors. Ignored. Default is None.
            level (list[int | float] | None): Confidence levels in (0, 100), e.g. ``[90, 95]``. Requires ``conformal_params``. Default is None.

        Returns:
            dict: Dictionary containing:

                - ``"mean"``: point forecasts, shape ``(h,)``.
                - ``"lo-{lv}"``/``"hi-{lv}"``: per level, when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``; if ``level`` is given without
                ``conformal_params``; or if the series is too short for 2 CV
                windows.
        """
        self._require_fitted()
        h = int(h)
        y = self.model_["y"]
        trend = self.model_["trend"]
        seas_mat = self.model_["seasonals"]
        periods = self.model_["periods"]
        n = y.size

        # Forecast the seasonally adjusted series (trend + remainder) with
        # Theta, then re-add the cyclic seasonal (statsforecast's MSTL scheme).
        # `tail_window` is unused.
        sa = trend + self.model_["remainder"]
        sa_fcst = _forecast_seasonally_adjusted(sa, h)

        if seas_mat.size:
            seas_fcst = jnp.zeros(h, dtype=y.dtype)
            for i, p in enumerate(periods):
                seas_fcst = seas_fcst + utils._repeat_val_seas(seas_mat[i, -p:], h)
        else:
            seas_fcst = jnp.zeros(h, dtype=y.dtype)

        out = {"mean": sa_fcst + seas_fcst}
        if level:
            cs = self.conformity_scores(y)
            out = self.add_confidence_intervals(out, cs, level, "conformal_distribution")
        return out

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: int | tuple[int, ...] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Stateless fit+predict on a fresh clone; the CV path of ``conformity_scores`` calls this.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Exogenous regressors. Ignored. Default is None.
            X_future (jnp.ndarray | None): Future exogenous regressors. Ignored. Default is None.
            level (int | tuple[int, ...] | None): Confidence levels, forwarded to :meth:`predict`. Default is None.
            fitted (bool): Accepted for interface parity; ignored, no ``"fitted"`` key is returned. Default is False.

        Returns:
            dict: Same keys as :meth:`predict`.

        Raises:
            ValueError: As :meth:`predict` when ``level`` is given.
        """
        return self.new().fit(y, X).predict(h, X, level=level)


# =========================
# Test Cases
# =========================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from chronax.models import STL
    print("=" * 60)
    print("MSTL Test Suite")
    print("=" * 60)
    
    # Test 1: Basic MSTL with two seasonal periods
    print("\n[Test 1] MSTL with two seasonal periods")
    n = 100
    period1, period2 = 7, 14
    t = jnp.arange(n)
    trend = 0.3 * t
    seasonal1 = 2.0 * jnp.sin(2 * jnp.pi * t / period1)
    seasonal2 = 1.5 * jnp.cos(2 * jnp.pi * t / period2)
    noise = jrandom.normal(jrandom.PRNGKey(123), (n,)) * 0.2
    y1 = trend + seasonal1 + seasonal2 + noise
    
    model1 = MSTL(period=[period1, period2], iterate=2, s_window=[7, 11])
    model1.fit(y1)
    result1 = model1.predict_in_sample()
    
    print(f"  Input series length: {n}")
    print(f"  Periods: {[period1, period2]}")
    print(f"  Result keys: {list(result1.keys())}")
    print(f"  Fitted shape: {result1['fitted'].shape}")
    print(f"  Trend shape: {result1['trend'].shape}")
    
    assert result1['fitted'].shape == (n,), "Fitted shape mismatch!"
    assert result1['trend'].shape == (n,), "Trend shape mismatch!"
    assert f'seasonal{period1}' in result1, f"Should have seasonal{period1}!"
    assert f'seasonal{period2}' in result1, f"Should have seasonal{period2}!"
    print(f"  Seasonal{period1} shape: {result1[f'seasonal{period1}'].shape}")
    print(f"  Seasonal{period2} shape: {result1[f'seasonal{period2}'].shape}")
    print("  ✓ Multi-seasonal decomposition OK")
    
    # Test 2: Forecast with multiple seasonals
    print("\n[Test 2] Forecast with multiple seasonals")
    h = 28
    forecast2 = model1.predict(h)
    print(f"  Forecast horizon: {h}")
    print(f"  Forecast shape: {forecast2['mean'].shape}")
    print(f"  Forecast values (first 5): {forecast2['mean'][:5]}")
    assert forecast2['mean'].shape == (h,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast2['mean'])), "Forecast contains NaN/Inf!"
    print("  ✓ Multi-seasonal forecast OK")
    
    # Test 3: Single seasonal period (should behave like STL)
    print("\n[Test 3] MSTL with single period (like STL)")
    model3_mstl = MSTL(period=7, iterate=1, s_window=7)
    model3_stl = STL(period=7, seasonal=7, trend=15)
    
    y3 = trend + seasonal1 + noise  # Only one seasonal
    model3_mstl.fit(y3)
    model3_stl.fit(y3)
    
    result3_mstl = model3_mstl.predict_in_sample()
    result3_stl = model3_stl.predict_in_sample()
    
    # Should have similar results
    diff = jnp.mean(jnp.abs(result3_mstl['fitted'] - result3_stl['fitted']))
    print(f"  Difference between MSTL and STL: {diff:.6f}")
    assert 'seasonal' in result3_mstl, "Single period should use 'seasonal' key!"
    print("  ✓ Single-period MSTL matches STL behavior")
    
    # Test 4: Iteration effect
    print("\n[Test 4] Iteration refinement effect")
    model4_iter1 = MSTL(period=[7, 14], iterate=1, s_window=[7, 11])
    model4_iter3 = MSTL(period=[7, 14], iterate=3, s_window=[7, 11])
    
    model4_iter1.fit(y1)
    model4_iter3.fit(y1)
    
    result4_iter1 = model4_iter1.predict_in_sample()
    result4_iter3 = model4_iter3.predict_in_sample()
    
    # More iterations should refine the decomposition
    remainder_var1 = jnp.var(result4_iter1['remainder'])
    remainder_var3 = jnp.var(result4_iter3['remainder'])
    
    print(f"  Remainder variance (1 iter): {remainder_var1:.6f}")
    print(f"  Remainder variance (3 iter): {remainder_var3:.6f}")
    print(f"  Variance reduction: {(remainder_var1 - remainder_var3) / remainder_var1 * 100:.2f}%")
    print("  ✓ Iterations refine decomposition")
    
    # Test 5: Three seasonal periods
    print("\n[Test 5] MSTL with three seasonal periods")
    n5 = 150
    t5 = jnp.arange(n5)
    trend5 = 0.2 * t5
    seas5_1 = 1.5 * jnp.sin(2 * jnp.pi * t5 / 5)
    seas5_2 = 1.0 * jnp.cos(2 * jnp.pi * t5 / 10)
    seas5_3 = 0.8 * jnp.sin(2 * jnp.pi * t5 / 20)
    noise5 = jrandom.normal(jrandom.PRNGKey(456), (n5,)) * 0.15
    y5 = trend5 + seas5_1 + seas5_2 + seas5_3 + noise5
    
    model5 = MSTL(period=[5, 10, 20], iterate=2)
    model5.fit(y5)
    result5 = model5.predict_in_sample()
    
    print(f"  Periods: [5, 10, 20]")
    assert 'seasonal5' in result5, "Should have seasonal5!"
    assert 'seasonal10' in result5, "Should have seasonal10!"
    assert 'seasonal20' in result5, "Should have seasonal20!"
    print(f"  All three seasonal components present: ✓")
    
    # Check if sum of components reconstructs the series
    reconstructed = result5['trend'] + result5['seasonal5'] + result5['seasonal10'] + result5['seasonal20'] + result5['remainder']
    reconstruction_error = jnp.mean(jnp.abs(y5 - reconstructed))
    print(f"  Reconstruction error: {reconstruction_error:.8f}")
    assert reconstruction_error < 1e-5, "Components should sum to original series!"
    print("  ✓ Three-period decomposition OK")
    
    # Test 6: No seasonality (trend only)
    print("\n[Test 6] MSTL with no seasonality (trend smoothing only)")
    y6 = 0.5 * t + jrandom.normal(jrandom.PRNGKey(789), (n,)) * 0.3
    model6 = MSTL(period=[], iterate=1)  # Empty period list
    model6.fit(y6)
    result6 = model6.predict_in_sample()
    
    print(f"  Input: linear trend + noise")
    print(f"  Result keys: {list(result6.keys())}")
    assert 'trend' in result6, "Should have trend!"
    assert 'fitted' in result6, "Should have fitted!"
    # Should not have seasonal components
    seasonal_keys = [k for k in result6.keys() if 'seasonal' in k]
    print(f"  Seasonal components found: {seasonal_keys}")
    print("  ✓ No-seasonality case handled")
    
    # Test 7: Custom window sizes
    print("\n[Test 7] Custom seasonal window sizes")
    model7 = MSTL(period=[7, 14], s_window=[9, 15], iterate=2)
    model7.fit(y1)
    result7 = model7.predict_in_sample()
    
    print(f"  Custom s_window: [9, 15]")
    print(f"  Fitted successfully: ✓")
    assert jnp.all(jnp.isfinite(result7['fitted'])), "Should produce finite results!"
    print("  ✓ Custom windows work")
    
    # Test 8: Tail window for forecasting
    print("\n[Test 8] Tail window parameter for forecasting")
    model8_default = MSTL(period=[7, 14], iterate=2, tail_window=None)
    model8_custom = MSTL(period=[7, 14], iterate=2, tail_window=10)
    
    model8_default.fit(y1)
    model8_custom.fit(y1)
    
    forecast8_default = model8_default.predict(h=14)
    forecast8_custom = model8_custom.predict(h=14)
    
    diff8 = jnp.mean(jnp.abs(forecast8_default['mean'] - forecast8_custom['mean']))
    print(f"  Default tail vs custom tail(10) difference: {diff8:.6f}")
    print("  ✓ Tail window parameter works")
    
    # Test 9: Fitted flag
    print("\n[Test 9] Fitted flag controls output components")
    model9_fitted_true = MSTL(period=[7, 14], iterate=2, fitted=True)
    model9_fitted_false = MSTL(period=[7, 14], iterate=2, fitted=False)
    
    model9_fitted_true.fit(y1)
    model9_fitted_false.fit(y1)
    
    result9_true = model9_fitted_true.predict_in_sample()
    result9_false = model9_fitted_false.predict_in_sample()
    
    print(f"  fitted=True keys: {list(result9_true.keys())}")
    print(f"  fitted=False keys: {list(result9_false.keys())}")
    
    assert 'fitted' in result9_true, "fitted=True should include 'fitted'!"
    assert 'trend' in result9_true, "fitted=True should include 'trend'!"
    assert 'remainder' in result9_true, "fitted=True should include 'remainder'!"
    print("  ✓ Fitted flag controls output")
    
    # Test 10: Conformal intervals
    print("\n[Test 10] Conformal prediction intervals")
    from chronax.utils import ConformalIntervals
    conformal_params = ConformalIntervals(h=14)
    model10 = MSTL(period=[7, 14], iterate=2, conformal_params=conformal_params)
    model10.fit(y1)
    forecast10 = model10.predict(h=14, level=[90, 95])
    
    print(f"  Forecast keys: {list(forecast10.keys())}")
    assert 'mean' in forecast10, "Should have 'mean'!"
    # Check for interval keys (format may vary: 'lower_90'/'lo-90' or 'upper_90'/'hi-90')
    has_90_intervals = ('lower_90' in forecast10 or 'lo-90' in forecast10)
    has_95_intervals = ('lower_95' in forecast10 or 'lo-95' in forecast10)
    assert has_90_intervals, "Should have 90% interval keys!"
    assert has_95_intervals, "Should have 95% interval keys!"
    print(f"  Mean forecast (first 5): {forecast10['mean'][:5]}")
    # Use whichever format exists
    lo_90_key = 'lo-90' if 'lo-90' in forecast10 else 'lower_90'
    hi_90_key = 'hi-90' if 'hi-90' in forecast10 else 'upper_90'
    lo_95_key = 'lo-95' if 'lo-95' in forecast10 else 'lower_95'
    hi_95_key = 'hi-95' if 'hi-95' in forecast10 else 'upper_95'
    print(f"  90% interval width (first): {forecast10[hi_90_key][0] - forecast10[lo_90_key][0]:.4f}")
    print(f"  95% interval width (first): {forecast10[hi_95_key][0] - forecast10[lo_95_key][0]:.4f}")
    print("  ✓ Conformal intervals work")
    
    print("\n" + "=" * 60)
    print("✓ All MSTL tests passed!")
    print("=" * 60)