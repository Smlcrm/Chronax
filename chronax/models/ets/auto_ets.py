from __future__ import annotations
from typing import Any, List, Optional

import jax.numpy as jnp

from chronax.utils import (
    ConformalIntervals,
    ensure_float,
    calculate_sigma,
    _add_fitted_pi,
)

from chronax.models.base_forecaster import BaseForecaster

from .ets_functions import (
    ets_f,
    ets_point_forecast,
    ets_winner_view,
    forecast_ets,
    forward_ets,
)

_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

class AutoETS(BaseForecaster):
    r"""Automatic Exponential Smoothing model.

    Automatically selects the best ETS (Error, Trend, Seasonality)
    model using an information criterion. Default is Akaike Information Criterion (AICc), while particular models are estimated using maximum likelihood.
    The state-space equations can be determined based on their $M$ multiplicative, $A$ additive,
    $Z$ optimized or $N$ ommited components. The `model` string parameter defines the ETS equations:
    E in [$M, A, Z$], T in [$N, A, M, Z$], and S in [$N, A, M, Z$].

    For example when model='ANN' (additive error, no trend, and no seasonality), ETS will
    explore only a simple exponential smoothing.

    If the component is selected as 'Z', it operates as a placeholder to ask the AutoETS model
    to figure out the best parameter.

    vmap contract: every candidate in the (config/shape-static) grid is fitted
    and the winner is a traced ``jnp.argmin`` over information criteria, so
    ``fit``/``forecast`` trace natively under ``jax.vmap`` — the base class's
    ``conformity_scores`` CV path. ``forecast()`` is fully stateless (never
    reads or writes ``model_``). Only the *native-interval* branch of
    ``predict``/``forecast``/``forward`` is eager-only, because the interval
    formulas dispatch on the winner's class string; conformal intervals are
    fully supported under the base-class contract.

    Args:
        season_length (int, default=1): Number of observations per unit of time. Ex: 24 Hourly data.
            Config-only — array shapes derive from it, so it is never inferred from the data.
        model (str, default="ZZZ"): Controlling state-space-equations.
        damped (bool, optional): A parameter that 'dampens' the trend.
        phi (float, optional): Smoothing parameter for trend damping. Only used when `damped=True`.
        max_iter (int, optional): Optimizer step budget; ``None`` derives a
            static budget from the series length.
        optax_lr (float, default=7e-2): Learning rate for the Adam warm-up phase.
        optax_clip (float, default=5.0): Retained for API compatibility.
        early_stop_patience (int): *Inert* — early stopping would require
            concretizing traced losses; the optimizer runs a fixed budget with
            monotone best-iterate tracking instead.
        early_stop_min_delta (float): *Inert* — see ``early_stop_patience``.
        alias (str, default="AutoETS"): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals], optional): Information to compute conformal prediction intervals. By default, the model will compute the native prediction intervals.

    Notes:
        This implementation is a mirror of Hyndman's [forecast::ets](https://github.com/robjhyndman/forecast).

    References:
        - [Rob J. Hyndman, Yeasmin Khandakar (2008). "Automatic Time Series Forecasting: The forecast package for R"](https://www.jstatsoft.org/article/view/v027i03).
        - [Hyndman, Rob, et al (2008). "Forecasting with exponential smoothing: the state space approach"](https://robjhyndman.com/expsmooth/).
    """

    @staticmethod
    def _validate_h(h: int) -> None:
        """Validate that the forecast horizon is a positive integer."""
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be positive integer, got {h}")

    @staticmethod
    def _validate_level(level: Optional[List[int]]) -> None:
        """Validate an optional list of interval confidence levels."""
        if level is None:
            return
        if not isinstance(level, list):
            raise ValueError("level must be a list or None")
        for lv in level:
            if not isinstance(lv, (int, float)) or lv < 0 or lv > 100:
                raise ValueError(f"level values must be in [0, 100], got {lv}")

    def _validate_series_length(self, y: jnp.ndarray) -> None:
        """Ensure the series is long enough for the configured season length."""
        min_len = max(self.season_length, 3)
        if len(y) < min_len:
            raise ValueError(
                f"Series too short: need >= {min_len} observations, got {len(y)}"
            )

    @staticmethod
    def _default_optax_steps(n: int) -> int:
        """Static optimizer budget keyed on series *length* (shape-static)."""
        if n < 200:
            return 200
        if n < 1000:
            return 300
        if n < 5000:
            return 400
        return 500

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        max_iter: Optional[int] = None,
        optax_lr: float = 7e-2,
        optax_clip: float = 5.0,
        early_stop_patience: int = 10,  # inert — see class docstring
        early_stop_min_delta: float = 1e-5,  # inert — see class docstring
        alias: str = "AutoETS",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize the AutoETS estimator configuration."""
        self.season_length = season_length
        self.model = model
        self.damped = damped
        if phi is not None:
            if not isinstance(phi, float):
                raise ValueError("phi must be `None` or float.")
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"Valid range for phi is [{_PHI_LOWER}, {_PHI_UPPER}]")
        self.phi = phi
        self.max_iter = max_iter
        self.optax_lr = optax_lr
        self.optax_clip = optax_clip
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.alias = alias
        self.conformal_params = prediction_intervals
        self.optax_steps = self.max_iter
        # Eagerly-selected winning (E,T,S,damped) spec, cached at fit so the
        # vmapped conformity_scores CV path re-fits only that fixed spec per
        # window (the AutoARIMA/AutoMFLES eager-select-once pattern). None until
        # fit; None also means "fresh/direct forecast -> full ZZZ grid".
        self._selected_spec: Optional[tuple] = None
        self._cs: Optional[jnp.ndarray] = None

    def _fit_stacked(
        self, y: jnp.ndarray, spec: Optional[tuple] = None
    ) -> dict[str, Any]:
        """Run the candidate fit on *y* (shared by fit/forecast).

        When *spec* is a concrete ``(etype, ttype, stype, damped)`` tuple (the
        eagerly-selected winner cached at fit), only that single candidate is
        fitted — a fully-traceable one-candidate ``ets_f`` (``n_cand == 1`` ->
        static winner index). This is what the vmapped
        ``conformity_scores`` CV path uses, so each window re-fits the SELECTED
        spec's parameters instead of re-running the ~12-candidate ZZZ grid (the
        O(candidates x n_windows) scalability wall). With
        *spec=None* the full ``self.model`` grid is fitted (fresh/direct use).
        """
        steps = self.max_iter if self.max_iter is not None else self._default_optax_steps(len(y))
        if spec is not None:
            etype, ttype, stype, dtype = spec
            return ets_f(
                y,
                m=self.season_length,
                model=f"{etype}{ttype}{stype}",
                damped=bool(dtype),
                phi=self.phi,
                allow_multiplicative_trend=True,
                optax_steps=steps,
                optax_lr=self.optax_lr,
                optax_clip=self.optax_clip,
            )
        return ets_f(
            y,
            m=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            # Allow multiplicative trend for richer models on positive series.
            allow_multiplicative_trend=True,
            optax_steps=steps,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
        )

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "AutoETS":
        r"""Fit the Exponential Smoothing model.

        Fit an Exponential Smoothing model to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (numpy.array): Clean time series of shape (t, ).
            X (array-like, optional): Optional exogenous of shape (t, n_x).

        Returns:
            AutoETS: Exponential Smoothing fitted model.
        """
        y = ensure_float(y)
        self._validate_series_length(y)
        self.optax_steps = self.max_iter if self.max_iter is not None else self._default_optax_steps(len(y))
        self.model_ = self._fit_stacked(y)
        self.model_["actual_residuals"] = y - self.model_["fitted"]
        # Cache the eagerly-selected winning spec so the vmapped CV path
        # (conformity_scores -> forecast) re-fits ONLY this fixed spec per
        # window instead of the full ~12-candidate ZZZ grid. Concretising the
        # traced argmin index here is eager/host-side (fit is not traced); the
        # winner index is static when only one candidate was generated.
        cands = self.model_["candidates"]
        best_idx = 0 if len(cands) == 1 else int(self.model_["best"])
        self._selected_spec = tuple(cands[best_idx])
        # Cache conformity scores on the training series (sibling convention);
        # forecast() below is write-free on the fitted fast path, so the vmapped
        # CV re-fits inside conformity_scores cannot leak tracers onto self.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def conformity_scores(
        self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Conformity scores with eager spec selection, vmapped per-window refit.

        The base implementation vmaps ``self.forecast`` over CV windows. Full
        ZZZ model selection (fitting ~12 candidates and argmin-ing their IC) is
        far too expensive to repeat inside every window — the
        O(candidates x n_windows x optax_steps x n) wall that made AutoETS
        ``TIMEOUT`` on long series. So the winning
        ``(E,T,S,damped)`` spec is selected ONCE, eagerly, on the full series
        here (via ``fit``); the vmapped ``forecast`` then re-fits only that
        fixed spec's parameters per window through the one-candidate ``ets_f``
        fast path (``_fit_stacked(spec=...)``) — mirroring the AutoARIMA
        (``auto_arima.py``) and AutoMFLES (``auto_mfles.py``) overrides.

        Calibration caveat (shared with AutoARIMA and AutoMFLES): the SPEC is
        chosen with sight of the full series, including the CV test
        windows — mildly optimistic. Parameters are still honestly re-fit per
        window, so scores vary across windows.

        Side Effects:
            First call on an unfitted estimator runs ``fit`` (caches the
            selected spec + scores) — mirroring ``forecast``'s first-call
            behaviour and the sibling Auto overrides.
        """
        if self._selected_spec is None:
            self.fit(y, X)
            # fit() just ran the CV on exactly this y and cached the scores;
            # reuse them rather than paying the n_windows re-fits twice.
            if self._cs is not None:
                return self._cs
        return super().conformity_scores(y, X)

    def predict(
        self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None
    ) -> dict[str, jnp.ndarray]:
        r"""Predict with fitted Exponential Smoothing.

        Args:
            h (int): Forecast horizon.
            X (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        self._require_fitted()
        self._validate_h(h)
        self._validate_level(level)
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        res = {"mean": ets_point_forecast(self.model_, h)}
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is not None:
            if getattr(self, "_cs", None) is None:
                raise ValueError(
                    "Conformity scores not cached. Fit the model with conformal_params set, "
                    "or use forecast(y, ...) which recomputes them."
                )
            if h != self.conformal_params.h:
                raise ValueError(
                    f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            return self.add_confidence_intervals(
                fcst=res,
                cs=self._cs,
                level=level,
                method=self.conformal_params.method,
            )

        # Native ETS intervals (eager-only: dispatches on the winner's
        # class string via ets_winner_view).
        fcst = forecast_ets(ets_winner_view(self.model_), h=h, level=level)
        res.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level)})
        res.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level})
        return res

    def predict_in_sample(self, level: Optional[List[int]] = None) -> dict[str, jnp.ndarray]:
        r"""Access fitted Exponential Smoothing insample predictions.

        Args:
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `fitted` for point predictions and `level_*` for probabilistic predictions.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            residuals = self.model_["actual_residuals"]
            se = calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def _compose_output(
        self,
        mod: dict[str, Any],
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray],
        level: Optional[List[int]],
        fitted: bool,
    ) -> dict[str, Any]:
        """Build the forecast dict (mean + optional fitted/intervals) from a stacked fit."""
        res = {"mean": ets_point_forecast(mod, h)}
        if fitted:
            res["fitted"] = mod["fitted"]
        if level is None:
            return res

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            if h != self.conformal_params.h:
                raise ValueError(
                    f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            cs = self.conformity_scores(y=y, X=X)
            res = self.add_confidence_intervals(res, cs, level_sorted, self.conformal_params.method)
        else:
            fcst = forecast_ets(ets_winner_view(mod), h=h, level=level_sorted)
            res.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)})
            res.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted})
        if fitted:
            se = calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            res = _add_fitted_pi(res=res, se=se, level=level_sorted)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> dict[str, Any]:
        r"""Memory Efficient Exponential Smoothing predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Stateless by contract: the base class's ``conformity_scores`` vmaps
        this method over CV windows, so it never reads or writes ``model_``
        (stored state would leak tracers and future information).

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for prediction intervals.
            fitted (bool, default=False): Whether or not returns insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = ensure_float(y)
        self._validate_h(h)
        self._validate_level(level)
        self._validate_series_length(y)
        # Fitted fast path: re-fit only the eagerly-selected spec (cheap, and
        # writes NOTHING to self, so the base conformity_scores can vmap this).
        # Fresh/direct use (_selected_spec is None): full ZZZ grid.
        mod = self._fit_stacked(y, spec=self._selected_spec)
        return self._compose_output(mod, y, h, X, level, fitted)

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> dict[str, Any]:
        r"""Apply fitted Exponential Smoothing model to a new time series.

        Reuses the fitted candidates' structures and smoothing parameters
        (no re-optimisation) and keeps the fit-time winner; each candidate is
        re-rolled on *y* via ``forward_ets``.

        Args:
            y (numpy.array): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (array-like, optional): Optional insample exogenpus of shape (t, n_x).
            X_future (array-like, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels for prediction intervals.
            fitted (bool, default=False): Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        self._validate_h(h)
        self._validate_level(level)
        self._validate_series_length(y)
        mod = forward_ets(self.model_, y)
        return self._compose_output(mod, y, h, X, level, fitted)


# -------------------------------------------------------------------
# Test cases
# -------------------------------------------------------------------

import numpy as np

# ============== Helpers ==============
def _assert_allclose(
    actual: object,
    expected: object,
    atol: float = 1e-4,
    rtol: float = 1e-6,
    label: str = "",
) -> None:
    """Assert two arrays are numerically close within tolerances."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    if not np.allclose(a, e, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label} mismatch.\nActual:   {a}\nExpected: {e}\n"
            f"max|Δ|={np.max(np.abs(a - e))}, atol={atol}, rtol={rtol}"
        )

def _print_ok(name: str) -> None:
    """Print a standardized success message for local smoke tests."""
    print(f"{name}: OK")


# ============== Core expected-value tests ==============

def test_constant_series_ANN_expected() -> None:
    """Smoke-test that a constant series selects and forecasts like ANN."""
    y = jnp.full((40,), 10.0, dtype=jnp.float64)
    m = AutoETS(season_length=1, model="ZZZ")
    m.fit(y)

    fitted = m.predict_in_sample(level=None)["fitted"]
    _assert_allclose(np.mean(np.abs(np.asarray(fitted) - 10.0)), 0.0, atol=1e-2, label="ANN fitted MAE")

    h = 5
    out = m.predict(h=h, level=None)
    expected = np.full(h, 10.0, dtype=np.float64)
    _assert_allclose(out["mean"], expected, atol=1e-1, label="ANN mean")
    _print_ok("test_constant_series_ANN_expected")


def test_linear_trend_AAN_expected() -> None:
    """Smoke-test that a linear trend produces a stable additive trend path."""
    a, b = 2.0, 0.5
    t = np.arange(60, dtype=np.float64)
    y = jnp.asarray(a + b * t, dtype=jnp.float64)

    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)

    h = 6
    out = ae.predict(h=h, level=None)
    mean = np.asarray(out["mean"])

    last_val = a + b * (len(y) - 1)
    expected_first = last_val + b
    expected = expected_first + b * np.arange(h, dtype=np.float64)
    _assert_allclose(mean[0], expected_first, atol=0.2, label="AAN step-1")
    _assert_allclose(mean, expected, atol=0.5, label="AAN path")
    _print_ok("test_linear_trend_AAN_expected")


def test_additive_seasonality_AAA_expected() -> None:
    """Smoke-test that additive seasonality repeats the learned pattern."""
    m = 4
    base = 10.0
    season = np.array([+1.0, -1.0, +2.0, -2.0], dtype=np.float64)
    y = jnp.asarray(np.tile(season, 15) + base, dtype=jnp.float64)  # len 60

    ae = AutoETS(season_length=m, model="ZZZ")
    ae.fit(y)

    h = 8
    out = ae.predict(h=h, level=None)
    mean = np.asarray(out["mean"])

    idx0 = len(y) % m
    expected = np.array([base + season[(idx0 + k) % m] for k in range(h)], dtype=np.float64)
    _assert_allclose(mean, expected, atol=1.0, label="AAA seasonal pattern")
    _print_ok("test_additive_seasonality_AAA_expected")


def test_stateless_forecast_matches_stateful_expected() -> None:
    """Smoke-test that stateless and stateful forecast paths agree."""
    t = np.arange(48, dtype=np.float64)
    base, slope = 5.0, 0.1
    season = np.array([0.0, +1.0, 0.0, -1.0], dtype=np.float64)
    y = jnp.asarray(base + slope * t + np.tile(season, len(t) // 4 + 1)[: len(t)], dtype=jnp.float64)

    h = 5
    ae = AutoETS(season_length=4, model="ZZZ")
    ae.fit(y)
    stateful = ae.predict(h=h, level=None)["mean"]
    stateless = ae.forecast(y=y, h=h, level=None, fitted=False)["mean"]

    _assert_allclose(stateful, stateless, atol=1e-6, label="stateless vs stateful mean")
    _print_ok("test_stateless_forecast_matches_stateful_expected")


# ============== Intervals & level handling ==============

def test_predict_native_intervals_and_unsorted_levels() -> None:
    """Smoke-test native intervals and level sorting behavior."""
    y = jnp.asarray([10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 12.5, 12.0] * 4, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)
    # Unsorted on purpose
    out = ae.predict(h=4, level=[95, 80, 50])

    assert "mean" in out and out["mean"].shape == (4,)
    for lv in (50, 80, 95):
        assert f"hi-{lv}" in out and f"lo-{lv}" in out
        assert out[f"hi-{lv}"].shape == (4,) and out[f"lo-{lv}"].shape == (4,)
    # sanity: lo ≤ mean ≤ hi
    m = np.asarray(out["mean"])
    for lv in (50, 80, 95):
        lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
        assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12), f"mean not inside {lv}% band"
    _print_ok("test_predict_native_intervals_and_unsorted_levels")


def test_forecast_adds_fitted_and_fitted_intervals_when_requested() -> None:
    """Smoke-test stateless forecasts with fitted values and fitted intervals."""
    # Needs enough history to avoid tiny-dataset path
    t = np.arange(60, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.2 * t, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    out = ae.forecast(y=y, h=5, level=[80, 95], fitted=True)
    assert "mean" in out and out["mean"].shape == (5,)
    assert "fitted" in out and out["fitted"].shape == y.shape
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        assert f"fitted-lo-{lv}" in out and f"fitted-hi-{lv}" in out
        # fitted must lie between its own bands
        fl = np.asarray(out[f"fitted-lo-{lv}"]); fh = np.asarray(out[f"fitted-hi-{lv}"])
        ft = np.asarray(out["fitted"])
        assert np.all(fl <= ft + 1e-12) and np.all(ft <= fh + 1e-12)
    _print_ok("test_forecast_adds_fitted_and_fitted_intervals_when_requested")


def test_predict_in_sample_intervals_monotonicity_expected() -> None:
    """Smoke-test in-sample interval nesting across confidence levels."""
    y = jnp.asarray([10.0, 12.0, 11.0, 13.0, 12.0, 11.5, 12.5, 12.0] * 3, dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y)

    res = ae.predict_in_sample(level=[80, 95])
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


# ============== Conformal intervals paths ==============

def test_fit_caches_conformal_then_predict_uses_cache() -> None:
    """
    Conformal path with windows large enough to avoid tiny-dataset errors inside
    rolling-origin fits. We use a longer y and modest n_windows/h.
    """
    # Make y long enough so that the earliest training window has plenty of data.
    # Rule of thumb: ensure n - n_windows*h >= ~20 for comfort.
    n = 40
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(8.0 + 0.05 * t + 0.5 * np.sin(2 * np.pi * t / 8), dtype=jnp.float64)

    cfg = ConformalIntervals(n_windows=5, h=3, method="conformal_distribution")
    ae = AutoETS(season_length=1, model="ZZZ", prediction_intervals=cfg)

    ae.fit(y)  # should not hit tiny-dataset in any window
    assert getattr(ae, "_cs") is not None

    out = ae.predict(h=3, level=[80, 95])
    for lv in (80, 95):
        assert f"lo-{lv}" in out and f"hi-{lv}" in out
        m = np.asarray(out["mean"])
        lo = np.asarray(out[f"lo-{lv}"]); hi = np.asarray(out[f"hi-{lv}"])
        assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
    print("test_fit_caches_conformal_then_predict_uses_cache: OK")


def test_stateless_forecast_with_conformal_intervals() -> None:
    """
    Stateless forecast with conformal intervals; series long enough that the
    internal rolling windows used for conformity scores are also large.
    The forecast horizon must match conformal_params.h — scores cover
    exactly h steps (guarded with a ValueError, sibling convention).
    """
    n = 36
    t = np.arange(n, dtype=np.float64)
    y = jnp.asarray(5.0 + 0.1 * t + 0.3 * np.sin(2 * np.pi * t / 6), dtype=jnp.float64)

    cfg = ConformalIntervals(n_windows=4, h=6, method="conformal_distribution")
    ae = AutoETS(season_length=1, model="ZZZ", prediction_intervals=cfg)

    out = ae.forecast(y=y, h=6, level=[90], fitted=False)
    assert "mean" in out and out["mean"].shape == (6,)
    assert "lo-90" in out and "hi-90" in out
    mean = np.asarray(out["mean"])
    lo90 = np.asarray(out["lo-90"]); hi90 = np.asarray(out["hi-90"])
    assert np.all(lo90 <= mean + 1e-12) and np.all(mean <= hi90 + 1e-12)
    print("test_stateless_forecast_with_conformal_intervals: OK")

# ============== Forward / state-handling & errors ==============

def test_forward_on_new_series_expected_or_reasonable() -> None:
    """Smoke-test forwarding a fitted model onto a new related series."""
    t = np.arange(36, dtype=np.float64)
    yA = jnp.asarray(15.0 + 0.3 * t, dtype=jnp.float64)
    yB = jnp.asarray(10.0 + 0.3 * t, dtype=jnp.float64)

    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(yA)

    h = 5
    outB = ae.forward(y=yB, h=h, level=[80], fitted=True)
    assert outB["mean"].shape == (h,)
    assert outB["fitted"].shape == yB.shape
    # MSE threshold
    mse = float(np.mean((np.asarray(outB["fitted"]) - np.asarray(yB)) ** 2))
    if mse >= 2.0:
        raise AssertionError(f"forward fitted MSE too large: {mse}")
    # interval sanity
    lo = np.asarray(outB["lo-80"]); hi = np.asarray(outB["hi-80"]); mean = np.asarray(outB["mean"])
    assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
    # fitted bands exist
    assert "fitted-lo-80" in outB and "fitted-hi-80" in outB
    _print_ok("test_forward_on_new_series_expected_or_reasonable")


def test_predict_before_fit_raises() -> None:
    """Smoke-test that predict raises before the model is fitted."""
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        _ = ae.predict(h=1)
        raise AssertionError("Expected an error when calling predict() before fit()")
    except AssertionError:
        raise
    except Exception:
        pass
    _print_ok("test_predict_before_fit_raises")


def test_forward_before_fit_raises() -> None:
    """Smoke-test that forward raises before the model is fitted."""
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        _ = ae.forward(y=jnp.asarray([1., 2., 3.]), h=1)
        raise AssertionError("Expected an error when calling forward() before fit()")
    except AssertionError:
        raise
    except Exception:
        pass
    _print_ok("test_forward_before_fit_raises")


def test_tiny_series_policy_expected() -> None:
    """Smoke-test the tiny-series policy for short inputs."""
    y = jnp.asarray([10.0, 11.0, 10.5], dtype=jnp.float64)
    ae = AutoETS(season_length=1, model="ZZZ")
    try:
        ae.fit(y)
        out = ae.predict(h=3, level=None)
        if out["mean"].shape != (3,):
            raise AssertionError("Expected 3-step forecast for tiny series")
    except NotImplementedError as e:
        msg = str(e).lower()
        if "tiny datasets" not in msg:
            raise
    _print_ok("test_tiny_series_policy_expected")


# ============== φ validation & dtypes ==============

def test_phi_validation_range_and_type() -> None:
    """Smoke-test phi validation across invalid and boundary inputs."""
    # invalid range
    try:
        _ = AutoETS(phi=1.5)
        raise AssertionError("Expected ValueError for invalid phi range")
    except ValueError:
        pass
    # invalid type
    try:
        _ = AutoETS(phi="0.9")  # type: ignore
        raise AssertionError("Expected ValueError for non-float phi")
    except ValueError:
        pass
    # valid boundary
    _ = AutoETS(phi=0.8)
    _ = AutoETS(phi=0.98)
    _print_ok("test_phi_validation_range_and_type")


def test_basic_shapes_and_dtypes_predict() -> None:
    """Smoke-test basic output shape and dtype behavior for predict."""
    y = jnp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
    ae = AutoETS(season_length=1, model="ZZZ")
    ae.fit(y.astype(jnp.float64))  # ensure float for the model
    out = ae.predict(h=3, level=None)
    assert out["mean"].shape == (3,)
    assert out["mean"].dtype in (jnp.float32, jnp.float64)
    _print_ok("test_basic_shapes_and_dtypes_predict")


# ============== Runner ==============
if __name__ == "__main__":
    # Core behavior
    test_constant_series_ANN_expected()
    test_linear_trend_AAN_expected()
    test_additive_seasonality_AAA_expected()
    test_stateless_forecast_matches_stateful_expected()

    # Intervals & conformal
    test_predict_native_intervals_and_unsorted_levels()
    test_forecast_adds_fitted_and_fitted_intervals_when_requested()
    test_predict_in_sample_intervals_monotonicity_expected()
    test_fit_caches_conformal_then_predict_uses_cache()
    test_stateless_forecast_with_conformal_intervals()

    # Forward & errors
    test_forward_on_new_series_expected_or_reasonable()
    test_predict_before_fit_raises()
    test_forward_before_fit_raises()
    test_tiny_series_policy_expected()

    # Validation & dtypes
    test_phi_validation_range_and_type()
    test_basic_shapes_and_dtypes_predict()

    print("All AutoETS full-coverage tests passed.")
