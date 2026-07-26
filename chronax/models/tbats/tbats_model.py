"""User-facing AutoTBATS and TBATS forecaster classes.

This module provides:

* :class:`AutoTBATS` — automatic TBATS model selection across Box–Cox,
  trend, damping, and ARMA configurations.
* :class:`TBATS` — fixed-configuration TBATS with sensible defaults
  (Box–Cox on, trend on, damping off, no ARMA).

Both classes wrap the low-level routines in :mod:`tbats_core` and inherit
the common ``fit`` / ``predict`` / ``forecast`` interface from
:class:`BaseForecaster`.

vmap-native contract: :meth:`AutoTBATS.forecast` is
fully stateless — it re-fits on the passed ``y`` and never reads or writes
``self`` — so ``BaseForecaster.conformity_scores`` can vmap it over CV
windows, with every window getting its own honest fit.  Candidates in the
selection grid can differ in state dimension (trend on/off), so all outputs
are assembled per candidate (each with its own static config) and the
winner's rows are selected with ``jnp.take`` on the traced argmin index —
the AutoCES pattern.

Eager input validation (NaN/Inf, strict positivity for forced Box-Cox)
lives in :meth:`fit` only: raising on data VALUES cannot trace.  The
stateless ``forecast`` follows standard JAX semantics instead — NaN in,
NaN out — and a forced-Box-Cox fit on non-positive data degrades via the
core's NaN-lambda sentinel rather than raising.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import config

from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from .tbats_core import (
    _ensure_pos_strict as _ensure_pos_strict,
    _inv_boxcox as _inv_boxcox,
    compute_sigmah as _compute_sigmah,
    tbats_forecast as _tbats_forecast,
    tbats_selection as _tbats_selection,
)
from chronax.utils import (
    _add_fitted_pi,
    _calculate_intervals,
    calculate_sigma as _calculate_sigma,
    ensure_float as _ensure_float,
)

config.update("jax_enable_x64", True)


# ── Utility ────────────────────────────────────────────────────────────

def _clamp_bc_domain(v: jnp.ndarray, lam: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Clamp *v* to the valid domain of the inverse Box–Cox transform.

    For ``λ > 0`` the inverse requires ``1 + λ·v > 0``; for ``λ < 0`` it
    requires ``1 + λ·v < 0``.  The log case (``λ ≈ 0``) has no constraint.

    Trace-native: *lam* may be a traced scalar (it comes out of the fitted
    candidate), so the three λ-regimes are combined with ``jnp.where``
    instead of Python branches.  A NaN λ (the core's degraded-Box-Cox
    sentinel) selects neither clamp and returns *v* unchanged.
    """
    lam = jnp.asarray(lam)
    safe_lam = jnp.where(jnp.abs(lam) < 1e-8, jnp.inf, lam)  # log case: thresh -> -0
    thresh = -1.0 / safe_lam
    return jnp.where(
        lam > 1e-8,
        jnp.maximum(v, thresh + eps),
        jnp.where(lam < -1e-8, jnp.minimum(v, thresh - eps), v),
    )


def _bc_original_scale(v: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Back-transform *v* to the original scale under Box-Cox λ.

    NaN λ (degraded Box-Cox: the candidate was fitted on the raw series)
    means *v* already IS original-scale, so it passes through unchanged.
    """
    return jnp.where(jnp.isnan(lam), v, _inv_boxcox(_clamp_bc_domain(v, lam), lam))


def _select_winner(rows: List[Dict[str, jnp.ndarray]], best: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Stack per-candidate output dicts and take the winner's row per key."""
    return {
        key: jnp.take(jnp.stack([r[key] for r in rows]), best, axis=0)
        for key in rows[0]
    }


def _candidate_predict(cand: Dict[str, Any], h: int, level: Optional[List[int]]) -> Dict[str, jnp.ndarray]:
    """Original-scale h-step outputs for ONE fitted candidate.

    The candidate's Box-Cox-ness is static (None vs jnp scalar λ), so the
    branch on ``lam is None`` is structural and trace-safe.  When Box–Cox is
    active, intervals are built on the transform scale (centred at
    ``mean_bc``) and inverted back; monotonicity (``lo ≤ mean ≤ hi``) is
    enforced to handle ULP edge cases when σ(h) ≈ 0.
    """
    fcst = _tbats_forecast(cand, h)  # {"mean": orig, "mean_bc": transform or None}
    res: Dict[str, jnp.ndarray] = {"mean": fcst["mean"]}

    lam = cand["BoxCox_lambda"]
    if lam is not None:
        # Align mean to the same (clamped) center used for the PIs
        res["mean"] = _bc_original_scale(fcst["mean_bc"], lam)

    if level is not None and len(level) > 0:
        levels = sorted(int(l) for l in level)
        sigmah = _compute_sigmah(cand, h)  # σ(h) on model scale

        if lam is None:
            res.update(_calculate_intervals(res, levels, h, sigmah))
        else:
            pred_int_trans = _calculate_intervals({"mean": fcst["mean_bc"]}, levels, h, sigmah)
            for k, v in pred_int_trans.items():
                res[k] = _bc_original_scale(v, lam)

        # --- Enforce monotonicity: lo ≤ mean ≤ hi (handles σ≈0 ULPs) ---
        m = res["mean"]
        for L in levels:
            lo_k, hi_k = f"lo-{L}", f"hi-{L}"
            res[lo_k] = jnp.minimum(res[lo_k], m)
            res[hi_k] = jnp.maximum(res[hi_k], m)

    return res


def _candidate_forecast(
    cand: Dict[str, Any], h: int, level: Optional[List[int]], fitted: bool,
) -> Dict[str, jnp.ndarray]:
    """`_candidate_predict` plus optional in-sample fitted values and PIs."""
    res = _candidate_predict(cand, h, level)
    if not fitted:
        return res

    lam = cand["BoxCox_lambda"]
    fitted_model_scale = cand["fitted"].ravel()
    res["fitted"] = (
        fitted_model_scale if lam is None else _bc_original_scale(fitted_model_scale, lam)
    )

    if level is not None and len(level) > 0:
        levels = sorted(int(l) for l in level)
        se = _calculate_sigma(cand["errors"], cand["errors"].shape[1])
        fitted_pi = _add_fitted_pi({"fitted": fitted_model_scale}, se, levels)
        for k, v in fitted_pi.items():
            if k == "fitted":
                continue
            res[k] = v if lam is None else _bc_original_scale(v, lam)

        f = res["fitted"]
        for L in levels:
            lo_k, hi_k = f"fitted-lo-{L}", f"fitted-hi-{L}"
            if lo_k in res:
                res[lo_k] = jnp.minimum(res[lo_k], f)
            if hi_k in res:
                res[hi_k] = jnp.maximum(res[hi_k], f)

    return res


def _candidate_in_sample(cand: Dict[str, Any], level: Optional[Tuple[int, ...]]) -> Dict[str, jnp.ndarray]:
    """Original-scale in-sample fitted values (and PIs) for ONE candidate."""
    res: Dict[str, jnp.ndarray] = {"fitted": cand["fitted"].ravel()}

    if level is not None:
        levels = sorted(int(l) for l in level)
        n = int(cand["errors"].shape[1])

        # --- Guard: ensure nonnegative, finite SE and add a tiny floor ---
        se = _calculate_sigma(cand["errors"], n)  # scalar
        se = jnp.asarray(se)
        se = jnp.where(jnp.isfinite(se), se, 0.0)
        se = jnp.maximum(jnp.abs(se), jnp.array(1e-12, dtype=se.dtype))

        sigma_vec = jnp.full((n,), se, dtype=cand["errors"].dtype)

        # Build intervals in model space around fitted
        ints = _calculate_intervals({"mean": res["fitted"]}, levels, n, sigma_vec)
        res = {**res, **ints}

        # --- Enforce monotonicity: lo ≤ fitted ≤ hi ---
        f = res["fitted"]
        for L in levels:
            lo_k, hi_k = f"lo-{L}", f"hi-{L}"
            res[lo_k] = jnp.minimum(res[lo_k], f)
            res[hi_k] = jnp.maximum(res[hi_k], f)

    lam = cand["BoxCox_lambda"]
    if lam is not None:
        res = {k: _bc_original_scale(v, lam) for k, v in res.items()}
    return res


class AutoTBATS(BaseForecaster):
    """Automatic TBATS forecaster with model selection.

    TBATS decomposes a time series into **level**, **trend**, and one or more
    **seasonal** components represented by trigonometric (Fourier) terms,
    with optional **Box–Cox** variance stabilisation and **ARMA** residual
    modelling.

    ``AutoTBATS`` evaluates a grid of configurations (Box–Cox on/off, trend
    on/off, damped trend on/off) and selects the model that minimises AIC.

    Parameters
    ----------
    season_length : int or list of int
        Seasonal period(s).  Pass a single ``int`` for one seasonal cycle
        (e.g. ``12`` for monthly) or a ``list`` for multi-seasonality
        (e.g. ``[7, 365]`` for daily data with weekly + annual cycles).
    use_boxcox : bool or None, default ``None``
        Whether to apply a Box–Cox transformation.  Follows the
        StatsForecast/R convention: ``None`` tries both on and off during
        model selection, ``True`` forces it on, ``False`` forces it off.
    bc_lower_bound : float, default ``-1.0``
        Lower bound for the Box–Cox λ parameter.
    bc_upper_bound : float, default ``2.0``
        Upper bound for the Box–Cox λ parameter.
    use_trend : bool or None, default ``None``
        Whether to include a trend component.  ``None`` tries both.
    use_damped_trend : bool or None, default ``None``
        Whether to damp the trend.  ``None`` tries both.
    use_arma_errors : bool, default ``False``
        Whether to add ARMA structure on the residuals.
    alias : str, default ``"AutoTBATS"``
        Display name for the model.
    conformal_params : ConformalIntervals or None, default ``None``
        Configuration for the base-class ``conformity_scores`` CV path.
        ``predict``/``forecast`` intervals are native (Gaussian σ(h)-based).

    Attributes
    ----------
    model_ : dict or None
        Selection state after :meth:`fit`: the fitted ``candidates`` list,
        the traced ``best`` argmin index, plus winner-selected leaves
        (``aic``, ``fitted``, ``errors``, ``k_vector``, ``BoxCox_lambda``,
        ...).  ``None`` before the first fit.
    only_conformal_intervals : bool
        ``False`` — this model supports **both** native Gaussian intervals
        and conformal intervals.

    Examples
    --------
    >>> tbats = AutoTBATS(season_length=12, use_boxcox=False)
    >>> tbats.fit(y_train)
    >>> tbats.predict(h=12, level=[80, 95])
    {'mean': Array([...]), 'lo-80': ..., 'hi-95': ..., ...}
    """

    uses_exog: bool = False

    def __init__(
        self,
        season_length: Union[int, List[int], str],
        use_boxcox: Optional[bool] = None,
        bc_lower_bound: float = -1.0,
        bc_upper_bound: float = 2.0,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        use_arma_errors: bool = False,
        alias: str = "AutoTBATS",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize the AutoTBATS estimator configuration.

        ``season_length`` may be an int, a list of ints (multi-seasonal), or the
        literal ``"auto"`` to infer a single dominant period from the training
        series at fit (via ``detect_period``; resolved once and cached).
        """
        if isinstance(season_length, str):
            # "auto": resolved once at fit via detect_period. A concrete int/list
            # is used verbatim (statsforecast parity).
            if season_length != "auto":
                raise ValueError(
                    f'season_length string must be "auto", got {season_length!r}'
                )
            self.season_length = "auto"
        else:
            if isinstance(season_length, int):
                season_length = [season_length]
            self.season_length = list(season_length)
        # Resolved concrete period list cached at fit (season_length="auto" ->
        # [detect_period(y)] once); the stateless/vmapped forecast reuses it so a
        # traced CV window never re-detects. None until fit / a fresh forecast.
        self._m_eff: Optional[List[int]] = None
        self.use_boxcox: Optional[bool] = use_boxcox
        self.bc_lower_bound: float = bc_lower_bound
        self.bc_upper_bound: float = bc_upper_bound
        self.use_trend: Optional[bool] = use_trend
        self.use_damped_trend: Optional[bool] = use_damped_trend
        self.use_arma_errors: bool = use_arma_errors

        self.alias: str = alias
        self.conformal_params: Optional[ConformalIntervals] = conformal_params
        self.model_: Optional[Dict[str, Any]] = None
        self.only_conformal_intervals: bool = False

    def _effective_periods(self, y: jnp.ndarray) -> List[int]:
        """Resolve ``season_length`` to a concrete list of periods (eager).

        Returns the fit-cached ``_m_eff`` when present (so the vmapped/stateless
        forecast reuses the fit-time period and never re-detects on a traced CV
        window); otherwise resolves ``"auto"`` → ``[detect_period(y)]`` on the
        concrete series, or returns the explicit list verbatim.
        """
        if self._m_eff is not None:
            return self._m_eff
        if self.season_length == "auto":
            return [self._resolve_season_length("auto", y)]
        return self.season_length

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "AutoTBATS":
        """Fit the TBATS model to training data.

        Runs the full model-selection grid (Box–Cox, trend, damping) and
        stores the fitted candidates + winner index in :attr:`model_`.

        This is the eager, user-facing entry point, so it also VALUE-validates
        the input (which cannot trace):  NaN/Inf raise, and strictly positive
        data is required when Box–Cox is forced on.

        Parameters
        ----------
        y : jnp.ndarray
            One-dimensional time series of shape ``(n,)``.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.

        Returns
        -------
        AutoTBATS
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If *y* contains ``NaN`` or ``Inf`` values, or non-positive values
            with ``use_boxcox=True``.
        """
        y = _ensure_float(y)

        # Resolve season_length="auto" fresh on this series (eager) and cache it
        # so the stateless/vmapped forecast reuses the same period under CV.
        periods = ([self._resolve_season_length("auto", y)]
                   if self.season_length == "auto" else self.season_length)
        self._m_eff = periods

        # Box–Cox forced on -> inputs must be strictly positive.
        if self.use_boxcox is True:
            y = _ensure_pos_strict(y)

        # Friendly heads-up (core will enforce this anyway):
        # when the sample is short relative to the largest season, damped trend and ARMA are curtailed.
        if len(periods) > 0:
            mmax = int(max(periods))
            if y.shape[0] < 3 * mmax:
                warnings.warn(
                    "Short sample vs. seasonality: damped trend and ARMA may be disabled for stability.",
                    RuntimeWarning,
                )

        # Input validation
        if jnp.any(jnp.isnan(y)) or jnp.any(jnp.isinf(y)):
            raise ValueError("Input series contains NaN or Inf values")

        # Fit the full candidate grid with argmin selection
        self.model_ = _tbats_selection(
            y=y,
            seasonal_periods=periods,
            use_boxcox=self.use_boxcox,
            bc_lower=self.bc_lower_bound,
            bc_upper=self.bc_upper_bound,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors,
        )
        return self

    def conformity_scores(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Resolve season_length="auto" eagerly, then delegate to the base CV path.

        The base ``conformity_scores`` vmaps ``self.forecast`` over CV windows and
        does NOT call ``fit`` (the invariant ``new().conformity_scores(y)`` is
        fit-less; TBATS also never caches ``_cs`` at fit). With
        ``season_length="auto"`` and no prior fit, the vmapped forecast's
        ``_effective_periods`` fallback would re-run ``detect_period`` on a *traced*
        window (ConcretizationError). Resolving ``_m_eff`` here on the concrete ``y``
        (eager, before the vmap) makes every vmapped forecast reuse a concrete period
        list — the eager-select-once pattern (AutoARIMA/AutoETS/AutoMFLES).
        """
        y = _ensure_float(y)
        self._m_eff = ([self._resolve_season_length("auto", y)]
                       if self.season_length == "auto" else self.season_length)
        return super().conformity_scores(y=y, X=X)

    def _fitted_candidates(self) -> Tuple[List[Dict[str, Any]], jnp.ndarray]:
        """Return (candidates, best) from model_, with fit/version guards."""
        if getattr(self, "model_", None) is None:
            raise RuntimeError(
                "TBATS model is not fitted yet. Call `fit(y)` before predicting."
            )
        if "candidates" not in self.model_:
            raise RuntimeError(
                "model_ was produced by a pre-vmap-refactor TBATS version; "
                "re-fit the model with fit(y)."
            )
        return self.model_["candidates"], self.model_["best"]

    def predict_in_sample(
        self,
        level: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Return in-sample fitted values (and optional prediction intervals).

        Fitted values live on the model (working) scale.  When Box–Cox is
        active for the winning candidate, they are automatically
        back-transformed to the **original** scale before being returned.

        Parameters
        ----------
        level : tuple of int or None
            Confidence levels in ``[0, 100]``.  When provided, symmetric
            intervals are built around the fitted values using the residual
            standard error, and monotonicity (``lo ≤ fitted ≤ hi``) is
            enforced.

        Returns
        -------
        dict
            ``{"fitted": jnp.ndarray}`` of shape ``(n,)``.  When *level* is
            given, also contains ``"lo-{level}"`` and ``"hi-{level}"``
            keys.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        candidates, best = self._fitted_candidates()
        rows = [_candidate_in_sample(c, level) for c in candidates]
        return _select_winner(rows, best)

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Generate *h*-step-ahead forecasts from the fitted model.

        Every stored candidate is forecast with its own static config and the
        AIC winner's rows are selected with ``jnp.take``.  When Box–Cox is
        active, prediction intervals are built on the **transform** scale
        (centred at ``mean_bc``) and then inverted back to the original
        scale, with monotonicity (``lo ≤ mean ≤ hi``) enforced.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.
        level : list of int or None
            Confidence levels in ``[0, 100]`` for prediction intervals.

        Returns
        -------
        dict
            Always contains ``"mean"`` of shape ``(h,)``.  When *level* is
            given, also contains ``"lo-{level}"`` and ``"hi-{level}"``
            keys.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        candidates, best = self._fitted_candidates()
        rows = [_candidate_predict(c, h, level) for c in candidates]
        return _select_winner(rows, best)

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Stateless fit-and-predict in a single call.

        Runs the full model-selection grid on *y* and produces *h*-step-ahead
        forecasts (optionally with in-sample fitted values and prediction
        intervals).  Deliberately reads and writes NOTHING on ``self`` beyond
        configuration: the base class's ``conformity_scores`` vmaps this
        method over CV windows, so it must trace natively and every window
        must get its own honest re-fit.  Use :meth:`fit` +
        :attr:`model_` when you need to inspect the fitted state.

        Because raising on data VALUES cannot trace, this method does not
        value-validate: NaN/Inf inputs propagate NaN outputs, and a forced
        Box–Cox fit on non-positive data degrades to an untransformed fit via
        the core's NaN-λ sentinel.  :meth:`fit` keeps the eager validation.

        Parameters
        ----------
        y : jnp.ndarray
            One-dimensional time series of shape ``(n,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.
        X_future : jnp.ndarray or None
            Ignored — present for API compatibility.
        level : list of int or None
            Confidence levels in ``[0, 100]`` for prediction intervals.
        fitted : bool, default ``False``
            If ``True``, include in-sample fitted values (back-transformed
            when Box–Cox is active) in the output under ``"fitted"``.

        Returns
        -------
        dict
            Always contains ``"mean"`` of shape ``(h,)``.  Optionally
            includes ``"fitted"``, ``"lo-{level}"``, ``"hi-{level}"``,
            ``"fitted-lo-{level}"``, and ``"fitted-hi-{level}"``.

        Raises
        ------
        ValueError
            If ``use_damped_trend=True`` is combined with ``use_trend=False``
            (config error — static, raised at trace time too).
        """
        y = _ensure_float(y)

        mod = _tbats_selection(
            y=y,
            seasonal_periods=self._effective_periods(y),
            use_boxcox=self.use_boxcox,
            bc_lower=self.bc_lower_bound,
            bc_upper=self.bc_upper_bound,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors,
        )

        rows = [_candidate_forecast(c, h, level, fitted) for c in mod["candidates"]]
        return _select_winner(rows, mod["best"])


class TBATS(AutoTBATS):
    """Fixed-configuration TBATS forecaster.

    A convenience subclass of :class:`AutoTBATS` with sensible defaults for
    a single, fully specified TBATS configuration:

    * Box–Cox **on** (``use_boxcox=True`` — forced, not grid-searched)
    * Trend **on** (``use_trend=True``)
    * Damping **off** (``use_damped_trend=False``)
    * ARMA errors **off** (``use_arma_errors=False``)

    Because the configuration is fixed, no model-selection grid is
    evaluated — :meth:`fit` trains a single candidate model.

    Parameters
    ----------
    season_length : int or list of int
        Seasonal period(s).
    use_boxcox : bool or None, default ``True``
        Apply Box–Cox transformation (``True`` forces on, ``None`` tries
        both, ``False`` forces off).
    bc_lower_bound : float, default ``-1.0``
        Lower bound for the Box–Cox λ parameter.
    bc_upper_bound : float, default ``2.0``
        Upper bound for the Box–Cox λ parameter.
    use_trend : bool or None, default ``True``
        Include a trend component.
    use_damped_trend : bool or None, default ``False``
        Damp the trend toward zero.
    use_arma_errors : bool, default ``False``
        Add ARMA structure on the residuals.
    alias : str, default ``"TBATS"``
        Display name for the model.
    conformal_params : ConformalIntervals or None, default ``None``
        Configuration for the base-class ``conformity_scores`` CV path.

    See Also
    --------
    AutoTBATS : Automatic model selection across TBATS configurations.

    Examples
    --------
    >>> model = TBATS(season_length=12)
    >>> out = model.forecast(y_train, h=12, level=[80, 95])
    """

    def __init__(
        self,
        season_length: Union[int, List[int], str],
        use_boxcox: Optional[bool] = True,
        bc_lower_bound: float = -1.0,
        bc_upper_bound: float = 2.0,
        use_trend: Optional[bool] = True,
        use_damped_trend: Optional[bool] = False,
        use_arma_errors: bool = False,
        alias: str = "TBATS",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize a fixed-configuration TBATS estimator (``season_length`` may be ``"auto"``)."""
        super().__init__(
            season_length=season_length,
            use_boxcox=use_boxcox,
            bc_lower_bound=bc_lower_bound,
            bc_upper_bound=bc_upper_bound,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
            alias=alias,
            conformal_params=conformal_params,
        )
