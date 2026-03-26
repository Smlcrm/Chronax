"""User-facing AutoTBATS and TBATS forecaster classes.

This module provides:

* :class:`AutoTBATS` — automatic TBATS model selection across Box–Cox,
  trend, damping, and ARMA configurations.
* :class:`TBATS` — fixed-configuration TBATS with sensible defaults
  (Box–Cox on, trend on, damping off, no ARMA).

Both classes wrap the low-level routines in :mod:`tbats_core` and inherit
the common ``fit`` / ``predict`` / ``forecast`` interface from
:class:`BaseForecaster`.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import config

from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from .tbats_core import (
    _boxcox as _boxcox,
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

def _clamp_bc_domain(v: jnp.ndarray, lam: float, eps: float = 1e-9) -> jnp.ndarray:
    """Clamp *v* to the valid domain of the inverse Box–Cox transform.

    For ``λ > 0`` the inverse requires ``1 + λ·v > 0``; for ``λ < 0`` it
    requires ``1 + λ·v < 0``.  The log case (``λ ≈ 0``) has no constraint.
    """
    if jnp.abs(lam) < 1e-8:
        return v  # log case — exp is defined everywhere
    thresh = -1.0 / lam
    if lam > 0:
        return jnp.maximum(v, thresh + eps)
    return jnp.minimum(v, thresh - eps)


class AutoTBATS(BaseForecaster):
    """Automatic TBATS forecaster with model selection.

    TBATS decomposes a time series into **level**, **trend**, and one or more
    **seasonal** components represented by trigonometric (Fourier) terms,
    with optional **Box–Cox** variance stabilisation and **ARMA** residual
    modelling.

    ``AutoTBATS`` evaluates a grid of configurations (Box–Cox on/off, trend
    on/off, damped trend on/off, ARMA on/off) and selects the model that
    minimises AIC.

    Parameters
    ----------
    season_length : int or list of int
        Seasonal period(s).  Pass a single ``int`` for one seasonal cycle
        (e.g. ``12`` for monthly) or a ``list`` for multi-seasonality
        (e.g. ``[7, 365]`` for daily data with weekly + annual cycles).
    use_boxcox : bool or None, default ``None``
        Whether to apply a Box–Cox transformation.  ``None`` tries both
        on and off during model selection.
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
        Configuration for conformal prediction intervals.

    Attributes
    ----------
    model_ : dict or None
        Full model state after :meth:`fit`, including estimated
        parameters, fitted values, residuals, AIC, Box–Cox λ, etc.
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
        season_length: Union[int, List[int]],
        use_boxcox: Optional[bool] = None,
        bc_lower_bound: float = -1.0,
        bc_upper_bound: float = 2.0,
        use_trend: Optional[bool] = None,
        use_damped_trend: Optional[bool] = None,
        use_arma_errors: bool = False,
        alias: str = "AutoTBATS",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize the AutoTBATS estimator configuration."""
        if isinstance(season_length, int):
            season_length = [season_length]
        self.season_length: List[int] = list(season_length)
        self.use_boxcox: Optional[bool] = use_boxcox
        self.bc_lower_bound: float = bc_lower_bound
        self.bc_upper_bound: float = bc_upper_bound
        self.use_trend: Optional[bool] = use_trend
        self.use_damped_trend: Optional[bool] = use_damped_trend
        self.use_arma_errors: bool = use_arma_errors

        self.alias: str = alias
        self.conformal_params: Optional[ConformalIntervals] = conformal_params
        self.model_: Optional[Dict[str, Any]] = None
        self._cs: Optional[jnp.ndarray] = None
        self.only_conformal_intervals: bool = False

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "AutoTBATS":
        """Fit the TBATS model to training data.

        Runs the full model-selection grid (Box–Cox, trend, damping, ARMA)
        and stores the winning configuration in :attr:`model_`.

        Parameters
        ----------
        y : jnp.ndarray
            One-dimensional time series of shape ``(n,)``.  Must be finite;
            if ``use_boxcox`` is enabled, values must be strictly positive.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.

        Returns
        -------
        AutoTBATS
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If *y* contains ``NaN`` or ``Inf`` values.
        RuntimeWarning
            If the sample is short relative to the largest seasonal period.
        """
        y = _ensure_float(y)

        # Extra defensive: if Box–Cox is enabled, make sure inputs are strictly positive.
        if self.use_boxcox:
            y = _ensure_pos_strict(y)

        # Friendly heads-up (core will enforce this anyway):
        # when the sample is short relative to the largest season, damped trend and ARMA are curtailed.
        if len(self.season_length) > 0:
            mmax = int(max(self.season_length))
            if y.shape[0] < 3 * mmax:
                warnings.warn(
                    "Short sample vs. seasonality: damped trend and ARMA may be disabled for stability.",
                    RuntimeWarning,
                )

        # Input validation
        if jnp.any(jnp.isnan(y)) or jnp.any(jnp.isinf(y)):
            raise ValueError("Input series contains NaN or Inf values")

        # Fit model with automatic selection
        self.model_ = _tbats_selection(
            y=y,
            seasonal_periods=self.season_length,
            use_boxcox=self.use_boxcox,
            bc_lower=self.bc_lower_bound,
            bc_upper=self.bc_upper_bound,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors,
        )

        # Store fitted config so forecast() uses the same config under vmap
        self._fitted_config = self.model_.get("_config", {
            'use_boxcox': self.use_boxcox,
            'use_trend': self.use_trend,
            'use_damped_trend': self.use_damped_trend,
        })

        # Pre-compute conformity scores if requested
        if self.conformal_params is not None:
            saved_model = self.model_  # save before vmap (which overwrites via forecast)
            self._cs = self.conformity_scores(y=y, X=X)
            self.model_ = saved_model  # restore concrete model after vmap
        else:
            self._cs = None

        return self
        
    def predict_in_sample(
        self,
        level: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Return in-sample fitted values (and optional prediction intervals).

        Fitted values live on the model (working) scale.  When Box–Cox was
        used during :meth:`fit`, they are automatically back-transformed to
        the **original** scale before being returned.

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
        if getattr(self, "model_", None) is None:
            raise RuntimeError("TBATS model is not fitted yet. Call `fit(y)` before `predict_in_sample()`.")

        # Fitted is on model/working scale; we’ll back-transform at the end if Box–Cox was used
        res = {"fitted": self.model_["fitted"].ravel()}

        if level is not None:
            levels = sorted(int(l) for l in level)
            n = int(self.model_["errors"].shape[1])

            # --- Guard: ensure nonnegative, finite SE and add a tiny floor ---
            se = _calculate_sigma(self.model_["errors"], n)          # scalar
            se = jnp.asarray(se)
            se = jnp.where(jnp.isfinite(se), se, 0.0)
            se = jnp.maximum(jnp.abs(se), jnp.array(1e-12, dtype=se.dtype))

            sigma_vec = jnp.full((n,), se, dtype=self.model_["errors"].dtype)

            # Build intervals in model space around fitted
            tmp = {"mean": res["fitted"]}
            ints = _calculate_intervals(tmp, levels, n, sigma_vec)
            res = {**res, **ints}

            # --- Enforce monotonicity: lo ≤ fitted ≤ hi ---
            f = res["fitted"]
            for L in levels:
                lo_k, hi_k = f"lo-{L}", f"hi-{L}"
                res[lo_k] = jnp.minimum(res[lo_k], f)
                res[hi_k] = jnp.maximum(res[hi_k], f)

        lam = self.model_.get("BoxCox_lambda", None)
        if lam is not None:
            out = {}
            for k, v in res.items():
                v_clamped = _clamp_bc_domain(v, lam)
                out[k] = _inv_boxcox(v_clamped, lam)  # now guaranteed > 0
            return out

        return res


    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Generate *h*-step-ahead forecasts from the fitted model.

        When Box–Cox is active, prediction intervals are built on the
        **transform** scale (centred at ``mean_bc``) and then inverted back
        to the original scale.  Monotonicity (``lo ≤ mean ≤ hi``) is
        enforced to handle ULP edge cases when σ(h) ≈ 0.

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
        if getattr(self, "model_", None) is None:
            raise RuntimeError("TBATS model is not fitted yet. Call `fit(y)` before `predict(h)`.")

        # Core forecast
        fcst = _tbats_forecast(self.model_, h)  # {"mean": orig, "mean_bc": transform or None}
        res: Dict[str, jnp.ndarray] = {"mean": fcst["mean"]}

        lam = self.model_.get("BoxCox_lambda", None)
        if lam is not None:
            # Align mean to the same center used for PIs
            mean_trans = fcst.get("mean_bc", None)
            if mean_trans is None:
                mean_trans = _boxcox(res["mean"], lam)  # fallback if core not patched
            res["mean"] = _inv_boxcox(_clamp_bc_domain(mean_trans, lam), lam)

        if level is not None and len(level) > 0:
            levels = sorted(int(l) for l in level)
            sigmah = _compute_sigmah(self.model_, h)  # σ(h) on model scale

            if lam is None:
                # No transform: intervals around original-scale mean
                pred_int = _calculate_intervals(res, levels, h, sigmah)
                res.update(pred_int)
            else:
                # Box–Cox: build in transform space, then invert
                mean_trans = fcst.get("mean_bc", None)
                if mean_trans is None:
                    mean_trans = _boxcox(res["mean"], lam)
                pred_int_trans = _calculate_intervals({"mean": mean_trans}, levels, h, sigmah)
                for k, v in pred_int_trans.items():
                    res[k] = _inv_boxcox(_clamp_bc_domain(v, lam), lam)

            # --- Enforce monotonicity: lo ≤ mean ≤ hi (handles σ≈0 ULPs) ---
            m = res["mean"]
            for L in levels:
                lo_k, hi_k = f"lo-{L}", f"hi-{L}"
                res[lo_k] = jnp.minimum(res[lo_k], m)
                res[hi_k] = jnp.maximum(res[hi_k], m)

        return res

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

        Runs the full model-selection grid on *y*, produces *h*-step-ahead
        forecasts, and (optionally) returns in-sample fitted values and
        prediction intervals.  The fitted model is stored in :attr:`model_`
        as a side effect for downstream inspection.

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
            If *y* contains ``NaN`` or ``Inf`` values.
        """
        y = _ensure_float(y)

        # Validation: skip under vmap where values are traced
        try:
            if self.use_boxcox is True:
                y = _ensure_pos_strict(y)
            if jnp.any(jnp.isnan(y)) or jnp.any(jnp.isinf(y)):
                raise ValueError("Input series contains NaN or Inf values")
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError):
            pass  # Under vmap, skip eager validation

        # When called under vmap (from conformity_scores), reuse the fitted
        # model's configuration to ensure a single combo (fixed state-space
        # dimension). This is correct: conformal scores measure how well a
        # SPECIFIC model config performs across windows.
        use_boxcox = self.use_boxcox
        use_trend = self.use_trend
        use_damped_trend = self.use_damped_trend
        k_vector = None
        if hasattr(self, '_fitted_config'):
            use_boxcox = self._fitted_config['use_boxcox']
            use_trend = self._fitted_config['use_trend']
            use_damped_trend = self._fitted_config['use_damped_trend']
        if hasattr(self, 'model_') and self.model_ is not None and 'k_vector' in self.model_:
            k_vector = self.model_['k_vector']

        mod = _tbats_selection(
            y=y,
            seasonal_periods=self.season_length,
            use_boxcox=use_boxcox,
            bc_lower=self.bc_lower_bound,
            bc_upper=self.bc_upper_bound,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=self.use_arma_errors,
            k_vector=k_vector,
        )

        self.model_ = mod

        fcst = _tbats_forecast(mod, h)
        res: Dict[str, jnp.ndarray] = {"mean": fcst["mean"]}

        lam = mod.get("BoxCox_lambda", None)

        if fitted:
            if lam is None:
                res["fitted"] = mod["fitted"].ravel()
            else:
                fitted_bc = mod["fitted"].ravel()
                res["fitted"] = _inv_boxcox(_clamp_bc_domain(fitted_bc, lam), lam)

        if level is not None:
            levels = sorted(int(l) for l in level)
            sigmah = _compute_sigmah(mod, h)

            if lam is None:
                pred_int = _calculate_intervals(res, levels, h, sigmah)
                res.update(pred_int)
                if fitted:
                    se = _calculate_sigma(mod["errors"], mod["errors"].shape[1])
                    fitted_pred_int = _add_fitted_pi({"fitted": res["fitted"]}, se, levels)
                    res.update(fitted_pred_int)
            else:
                mean_trans = fcst.get("mean_bc", None)
                if mean_trans is None:
                    mean_trans = _boxcox(res["mean"], lam)

                pred_int_trans = _calculate_intervals({"mean": mean_trans}, levels, h, sigmah)
                for k, v in pred_int_trans.items():
                    res[k] = _inv_boxcox(_clamp_bc_domain(v, lam), lam)

                if fitted:
                    se = _calculate_sigma(mod["errors"], mod["errors"].shape[1])
                    fitted_bc = mod["fitted"].ravel()
                    fitted_int_trans = _add_fitted_pi({"fitted": fitted_bc}, se, levels)
                    for k, v in fitted_int_trans.items():
                        if k != "fitted":
                            res[k] = _inv_boxcox(_clamp_bc_domain(v, lam), lam)

            m = res["mean"]
            for L in levels:
                lo_k, hi_k = f"lo-{L}", f"hi-{L}"
                res[lo_k] = jnp.minimum(res[lo_k], m)
                res[hi_k] = jnp.maximum(res[hi_k], m)

            if fitted:
                f = res["fitted"]
                for L in levels:
                    lo_k, hi_k = f"fitted-lo-{L}", f"fitted-hi-{L}"
                    if lo_k in res:
                        res[lo_k] = jnp.minimum(res[lo_k], f)
                    if hi_k in res:
                        res[hi_k] = jnp.maximum(res[hi_k], f)

        return res



class TBATS(AutoTBATS):
    """Fixed-configuration TBATS forecaster.

    A convenience subclass of :class:`AutoTBATS` with sensible defaults for
    a single, fully specified TBATS configuration:

    * Box–Cox **on** (``use_boxcox=True``)
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
        Apply Box–Cox transformation.
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
        Configuration for conformal prediction intervals.

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
        season_length: Union[int, List[int]],
        use_boxcox: Optional[bool] = True,
        bc_lower_bound: float = -1.0,
        bc_upper_bound: float = 2.0,
        use_trend: Optional[bool] = True,
        use_damped_trend: Optional[bool] = False,
        use_arma_errors: bool = False,
        alias: str = "TBATS",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize a fixed-configuration TBATS estimator."""
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
