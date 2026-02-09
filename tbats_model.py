# tbats_model.py
from __future__ import annotations
from typing import List, Optional, Dict, Any, Union, Tuple

import warnings
import jax.numpy as jnp

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
from utils import ensure_float as _ensure_float, calculate_sigma as _calculate_sigma, _add_fitted_pi, _calculate_intervals
from jax import config
config.update("jax_enable_x64", True)

from tbats_core import (
    tbats_selection as _tbats_selection,
    tbats_forecast as _tbats_forecast,
    compute_sigmah as _compute_sigmah,
    _inv_boxcox as _inv_boxcox,
    _boxcox as _boxcox,
    _ensure_pos_strict as _ensure_pos_strict,
)


class AutoTBATS(BaseForecaster):
    """
    AutoTBATS forecaster with automatic model selection.
    """

    uses_exog = False

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
    ):
        if isinstance(season_length, int):
            season_length = [season_length]
        self.season_length = list(season_length)
        self.use_boxcox = use_boxcox
        self.bc_lower_bound = bc_lower_bound
        self.bc_upper_bound = bc_upper_bound
        self.use_trend = use_trend
        self.use_damped_trend = use_damped_trend
        self.use_arma_errors = use_arma_errors

        self.alias = alias
        self.conformal_params = conformal_params
        self.model_: Optional[Dict[str, Any]] = None
        self._cs = None
        self.only_conformal_intervals = False

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None):
        """
        Fit the TBATS model to training data.
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

        # Pre-compute conformity scores if requested
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None

        return self
        
    def predict_in_sample(self, level: Optional[Tuple[int]] = None):
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
            # ---- Safe inverse: clamp to valid domain before _inv_boxcox ----
            def _clamp_bc_domain(v: jnp.ndarray, lam: float, eps: float = 1e-9) -> jnp.ndarray:
                if jnp.abs(lam) < 1e-8:
                    # log case → exp is defined for all real v, no clamp needed
                    return v
                thresh = -1.0 / lam
                if lam > 0:
                    # need v >= -1/lam
                    return jnp.maximum(v, thresh + eps)
                else:
                    # lam < 0 → need v <= -1/lam
                    return jnp.minimum(v, thresh - eps)

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
        """
        Predict with the fitted TBATS model.

        - Requires tbats_core.tbats_forecast to return:
            {"mean": original-scale mean, "mean_bc": transform-scale mean or None}
        - When Box–Cox is active, PIs are built on the transform scale centered at mean_bc,
        then inverted. The returned 'mean' is aligned to inv_boxcox(mean_bc).
        - Finally, we enforce lo ≤ mean ≤ hi to avoid ULP issues when σ(h)≈0.
        """
        if getattr(self, "model_", None) is None:
            raise RuntimeError("TBATS model is not fitted yet. Call `fit(y)` before `predict(h)`.")

        # Core forecast
        fcst = _tbats_forecast(self.model_, h)  # {"mean": orig, "mean_bc": transform or None}
        res: Dict[str, jnp.ndarray] = {"mean": fcst["mean"]}

        lam = self.model_.get("BoxCox_lambda", None)
        if lam is not None:
            def _clamp_bc_domain(v: jnp.ndarray, lam: float, eps: float = 1e-9) -> jnp.ndarray:
                if jnp.abs(lam) < 1e-8:
                    return v
                thresh = -1.0 / lam
                if lam > 0:
                    return jnp.maximum(v, thresh + eps)
                return jnp.minimum(v, thresh - eps)

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
        X: Optional[jnp.ndarray] = None,        # API parity; ignored
        X_future: Optional[jnp.ndarray] = None, # API parity; ignored
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Stateless forecast (fit on `y`, then predict `h`)."""
        y = _ensure_float(y)

        if self.use_boxcox is True:
            y = _ensure_pos_strict(y)

        if jnp.any(jnp.isnan(y)) or jnp.any(jnp.isinf(y)):
            raise ValueError("Input series contains NaN or Inf values")

        mod = _tbats_selection(
            y=y,
            seasonal_periods=self.season_length,
            use_boxcox=self.use_boxcox,
            bc_lower=self.bc_lower_bound,
            bc_upper=self.bc_upper_bound,
            use_trend=self.use_trend,
            use_damped_trend=self.use_damped_trend,
            use_arma_errors=self.use_arma_errors,
        )

        self.model_ = mod

        fcst = _tbats_forecast(mod, h)
        res: Dict[str, jnp.ndarray] = {"mean": fcst["mean"]}

        lam = mod.get("BoxCox_lambda", None)

        def _clamp_bc_domain(v: jnp.ndarray, lam: float, eps: float = 1e-9) -> jnp.ndarray:
            if jnp.abs(lam) < 1e-8:
                return v
            thresh = -1.0 / lam
            if lam > 0:
                return jnp.maximum(v, thresh + eps)
            return jnp.minimum(v, thresh - eps)

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
    """
    TBATS model with fixed configuration.
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
    ):
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