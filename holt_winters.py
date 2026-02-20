"""
Holt-Winters' Exponential Smoothing — JAX-accelerated implementation.

Implements triple exponential smoothing with additive/multiplicative error
and seasonality variants, damped trend, and two-phase ADAM + L-BFGS optimization.

Features:
- Additive and multiplicative error models
- Additive and multiplicative seasonality
- Damped and non-damped trend variants
- Native prediction intervals (analytical formulas)
- Conformal prediction intervals support
- JAX JIT compilation for performance
- Four prediction methods: fit/predict, predict_in_sample, forecast, forward

Instance Attributes:
1. season_length: int - Number of observations per unit of time (e.g., 12 for monthly)
2. error_type: str - Type of error: 'A' (additive) or 'M' (multiplicative)
3. season_type: str - Type of seasonality: 'A' (additive) or 'M' (multiplicative)
4. damped: bool - Whether to use damped trend
5. phi: float | None - Damping parameter (0.8-0.98), used only if damped=True
6. alias: str - Custom name for the model
7. conformal_params: ConformalIntervals | None - Parameters for conformal prediction intervals
8. model_: dict - Fitted model parameters (created after fit())
   - fitted: In-sample fitted values
   - level: Final level state
   - trend: Final trend state
   - seasonal: Final seasonal states (array of length season_length)
   - alpha, beta, gamma: Estimated smoothing parameters
   - sigma: Residual standard error
   - residuals: Forecast residuals
   - y_train: Original training data (for conformal prediction)

Methods:
1. __init__() - Initialize with error, season, and damping parameters
2. fit(y, X=None) - Fit model to training data via ADAM + L-BFGS optimization
3. predict(h, X=None, level=None) - Generate forecasts with fitted model
4. predict_in_sample(level=None) - Return fitted values with optional intervals
5. forecast(y, h, X=None, X_future=None, level=None, fitted=False) - Stateless prediction
6. forward(y, h, X=None, X_future=None, level=None, fitted=False) - Apply fitted model to new data

Implementation Notes:
- Sigmoid reparameterization for admissibility-constrained smoothing parameters
- States initialized via classical decomposition with trend estimation
- Supports 8 model variants: AAA, AAM, MAA, MAM (+ damped versions)
- Analytical interval formulas from Hyndman et al. (2008) and Taylor (2003)
- Seasonal constraint: additive sums to 0, multiplicative averages to 1
"""
import jax
import jax.numpy as jnp
import optax
import utils
from conformal_intervals import ConformalIntervals
from base_forecaster import BaseForecaster
from jax import lax


# =============================================================================
# Constants
# =============================================================================

_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

_EPS_PARAM = 1e-4
_ADAM_STEPS = 100
_ADAM_LR = 0.05
_LBFGS_STEPS = 50
_LBFGS_MEMORY = 8
_LBFGS_LS_STEPS = 20
_EPSILON = 1e-10

__all__ = ['HoltWinters']


# =============================================================================
# Core Math — Module-level JIT'd functions
# =============================================================================

def _to_constrained(p_raw, l0_decomp, b0_decomp, s0_decomp,
                    l_scale, b_scale, s_scale,
                    is_additive_season, season_length):
    """Convert unconstrained parameters to constrained space via sigmoid."""
    m = season_length
    eps = _EPS_PARAM

    alpha = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[0])
    beta  = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[1])
    gamma = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[2])

    l0 = l0_decomp + p_raw[3] * l_scale
    b0 = b0_decomp + p_raw[4] * b_scale
    s_free = s0_decomp[:m-1] + p_raw[5:m+4] * s_scale

    if is_additive_season:
        s_last = -jnp.sum(s_free)
    else:
        s_last = m - jnp.sum(s_free)
    s0 = jnp.concatenate([s_free, s_last[None]])

    return alpha, beta, gamma, l0, b0, s0


def _hw_core(alpha, beta, gamma, l0, b0, s0, y,
             phi, is_additive_error, is_additive_season, season_length):
    """Forward pass and MLE loss for Holt-Winters.

    Returns loss, fitted_vals, final_level, final_trend, final_seasonal, residuals.
    """
    m = season_length
    n = y.shape[0]

    def step_AAA(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = level_prev + phi_trend + s_prev
        level = a * (y_t - s_prev) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t - level) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    def step_AAM(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = (level_prev + phi_trend) * s_prev
        level = a * (y_t / jnp.maximum(s_prev, _EPSILON)) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t / jnp.maximum(level, _EPSILON)) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    def step_MAA(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = level_prev + phi_trend + s_prev
        level = a * (y_t - s_prev) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t - level) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    def step_MAM(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = (level_prev + phi_trend) * s_prev
        level = a * (y_t / jnp.maximum(s_prev, _EPSILON)) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t / jnp.maximum(level, _EPSILON)) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    if is_additive_error and is_additive_season:
        step_fn = step_AAA
    elif is_additive_error and not is_additive_season:
        step_fn = step_AAM
    elif not is_additive_error and is_additive_season:
        step_fn = step_MAA
    else:
        step_fn = step_MAM

    init_carry = (l0, b0, s0, alpha, beta, gamma)
    final_carry, fitted_vals = lax.scan(step_fn, init_carry, y)
    final_level, final_trend, final_seasonal, _, _, _ = final_carry
    residuals = y - fitted_vals

    sse = jnp.sum(residuals ** 2)
    loss = n * jnp.log(jnp.maximum(sse, _EPSILON))

    return loss, fitted_vals, final_level, final_trend, final_seasonal, residuals


# =============================================================================
# Optimizer — ADAM + L-BFGS
# =============================================================================

def _optimize_hw(y, l0_decomp, b0_decomp, s0_decomp,
                 phi, is_additive_error, is_additive_season, season_length):
    """Two-phase ADAM + L-BFGS optimization for Holt-Winters.

    Parameters
    ----------
    y : jnp.ndarray
        Time series of shape (n,).
    l0_decomp, b0_decomp, s0_decomp : float/jnp.ndarray
        Initial state estimates from classical decomposition.
    phi : float
        Damping factor (static arg).
    is_additive_error : bool
        Whether error type is additive (static arg).
    is_additive_season : bool
        Whether season type is additive (static arg).
    season_length : int
        Number of observations per season (static arg).

    Returns
    -------
    tuple
        (fitted, level, trend, seasonal, residuals, alpha, beta, gamma)
    """
    m = season_length
    dtype = y.dtype

    l_scale = jnp.std(y)
    b_scale = jnp.std(jnp.diff(y))
    if is_additive_season:
        s_scale = jnp.std(y)
    else:
        s_scale = jnp.std(y) / jnp.maximum(jnp.mean(y), _EPSILON)

    def loss_fn(p_raw):
        p_raw = jnp.asarray(p_raw, dtype=dtype)
        alpha, beta, gamma, l0, b0, s0 = _to_constrained(
            p_raw, l0_decomp, b0_decomp, s0_decomp,
            l_scale, b_scale, s_scale, is_additive_season, m)
        loss, *_ = _hw_core(alpha, beta, gamma, l0, b0, s0, y,
                            phi, is_additive_error, is_additive_season, m)
        return loss

    vg_fn = jax.value_and_grad(loss_fn)

    # sigmoid(-2.0) ≈ 0.12, sigmoid(-5.0) ≈ 0.007
    p0_smooth = jnp.array([-2.0, -5.0, -2.0], dtype=dtype)
    p0_states = jnp.zeros(m + 1, dtype=dtype)
    p0 = jnp.concatenate([p0_smooth, p0_states])

    inf_val = jnp.asarray(float('inf'), dtype=dtype)

    # Phase 1: ADAM warm-up
    adam_opt = optax.adam(_ADAM_LR)
    adam_state = adam_opt.init(p0)

    def adam_step(carry, _):
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = adam_opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_p = jnp.where(improved, new_p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    adam_carry = (p0, adam_state, p0, inf_val)
    (_, _, adam_best_p, adam_best_loss), _ = lax.scan(
        adam_step, adam_carry, None, length=_ADAM_STEPS)

    # Phase 2: L-BFGS refinement
    lbfgs_solver = optax.lbfgs(
        memory_size=_LBFGS_MEMORY,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=_LBFGS_LS_STEPS, initial_guess_strategy="one"))
    lbfgs_state = lbfgs_solver.init(adam_best_p)

    def lbfgs_step(carry, _):
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = lbfgs_solver.update(
            grads, state, p, value=loss, grad=grads, value_fn=loss_fn)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_p = jnp.where(improved, p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    lbfgs_carry = (adam_best_p, lbfgs_state, adam_best_p, adam_best_loss)
    (_, _, lbfgs_best_p, lbfgs_best_loss), _ = lax.scan(
        lbfgs_step, lbfgs_carry, None, length=_LBFGS_STEPS)

    # Pick best
    use_lbfgs = jnp.isfinite(lbfgs_best_loss) & (lbfgs_best_loss < adam_best_loss)
    best_p = jnp.where(use_lbfgs, lbfgs_best_p, adam_best_p)

    # Final forward pass with optimal params
    alpha, beta, gamma, l0, b0, s0 = _to_constrained(
        best_p, l0_decomp, b0_decomp, s0_decomp,
        l_scale, b_scale, s_scale, is_additive_season, m)
    loss, fitted, level, trend, seasonal, residuals = _hw_core(
        alpha, beta, gamma, l0, b0, s0, y,
        phi, is_additive_error, is_additive_season, m)

    return fitted, level, trend, seasonal, residuals, alpha, beta, gamma

_optimize_hw = jax.jit(_optimize_hw, static_argnums=(4, 5, 6, 7))


# =============================================================================
# HoltWinters Class
# =============================================================================

class HoltWinters(BaseForecaster):
    r"""Holt-Winters' seasonal exponential smoothing.

    Parameters
    ----------
    season_length : int, default 12
        Number of observations per unit of time. Must be >= 2.
    error_type : str, default 'A'
        Error type: 'A' (additive) or 'M' (multiplicative).
    season_type : str, default 'A'
        Seasonality type: 'A' (additive) or 'M' (multiplicative).
    damped : bool | None, default None
        Whether to use damped trend. None treated as False.
    phi : float | None, default None
        Damping parameter in [0.8, 0.98]. Only used if damped=True.
    alias : str, default 'HoltWinters'
        Custom name for the model.
    conformal_params : ConformalIntervals | None, default None
        Parameters for conformal prediction intervals.
    allow_extended_iterations : bool, default False
        No-op, kept for API compatibility.
    iteration_scaling : str, default 'quadratic'
        No-op, kept for API compatibility.
    """

    @staticmethod
    def _validate_h(h: int) -> None:
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be a positive integer, got {h}")

    @staticmethod
    def _validate_level(level: list[int] | None) -> None:
        if level is not None:
            if not isinstance(level, list):
                raise ValueError(f"level must be a list or None, got {type(level).__name__}")
            if any(not isinstance(lv, (int, float)) or lv < 0 or lv > 100 for lv in level):
                raise ValueError("All level values must be numbers between 0 and 100")

    def _initialize_states(self, y: jnp.ndarray) -> tuple[float, float, jnp.ndarray]:
        """Initialize level, trend, and seasonal states via classical decomposition.

        Returns (l0, b0, s0) where s0 has shape (season_length,).
        """
        m = self.season_length
        n = len(y)
        n_seasons = n // m

        if n_seasons < 2:
            l0 = float(jnp.mean(y[:m])) if n >= m else float(y[0])
            b0 = 0.0
            if n >= m:
                s0 = y[:m] - l0 if self.season_type == 'A' else y[:m] / jnp.maximum(l0, _EPSILON)
            else:
                s0 = jnp.zeros(m, dtype=jnp.float32) if self.season_type == 'A' else jnp.ones(m, dtype=jnp.float32)
            return l0, b0, s0.astype(jnp.float32)

        n_complete = n_seasons * m
        y_reshaped = y[:n_complete].reshape(n_seasons, m)
        seasonal_avgs = jnp.mean(y_reshaped, axis=1)

        # OLS on seasonal averages for trend
        t = jnp.arange(n_seasons, dtype=jnp.float32)
        t_mean = jnp.mean(t)
        avg_mean = jnp.mean(seasonal_avgs)
        cov_ta = jnp.sum((t - t_mean) * (seasonal_avgs - avg_mean))
        var_t = jnp.sum((t - t_mean) ** 2)
        b0_per_season = float(cov_ta / jnp.maximum(var_t, _EPSILON))
        l0 = float(avg_mean - b0_per_season * t_mean)
        b0 = b0_per_season / m
        l0 = l0 - b0 * (m - 1) / 2.0

        # Detrend and compute seasonal pattern
        trend_vals = l0 + b0 * jnp.arange(n_complete, dtype=jnp.float32)
        y_complete = y[:n_complete]
        if self.season_type == 'A':
            detrended = y_complete - trend_vals
        else:
            detrended = y_complete / jnp.maximum(trend_vals, _EPSILON)

        s0 = jnp.mean(detrended.reshape(n_seasons, m), axis=0)

        # Normalize: additive sums to 0, multiplicative averages to 1
        if self.season_type == 'A':
            s0 = s0 - jnp.mean(s0)
        else:
            s0 = s0 / jnp.maximum(jnp.mean(s0), _EPSILON)

        return l0, b0, s0.astype(jnp.float32)

    def _get_phi(self) -> float:
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        return 1.0

    def _compute_base_variance(self, t, alpha, beta, phi):
        """Compute base variance for trend component (Hyndman et al. 2008)."""
        if self.damped and phi < 0.9999:
            denom = jnp.maximum(1 - phi, _EPSILON)
            denom2 = jnp.maximum(1 - phi**2, _EPSILON)
            trend_var = (beta * phi * t) / denom**2 * (2 * alpha * denom + beta * phi) \
                       - (beta * phi * (1 - phi**t)) / (denom**2 * denom2) \
                       * (2 * alpha * denom2 + beta * phi * (1 + 2 * phi - phi**t))
            return 1 + alpha**2 * (t - 1) + trend_var
        else:
            exp1 = alpha**2 + alpha * beta * t + (1 / 6) * beta**2 * t * (2 * t - 1)
            return 1 + (t - 1) * exp1

    def _validate_forecast_inputs(self, y, h, level):
        """Validate and convert inputs for forecast/forward methods."""
        y = utils.ensure_float(y)
        if len(y) < self.season_length:
            raise ValueError(
                f"Time series must have at least {self.season_length} observations "
                f"(season_length), got {len(y)}"
            )
        self._validate_h(h)
        self._validate_level(level)
        return y

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Fit model parameters and return results dictionary."""
        l0, b0, s0 = self._initialize_states(y)
        phi = self._get_phi()
        is_additive_error = self.error_type == 'A'
        is_additive_season = self.season_type == 'A'

        fitted, level, trend, seasonal, residuals, alpha, beta, gamma = _optimize_hw(
            y, l0, b0, s0, phi, is_additive_error, is_additive_season, self.season_length)

        n_params = 3 + 2 + (self.season_length - 1)
        return {
            'fitted': fitted, 'level': level, 'trend': trend,
            'seasonal': seasonal, 'residuals': residuals,
            'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
            'sigma': utils.calculate_sigma(residuals, len(y) - n_params),
        }

    def _generate_forecasts(self, level, trend, seasonal, phi, h):
        """Generate h-step ahead point forecasts (vectorized)."""
        m = self.season_length
        t_vals = jnp.arange(1, h + 1, dtype=jnp.float32)

        if phi == 1.0:
            trend_components = t_vals * trend
        else:
            trend_components = phi * (1 - phi**t_vals) / (1 - phi) * trend

        s_components = seasonal[jnp.arange(h) % m]

        if self.season_type == 'A':
            return level + trend_components + s_components
        else:
            return (level + trend_components) * s_components

    def _calculate_native_intervals(self, mean, sigma, alpha, beta, gamma, phi, h):
        """Calculate prediction interval width (sigmah) using analytical formulas.

        Based on Hyndman et al. (2008) and Taylor (2003).
        """
        t = jnp.arange(1, h + 1, dtype=jnp.float32)
        base_var = self._compute_base_variance(t, alpha, beta, phi)
        seasonal_var = gamma**2 * ((t - 1) // self.season_length + 1)

        if self.error_type == 'A':
            if self.season_type == 'A':
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var)
            else:
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var) * jnp.abs(mean)
        else:
            sigmah = sigma * jnp.sqrt(base_var + seasonal_var) * jnp.abs(mean)

        return sigmah

    def _add_interval_bounds(self, res, values, sigmah, level, prefix=''):
        """Add lo/hi prediction interval bounds to result dict."""
        for lv in reversed(level):
            alpha_level = (100 - lv) / 100
            z = utils._jax_norm_ppf(1 - alpha_level / 2)
            res[f'{prefix}lo-{lv}'] = values - z * sigmah
            res[f'{prefix}hi-{lv}'] = values + z * sigmah
        return res

    def _compute_forecast_with_intervals(self, result, phi, h, level, fitted, y, X):
        """Generate forecasts and optionally add intervals and fitted values."""
        mean = self._generate_forecasts(
            result['level'], result['trend'], result['seasonal'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                temp_model = getattr(self, 'model_', None)
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res, cs=cs, level=level,
                    method=self.conformal_params.method)
                if temp_model is not None:
                    self.model_ = temp_model
                else:
                    delattr(self, 'model_')
            else:
                sigmah = self._calculate_native_intervals(
                    mean, result['sigma'], result['alpha'],
                    result['beta'], result['gamma'], phi, h)
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res, result['fitted'], result['sigma'], level, prefix='fitted-')

        return res

    def __init__(
        self,
        season_length: int = 12,
        error_type: str = 'A',
        season_type: str = 'A',
        damped: bool | None = None,
        phi: float | None = None,
        alias: str = "HoltWinters",
        conformal_params: ConformalIntervals | None = None,
        allow_extended_iterations: bool = False,
        iteration_scaling: str = "quadratic",
    ):
        if not isinstance(season_length, int) or season_length < 2:
            raise ValueError(f"season_length must be an integer >= 2, got {season_length}")
        if error_type not in ('A', 'M'):
            raise ValueError(f"error_type must be 'A' or 'M', got '{error_type}'")
        if season_type not in ('A', 'M'):
            raise ValueError(f"season_type must be 'A' or 'M', got '{season_type}'")
        if phi is not None:
            if not isinstance(phi, (float, int)):
                raise ValueError(f"phi must be None or a number, got {type(phi).__name__}")
            phi = float(phi)
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"phi must be in range [{_PHI_LOWER}, {_PHI_UPPER}], got {phi}")
        if conformal_params is not None and not isinstance(conformal_params, ConformalIntervals):
            raise ValueError(
                f"conformal_params must be a ConformalIntervals instance, got {type(conformal_params).__name__}"
            )
        if iteration_scaling not in ("cubic", "quadratic"):
            raise ValueError(f"iteration_scaling must be 'cubic' or 'quadratic', got '{iteration_scaling}'")

        self.season_length = season_length
        self.error_type = error_type
        self.season_type = season_type
        self.damped = damped if damped is not None else False
        self.phi = phi
        self.alias = alias
        self.conformal_params = conformal_params
        self.allow_extended_iterations = allow_extended_iterations
        self.iteration_scaling = iteration_scaling

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> 'HoltWinters':
        r"""Fit the Holt-Winters model.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,).
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (n, n_x).

        Returns
        -------
        HoltWinters
            Fitted model instance.
        """
        y = utils.ensure_float(y)
        if len(y) < self.season_length:
            raise ValueError(
                f"Time series must have at least {self.season_length} observations "
                f"(season_length), got {len(y)}"
            )
        result = self._fit_parameters(y)
        result['y_train'] = y
        self.model_ = result
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None,
                level: list[int] | None = None) -> dict:
        r"""Predict with fitted Holt-Winters.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (h, n_x).
        level : list of int or None, default None
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Keys: 'mean' and optionally 'lo-{lv}', 'hi-{lv}'.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict(). Call fit() first.")
        self._validate_h(h)
        self._validate_level(level)

        phi = self._get_phi()
        mean = self._generate_forecasts(
            self.model_['level'], self.model_['trend'],
            self.model_['seasonal'], phi, h)
        res = {'mean': mean}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                cs = self.conformity_scores(y=self.model_['y_train'], X=X)
                res = self.add_confidence_intervals(
                    fcst=res, cs=cs, level=level,
                    method=self.conformal_params.method)
            else:
                sigmah = self._calculate_native_intervals(
                    mean, self.model_['sigma'], self.model_['alpha'],
                    self.model_['beta'], self.model_['gamma'], phi, h)
                res = self._add_interval_bounds(res, mean, sigmah, level)

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        r"""Access fitted Holt-Winters in-sample predictions.

        Parameters
        ----------
        level : list of int or None, default None
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Keys: 'fitted' and optionally 'fitted-lo-{lv}', 'fitted-hi-{lv}'.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict_in_sample(). Call fit() first.")
        self._validate_level(level)

        res = {'fitted': self.model_['fitted']}
        if level is not None:
            level = sorted(level)
            res = self._add_interval_bounds(
                res, self.model_['fitted'], self.model_['sigma'],
                level, prefix='fitted-')
        return res

    def forecast(self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
                 X_future: jnp.ndarray | None = None, level: list[int] | None = None,
                 fitted: bool = False) -> dict:
        r"""Memory-efficient Holt-Winters predictions.

        Fits and forecasts without storing model state.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,).
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (n, n_x).
        X_future : jnp.ndarray or None, default None
            Optional future exogenous of shape (h, n_x).
        level : list of int or None, default None
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default False
            Whether to return in-sample predictions.

        Returns
        -------
        dict
            Keys: 'mean', and optionally 'fitted', 'lo-{lv}', 'hi-{lv}'.
        """
        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)

    def forward(self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
                X_future: jnp.ndarray | None = None, level: list[int] | None = None,
                fitted: bool = False) -> dict:
        r"""Apply fitted Holt-Winters model to a new time series.

        Uses the model structure from the original fit but re-estimates parameters.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,).
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Optional exogenous of shape (n, n_x).
        X_future : jnp.ndarray or None, default None
            Optional future exogenous of shape (h, n_x).
        level : list of int or None, default None
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default False
            Whether to return in-sample predictions.

        Returns
        -------
        dict
            Keys: 'mean', and optionally 'fitted', 'lo-{lv}', 'hi-{lv}'.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling forward(). Call fit() first.")
        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)
