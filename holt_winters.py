"""
Holt-Winters' Exponential Smoothing Model.

This module implements Holt-Winters' seasonal method (triple exponential smoothing)
with full JAX acceleration and compatibility with statsforecast's API.

Features:
- Additive and multiplicative error models
- Additive and multiplicative seasonality
- Damped and non-damped trend variants
- Native prediction intervals (analytical formulas)
- Conformal prediction intervals support
- JAX JIT compilation for performance
- Four prediction methods: fit/predict, predict_in_sample, forecast, forward

The model fits level, trend, and seasonal smoothing parameters (alpha, beta, gamma)
using maximum likelihood optimization via gradient descent.

Instance Attributes:
1. season_length: int - Number of observations per unit of time (e.g., 12 for monthly)
2. error_type: str - Type of error: 'A' (additive) or 'M' (multiplicative)
3. season_type: str - Type of seasonality: 'A' (additive) or 'M' (multiplicative)
4. damped: bool - Whether to use damped trend
5. phi: float | None - Damping parameter (0.8-0.98), used only if damped=True
6. alias: str - Custom name for the model
7. conformal_params: ConformalIntervals | None - Parameters for conformal prediction intervals
8. allow_extended_iterations: bool - Whether to allow extended iteration counts for difficult series
9. iteration_scaling: str - Scaling method for adaptive iterations ("quadratic" or "cubic")
10. model_: dict - Fitted model parameters (created after fit())
   - fitted: In-sample fitted values
   - level: Final level state
   - trend: Final trend state
   - seasonal: Final seasonal states (array of length season_length)
   - alpha: Estimated level smoothing parameter
   - beta: Estimated trend smoothing parameter
   - gamma: Estimated seasonal smoothing parameter
   - sigma: Residual standard error
   - residuals: Forecast residuals
   - y_train: Original training data (for conformal prediction)

Class Attributes:
1. uses_exog: bool - Whether model supports exogenous variables (False for Holt-Winters)

Methods:
1. __init__() - Initialize Holt-Winters model with error, season, and damping parameters
2. fit(y, X=None) - Fit model to training data, estimates alpha, beta, gamma via optimization
3. predict(h, X=None, level=None) - Generate forecasts with fitted model
4. predict_in_sample(level=None) - Return fitted values with optional intervals
5. forecast(y, h, X=None, X_future=None, level=None, fitted=False) - Stateless prediction
6. forward(y, h, X=None, X_future=None, level=None, fitted=False) - Apply fitted model to new data

Helper Methods:
- _validate_h() - Validate forecast horizon parameter
- _validate_level() - Validate prediction interval levels
- _initialize_states() - Initialize level, trend, and seasonal states via decomposition
- _get_phi() - Get damping factor
- _estimate_iterations() - Estimate optimal iteration count based on data complexity
- _fit_parameters() - Core optimization routine using JAX/optax
- _generate_forecasts() - Compute h-step ahead point forecasts with seasonality
- _calculate_native_intervals() - Analytical prediction interval formulas with seasonality
- _add_interval_bounds() - Add interval bounds to result dictionary
- _compute_base_variance() - Compute base variance for trend component
- _validate_forecast_inputs() - Validate inputs for forecast/forward methods
- _compute_forecast_with_intervals() - Generate forecasts and add intervals

Implementation Notes:
- Uses optax.adam optimizer with exponential learning rate decay
- Adaptive iteration count based on data complexity (50-600 iterations)
- Module-level JIT functions for efficient compilation caching
- Supports 8 model variants: AAA, AAM, MAA, MAM (+ damped versions)
- Requires at least season_length observations for fitting
- States initialized via classical decomposition with trend estimation
- Analytical interval formulas from Hyndman et al. (2008) and Taylor (2003)
- Seasonal normalization: additive sums to 0, multiplicative averages to 1
"""
import jax
import jax.numpy as jnp
import utils
import optax
from conformal_intervals import ConformalIntervals
from base_forecaster import BaseForecaster
from jax import lax

# Validation constants
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

# Optimization constants
_INIT_ALPHA = 0.3
_INIT_BETA = 0.1
_INIT_GAMMA = 0.1
_N_PARAMS = 3  # alpha, beta, and gamma
_LEARNING_RATE = 0.01
_LR_DECAY_STEPS = 500
_LR_DECAY_RATE = 0.9
_EPSILON = 1e-10  # For numerical stability

# Adaptive iteration constants
_MIN_ITER = 50
_MAX_ITER = 350
_MAX_ITER_EXTENDED = 600

__all__ = ['HoltWinters']


# =============================================================================
# Module-level JIT-compiled optimization functions
# =============================================================================

def _run_hw_optimization(y, l0, b0, s0, phi, is_additive_error, is_additive_season, season_length, n_iters):
    """Module-level JIT-compiled optimization loop for Holt-Winters.

    Parameters
    ----------
    y : jnp.ndarray
        Time series data
    l0 : float
        Initial level
    b0 : float
        Initial trend
    s0 : jnp.ndarray
        Initial seasonal states (length season_length)
    phi : float
        Damping factor (1.0 for non-damped)
    is_additive_error : bool
        True for additive error, False for multiplicative
    is_additive_season : bool
        True for additive seasonality, False for multiplicative
    season_length : int
        Number of periods in a season
    n_iters : int
        Number of optimization iterations

    Returns
    -------
    tuple
        (best_params, raw_fit_result) where best_params is [alpha, beta, gamma]
        and raw_fit_result is dict with fitted values, level, trend, seasonal, residuals
    """
    scheduler = optax.exponential_decay(
        init_value=_LEARNING_RATE,
        transition_steps=_LR_DECAY_STEPS,
        decay_rate=_LR_DECAY_RATE
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=scheduler)
    )

    n = len(y)
    m = season_length

    # Define step functions for all 4 model variants
    def step_AAA(carry, y_t):
        """Additive error, Additive seasonality (AAA)."""
        level_prev, trend_prev, seasonal_prev, alpha, beta, gamma = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[m - 1]  # s_{t-m}

        y_hat = level_prev + phi_trend + s_prev

        level = alpha * (y_t - s_prev) + (1 - alpha) * (level_prev + phi_trend)
        trend = beta * (level - level_prev) + (1 - beta) * phi_trend
        new_seasonal = gamma * (y_t - level) + (1 - gamma) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(jnp.float32(new_seasonal))

        return (level, trend, seasonal_new, alpha, beta, gamma), y_hat

    def step_AAM(carry, y_t):
        """Additive error, Multiplicative seasonality (AAM)."""
        level_prev, trend_prev, seasonal_prev, alpha, beta, gamma = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[m - 1]  # s_{t-m}

        y_hat = (level_prev + phi_trend) * s_prev

        level = alpha * (y_t / jnp.maximum(s_prev, _EPSILON)) + (1 - alpha) * (level_prev + phi_trend)
        trend = beta * (level - level_prev) + (1 - beta) * phi_trend
        new_seasonal = gamma * (y_t / jnp.maximum(level, _EPSILON)) + (1 - gamma) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(jnp.float32(new_seasonal))

        return (level, trend, seasonal_new, alpha, beta, gamma), y_hat

    def step_MAA(carry, y_t):
        """Multiplicative error, Additive seasonality (MAA)."""
        level_prev, trend_prev, seasonal_prev, alpha, beta, gamma = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[m - 1]  # s_{t-m}

        y_hat = level_prev + phi_trend + s_prev
        epsilon = (y_t - y_hat) / jnp.maximum(jnp.abs(y_hat), _EPSILON)

        level = (level_prev + phi_trend - s_prev) + (level_prev + phi_trend) * alpha * epsilon
        trend = phi_trend + beta * (level_prev + phi_trend) * epsilon
        new_seasonal = s_prev + gamma * (level_prev + phi_trend) * epsilon

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(jnp.float32(new_seasonal))

        return (level, trend, seasonal_new, alpha, beta, gamma), y_hat

    def step_MAM(carry, y_t):
        """Multiplicative error, Multiplicative seasonality (MAM)."""
        level_prev, trend_prev, seasonal_prev, alpha, beta, gamma = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[m - 1]  # s_{t-m}

        y_hat = (level_prev + phi_trend) * s_prev
        epsilon = (y_t - y_hat) / jnp.maximum(jnp.abs(y_hat), _EPSILON)

        level = (level_prev + phi_trend) * (1 + alpha * epsilon)
        trend = phi_trend + beta * (level_prev + phi_trend) * epsilon
        new_seasonal = s_prev * (1 + gamma * epsilon)

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(jnp.float32(new_seasonal))

        return (level, trend, seasonal_new, alpha, beta, gamma), y_hat

    # Select appropriate step function based on model type
    if is_additive_error and is_additive_season:
        step_fn = step_AAA
    elif is_additive_error and not is_additive_season:
        step_fn = step_AAM
    elif not is_additive_error and is_additive_season:
        step_fn = step_MAA
    else:
        step_fn = step_MAM

    def raw_fit(params_abg):
        alpha, beta, gamma = params_abg
        init_carry = (l0, b0, s0, alpha, beta, gamma)
        final_carry, fitted_vals = lax.scan(step_fn, init_carry, y)
        final_level, final_trend, final_seasonal, _, _, _ = final_carry
        return fitted_vals, final_level, final_trend, final_seasonal, y - fitted_vals

    def loss_fn(params_abg):
        fitted, _, _, _, residuals = raw_fit(params_abg)
        sse = jnp.sum(residuals ** 2)
        if is_additive_error:
            return n * jnp.log(jnp.maximum(sse, _EPSILON))
        else:
            log_det = 2 * jnp.sum(jnp.log(jnp.maximum(jnp.abs(fitted), _EPSILON)))
            return n * jnp.log(jnp.maximum(sse, _EPSILON)) + log_det

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Track best params during optimization (prevents overshoot)
    def opt_step(carry, _):
        params, opt_state, best_params, best_loss = carry
        loss, grads = value_and_grad_fn(params)
        loss = jnp.float32(loss)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jnp.clip(new_params, 0.0001, 0.9999)

        # Track best parameters (minimum loss)
        improved = loss < best_loss
        new_best_params = jnp.where(improved, params, best_params)
        new_best_loss = jnp.where(improved, loss, best_loss)

        return (new_params, opt_state, new_best_params, new_best_loss), loss

    init_params = jnp.array([_INIT_ALPHA, _INIT_BETA, _INIT_GAMMA])
    opt_state = optimizer.init(init_params)
    init_carry = (init_params, opt_state, init_params, jnp.float32(jnp.inf))
    (_, _, best_params, _), _ = lax.scan(opt_step, init_carry, None, length=n_iters)

    # Get final results with best parameters
    fitted, final_level, final_trend, final_seasonal, residuals = raw_fit(best_params)

    return best_params, {
        'fitted': fitted,
        'level': final_level,
        'trend': final_trend,
        'seasonal': final_seasonal,
        'residuals': residuals,
        'alpha': best_params[0],
        'beta': best_params[1],
        'gamma': best_params[2],
    }


# JIT with static args: is_additive_error, is_additive_season, season_length, n_iters
_run_hw_optimization_jit = jax.jit(_run_hw_optimization, static_argnums=(5, 6, 7, 8))


class HoltWinters(BaseForecaster):
    # Helper methods
    @staticmethod
    def _validate_h(h: int) -> None:
        """Validate forecast horizon parameter."""
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be a positive integer, got {h}")

    @staticmethod
    def _validate_level(level: list[int] | None) -> None:
        """Validate prediction interval level parameter."""
        if level is not None:
            if not isinstance(level, list):
                raise ValueError(f"level must be a list or None, got {type(level).__name__}")
            if any(not isinstance(lv, (int, float)) or lv < 0 or lv > 100 for lv in level):
                raise ValueError("All level values must be numbers between 0 and 100")

    def _initialize_states(self, y: jnp.ndarray) -> tuple[float, float, jnp.ndarray]:
        """Initialize level, trend, and seasonal states using JAX.

        Uses classical decomposition approach:
        - Level: mean of first season
        - Trend: average slope across seasons
        - Seasonal: average seasonal pattern
        """
        m = self.season_length
        n = len(y)
        n_seasons = n // m

        if n_seasons < 2:
            # Simple fallback
            l0 = float(jnp.mean(y[:m])) if n >= m else float(y[0])
            b0 = 0.0
            if n >= m:
                s0 = y[:m] - l0 if self.season_type == 'A' else y[:m] / jnp.maximum(l0, _EPSILON)
            else:
                s0 = jnp.zeros(m, dtype=jnp.float32) if self.season_type == 'A' else jnp.ones(m, dtype=jnp.float32)
            return l0, b0, s0.astype(jnp.float32)

        # Compute seasonal averages via reshape (uniform shapes for JAX)
        n_complete = n_seasons * m
        y_reshaped = y[:n_complete].reshape(n_seasons, m)  # (n_seasons, m)
        seasonal_avgs = jnp.mean(y_reshaped, axis=1)  # Mean of each season

        # Linear regression on seasonal averages for trend
        t = jnp.arange(n_seasons, dtype=jnp.float32)
        t_mean = jnp.mean(t)
        avg_mean = jnp.mean(seasonal_avgs)

        cov_ta = jnp.sum((t - t_mean) * (seasonal_avgs - avg_mean))
        var_t = jnp.sum((t - t_mean) ** 2)
        b0 = float(cov_ta / jnp.maximum(var_t, _EPSILON))
        l0 = float(avg_mean - b0 * t_mean)

        # Detrend and compute seasonal pattern via reshape (avoids variable-length indexing)
        trend_vals = l0 + b0 * jnp.repeat(jnp.arange(n_seasons, dtype=jnp.float32), m)
        y_complete = y[:n_complete]
        if self.season_type == 'A':
            detrended = y_complete - trend_vals
        else:
            detrended = y_complete / jnp.maximum(trend_vals, _EPSILON)

        # Average by position within season using reshape (uniform shapes)
        detrended_reshaped = detrended.reshape(n_seasons, m)  # (n_seasons, m)
        s0 = jnp.mean(detrended_reshaped, axis=0)  # Average across seasons per position

        # Normalize
        if self.season_type == 'A':
            s0 = s0 - jnp.mean(s0)
        else:
            s0 = s0 / jnp.maximum(jnp.mean(s0), _EPSILON)

        return l0, b0, s0.astype(jnp.float32)

    def _get_phi(self) -> float:
        """Get damping factor phi."""
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        else:
            return 1.0

    def _estimate_iterations(self, y: jnp.ndarray) -> int:
        """Estimate iterations based on noise, seasonality, and trend."""
        n = len(y)
        m = self.season_length

        # 1. Noise: CV of first differences
        diffs = y[1:] - y[:-1]
        cv_diffs = jnp.std(diffs) / jnp.maximum(jnp.abs(jnp.mean(diffs)), _EPSILON)
        noise_score = float(jnp.clip(cv_diffs / 5.0, 0.0, 1.0))

        # 2. Seasonality difficulty: 1 - ACF(m)
        y_c = y - jnp.mean(y)
        var_y = jnp.var(y)
        n_acf = n - m
        if n_acf > 0:
            acf_m = jnp.sum(y_c[:n_acf] * y_c[m:]) / (n_acf * jnp.maximum(var_y, _EPSILON))
            seasonality_difficulty = float(1.0 - jnp.maximum(jnp.clip(acf_m, -1.0, 1.0), 0.0))
        else:
            seasonality_difficulty = 1.0

        # 3. Trend clarity
        t = jnp.arange(n, dtype=jnp.float32)
        y_mean, t_mean = jnp.mean(y), jnp.mean(t)
        slope = jnp.sum((t - t_mean) * (y - y_mean)) / jnp.maximum(jnp.sum((t - t_mean)**2), _EPSILON)
        y_pred = y_mean + slope * (t - t_mean)
        ss_res = jnp.sum((y - y_pred)**2)
        ss_tot = jnp.sum((y - y_mean)**2)
        r_sq = 1 - ss_res / jnp.maximum(ss_tot, _EPSILON)
        trend_difficulty = float(1.0 - jnp.clip(r_sq, 0.0, 1.0))

        # 4. Season length factor
        season_length_score = float(jnp.clip((m - 4) / 48, 0.0, 1.0))

        # 5. Coverage factor
        n_seasons = n // m
        coverage_score = float(jnp.clip((4 - n_seasons) / 4, 0.0, 1.0)) if n_seasons < 4 else 0.0

        # Weighted combination
        complexity = (0.30 * noise_score + 0.35 * seasonality_difficulty +
                      0.15 * trend_difficulty + 0.10 * season_length_score + 0.10 * coverage_score)

        # Scale to range
        min_iters = _MIN_ITER
        max_iters = _MAX_ITER_EXTENDED if self.allow_extended_iterations else _MAX_ITER
        exponent = {"cubic": 3.0, "quadratic": 2.0}[self.iteration_scaling]

        return int(min_iters + (complexity ** exponent) * (max_iters - min_iters))

    def _compute_base_variance(
        self,
        t: jnp.ndarray,
        alpha: float,
        beta: float,
        phi: float
    ) -> jnp.ndarray:
        """Compute base variance for trend component.

        This calculation is identical for both additive and multiplicative
        error models, so it's extracted to eliminate duplication.

        Parameters
        ----------
        t : jnp.ndarray
            Time steps (1 to h)
        alpha : float
            Level smoothing parameter
        beta : float
            Trend smoothing parameter
        phi : float
            Damping factor

        Returns
        -------
        jnp.ndarray
            Base variance for each time step
        """
        if self.damped and phi < 0.9999:
            # Damped trend variance calculation
            denom = jnp.maximum(1 - phi, _EPSILON)
            denom2 = jnp.maximum(1 - phi**2, _EPSILON)
            trend_var = (beta * phi * t) / denom**2 * (2 * alpha * denom + beta * phi) \
                       - (beta * phi * (1 - phi**t)) / (denom**2 * denom2) \
                       * (2 * alpha * denom2 + beta * phi * (1 + 2 * phi - phi**t))
            return 1 + alpha**2 * (t - 1) + trend_var
        else:
            # Non-damped trend variance calculation
            exp1 = alpha**2 + alpha * beta * t + (1 / 6) * beta**2 * t * (2 * t - 1)
            return 1 + (t - 1) * exp1

    def _validate_forecast_inputs(
        self,
        y: jnp.ndarray,
        h: int,
        level: list[int] | None
    ) -> jnp.ndarray:
        """Validate inputs for forecast/forward methods.

        Parameters
        ----------
        y : jnp.ndarray
            Input time series
        h : int
            Forecast horizon
        level : list[int] | None
            Prediction interval levels

        Returns
        -------
        jnp.ndarray
            Validated and converted y as float32

        Raises
        ------
        ValueError
            If y is too short, h is invalid, or level values are invalid
        """
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
        n_iters = self._estimate_iterations(y)
        is_additive_error = self.error_type == 'A'
        is_additive_season = self.season_type == 'A'

        # Use module-level JIT function
        best_params, result = _run_hw_optimization_jit(
            y, l0, b0, s0, phi, is_additive_error, is_additive_season,
            self.season_length, n_iters
        )

        # Convert to Python floats for storage
        result['alpha'] = float(best_params[0])
        result['beta'] = float(best_params[1])
        result['gamma'] = float(best_params[2])
        result['sigma'] = utils.calculate_sigma(result['residuals'], len(y) - _N_PARAMS)
        return result

    def _generate_forecasts(self, level: float, trend: float, seasonal: jnp.ndarray, phi: float, h: int) -> jnp.ndarray:
        """Generate h-step ahead forecasts (vectorized)."""
        m = self.season_length
        t_vals = jnp.arange(1, h + 1, dtype=jnp.float32)

        # Vectorized trend component with damping
        if phi == 1.0:
            trend_components = t_vals * trend
        else:
            # Sum of geometric series: phi * (1 - phi^t) / (1 - phi)
            trend_components = phi * (1 - phi**t_vals) / (1 - phi) * trend

        # Vectorized seasonal indices (wrapping around)
        seasonal_indices = jnp.arange(h) % m
        s_components = seasonal[seasonal_indices]

        # Combine components
        if self.season_type == 'A':
            # Additive: level + trend + seasonal
            forecasts = level + trend_components + s_components
        else:
            # Multiplicative: (level + trend) * seasonal
            forecasts = (level + trend_components) * s_components

        return forecasts

    def _calculate_native_intervals(
        self,
        mean: jnp.ndarray,
        sigma: float,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        h: int
    ) -> jnp.ndarray:
        """Calculate native prediction interval width (sigmah).

        Uses analytical formulas for Holt-Winters models.
        Based on research by Hyndman et al. (2008) and Taylor (2003).

        For additive seasonality (AAA/MAA):
        - Base variance grows with forecast horizon
        - Additional variance from seasonal uncertainty

        For multiplicative seasonality (AAM/MAM):
        - Variance scales with forecast level
        """
        t = jnp.arange(1, h + 1, dtype=jnp.float32)

        # Compute base variance (same for both error types)
        base_var = self._compute_base_variance(t, alpha, beta, phi)

        # Seasonal variance component (simplified approximation based on gamma)
        seasonal_var = gamma**2 * ((t - 1) // self.season_length + 1)

        # Combine variances based on error and season types
        if self.error_type == 'A':
            # Additive error models
            if self.season_type == 'A':
                # AAA model: additive seasonal
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var)
            else:
                # AAM model: multiplicative seasonal
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var) * jnp.abs(mean)
        else:
            # Multiplicative error models (MAA/MAM)
            sigmah_base = jnp.sqrt(base_var + seasonal_var)
            sigmah = sigma * sigmah_base * jnp.abs(mean)

        return sigmah

    def _add_interval_bounds(
        self,
        res: dict,
        values: jnp.ndarray,
        sigmah: jnp.ndarray | float,
        level: list[int],
        prefix: str = ''
    ) -> dict:
        """Add prediction interval bounds to result dictionary."""
        for lv in reversed(level):
            alpha_level = (100 - lv) / 100
            z = utils._jax_norm_ppf(1 - alpha_level / 2)
            res[f'{prefix}lo-{lv}'] = values - z * sigmah
            res[f'{prefix}hi-{lv}'] = values + z * sigmah
        return res

    def _compute_forecast_with_intervals(
        self,
        result: dict,
        phi: float,
        h: int,
        level: list[int] | None,
        fitted: bool,
        y: jnp.ndarray,
        X: jnp.ndarray | None
    ) -> dict:
        """Generate forecasts and add intervals if requested.

        This consolidates common logic from forecast() and forward() methods.

        Parameters
        ----------
        result : dict
            Fitted model results from _fit_parameters()
        phi : float
            Damping factor
        h : int
            Forecast horizon
        level : list[int] | None
            Confidence levels for prediction intervals
        fitted : bool
            Whether to include fitted values
        y : jnp.ndarray
            Time series data (for conformal intervals)
        X : jnp.ndarray | None
            Exogenous variables (for API consistency)

        Returns
        -------
        dict
            Dictionary with forecasts and optional intervals
        """
        # Generate point forecasts
        mean = self._generate_forecasts(
            result['level'],
            result['trend'],
            result['seasonal'],
            phi,
            h
        )
        res = {'mean': mean}

        # Add fitted values if requested
        if fitted:
            res['fitted'] = result['fitted']

        # Add prediction intervals if requested
        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                # Use conformal prediction intervals
                temp_model = getattr(self, 'model_', None)
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
                # Restore previous model state
                if temp_model is not None:
                    self.model_ = temp_model
                else:
                    delattr(self, 'model_')
            else:
                # Use native analytical prediction intervals
                sigmah = self._calculate_native_intervals(
                    mean,
                    result['sigma'],
                    result['alpha'],
                    result['beta'],
                    result['gamma'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

            # Add fitted intervals if requested
            if fitted:
                res = self._add_interval_bounds(
                    res,
                    result['fitted'],
                    result['sigma'],
                    level,
                    prefix='fitted-'
                )

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
        """
        Holt-Winters' seasonal exponential smoothing method.

        Parameters
        ----------
        season_length : int, default=12
            Number of observations per unit of time (e.g., 12 for monthly data
            with yearly seasonality). Must be at least 2.
        error_type : str, default='A'
            Type of error: 'A' (additive) or 'M' (multiplicative).
            Must be either 'A' or 'M'.
        season_type : str, default='A'
            Type of seasonality: 'A' (additive) or 'M' (multiplicative).
            Must be either 'A' or 'M'.
        damped : bool | None, default=None
            Whether to use damped trend. If None, treated as False (non-damped).
        phi : float | None, default=None
            Damping parameter, must be in [0.8, 0.98]. Only used if damped=True.
            If damped=True and phi=None, defaults to 0.9.
        alias : str, default="HoltWinters"
            Custom name for the model.
        conformal_params : ConformalIntervals | None, default=None
            Parameters for conformal prediction intervals. If None, uses native
            analytical prediction intervals.
        allow_extended_iterations : bool, default=False
            Whether to allow extended iteration counts (up to 600) for difficult
            series. Default max is 350.
        iteration_scaling : str, default="quadratic"
            Scaling method for adaptive iterations. "quadratic" (default) gives
            moderate scaling, "cubic" gives more aggressive scaling for complex series.

        Raises
        ------
        ValueError
            If season_length < 2.
            If error_type is not 'A' or 'M'.
            If season_type is not 'A' or 'M'.
            If phi is not a float when provided.
            If phi is outside the valid range [0.8, 0.98].
            If conformal_params is not a ConformalIntervals instance.
            If iteration_scaling is not 'quadratic' or 'cubic'.
        """
        # Validate season_length
        if not isinstance(season_length, int) or season_length < 2:
            raise ValueError(f"season_length must be an integer >= 2, got {season_length}")

        # Validate error_type
        if error_type not in ('A', 'M'):
            raise ValueError(
                f"error_type must be 'A' (additive) or 'M' (multiplicative), got '{error_type}'"
            )

        # Validate season_type
        if season_type not in ('A', 'M'):
            raise ValueError(
                f"season_type must be 'A' (additive) or 'M' (multiplicative), got '{season_type}'"
            )

        # Validate phi
        if phi is not None:
            if not isinstance(phi, (float, int)):
                raise ValueError(f"phi must be None or a number, got {type(phi).__name__}")
            phi = float(phi)
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"phi must be in range [{_PHI_LOWER}, {_PHI_UPPER}], got {phi}")

        # Validate conformal_params
        if conformal_params is not None and not isinstance(conformal_params, ConformalIntervals):
            raise ValueError(
                f"conformal_params must be a ConformalIntervals instance, got {type(conformal_params).__name__}"
            )

        # Validate iteration_scaling
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

    def fit(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ) -> 'HoltWinters':
        """Fit the Holt-Winters model to training data.

        This method estimates the smoothing parameters (alpha, beta, gamma) and
        computes the level, trend, and seasonal states by maximizing the
        log-likelihood using gradient descent optimization with JAX.

        Parameters
        ----------
        y : jnp.ndarray
            Training time series data of shape (n,). Must have at least
            2 * season_length observations for robust fitting.
        X : jnp.ndarray | None, default=None
            Exogenous variables (not currently used, included for API consistency).

        Returns
        -------
        self : HoltWinters
            The fitted model instance.

        Raises
        ------
        ValueError
            If y has fewer than season_length observations.

        Notes
        -----
        The fitted model stores the following in `self.model_`:
        - fitted: In-sample fitted values
        - level: Final level state after processing all observations
        - trend: Final trend state after processing all observations
        - seasonal: Final seasonal states (array of length season_length)
        - alpha: Estimated level smoothing parameter
        - beta: Estimated trend smoothing parameter
        - gamma: Estimated seasonal smoothing parameter
        - sigma: Residual standard error
        - residuals: Forecast residuals
        - y_train: Original training data (for conformal prediction)
        """
        y = utils.ensure_float(y)

        # Validate minimum series length
        if len(y) < self.season_length:
            raise ValueError(
                f"Time series must have at least {self.season_length} observations "
                f"(season_length), got {len(y)}"
            )

        result = self._fit_parameters(y)
        result['y_train'] = y  # Store for conformal prediction
        self.model_ = result
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
    ) -> dict:
        """
        Predict with fitted Holt-Winters model.

        Parameters
        ----------
        h : int
            Forecast horizon (must be positive).
        X : jnp.ndarray, optional
            Exogenous variables (not used, included for API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Dictionary with entries 'mean' for point predictions and
            'lo-{level}' and 'hi-{level}' for probabilistic predictions.

        Raises
        ------
        ValueError
            If model is not fitted, if h is not positive, or if level values
            are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict(). Call fit() first.")

        self._validate_h(h)
        self._validate_level(level)

        phi = self._get_phi()
        mean = self._generate_forecasts(
            self.model_['level'],
            self.model_['trend'],
            self.model_['seasonal'],
            phi,
            h
        )
        res = {'mean': mean}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                cs = self.conformity_scores(y=self.model_['y_train'], X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    self.model_['sigma'],
                    self.model_['alpha'],
                    self.model_['beta'],
                    self.model_['gamma'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

        return res

    def predict_in_sample(
        self,
        level: list[int] | None = None,
    ) -> dict:
        """
        Access fitted Holt-Winters model insample predictions.

        Parameters
        ----------
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            Dictionary with entries 'fitted' for point predictions and
            'fitted-lo-{level}' and 'fitted-hi-{level}' for probabilistic predictions.

        Raises
        ------
        ValueError
            If model is not fitted or if level values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict_in_sample(). Call fit() first.")

        self._validate_level(level)

        res = {'fitted': self.model_['fitted']}

        if level is not None:
            level = sorted(level)
            res = self._add_interval_bounds(
                res,
                self.model_['fitted'],
                self.model_['sigma'],
                level,
                prefix='fitted-'
            )

        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Memory efficient Holt-Winters predictions.

        This method avoids memory burden from object storage.
        It is analogous to fit_predict without storing information.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). Must have at least season_length observations.
        h : int
            Forecast horizon (must be positive).
        X : jnp.ndarray, optional
            Insample exogenous variables (not used, included for API consistency).
        X_future : jnp.ndarray, optional
            Future exogenous variables (not used, included for API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default=False
            Whether to return insample predictions.

        Returns
        -------
        dict
            Dictionary with entries 'mean' for point predictions,
            'fitted' for insample predictions (if fitted=True),
            and 'lo-{level}' and 'hi-{level}' for probabilistic predictions.

        Raises
        ------
        ValueError
            If y has fewer than season_length observations, if h is not positive,
            or if level values are outside [0, 100].
        """
        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Apply fitted Holt-Winters model to a new time series.

        This method uses the model structure (error_type, season_type, damped, phi)
        from the original fit, but re-estimates parameters on the new data.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). Must have at least season_length observations.
        h : int
            Forecast horizon (must be positive).
        X : jnp.ndarray, optional
            Insample exogenous variables (not used, included for API consistency).
        X_future : jnp.ndarray, optional
            Future exogenous variables (not used, included for API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default=False
            Whether to return insample predictions.

        Returns
        -------
        dict
            Dictionary with entries 'mean' for point predictions,
            'fitted' for insample predictions (if fitted=True),
            and 'lo-{level}' and 'hi-{level}' for probabilistic predictions.

        Raises
        ------
        ValueError
            If model is not fitted, if y has fewer than season_length observations,
            if h is not positive, or if level values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling forward(). Call fit() first.")

        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)
