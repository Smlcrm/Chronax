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
8. model_: dict - Fitted model parameters (created after fit())
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
- _fit_parameters() - Core optimization routine using JAX/optax
- _generate_forecasts() - Compute h-step ahead point forecasts with seasonality
- _calculate_native_intervals() - Analytical prediction interval formulas with seasonality
- _add_interval_bounds() - Add interval bounds to result dictionary
- _compute_base_variance() - Compute base variance for trend component
- _validate_forecast_inputs() - Validate inputs for forecast/forward methods
- _compute_forecast_with_intervals() - Generate forecasts and add intervals

Implementation Notes:
- Uses optax.adam optimizer with exponential learning rate decay
- Default 1500 iterations for parameter optimization (more complex than Holt)
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
_N_ITER = 1500
_LEARNING_RATE = 0.01
_LR_DECAY_STEPS = 500
_LR_DECAY_RATE = 0.9
_EPSILON = 1e-10  # For numerical stability

__all__ = ['HoltWinters']

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
        """Initialize level, trend, and seasonal states.

        Uses classical decomposition approach:
        - Level: mean of first season
        - Trend: average slope across seasons
        - Seasonal: average seasonal pattern
        """
        import numpy as np

        m = self.season_length
        n = len(y)
        y_np = np.array(y)

        # Need at least 2 full seasons for robust initialization
        n_seasons = n // m

        if n_seasons < 2:
            # Fall back to simple initialization
            l0 = float(np.mean(y_np[:m]) if n >= m else y_np[0])
            b0 = 0.0
            if n >= m:
                s0 = np.array(y_np[:m]) - l0
            else:
                s0 = np.zeros(m)
        else:
            # Compute seasonal averages
            seasonal_avgs = []
            for i in range(n_seasons):
                start_idx = i * m
                end_idx = min(start_idx + m, n)
                if end_idx - start_idx == m:
                    seasonal_avgs.append(np.mean(y_np[start_idx:end_idx]))

            # Estimate trend from seasonal averages
            if len(seasonal_avgs) >= 2:
                t_vals = np.arange(len(seasonal_avgs))
                X_trend = np.column_stack([np.ones(len(seasonal_avgs)), t_vals])
                coef = np.linalg.lstsq(X_trend, np.array(seasonal_avgs), rcond=None)[0]
                l0, b0 = float(coef[0]), float(coef[1])
            else:
                l0 = seasonal_avgs[0] if seasonal_avgs else float(y_np[0])
                b0 = 0.0

            # Compute seasonal components
            # Detrend the series first
            detrended = np.zeros(n)
            for i in range(n):
                trend_val = l0 + b0 * (i // m)
                if self.season_type == 'A':
                    detrended[i] = y_np[i] - trend_val
                else:  # Multiplicative
                    detrended[i] = y_np[i] / (trend_val + _EPSILON) if trend_val != 0 else 1.0

            # Average seasonal indices
            s0 = np.zeros(m)
            for j in range(m):
                indices = detrended[j::m]
                s0[j] = np.mean(indices) if len(indices) > 0 else 0.0

            # Normalize seasonal components
            if self.season_type == 'A':
                s0 = s0 - np.mean(s0)
            else:  # Multiplicative
                mean_s = np.mean(s0)
                s0 = s0 / mean_s if mean_s != 0 else np.ones(m)

        return l0, b0, jnp.array(s0, dtype=jnp.float32)

    def _get_phi(self) -> float:
        """Get damping factor phi."""
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        else:
            return 1.0

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
        m = self.season_length

        def step_update(carry, y_t):
            level_prev, trend_prev, seasonal_prev, alpha, beta, gamma = carry

            # Compute forecast
            phi_trend = phi * trend_prev
            s_prev = seasonal_prev[m - 1]  # s_{t-m}

            if self.season_type == 'A':
                y_hat = level_prev + phi_trend + s_prev
            else:  # Multiplicative
                y_hat = (level_prev + phi_trend) * s_prev

            # Update equations
            if self.error_type == 'A':
                # Additive error
                if self.season_type == 'A':
                    # AAA model
                    level = alpha * (y_t - s_prev) + (1 - alpha) * (level_prev + phi_trend)
                    trend = beta * (level - level_prev) + (1 - beta) * phi_trend
                    new_seasonal = gamma * (y_t - level) + (1 - gamma) * s_prev
                else:
                    # AAM model
                    level = alpha * (y_t / jnp.maximum(s_prev, _EPSILON)) + (1 - alpha) * (level_prev + phi_trend)
                    trend = beta * (level - level_prev) + (1 - beta) * phi_trend
                    new_seasonal = gamma * (y_t / jnp.maximum(level, _EPSILON)) + (1 - gamma) * s_prev
            else:
                # Multiplicative error
                epsilon = (y_t - y_hat) / jnp.maximum(jnp.abs(y_hat), _EPSILON)
                if self.season_type == 'A':
                    # MAA model
                    level = (level_prev + phi_trend - s_prev) + (level_prev + phi_trend) * alpha * epsilon
                    trend = phi_trend + beta * (level_prev + phi_trend) * epsilon
                    new_seasonal = s_prev + gamma * (level_prev + phi_trend) * epsilon
                else:
                    # MAM model
                    level = (level_prev + phi_trend) * (1 + alpha * epsilon)
                    trend = phi_trend + beta * (level_prev + phi_trend) * epsilon
                    new_seasonal = s_prev * (1 + gamma * epsilon)

            # Roll seasonal array
            seasonal_new = jnp.roll(seasonal_prev, -1)
            seasonal_new = seasonal_new.at[-1].set(new_seasonal)

            return (level, trend, seasonal_new, alpha, beta, gamma), y_hat

        l0_fixed, b0_fixed, s0_fixed = l0, b0, s0

        @jax.jit
        def raw_fit(params_abg):
            alpha, beta, gamma = params_abg
            init_carry = (l0_fixed, b0_fixed, s0_fixed, alpha, beta, gamma)
            final_carry, fitted_vals = lax.scan(step_update, init_carry, y)
            final_level, final_trend, final_seasonal, _, _, _ = final_carry
            residuals = y - fitted_vals
            return {
                'fitted': fitted_vals,
                'level': final_level,
                'trend': final_trend,
                'seasonal': final_seasonal,
                'residuals': residuals,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
            }

        @jax.jit
        def _likelihood_loss(params_abg):
            result = raw_fit(params_abg)
            residuals = result['residuals']
            sse = jnp.sum(residuals ** 2)
            if self.error_type == 'M':
                fitted_vals = result['fitted']
                log_det = 2 * jnp.sum(jnp.log(jnp.maximum(jnp.abs(fitted_vals), _EPSILON)))
                return len(y) * jnp.log(jnp.maximum(sse, _EPSILON)) + log_det
            else:
                return len(y) * jnp.log(jnp.maximum(sse, _EPSILON))

        init_params = jnp.array([_INIT_ALPHA, _INIT_BETA, _INIT_GAMMA])
        scheduler = optax.exponential_decay(
            init_value=_LEARNING_RATE,
            transition_steps=_LR_DECAY_STEPS,
            decay_rate=_LR_DECAY_RATE
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=scheduler)
        )
        opt_state = optimizer.init(init_params)
        params = init_params

        @jax.jit
        def step(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(_likelihood_loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = params.at[:_N_PARAMS].set(jnp.clip(params[:_N_PARAMS], 0.0001, 0.9999))
            return (params, opt_state), loss

        (final_params, _), _ = jax.jit(lambda: lax.scan(step, (params, opt_state), jnp.arange(_N_ITER)))()

        result = raw_fit(final_params)
        result['alpha'] = float(final_params[0])
        result['beta'] = float(final_params[1])
        result['gamma'] = float(final_params[2])
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

        Raises
        ------
        ValueError
            If season_length < 2.
            If error_type is not 'A' or 'M'.
            If season_type is not 'A' or 'M'.
            If phi is not a float when provided.
            If phi is outside the valid range [0.8, 0.98].
            If conformal_params is not a ConformalIntervals instance.
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

        self.season_length = season_length
        self.error_type = error_type
        self.season_type = season_type
        self.damped = damped if damped is not None else False
        self.phi = phi
        self.alias = alias
        self.conformal_params = conformal_params

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


# def test():
#     """Comprehensive test suite for Holt-Winters model."""
#     print("="*60)
#     print("Running Holt-Winters model tests...")
#     print("="*60)
#     passed = 0
#     failed = 0

#     # Test 1: Basic fit/predict (AAA Model)
#     try:
#         print("\nTest 1: Basic fit/predict (AAA Model)")
#         # Create seasonal data: 3 years of monthly data with trend and seasonality
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal + np.random.normal(0, 2, 36), dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model.fit(y)
#         result = model.predict(h=12)

#         assert 'mean' in result, "Result should contain 'mean' key"
#         assert len(result['mean']) == 12, "Forecast length should be 12"
#         assert jnp.all(jnp.isfinite(result['mean'])), "Forecasts should be finite"

#         print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
#         print(f"  Fitted beta: {model.model_['beta']:.4f}")
#         print(f"  Fitted gamma: {model.model_['gamma']:.4f}")
#         print(f"  First 3 forecasts: {result['mean'][:3]}")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 2: MAM Model (Multiplicative error, Additive trend, Multiplicative seasonality)
#     try:
#         print("\nTest 2: MAM Model")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend * seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='M', season_type='M', damped=False)
#         model.fit(y)
#         result = model.predict(h=12)

#         assert 'mean' in result, "Result should contain 'mean' key"
#         assert len(result['mean']) == 12, "Forecast length should be 12"

#         print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
#         print(f"  Fitted beta: {model.model_['beta']:.4f}")
#         print(f"  Fitted gamma: {model.model_['gamma']:.4f}")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 3: AAM Model (Additive error, Additive trend, Multiplicative seasonality)
#     try:
#         print("\nTest 3: AAM Model")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend * seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='M', damped=False)
#         model.fit(y)
#         result = model.predict(h=12)

#         assert 'mean' in result, "Should have forecasts"
#         assert len(result['mean']) == 12, "Forecast length should be 12"

#         print(f"  Model fitted successfully")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 4: MAA Model (Multiplicative error, Additive trend, Additive seasonality)
#     try:
#         print("\nTest 4: MAA Model")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='M', season_type='A', damped=False)
#         model.fit(y)
#         result = model.predict(h=12)

#         assert 'mean' in result, "Should have forecasts"
#         assert len(result['mean']) == 12, "Forecast length should be 12"

#         print(f"  Model fitted successfully")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 5: Damped Trend
#     try:
#         print("\nTest 5: Damped Trend")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal, dtype=jnp.float32)

#         # Non-damped model
#         model_nodamp = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model_nodamp.fit(y)
#         result_nodamp = model_nodamp.predict(h=24)

#         # Damped model
#         model_damp = HoltWinters(season_length=12, error_type='A', season_type='A', damped=True, phi=0.9)
#         model_damp.fit(y)
#         result_damp = model_damp.predict(h=24)

#         # At long horizons, damped should be lower (due to dampening trend)
#         last_forecast_damp = result_damp['mean'][-1]
#         last_forecast_nodamp = result_nodamp['mean'][-1]

#         assert last_forecast_damp < last_forecast_nodamp, \
#             "Damped forecast should be lower at long horizons"

#         print(f"  Non-damped h=24: {last_forecast_nodamp:.2f}")
#         print(f"  Damped h=24: {last_forecast_damp:.2f}")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 6: Prediction Intervals
#     try:
#         print("\nTest 6: Prediction Intervals")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model.fit(y)
#         result = model.predict(h=12, level=[80, 95])

#         assert 'lo-80' in result, "Should have lo-80"
#         assert 'hi-80' in result, "Should have hi-80"
#         assert 'lo-95' in result, "Should have lo-95"
#         assert 'hi-95' in result, "Should have hi-95"

#         # Check ordering for first forecast
#         assert result['lo-95'][0] < result['lo-80'][0], "lo-95 < lo-80"
#         assert result['lo-80'][0] < result['mean'][0], "lo-80 < mean"
#         assert result['mean'][0] < result['hi-80'][0], "mean < hi-80"
#         assert result['hi-80'][0] < result['hi-95'][0], "hi-80 < hi-95"

#         print(f"  h=1: [{result['lo-95'][0]:.2f}, {result['mean'][0]:.2f}, {result['hi-95'][0]:.2f}]")
#         print(f"  h=12: [{result['lo-95'][11]:.2f}, {result['mean'][11]:.2f}, {result['hi-95'][11]:.2f}]")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 7: Different Season Lengths
#     try:
#         print("\nTest 7: Different Season Lengths")

#         # Quarterly data (season_length=4)
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(28)  # 7 years quarterly
#         trend = 100 + 2 * t
#         seasonal = 5 * np.sin(2 * np.pi * t / 4)
#         y_quarterly = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model_q = HoltWinters(season_length=4, error_type='A', season_type='A')
#         model_q.fit(y_quarterly)
#         result_q = model_q.predict(h=4)
#         assert len(result_q['mean']) == 4, "Should forecast 4 quarters"

#         # Weekly data (season_length=7)
#         t = np.arange(35)  # 5 weeks
#         trend = 100 + t
#         seasonal = 3 * np.sin(2 * np.pi * t / 7)
#         y_weekly = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model_w = HoltWinters(season_length=7, error_type='A', season_type='A')
#         model_w.fit(y_weekly)
#         result_w = model_w.predict(h=7)
#         assert len(result_w['mean']) == 7, "Should forecast 7 days"

#         print(f"  Quarterly (season=4): ✓")
#         print(f"  Weekly (season=7): ✓")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 8: predict_in_sample()
#     try:
#         print("\nTest 8: predict_in_sample()")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model.fit(y)
#         result = model.predict_in_sample(level=[95])

#         assert 'fitted' in result, "Should have fitted values"
#         assert len(result['fitted']) == len(y), "Fitted should match training length"
#         assert 'fitted-lo-95' in result, "Should have fitted intervals"
#         assert 'fitted-hi-95' in result, "Should have fitted intervals"

#         print(f"  Fitted length: {len(result['fitted'])}")
#         print(f"  First fitted value: {result['fitted'][0]:.2f}")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 9: forecast() Method
#     try:
#         print("\nTest 9: forecast() Method (Stateless)")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y = jnp.array(trend + seasonal, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         # Don't call fit()
#         result = model.forecast(y, h=12, fitted=True, level=[95])

#         assert 'mean' in result, "Should have forecasts"
#         assert 'fitted' in result, "Should have fitted values with fitted=True"
#         assert len(result['mean']) == 12, "Forecast length should be 12"
#         assert len(result['fitted']) == len(y), "Fitted length should match y"

#         print(f"  Forecast h=1: {result['mean'][0]:.2f}")
#         print(f"  Includes fitted values and intervals")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 10: forward() Method
#     try:
#         print("\nTest 10: forward() Method")
#         import numpy as np
#         np.random.seed(42)
#         t = np.arange(36)
#         trend = 100 + 2 * t
#         seasonal = 10 * np.sin(2 * np.pi * t / 12)
#         y1 = jnp.array(trend + seasonal, dtype=jnp.float32)

#         # Different scale
#         trend2 = 200 + 3 * t
#         seasonal2 = 15 * np.sin(2 * np.pi * t / 12)
#         y2 = jnp.array(trend2 + seasonal2, dtype=jnp.float32)

#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model.fit(y1)

#         # Apply to new series
#         result = model.forward(y2, h=6)

#         assert 'mean' in result, "Should have forecasts"
#         assert len(result['mean']) == 6, "Forecast length should be 6"
#         # Forecasts should follow y2 scale, not y1
#         assert result['mean'][0] > 200, "Forecast should follow y2 scale"

#         print(f"  y1 range: [{float(y1.min()):.1f}, {float(y1.max()):.1f}]")
#         print(f"  y2 range: [{float(y2.min()):.1f}, {float(y2.max()):.1f}]")
#         print(f"  Forward forecast: {result['mean'][0]:.2f}")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Test 11: Edge Cases
#     try:
#         print("\nTest 11: Edge Cases")

#         # Test minimum data length (should work with season_length observations)
#         import numpy as np
#         y_min = jnp.array(np.arange(12, dtype=np.float32) + 100)
#         model = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         model.fit(y_min)
#         result = model.predict(h=4)
#         assert len(result['mean']) == 4, "Should work with season_length observations"

#         # Test that insufficient data fails
#         y_fail = jnp.array(np.arange(10, dtype=np.float32))
#         model2 = HoltWinters(season_length=12, error_type='A', season_type='A', damped=False)
#         try:
#             model2.fit(y_fail)
#             raise AssertionError("Should raise ValueError for len(y) < season_length")
#         except ValueError:
#             pass  # Expected

#         print(f"  Minimum length (season_length=12): {len(y_min)} ✓")
#         print(f"  Rejects insufficient data ✓")
#         print("  ✓ PASSED")
#         passed += 1
#     except Exception as e:
#         print(f"  ✗ FAILED: {e}")
#         failed += 1

#     # Summary
#     print("\n" + "="*60)
#     print(f"Tests passed: {passed}/{passed+failed}")
#     if failed == 0:
#         print("All tests passed! ✓")
#     else:
#         print(f"{failed} test(s) failed.")
#     print("="*60)


# if __name__ == '__main__':
#     test()