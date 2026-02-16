"""
Holt's Linear Exponential Smoothing Model.

This module implements Holt's linear trend method (double exponential smoothing)
with full JAX acceleration and compatibility with statsforecast's API.

Features:
- Additive and multiplicative error models
- Damped and non-damped trend variants
- Native prediction intervals (analytical formulas)
- Conformal prediction intervals support
- JAX JIT compilation for performance
- Four prediction methods: fit/predict, predict_in_sample, forecast, forward

The model fits level and trend smoothing parameters (alpha, beta) using
maximum likelihood optimization via gradient descent.

Instance Attributes:
1. season_length: int - Number of observations per unit of time (kept for API consistency)
2. error_type: str - Type of error: 'A' (additive) or 'M' (multiplicative)
3. damped: bool - Whether to use damped trend
4. phi: float | None - Damping parameter (0.8-0.98), used only if damped=True
5. alias: str - Custom name for the model
6. conformal_params: ConformalIntervals | None - Parameters for conformal prediction intervals
7. allow_extended_iterations: bool - Whether to allow extended iteration counts for difficult series
8. iteration_scaling: str - Scaling method for adaptive iterations ("quadratic" or "cubic")
9. model_: dict - Fitted model parameters (created after fit())
   - fitted: In-sample fitted values
   - level: Final level state
   - trend: Final trend state
   - alpha: Estimated level smoothing parameter
   - beta: Estimated trend smoothing parameter
   - sigma: Residual standard error
   - residuals: Forecast residuals
   - y_train: Original training data (for conformal prediction)

Class Attributes:
1. uses_exog: bool - Whether model supports exogenous variables (False for Holt)

Methods:
1. __init__() - Initialize Holt model with error type and damping parameters
2. fit(y, X=None) - Fit model to training data, estimates alpha and beta via optimization
3. predict(h, X=None, level=None) - Generate forecasts with fitted model
4. predict_in_sample(level=None) - Return fitted values with optional intervals
5. forecast(y, h, X=None, X_future=None, level=None, fitted=False) - Stateless prediction
6. forward(y, h, X=None, X_future=None, level=None, fitted=False) - Apply fitted model to new data

Helper Methods:
- _validate_h() - Validate forecast horizon parameter
- _validate_level() - Validate prediction interval levels
- _initialize_states() - Initialize level and trend via linear regression
- _get_phi() - Get damping factor
- _estimate_iterations() - Estimate optimal iteration count based on data complexity
- _fit_parameters() - Core optimization routine using JAX/optax
- _generate_forecasts() - Compute h-step ahead point forecasts
- _calculate_native_intervals() - Analytical prediction interval formulas
- _add_interval_bounds() - Add interval bounds to result dictionary

Implementation Notes:
- Uses optax.adam optimizer with exponential learning rate decay
- Adaptive iteration count based on data complexity (30-400 iterations)
- Module-level JIT functions for efficient compilation caching
- Supports both native (analytical) and conformal prediction intervals
- Level and trend initialized via linear regression on first 10 observations
- Analytical interval formulas from Hyndman et al. (2008)
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
_N_PARAMS = 2  # alpha and beta
_LEARNING_RATE = 0.01
_LR_DECAY_STEPS = 500
_LR_DECAY_RATE = 0.9
_EPSILON = 1e-10  # For numerical stability

# Adaptive iteration constants
_MIN_ITER = 30
_MAX_ITER = 200
_MAX_ITER_EXTENDED = 400

__all__ = ['Holt']


# =============================================================================
# Module-level JIT-compiled optimization functions
# =============================================================================

def _run_holt_optimization(y, l0, b0, phi, is_additive, n_iters):
    """Module-level JIT-compiled optimization loop for Holt.

    Parameters
    ----------
    y : jnp.ndarray
        Time series data
    l0 : float
        Initial level
    b0 : float
        Initial trend
    phi : float
        Damping factor (1.0 for non-damped)
    is_additive : bool
        True for additive error, False for multiplicative
    n_iters : int
        Number of optimization iterations

    Returns
    -------
    tuple
        (best_params, raw_fit_result) where best_params is [alpha, beta]
        and raw_fit_result is dict with fitted values, level, trend, residuals
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

    def step_additive(carry, y_t):
        level_prev, trend_prev, alpha, beta = carry
        y_hat = level_prev + phi * trend_prev
        level = alpha * y_t + (1 - alpha) * y_hat
        trend = beta * (level - level_prev) + (1 - beta) * phi * trend_prev
        return (level, trend, alpha, beta), y_hat

    def step_multiplicative(carry, y_t):
        level_prev, trend_prev, alpha, beta = carry
        y_hat = level_prev + phi * trend_prev
        epsilon = (y_t - y_hat) / jnp.maximum(jnp.abs(y_hat), _EPSILON)
        level = y_hat * (1 + alpha * epsilon)
        trend = phi * trend_prev + beta * y_hat * epsilon
        return (level, trend, alpha, beta), y_hat

    step_fn = step_additive if is_additive else step_multiplicative

    def raw_fit(params_ab):
        alpha, beta = params_ab
        init_carry = (l0, b0, alpha, beta)
        final_carry, fitted_vals = lax.scan(step_fn, init_carry, y)
        return fitted_vals, final_carry[0], final_carry[1], y - fitted_vals

    def loss_fn(params_ab):
        fitted, _, _, residuals = raw_fit(params_ab)
        sse = jnp.sum(residuals ** 2)
        if is_additive:
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

    init_params = jnp.array([_INIT_ALPHA, _INIT_BETA])
    opt_state = optimizer.init(init_params)
    init_carry = (init_params, opt_state, init_params, jnp.float32(jnp.inf))
    (_, _, best_params, _), _ = lax.scan(opt_step, init_carry, None, length=n_iters)

    # Get final results with best parameters
    fitted, final_level, final_trend, residuals = raw_fit(best_params)

    return best_params, {
        'fitted': fitted,
        'level': final_level,
        'trend': final_trend,
        'residuals': residuals,
        'alpha': best_params[0],
        'beta': best_params[1],
    }


# JIT with static args: is_additive and n_iters
_run_holt_optimization_jit = jax.jit(_run_holt_optimization, static_argnums=(4, 5))


class Holt(BaseForecaster):
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

    def _initialize_states(self, y: jnp.ndarray) -> tuple[float, float]:
        """Initialize level and trend via linear regression using JAX."""
        n = len(y)
        n_init = min(10, n // 2)

        if n_init >= 2:
            # Linear regression: y = l0 + b0 * t
            t = jnp.arange(n_init, dtype=jnp.float32)
            y_init = y[:n_init]

            t_mean = jnp.mean(t)
            y_mean = jnp.mean(y_init)

            # Slope: cov(t, y) / var(t)
            cov_ty = jnp.sum((t - t_mean) * (y_init - y_mean))
            var_t = jnp.sum((t - t_mean) ** 2)
            b0 = cov_ty / jnp.maximum(var_t, _EPSILON)

            # Intercept
            l0 = y_mean - b0 * t_mean

            return float(l0), float(b0)
        else:
            return float(y[0]), 0.0

    def _get_phi(self) -> float:
        """Get damping factor phi."""
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        else:
            return 1.0

    def _estimate_iterations(self, y: jnp.ndarray) -> int:
        """Estimate iterations based on noise and trend clarity."""
        n = len(y)

        # 1. Noise: CV of first differences
        diffs = y[1:] - y[:-1]
        cv_diffs = jnp.std(diffs) / jnp.maximum(jnp.abs(jnp.mean(diffs)), _EPSILON)
        noise_score = float(jnp.clip(cv_diffs / 5.0, 0.0, 1.0))

        # 2. Trend clarity: 1 - R^2 of linear fit
        t = jnp.arange(n, dtype=jnp.float32)
        y_mean, t_mean = jnp.mean(y), jnp.mean(t)
        slope = jnp.sum((t - t_mean) * (y - y_mean)) / jnp.maximum(jnp.sum((t - t_mean)**2), _EPSILON)
        y_pred = y_mean + slope * (t - t_mean)
        ss_res = jnp.sum((y - y_pred)**2)
        ss_tot = jnp.sum((y - y_mean)**2)
        r_sq = 1 - ss_res / jnp.maximum(ss_tot, _EPSILON)
        trend_difficulty = float(1.0 - jnp.clip(r_sq, 0.0, 1.0))

        # 3. Length penalty for short series
        length_score = float(jnp.clip((50 - n) / 50, 0.0, 1.0)) if n < 50 else 0.0

        # Weighted combination
        complexity = 0.50 * noise_score + 0.35 * trend_difficulty + 0.15 * length_score

        # Scale to range
        min_iters = _MIN_ITER
        max_iters = _MAX_ITER_EXTENDED if self.allow_extended_iterations else _MAX_ITER
        exponent = {"cubic": 3.0, "quadratic": 2.0}[self.iteration_scaling]

        return int(min_iters + (complexity ** exponent) * (max_iters - min_iters))

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Fit model parameters and return results dictionary."""
        l0, b0 = self._initialize_states(y)
        phi = self._get_phi()
        n_iters = self._estimate_iterations(y)
        is_additive = self.error_type == 'A'

        # Use module-level JIT function
        best_params, result = _run_holt_optimization_jit(
            y, l0, b0, phi, is_additive, n_iters
        )

        # Convert to Python floats for storage
        result['alpha'] = float(best_params[0])
        result['beta'] = float(best_params[1])
        result['sigma'] = utils.calculate_sigma(result['residuals'], len(y) - _N_PARAMS)
        return result

    def _generate_forecasts(self, level: float, trend: float, phi: float, h: int) -> jnp.ndarray:
        """Generate h-step ahead forecasts."""
        t = jnp.arange(1, h + 1, dtype=jnp.float32)
        if phi == 1.0:
            return level + t * trend
        else:
            phi_sum = phi * (1 - phi**t) / (1 - phi)
            return level + trend * phi_sum

    def _calculate_native_intervals(
        self,
        mean: jnp.ndarray,
        sigma: float,
        alpha: float,
        beta: float,
        phi: float,
        h: int
    ) -> jnp.ndarray:
        """Calculate native prediction interval width (sigmah).

        Uses analytical formulas from Hyndman et al. (2008) for:
        - AAN: Additive error, Additive trend, No seasonality
        - AAdN: Additive error, Additive damped trend, No seasonality
        - MAN: Multiplicative error, Additive trend, No seasonality
        - MAdN: Multiplicative error, Additive damped trend, No seasonality
        """
        t = jnp.arange(1, h + 1, dtype=jnp.float32)

        if self.error_type == 'A':
            if self.damped and phi < 0.9999:  # Avoid division by zero
                # Damped trend case - use full formula
                denom = jnp.maximum(1 - phi, _EPSILON)
                denom2 = jnp.maximum(1 - phi**2, _EPSILON)
                exp2 = (beta * phi * t) / denom ** 2
                exp3 = 2 * alpha * denom + beta * phi
                exp4 = (beta * phi * (1 - phi**t)) / (denom ** 2 * denom2)
                exp5 = 2 * alpha * denom2 + beta * phi * (1 + 2 * phi - phi**t)
                sigmah = sigma * jnp.sqrt(1 + alpha**2 * (t - 1) + exp2 * exp3 - exp4 * exp5)
            else:
                # Non-damped trend case (or phi very close to 1.0)
                exp1 = alpha**2 + alpha * beta * t + (1 / 6) * beta**2 * t * (2 * t - 1)
                sigmah = sigma * jnp.sqrt(1 + (t - 1) * exp1)
        else:
            # Multiplicative error
            if self.damped and phi < 0.9999:  # Avoid division by zero
                denom = jnp.maximum(1 - phi, _EPSILON)
                denom2 = jnp.maximum(1 - phi**2, _EPSILON)
                exp2 = (beta * phi * t) / denom ** 2
                exp3 = 2 * alpha * denom + beta * phi
                exp4 = (beta * phi * (1 - phi**t)) / (denom ** 2 * denom2)
                exp5 = 2 * alpha * denom2 + beta * phi * (1 + 2 * phi - phi**t)
                sigmah_base = jnp.sqrt(1 + alpha**2 * (t - 1) + exp2 * exp3 - exp4 * exp5)
            else:
                exp1 = alpha**2 + alpha * beta * t + (1 / 6) * beta**2 * t * (2 * t - 1)
                sigmah_base = jnp.sqrt(1 + (t - 1) * exp1)
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

    def __init__(
        self,
        season_length: int = 1,
        error_type: str = 'A',
        damped: bool | None = None,
        phi: float | None = None,
        alias: str = "Holt",
        conformal_params: ConformalIntervals | None = None,
        allow_extended_iterations: bool = False,
        iteration_scaling: str = "quadratic",
    ):
        """
        Holt's linear exponential smoothing method.

        Parameters
        ----------
        season_length : int, default=1
            Number of observations per unit of time. (Not used in current
            implementation but kept for API consistency.)
        error_type : str, default='A'
            Type of error: 'A' (additive) or 'M' (multiplicative).
            Must be either 'A' or 'M'.
        damped : bool | None, default=None
            Whether to use damped trend. If None, treated as False (non-damped).
        phi : float | None, default=None
            Damping parameter, must be in [0.8, 0.98]. Only used if damped=True.
            If damped=True and phi=None, defaults to 0.9.
        alias : str, default="Holt"
            Custom name for the model.
        conformal_params : ConformalIntervals | None, default=None
            Parameters for conformal prediction intervals. If None, uses native
            analytical prediction intervals.
        allow_extended_iterations : bool, default=False
            Whether to allow extended iteration counts (up to 400) for difficult
            series. Default max is 200.
        iteration_scaling : str, default="quadratic"
            Scaling method for adaptive iterations. "quadratic" (default) gives
            moderate scaling, "cubic" gives more aggressive scaling for complex series.

        Raises
        ------
        ValueError
            If error_type is not 'A' or 'M'.
            If phi is not a float when provided.
            If phi is outside the valid range [0.8, 0.98].
            If conformal_params is not a ConformalIntervals instance.
            If iteration_scaling is not 'quadratic' or 'cubic'.
        """
        # Validate error_type
        if error_type not in ('A', 'M'):
            raise ValueError(
                f"error_type must be 'A' (additive) or 'M' (multiplicative), got '{error_type}'"
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
    ) -> 'Holt':
        """Fit the Holt model to training data.

        This method estimates the smoothing parameters (alpha, beta) and computes
        the level and trend states by maximizing the log-likelihood using gradient
        descent optimization with JAX.

        Parameters
        ----------
        y : jnp.ndarray
            Training time series data of shape (n,). Must have at least 2 observations.
        X : jnp.ndarray | None, default=None
            Exogenous variables (not currently used, included for API consistency).

        Returns
        -------
        self : Holt
            The fitted model instance.

        Raises
        ------
        ValueError
            If y has fewer than 2 observations.

        Notes
        -----
        The fitted model stores the following in `self.model_`:
        - fitted: In-sample fitted values
        - level: Final level state after processing all observations
        - trend: Final trend state after processing all observations
        - alpha: Estimated level smoothing parameter
        - beta: Estimated trend smoothing parameter
        - sigma: Residual standard error
        - residuals: Forecast residuals
        - y_train: Original training data (for conformal prediction)
        """
        y = utils.ensure_float(y)

        # Validate minimum series length
        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")

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
        Predict with fitted Holt model.

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
        Access fitted Holt model insample predictions.

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
        Memory efficient Holt predictions.

        This method avoids memory burden from object storage.
        It is analogous to fit_predict without storing information.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). Must have at least 2 observations.
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
            If y has fewer than 2 observations, if h is not positive, or
            if level values are outside [0, 100].
        """
        y = utils.ensure_float(y)
        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")
        self._validate_h(h)
        self._validate_level(level)
        result = self._fit_parameters(y)
        phi = self._get_phi()

        mean = self._generate_forecasts(result['level'], result['trend'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                temp_model = self.model_ if hasattr(self, 'model_') else None
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
                if temp_model is not None:
                    self.model_ = temp_model
                else:
                    delattr(self, 'model_')
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    result['sigma'],
                    result['alpha'],
                    result['beta'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res,
                    result['fitted'],
                    result['sigma'],
                    level,
                    prefix='fitted-'
                )

        return res

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
        Apply fitted Holt model to a new time series.

        This method uses the model structure (error_type, damped, phi) from
        the original fit, but re-estimates parameters on the new data.

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). Must have at least 2 observations.
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
            If model is not fitted, if y has fewer than 2 observations,
            if h is not positive, or if level values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling forward(). Call fit() first.")

        y = utils.ensure_float(y)
        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")
        self._validate_h(h)
        self._validate_level(level)
        result = self._fit_parameters(y)
        phi = self._get_phi()

        mean = self._generate_forecasts(result['level'], result['trend'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                temp_model = self.model_
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
                self.model_ = temp_model
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    result['sigma'],
                    result['alpha'],
                    result['beta'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res,
                    result['fitted'],
                    result['sigma'],
                    level,
                    prefix='fitted-'
                )

        return res
