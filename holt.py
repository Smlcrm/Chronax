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

Example:
    >>> import jax.numpy as jnp
    >>> from holt import Holt
    >>>
    >>> # Create and fit model
    >>> model = Holt(error_type='A', damped=False)
    >>> y = jnp.array([10, 12, 15, 18, 22, 27, 33, 40])
    >>> model.fit(y)
    >>>
    >>> # Generate forecasts with 95% prediction intervals
    >>> predictions = model.predict(h=5, level=[95])
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
_N_ITER = 1000
_LEARNING_RATE = 0.01
_LR_DECAY_STEPS = 500
_LR_DECAY_RATE = 0.9
_EPSILON = 1e-10  # For numerical stability

__all__ = ['Holt']

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
        """Initialize level and trend via linear regression."""
        n_init = min(10, len(y) // 2)
        if n_init >= 2:
            import numpy as np
            X_init = np.column_stack([np.ones(n_init), np.arange(n_init)])
            coef = np.linalg.lstsq(X_init, np.array(y[:n_init]), rcond=None)[0]
            l0, b0 = float(coef[0]), float(coef[1])
        else:
            l0, b0 = float(y[0]), 0.0
        return l0, b0

    def _get_phi(self) -> float:
        """Get damping factor phi."""
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        else:
            return 1.0

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Fit model parameters and return results dictionary."""
        l0, b0 = self._initialize_states(y)
        phi = self._get_phi()

        def step_update(carry, y_t):
            level_prev, trend_prev, alpha, beta = carry
            y_hat = level_prev + phi * trend_prev
            if self.error_type == 'A':
                level = alpha * y_t + (1 - alpha) * (level_prev + phi * trend_prev)
                trend = beta * (level - level_prev) + (1 - beta) * phi * trend_prev
            else:
                epsilon = (y_t - y_hat) / jnp.maximum(jnp.abs(y_hat), _EPSILON)
                level = (level_prev + phi * trend_prev) * (1 + alpha * epsilon)
                trend = phi * trend_prev + beta * (level_prev + phi * trend_prev) * epsilon
            return (level, trend, alpha, beta), y_hat

        l0_fixed, b0_fixed = l0, b0

        @jax.jit
        def raw_fit(params_ab):
            alpha, beta = params_ab
            init_carry = (l0_fixed, b0_fixed, alpha, beta)
            final_carry, fitted_vals = lax.scan(step_update, init_carry, y)
            final_level, final_trend, _, _ = final_carry
            residuals = y - fitted_vals
            return {
                'fitted': fitted_vals,
                'level': final_level,
                'trend': final_trend,
                'residuals': residuals,
                'alpha': alpha,
                'beta': beta,
            }

        @jax.jit
        def _likelihood_loss(params_ab):
            result = raw_fit(params_ab)
            residuals = result['residuals']
            sse = jnp.sum(residuals ** 2)
            if self.error_type == 'M':
                fitted_vals = result['fitted']
                log_det = 2 * jnp.sum(jnp.log(jnp.maximum(jnp.abs(fitted_vals), _EPSILON)))
                return len(y) * jnp.log(jnp.maximum(sse, _EPSILON)) + log_det
            else:
                return len(y) * jnp.log(jnp.maximum(sse, _EPSILON))

        init_params = jnp.array([_INIT_ALPHA, _INIT_BETA])
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

        Raises
        ------
        ValueError
            If error_type is not 'A' or 'M'.
            If phi is not a float when provided.
            If phi is outside the valid range [0.8, 0.98].
            If conformal_params is not a ConformalIntervals instance.
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

        self.season_length = season_length
        self.error_type = error_type
        self.damped = damped if damped is not None else False
        self.phi = phi
        self.alias = alias
        self.conformal_params = conformal_params

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


def test():
    """Comprehensive test suite for Holt model."""
    print("="*60)
    print("Running Holt model tests...")
    print("="*60)
    passed = 0
    failed = 0

    # Test 1: Basic fit/predict (Additive Error)
    try:
        print("\nTest 1: Basic fit/predict (Additive Error)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=5)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 5, "Forecast length should be 5"
        assert jnp.all(jnp.isfinite(result['mean'])), "Forecasts should be finite"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print(f"  First 3 forecasts: {result['mean'][:3]}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 2: Multiplicative Error
    try:
        print("\nTest 2: Multiplicative Error")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='M', damped=False)
        model.fit(y)
        result = model.predict(h=5)

        assert 'mean' in result, "Result should contain 'mean' key"
        assert len(result['mean']) == 5, "Forecast length should be 5"

        print(f"  Fitted alpha: {model.model_['alpha']:.4f}")
        print(f"  Fitted beta: {model.model_['beta']:.4f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 3: Damped Trend
    try:
        print("\nTest 3: Damped Trend")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])

        # Non-damped model
        model_nodamp = Holt(error_type='A', damped=False)
        model_nodamp.fit(y)
        result_nodamp = model_nodamp.predict(h=10)

        # Damped model
        model_damp = Holt(error_type='A', damped=True, phi=0.9)
        model_damp.fit(y)
        result_damp = model_damp.predict(h=10)

        # At longer horizons, damped should be less than non-damped
        last_forecast_damp = result_damp['mean'][-1]
        last_forecast_nodamp = result_nodamp['mean'][-1]

        assert last_forecast_damp < last_forecast_nodamp, \
            "Damped forecast should be lower at long horizons"

        print(f"  Non-damped h=10: {last_forecast_nodamp:.2f}")
        print(f"  Damped h=10: {last_forecast_damp:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 4: Prediction Intervals (Native)
    try:
        print("\nTest 4: Prediction Intervals (Native)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y)
        result = model.predict(h=5, level=[80, 95])

        assert 'lo-80' in result, "Should have lo-80"
        assert 'hi-80' in result, "Should have hi-80"
        assert 'lo-95' in result, "Should have lo-95"
        assert 'hi-95' in result, "Should have hi-95"

        # Check ordering: lo-95 < lo-80 < mean < hi-80 < hi-95
        for i in range(5):
            assert result['lo-95'][i] < result['lo-80'][i], "lo-95 < lo-80"
            assert result['lo-80'][i] < result['mean'][i], "lo-80 < mean"
            assert result['mean'][i] < result['hi-80'][i], "mean < hi-80"
            assert result['hi-80'][i] < result['hi-95'][i], "hi-80 < hi-95"

        print(f"  h=1: [{result['lo-95'][0]:.2f}, {result['mean'][0]:.2f}, {result['hi-95'][0]:.2f}]")
        print(f"  h=5: [{result['lo-95'][4]:.2f}, {result['mean'][4]:.2f}, {result['hi-95'][4]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 5: Conformal Intervals
    try:
        print("\nTest 5: Conformal Intervals")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 57.0])
        conformal = ConformalIntervals(n_windows=3, h=2)
        model = Holt(error_type='A', damped=False, conformal_params=conformal)
        model.fit(y)
        result = model.predict(h=2, level=[95])

        assert 'lo-95' in result, "Should have conformal lo-95"
        assert 'hi-95' in result, "Should have conformal hi-95"

        print(f"  Conformal interval h=1: [{result['lo-95'][0]:.2f}, {result['hi-95'][0]:.2f}]")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 6: predict_in_sample()
    try:
        print("\nTest 6: predict_in_sample()")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y)
        result = model.predict_in_sample(level=[95])

        assert 'fitted' in result, "Should have fitted values"
        assert len(result['fitted']) == len(y), "Fitted should match training length"
        assert 'fitted-lo-95' in result, "Should have fitted intervals"
        assert 'fitted-hi-95' in result, "Should have fitted intervals"

        print(f"  Fitted length: {len(result['fitted'])}")
        print(f"  First fitted value: {result['fitted'][0]:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 7: forecast() Method
    try:
        print("\nTest 7: forecast() Method (Stateless)")
        y = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        model = Holt(error_type='A', damped=False)
        # Don't call fit()
        result = model.forecast(y, h=5, fitted=True, level=[95])

        assert 'mean' in result, "Should have forecasts"
        assert 'fitted' in result, "Should have fitted values with fitted=True"
        assert len(result['mean']) == 5, "Forecast length should be 5"
        assert len(result['fitted']) == len(y), "Fitted length should match y"

        print(f"  Forecast h=1: {result['mean'][0]:.2f}")
        print(f"  Includes fitted values and intervals")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 8: forward() Method
    try:
        print("\nTest 8: forward() Method")
        y1 = jnp.array([10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
        y2 = jnp.array([100.0, 102.0, 105.0, 108.0, 112.0, 117.0, 123.0, 130.0])

        model = Holt(error_type='A', damped=False)
        model.fit(y1)

        # Apply to new series
        result = model.forward(y2, h=3)

        assert 'mean' in result, "Should have forecasts"
        assert len(result['mean']) == 3, "Forecast length should be 3"
        # Forecasts should be in range of y2, not y1
        assert result['mean'][0] > 100, "Forecast should follow y2 scale"

        print(f"  y1 range: [{float(y1.min()):.1f}, {float(y1.max()):.1f}]")
        print(f"  y2 range: [{float(y2.min()):.1f}, {float(y2.max()):.1f}]")
        print(f"  Forward forecast: {result['mean'][0]:.2f}")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Test 9: Edge Cases
    try:
        print("\nTest 9: Edge Cases")

        # Test minimum data length (should work with 2 observations)
        y_min = jnp.array([10.0, 12.0])
        model = Holt(error_type='A', damped=False)
        model.fit(y_min)
        result = model.predict(h=2)
        assert len(result['mean']) == 2, "Should work with 2 observations"

        # Test that 1 observation fails
        y_fail = jnp.array([10.0])
        model2 = Holt(error_type='A', damped=False)
        try:
            model2.fit(y_fail)
            raise AssertionError("Should raise ValueError for len(y) < 2")
        except ValueError:
            pass  # Expected

        print(f"  Minimum length (2): {len(y_min)} ✓")
        print(f"  Rejects length 1 ✓")
        print("  ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        failed += 1

    # Summary
    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{passed+failed}")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed.")
    print("="*60)


if __name__ == '__main__':
    test()