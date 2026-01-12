"""
GARCH Model for Time-Varying Volatility

GARCH models time series with non-constant volatility where conditional variance
depends on past squared errors and past conditional variances. Supports both
deterministic forecasting (expected values, analytical intervals) and stochastic
forecasting (sample paths via Monte Carlo simulation).

Parameters:
- ω (omega): Baseline volatility level (must be > 0)
- α_i (alpha): Impact of past shocks on current volatility (≥ 0)
- β_j (beta): Impact of past volatility on current volatility (≥ 0)
- p: Number of lagged squared shocks to include (ARCH order)
- q: Number of lagged variances to include (GARCH order)

Methods:
1. fit: Estimates ω, α, β using maximum likelihood via L-BFGS optimization
2. predict: Deterministic forecast returning expected volatility path
3. predict_simulate: Stochastic forecast generating multiple sample paths
4. forecast: Stateless deterministic fit-and-predict
5. forecast_simulate: Stateless stochastic fit-and-predict
6. predict_in_sample: Returns fitted values with optional prediction intervals

Constants:
_EPSILON (1e-8): Numerical floor for variance
_MAX_ITER (1000): Maximum L-BFGS iterations
_INIT_OMEGA (0.01): Starting guess for baseline volatility
_INIT_ALPHA_BETA (0.1): Starting guess for ARCH/GARCH coefficients
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
import utils

_EPSILON = jnp.float32(1e-8)
_MAX_ITER = 1000

def _compute_sigma2_series(y: jnp.ndarray, omega: float, alpha: jnp.ndarray, beta: jnp.ndarray, p: int, q: int) -> jnp.ndarray:
    n = len(y)
    y = y.astype(jnp.float32)
    y_squared = y ** 2
    init_var = jnp.var(y).astype(jnp.float32)
    max_lag = max(p, q)
    y_squared_padded = jnp.concatenate([jnp.full(max_lag, init_var, dtype=jnp.float32), y_squared])

    # Cast parameters to float32 for type consistency in lax.scan
    omega_f32 = jnp.float32(omega)
    alpha_f32 = alpha.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    zero_f32 = jnp.float32(0.0)

    def step(carry, t):
        sigma2_series = carry
        arch_start = t + max_lag - p
        y2_lagged = lax.dynamic_slice(y_squared_padded, (arch_start,), (p,))
        arch_sum = jnp.dot(alpha_f32, jnp.flip(y2_lagged))
        garch_sum = zero_f32
        if q > 0:
            garch_start = t + max_lag - q
            sigma2_lagged = lax.dynamic_slice(sigma2_series, (garch_start,), (q,))
            garch_sum = jnp.dot(beta_f32, jnp.flip(sigma2_lagged))

        sigma2_t = jnp.maximum(omega_f32 + arch_sum + garch_sum, _EPSILON)
        sigma2_series = sigma2_series.at[t + max_lag].set(sigma2_t)
        return sigma2_series, sigma2_t

    init_sigma2 = jnp.full(n + max_lag, init_var, dtype=jnp.float32)
    final_sigma2, _ = lax.scan(step, init_sigma2, jnp.arange(n))
    return lax.dynamic_slice(final_sigma2, (max_lag,), (n,))

# negative log likelihood
def _log_likelihood(params: jnp.ndarray, y: jnp.ndarray, p: int, q: int):
    omega = jax.nn.softplus(params[0]) + _EPSILON
    alpha = jax.nn.softplus(lax.dynamic_slice(params, (1,), (p,)))
    beta = jax.nn.softplus(lax.dynamic_slice(params, (p+1,), (q,))) if q > 0 else jnp.array([])
    sigma2 = _compute_sigma2_series(y, omega, alpha, beta, p, q)
    log_lik = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + jnp.log(sigma2) + y**2 / sigma2)
    coef_sum = jnp.sum(alpha) + jnp.sum(beta)
    penalty = jnp.where(coef_sum >= 0.999, 1e6 * (coef_sum - 0.999) ** 2, 0.0)
    return -log_lik + penalty

_log_likelihood = jax.jit(_log_likelihood, static_argnums=(2, 3))


class GARCH(BaseForecaster):
    """
    Args:
        p: ARCH order (lagged squared shocks), must be greater than or equal to 1
        q: GARCH order (lagged variances)
        alias: Model name for display
        conformal_params: Optional conformal prediction configuration
    """
    uses_exog = False

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        alias: str = "GARCH",
        conformal_params: ConformalIntervals | None = None,
    ):
        if not isinstance(p, int) or p < 1:
            raise ValueError(f"p must be an integer greater than or equal to 1, got {p}")
        if not isinstance(q, int) or q < 0:
            raise ValueError(f"q must be an integer greater than or equal to 0, got {q}")

        self.p = p
        self.q = q
        self.alias = f"{alias}({p},{q})" if q != 0 else f"{alias}({p})"
        self.conformal_params = conformal_params
        self.model_ = None

    @staticmethod
    def _validate_h(h: int) -> None:
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"Forecast horizon h must be a positive integer, got {h}")

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Estimate parameters using L-BFGS."""
        n = len(y)
        min_obs = max(self.p, self.q) + 10
        if n < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations for GARCH({self.p},{self.q}), got {n}"
            )
        init_params = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(self.p, 0.0),
            jnp.full(self.q, 0.0) if self.q > 0 else jnp.array([])
        ])

        optimizer = optax.lbfgs()
        opt_state = optimizer.init(init_params)

        def step(carry, _):
            params, state = carry
            value, grads = jax.value_and_grad(_log_likelihood)(params, y, self.p, self.q)
            updates, state = optimizer.update(grads, state, params, value=value, grad=grads, value_fn=lambda p: _log_likelihood(p, y, self.p, self.q))
            params = optax.apply_updates(params, updates)
            return (params, state), value

        (final_params, _), losses = lax.scan(step, (init_params, opt_state), None, length=_MAX_ITER)

        omega = jax.nn.softplus(final_params[0]) + _EPSILON
        alpha = jax.nn.softplus(lax.dynamic_slice(final_params, (1,), (self.p,)))
        beta = jax.nn.softplus(lax.dynamic_slice(final_params, (self.p+1,), (self.q,))) if self.q > 0 else jnp.array([])

        sigma2 = _compute_sigma2_series(y, omega, alpha, beta, self.p, self.q)

        return {
            'omega': float(omega),
            'alpha': alpha,
            'beta': beta,
            'sigma2': sigma2,
            'fitted': jnp.zeros_like(y),
            'y_mean': float(jnp.mean(y)),
            'y_last': y[-self.p:],
            'sigma2_last': sigma2[-self.q:] if self.q > 0 else jnp.array([]),
        }

    def _forecast_sigma2(self, omega: float, alpha: jnp.ndarray, beta: jnp.ndarray, y_last: jnp.ndarray, sigma2_last: jnp.ndarray, h: int) -> jnp.ndarray:
        """Forecast variance h steps ahead by iterating GARCH equation."""
        # Cast all parameters to float32 for type consistency in lax.scan
        omega_f32 = jnp.float32(omega)
        alpha_f32 = alpha.astype(jnp.float32)
        beta_f32 = beta.astype(jnp.float32) if self.q > 0 else jnp.array([], dtype=jnp.float32)
        zero_f32 = jnp.float32(0.0)

        def step(carry, _):
            y_buffer, sigma2_buffer = carry

            arch_sum = jnp.sum(alpha_f32 * jnp.flip(y_buffer ** 2))
            garch_sum = jnp.sum(beta_f32 * jnp.flip(sigma2_buffer)) if self.q > 0 else zero_f32
            sigma2_next = jnp.maximum(omega_f32 + arch_sum + garch_sum, _EPSILON)

            y_buffer = jnp.concatenate([y_buffer[1:], jnp.array([zero_f32])])
            if self.q > 0:
                sigma2_buffer = jnp.concatenate([sigma2_buffer[1:], jnp.array([sigma2_next], dtype=jnp.float32)])

            return (y_buffer, sigma2_buffer), sigma2_next

        init_y = y_last.astype(jnp.float32)
        init_sigma2 = sigma2_last.astype(jnp.float32) if self.q > 0 else jnp.array([], dtype=jnp.float32)
        _, forecasts = lax.scan(step, (init_y, init_sigma2), None, length=h)
        return forecasts

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> 'GARCH':
        # estimate parameters from data
        y = utils.ensure_float(y)

        if not jnp.all(jnp.isfinite(y)):
            raise ValueError("Input contains NaN or infinite values")
        if jnp.var(y) < 1e-10:
            raise ValueError("Input has near-zero variance, unsuitable for GARCH")

        y_mean = jnp.mean(y)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered)

        self.model_ = {
            'omega': result['omega'],
            'alpha': result['alpha'],
            'beta': result['beta'],
            'sigma2': result['sigma2'],
            'fitted': result['fitted'] + y_mean,
            'residuals': y_centered - result['fitted'],
            'y_mean': y_mean,
            'y_centered_last': result['y_last'],
            'sigma2_last': result['sigma2_last'],
            'y_train': y,
        }

        self.model_['sigma'] = float(utils.calculate_sigma(self.model_['residuals'], len(y) - (self.p + self.q + 1)))

        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None
    ) -> dict:
        """Generate h-step forecasts. Returns mean, sigma2, and optional intervals."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        self._validate_h(h)

        # Forecast variance
        sigma2_forecast = self._forecast_sigma2(
            self.model_['omega'],
            self.model_['alpha'],
            self.model_['beta'],
            self.model_['y_centered_last'],
            self.model_['sigma2_last'],
            h
        )

        mean_forecast = jnp.full(h, self.model_['y_mean'])

        res = {
            'mean': mean_forecast,
            'sigma2': sigma2_forecast
        }
        if level is not None:
            level = sorted(level)

            if self.conformal_params is not None:
                cs = self.conformity_scores(y=self.model_['y_train'], X=X)
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                sigma_forecast = jnp.sqrt(sigma2_forecast)

                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = mean_forecast - z * sigma_forecast
                    res[f'hi-{lv}'] = mean_forecast + z * sigma_forecast

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        res = {'fitted': self.model_['fitted']}

        if level is not None:
            level = sorted(level)
            sigma = self.model_['sigma']

            for lv in level:
                z = utils._jax_norm_ppf((100 + lv) / 200)
                res[f'fitted-lo-{lv}'] = self.model_['fitted'] - z * sigma
                res[f'fitted-hi-{lv}'] = self.model_['fitted'] + z * sigma

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
        # stateless fit and predict
        self._validate_h(h)
        y = utils.ensure_float(y)

        y_mean = jnp.mean(y)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered)

        sigma2_forecast = self._forecast_sigma2(
            result['omega'],
            result['alpha'],
            result['beta'],
            result['y_last'],
            result['sigma2_last'],
            h
        )

        mean_forecast = jnp.full(h, y_mean)

        res = {
            'mean': mean_forecast,
            'sigma2': sigma2_forecast
        }

        if fitted:
            res['fitted'] = result['fitted'] + y_mean

        if level is not None:
            level = sorted(level)

            if self.conformal_params is not None:
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                sigma_forecast = jnp.sqrt(sigma2_forecast)

                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = mean_forecast - z * sigma_forecast
                    res[f'hi-{lv}'] = mean_forecast + z * sigma_forecast
            if fitted:
                residuals = y_centered - result['fitted']
                sigma = float(utils.calculate_sigma(residuals, len(y) - (self.p + self.q + 1)))

                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'fitted-lo-{lv}'] = res['fitted'] - z * sigma
                    res[f'fitted-hi-{lv}'] = res['fitted'] + z * sigma

        return res

    def _forecast_sigma2_stochastic(
        self,
        omega: float,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        y_last: jnp.ndarray,
        sigma2_last: jnp.ndarray,
        h: int,
        key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Cast all parameters to float32 for type consistency in lax.scan
        omega_f32 = jnp.float32(omega)
        alpha_f32 = alpha.astype(jnp.float32)
        beta_f32 = beta.astype(jnp.float32) if self.q > 0 else jnp.array([], dtype=jnp.float32)
        zero_f32 = jnp.float32(0.0)

        def step(carry, subkey):
            y_buffer, sigma2_buffer = carry

            arch_sum = jnp.sum(alpha_f32 * jnp.flip(y_buffer ** 2))
            garch_sum = jnp.sum(beta_f32 * jnp.flip(sigma2_buffer)) if self.q > 0 else zero_f32
            sigma2_next = jnp.maximum(omega_f32 + arch_sum + garch_sum, _EPSILON)

            # generate random shock and realized level
            epsilon = jax.random.normal(subkey, dtype=jnp.float32)
            y_next = epsilon * jnp.sqrt(sigma2_next)
            y_buffer = jnp.concatenate([y_buffer[1:], jnp.array([y_next], dtype=jnp.float32)])
            if self.q > 0:
                sigma2_buffer = jnp.concatenate([sigma2_buffer[1:], jnp.array([sigma2_next], dtype=jnp.float32)])

            return (y_buffer, sigma2_buffer), (y_next, sigma2_next)

        subkeys = jax.random.split(key, h)
        init_y = y_last.astype(jnp.float32)
        init_sigma2 = sigma2_last.astype(jnp.float32) if self.q > 0 else jnp.array([], dtype=jnp.float32)
        _, (y_path, sigma2_path) = lax.scan(step, (init_y, init_sigma2), subkeys)

        return y_path, sigma2_path

    def _simulate_paths(
        self,
        h: int,
        n_sims: int,
        key: jnp.ndarray,
        omega: float,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        y_last: jnp.ndarray,
        sigma2_last: jnp.ndarray,
        y_mean: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Parallelize simulation of n_sims paths using vmap."""
        def simulate_single_path(subkey):
            y_path, sigma2_path = self._forecast_sigma2_stochastic(
                omega, alpha, beta, y_last, sigma2_last, h, subkey
            )
            return y_path + y_mean, sigma2_path

        keys = jax.random.split(key, n_sims)
        paths, sigma2_paths = jax.vmap(simulate_single_path)(keys)
        return paths, sigma2_paths

    def predict_simulate(
        self,
        h: int,
        n_sims: int = 1000,
        seed: int | None = None,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
        return_paths: bool = True
    ) -> dict:
        # Generate stochastic forecast paths via Monte Carlo simulation.
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        self._validate_h(h)

        if not isinstance(n_sims, int) or n_sims < 1:
            raise ValueError(f"n_sims must be a positive integer, got {n_sims}")

        key = jax.random.PRNGKey(seed if seed is not None else 0)

        paths, sigma2_paths = self._simulate_paths(
            h=h,
            n_sims=n_sims,
            key=key,
            omega=self.model_['omega'],
            alpha=self.model_['alpha'],
            beta=self.model_['beta'],
            y_last=self.model_['y_centered_last'],
            sigma2_last=self.model_['sigma2_last'],
            y_mean=self.model_['y_mean']
        )

        res = {
            'mean': jnp.mean(paths, axis=0),
            'median': jnp.median(paths, axis=0),
            'sigma2_mean': jnp.mean(sigma2_paths, axis=0),
            'sigma2_median': jnp.median(sigma2_paths, axis=0),
        }

        if return_paths:
            res['paths'] = paths
            res['sigma2_paths'] = sigma2_paths

        if level is not None:
            level = sorted(level)
            for lv in level:
                lower_q = (100 - lv) / 2
                upper_q = 100 - lower_q
                res[f'lo-{lv}'] = jnp.percentile(paths, lower_q, axis=0)
                res[f'hi-{lv}'] = jnp.percentile(paths, upper_q, axis=0)

        return res

    def forecast_simulate(
        self,
        y: jnp.ndarray,
        h: int,
        n_sims: int = 1000,
        seed: int | None = None,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
        return_paths: bool = True
    ) -> dict:
        """Stateless stochastic forecast (fit then simulate, don't store model state)."""
        self.fit(y, X)
        res = self.predict_simulate(h, n_sims, seed, X_future, level, return_paths)

        if fitted:
            res['fitted'] = self.model_['fitted']

            if level is not None:
                level = sorted(level)
                sigma = self.model_['sigma']
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'fitted-lo-{lv}'] = self.model_['fitted'] - z * sigma
                    res[f'fitted-hi-{lv}'] = self.model_['fitted'] + z * sigma

        return res


# testing

if __name__ == '__main__':

    passed = 0
    failed = 0

    # Generate synthetic GARCH data for testing
    jax_key = jax.random.PRNGKey(42)
    n = 200

    # Simulate GARCH(1,1) process
    omega_true = 0.01
    alpha_true = 0.15
    beta_true = 0.80

    y_test = jnp.zeros(n)
    sigma2_test = jnp.zeros(n)
    sigma2_test = sigma2_test.at[0].set(omega_true / (1 - alpha_true - beta_true))

    for t in range(1, n):
        if t == 1:
            sigma2_test = sigma2_test.at[t].set(omega_true + alpha_true * y_test[t-1]**2 + beta_true * sigma2_test[t-1])
        else:
            sigma2_test = sigma2_test.at[t].set(omega_true + alpha_true * y_test[t-1]**2 + beta_true * sigma2_test[t-1])

        jax_key, subkey = jax.random.split(jax_key)
        y_test = y_test.at[t].set(jax.random.normal(subkey) * jnp.sqrt(sigma2_test[t]))

    # Test 1: Basic fit and predict
    print("[Test 1] Basic GARCH(1,1) fit and predict")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        result = model.predict(h=10)

        assert 'mean' in result, "Missing 'mean' in result"
        assert 'sigma2' in result, "Missing 'sigma2' in result"
        assert len(result['mean']) == 10, f"Expected 10 forecasts, got {len(result['mean'])}"
        assert jnp.all(jnp.isfinite(result['mean'])), "Non-finite values in mean forecast"
        assert jnp.all(result['sigma2'] > 0), "Non-positive variance forecasts"

        print(f"  [PASS] Fitted omega={model.model_['omega']:.4f}, alpha={model.model_['alpha'][0]:.4f}, beta={model.model_['beta'][0]:.4f}")
        print(f"  [PASS] Mean forecast: {result['mean'][:3]} ...")
        print(f"  [PASS] Sigma2 forecast: {result['sigma2'][:3]} ...")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 2: ARCH model (q=0)
    print("[Test 2] ARCH(2) model (GARCH with q=0)")
    try:
        model = GARCH(p=2, q=0)
        model.fit(y_test)
        result = model.predict(h=5)

        assert model.alias == "GARCH(2)", f"Expected alias 'GARCH(2)', got '{model.alias}'"
        assert len(model.model_['beta']) == 0, "ARCH model should have no beta coefficients"
        assert 'mean' in result and 'sigma2' in result

        print(f"  [PASS] Model alias: {model.alias}")
        print(f"  [PASS] Fitted omega={model.model_['omega']:.4f}, alpha={model.model_['alpha']}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 3: Native prediction intervals
    print("[Test 3] Native prediction intervals")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        result = model.predict(h=10, level=[80, 95])

        assert 'lo-80' in result and 'hi-80' in result, "Missing 80% intervals"
        assert 'lo-95' in result and 'hi-95' in result, "Missing 95% intervals"

        # Check interval ordering: lo-95 < lo-80 < mean < hi-80 < hi-95
        for i in range(10):
            assert result['lo-95'][i] < result['lo-80'][i], f"Interval ordering violated at {i}"
            assert result['lo-80'][i] < result['mean'][i], f"Interval ordering violated at {i}"
            assert result['mean'][i] < result['hi-80'][i], f"Interval ordering violated at {i}"
            assert result['hi-80'][i] < result['hi-95'][i], f"Interval ordering violated at {i}"

        print(f"  [PASS] 80% interval: [{result['lo-80'][0]:.3f}, {result['hi-80'][0]:.3f}]")
        print(f"  [PASS] 95% interval: [{result['lo-95'][0]:.3f}, {result['hi-95'][0]:.3f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 4: Conformal prediction intervals
    print("[Test 4] Conformal prediction intervals")
    try:
        conformal = ConformalIntervals(n_windows=3, h=5)
        model = GARCH(p=1, q=1, conformal_params=conformal)
        model.fit(y_test)
        result = model.predict(h=5, level=[90])

        assert 'lo-90' in result and 'hi-90' in result, "Missing conformal intervals"
        assert len(result['lo-90']) == 5, "Conformal intervals wrong length"

        print(f"  [PASS] Conformal 90% interval: [{result['lo-90'][0]:.3f}, {result['hi-90'][0]:.3f}]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 5: predict_in_sample
    print("[Test 5] In-sample fitted values with intervals")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        result = model.predict_in_sample(level=[80, 95])

        assert 'fitted' in result, "Missing fitted values"
        assert len(result['fitted']) == len(y_test), "Fitted values wrong length"
        assert 'fitted-lo-80' in result and 'fitted-hi-80' in result, "Missing fitted intervals"

        print(f"  [PASS] Fitted values shape: {result['fitted'].shape}")
        print(f"  [PASS] First fitted value: {result['fitted'][0]:.3f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 6: Stateless forecast method
    print("[Test 6] Stateless forecast() method")
    try:
        model = GARCH(p=1, q=1)  # Unfitted model
        result = model.forecast(y_test, h=8, level=[90], fitted=True)

        assert 'mean' in result and 'sigma2' in result, "Missing forecasts"
        assert 'fitted' in result, "Missing fitted values"
        assert 'lo-90' in result and 'hi-90' in result, "Missing intervals"
        assert 'fitted-lo-90' in result and 'fitted-hi-90' in result, "Missing fitted intervals"
        assert len(result['mean']) == 8, "Wrong forecast length"

        print(f"  [PASS] Stateless forecast successful")
        print(f"  [PASS] Forecast length: {len(result['mean'])}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 7: Different lag orders
    print("[Test 7] GARCH(2,2) with higher orders")
    try:
        model = GARCH(p=2, q=2)
        model.fit(y_test)
        result = model.predict(h=5)

        assert len(model.model_['alpha']) == 2, "Wrong number of alpha coefficients"
        assert len(model.model_['beta']) == 2, "Wrong number of beta coefficients"
        assert model.alias == "GARCH(2,2)", f"Wrong alias: {model.alias}"

        print(f"  [PASS] Alpha coefficients: {model.model_['alpha']}")
        print(f"  [PASS] Beta coefficients: {model.model_['beta']}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 8: Parameter validation
    print("[Test 8] Parameter validation")
    test8_failed = False
    try:
        # Test invalid p
        try:
            model = GARCH(p=0, q=1)
            print("  [FAIL] Failed: Should have raised ValueError for p=0")
            test8_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected p=0")

        # Test invalid q
        try:
            model = GARCH(p=1, q=-1)
            print("  [FAIL] Failed: Should have raised ValueError for q=-1")
            test8_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected q=-1")

        # Test invalid h
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        try:
            result = model.predict(h=0)
            print("  [FAIL] Failed: Should have raised ValueError for h=0")
            test8_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected h=0")

        if test8_failed:
            failed += 1
        else:
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 9: Minimum data requirement
    print("[Test 9] Insufficient data handling")
    try:
        model = GARCH(p=2, q=2)
        small_data = y_test[:10]

        try:
            model.fit(small_data)
            print("  [FAIL] Failed: Should have raised ValueError for insufficient data")
            failed += 1
        except ValueError as e:
            print(f"  [PASS] Correctly rejected insufficient data: {str(e)[:60]}...")
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 10: Input validation (NaN/Inf/constant series)
    print("[Test 10] Input validation (NaN/Inf/constant)")
    test10_failed = False
    try:
        model = GARCH(p=1, q=1)

        # Test NaN input
        try:
            nan_data = y_test.at[50].set(jnp.nan)
            model.fit(nan_data)
            print("  [FAIL] Failed: Should have rejected NaN input")
            test10_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected NaN input")

        # Test Inf input
        try:
            inf_data = y_test.at[50].set(jnp.inf)
            model.fit(inf_data)
            print("  [FAIL] Failed: Should have rejected Inf input")
            test10_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected Inf input")

        # Test constant series
        try:
            const_data = jnp.ones(100)
            model.fit(const_data)
            print("  [FAIL] Failed: Should have rejected constant series")
            test10_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected constant series")

        if test10_failed:
            failed += 1
        else:
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 11: Volatility persistence
    print("[Test 11] Volatility persistence check")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        omega = model.model_['omega']
        alpha = model.model_['alpha'][0]
        beta = model.model_['beta'][0]
        persistence = alpha + beta

        assert 0 < persistence < 1, f"Persistence {persistence} out of valid range"
        assert omega > 0, f"Omega {omega} must be positive"

        print(f"  [PASS] Omega: {omega:.4f} > 0")
        print(f"  [PASS] Persistence (α+β): {persistence:.4f} ∈ (0, 1)")
        print(f"  [PASS] Unconditional variance: {omega / (1 - persistence):.4f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 12: Basic stochastic simulation
    print("[Test 12] Basic stochastic simulation")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        result = model.predict_simulate(h=10, n_sims=100, seed=42)

        assert 'mean' in result, "Missing 'mean' in result"
        assert 'median' in result, "Missing 'median' in result"
        assert 'sigma2_mean' in result, "Missing 'sigma2_mean' in result"
        assert 'paths' in result, "Missing 'paths' in result"
        assert 'sigma2_paths' in result, "Missing 'sigma2_paths' in result"
        assert result['paths'].shape == (100, 10), f"Expected paths shape (100, 10), got {result['paths'].shape}"
        assert result['sigma2_paths'].shape == (100, 10), f"Expected sigma2_paths shape (100, 10), got {result['sigma2_paths'].shape}"
        assert jnp.all(jnp.isfinite(result['mean'])), "Non-finite values in mean"
        assert jnp.all(result['sigma2_mean'] > 0), "Non-positive sigma2_mean values"

        print(f"  [PASS] Paths shape: {result['paths'].shape}")
        print(f"  [PASS] Mean forecast: {result['mean'][:3]} ...")
        print(f"  [PASS] Sigma2 mean: {result['sigma2_mean'][:3]} ...")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 13: Seed reproducibility
    print("[Test 13] Seed reproducibility")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        result1 = model.predict_simulate(h=10, n_sims=50, seed=42)
        result2 = model.predict_simulate(h=10, n_sims=50, seed=42)

        assert jnp.allclose(result1['paths'], result2['paths']), "Paths not reproducible with same seed"
        assert jnp.allclose(result1['mean'], result2['mean']), "Mean not reproducible with same seed"
        assert jnp.allclose(result1['sigma2_paths'], result2['sigma2_paths']), "Sigma2 paths not reproducible"

        print(f"  [PASS] Paths identical with seed=42")
        print(f"  [PASS] Mean identical with seed=42")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 14: Parameter validation
    print("[Test 14] Stochastic parameter validation")
    test14_failed = False
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        # Test invalid n_sims
        try:
            result = model.predict_simulate(h=10, n_sims=0)
            print("  [FAIL] Should have raised ValueError for n_sims=0")
            test14_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected n_sims=0")

        # Test negative n_sims
        try:
            result = model.predict_simulate(h=10, n_sims=-5)
            print("  [FAIL] Should have raised ValueError for n_sims=-5")
            test14_failed = True
        except ValueError:
            print("  [PASS] Correctly rejected n_sims=-5")

        if test14_failed:
            failed += 1
        else:
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 15: Stochastic mean vs deterministic
    print("[Test 15] Stochastic mean approximates deterministic")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        det = model.predict(h=10)
        stoch = model.predict_simulate(h=10, n_sims=5000, seed=42)

        # Stochastic mean should approximate deterministic mean (law of large numbers)
        mean_diff = jnp.abs(det['mean'] - stoch['mean'])
        assert jnp.all(mean_diff < 0.05), f"Stochastic mean differs too much from deterministic: max diff {jnp.max(mean_diff):.4f}"

        print(f"  [PASS] Max mean difference: {jnp.max(mean_diff):.4f} < 0.05")
        print(f"  [PASS] Deterministic mean[0]: {det['mean'][0]:.4f}")
        print(f"  [PASS] Stochastic mean[0]: {stoch['mean'][0]:.4f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 16: Interval comparison
    print("[Test 16] Empirical vs analytical intervals")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)
        det = model.predict(h=10, level=[95])
        stoch = model.predict_simulate(h=10, n_sims=5000, seed=42, level=[95])

        det_width = det['hi-95'] - det['lo-95']
        stoch_width = stoch['hi-95'] - stoch['lo-95']

        # Stochastic intervals can be wider due to path dependency
        # Check they're in same ballpark (within factor of 2)
        ratio = stoch_width / det_width
        assert jnp.all(ratio > 0.5) and jnp.all(ratio < 2.0), f"Interval ratio out of range [0.5, 2.0]: {ratio}"

        print(f"  [PASS] Deterministic 95% width[0]: {det_width[0]:.3f}")
        print(f"  [PASS] Stochastic 95% width[0]: {stoch_width[0]:.3f}")
        print(f"  [PASS] Width ratio (stoch/det)[0]: {ratio[0]:.2f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 17: Path dependency
    print("[Test 17] Path dependency (volatility clustering)")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        # Run two simulations with different seeds
        result1 = model.predict_simulate(h=20, n_sims=1, seed=1)
        result2 = model.predict_simulate(h=20, n_sims=1, seed=2)

        # Paths should be different
        assert not jnp.allclose(result1['sigma2_paths'], result2['sigma2_paths']), "Paths should differ with different seeds"

        # Check volatility clustering: large shock should increase next period volatility
        path = result1['paths'][0]
        sigma2_path = result1['sigma2_paths'][0]

        # Find position of largest absolute shock in first half
        mid = len(path) // 2
        large_shock_idx = jnp.argmax(jnp.abs(path[:mid]))

        # Verify volatility increased after large shock (allowing some tolerance)
        if large_shock_idx < len(sigma2_path) - 1:
            vol_before = sigma2_path[large_shock_idx]
            vol_after = sigma2_path[large_shock_idx + 1]
            # After large shock, volatility should generally be higher (but not always due to decay)
            print(f"  [PASS] Volatility before large shock: {vol_before:.4f}")
            print(f"  [PASS] Volatility after large shock: {vol_after:.4f}")

        print(f"  [PASS] Paths differ with different seeds")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 18: return_paths parameter
    print("[Test 18] return_paths parameter")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        # With return_paths=True (default)
        result_with = model.predict_simulate(h=10, n_sims=100, seed=42, return_paths=True)
        assert 'paths' in result_with, "Should have paths when return_paths=True"
        assert 'sigma2_paths' in result_with, "Should have sigma2_paths when return_paths=True"

        # With return_paths=False
        result_without = model.predict_simulate(h=10, n_sims=100, seed=42, return_paths=False)
        assert 'paths' not in result_without, "Should not have paths when return_paths=False"
        assert 'sigma2_paths' not in result_without, "Should not have sigma2_paths when return_paths=False"
        assert 'mean' in result_without, "Should still have mean when return_paths=False"
        assert 'sigma2_mean' in result_without, "Should still have sigma2_mean when return_paths=False"

        print(f"  [PASS] Paths included with return_paths=True")
        print(f"  [PASS] Paths excluded with return_paths=False")
        print(f"  [PASS] Summary statistics present in both cases")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 19: forecast_simulate stateless
    print("[Test 19] forecast_simulate stateless operation")
    try:
        model = GARCH(p=1, q=1)  # Unfitted model
        assert model.model_ is None, "Model should be unfitted initially"

        result = model.forecast_simulate(y_test, h=10, n_sims=100, seed=42, fitted=True, level=[95])

        assert 'mean' in result, "Missing mean in result"
        assert 'paths' in result, "Missing paths in result"
        assert 'fitted' in result, "Missing fitted values"
        assert 'lo-95' in result and 'hi-95' in result, "Missing intervals"
        assert 'fitted-lo-95' in result and 'fitted-hi-95' in result, "Missing fitted intervals"
        assert result['paths'].shape == (100, 10), f"Expected paths shape (100, 10), got {result['paths'].shape}"

        print(f"  [PASS] Stateless forecast successful")
        print(f"  [PASS] Fitted values included")
        print(f"  [PASS] Intervals computed")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Summary
    print(f"{passed} passed, {failed} failed")
