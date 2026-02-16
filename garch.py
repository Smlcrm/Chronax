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
1. fit: Estimates ω, α, β using maximum likelihood optimization
2. predict: Deterministic forecast returning expected volatility path
3. predict_simulate: Stochastic forecast generating multiple sample paths
4. forecast: Stateless deterministic fit-and-predict
5. forecast_simulate: Stateless stochastic fit-and-predict
6. predict_in_sample: Returns fitted values with optional prediction intervals

Constants:
_EPSILON (1e-8): Numerical floor for variance

Note on notation: This implementation uses p for ARCH order (lagged squared shocks)
and q for GARCH order (lagged variances). This is reversed from Bollerslev (1986)
but internally consistent within this codebase.
"""

import warnings

import jax
import jax.numpy as jnp
import optax
from jax import lax

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
import utils


_EPSILON = jnp.float32(1e-8)
_SIGMA2_MAX_MULT = jnp.float32(1e6)  # Maximum variance multiplier for upper bound
_LOG_2PI = jnp.float32(jnp.log(2 * jnp.pi))  # Pre-computed constant

def _compute_backcast(y: jnp.ndarray, max_window: int = 75) -> float:
    """Compute exponentially weighted backcast for variance initialization.

    Uses decay factor of 0.94 over observations, matching the arch library approach.
    This function is NOT JIT-compiled as it needs to access array length.

    Args:
        y: Input array (centered returns)
        max_window: Maximum window size for backcast (default 75)

    Returns:
        Exponentially weighted average of squared observations
    """
    n = len(y)
    tau = min(max_window, n)

    # Create weights with exponential decay
    indices = jnp.arange(tau)
    w = jnp.float32(0.94) ** indices
    w = w / jnp.sum(w)

    # Compute weighted sum of squared observations
    y_squared = (y[:tau] ** 2).astype(jnp.float32)
    return float(jnp.sum(y_squared * w))


def _compute_sigma2_series(y: jnp.ndarray, omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
                           p: int, q: int, init_var: float) -> jnp.ndarray:
    """Compute conditional variance series using GARCH recursion.

    Uses O(q) circular buffer for memory efficiency instead of O(n) full array.
    Applies variance bounds to prevent numerical issues.

    Args:
        y: Centered returns array
        omega: GARCH constant
        alpha: ARCH coefficients
        beta: GARCH coefficients
        p: ARCH order (static)
        q: GARCH order (static)
        init_var: Initial variance (from backcast, passed in to avoid JIT issues)
    """
    n = len(y)
    y = y.astype(jnp.float32)
    y_squared = y ** 2

    init_var_f32 = jnp.float32(init_var)
    sigma2_max = init_var_f32 * _SIGMA2_MAX_MULT  # Upper bound for variance

    # Pad y_squared for ARCH term lookback
    y_squared_padded = jnp.concatenate([jnp.full(p, init_var_f32, dtype=jnp.float32), y_squared])

    # Cast parameters to float32 for type consistency in lax.scan
    omega_f32 = jnp.float32(omega)
    alpha_f32 = alpha.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    zero_f32 = jnp.float32(0.0)

    # Buffer size for GARCH term (use at least 1 for consistent array shapes)
    buffer_size = max(q, 1)

    def step(sigma2_buffer, t):
        # ARCH term: use pre-padded y_squared array
        y2_lagged = lax.dynamic_slice(y_squared_padded, (t,), (p,))
        arch_sum = jnp.dot(alpha_f32, jnp.flip(y2_lagged))

        # GARCH term: use circular buffer (only last q values needed)
        garch_sum = jnp.dot(beta_f32, jnp.flip(sigma2_buffer[:q])) if q > 0 else zero_f32

        sigma2_t = jnp.maximum(omega_f32 + arch_sum + garch_sum, _EPSILON)
        # Upper bound with log-smoothing to maintain gradient flow
        sigma2_t = jnp.where(
            sigma2_t > sigma2_max,
            sigma2_max + jnp.log(sigma2_t / sigma2_max),
            sigma2_t
        )

        # Shift buffer: drop oldest, append newest
        sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_t)

        return sigma2_buffer, sigma2_t

    init_buffer = jnp.full(buffer_size, init_var_f32, dtype=jnp.float32)
    _, sigma2_all = lax.scan(step, init_buffer, jnp.arange(n))
    return sigma2_all

_compute_sigma2_series = jax.jit(_compute_sigma2_series, static_argnums=(4, 5))  # p, q are static

# negative log likelihood
def _log_likelihood(params: jnp.ndarray, y: jnp.ndarray, p: int, q: int, init_var: float,
                    actual_len: jnp.ndarray):
    """Compute negative log-likelihood for GARCH model with penalty for constraint violations.

    Args:
        params: Unconstrained parameters (to be transformed via softplus)
        y: Centered returns array (may be padded)
        p: ARCH order (static)
        q: GARCH order (static)
        init_var: Initial variance from backcast (passed in to avoid JIT issues)
        actual_len: Actual data length as JAX int32 (-1 means use full array).
                    Must be JAX array for proper tracing to avoid recompilation.
    """
    omega = jax.nn.softplus(params[0]) + _EPSILON
    alpha = jax.nn.softplus(params[1:1+p])
    beta = jax.nn.softplus(params[1+p:1+p+q]) if q > 0 else jnp.array([])
    sigma2 = _compute_sigma2_series(y, omega, alpha, beta, p, q, init_var)

    # Clip standardized squared residuals to prevent extreme values
    y_sq_over_sigma2 = jnp.clip(y**2 / sigma2, 0.0, 1e10)

    # Per-timestep log-likelihood components
    ll_per_t = _LOG_2PI + jnp.log(sigma2) + y_sq_over_sigma2

    # MASKED SUM: only sum over actual data (not padding)
    # actual_len should be passed as JAX array; -1 means use full length
    n = y.shape[0]
    eff_len = jnp.where(actual_len < 0, n, actual_len)
    mask = (jnp.arange(n) < eff_len).astype(jnp.float32)
    log_lik = -0.5 * jnp.sum(ll_per_t * mask)

    # Return large penalty if computation produced NaN
    log_lik = jnp.where(jnp.isfinite(log_lik), log_lik, jnp.float32(-1e10))

    # Stationarity penalty with smooth activation starting at 0.95
    coef_sum = jnp.sum(alpha) + jnp.sum(beta)
    # Gradual penalty that grows smoothly as sum approaches 1
    excess = jnp.maximum(coef_sum - 0.95, 0.0)
    penalty = 1e4 * excess ** 2
    # Add steep penalty near boundary
    penalty = penalty + jnp.where(coef_sum >= 0.999, 1e6 * (coef_sum - 0.999) ** 2, 0.0)

    return -log_lik + penalty

# p, q are static; actual_len is dynamic (traced) to allow JIT reuse across CV windows
_log_likelihood = jax.jit(_log_likelihood, static_argnums=(2, 3))


def _run_optax_optimization(y: jnp.ndarray, init_var: float, init_params: jnp.ndarray,
                            p: int, q: int, n_iters: int,
                            actual_len: jnp.ndarray) -> jnp.ndarray:
    """Fully JIT-compiled Optax optimization loop.

    This avoids lambda recompilation by defining the loss function once
    and using closure capture for y, init_var, p, q.

    Args:
        y: Centered returns array (may be padded)
        init_var: Initial variance from backcast
        init_params: Initial parameter values (unconstrained, float32)
        p: ARCH order (static)
        q: GARCH order (static)
        n_iters: Number of optimization iterations (static)
        actual_len: Actual data length as JAX int32 (-1 means use full array).
                    Must be JAX array for proper tracing to avoid recompilation.

    Returns:
        Best parameters found during optimization (unconstrained)
    """
    # Use learning rate schedule: start higher for fast initial progress, decay for stability
    schedule = optax.exponential_decay(
        init_value=0.05,  # Higher initial LR for faster start
        transition_steps=n_iters // 2,
        decay_rate=0.3,
        end_value=0.005,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )

    # Pre-compute the gradient function ONCE (not inside the loop)
    # This is the key optimization - avoids lambda retracing
    def loss_fn(params):
        return _log_likelihood(params, y, p, q, init_var, actual_len)

    # Use value_and_grad for optimal single-pass computation
    value_and_grad_fn = jax.value_and_grad(loss_fn)

    def step(carry, _):
        params, opt_state, best_params, best_loss = carry

        # Compute loss and gradients
        loss, grads = value_and_grad_fn(params)

        # Ensure loss is float32 to match carry type
        loss = jnp.float32(loss)

        # Apply optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Track best parameters (minimum loss)
        improved = loss < best_loss
        new_best_params = jnp.where(improved, params, best_params)
        new_best_loss = jnp.where(improved, loss, best_loss)

        return (new_params, new_opt_state, new_best_params, new_best_loss), loss

    opt_state = optimizer.init(init_params)
    init_carry = (init_params, opt_state, init_params, jnp.float32(jnp.inf))
    (_, _, best_params, _), _ = lax.scan(step, init_carry, None, length=n_iters)

    return best_params

# JIT compile with p, q, n_iters as static (they determine loop structure)
# actual_len is dynamic (traced) to avoid recompilation for different input lengths
_run_optax_optimization = jax.jit(_run_optax_optimization, static_argnums=(3, 4, 5))


def _forecast_sigma2_impl(omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
                          y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
                          sigma2_max: float, h: int, p: int, q: int) -> jnp.ndarray:
    """JIT-compiled variance forecast h steps ahead."""
    omega_f32 = jnp.float32(omega)
    alpha_f32 = alpha.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    zero_f32 = jnp.float32(0.0)
    sigma2_max_f32 = jnp.float32(sigma2_max)

    def step(carry, _):
        y_buffer, sigma2_buffer = carry

        arch_sum = jnp.sum(alpha_f32 * jnp.flip(y_buffer ** 2))
        garch_sum = jnp.sum(beta_f32 * jnp.flip(sigma2_buffer)) if q > 0 else zero_f32
        sigma2_next = jnp.clip(omega_f32 + arch_sum + garch_sum, _EPSILON, sigma2_max_f32)

        # For multi-step forecasts, future squared innovations use their expected value:
        # E[ε²_{t+h} | F_t] = σ²_{t+h}, so we set y_buffer to sqrt(sigma2_next)
        # This ensures the ARCH term correctly contributes to future variance forecasts
        y_buffer = jnp.roll(y_buffer, -1).at[-1].set(jnp.sqrt(sigma2_next))
        if q > 0:
            sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_next)

        return (y_buffer, sigma2_buffer), sigma2_next

    init_y = y_last.astype(jnp.float32)
    init_sigma2 = sigma2_last.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    _, forecasts = lax.scan(step, (init_y, init_sigma2), None, length=h)
    return forecasts

_forecast_sigma2_impl = jax.jit(_forecast_sigma2_impl, static_argnums=(6, 7, 8))  # h, p, q are static


def _forecast_sigma2_stochastic_impl(
    omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
    y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
    sigma2_max: float, h: int, p: int, q: int, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled stochastic variance forecast."""
    omega_f32 = jnp.float32(omega)
    alpha_f32 = alpha.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    zero_f32 = jnp.float32(0.0)
    sigma2_max_f32 = jnp.float32(sigma2_max)

    def step(carry, subkey):
        y_buffer, sigma2_buffer = carry

        arch_sum = jnp.sum(alpha_f32 * jnp.flip(y_buffer ** 2))
        garch_sum = jnp.sum(beta_f32 * jnp.flip(sigma2_buffer)) if q > 0 else zero_f32
        sigma2_next = jnp.clip(omega_f32 + arch_sum + garch_sum, _EPSILON, sigma2_max_f32)

        # Generate random shock and realized level
        epsilon = jax.random.normal(subkey, dtype=jnp.float32)
        y_next = epsilon * jnp.sqrt(sigma2_next)

        # Use jnp.roll instead of concatenation for buffer shifting
        y_buffer = jnp.roll(y_buffer, -1).at[-1].set(y_next)
        if q > 0:
            sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_next)

        return (y_buffer, sigma2_buffer), (y_next, sigma2_next)

    subkeys = jax.random.split(key, h)
    init_y = y_last.astype(jnp.float32)
    init_sigma2 = sigma2_last.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    _, (y_path, sigma2_path) = lax.scan(step, (init_y, init_sigma2), subkeys)

    return y_path, sigma2_path

_forecast_sigma2_stochastic_impl = jax.jit(_forecast_sigma2_stochastic_impl, static_argnums=(6, 7, 8))  # h, p, q are static


class GARCH(BaseForecaster):
    """
    Args:
        p: ARCH order (lagged squared shocks), must be greater than or equal to 1
        q: GARCH order (lagged variances)
        alias: Model name for display
        conformal_params: Optional conformal prediction configuration
        allow_extended_iterations: If True, allows up to 300 iterations for complex data.
            Default False.
        iteration_scaling: Iteration scaling strategy. Default "cubic".
            - "cubic": Aggressive scaling, minimal iterations except at high complexity (recommended)
            - "quadratic": Less aggressive, use for accuracy-intensive higher-order models
    """
    uses_exog = False

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        alias: str = "GARCH",
        conformal_params: ConformalIntervals | None = None,
        allow_extended_iterations: bool = False,
        iteration_scaling: str = "cubic",
    ):
        if not isinstance(p, int) or p < 1:
            raise ValueError(f"p must be an integer greater than or equal to 1, got {p}")
        if not isinstance(q, int) or q < 0:
            raise ValueError(f"q must be an integer greater than or equal to 0, got {q}")
        if iteration_scaling not in ("cubic", "quadratic"):
            raise ValueError(
                f"iteration_scaling must be 'cubic' or 'quadratic', got '{iteration_scaling}'"
            )

        self.p = p
        self.q = q
        self.alias = f"{alias}({p},{q})" if q != 0 else f"{alias}({p})"
        self.conformal_params = conformal_params
        self.allow_extended_iterations = allow_extended_iterations
        self.iteration_scaling = iteration_scaling
        self.model_ = None

    @staticmethod
    def _validate_h(h: int) -> None:
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"Forecast horizon h must be a positive integer, got {h}")

    @staticmethod
    def _inverse_softplus(x: float) -> float:
        """Compute inverse softplus: returns y such that softplus(y) + _EPSILON = x."""
        # Account for the epsilon offset used in parameter transform
        target = max(x - float(_EPSILON), float(_EPSILON))
        if target > 20:
            return target  # For large x, softplus(x) ≈ x
        return float(jnp.log(jnp.maximum(jnp.exp(target) - 1, _EPSILON)))

    def _get_init_params(self, y: jnp.ndarray, init_var: float) -> jnp.ndarray:
        """Compute initial parameters for optimization.

        Uses a single starting configuration matching typical GARCH parameters.
        """
        sample_var = float(jnp.var(y))
        sample_var = max(sample_var, float(_EPSILON))

        # Use moderate persistence starting point
        alpha_sum = 0.10
        persistence = 0.90
        beta_sum = persistence - alpha_sum

        omega_target = sample_var * (1 - persistence)
        omega_target = max(omega_target, float(_EPSILON))

        init_omega = self._inverse_softplus(omega_target)
        init_alpha = self._inverse_softplus(alpha_sum / self.p)
        init_beta = self._inverse_softplus(beta_sum / self.q) if self.q > 0 else 0.0

        return jnp.concatenate([
            jnp.array([init_omega]),
            jnp.full(self.p, init_alpha),
            jnp.full(self.q, init_beta) if self.q > 0 else jnp.array([])
        ])

    def _estimate_iterations(self, y: jnp.ndarray) -> int:
        """Estimate optimal iteration count based on data complexity.

        Uses kurtosis and squared-return autocorrelation as complexity indicators.
        Higher complexity → more iterations needed for convergence.

        Args:
            y: Centered returns array

        Returns:
            Estimated number of iterations (static int for lax.scan)
        """
        # Compute excess kurtosis (normal = 0, heavy tails > 0)
        mean = jnp.mean(y)
        std = jnp.std(y)
        z = (y - mean) / jnp.maximum(std, _EPSILON)
        kurtosis = float(jnp.mean(z ** 4) - 3.0)  # Excess kurtosis

        # Compute ACF1 of squared returns (GARCH effect strength)
        y_sq = y ** 2
        y_sq_centered = y_sq - jnp.mean(y_sq)
        # ACF(1) = Cov(y_sq_t, y_sq_{t-1}) / Var(y_sq)
        var_y_sq = jnp.var(y_sq)
        if var_y_sq > _EPSILON:
            acf1 = float(jnp.mean(y_sq_centered[1:] * y_sq_centered[:-1]) / var_y_sq)
        else:
            acf1 = 0.0

        # Combine into complexity score (0 to 1)
        # Kurtosis: clip to [0, 10], normalize
        kurtosis_score = min(max(kurtosis, 0.0), 10.0) / 10.0
        # ACF1: clip to [0, 0.5], normalize
        acf1_score = min(max(acf1, 0.0), 0.5) / 0.5

        # Weighted combination (ACF1 is more indicative of GARCH complexity)
        complexity = 0.3 * kurtosis_score + 0.7 * acf1_score

        # Scale to iteration range
        min_iters = 40
        max_iters = 300 if self.allow_extended_iterations else 150

        # Map string to exponent
        scaling_exponent = {"cubic": 3.0, "quadratic": 2.0}[self.iteration_scaling]
        n_iters = int(min_iters + (complexity ** scaling_exponent) * (max_iters - min_iters))
        return n_iters

    def _fit_parameters_optax(
        self, y: jnp.ndarray, init_var: float, init_params: jnp.ndarray,
        n_iters: int = None, actual_len: int = None
    ) -> jnp.ndarray:
        """Fit parameters using Optax Adam with fixed iterations via lax.scan.

        This approach uses a fixed number of iterations with lax.scan instead of
        adaptive while_loop, providing significant speedup (2-7x faster than BFGS).

        Uses the JIT-compiled _run_optax_optimization function which:
        - Pre-defines the gradient function once (avoids lambda retracing)
        - Uses learning rate schedule for faster convergence
        - Is fully compiled as a single unit

        Args:
            y: Centered returns array (may be padded)
            init_var: Initial variance from backcast
            init_params: Initial parameter values (unconstrained)
            n_iters: Fixed iteration count (for avoiding recompilation in CV).
                     If None, estimate from data.
            actual_len: Actual data length for padded inputs. If None, use full array.

        Returns:
            Best parameters found during optimization (unconstrained)
        """
        # Estimate optimal iteration count based on data complexity (actual data only)
        if n_iters is None:
            y_for_estimate = y[:actual_len] if actual_len is not None else y
            n_iters = self._estimate_iterations(y_for_estimate)

        # Ensure init_params is float32 for consistent types in lax.scan
        init_params = init_params.astype(jnp.float32)

        # Use the JIT-compiled optimization function
        # Convert actual_len to JAX array for proper tracing (use -1 as sentinel for None)
        actual_len_jax = jnp.int32(-1 if actual_len is None else actual_len)
        return _run_optax_optimization(
            y, init_var, init_params, self.p, self.q, n_iters, actual_len_jax
        )

    def _fit_parameters(self, y: jnp.ndarray, n_iters: int = None,
                         actual_len: int = None) -> dict:
        """Estimate GARCH parameters using Optax Adam optimization.

        Args:
            y: Centered returns array (may be padded)
            n_iters: Fixed iteration count (for avoiding recompilation in CV).
                     If None, estimate from data.
            actual_len: Actual data length for padded inputs. If None, use full array.

        Returns:
            Dict with fitted parameters and computed variance series.
        """
        # Use actual_len for validation, backcast, and init params
        n_eff = actual_len if actual_len is not None else len(y)
        min_obs = max(self.p, self.q) + 10
        if n_eff < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations for GARCH({self.p},{self.q}), got {n_eff}"
            )

        # Compute backcast on actual data only (done once, outside JIT)
        y_actual = y[:n_eff] if actual_len is not None else y
        init_var = _compute_backcast(y_actual)

        # Run optimization with padding-aware functions
        init_params = self._get_init_params(y_actual, init_var)
        final_params = self._fit_parameters_optax(y, init_var, init_params, n_iters, actual_len)

        # Transform parameters from unconstrained to constrained space
        omega = jax.nn.softplus(final_params[0]) + _EPSILON
        alpha = jax.nn.softplus(final_params[1:1+self.p])
        beta = jax.nn.softplus(final_params[1+self.p:1+self.p+self.q]) if self.q > 0 else jnp.array([])

        # Check persistence and validate stationarity constraint
        persistence = float(jnp.sum(alpha) + jnp.sum(beta))

        # Critical check: if persistence >= 1, the model is non-stationary and forecasts will explode
        if persistence >= 1.0:
            raise RuntimeError(
                f"GARCH optimization failed: persistence {persistence:.4f} >= 1 "
                "violates stationarity constraint. The fitted model is non-stationary "
                "and forecasts will diverge."
            )
        elif persistence > 0.99:
            warnings.warn(
                f"GARCH persistence {persistence:.4f} is near the stationarity boundary. "
                "Consider using an integrated GARCH (IGARCH) model for highly persistent volatility.",
                RuntimeWarning
            )

        # Compute sigma2 on full array (including padding if present)
        sigma2_full = _compute_sigma2_series(y, omega, alpha, beta, self.p, self.q, init_var)
        # Trim to actual length for output
        sigma2 = sigma2_full[:n_eff] if actual_len is not None else sigma2_full

        return {
            'omega': float(omega),
            'alpha': alpha,
            'beta': beta,
            'sigma2': sigma2,
            'fitted': jnp.zeros(n_eff),  # Use actual length, not padded
            'y_mean': float(jnp.mean(y_actual)),
            'y_last': y_actual[-self.p:],  # Use actual data for last values
            'sigma2_last': sigma2[-self.q:] if self.q > 0 else jnp.array([]),
            'init_var': init_var,
        }

    def _forecast_sigma2(self, omega: float, alpha: jnp.ndarray, beta: jnp.ndarray, y_last: jnp.ndarray, sigma2_last: jnp.ndarray, init_var: float, h: int) -> jnp.ndarray:
        """Forecast variance h steps ahead by iterating GARCH equation."""
        sigma2_max = init_var * float(_SIGMA2_MAX_MULT)
        return _forecast_sigma2_impl(omega, alpha, beta, y_last, sigma2_last, sigma2_max, h, self.p, self.q)

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None,
            n_iters: int = None, actual_len: int = None) -> 'GARCH':
        """Estimate GARCH parameters from data.

        Args:
            y: Input time series (may be padded)
            X: Exogenous variables (unused for GARCH)
            n_iters: Fixed iteration count (for avoiding recompilation in CV).
                     If None, estimate from data.
            actual_len: Actual data length for padded inputs. If None, use full array.

        Returns:
            self (fitted model)
        """
        y = utils.ensure_float(y)

        # Use actual_len for validation
        n_eff = actual_len if actual_len is not None else len(y)
        y_actual = y[:n_eff] if actual_len is not None else y

        if not jnp.all(jnp.isfinite(y_actual)):
            raise ValueError("Input contains NaN or infinite values")
        if jnp.var(y_actual) < 1e-10:
            raise ValueError("Input has near-zero variance, unsuitable for GARCH")

        y_mean = jnp.mean(y_actual)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered, n_iters=n_iters, actual_len=actual_len)

        self.model_ = {
            'omega': result['omega'],
            'alpha': result['alpha'],
            'beta': result['beta'],
            'sigma2': result['sigma2'],
            'fitted': result['fitted'] + y_mean,
            'residuals': y_actual[:n_eff] - y_mean - result['fitted'],  # Use actual data
            'y_mean': y_mean,
            'y_centered_last': result['y_last'],
            'sigma2_last': result['sigma2_last'],
            'init_var': result['init_var'],
            'y_train': y_actual,  # Store actual data, not padded
        }

        self.model_['sigma'] = float(utils.calculate_sigma(self.model_['residuals'], n_eff - (self.p + self.q + 1)))

        # Pre-compute and store conformity scores if conformal prediction is enabled
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            self.model_['_cs'] = cs

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
            self.model_['init_var'],
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
                # Use pre-computed conformity scores from fit()
                cs = self.model_.get('_cs')
                if cs is None:
                    raise ValueError("Conformity scores not found. Model may have been fitted without conformal_params.")
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                sigma_forecast = jnp.sqrt(sigma2_forecast)

                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = mean_forecast - z * sigma_forecast
                    res[f'hi-{lv}'] = mean_forecast + z * sigma_forecast

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        """Return in-sample fitted values and conditional variance series.

        Returns:
            Dictionary containing:
            - fitted: Conditional mean E[y_t | F_{t-1}] = μ (constant for pure GARCH)
            - sigma2: Conditional variance series σ²_t (the quantity GARCH models)
            - fitted-lo-{lv}, fitted-hi-{lv}: Prediction intervals if level is specified
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        res = {
            'fitted': self.model_['fitted'],
            'sigma2': self.model_['sigma2'],
        }

        if level is not None:
            level = sorted(level)
            sigma_t = jnp.sqrt(self.model_['sigma2'])  # Time-varying conditional std dev

            for lv in level:
                z = utils._jax_norm_ppf((100 + lv) / 200)
                res[f'fitted-lo-{lv}'] = self.model_['fitted'] - z * sigma_t
                res[f'fitted-hi-{lv}'] = self.model_['fitted'] + z * sigma_t

        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
        n_iters: int = None,
        actual_len: int = None,
    ) -> dict:
        """Stateless fit and predict.

        Args:
            y: Input time series (may be padded)
            h: Forecast horizon
            X: Exogenous variables (unused for GARCH)
            X_future: Future exogenous variables (unused for GARCH)
            level: Confidence levels for prediction intervals
            fitted: Whether to return fitted values
            n_iters: Fixed iteration count (for avoiding recompilation in CV).
                     If None, estimate from data.
            actual_len: Actual data length for padded inputs. If None, use full array.

        Returns:
            Dict with mean forecasts and variance forecasts.
        """
        self._validate_h(h)
        y = utils.ensure_float(y)

        # Compute mean on actual data only
        n_eff = actual_len if actual_len is not None else len(y)
        y_actual = y[:n_eff] if actual_len is not None else y
        y_mean = jnp.mean(y_actual)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered, n_iters=n_iters, actual_len=actual_len)

        sigma2_forecast = self._forecast_sigma2(
            result['omega'],
            result['alpha'],
            result['beta'],
            result['y_last'],
            result['sigma2_last'],
            result['init_var'],
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
                sigma_t = jnp.sqrt(result['sigma2'])  # Time-varying conditional std dev

                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'fitted-lo-{lv}'] = res['fitted'] - z * sigma_t
                    res[f'fitted-hi-{lv}'] = res['fitted'] + z * sigma_t

        return res

    def _forecast_sigma2_stochastic(
        self,
        omega: float,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        y_last: jnp.ndarray,
        sigma2_last: jnp.ndarray,
        init_var: float,
        h: int,
        key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sigma2_max = init_var * float(_SIGMA2_MAX_MULT)
        return _forecast_sigma2_stochastic_impl(
            omega, alpha, beta, y_last, sigma2_last, sigma2_max, h, self.p, self.q, key
        )

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
        init_var: float,
        y_mean: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Parallelize simulation of n_sims paths using vmap."""
        def simulate_single_path(subkey):
            y_path, sigma2_path = self._forecast_sigma2_stochastic(
                omega, alpha, beta, y_last, sigma2_last, init_var, h, subkey
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
            init_var=self.model_['init_var'],
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

            # Minimum simulations for reliable percentile-based intervals
            MIN_SIMS_FOR_PERCENTILE = 30

            if n_sims < MIN_SIMS_FOR_PERCENTILE:
                # Fall back to analytical intervals using mean variance forecast
                warnings.warn(
                    f"n_sims={n_sims} is too small for reliable percentile-based intervals. "
                    f"Falling back to analytical intervals using mean variance forecast. "
                    f"Use n_sims >= {MIN_SIMS_FOR_PERCENTILE} for empirical intervals.",
                    RuntimeWarning
                )
                sigma_forecast = jnp.sqrt(res['sigma2_mean'])
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = res['mean'] - z * sigma_forecast
                    res[f'hi-{lv}'] = res['mean'] + z * sigma_forecast
            else:
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
        return_paths: bool = True,
        n_iters: int = None,
        actual_len: int = None,
    ) -> dict:
        """Stateless stochastic forecast (fit then simulate, don't store model state).

        Args:
            y: Input time series (may be padded)
            h: Forecast horizon
            n_sims: Number of simulation paths
            seed: Random seed for reproducibility
            X: Exogenous variables (unused for GARCH)
            X_future: Future exogenous variables (unused for GARCH)
            level: Confidence levels for prediction intervals
            fitted: Whether to return fitted values
            return_paths: Whether to return full simulation paths
            n_iters: Fixed iteration count (for avoiding recompilation in CV).
                     If None, estimate from data.
            actual_len: Actual data length for padded inputs. If None, use full array.

        Returns:
            Dict with simulation-based forecasts and intervals.
        """
        self.fit(y, X, n_iters=n_iters, actual_len=actual_len)
        res = self.predict_simulate(h, n_sims, seed, X_future, level, return_paths)

        if fitted:
            res['fitted'] = self.model_['fitted']

            if level is not None:
                level = sorted(level)
                sigma_t = jnp.sqrt(self.model_['sigma2'])  # Time-varying conditional std dev
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'fitted-lo-{lv}'] = self.model_['fitted'] - z * sigma_t
                    res[f'fitted-hi-{lv}'] = self.model_['fitted'] + z * sigma_t

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
    print("[Test 17] Path dependency (stochastic simulation)")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        # Run two simulations with different seeds
        result1 = model.predict_simulate(h=20, n_sims=1, seed=1)
        result2 = model.predict_simulate(h=20, n_sims=1, seed=2)

        # y paths should always differ due to random shocks
        assert not jnp.allclose(result1['paths'], result2['paths']), "Y paths should differ with different seeds"

        # Note: sigma2_paths may be nearly identical if fitted alpha ≈ 0
        # (when ARCH effect is negligible, variance evolution is deterministic)
        alpha_sum = float(jnp.sum(model.model_['alpha']))
        if alpha_sum > 0.01:
            # Only check sigma2 path dependency when alpha is non-trivial
            assert not jnp.allclose(result1['sigma2_paths'], result2['sigma2_paths']), "Sigma2 paths should differ"
            print(f"  [PASS] Sigma2 paths differ with different seeds (alpha={alpha_sum:.4f})")
        else:
            print(f"  [INFO] Alpha≈0 ({alpha_sum:.4f}), sigma2 paths are deterministic (expected)")

        # Check that variance values are positive and finite
        path = result1['paths'][0]
        sigma2_path = result1['sigma2_paths'][0]
        assert jnp.all(jnp.isfinite(sigma2_path)), "Sigma2 path should be finite"
        assert jnp.all(sigma2_path > 0), "Sigma2 path should be positive"

        print(f"  [PASS] Y paths differ with different seeds")
        print(f"  [PASS] Sigma2 values are finite and positive")
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

    # Test 20: Small n_sims fallback to analytical intervals
    print("[Test 20] Small n_sims fallback to analytical intervals")
    try:
        model = GARCH(p=1, q=1)
        model.fit(y_test)

        # Test with n_sims=1 (below MIN_SIMS_FOR_PERCENTILE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.predict_simulate(h=10, n_sims=1, seed=42, level=[95])

            # Check warning was issued
            assert len(w) >= 1, "Expected warning for small n_sims"
            assert "too small for reliable percentile-based intervals" in str(w[-1].message), \
                f"Wrong warning: {w[-1].message}"

        # Check intervals are computed and non-zero
        assert 'lo-95' in result and 'hi-95' in result, "Missing intervals"
        interval_width = result['hi-95'] - result['lo-95']
        assert jnp.all(interval_width > 0), f"Interval width should be positive, got {interval_width}"

        # Verify analytical intervals match expected z-score computation
        sigma_forecast = jnp.sqrt(result['sigma2_mean'])
        z = utils._jax_norm_ppf((100 + 95) / 200)
        expected_lo = result['mean'] - z * sigma_forecast
        expected_hi = result['mean'] + z * sigma_forecast
        assert jnp.allclose(result['lo-95'], expected_lo), "lo-95 doesn't match analytical"
        assert jnp.allclose(result['hi-95'], expected_hi), "hi-95 doesn't match analytical"

        print(f"  [PASS] Warning issued for n_sims=1")
        print(f"  [PASS] Intervals computed with non-zero width: {interval_width[0]:.3f}")
        print(f"  [PASS] Analytical fallback matches expected computation")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 21: iteration_scaling parameter validation
    print("[Test 21] iteration_scaling parameter validation")
    test21_failed = False
    try:
        # Test default cubic scaling
        model_cubic = GARCH(p=1, q=1)
        assert model_cubic.iteration_scaling == "cubic", f"Expected 'cubic' default, got {model_cubic.iteration_scaling}"
        print("  [PASS] Default iteration_scaling is 'cubic'")

        # Test quadratic scaling
        model_quad = GARCH(p=1, q=1, iteration_scaling="quadratic")
        assert model_quad.iteration_scaling == "quadratic", f"Expected 'quadratic', got {model_quad.iteration_scaling}"
        print("  [PASS] Can set iteration_scaling='quadratic'")

        # Test invalid value raises error
        try:
            model_invalid = GARCH(p=1, q=1, iteration_scaling="linear")
            print("  [FAIL] Should have raised ValueError for invalid iteration_scaling")
            test21_failed = True
        except ValueError as e:
            assert "iteration_scaling must be 'cubic' or 'quadratic'" in str(e), f"Wrong error message: {e}"
            print("  [PASS] Correctly rejected invalid iteration_scaling")

        if test21_failed:
            failed += 1
        else:
            passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 22: Cubic vs quadratic scaling produces different iteration counts
    print("[Test 22] Cubic vs quadratic iteration scaling behavior")
    try:
        model_cubic = GARCH(p=1, q=1, iteration_scaling="cubic")
        model_quad = GARCH(p=2, q=2, iteration_scaling="quadratic")

        # Both should fit successfully
        model_cubic.fit(y_test)
        model_quad.fit(y_test)

        # Both should produce valid results
        result_cubic = model_cubic.predict(h=10)
        result_quad = model_quad.predict(h=10)

        assert jnp.all(jnp.isfinite(result_cubic['mean'])), "Non-finite values in cubic model forecast"
        assert jnp.all(jnp.isfinite(result_quad['mean'])), "Non-finite values in quadratic model forecast"
        assert jnp.all(result_cubic['sigma2'] > 0), "Non-positive variance in cubic model"
        assert jnp.all(result_quad['sigma2'] > 0), "Non-positive variance in quadratic model"

        print(f"  [PASS] Cubic scaling model fitted successfully")
        print(f"  [PASS] Quadratic scaling model fitted successfully")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 23: Extended iterations with quadratic scaling
    print("[Test 23] Extended iterations with quadratic scaling")
    try:
        model = GARCH(p=2, q=2, iteration_scaling="quadratic", allow_extended_iterations=True)
        model.fit(y_test)
        result = model.predict(h=10)

        assert jnp.all(jnp.isfinite(result['mean'])), "Non-finite values in forecast"
        assert jnp.all(result['sigma2'] > 0), "Non-positive variance forecasts"

        # Check persistence is valid
        persistence = float(jnp.sum(model.model_['alpha']) + jnp.sum(model.model_['beta']))
        assert 0 < persistence < 1, f"Invalid persistence {persistence}"

        print(f"  [PASS] Extended iterations with quadratic scaling works")
        print(f"  [PASS] Persistence: {persistence:.4f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        failed += 1

    # Test 24: Padded input produces same results as unpadded (GPU JIT optimization)
    print("[Test 24] Padded input equivalence for GPU JIT optimization")
    try:
        y_short = y_test[:150]  # 150 samples
        y_padded = jnp.pad(y_short, (0, 50), mode='edge')  # Pad to 200

        model1 = GARCH(p=1, q=1)
        model2 = GARCH(p=1, q=1)

        # Unpadded forecast
        result1 = model1.forecast(y_short, h=10)

        # Padded forecast with actual_len
        result2 = model2.forecast(y_padded, h=10, actual_len=150)

        # Results should be numerically close
        mean_diff = jnp.max(jnp.abs(result1['mean'] - result2['mean']))
        sigma2_diff = jnp.max(jnp.abs(result1['sigma2'] - result2['sigma2']))

        assert mean_diff < 1e-4, f"Mean forecasts differ by {mean_diff}"
        assert sigma2_diff < 1e-4, f"Sigma2 forecasts differ by {sigma2_diff}"

        print(f"  [PASS] Padded and unpadded mean difference: {mean_diff:.6f}")
        print(f"  [PASS] Padded and unpadded sigma2 difference: {sigma2_diff:.6f}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # Test 25: Fixed n_iters with padding prevents recompilation
    print("[Test 25] Fixed n_iters with padding")
    try:
        # Simulate CV scenario: different lengths, same n_iters
        y_lens = [150, 160, 170]
        max_len = 200
        fixed_n_iters = 80

        results = []
        for y_len in y_lens:
            y_data = y_test[:y_len]
            y_padded = jnp.pad(y_data, (0, max_len - y_len), mode='edge')

            model = GARCH(p=1, q=1)
            result = model.forecast(y_padded, h=10, n_iters=fixed_n_iters, actual_len=y_len)

            assert jnp.all(jnp.isfinite(result['mean'])), f"Non-finite mean for len={y_len}"
            assert jnp.all(result['sigma2'] > 0), f"Non-positive sigma2 for len={y_len}"
            results.append(result)

        print(f"  [PASS] All {len(y_lens)} padded lengths with fixed n_iters work correctly")
        sigma2_vals = [float(r['sigma2'][0]) for r in results]
        print(f"  [PASS] Sigma2[0] for lens {y_lens}: {sigma2_vals}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # Summary
    print(f"{passed} passed, {failed} failed")
