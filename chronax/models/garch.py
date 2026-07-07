"""
GARCH Model — JAX-accelerated implementation.

Models time series with non-constant volatility where conditional variance
depends on past squared errors and past conditional variances. Supports both
deterministic forecasting (analytical intervals) and stochastic forecasting
(Monte Carlo simulation paths).

Features:
- GARCH(p,q) and pure ARCH(p) models
- Primary: jaxopt.LBFGSB with box-constrained direct parameterization
- Fallback: Optax ADAM + L-BFGS with softplus reparameterization (lax.cond)
- Multi-start optimization: every candidate (ACF-based, uniform,
  high/low-persistence) is fit via a vmapped LBFGSB run, winner by jnp.argmin
- vmap-native fit/forecast: no host casts or Python control flow on traced
  values, so conformity_scores' vmapped CV path traces end-to-end
- O(log n) parallel variance recursion for q<=1 via associative scan
- Deterministic and stochastic (Monte Carlo) forecasting
- Native and conformal prediction intervals
- Padded-input support for GPU JIT reuse in cross-validation

Instance Attributes:
1. p: int - ARCH order (lagged squared shocks)
2. q: int - GARCH order (lagged variances)
3. alias: str - Display name for the model
4. conformal_params: ConformalIntervals | None - Conformal prediction config
5. allow_extended_iterations: bool - Use the extended optax-fallback budget (120 vs 80)
6. iteration_scaling: str - Deprecated/inert (formerly scaled a data-adaptive iteration count)
7. model_: dict - Fitted model state (created after fit())
   - omega, alpha, beta: Estimated GARCH parameters
   - sigma2: Conditional variance series
   - fitted: In-sample fitted values (conditional mean)
   - residuals: Fit residuals
   - y_mean: Training series mean
   - y_centered_last: Last p centered observations
   - sigma2_last: Last q conditional variances
   - init_var: Backcast initial variance

Methods:
1. __init__() - Initialize with ARCH/GARCH orders and optimization settings
2. fit(y, X=None) - Fit model via LBFGSB optimization
3. predict(h, X=None, level=None, simulate=False, ...) - h-step forecast (deterministic or Monte Carlo)
4. predict_in_sample(level=None) - Return fitted values with optional intervals
5. forecast(y, h, ..., simulate=False, ...) - Stateless fit-and-predict (deterministic or Monte Carlo)

Implementation Notes:
- Notation: p = ARCH order, q = GARCH order (reversed from Bollerslev 1986)
- Primary path: direct positive params with LBFGSB box bounds (omega: [1e-7, 1e4], alpha/beta: [1e-7, 0.9999])
- Fallback path: softplus(unconstrained) + epsilon with stationarity penalty
- Post-fit stationarity clamping at 0.999 with omega recomputation
- Variance recursion uses O(q) circular buffer for q>1, associative scan for q<=1
- Backcast initialization with exponential decay (tau=0.94, max window=75)
"""

import jax
import jax.numpy as jnp
import optax
from jax import lax

import jaxopt

from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals
from chronax import utils

__all__ = ['GARCH']


# =============================================================================
# Constants
# =============================================================================

_EPSILON = jnp.float32(1e-8)
_SIGMA2_MAX_MULT = jnp.float32(1e6)
_LOG_2PI = jnp.float32(jnp.log(2 * jnp.pi))
_LBFGS_STEPS = 40
_LBFGS_MEMORY = 15
_LBFGS_LS_STEPS = 20
_LBFGSB_MAXITER = 50
_LBFGSB_HISTORY = 15
_LBFGSB_TOL = 1e-5
_LBFGSB_MAXLS = 30

# Optax-fallback iteration budgets (static, config-derived — a data-dependent
# count cannot feed a jit static arg; CLAUDE.md §10). The former adaptive
# heuristic mapped data complexity into 20–80 (120 extended); both scan phases
# track best-params monotonically, so running the full budget converges
# equal-or-better than any adaptive cutoff.
_MAX_ITER = 80
_MAX_ITER_EXTENDED = 120

# =============================================================================
# Core Math — Module-level JIT'd functions
# =============================================================================

def _compute_backcast(y: jnp.ndarray, max_window: int = 75) -> jnp.ndarray:
    """Exponentially weighted backcast for variance initialization.

    Not JIT-compiled itself, but traces under jit/vmap: the window length
    derives from the array's static shape, never from its values.

    Parameters
    ----------
    y : jnp.ndarray
        Centered returns.
    max_window : int, default 75
        Maximum lookback window.

    Returns
    -------
    jnp.ndarray
        Scalar exponentially weighted average of squared observations.
    """
    tau = min(max_window, len(y))
    w = jnp.float32(0.94) ** jnp.arange(tau)
    w = w / jnp.sum(w)
    y_squared = (y[:tau] ** 2).astype(jnp.float32)
    return jnp.sum(y_squared * w)


def _garch_associative_op(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Associative operator for GARCH(p,1) linear recurrence composition.

    For sigma2_t = a_t + b * sigma2_{t-1}, composition of two affine maps:
    (a_l, b_l) composed with (a_r, b_r) = (a_r + b_r * a_l, b_r * b_l)
    """
    a_l, b_l = left
    a_r, b_r = right
    return (a_r + b_r * a_l, b_r * b_l)


def _compute_arch_terms(
    y_squared: jnp.ndarray, omega: jnp.ndarray, alpha: jnp.ndarray,
    init_var: jnp.ndarray, p: int,
) -> jnp.ndarray:
    """Vectorized ARCH summation: omega + sum_i(alpha_i * y^2_{t-i}).

    Pre-computes all ARCH terms in parallel via lag indexing, avoiding
    sequential dynamic_slice calls. Used by q<=1 paths (pure ARCH and
    associative scan); the q>1 sequential path uses inline dynamic_slice.
    """
    n = y_squared.shape[0]
    y2_padded = jnp.concatenate([jnp.full(p, init_var, dtype=jnp.float32), y_squared])
    lags = p - 1 - jnp.arange(p)
    indices = lags[:, None] + jnp.arange(n)
    y2_lags = y2_padded[indices]  # (p, n)
    return omega + jnp.dot(alpha, y2_lags)


def _compute_sigma2_parallel_q1(
    arch_terms: jnp.ndarray, beta_scalar: jnp.ndarray, init_var: jnp.ndarray,
) -> jnp.ndarray:
    """O(log n) parallel variance recursion for q=1 via associative scan.

    Exploits the linear recurrence sigma2_t = a_t + beta * sigma2_{t-1}
    by composing affine maps in parallel using JAX's associative_scan.
    """
    n = arch_terms.shape[0]
    b = jnp.full(n, beta_scalar, dtype=jnp.float32)
    A, B = jax.lax.associative_scan(_garch_associative_op, (arch_terms, b))
    sigma2 = A + B * init_var
    return jnp.maximum(sigma2, _EPSILON)


def _compute_sigma2_series(y: jnp.ndarray, omega: float, alpha: jnp.ndarray,
                           beta: jnp.ndarray, p: int, q: int,
                           init_var: float) -> jnp.ndarray:
    """Compute conditional variance series via GARCH recursion.

    Dispatches to O(log n) parallel path for q<=1 or O(n) sequential for q>1.

    Parameters
    ----------
    y : jnp.ndarray
        Centered returns.
    omega : float
        GARCH intercept.
    alpha : jnp.ndarray
        ARCH coefficients of length p.
    beta : jnp.ndarray
        GARCH coefficients of length q.
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).
    init_var : float
        Backcast initial variance.

    Returns
    -------
    jnp.ndarray
        Conditional variance series of length n.
    """
    n = len(y)
    y = y.astype(jnp.float32)
    y_squared = y ** 2

    init_var_f32 = jnp.float32(init_var)
    omega_f32 = jnp.float32(omega)
    alpha_f32 = alpha.astype(jnp.float32)

    if q == 0:
        # Pure ARCH: no recursion, sigma2 = arch_terms
        arch_terms = _compute_arch_terms(y_squared, omega_f32, alpha_f32, init_var_f32, p)
        return jnp.maximum(arch_terms, _EPSILON)
    elif q == 1:
        # GARCH(p,1): O(log n) parallel via associative scan
        arch_terms = _compute_arch_terms(y_squared, omega_f32, alpha_f32, init_var_f32, p)
        return _compute_sigma2_parallel_q1(arch_terms, beta[0].astype(jnp.float32), init_var_f32)
    else:
        # q > 1: sequential scan with circular buffer and log-smoothed bound.
        # dynamic_slice inside the scan fuses better in XLA than pre-computing.
        sigma2_max = init_var_f32 * _SIGMA2_MAX_MULT
        y_squared_padded = jnp.concatenate([jnp.full(p, init_var_f32, dtype=jnp.float32), y_squared])
        beta_f32 = beta.astype(jnp.float32)
        buffer_size = max(q, 1)

        def step(sigma2_buffer, t):
            y2_lagged = lax.dynamic_slice(y_squared_padded, (t,), (p,))
            arch_sum = jnp.dot(alpha_f32, jnp.flip(y2_lagged))
            garch_sum = jnp.dot(beta_f32, jnp.flip(sigma2_buffer[:q]))
            sigma2_t = jnp.maximum(omega_f32 + arch_sum + garch_sum, _EPSILON)
            sigma2_t = jnp.where(
                sigma2_t > sigma2_max,
                sigma2_max + jnp.log(sigma2_t / sigma2_max),
                sigma2_t
            )
            sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_t)
            return sigma2_buffer, sigma2_t

        init_buffer = jnp.full(buffer_size, init_var_f32, dtype=jnp.float32)
        _, sigma2_all = lax.scan(step, init_buffer, jnp.arange(n))
        return sigma2_all

_compute_sigma2_series = jax.jit(_compute_sigma2_series, static_argnums=(4, 5))


def _log_likelihood(params: jnp.ndarray, y: jnp.ndarray, p: int, q: int,
                    init_var: float, actual_len: jnp.ndarray,
                    sample_var: jnp.ndarray) -> jnp.ndarray:
    """Penalized negative log-likelihood for GARCH.

    Parameters
    ----------
    params : jnp.ndarray
        Unconstrained parameters (transformed via softplus).
    y : jnp.ndarray
        Centered returns (may be padded).
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).
    init_var : float
        Backcast initial variance.
    actual_len : jnp.ndarray
        Actual data length as JAX int32; -1 means use full array.
    sample_var : jnp.ndarray
        Sample variance for variance targeting; -1.0 disables VT.

    Returns
    -------
    jnp.ndarray
        Scalar negative log-likelihood plus stationarity penalty.
    """
    params = jnp.asarray(params, dtype=jnp.float32)
    alpha = jax.nn.softplus(params[1:1+p])
    beta = jax.nn.softplus(params[1+p:1+p+q]) if q > 0 else jnp.array([])

    # Omega: variance targeting or free parameter
    coef_sum = jnp.sum(alpha) + jnp.sum(beta)
    omega_free = jax.nn.softplus(params[0]) + _EPSILON
    omega_vt = sample_var * jnp.maximum(1.0 - coef_sum, _EPSILON)
    omega = jnp.where(sample_var >= 0, omega_vt, omega_free)

    sigma2 = _compute_sigma2_series(y, omega, alpha, beta, p, q, init_var)

    y_sq_over_sigma2 = y**2 / jnp.maximum(sigma2, _EPSILON)
    ll_per_t = _LOG_2PI + jnp.log(sigma2) + y_sq_over_sigma2

    # Masked mean: normalize by effective length for scale-invariant gradients
    n = y.shape[0]
    eff_len = jnp.where(actual_len < 0, n, actual_len)
    mask = (jnp.arange(n) < eff_len).astype(jnp.float32)
    eff_len_f32 = jnp.maximum(eff_len.astype(jnp.float32), 1.0)
    nll = 0.5 * jnp.sum(ll_per_t * mask) / eff_len_f32
    nll = jnp.where(jnp.isfinite(nll), nll, jnp.float32(1e10))

    # Stationarity penalty: smooth activation at 0.95, steep near 1.0
    excess = jnp.maximum(coef_sum - 0.95, 0.0)
    penalty = 1e5 * excess ** 2
    penalty = penalty + jnp.where(coef_sum >= 0.999, 1e6 * (coef_sum - 0.999) ** 2, 0.0)

    return nll + penalty

_log_likelihood = jax.jit(_log_likelihood, static_argnums=(2, 3))


def _log_likelihood_direct(params: jnp.ndarray, y: jnp.ndarray, p: int,
                           q: int, init_var: float,
                           actual_len: jnp.ndarray) -> jnp.ndarray:
    """NLL for direct parameterization (no softplus, no penalty).

    Parameters are already positive via LBFGSB box bounds.
    Starts NLL computation from index max(p,q) to avoid initialization
    transient, matching statsforecast's approach.
    """
    params = jnp.asarray(params, dtype=jnp.float32)
    omega = jnp.maximum(params[0], _EPSILON)
    alpha = jnp.maximum(params[1:1+p], _EPSILON)
    beta = jnp.maximum(params[1+p:1+p+q], _EPSILON) if q > 0 else jnp.array([], dtype=jnp.float32)

    sigma2 = _compute_sigma2_series(y, omega, alpha, beta, p, q, init_var)

    y_sq_over_sigma2 = y**2 / jnp.maximum(sigma2, _EPSILON)
    ll_per_t = _LOG_2PI + jnp.log(jnp.maximum(sigma2, _EPSILON)) + y_sq_over_sigma2

    n = y.shape[0]
    eff_len = jnp.where(actual_len < 0, n, actual_len)
    start_idx = max(p, q)
    mask = ((jnp.arange(n) >= start_idx) & (jnp.arange(n) < eff_len)).astype(jnp.float32)
    count = jnp.maximum(jnp.sum(mask), 1.0)
    nll = 0.5 * jnp.sum(ll_per_t * mask) / count
    nll = jnp.where(jnp.isfinite(nll), nll, jnp.float32(1e10))
    return nll

_log_likelihood_direct = jax.jit(_log_likelihood_direct, static_argnums=(2, 3))


def _run_lbfgsb_optimization(y: jnp.ndarray, init_var: float,
                              init_params_direct: jnp.ndarray,
                              p: int, q: int,
                              actual_len: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optimize GARCH via jaxopt.LBFGSB with box constraints.

    Parameters
    ----------
    y : jnp.ndarray
        Centered returns (may be padded).
    init_var : float
        Backcast initial variance.
    init_params_direct : jnp.ndarray
        Starting parameters in constrained (positive) space.
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).
    actual_len : jnp.ndarray
        Actual data length as JAX int32; -1 means use full array.

    Returns
    -------
    tuple
        (best_params, best_loss) — optimized params and final NLL.
    """
    def loss_fn(params):
        return _log_likelihood_direct(params, y, p, q, init_var, actual_len)

    n_params = 1 + p + q
    # Bounds follow the candidate dtype: mixed f32 params / f64 bounds would
    # promote inside the solver's while_loop carry under global x64.
    dtype = init_params_direct.dtype
    lower = jnp.full(n_params, 1e-7, dtype=dtype)
    upper_alpha_beta = jnp.full(p + q, 0.9999, dtype=dtype)
    upper_omega = jnp.array([1e4], dtype=dtype)
    upper = jnp.concatenate([upper_omega, upper_alpha_beta])
    bounds = (lower, upper)

    solver = jaxopt.LBFGSB(
        fun=loss_fn, maxiter=_LBFGSB_MAXITER, tol=_LBFGSB_TOL,
        history_size=_LBFGSB_HISTORY, maxls=_LBFGSB_MAXLS,
        jit=True, implicit_diff=False,
    )
    result = solver.run(init_params_direct, bounds=bounds)
    return result.params, loss_fn(result.params)

_run_lbfgsb_optimization = jax.jit(_run_lbfgsb_optimization, static_argnums=(3, 4))


def _run_optax_optimization(y: jnp.ndarray, init_var: float,
                            init_params: jnp.ndarray, p: int, q: int,
                            n_iters: int,
                            actual_len: jnp.ndarray,
                            sample_var: jnp.ndarray) -> jnp.ndarray:
    """Two-phase optimization: ADAM warm-up then L-BFGS refinement.

    Parameters
    ----------
    y : jnp.ndarray
        Centered returns (may be padded).
    init_var : float
        Backcast initial variance.
    init_params : jnp.ndarray
        Unconstrained starting parameters (float32).
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).
    n_iters : int
        ADAM iteration count (static).
    actual_len : jnp.ndarray
        Actual data length as JAX int32; -1 means use full array.
    sample_var : jnp.ndarray
        Sample variance for variance targeting; -1.0 disables VT.

    Returns
    -------
    jnp.ndarray
        Best parameters found (unconstrained).
    """
    def loss_fn(params):
        params = jnp.asarray(params, dtype=jnp.float32)
        return _log_likelihood(params, y, p, q, init_var, actual_len, sample_var)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    # Phase 1: Short ADAM warm-up (1/3 of budget), then L-BFGS refines
    adam_steps = max(n_iters // 3, 10)
    schedule = optax.cosine_decay_schedule(
        init_value=0.10, decay_steps=adam_steps, alpha=0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )

    def adam_step(carry, _):
        params, opt_state, best_params, best_loss = carry
        loss, grads = value_and_grad_fn(params)
        loss = jnp.float32(loss)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        improved = loss < best_loss
        new_best_params = jnp.where(improved, params, best_params)
        new_best_loss = jnp.where(improved, loss, best_loss)
        return (new_params, new_opt_state, new_best_params, new_best_loss), None

    opt_state = optimizer.init(init_params)
    adam_carry = (init_params, opt_state, init_params, jnp.float32(jnp.inf))
    (_, _, adam_best, adam_best_loss), _ = lax.scan(
        adam_step, adam_carry, None, length=adam_steps)

    # Phase 2: L-BFGS refinement
    lbfgs_solver = optax.lbfgs(
        memory_size=_LBFGS_MEMORY,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=_LBFGS_LS_STEPS, initial_guess_strategy="one"))
    lbfgs_state = lbfgs_solver.init(adam_best)

    def lbfgs_step(carry, _):
        params, state, best_params, best_loss = carry
        loss, grads = value_and_grad_fn(params)
        loss = jnp.float32(loss)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = lbfgs_solver.update(
            grads, state, params, value=loss, grad=grads, value_fn=loss_fn)
        new_params = optax.apply_updates(params, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        new_best_params = jnp.where(improved, params, best_params)
        new_best_loss = jnp.where(improved, loss, best_loss)
        return (new_params, new_state, new_best_params, new_best_loss), None

    lbfgs_carry = (adam_best, lbfgs_state, adam_best, jnp.float32(adam_best_loss))
    (_, _, lbfgs_best, lbfgs_best_loss), _ = lax.scan(
        lbfgs_step, lbfgs_carry, None, length=_LBFGS_STEPS)

    # Pick best between ADAM and L-BFGS
    use_lbfgs = jnp.isfinite(lbfgs_best_loss) & (lbfgs_best_loss < adam_best_loss)
    return jnp.where(use_lbfgs, lbfgs_best, adam_best)

_run_optax_optimization = jax.jit(_run_optax_optimization, static_argnums=(3, 4, 5))


def _forecast_sigma2_impl(omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
                          y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
                          sigma2_max: float, h: int, p: int,
                          q: int) -> jnp.ndarray:
    """Deterministic variance forecast h steps ahead.

    For multi-step forecasts, future squared innovations use E[e^2_{t+h}] = sigma^2_{t+h}.

    Parameters
    ----------
    omega : float
        GARCH intercept.
    alpha : jnp.ndarray
        ARCH coefficients.
    beta : jnp.ndarray
        GARCH coefficients.
    y_last : jnp.ndarray
        Last p observations (centered).
    sigma2_last : jnp.ndarray
        Last q conditional variances.
    sigma2_max : float
        Upper bound for variance clipping.
    h : int
        Forecast horizon (static).
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).

    Returns
    -------
    jnp.ndarray
        Variance forecasts of shape (h,).
    """
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

        y_buffer = jnp.roll(y_buffer, -1).at[-1].set(jnp.sqrt(sigma2_next))
        if q > 0:
            sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_next)
        return (y_buffer, sigma2_buffer), sigma2_next

    init_y = y_last.astype(jnp.float32)
    init_sigma2 = sigma2_last.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    _, forecasts = lax.scan(step, (init_y, init_sigma2), None, length=h)
    return forecasts

_forecast_sigma2_impl = jax.jit(_forecast_sigma2_impl, static_argnums=(6, 7, 8))


def _forecast_sigma2_stochastic_impl(
    omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
    y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
    sigma2_max: float, h: int, p: int, q: int, key: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stochastic variance forecast with random shocks.

    Parameters
    ----------
    omega : float
        GARCH intercept.
    alpha : jnp.ndarray
        ARCH coefficients.
    beta : jnp.ndarray
        GARCH coefficients.
    y_last : jnp.ndarray
        Last p observations (centered).
    sigma2_last : jnp.ndarray
        Last q conditional variances.
    sigma2_max : float
        Upper bound for variance clipping.
    h : int
        Forecast horizon (static).
    p : int
        ARCH order (static).
    q : int
        GARCH order (static).
    key : jnp.ndarray
        JAX PRNG key.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (y_path, sigma2_path) each of shape (h,).
    """
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

        epsilon = jax.random.normal(subkey, dtype=jnp.float32)
        y_next = epsilon * jnp.sqrt(sigma2_next)

        y_buffer = jnp.roll(y_buffer, -1).at[-1].set(y_next)
        if q > 0:
            sigma2_buffer = jnp.roll(sigma2_buffer, -1).at[-1].set(sigma2_next)
        return (y_buffer, sigma2_buffer), (y_next, sigma2_next)

    subkeys = jax.random.split(key, h)
    init_y = y_last.astype(jnp.float32)
    init_sigma2 = sigma2_last.astype(jnp.float32) if q > 0 else jnp.array([], dtype=jnp.float32)
    _, (y_path, sigma2_path) = lax.scan(step, (init_y, init_sigma2), subkeys)
    return y_path, sigma2_path

_forecast_sigma2_stochastic_impl = jax.jit(_forecast_sigma2_stochastic_impl, static_argnums=(6, 7, 8))


# =============================================================================
# GARCH Class
# =============================================================================

class GARCH(BaseForecaster):
    r"""GARCH model.

    Models time-varying volatility where conditional variance depends on
    past squared errors and past conditional variances.

    Parameters
    ----------
    p : int, default 1
        ARCH order (lagged squared shocks), must be >= 1.
    q : int, default 1
        GARCH order (lagged variances), must be >= 0.
    alias : str, default 'GARCH'
        Display name for the model.
    conformal_params : ConformalIntervals or None, default None
        Configuration for conformal prediction intervals.
    allow_extended_iterations : bool, default False
        Whether to use the extended optax-fallback iteration budget
        (120 instead of 80).
    iteration_scaling : str, default 'cubic'
        Deprecated/inert. Formerly scaled a data-adaptive iteration count,
        which cannot feed a jit static argument (CLAUDE.md §10); the budget
        is now fixed by allow_extended_iterations alone. The value is still
        validated for API compatibility.
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
    def _inverse_softplus(x: jnp.ndarray) -> jnp.ndarray:
        """Return y such that softplus(y) + _EPSILON = x, elementwise.

        Traceable: both branches evaluate under jnp.where; for target > 20
        softplus is identity to float precision, and the inner clamp keeps the
        untaken expm1 branch finite (no inf/NaN leaking through the select).
        """
        target = jnp.maximum(jnp.asarray(x) - 1e-8, 1e-8)
        return jnp.where(
            target > 20.0,
            target,
            jnp.log(jnp.maximum(jnp.expm1(jnp.minimum(target, 20.0)), 1e-8)),
        )

    def _get_init_params(self, y: jnp.ndarray, init_var: jnp.ndarray) -> jnp.ndarray:
        """Compute starting parameters, adaptive to model order.

        Uses ACF of squared observations for per-lag alpha proportions
        when p > 1, giving better initialization for higher-order models.

        Fully traceable: every value derived from ``y`` stays a jnp scalar,
        the degenerate-variance guard is a ``jnp.where`` select, and the lag
        loop runs over the *config-static* order p.
        """
        sample_var = jnp.maximum(jnp.var(y), 1e-8)

        if self.q == 0:
            alpha_sum = 0.05
            beta_sum = 0.0
            omega_target = jnp.maximum(sample_var * 0.95, 1e-8)
        else:
            alpha_sum = 0.05
            beta_sum = 0.90
            omega_target = jnp.maximum(sample_var * 0.05, 1e-8)

        init_omega = self._inverse_softplus(omega_target)

        # ACF-based per-lag alpha proportions for p > 1
        if self.p > 1:
            e2 = y ** 2
            e2_centered = e2 - jnp.mean(e2)
            var_e2 = jnp.var(e2)
            acf = jnp.stack([
                jnp.mean(e2_centered[k:] * e2_centered[:-k])
                for k in range(1, self.p + 1)
            ]) / jnp.maximum(var_e2, 1e-8)
            acf = jnp.maximum(acf, 0.01 / self.p)
            acf_targets = alpha_sum * acf / jnp.sum(acf)
            uniform_targets = jnp.full(self.p, alpha_sum / self.p)
            alpha_targets = jnp.where(var_e2 > 1e-8, acf_targets, uniform_targets)
            init_alphas = self._inverse_softplus(alpha_targets)
        else:
            init_alphas = jnp.full(self.p, self._inverse_softplus(alpha_sum / self.p))

        init_beta = (
            self._inverse_softplus(beta_sum / self.q) if self.q > 0
            else jnp.asarray(0.0)
        )

        return jnp.concatenate([
            jnp.reshape(init_omega, (1,)),
            init_alphas,
            jnp.full(self.q, init_beta) if self.q > 0 else jnp.zeros(0),
        ])

    def _fit_parameters_optax(
        self, y: jnp.ndarray, init_var: float, init_params: jnp.ndarray,
        n_iters: int | None = None, actual_len: int | None = None,
        sample_var: float = -1.0
    ) -> jnp.ndarray:
        """Run ADAM + L-BFGS optimization via the JIT-compiled loop.

        Parameters
        ----------
        y : jnp.ndarray
            Centered returns (may be padded).
        init_var : float
            Backcast initial variance.
        init_params : jnp.ndarray
            Unconstrained starting parameters.
        n_iters : int or None, default None
            Fixed iteration count. None uses the config-derived budget
            (_MAX_ITER, or _MAX_ITER_EXTENDED with allow_extended_iterations).
        actual_len : int or None, default None
            Actual data length for padded inputs.
        sample_var : float, default -1.0
            Sample variance for variance targeting; -1.0 disables VT.

        Returns
        -------
        jnp.ndarray
            Best parameters found (unconstrained).
        """
        if n_iters is None:
            n_iters = _MAX_ITER_EXTENDED if self.allow_extended_iterations else _MAX_ITER

        init_params = init_params.astype(jnp.float32)
        actual_len_jax = jnp.int32(-1 if actual_len is None else actual_len)
        sample_var_jax = jnp.asarray(sample_var, dtype=jnp.float32)
        return _run_optax_optimization(
            y, init_var, init_params, self.p, self.q, n_iters,
            actual_len_jax, sample_var_jax
        )

    def _fit_parameters(self, y: jnp.ndarray, n_iters: int | None = None,
                         actual_len: int | None = None) -> dict:
        """Estimate GARCH parameters via LBFGSB (primary) or ADAM + L-BFGS (fallback).

        Parameters
        ----------
        y : jnp.ndarray
            Centered returns (may be padded).
        n_iters : int or None, default None
            Fixed optax-fallback iteration count. None uses the config-derived
            budget (_MAX_ITER, or _MAX_ITER_EXTENDED with allow_extended_iterations).
        actual_len : int or None, default None
            Actual data length for padded inputs.

        Returns
        -------
        dict
            Fitted parameters, variance series, and state for forecasting.
        """
        n_eff = actual_len if actual_len is not None else len(y)
        min_obs = max(self.p, self.q) + 10
        if n_eff < min_obs:
            raise ValueError(
                f"Need at least {min_obs} observations for GARCH({self.p},{self.q}), got {n_eff}"
            )

        y_actual = y[:n_eff] if actual_len is not None else y

        # Rescale data to normalize the loss landscape (arch library technique)
        scale = jnp.maximum(jnp.std(y_actual), 1e-6)
        y_scaled = y / scale

        y_scaled_actual = y_scaled[:n_eff] if actual_len is not None else y_scaled
        init_var = _compute_backcast(y_scaled_actual)
        init_params = self._get_init_params(y_scaled_actual, init_var)

        actual_len_jax = jnp.int32(-1 if actual_len is None else actual_len)
        sample_var_val = jnp.maximum(jnp.var(y_scaled_actual), 1e-8)

        # Build candidates in DIRECT (constrained, positive) param space
        # Candidate 1: ACF-based (transform from softplus space)
        acf_direct = jax.nn.softplus(init_params)
        acf_direct = acf_direct.at[0].add(_EPSILON)

        # Candidate 2: SF-style uniform 0.1
        sf_direct = jnp.full(1 + self.p + self.q, 0.1)

        if self.q > 0:
            # Candidate 3: High-persistence (alpha=0.05, beta=0.90)
            hi_direct = jnp.concatenate([
                jnp.reshape(0.05 * sample_var_val, (1,)),
                jnp.full(self.p, 0.05 / self.p),
                jnp.full(self.q, 0.90 / self.q)
            ])
            # Candidate 4: Low-persistence (alpha=0.15, beta=0.70)
            lo_direct = jnp.concatenate([
                jnp.reshape(0.20 * sample_var_val, (1,)),
                jnp.full(self.p, 0.15 / self.p),
                jnp.full(self.q, 0.70 / self.q)
            ])
            candidates = [acf_direct, sf_direct, hi_direct, lo_direct]
        else:
            # Pure ARCH: third candidate with higher alpha
            hi_direct = jnp.concatenate([
                jnp.reshape(0.10 * sample_var_val, (1,)),
                jnp.full(self.p, 0.20 / self.p)
            ])
            candidates = [acf_direct, sf_direct, hi_direct]

        # Multi-start: optimize EVERY candidate (count is config-static) with a
        # vmapped LBFGSB run and pick the winner by NLL — replaces the former
        # init-loss prescreen + Python sorted/try-except selection, which
        # concretized traced values (ces.py's stacked-fits + argmin pattern).
        # Optimizer space is pinned to float32: the NLL kernels compute in
        # float32 regardless, so wider candidate dtypes buy cost, not precision.
        cand_stack = jnp.stack(candidates).astype(jnp.float32)

        def _run_one(cand):
            return _run_lbfgsb_optimization(
                y_scaled, init_var, cand, self.p, self.q, actual_len_jax)

        params_stack, losses = jax.vmap(_run_one)(cand_stack)
        losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
        best_params = jnp.take(params_stack, jnp.argmin(losses), axis=0)
        lbfgsb_ok = jnp.isfinite(losses).any()

        # Variance targeting only for p+q >= 4 (biases low-order MLE);
        # config-static choice.
        sv = (
            sample_var_val.astype(jnp.float32)
            if self.p + self.q >= 4 else jnp.float32(-1.0)
        )
        if n_iters is None:
            n_iters = _MAX_ITER_EXTENDED if self.allow_extended_iterations else _MAX_ITER
        init_params_f32 = init_params.astype(jnp.float32)
        empty_beta = jnp.zeros(0, dtype=jnp.float32)

        # Fallback to ADAM+L-BFGS only when every LBFGSB start produced a
        # non-finite NLL. lax.cond keeps the fallback off the eager/jit fast
        # path (under vmap batching both branches run — the cost of a fallback
        # that stays traceable; a host `if best is None` cannot).
        def _lbfgsb_branch(_):
            omega = jnp.maximum(best_params[0], _EPSILON)
            alpha = jnp.maximum(best_params[1:1 + self.p], _EPSILON)
            beta = (
                jnp.maximum(best_params[1 + self.p:1 + self.p + self.q], _EPSILON)
                if self.q > 0 else empty_beta
            )
            return omega, alpha, beta

        def _optax_branch(_):
            final_params = _run_optax_optimization(
                y_scaled, init_var, init_params_f32, self.p, self.q, n_iters,
                actual_len_jax, sv)
            omega = jax.nn.softplus(final_params[0]) + _EPSILON
            alpha = jax.nn.softplus(final_params[1:1 + self.p])
            beta = (
                jax.nn.softplus(final_params[1 + self.p:1 + self.p + self.q])
                if self.q > 0 else empty_beta
            )
            return omega, alpha, beta

        omega, alpha, beta = lax.cond(lbfgsb_ok, _lbfgsb_branch, _optax_branch, None)

        persistence = jnp.sum(alpha) + jnp.sum(beta)

        # Post-fit stationarity clamping: scale alpha+beta to 0.999 if needed.
        # Use 0.999 (not 0.98) to allow high-persistence financial data (e.g.
        # S&P 500) to retain accurate MLE parameters while still preventing
        # near-IGARCH instability. The clamp caps persistence at 0.999, so the
        # old `persistence >= 1.0` raise was unreachable (NaN comparisons fall
        # through it too) — removed rather than kept as untraceable dead code.
        clamped = persistence > 0.999
        scale_factor = jnp.where(clamped, 0.999 / jnp.maximum(persistence, 1e-8), 1.0)
        alpha = alpha * scale_factor
        beta = beta * scale_factor
        # Recompute omega so unconditional variance matches sample variance.
        # Without this, clamping collapses the unconditional variance
        # (omega/(1-persistence) becomes much smaller than sample_var).
        # astype keeps omega on the cond-branch dtype (f32) — sample_var_val
        # follows y and would silently promote the select under x64 (§10).
        omega = jnp.where(
            clamped, sample_var_val.astype(omega.dtype) * (1.0 - 0.999), omega
        )

        # Compute sigma2 in scaled space, then un-scale
        sigma2_scaled = _compute_sigma2_series(
            y_scaled, omega, alpha, beta, self.p, self.q, init_var)
        sigma2_full = sigma2_scaled * (scale ** 2)
        sigma2 = sigma2_full[:n_eff] if actual_len is not None else sigma2_full

        # Un-scale variance parameters
        omega_unscaled = omega * (scale ** 2)
        init_var_unscaled = init_var * (scale ** 2)

        return {
            'omega': omega_unscaled,
            'alpha': alpha,
            'beta': beta,
            'sigma2': sigma2,
            'fitted': jnp.zeros(n_eff),
            'y_mean': jnp.mean(y_actual),
            'y_last': y_actual[-self.p:],
            'sigma2_last': sigma2[-self.q:] if self.q > 0 else jnp.array([]),
            'init_var': init_var_unscaled,
        }

    def _forecast_sigma2(self, omega: float, alpha: jnp.ndarray,
                         beta: jnp.ndarray, y_last: jnp.ndarray,
                         sigma2_last: jnp.ndarray, init_var: float,
                         h: int) -> jnp.ndarray:
        """Forecast variance h steps ahead."""
        sigma2_max = init_var * _SIGMA2_MAX_MULT
        return _forecast_sigma2_impl(
            omega, alpha, beta, y_last, sigma2_last, sigma2_max, h, self.p, self.q
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None,
            n_iters: int | None = None,
            actual_len: int | None = None) -> 'GARCH':
        """Fit GARCH model to data.

        Parameters
        ----------
        y : jnp.ndarray
            Input time series (may be padded).
        X : jnp.ndarray or None, default None
            Exogenous variables (unused).
        n_iters : int or None, default None
            Fixed optax-fallback iteration count. None uses the config-derived
            budget (_MAX_ITER, or _MAX_ITER_EXTENDED with allow_extended_iterations).
        actual_len : int or None, default None
            Actual data length for padded inputs.

        Returns
        -------
        GARCH
            Fitted model (self).
        """
        y = utils.ensure_float(y)
        n_eff = actual_len if actual_len is not None else len(y)
        y_actual = y[:n_eff] if actual_len is not None else y

        # No data-dependent validation raises here: booleans on traced values
        # cannot branch under vmap (CLAUDE.md §1 rule 2). NaN input propagates
        # NaN outputs; near-constant input degenerates gracefully (the internal
        # scale is clamped at 1e-6, so forecasts collapse to the series mean).
        y_mean = jnp.mean(y_actual)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered, n_iters=n_iters, actual_len=actual_len)

        self.model_ = {
            'omega': result['omega'],
            'alpha': result['alpha'],
            'beta': result['beta'],
            'sigma2': result['sigma2'],
            'fitted': result['fitted'] + y_mean,
            'residuals': y_actual[:n_eff] - y_mean - result['fitted'],
            'y_mean': y_mean,
            'y_centered_last': result['y_last'],
            'sigma2_last': result['sigma2_last'],
            'init_var': result['init_var'],
            'y_train': y_actual,
        }
        self.model_['sigma'] = utils.calculate_sigma(
            self.model_['residuals'], n_eff - (self.p + self.q + 1)
        )

        if self.conformal_params is not None:
            self.model_['_cs'] = self.conformity_scores(y=y, X=X)

        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
        simulate: bool = False,
        n_sims: int = 1000,
        seed: int | None = None,
        return_paths: bool = True,
    ) -> dict:
        """Generate h-step forecasts (deterministic or Monte Carlo).

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Exogenous variables (unused).
        level : list[int] or None, default None
            Confidence levels for prediction intervals. When simulate=False,
            intervals are analytical (normal quantiles). When simulate=True,
            intervals are percentile-based from simulation paths.
        simulate : bool, default False
            Use Monte Carlo simulation instead of analytical forecasting.
        n_sims : int, default 1000
            Number of simulation paths (only used when simulate=True).
        seed : int or None, default None
            PRNG seed for reproducibility (only used when simulate=True).
        return_paths : bool, default True
            Include full simulation paths in output (only used when simulate=True).

        Returns
        -------
        dict
            When simulate=False:
                Keys: 'mean', 'sigma2', and optionally 'lo-{lv}', 'hi-{lv}'.
            When simulate=True:
                Keys: 'mean', 'median', 'sigma2_mean', 'sigma2_median',
                and optionally 'paths', 'sigma2_paths', 'lo-{lv}', 'hi-{lv}'.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        self._validate_h(h)

        if simulate:
            if not isinstance(n_sims, int) or n_sims < 1:
                raise ValueError(f"n_sims must be a positive integer, got {n_sims}")

            key = jax.random.PRNGKey(seed if seed is not None else 0)
            paths, sigma2_paths = self._simulate_paths(
                h, n_sims, key,
                self.model_['omega'], self.model_['alpha'], self.model_['beta'],
                self.model_['y_centered_last'], self.model_['sigma2_last'],
                self.model_['init_var'], self.model_['y_mean']
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
                if n_sims < 30:
                    raise ValueError(
                        f"n_sims={n_sims} is too small for reliable percentile-based intervals. "
                        "Use n_sims >= 30."
                    )
                for lv in level:
                    lower_q = (100 - lv) / 2
                    upper_q = 100 - lower_q
                    res[f'lo-{lv}'] = jnp.percentile(paths, lower_q, axis=0)
                    res[f'hi-{lv}'] = jnp.percentile(paths, upper_q, axis=0)

            return res

        sigma2_forecast = self._forecast_sigma2(
            self.model_['omega'], self.model_['alpha'], self.model_['beta'],
            self.model_['y_centered_last'], self.model_['sigma2_last'],
            self.model_['init_var'], h
        )
        mean_forecast = jnp.full(h, self.model_['y_mean'])

        res = {'mean': mean_forecast, 'sigma2': sigma2_forecast}
        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                if h != self.conformal_params.h:
                    raise ValueError(
                        f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                        "conformity scores cover exactly conformal_params.h steps."
                    )
                cs = self.model_.get('_cs')
                if cs is None:
                    raise ValueError("Conformity scores not found.")
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                sigma_forecast = jnp.sqrt(sigma2_forecast)
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = mean_forecast - z * sigma_forecast
                    res[f'hi-{lv}'] = mean_forecast + z * sigma_forecast

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        """Return in-sample fitted values and conditional variance.

        Parameters
        ----------
        level : list[int] or None, default None
            Confidence levels for fitted prediction intervals.

        Returns
        -------
        dict
            Keys: 'fitted', 'sigma2', and optionally 'fitted-lo-{lv}', 'fitted-hi-{lv}'.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        res = {'fitted': self.model_['fitted'], 'sigma2': self.model_['sigma2']}
        if level is not None:
            level = sorted(level)
            sigma_t = jnp.sqrt(self.model_['sigma2'])
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
        n_iters: int | None = None,
        actual_len: int | None = None,
        simulate: bool = False,
        n_sims: int = 1000,
        seed: int | None = None,
        return_paths: bool = True,
    ) -> dict:
        """Stateless fit-and-predict (deterministic or Monte Carlo).

        Parameters
        ----------
        y : jnp.ndarray
            Input time series (may be padded).
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Exogenous variables (unused).
        X_future : jnp.ndarray or None, default None
            Future exogenous variables (unused).
        level : list[int] or None, default None
            Confidence levels for prediction intervals. When simulate=True,
            forecast intervals are percentile-based from paths, while fitted
            intervals remain analytical.
        fitted : bool, default False
            Whether to return in-sample fitted values.
        n_iters : int or None, default None
            Fixed optax-fallback iteration count. None uses the config-derived
            budget (_MAX_ITER, or _MAX_ITER_EXTENDED with allow_extended_iterations).
        actual_len : int or None, default None
            Actual data length for padded inputs.
        simulate : bool, default False
            Use Monte Carlo simulation instead of analytical forecasting.
        n_sims : int, default 1000
            Number of simulation paths (only used when simulate=True).
        seed : int or None, default None
            PRNG seed for reproducibility (only used when simulate=True).
        return_paths : bool, default True
            Include full simulation paths in output (only used when simulate=True).

        Returns
        -------
        dict
            Keys: 'mean', 'sigma2', and optionally intervals and fitted values.
            When simulate=True, also includes 'median', 'sigma2_mean',
            'sigma2_median', and optionally 'paths', 'sigma2_paths'.
        """
        if simulate:
            # Fit a clone: forecast() is contractually stateless, and fit()
            # attribute writes under a caller's vmap would leak tracers onto
            # this estimator (the §5.1 clone pattern).
            m = self.new()
            m.fit(y, X, n_iters=n_iters, actual_len=actual_len)
            res = m.predict(h, X_future, level, simulate=True,
                            n_sims=n_sims, seed=seed, return_paths=return_paths)

            if fitted:
                res['fitted'] = m.model_['fitted']
                if level is not None:
                    level = sorted(level)
                    sigma_t = jnp.sqrt(m.model_['sigma2'])
                    for lv in level:
                        z = utils._jax_norm_ppf((100 + lv) / 200)
                        res[f'fitted-lo-{lv}'] = m.model_['fitted'] - z * sigma_t
                        res[f'fitted-hi-{lv}'] = m.model_['fitted'] + z * sigma_t

            return res

        self._validate_h(h)
        y = utils.ensure_float(y)

        n_eff = actual_len if actual_len is not None else len(y)
        y_actual = y[:n_eff] if actual_len is not None else y
        y_mean = jnp.mean(y_actual)
        y_centered = y - y_mean
        result = self._fit_parameters(y_centered, n_iters=n_iters, actual_len=actual_len)

        sigma2_forecast = self._forecast_sigma2(
            result['omega'], result['alpha'], result['beta'],
            result['y_last'], result['sigma2_last'], result['init_var'], h
        )
        mean_forecast = jnp.full(h, y_mean)

        res = {'mean': mean_forecast, 'sigma2': sigma2_forecast}

        if fitted:
            res['fitted'] = result['fitted'] + y_mean

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                if h != self.conformal_params.h:
                    raise ValueError(
                        f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                        "conformity scores cover exactly conformal_params.h steps."
                    )
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
            else:
                sigma_forecast = jnp.sqrt(sigma2_forecast)
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'lo-{lv}'] = mean_forecast - z * sigma_forecast
                    res[f'hi-{lv}'] = mean_forecast + z * sigma_forecast
            if fitted:
                sigma_t = jnp.sqrt(result['sigma2'])
                for lv in level:
                    z = utils._jax_norm_ppf((100 + lv) / 200)
                    res[f'fitted-lo-{lv}'] = res['fitted'] - z * sigma_t
                    res[f'fitted-hi-{lv}'] = res['fitted'] + z * sigma_t

        return res

    def _forecast_sigma2_stochastic(
        self, omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
        y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
        init_var: float, h: int, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Single stochastic path forecast."""
        sigma2_max = init_var * _SIGMA2_MAX_MULT
        return _forecast_sigma2_stochastic_impl(
            omega, alpha, beta, y_last, sigma2_last, sigma2_max, h, self.p, self.q, key
        )

    def _simulate_paths(
        self, h: int, n_sims: int, key: jnp.ndarray,
        omega: float, alpha: jnp.ndarray, beta: jnp.ndarray,
        y_last: jnp.ndarray, sigma2_last: jnp.ndarray,
        init_var: float, y_mean: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate n_sims paths in parallel via vmap."""
        def simulate_single_path(subkey):
            y_path, sigma2_path = self._forecast_sigma2_stochastic(
                omega, alpha, beta, y_last, sigma2_last, init_var, h, subkey
            )
            return y_path + y_mean, sigma2_path

        keys = jax.random.split(key, n_sims)
        paths, sigma2_paths = jax.vmap(simulate_single_path)(keys)
        return paths, sigma2_paths

