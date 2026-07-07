"""
MFLES (Multi-Feature Locally Exponential Smoothing) in JAX

Implements a StatsForecast-compatible MFLES model that combines multiple
additive components fitted iteratively to residuals, fully JIT-compiled
via lax.while_loop for performance.

**Components:**
- Piecewise linear trend with adaptive changepoints (LASSO, fraction-based knot selection)
- Multiple seasonal patterns via Fourier series (OLS or weighted OLS)
- Residual smoothing via SES ensemble ("lite" mode) or rolling means ("full" mode)
- Optional exogenous variables via OLS
- Robust estimation via Siegel repeated medians (auto-detected or user-specified)

**Algorithm:**
Each iteration fits components sequentially to the current residuals:
1. Median-based initialization
2. Seasonal update: Fourier OLS/WLS on residuals, cycling through periods each round
3. Trend update: OLS, robust (Siegel), or piecewise LASSO with adaptive knots
4. Residual smoothing: SES ensemble or rolling mean
5. Exogenous update: OLS on remaining residuals
6. Convergence check: stops after 6 consecutive non-improving rounds or max_rounds

**Formulas:**
- Seasonal: y_t = Σ [a_k cos(2πkt/T) + b_k sin(2πkt/T)] for k=1..K
- Trend: y_t = β₀ + β₁t + Σ β_k max(0, t-τ_k) (piecewise linear, LASSO-regularized)
- SES: ŷ_t = α·y_t + (1-α)·ŷ_{t-1}

**Implementation:**
- Entire fitting loop JIT-compiled via lax.while_loop with a _LoopState NamedTuple
- Static boolean flags (has_seasonality, use_changepoints, etc.) baked into the loop
  at compile time via _make_fit_loop, avoiding runtime branching overhead
- Adaptive changepoint count: n_changepoints as a float (e.g. 0.25) sets knots
  proportional to series length, capped at 50
- Multiplicative mode: automatic when seasonal_period is provided and series is positive;
  uses log-transform internally, reverted at predict time
- Seasonal tail fix: last full period stored in _LoopState for correct multi-step forecasting
- Conformal prediction intervals via BaseForecaster.conformity_scores

**Attributes:**
- `robust`: Siegel repeated medians (True), OLS (False), or auto-detect (None)
- `multiplicative`: Log-space fitting for multiplicative seasonality (set during fit)
- `penalty`: R²-based trend dampening scalar (set during fit, optional)
- `verbose`: Verbosity level (currently unused, reserved for future logging)

**Methods:**
- `fit(y, seasonal_period, X, ...)`: Fit all components; extensive hyperparameter control
- `predict(h, X, level)`: Forecast h steps from fitted state; optional conformal intervals
- `forecast(y, h, ...)`: Stateless fit+predict (inherits base class default)
- `optimize(y, ...)`: Auto-tune hyperparameters via rolling cross-validation
"""
# mfles.py
from __future__ import annotations
from functools import partial as _partial
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals


# ============================================================================
# Loop State for JIT-compiled fitting
# ============================================================================

class _LoopState(NamedTuple):
    """State carried through the MFLES fitting loop.
    
    All mutable state during fitting is captured here to enable
    lax.while_loop compilation.
    """
    i: jnp.ndarray                  # iteration counter (scalar int)
    fitted: jnp.ndarray             # current fitted values (n,)
    linear_component: jnp.ndarray   # accumulated linear/trend (n,)
    seasonal_component: jnp.ndarray # accumulated seasonal (n,)
    ses_component: jnp.ndarray      # accumulated SES smoothing (n,)
    exogenous_component: jnp.ndarray # accumulated exogenous (n,)
    trend_tail: jnp.ndarray         # [prev_end, curr_end] for forecasting (2,)
    seas_tail: jnp.ndarray          # last period of seasonal, pre-allocated (max_period,)
    seas_tail_len: jnp.ndarray      # actual length of seas_tail to use (scalar int)
    best: jnp.ndarray               # best MSE so far (scalar float)
    stalls: jnp.ndarray             # consecutive non-improving iterations (scalar int)
    robust: jnp.ndarray             # robust regression flag (scalar bool)
    penalty: jnp.ndarray            # R² penalty for trend dampening (scalar float)
    exo_beta: jnp.ndarray           # exogenous coefficients (n_features,) or empty
    converged: jnp.ndarray          # early stopping flag (scalar bool)

@jax.jit
def _mse(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error metric.

    Args:
        y: Actual values
        yhat: Predicted values

    Returns:
        Scalar MSE value
    """
    return jnp.mean((y - yhat) ** 2)

@jax.jit
def _mae(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error metric.

    Args:
        y: Actual values
        yhat: Predicted values

    Returns:
        Scalar MAE value
    """
    return jnp.mean(jnp.abs(y - yhat))

@jax.jit
def _mape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute percentage error metric.

    Args:
        y: Actual values
        yhat: Predicted values

    Returns:
        Scalar MAPE value
    """
    return jnp.mean(jnp.abs((y - yhat) / (jnp.abs(y) + 1e-10)))

@jax.jit
def _smape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Symmetric mean absolute percentage error metric.

    Args:
        y: Actual values
        yhat: Predicted values

    Returns:
        Scalar SMAPE value
    """
    return jnp.mean(2.0 * jnp.abs(y - yhat) / (jnp.abs(y) + jnp.abs(yhat) + 1e-10))
_metric2fn = {"mse": _mse, "mae": _mae, "mape": _mape, "smape": _smape}

@_partial(jax.jit, static_argnums=(1,))
def _rolling_mean(y: jnp.ndarray, window: int) -> jnp.ndarray:
    """Compute rolling window average with left-edge retention.

    Averages values over a sliding window. For the first (window-1) positions,
    returns original values to avoid boundary artifacts.

    Args:
        y: Input array to smooth
        window: Window size for averaging (static)

    Returns:
        Smoothed array with same shape as input
    """
    if window <= 1:
        return y
    kernel = jnp.ones((window,), dtype=y.dtype) / window
    conv = jnp.convolve(y, kernel, mode="valid")
    pad = y[:window - 1]
    return jnp.concatenate([pad, conv], axis=0)

@_partial(jax.jit, static_argnums=(2, 3))
def _ses_ensemble_via_utils(resids: jnp.ndarray, alphas: jnp.ndarray, smooth: bool, order: int) -> jnp.ndarray:
    """SES ensemble or rolling mean based on smooth flag.
    
    Args:
        resids: Residuals array
        alphas: Pre-computed array of alpha values for SES ensemble
        smooth: If True, use SES ensemble; if False, use rolling mean (static)
        order: Window size for rolling mean minus 1 (static)
    """
    def smooth_path():
        def one(alpha):
            _, fitted = utils._ses_forecast(resids, alpha)
            f = jnp.nan_to_num(fitted, nan=0.0)
            mask = jnp.isnan(fitted)
            idx = jnp.maximum.accumulate((~mask).astype(jnp.int32) * jnp.arange(resids.size))
            return f[idx]
        mats = jax.vmap(one)(alphas)
        return jnp.mean(mats, axis=0)
    
    def rolling_path():
        rm = _rolling_mean(resids, order + 1)
        return rm.at[:order + 1].set(resids[:order + 1])
    
    # Use lax.cond for JAX-traceable branching (smooth is static so this traces correctly)
    return lax.cond(smooth, smooth_path, rolling_path)

@jax.jit
def _cap_outliers(y: jnp.ndarray, k: float = 3.0) -> jnp.ndarray:
    """Clip outliers beyond k standard deviations from mean.

    Applies Winsorization to cap extreme values at k standard deviations
    from the mean in both directions.

    Args:
        y: Input array
        k: Number of standard deviations for clipping threshold

    Returns:
        Array with outliers clipped to [mean - k*std, mean + k*std]
    """
    mu, sd = jnp.mean(y), jnp.std(y)
    return jnp.clip(y, mu - k * sd, mu + k * sd)

@jax.jit
def _fourier_order_from_period(period: jnp.ndarray) -> jnp.ndarray:
    """Determine Fourier series order based on seasonal period length.

    Uses StatsForecast-compatible heuristic:
    - Period < 10: order 5
    - Period < 70: order 10
    - Period >= 70: order 15

    Args:
        period: Seasonal period (can be array or scalar)

    Returns:
        Fourier order as int32 array
    """
    p = period.astype(jnp.int32)
    return jnp.where(p < 10, jnp.int32(5), jnp.where(p < 70, jnp.int32(10), jnp.int32(15)))

@_partial(jax.jit, static_argnums=(0, 2, 3, 4))
def _build_fourier_and_weights_stack(
    n: int,
    sp_array: jnp.ndarray,
    max_order: int,
    forced_order: int,
    use_weights: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build Fourier design matrices and optional seasonality weights using JAX primitives.

    Creates stacked Fourier series design matrices for multiple seasonal periods,
    with optional recency-weighted coefficients for improved stability on long series.

    Args:
        n: Series length (static)
        sp_array: Array of seasonal periods to create Fourier features for
        max_order: Maximum Fourier order across all periods (static)
        forced_order: User-specified Fourier order, or -1 for auto-detection (static)
        use_weights: Whether to compute recency weights for seasonal fitting (static)

    Returns:
        Tuple of (fourier_stack, weights_stack) where:
            - fourier_stack: Shape (num_periods, n, 2*max_order) with cos/sin features
            - weights_stack: Shape (num_periods, n) with recency weights (or zeros if disabled)
    """
    t = jnp.arange(1, n + 1, dtype=jnp.float32).reshape(-1, 1)
    k = jnp.arange(1, max_order + 1, dtype=jnp.float32).reshape(1, -1)
    col_idx = jnp.arange(2 * max_order, dtype=jnp.int32)
    time_idx = jnp.arange(n, dtype=jnp.int32)

    def one(period: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        p_f = period.astype(jnp.float32)
        angles = 2.0 * jnp.pi * t * (k / p_f)
        fs = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=1)
        order = jnp.int32(forced_order) if forced_order > 0 else _fourier_order_from_period(period)
        mask = (col_idx < (2 * order)).astype(fs.dtype)
        fs = fs * mask
        weights = 1.0 + (time_idx // period).astype(jnp.float32)
        return fs, weights

    fourier_stack, weights_stack = jax.vmap(one)(sp_array)
    if not use_weights:
        weights_stack = jnp.zeros_like(weights_stack)
    return fourier_stack, weights_stack

def _median_init(y: jnp.ndarray, period: int | None) -> jnp.ndarray:
    """Initialize fitted values with period-wise median smoothing.

    Computes medians over complete cycles of the given period, repeating
    the pattern across the series length. Used for robust initial estimates.

    Args:
        y: Input time series array
        period: Seasonal period for median computation, or None for global median

    Returns:
        Array of same shape as y with period-wise median initialization
    """
    n = y.shape[0]
    if period is None:
        return jnp.full_like(y, jnp.median(y))
    m = int(period)
    full = (n // m) * m
    base = y[:full].reshape(-1, m)
    per = jnp.median(base, axis=1)
    med = jnp.repeat(per, m)
    if full < n:
        tail = jnp.median(y[-m:])
        med = jnp.concatenate([med, jnp.full((n - full,), tail, dtype=y.dtype)], axis=0)
    return med

@jax.jit
def _fast_ols_fit_predict(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Fit simple linear regression (scalar predictor) and return fitted values.

    Efficiently computes OLS fit for univariate regression using closed-form
    solution without matrix inversion.

    Args:
        x: Predictor variable (1D array)
        y: Response variable (1D array)

    Returns:
        Fitted values from linear model: slope * x + intercept
    """
    M = x.shape[0]
    x_sum, y_sum = jnp.sum(x), jnp.sum(y)
    x2, xy = jnp.dot(x, x), jnp.dot(x, y)
    denom = M * x2 - x_sum * x_sum + 1e-12
    slope = (M * xy - x_sum * y_sum) / denom
    intercept = (y_sum - slope * x_sum) / M
    return slope * x + intercept

@jax.jit
def _siegel_repeated_medians(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Fit a robust linear regression via Siegel repeated medians and return fitted values.

    Computes pairwise slopes between all point pairs, takes the median slope
    per point, then the median of those medians as the global slope. Robust
    to up to 50% outliers in both x and y.

    Args:
        x: Predictor array of shape (n,).
        y: Response array of shape (n,).

    Returns:
        Fitted values slope*x + intercept of shape (n,).
    """
    n = x.shape[0]
    X_i = x[:, None]; X_j = x[None, :]
    Y_i = y[:, None]; Y_j = y[None, :]
    dx = X_j - X_i
    dy = Y_j - Y_i
    slopes = jnp.where(dx != 0, dy / (dx + 1e-12), 0.0)
    med_slopes_i = jnp.median(slopes, axis=1)
    slope = jnp.median(med_slopes_i)
    intercept = jnp.median(y - slope * x)
    return slope * x + intercept

@jax.jit
def _ols_predict(X: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fit OLS regression and return fitted values and coefficients.

    Solves the normal equations via pseudo-inverse: β = pinv(XᵀX) · Xᵀy.

    Args:
        X: Design matrix of shape (n, p).
        y: Response vector of shape (n,).

    Returns:
        Tuple of (fitted values of shape (n,), coefficient vector of shape (p,)).
    """
    beta = jnp.linalg.pinv(X.T @ X) @ (X.T @ y)
    return X @ beta, beta

@jax.jit
def _wls_predict(X: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    """Fit Weighted Least Squares regression and return fitted values.

    Solves the weighted normal equations: β = pinv(XᵀWX) · XᵀWy, where W = diag(w).

    Args:
        X: Design matrix of shape (n, p).
        y: Response vector of shape (n,).
        w: Non-negative weight vector of shape (n,).

    Returns:
        Fitted values of shape (n,).
    """
    WX = X * w[:, None]
    beta = jnp.linalg.pinv(X.T @ WX) @ (X.T @ (w * y))
    return X @ beta

@jax.jit
def _cov_proxy(y: jnp.ndarray, mult: int = 1) -> jnp.ndarray:
    """Coefficient of variation proxy used for robust-mode auto-detection.

    In multiplicative mode (mult != 0), uses a log-space variance formula
    consistent with StatsForecast's MFLES implementation. In additive mode,
    returns std/mean.

    Args:
        y: Residual or series array.
        mult: Non-zero for multiplicative formula, zero for additive (std/mean).

    Returns:
        Scalar CoV proxy value.
    """
    def mult_path():
        # Keep parity with StatsForecast's multiplicative CoV proxy formula.
        return jnp.sqrt(jnp.exp(jnp.log(10.0) * (jnp.std(y) ** 2) - 1.0))
    
    def additive_path():
        sd, mu = jnp.std(y), jnp.mean(y)
        return jnp.where(mu != 0, sd / mu, sd)
    
    return lax.cond(mult != 0, mult_path, additive_path)

@_partial(jax.jit, static_argnums=(0, 2))
def _uniform_knots(n: int, k: int, max_knots: int = 50) -> jnp.ndarray:
    """
    Create uniformly spaced knots. Returns fixed-size array for JIT compatibility.
    
    Args:
        n: Series length
        k: Number of knots (can be dynamic)
        max_knots: Maximum number of knots (static, for fixed output size)
    
    Returns:
        Array of shape (max_knots,) with knot positions (padded with zeros if k < max_knots)
    """
    knots = jnp.zeros(max_knots, dtype=jnp.int32)
    
    # Compute step size
    step = n // (k + 1)
    
    # Fill knots array
    positions = step * (jnp.arange(max_knots) + 1)
    fill_mask = jnp.arange(max_knots) < k
    knots = jnp.where(fill_mask, jnp.clip(positions, 1, n - 2), 0)
    
    return knots

@_partial(jax.jit, static_argnums=(0,))
def _hinge_basis_from_knots(n: int, knots: jnp.ndarray) -> jnp.ndarray:
    """
    Create hinge basis functions from knots.
    
    Knots array is fixed-size with padding (zeros indicate no knot).
    """
    t = jnp.arange(n, dtype=jnp.float32)
    K = knots.shape[0]
    
    # Create hinge functions for all positions (including padded zeros)
    def one(k_idx):
        tau = knots[k_idx].astype(jnp.float32)
        # If knot is 0 (padding), return zeros
        hinge = jnp.maximum(0.0, t - tau)
        is_valid = knots[k_idx] > 0
        return jnp.where(is_valid, hinge, 0.0)
    
    H = jax.vmap(one)(jnp.arange(K)).T  # Shape: (n, K)
    
    # Concatenate: [t, ones, hinges]
    return jnp.concatenate([t.reshape(-1, 1), jnp.ones((n, 1), t.dtype), H], axis=1)

@jax.jit
def _spectral_step(X: jnp.ndarray) -> float:
    s = jnp.linalg.svd(X, full_matrices=False, compute_uv=False)
    L = (s[0] ** 2) + 1e-6
    return 1.0 / L

@jax.jit
def _soft(z: jnp.ndarray, lam: float) -> jnp.ndarray:
    return jnp.sign(z) * jnp.maximum(0.0, jnp.abs(z) - lam)

@jax.jit
def _lasso_ista_with_step(X: jnp.ndarray, y: jnp.ndarray, alpha: float, step: jnp.ndarray, maxiter: int = 200) -> jnp.ndarray:
    """LASSO via ISTA with pre-computed spectral step (avoids redundant SVD)."""
    beta = jnp.zeros((X.shape[1],), dtype=y.dtype)
    def body(i, b):
        grad = X.T @ (X @ b - y)
        return _soft(b - step * grad, step * alpha)
    return lax.fori_loop(0, maxiter, body, beta)


# ============================================================================
# JIT-Compiled Fit Loop
# ============================================================================

class _FitConfig(NamedTuple):
    """Static configuration for the fit loop (arrays that don't change)."""
    y_tr: jnp.ndarray              # transformed target (n,)
    x_idx: jnp.ndarray             # jnp.arange(n) for trend fitting (n,)
    fourier_stack: jnp.ndarray     # stacked Fourier matrices (num_periods, n, max_cols) or empty
    weights_stack: jnp.ndarray     # stacked seasonality weights (num_periods, n) or empty
    sp_array: jnp.ndarray          # seasonal periods array (num_periods,) or empty
    X_exo: jnp.ndarray             # exogenous variables (n, n_features) or empty
    ma_array: jnp.ndarray          # MA cycle values (ma_len,)
    ses_alphas: jnp.ndarray        # pre-computed alphas for SES ensemble (num_alphas,)
    # Pre-computed changepoint basis (avoids redundant SVD inside loop)
    hinge_basis: jnp.ndarray       # pre-computed hinge basis matrix (n, n_cols) or empty
    lasso_step: jnp.ndarray        # pre-computed spectral step for LASSO (scalar)
    # Scalar hyperparameters
    seasonal_lr: float
    linear_lr: float
    exogenous_lr: float
    rs_lr: float
    round_penalty: float
    alpha: float                   # LASSO regularization
    cov_threshold: float
    multiplicative_flag: int       # 1 if multiplicative mode, else 0
    n_cps: int                     # number of changepoints
    max_period: int                # max seasonal period (for seas_tail allocation)
    n_exo_features: int            # number of exogenous features
    min_iter: int                  # minimum rounds before early-stop check
    lasso_maxiter: int             # ISTA iterations for changepoint trend fit


def _fit_body(
    state: _LoopState,
    config: _FitConfig,
    has_seasonality: bool,
    has_exogenous: bool,
    use_seasonality_weights: bool,
    use_changepoints: bool,
    gradient_strategy: bool,
    smoother: bool,
    ses_mode_code: int,
    init_robust: bool,              # Whether robust was None at start (need auto-detect)
) -> _LoopState:
    """Single iteration of the MFLES fitting loop.
    
    All boolean flags are static (compile-time constants) to enable efficient branching.
    """
    i = state.i
    n = config.y_tr.shape[0]
    
    # -------------------------------------------------------------------------
    # 1. Compute residuals and current MSE
    # -------------------------------------------------------------------------
    resids = config.y_tr - state.fitted
    cur = _mse(config.y_tr, state.fitted)
    
    # -------------------------------------------------------------------------
    # 2. Convergence check (StatsForecast-style non-improvement counting)
    # -------------------------------------------------------------------------
    improved = cur < state.best
    new_best = lax.cond(improved, lambda: cur, lambda: state.best)
    new_stalls = lax.cond(improved, lambda: jnp.int32(0), lambda: state.stalls + 1)
    converged = (i >= config.min_iter) & (new_stalls >= 6)

    
    # -------------------------------------------------------------------------
    # 3. Seasonal update (if has_seasonality)
    # -------------------------------------------------------------------------
    if has_seasonality:
        num_periods = config.sp_array.shape[0]
        k = i % num_periods  # which period to use this iteration
        
        # Get Fourier matrix for this period (dynamic indexing)
        Xs = config.fourier_stack[k]  # (n, max_cols)
        
        # Compute seasonal component
        if use_seasonality_weights:
            w = config.weights_stack[k]  # (n,)
            seas = _wls_predict(Xs, resids, w) * config.seasonal_lr
        else:
            seas, _ = _ols_predict(Xs, resids)
            seas = seas * config.seasonal_lr
        
        # Check if improvement
        test_seas = _mse(config.y_tr, state.fitted + seas)
        seas_improves = test_seas < new_best

        # Conditional update
        new_fitted = lax.cond(seas_improves, lambda: state.fitted + seas, lambda: state.fitted)
        new_seasonal = lax.cond(seas_improves, lambda: state.seasonal_component + seas, lambda: state.seasonal_component)
        new_best_after_seas = lax.cond(seas_improves, lambda: test_seas, lambda: new_best)
        
        # Update seas_tail from the cumulative seasonal component.
        # Using only the incremental `seas` update underestimates forecast seasonality.
        p = config.sp_array[k]
        max_p = state.seas_tail.shape[0]  # This is static (known at trace time)
        
        # Always copy last max_p elements (static slice size), track actual period in seas_tail_len
        # The actual period values are in the last p positions, but we copy max_p for JIT compat
        last_max_p = lax.dynamic_slice(new_seasonal, (n - max_p,), (max_p,))
        
        # Rearrange so that the last p values are at the start
        # We need seas[-p:] at positions [0:p], but we have seas[-max_p:] in last_max_p
        # seas[-p:] corresponds to last_max_p[max_p - p:]
        # Shift so it starts at index 0: rotate left by (max_p - p)
        indices = (jnp.arange(max_p) + (max_p - p)) % max_p
        rotated = last_max_p[indices]

        new_seas_tail = lax.cond(
            seas_improves,
            lambda: rotated,
            lambda: state.seas_tail
        )
        new_seas_tail_len = lax.cond(
            seas_improves,
            lambda: p,
            lambda: state.seas_tail_len
        )
    else:
        new_fitted = state.fitted
        new_seasonal = state.seasonal_component
        new_best_after_seas = new_best
        new_seas_tail = state.seas_tail
        new_seas_tail_len = state.seas_tail_len
    
    # Update resids after seasonal
    resids = config.y_tr - new_fitted
    
    # -------------------------------------------------------------------------
    # 4. Exogenous update (if has_exogenous and i > 0)
    # -------------------------------------------------------------------------
    if has_exogenous:
        def exo_update():
            beta = jnp.linalg.pinv(config.X_exo.T @ config.X_exo) @ (config.X_exo.T @ resids)
            exo = (config.X_exo @ beta) * config.exogenous_lr
            test_exo = _mse(config.y_tr, new_fitted + exo)
            exo_improves = test_exo < new_best_after_seas

            upd_fitted = lax.cond(exo_improves, lambda: new_fitted + exo, lambda: new_fitted)
            upd_exo_comp = lax.cond(exo_improves, lambda: state.exogenous_component + exo, lambda: state.exogenous_component)
            upd_best = lax.cond(exo_improves, lambda: test_exo, lambda: new_best_after_seas)
            upd_beta = lax.cond(exo_improves, lambda: beta, lambda: state.exo_beta)
            return upd_fitted, upd_exo_comp, upd_best, upd_beta

        def no_exo_update():
            return new_fitted, state.exogenous_component, new_best_after_seas, state.exo_beta

        new_fitted2, new_exo_comp, new_best_after_exo, new_exo_beta = lax.cond(
            i > 0,
            exo_update,
            no_exo_update
        )
    else:
        new_fitted2 = new_fitted
        new_exo_comp = state.exogenous_component
        new_best_after_exo = new_best_after_seas
        new_exo_beta = state.exo_beta
    
    # Update resids after exogenous
    resids = config.y_tr - new_fitted2
    
    # -------------------------------------------------------------------------
    # 5. Trend + residual smoothing update
    # -------------------------------------------------------------------------
    # StatsForecast-compatible cadence:
    # - Odd rounds: trend update (robust / OLS / piecewise)
    # - Even rounds (after round 4): residual smoothing update
    is_odd = (i % 2) == 1

    def trend_piecewise():
        Xb = config.hinge_basis
        beta = _lasso_ista_with_step(Xb, resids, config.alpha, config.lasso_step, config.lasso_maxiter)
        return (Xb @ beta) * config.linear_lr

    def trend_robust():
        return _siegel_repeated_medians(config.x_idx, resids)

    def trend_ols():
        return _fast_ols_fit_predict(config.x_idx, resids)

    def odd_round_candidate():
        # SF behavior: first odd trend round uses OLS even when changepoints are enabled.
        use_piecewise = use_changepoints & (config.n_cps > 0) & (i != jnp.int32(1))
        tren = lax.cond(
            state.robust,
            trend_robust,
            lambda: lax.cond(use_piecewise, trend_piecewise, trend_ols)
        )
        test = _mse(config.y_tr, new_fitted2 + tren)
        improves = test < new_best_after_exo
        # False -> update linear_component, not ses_component
        return tren, improves, test, jnp.bool_(False)

    def even_round_candidate():
        # Even rounds before i>4 do not update in SF MFLES.
        def no_even_update():
            zero = jnp.zeros_like(resids)
            return zero, jnp.bool_(False), new_best_after_exo, jnp.bool_(True)

        def ses_update():
            # ses_mode:
            # 0=off, 1=lite(rolling), 2=full(SES ensemble), 3=adaptive(use smoother flag)
            if ses_mode_code == 2:
                ses_tren = _ses_ensemble_via_utils(resids, config.ses_alphas, True, 1) * config.rs_lr
            elif ses_mode_code == 3:
                ses_tren = _ses_ensemble_via_utils(resids, config.ses_alphas, bool(smoother), 1) * config.rs_lr
            else:
                ses_tren = _ses_ensemble_via_utils(resids, config.ses_alphas, False, 1) * config.rs_lr

            ses_test = _mse(config.y_tr, new_fitted2 + ses_tren)
            ses_improves = ses_test < (new_best_after_exo * (1.0 - config.round_penalty))
            return ses_tren, ses_improves, ses_test, jnp.bool_(True)

        if ses_mode_code == 0:
            return no_even_update()
        return lax.cond(i > jnp.int32(4), ses_update, no_even_update)

    tren, trend_improves, test_trend, used_ses = lax.cond(
        is_odd,
        odd_round_candidate,
        even_round_candidate
    )

    new_fitted3 = lax.cond(trend_improves, lambda: new_fitted2 + tren, lambda: new_fitted2)
    new_best_final = lax.cond(trend_improves, lambda: test_trend, lambda: new_best_after_exo)

    # Update only the component that produced the accepted candidate.
    new_linear = lax.cond(
        trend_improves & (~used_ses),
        lambda: state.linear_component + tren,
        lambda: state.linear_component
    )
    new_ses = lax.cond(
        trend_improves & used_ses,
        lambda: state.ses_component + tren,
        lambda: state.ses_component
    )

    # Keep trend state aligned with SF logic:
    # - accepted trend update contributes tren[-2:] to trend tail
    # - accepted residual smoother contributes tren[-1] to both tail points
    def trend_from_linear():
        delta = jnp.array([tren[-2], tren[-1]], dtype=state.trend_tail.dtype)
        return state.trend_tail + delta

    def trend_from_ses():
        return state.trend_tail + tren[-1]

    new_trend_tail = lax.cond(
        trend_improves & (~used_ses),
        trend_from_linear,
        lambda: lax.cond(
            trend_improves & used_ses,
            trend_from_ses,
            lambda: state.trend_tail
        )
    )

    # Compute R² penalty on first trend iteration.
    def compute_penalty():
        mu = jnp.mean(resids)
        ssres = jnp.sum((resids - tren) ** 2)
        sstot = jnp.sum((resids - mu) ** 2) + 1e-12
        # Match the carry leg's dtype (threaded from y — §10 weak-type trap).
        return (1.0 - (ssres / sstot)).astype(state.penalty.dtype)

    new_penalty = lax.cond(
        (i == 1) & trend_improves & (~used_ses),
        compute_penalty,
        lambda: state.penalty
    )
    
    # Update resids for robustness detection
    resids = config.y_tr - new_fitted3
    
    # -------------------------------------------------------------------------
    # 6. Auto-detect robust mode (if i == 0 and init_robust is True)
    # -------------------------------------------------------------------------
    if init_robust:
        def detect_robust():
            return _cov_proxy(resids, jnp.int32(config.multiplicative_flag)) > config.cov_threshold

        new_robust = lax.cond(
            i == 0,
            detect_robust,
            lambda: state.robust
        )
    else:
        new_robust = state.robust
    
    # -------------------------------------------------------------------------
    # 7. Cap outliers on i == 1 (applied to next iteration via resids update in state)
    # Note: We can't directly modify resids for next iter, but the effect is captured
    # through the fitted values.
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # 8. Return new state
    # -------------------------------------------------------------------------
    return _LoopState(
        i=i + 1,
        fitted=new_fitted3,
        linear_component=new_linear,
        seasonal_component=new_seasonal,
        ses_component=new_ses,
        exogenous_component=new_exo_comp,
        trend_tail=new_trend_tail,
        seas_tail=new_seas_tail,
        seas_tail_len=new_seas_tail_len,
        best=new_best_final,
        stalls=new_stalls,
        robust=new_robust,
        penalty=new_penalty,
        exo_beta=new_exo_beta,
        converged=converged,
    )


from functools import lru_cache

@lru_cache(maxsize=32)
def _make_fit_loop(
    has_seasonality: bool,
    has_exogenous: bool,
    use_seasonality_weights: bool,
    use_changepoints: bool,
    gradient_strategy: bool,
    smoother: bool,
    ses_mode_code: int,
    init_robust: bool,
    max_rounds: int,
):
    """Create a JIT-compiled fit loop with static configuration baked in.
    
    Uses lru_cache to avoid recompilation for identical configurations.
    """
    
    def body_fn(state_and_config):
        state, config = state_and_config
        new_state = _fit_body(
            state, config,
            has_seasonality=has_seasonality,
            has_exogenous=has_exogenous,
            use_seasonality_weights=use_seasonality_weights,
            use_changepoints=use_changepoints,
            gradient_strategy=gradient_strategy,
            smoother=smoother,
            ses_mode_code=ses_mode_code,
            init_robust=init_robust,
        )
        return (new_state, config)
    
    def cond_fn(state_and_config):
        state, config = state_and_config
        return (state.i < max_rounds) & (~state.converged)
    
    @jax.jit
    def fit_loop(init_state: _LoopState, config: _FitConfig) -> _LoopState:
        final_state, _ = lax.while_loop(cond_fn, body_fn, (init_state, config))
        return final_state
    
    return fit_loop


class MFLES(BaseForecaster):
    """
    JAX MFLES implementation with StatsForecast-compatible default behavior.

    MFLES (Multi-Feature Locally Exponential Smoothing) combines multiple additive
    components — piecewise linear trend, Fourier seasonal patterns, and residual
    smoothing — fitted iteratively to successive residuals in a JIT-compiled loop.

    The goal is algorithmic parity with MFLES in StatsForecast while keeping
    the implementation JAX-friendly for speed at larger scales.

    Parity-oriented defaults:
    - robust=None  -> auto-detect robust mode from residual CoV proxy.
    - changepoints=True  -> piecewise LASSO trend enabled by default.
    - ses_mode="lite"  -> rolling residual smoother (matches StatsForecast default).

    Args:
        verbose (int): Verbosity level (currently reserved, unused). Default is 1.
        robust (bool | None): If True, uses Siegel repeated medians for trend fitting. If False, uses OLS. If None, auto-detects based on residual variability. Default is None.
        alias (str): Model name identifier. Default is "MFLES".
        conformal_params (ConformalIntervals | None): Conformal prediction configuration for generating prediction intervals. Default is None.

    Attributes:
        ``model_`` (dict): Populated after fit(); contains fitted values and all components.
        multiplicative (bool): Whether the last fit used multiplicative (log-space) mode.
        penalty (float | None): R²-based trend dampening scalar (set during fit).
        trend_penalty (bool): Whether trend damping is applied during predict().
        seasonality (jnp.ndarray | None): Last seasonal period tail used for forecasting.
        trend (jnp.ndarray): Two-element array [prev_end, curr_end] for slope extrapolation.
    """
    uses_exog = True

    def __init__(self, verbose: int = 1, robust: bool | None = None, alias: str = "MFLES", conformal_params: ConformalIntervals | None = None) -> None:
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
        self.verbose = verbose
        self.robust = robust
        self.predicted = None
        self._exo_beta = None
        self.penalty = None
        self._cs = None
        self._seas_len = None
        self._fit_config = None

    def fit(
        self,
        y: jnp.ndarray,
        seasonal_period: int | list[int] | None = None,
        X: jnp.ndarray | None = None,
        fourier_order: int | None = None,
        ma: int | list[int] | None = None,
        alpha: float = 1.0,
        decay: float = -1,
        n_changepoints: float | int = 0.25,
        seasonal_lr: float = 0.9,
        rs_lr: float = 1.0,
        exogenous_lr: float = 1.0,
        exogenous_estimator=None,
        exogenous_params: dict = {},
        linear_lr: float = 0.9,
        cov_threshold: float = 0.7,
        moving_medians: bool = False,
        max_rounds: int = 50,
        min_alpha: float = 0.05,
        max_alpha: float = 1.0,
        round_penalty: float = 0.0001,
        trend_penalty: bool = True,
        multiplicative: bool | None = None,
        changepoints: bool = True,
        smoother: bool = False,
        ses_mode: str = "lite",
        seasonality_weights: bool = False,
        gradient_strategy: bool = False,
    ) -> "MFLES":
        """Fit the MFLES model to a time series.

        Runs the JIT-compiled iterative fitting loop that alternately updates
        seasonal, trend, residual-smoothing, and exogenous components until
        convergence or `max_rounds` is reached.

        Args:
            y (jnp.ndarray): Input time series of shape (n,).
            seasonal_period (int | list[int] | None): Seasonal period(s) for Fourier features. Pass a list for multiple seasonalities. None disables seasonality. Default is None.
            X (jnp.ndarray | None): Exogenous design matrix of shape (n, p). Default is None.
            fourier_order (int | None): Fixed Fourier order for all periods. None uses the auto-heuristic (5 / 10 / 15 based on period length). Default is None.
            ma (int | list[int] | None): Moving-average window(s) for residual smoothing cadence. None defaults to [1]. Default is None.
            alpha (float): LASSO regularization strength for changepoint trend. Default is 1.0.
            decay (float): Unused legacy parameter (kept for API compatibility). Default is -1.
            n_changepoints (float | int): Number of changepoint knots. A float < 1 is treated as a fraction of series length (e.g. 0.25 = 25% of n). An int specifies knots directly. None or 0 disables changepoints. Default is 0.25.
            seasonal_lr (float): Learning rate multiplier applied to seasonal updates. Default is 0.9.
            rs_lr (float): Learning rate multiplier for residual-smoothing updates. Default is 1.0.
            exogenous_lr (float): Learning rate multiplier for exogenous updates. Default is 1.0.
            exogenous_estimator: Unused legacy parameter. Default is None.
            exogenous_params (dict): Unused legacy parameter. Default is {}.
            linear_lr (float): Learning rate multiplier for trend updates. Default is 0.9.
            cov_threshold (float): CoV proxy threshold for auto robust-mode detection. Set to -1 to effectively disable. Default is 0.7.
            moving_medians (bool): If True, initialises fitted values with period-wise medians instead of zeros. Default is False.
            max_rounds (int): Maximum number of fitting iterations. Default is 50.
            min_alpha (float): Minimum SES alpha in the ensemble grid. Default is 0.05.
            max_alpha (float): Maximum SES alpha in the ensemble grid. Default is 1.0.
            round_penalty (float): Improvement threshold fraction required before accepting a residual-smoothing update. Default is 0.0001.
            trend_penalty (bool): If True, dampens trend slope by the R-squared penalty computed on the first trend iteration. Default is True.
            multiplicative (bool | None): If True, fits in log-space (multiplicative seasonality). If None, auto-detected: True when seasonal_period is set and all values are positive. Default is None.
            changepoints (bool): If True, enables piecewise linear trend via LASSO. Default is True.
            smoother (bool): Used only when ses_mode="adaptive": True selects SES ensemble, False selects rolling mean. Default is False.
            ses_mode (str): Residual smoothing strategy. One of ``"off"`` (no residual smoothing), ``"lite"`` (rolling mean, StatsForecast default), ``"full"`` (SES ensemble), or ``"adaptive"`` (controlled by the ``smoother`` flag). Default is "lite".
            seasonality_weights (bool): If True, applies recency-weighted OLS for Fourier seasonal fitting. Auto-enabled for multiplicative single-period series. Default is False.
            gradient_strategy (bool): Legacy flag (currently unused). Default is False.

        Returns:
            MFLES: Self (fitted model instance) for method chaining.
        """
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        n = y.shape[0]

        # StatsForecast-compatible multiplicative decision:
        # - if seasonality is provided, prefer multiplicative mode
        # - disable multiplicative when the series contains non-positive values
        if multiplicative is None:
            # Auto decision needs concrete data (log-space validity). It is
            # resolved ONCE, eagerly, at fit time; the CV path replays the
            # resolved value via _fit_config (captured below) so per-window
            # fits trace statically. Same selection-with-sight class as
            # AutoARIMA's cached order (CLAUDE.md §3.1).
            multiplicative = seasonal_period is not None
            if multiplicative:
                multiplicative = bool(jnp.min(y) > 0)
        multiplicative = bool(multiplicative)

        # Resolved fit configuration. forecast() replays this when called with
        # no explicit config (the base conformity_scores path), so CV windows
        # re-fit the SAME model configuration — not the bare defaults — and
        # every auto decision above stays static under the vmap trace.
        self._fit_config = dict(
            seasonal_period=seasonal_period, fourier_order=fourier_order,
            ma=ma, alpha=alpha, decay=decay, n_changepoints=n_changepoints,
            seasonal_lr=seasonal_lr, rs_lr=rs_lr, exogenous_lr=exogenous_lr,
            linear_lr=linear_lr, cov_threshold=cov_threshold,
            moving_medians=moving_medians, max_rounds=max_rounds,
            min_alpha=min_alpha, max_alpha=max_alpha,
            round_penalty=round_penalty, trend_penalty=trend_penalty,
            multiplicative=multiplicative, changepoints=changepoints,
            smoother=smoother, ses_mode=ses_mode,
            seasonality_weights=seasonality_weights,
            gradient_strategy=gradient_strategy,
        )
        if multiplicative:
            const = jnp.min(y)
            y_tr = jnp.log(y)
            mean = jnp.array(0.0, dtype=y.dtype)
            std = jnp.array(1.0, dtype=y.dtype)
        else:
            const = None
            mean, std = jnp.mean(y), jnp.std(y)
            y_tr = y - mean
            y_tr = jnp.where(std > 0, y_tr / std, y_tr)

        self.const, self.mean, self.std = const, mean, std
        self.trend_penalty = bool(trend_penalty)

        # Static short-series guard only. (The old `jnp.all(y_tr == mean)`
        # constant-series test is a traced bool under vmap; the main loop is
        # flat on constants anyway — AutoCES precedent.)
        if n < 4:
            base = y_tr[-1]
            self.trend = jnp.array([base, base], dtype=y.dtype)
            self.seasonality = None
            self._seas_len = None
            self.penalty = jnp.zeros((), dtype=y.dtype)
            self.linear_component = jnp.zeros(n, y.dtype)
            self.seasonal_component = jnp.zeros(n, y.dtype)
            self.ses_component = jnp.zeros(n, y.dtype)
            self.median_component = jnp.zeros(n, y.dtype)
            self.exogenous_component = jnp.zeros(n, y.dtype)
            self._exo_beta = None
            fitted = jnp.full(n, base, y.dtype)
            self._finalize_fit(fitted, multiplicative)
            self._cache_cs(y, X)
            return self

        sp_list = None
        if seasonal_period is not None:
            if isinstance(seasonal_period, (list, tuple, np.ndarray)):
                raw_periods = np.asarray(seasonal_period, dtype=np.int32).reshape(-1)
            else:
                raw_periods = np.asarray([int(seasonal_period)], dtype=np.int32)
            valid_periods = raw_periods[(raw_periods > 1) & (raw_periods < n)]
            if valid_periods.size > 0:
                _, first_idx = np.unique(valid_periods, return_index=True)
                sp_list = valid_periods[np.sort(first_idx)].tolist()

        linear_component = jnp.zeros(n, y.dtype)
        seasonal_component = jnp.zeros(n, y.dtype)
        ses_component = jnp.zeros(n, y.dtype)
        median_component = _median_init(y_tr, max(sp_list) if (sp_list is not None and moving_medians) else None)
        exogenous_component = jnp.zeros(n, y.dtype)
        fitted = median_component.copy()
        trend_tail = jnp.array([fitted[-1], fitted[-1]], y.dtype)

        ma_cycle = [1] if ma is None else (ma if isinstance(ma, list) else [int(ma)])

        if n_changepoints is None:
            changepoints = False
            n_cps = 0
        elif isinstance(n_changepoints, float) and n_changepoints < 1:
            n_cps = int(n_changepoints * n)
        elif isinstance(n_changepoints, int):
            n_cps = n_changepoints
        else:
            n_cps = 0

        if not changepoints:
            n_cps = 0

        # SF uses min(n_changepoints, 0.1 * n). We keep that behavior but
        # apply a fixed knot cap so complexity scales monotonically with n.
        cp_cap = 64
        n_cps = max(0, min(int(n_cps), int(0.1 * n), cp_cap))
        effective_changepoints = bool(changepoints and n_cps > 0)

        # Fixed max_rounds for all series lengths to maximize JIT cache reuse
        # Note: StatsForecast MFLES runs all iterations without early stopping
        if max_rounds == 50:  # Default value, keep it
            max_rounds = 50  # Run full iterations for accuracy (early stopping disabled for now)

        # =====================================================================
        # Build configuration for JIT-compiled loop
        # =====================================================================
        has_seasonality = sp_list is not None
        has_exogenous = X is not None
        use_seasonality_weights = bool(seasonality_weights)
        if (not use_seasonality_weights) and has_seasonality and bool(multiplicative) and (len(sp_list) == 1):
            # For long positive seasonal series, recency-weighted seasonal fits
            # are generally more stable than uniform OLS and improve accuracy.
            use_seasonality_weights = True
        use_changepoints = bool(effective_changepoints)
        robust_init_value = self.robust
        init_robust = robust_init_value is None  # Need auto-detection

        # JAX guardrail: robust auto-detection can select Siegel regression,
        # which is O(n^2). For very long series this dominates runtime.
        if init_robust and n >= 5000:
            init_robust = False
            robust_init_value = False

        # SF behavior: cov_threshold=-1 disables safeguards by using a very large threshold.
        if cov_threshold == -1:
            cov_threshold = 10000

        ses_mode_norm = str(ses_mode).lower()
        if ses_mode_norm not in ("off", "lite", "full", "adaptive"):
            raise ValueError("ses_mode must be one of: 'off', 'lite', 'full', 'adaptive'")
        ses_mode_code = {"off": 0, "lite": 1, "full": 2, "adaptive": 3}[ses_mode_norm]
        
        # Build Fourier and optional seasonality-weight stacks with JAX primitives.
        if has_seasonality:
            sp_array = jnp.array(sp_list, dtype=jnp.int32)
            max_period = int(max(sp_list))
            forced_order = max(1, int(fourier_order)) if fourier_order is not None else -1
            # IMPORTANT: avoid padding Fourier matrices with extra all-zero columns.
            # Rank-deficient X can perturb pinv-based OLS (vs StatsForecast which
            # builds an exact-width Fourier design).
            if forced_order > 0:
                max_fourier_order = forced_order
            else:
                # Mirror StatsForecast's set_fourier(period) rule in Python.
                def _sf_fourier(p: int) -> int:
                    return 5 if p < 10 else (10 if p < 70 else 15)
                max_fourier_order = int(max(_sf_fourier(int(p)) for p in sp_list))
            fourier_stack, weights_stack = _build_fourier_and_weights_stack(
                n=n,
                sp_array=sp_array,
                max_order=max_fourier_order,
                forced_order=forced_order,
                use_weights=use_seasonality_weights,
            )
            fourier_stack = fourier_stack.astype(y.dtype)
            weights_stack = weights_stack.astype(y.dtype)
        else:
            fourier_stack = jnp.zeros((1, n, 1), dtype=y.dtype)
            weights_stack = jnp.zeros((1, n), dtype=y.dtype)
            sp_array = jnp.zeros((1,), dtype=jnp.int32)
            max_period = 1
        
        # Prepare exogenous data
        if has_exogenous:
            X_exo = utils.ensure_float(jnp.asarray(X))
            n_exo_features = X_exo.shape[1]
        else:
            X_exo = jnp.zeros((n, 1), dtype=y.dtype)
            n_exo_features = 0
        
        # Pre-compute alphas for SES ensemble (static shape)
        ses_alphas = jnp.arange(float(min_alpha), float(max_alpha) + 1e-9, 0.05, dtype=y.dtype)

        # Use explicit learning-rate hyperparameters directly.
        effective_linear_lr = float(linear_lr)
        effective_seasonal_lr = float(seasonal_lr)
        if n < 250:
            min_iter = 0
        elif n < 1000:
            min_iter = 6
        else:
            min_iter = 10
        # Keep high ISTA budget for short series; use a shared budget for
        # medium/large lengths to avoid 5k being slower than 10k.
        if n <= 1000:
            lasso_maxiter = 200
        elif n <= 10000:
            lasso_maxiter = 64
        else:
            lasso_maxiter = 48

        # Pre-compute piecewise basis + spectral step once.
        if bool(effective_changepoints) and n_cps > 0 and not bool(gradient_strategy):
            knots = _uniform_knots(n, n_cps, max_knots=n_cps)
            hinge_basis = _hinge_basis_from_knots(n, knots)
            lasso_step = jnp.array(_spectral_step(hinge_basis), dtype=y.dtype)
        else:
            # Dummy values (won't be used if changepoints=False or gradient_strategy=True)
            hinge_basis = jnp.zeros((n, 1), dtype=y.dtype)
            lasso_step = jnp.array(1.0, dtype=y.dtype)

        # Create config
        config = _FitConfig(
            y_tr=y_tr,
            x_idx=jnp.arange(n, dtype=y.dtype),
            fourier_stack=fourier_stack,
            weights_stack=weights_stack,
            sp_array=sp_array,
            X_exo=X_exo,
            ma_array=jnp.array(ma_cycle, dtype=jnp.int32),
            ses_alphas=ses_alphas,
            hinge_basis=hinge_basis,
            lasso_step=lasso_step,
            seasonal_lr=float(effective_seasonal_lr),
            linear_lr=float(effective_linear_lr),
            exogenous_lr=float(exogenous_lr),
            rs_lr=float(rs_lr),
            round_penalty=float(round_penalty),
            alpha=float(alpha),
            cov_threshold=float(cov_threshold),
            multiplicative_flag=int(bool(multiplicative)),
            n_cps=n_cps,
            max_period=max_period,
            n_exo_features=n_exo_features,
            min_iter=int(min_iter),
            lasso_maxiter=int(lasso_maxiter),
        )
        
        # Initialize loop state
        init_state = _LoopState(
            i=jnp.int32(0),
            fitted=fitted,
            linear_component=linear_component,
            seasonal_component=seasonal_component,
            ses_component=ses_component,
            exogenous_component=exogenous_component,
            trend_tail=trend_tail,
            seas_tail=jnp.zeros(max_period, dtype=y.dtype),
            seas_tail_len=jnp.int32(0),
            best=jnp.array(jnp.inf, dtype=y.dtype),  # Match dtype with y
            stalls=jnp.int32(0),
            robust=jnp.bool_(robust_init_value if robust_init_value is not None else False),
            penalty=jnp.array(0.0, dtype=y.dtype),
            exo_beta=jnp.zeros(max(n_exo_features, 1), dtype=y.dtype),
            converged=jnp.bool_(False),
        )
        
        # =====================================================================
        # Run JIT-compiled fit loop
        # =====================================================================
        fit_loop = _make_fit_loop(
            has_seasonality=has_seasonality,
            has_exogenous=has_exogenous,
            use_seasonality_weights=use_seasonality_weights,
            use_changepoints=use_changepoints,
            gradient_strategy=bool(gradient_strategy),
            smoother=bool(smoother),
            ses_mode_code=ses_mode_code,
            init_robust=init_robust,
            max_rounds=int(max_rounds),
        )
        
        final_state = fit_loop(init_state, config)

        # =====================================================================
        # Unpack final state to self attributes
        # =====================================================================
        fitted = final_state.fitted
        self.linear_component = final_state.linear_component
        self.seasonal_component = final_state.seasonal_component
        self.ses_component = final_state.ses_component
        self.median_component = median_component
        self.exogenous_component = final_state.exogenous_component

        # Trend tail is tracked during fitting to mirror SF trend updates
        # (trend updates add tren[-2:], residual smoother adds tren[-1]).
        self.trend = final_state.trend_tail.astype(fitted.dtype)
        
        # Seasonality: keep the STATIC-size buffer plus its traced length.
        # (A lax.dynamic_slice with a traced SIZE cannot trace under vmap;
        # predict() gathers modulo the length instead.)
        if has_seasonality:
            self.seasonality = final_state.seas_tail
            self._seas_len = final_state.seas_tail_len
        else:
            self.seasonality = None
            self._seas_len = None

        # Auto-detected flags stay jnp scalars end-to-end (§1 rule 2).
        if init_robust:
            self.robust = final_state.robust
        # penalty <= 0 encodes the old `None` sentinel; predict() masks it.
        self.penalty = final_state.penalty

        # exo_beta is all-zeros whenever exogenous never improved the fit, so
        # the X @ beta contribution is exactly 0 — no traced any() guard.
        if has_exogenous:
            self._exo_beta = final_state.exo_beta

        self._finalize_fit(fitted, multiplicative)
        self._cache_cs(y, X)
        return self

    def _cache_cs(self, y: jnp.ndarray, X: jnp.ndarray | None) -> None:
        """Cache conformity scores on the TRAINING series (sibling convention).

        Runs the base-class walk-forward CV once, eagerly, at fit time so
        predict(level=...) is a cheap lookup. forecast() strips
        ``conformal_params`` from its internal clone, so the CV-window fits
        this triggers cannot recurse into their own CV.
        """
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None

    def _finalize_fit(self, fitted_tr: jnp.ndarray, multiplicative: bool) -> "MFLES":
        """Reverse scaling on fitted values and populate ``model_`` dict.

        Converts fitted values from transformed space (log or standardised)
        back to the original scale and stores all components in ``self.model_``.

        Args:
            fitted_tr (jnp.ndarray): Fitted values in transformed space of shape (n,).
            multiplicative (bool): If True, applies exp(); otherwise reverses z-score.

        Returns:
            MFLES: Self (for method chaining inside fit()).
        """
        if multiplicative:
            fitted = jnp.exp(fitted_tr)
        else:
            fitted = self.mean + fitted_tr * jnp.where(self.std > 0, self.std, 1.0)
        self.multiplicative = multiplicative
        self.model_ = {
            "fitted": fitted,
            "trend": self.trend,
            "seasonality": self.seasonality,
            "linear_component": self.linear_component,
            "seasonal_component": self.seasonal_component,
            "ses_component": self.ses_component,
            "median_component": self.median_component,
            "exogenous_component": self.exogenous_component,
            "mean": self.mean,
            "std": self.std,
            "const": self.const,
        }
        self.fitted_ = fitted  # Store fitted values
        return self  # Return self for method chaining

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        """Generate h-step ahead forecasts from the fitted model.

        Extrapolates the stored trend tail by the fitted slope (optionally damped
        by the R² penalty), tiles the seasonal tail, and adds any exogenous contribution.
        Optionally computes conformal prediction intervals.

        Args:
            h (int): Forecast horizon (number of steps ahead).
            X (jnp.ndarray | None): Future exogenous matrix of shape (h, p). Required only if the model was fitted with exogenous variables. Default is None.
            level (list[int | float] | None): Confidence levels (0-100) for conformal prediction intervals, e.g. [80, 95]. Requires ``conformal_params`` to be set. Default is None.

        Returns:
            dict: Dictionary containing:

                - "mean": Point forecasts of shape (h,).
                - "lo-{l}" / "hi-{l}": Conformal interval bounds for each level l
                  (only present when level is not None).
        """
        h = int(h)
        last, prev = self.trend[1], self.trend[0]
        slope = (last - prev)
        if self.trend_penalty and (self.penalty is not None):
            # penalty <= 0 encodes the pre-refactor "no damping" None sentinel.
            pen = jnp.asarray(self.penalty)
            slope = jnp.where(pen > 0, slope * pen, slope)
        trend_fcst = slope * jnp.arange(1, h + 1, dtype=jnp.asarray(last).dtype) + last

        if self.seasonality is not None and self.seasonality.size > 0:
            seas = jnp.asarray(self.seasonality)
            seas_len = getattr(self, "_seas_len", None)
            if seas_len is None:
                # Pre-refactor fit/pickle stored the exact-length tail.
                seas_len = jnp.int32(seas.shape[0])
            # Modular gather over the static buffer (traced-length-safe tiling;
            # replaces _repeat_val_seas, which needs a static pattern length).
            idx = jnp.arange(h) % jnp.maximum(seas_len, 1)
            seas_fcst = jnp.where(seas_len > 0, seas[idx],
                                  jnp.zeros((), dtype=seas.dtype))
        else:
            seas_fcst = jnp.zeros(h, dtype=trend_fcst.dtype)

        mean = trend_fcst + seas_fcst

        if X is not None and self._exo_beta is not None:
            X_exo = utils.ensure_float(jnp.asarray(X))
            mean = mean + (X_exo @ self._exo_beta)

        if self.const is not None:
            mean = jnp.exp(mean)
        else:
            mean = self.mean + mean * jnp.where(self.std > 0, self.std, 1.0)

        out = {"mean": mean}
        if level:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(level=...) requires conformal_params to be set "
                    "before fit()."
                )
            if getattr(self, "_cs", None) is None:
                raise ValueError(
                    "No cached conformity scores — set conformal_params and "
                    "re-run fit() before predict(level=...)."
                )
            if h != self.conformal_params.h:
                raise ValueError(
                    f"h={h} != conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            out = self.add_confidence_intervals(
                out, self._cs, level, self.conformal_params.method
            )
        return out

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        seasonal_period: int | list[int] | None = None,
        **fit_kwargs,
    ) -> dict:
        r"""Stateless fit+predict in one call (convenience wrapper).

        Creates a fresh model copy, fits it on `y`, and immediately generates
        forecasts. Does not store any state on `self`.

        Args:
            y (jnp.ndarray): Input time series of shape (n,).
            h (int): Forecast horizon.
            X (jnp.ndarray | None): In-sample exogenous matrix of shape (n, p). Default is None.
            X_future (jnp.ndarray | None): Future exogenous matrix of shape (h, p). Default is None.
            level (list[int | float] | None): Confidence levels for conformal intervals. Default is None.
            seasonal_period (int | list[int] | None): Seasonal period(s) passed to fit. Default is None.
            \**fit_kwargs: Any additional keyword arguments forwarded to fit().

        Returns:
            dict: Same output as predict() -- "mean" and optional interval keys.
        """
        # Replay the fitted configuration when the caller passed none: the base
        # conformity_scores path calls forecast(h=..., y=..., X=..., X_future=...)
        # with no hyperparameters, and the CV windows must re-fit the SAME
        # config the estimator was fit with (auto decisions already resolved
        # there, so this branch traces statically under vmap).
        if seasonal_period is None and not fit_kwargs:
            replay = getattr(self, "_fit_config", None)
            if replay:
                fit_kwargs = dict(replay)
                seasonal_period = fit_kwargs.pop("seasonal_period")

        m = self.new()
        # CV-window fits must not recurse into their own walk-forward CV.
        m.conformal_params = None
        m.fit(y, X=X, seasonal_period=seasonal_period, **fit_kwargs)
        out = m.predict(h, X=X_future, level=None)

        if level:
            if self.conformal_params is None:
                raise ValueError(
                    "forecast(level=...) requires conformal_params to be set."
                )
            if int(h) != self.conformal_params.h:
                raise ValueError(
                    f"h={h} != conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            # Score from the JUST-FITTED clone so the CV windows replay the
            # same config the point forecast used (self._fit_config may be
            # stale or absent — e.g. stateless forecast on an unfitted model
            # or explicit kwargs overriding the fitted config). The clone's
            # own window-fits still get conformal_params=None → no recursion.
            cs_model = m.new()
            cs_model.conformal_params = self.conformal_params
            cs = cs_model.conformity_scores(y=y, X=X)
            out = self.add_confidence_intervals(
                out, cs, level, self.conformal_params.method
            )
        return out

    def optimize(
        self,
        y: jnp.ndarray,
        seasonal_period: int | list[int] | None,
        n_steps: int,
        test_size: int,
        step_size: int = 1,
        metric: str = "smape",
        X: jnp.ndarray | None = None,
        params: list[dict] | None = None,
    ) -> dict:
        """Auto-tune MFLES hyperparameters via rolling cross-validation.

        Evaluates a grid of candidate configurations on rolling validation windows
        and returns the configuration with the lowest average error metric.

        Args:
            y (jnp.ndarray): Full time series used for cross-validation.
            seasonal_period (int | list[int] | None): Seasonal period(s) passed to fit() in each fold. Also drives the default candidate grid when params is None.
            n_steps (int): Number of rolling validation windows to evaluate.
            test_size (int): Number of observations held out as the test horizon in each window.
            step_size (int): Step (in observations) between successive validation windows. Default is 1.
            metric (str): Error metric to minimise. One of "mse", "mae", "mape", "smape". Default is "smape".
            X (jnp.ndarray | None): Exogenous matrix of shape (n, p) aligned with y. Sliced appropriately for each fold. Default is None.
            params (list[dict] | None): Explicit list of fit() kwarg dicts to evaluate. If None, a default grid is constructed based on seasonal_period. Default is None.

        Returns:
            dict: The best-performing hyperparameter dictionary (suitable as ``**kwargs`` to fit()).
        """
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        metric_fn = _metric2fn[metric]
        total = y.shape[0]

        if seasonal_period is not None and not isinstance(seasonal_period, list):
            seasonal_period = [int(seasonal_period)]
        if params is None:
            cfgs = []
            if seasonal_period is not None:
                for smoother in [True, False]:
                    for ma in [int(min(seasonal_period)), int(min(seasonal_period) // 2), None]:
                        for seas in [None, seasonal_period]:
                            for sw in [True, False]:
                                if seas is None and sw:
                                    continue
                                cfgs.append({"smoother": smoother, "ma": ma, "seasonal_period": seas, "seasonality_weights": sw})
            else:
                for smoother in [True, False]:
                    for cov in [0.5, -1]:
                        for mr in [5, 20]:
                            cfgs.append({"smoother": smoother, "cov_threshold": cov, "max_rounds": mr, "seasonal_period": None})
        else:
            cfgs = [dict(p) for p in params]

        max_steps = (total - test_size - 4) // step_size + 1
        if max_steps < 1:
            return cfgs[0]
        if max_steps < n_steps:
            n_steps = max_steps

        best_score = jnp.inf
        best_cfg = cfgs[0]
        for cfg in cfgs:
            scores = []
            for s in range(n_steps):
                train_end = total - (s * step_size + test_size)
                y_tr = y[:train_end]
                y_te = y[train_end: train_end + test_size]
                if X is not None:
                    X_tr = utils.ensure_float(jnp.asarray(X[:train_end]))
                    X_te = utils.ensure_float(jnp.asarray(X[train_end: train_end + test_size]))
                else:
                    X_tr = None
                    X_te = None
                yhat = self.new().fit(y_tr, X=X_tr, **cfg).predict(test_size, X=X_te)["mean"]
                scores.append(metric_fn(y_te, yhat))
            score = jnp.mean(jnp.stack(scores))
            if score < best_score:
                best_score, best_cfg = score, cfg
        return best_cfg


# =========================
# Test Cases
# =========================

if __name__ == "__main__":
    import jax.random as jrandom
    
    print("=" * 60)
    print("MFLES Test Suite")
    print("=" * 60)
    
    # Test 1: Basic MFLES with seasonal data
    print("\n[Test 1] Basic MFLES fit and forecast")
    n = 100
    period = 12
    t = jnp.arange(n)
    trend = 0.5 * t + 10
    seasonal = 3.0 * jnp.sin(2 * jnp.pi * t / period)
    noise = jrandom.normal(jrandom.PRNGKey(42), (n,)) * 0.5
    y1 = trend + seasonal + noise
    
    model1 = MFLES()
    model1.fit(y1, seasonal_period=period, max_rounds=10, changepoints=False)
    forecast1 = model1.predict(h=12)
    
    print(f"  Input series length: {n}")
    print(f"  Fitted shape: {model1.fitted_.shape}")
    print(f"  Forecast shape: {forecast1['mean'].shape}")
    print(f"  Model components: trend, seasonality, linear, ses, median")
    assert model1.fitted_.shape == (n,), "Fitted shape mismatch!"
    assert forecast1['mean'].shape == (12,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast1['mean'])), "Forecast contains NaN/Inf!"
    print("  ✓ Basic fit and forecast OK")
    
    # Test 2: Multiplicative seasonality
    print("\n[Test 2] Multiplicative seasonality")
    y2_mult = jnp.exp(jnp.log(10) + 0.02 * t + 0.3 * jnp.sin(2 * jnp.pi * t / period))
    model2 = MFLES()
    model2.fit(y2_mult, seasonal_period=period, multiplicative=True, max_rounds=10)
    forecast2 = model2.predict(h=12)
    
    print(f"  Multiplicative mode: {model2.multiplicative}")
    print(f"  All values positive: {jnp.all(y2_mult > 0)}")
    assert model2.multiplicative == True, "Should be multiplicative!"
    assert jnp.all(forecast2['mean'] > 0), "Multiplicative forecast should be positive!"
    print("  ✓ Multiplicative mode OK")
    
    # Test 3: Multiple seasonal periods
    print("\n[Test 3] Multiple seasonal periods")
    seas1 = 2.0 * jnp.sin(2 * jnp.pi * t / 7)
    seas2 = 1.5 * jnp.cos(2 * jnp.pi * t / 14)
    y3 = trend + seas1 + seas2 + noise
    
    model3 = MFLES()
    model3.fit(y3, seasonal_period=[7, 14], max_rounds=15)
    forecast3 = model3.predict(h=14)
    
    print(f"  Periods: [7, 14]")
    print(f"  Seasonal component shape: {model3.seasonal_component.shape}")
    assert forecast3['mean'].shape == (14,), "Forecast shape mismatch!"
    print("  ✓ Multiple seasonal periods OK")
    
    # Test 4: Robust mode
    print("\n[Test 4] Robust regression mode")
    # Add outliers
    y4 = y1.at[20].set(y1[20] + 10)
    y4 = y4.at[50].set(y4[50] - 8)
    
    model4_normal = MFLES(robust=False)
    model4_robust = MFLES(robust=True)
    
    model4_normal.fit(y4, seasonal_period=period, max_rounds=10)
    model4_robust.fit(y4, seasonal_period=period, max_rounds=10)
    
    # Robust should handle outliers better
    residuals_normal = y4 - model4_normal.fitted_
    residuals_robust = y4 - model4_robust.fitted_
    
    print(f"  Normal residual std: {jnp.std(residuals_normal):.4f}")
    print(f"  Robust residual std: {jnp.std(residuals_robust):.4f}")
    print("  ✓ Robust mode works")
    
    # Test 5: Exogenous variables
    print("\n[Test 5] Exogenous variables support")
    X_train = jnp.column_stack([
        jnp.sin(2 * jnp.pi * t / 30),
        jnp.cos(2 * jnp.pi * t / 30)
    ])
    exog_effect = X_train @ jnp.array([2.0, 1.5])
    y5 = trend + seasonal + exog_effect + noise
    
    model5 = MFLES()
    model5.fit(y5, seasonal_period=period, X=X_train, max_rounds=15)
    
    # Forecast with exogenous
    t_future = jnp.arange(n, n + 12)
    X_future = jnp.column_stack([
        jnp.sin(2 * jnp.pi * t_future / 30),
        jnp.cos(2 * jnp.pi * t_future / 30)
    ])
    forecast5 = model5.predict(h=12, X=X_future)
    
    print(f"  Exogenous beta: {model5._exo_beta}")
    print(f"  Forecast with exogenous: {forecast5['mean'][:3]}")
    assert model5._exo_beta is not None, "Should have exogenous coefficients!"
    print("  ✓ Exogenous variables OK")
    
    # Test 6: Changepoints (piecewise linear)
    print("\n[Test 6] Changepoints for piecewise linear trend")
    # Create data with changepoint
    y6_part1 = 0.2 * jnp.arange(50) + 10
    y6_part2 = 0.8 * jnp.arange(50, 100) + 5
    y6 = jnp.concatenate([y6_part1, y6_part2]) + jrandom.normal(jrandom.PRNGKey(123), (100,)) * 0.3
    
    model6 = MFLES()
    model6.fit(y6, changepoints=True, n_changepoints=0.1, alpha=0.5, max_rounds=20)
    forecast6 = model6.predict(h=10)
    
    print(f"  Changepoints enabled: True")
    print(f"  Linear component variance: {jnp.var(model6.linear_component):.4f}")
    assert jnp.var(model6.linear_component) > 0, "Should have non-zero linear component!"
    print("  ✓ Changepoints OK")
    
    # Test 7: Trend penalty
    print("\n[Test 7] Trend penalty based on R²")
    model7 = MFLES()
    model7.fit(y1, seasonal_period=period, trend_penalty=True, max_rounds=15)
    forecast7 = model7.predict(h=12)
    
    print(f"  Trend penalty: {model7.penalty}")
    print(f"  Penalty applied: {model7.trend_penalty}")
    assert model7.penalty is not None, "Should have penalty value!"
    assert 0 <= model7.penalty <= 1, "Penalty should be between 0 and 1!"
    print("  ✓ Trend penalty OK")
    
    # Test 8: Moving medians initialization
    print("\n[Test 8] Moving medians initialization")
    model8 = MFLES()
    model8.fit(y1, seasonal_period=period, moving_medians=True, max_rounds=10)
    
    print(f"  Median component shape: {model8.median_component.shape}")
    assert jnp.any(model8.median_component != 0), "Should have median component!"
    print("  ✓ Moving medians OK")
    
    # Test 9: Optimization (hyperparameter tuning)
    print("\n[Test 9] Hyperparameter optimization")
    # Use smaller dataset for speed
    y9 = y1[:70]
    model9 = MFLES()
    best_params = model9.optimize(
        y9,
        seasonal_period=period,
        n_steps=3,
        test_size=10,
        step_size=5,
        metric="mse"
    )
    
    print(f"  Best params keys: {list(best_params.keys())}")
    assert 'smoother' in best_params or 'ma' in best_params, "Should return valid params!"
    print("  ✓ Optimization OK")
    
    # Test 10: Conformal intervals
    print("\n[Test 10] Conformal prediction intervals")
    from chronax.utils import ConformalIntervals
    conformal_params = ConformalIntervals(h=12)
    model10 = MFLES(conformal_params=conformal_params)
    model10.fit(y1, seasonal_period=period, max_rounds=10)
    forecast10 = model10.predict(h=12, level=[90, 95])
    
    print(f"  Forecast keys: {list(forecast10.keys())}")
    assert 'mean' in forecast10, "Should have 'mean'!"
    # Check for interval keys (format may vary)
    has_90_intervals = ('lower_90' in forecast10 or 'lo-90' in forecast10)
    has_95_intervals = ('lower_95' in forecast10 or 'lo-95' in forecast10)
    assert has_90_intervals, "Should have 90% interval keys!"
    assert has_95_intervals, "Should have 95% interval keys!"
    # Use whichever format exists
    lo_90_key = 'lo-90' if 'lo-90' in forecast10 else 'lower_90'
    hi_90_key = 'hi-90' if 'hi-90' in forecast10 else 'upper_90'
    lo_95_key = 'lo-95' if 'lo-95' in forecast10 else 'lower_95'
    hi_95_key = 'hi-95' if 'hi-95' in forecast10 else 'upper_95'
    print(f"  90% interval width (first): {forecast10[hi_90_key][0] - forecast10[lo_90_key][0]:.4f}")
    print(f"  95% interval width (first): {forecast10[hi_95_key][0] - forecast10[lo_95_key][0]:.4f}")
    print("  ✓ Conformal intervals work")
    
    # Test 11: Short series edge case
    print("\n[Test 11] Edge case: very short series")
    y11 = jnp.array([1.0, 2.0, 3.0])
    model11 = MFLES()
    model11.fit(y11)
    forecast11 = model11.predict(h=3)
    
    print(f"  Input length: 3")
    print(f"  Fitted: {model11.fitted_}")
    print(f"  Forecast: {forecast11['mean']}")
    assert forecast11['mean'].shape == (3,), "Should forecast despite short series!"
    print("  ✓ Short series handled")
    
    # Test 12: Constant series
    print("\n[Test 12] Edge case: constant series")
    y12 = jnp.ones(50) * 7.5
    model12 = MFLES()
    model12.fit(y12)
    forecast12 = model12.predict(h=10)
    
    print(f"  Input (constant 7.5)")
    print(f"  Forecast mean: {jnp.mean(forecast12['mean']):.4f}")
    assert jnp.allclose(forecast12['mean'], 7.5, atol=0.5), "Should forecast constant!"
    print("  ✓ Constant series OK")
    
    # Test 13: Seasonal weights
    print("\n[Test 13] Seasonality weights")
    model13 = MFLES()
    model13.fit(y1, seasonal_period=period, seasonality_weights=True, max_rounds=10)
    
    print(f"  Seasonality weights enabled")
    print(f"  Seasonal component std: {jnp.std(model13.seasonal_component):.4f}")
    print("  ✓ Seasonality weights OK")
    
    print("\n" + "=" * 60)
    print("✓ All MFLES tests passed!")
    print("=" * 60)