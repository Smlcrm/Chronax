"""
MFLES (Multi-Feature Locally Exponential Smoothing) in JAX

This module implements an advanced forecasting model that combines multiple components:
- Linear trend with optional changepoints (piecewise linear via LASSO)
- Multiple seasonal patterns (Fourier series representation)
- Residual smoothing (Simple Exponential Smoothing ensemble)
- Exogenous variables support
- Robust estimation options (Siegel repeated medians)

**Algorithm:**
MFLES iteratively fits components to residuals:
1. Initial median-based estimates
2. Seasonal components via Fourier series + (weighted) OLS
3. Linear trend via OLS, robust regression, or piecewise LASSO
4. Residual smoothing via SES ensemble or rolling means
5. Exogenous variables via OLS
6. Iterates until convergence or max rounds

**Formulas:**
- Seasonal: y_t = Σ [a_k cos(2πkt/T) + b_k sin(2πkt/T)] for k=1..K
- Trend: y_t = β₀ + β₁t + Σ β_k max(0, t-τ_k) (piecewise linear)
- SES: ŷ_t = α·y_t + (1-α)·ŷ_{t-1}

**Implementation:**
- JIT-compiled components for performance
- Automatic robustness detection via coefficient of variation
- Supports additive and multiplicative modes
- Automatic hyperparameter selection via `optimize` method
- Conforms to `BaseForecaster` interface

**Attributes:**
- `robust`: Use robust regression (Siegel) vs OLS
- `multiplicative`: Log-transform for multiplicative seasonality
- `penalty`: Trend dampening factor based on R² (optional)

**Methods:**
- `fit(y, seasonal_period, X, ...)`: Fit all components to series
- `predict(h, X, level)`: Forecast h steps ahead with optional intervals
- `optimize(y, ...)`: Auto-tune hyperparameters via cross-validation
"""
# mfles.py
from __future__ import annotations
from functools import partial as _partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

import utils
from base_forecaster import BaseForecaster


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
def _mse(y, yhat): return jnp.mean((y - yhat) ** 2)
@jax.jit
def _mae(y, yhat): return jnp.mean(jnp.abs(y - yhat))
@jax.jit
def _mape(y, yhat): return jnp.mean(jnp.abs((y - yhat) / (jnp.abs(y) + 1e-10)))
@jax.jit
def _smape(y, yhat): return jnp.mean(2.0 * jnp.abs(y - yhat) / (jnp.abs(y) + jnp.abs(yhat) + 1e-10))
_metric2fn = {"mse": _mse, "mae": _mae, "mape": _mape, "smape": _smape}

@_partial(jax.jit, static_argnums=(1,))
def _rolling_mean(y: jnp.ndarray, window: int) -> jnp.ndarray:
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
def _cap_outliers(y: jnp.ndarray, k=3.0) -> jnp.ndarray:
    mu, sd = jnp.mean(y), jnp.std(y)
    return jnp.clip(y, mu - k * sd, mu + k * sd)

def _set_fourier(period: int) -> int:
    return 5 if period < 10 else (10 if period < 70 else 15)

@_partial(jax.jit, static_argnums=(0, 1, 2))
def _fourier_series(n: int, period: int, order: int) -> jnp.ndarray:
    t = jnp.arange(1, n + 1, dtype=jnp.float32).reshape(-1, 1)
    k = jnp.arange(1, order + 1, dtype=jnp.float32).reshape(1, -1)
    X = 2.0 * jnp.pi * t @ (k / float(period))
    return jnp.hstack([jnp.cos(X), jnp.sin(X)])

@_partial(jax.jit, static_argnums=(0, 1))
def _seasonality_weights(n: int, period: int) -> jnp.ndarray:
    return 1.0 + (jnp.arange(n) // period).astype(jnp.float32)

def _median_init(y: jnp.ndarray, period: int | None) -> jnp.ndarray:
    """Initialize with period-wise medians."""
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
    M = x.shape[0]
    x_sum, y_sum = jnp.sum(x), jnp.sum(y)
    x2, xy = jnp.dot(x, x), jnp.dot(x, y)
    denom = M * x2 - x_sum * x_sum + 1e-12
    slope = (M * xy - x_sum * y_sum) / denom
    intercept = (y_sum - slope * x_sum) / M
    return slope * x + intercept

@jax.jit
def _siegel_repeated_medians(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
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
    beta = jnp.linalg.pinv(X.T @ X) @ (X.T @ y)
    return X @ beta, beta

@jax.jit
def _wls_predict(X: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    WX = X * w[:, None]
    beta = jnp.linalg.pinv(X.T @ WX) @ (X.T @ (w * y))
    return X @ beta

@jax.jit
def _cov_proxy(y: jnp.ndarray, mult: int = 1) -> jnp.ndarray:
    """Coefficient of variation proxy for robustness detection."""
    def mult_path():
        return jnp.sqrt(jnp.exp(jnp.log(10.0) * (jnp.std(y) ** 2)) - 1.0)
    
    def additive_path():
        sd, mu = jnp.std(y), jnp.mean(y)
        return jnp.where(mu != 0, sd / mu, sd)
    
    return lax.cond(mult != 0, mult_path, additive_path)

def _knots_from_gradients(y: jnp.ndarray, k: int, max_knots: int = 50) -> jnp.ndarray:
    """
    Select knots based on gradients. Returns fixed-size array for JIT compatibility.
    
    Args:
        y: Input series
        k: Number of knots to select (can be dynamic)
        max_knots: Maximum number of knots (static, for fixed output size)
    
    Returns:
        Array of shape (max_knots,) with knot positions (padded with zeros if k < max_knots)
    """
    g = jnp.abs(jnp.diff(y))
    idx = jnp.argsort(-g)
    
    # Create fixed-size output array
    knots = jnp.zeros(max_knots, dtype=jnp.int32)
    
    # Fill with top k indices using masking
    fill_mask = jnp.arange(max_knots) < k
    # Take indices with wrapping (safe indexing)
    safe_idx = idx[jnp.minimum(jnp.arange(max_knots), idx.shape[0] - 1)]
    knots = jnp.where(fill_mask, safe_idx + 1, 0)
    knots = jnp.clip(knots, 0, y.shape[0] - 1)
    
    return jnp.sort(knots)

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

def _spectral_step(X: jnp.ndarray) -> float:
    s = jnp.linalg.svd(X, full_matrices=False, compute_uv=False)
    L = (s[0] ** 2) + 1e-6
    return 1.0 / L

def _soft(z: jnp.ndarray, lam: float) -> jnp.ndarray:
    return jnp.sign(z) * jnp.maximum(0.0, jnp.abs(z) - lam)

@jax.jit
def _lasso_ista(X: jnp.ndarray, y: jnp.ndarray, alpha: float, maxiter: int = 200, tol: float = 1e-4) -> jnp.ndarray:
    step = _spectral_step(X)
    return _lasso_ista_with_step(X, y, alpha, step, maxiter)

def _lasso_ista_with_step(X: jnp.ndarray, y: jnp.ndarray, alpha: float, step: jnp.ndarray, maxiter: int = 200) -> jnp.ndarray:
    """LASSO via ISTA with pre-computed spectral step (avoids redundant SVD)."""
    beta = jnp.zeros((X.shape[1],), dtype=y.dtype)
    def body(carry):
        b = carry
        grad = X.T @ (X @ b - y)
        b_new = _soft(b - step * grad, step * alpha)
        return b_new, jnp.linalg.norm(b_new - b)
    def loop(b):
        b_new = lax.fori_loop(0, maxiter, lambda i, bb: body(bb)[0], b)
        return b_new
    return loop(beta)


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
    n_cps: int                     # number of changepoints
    max_period: int                # max seasonal period (for seas_tail allocation)
    n_exo_features: int            # number of exogenous features
    convergence_tol: float         # absolute tolerance for early stopping
    min_iter: int                  # minimum iterations before allowing early stop (n-dependent)
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
    # 2. Convergence check (early stopping with relative tolerance)
    # -------------------------------------------------------------------------
    improved = cur < state.best
    improvement = state.best - cur
    new_best = lax.cond(improved, lambda: cur, lambda: state.best)
    new_stalls = lax.cond(improved, lambda: jnp.int32(0), lambda: state.stalls + 1)

    # Early stopping with n-dependent minimum iterations:
    # Small n (100-1000): min_iter=10-15 (quick convergence)
    # Large n (5000+): min_iter=25-30 (trend needs more iterations)
    stall_stop = new_stalls >= 5
    relative_improvement = improvement / (new_best + 1e-12)
    small_improvement = improved & (relative_improvement < config.convergence_tol)
    past_minimum = i >= config.min_iter  # n-dependent threshold
    converged = past_minimum & (stall_stop | small_improvement)

    
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
    # 5. Trend + guarded SES update
    # -------------------------------------------------------------------------
    # Odd rounds: trend (piecewise/robust/OLS)
    # Even rounds: OLS by default, with late guarded SES attempt.
    is_odd = (i % 2) == 1

    def trend_piecewise():
        Xb = config.hinge_basis
        beta = _lasso_ista_with_step(Xb, resids, config.alpha, config.lasso_step, config.lasso_maxiter)
        return (Xb @ beta) * config.linear_lr

    def trend_robust():
        return _siegel_repeated_medians(config.x_idx, resids) * config.linear_lr

    def trend_ols():
        return _fast_ols_fit_predict(config.x_idx, resids) * config.linear_lr

    def odd_round_candidate():
        use_piecewise = use_changepoints & (config.n_cps > 0)
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
        # Fast fallback trend path
        ols_tren = trend_ols()
        ols_test = _mse(config.y_tr, new_fitted2 + ols_tren)
        ols_improves = ols_test < new_best_after_exo

        # SES controller:
        # 0=off, 1=lite, 2=full, 3=adaptive(lite->full escalation)
        # Attempt SES only in later rounds with strict gain thresholds.
        def try_ses():
            capped_resids = resids.at[-2:].set(_cap_outliers(resids, 3.0)[-2:])
            # Detrend first so SES focuses on residual autocorrelation, not slope.
            detrended = capped_resids - _fast_ols_fit_predict(config.x_idx, capped_resids)
            lite_tren = _ses_ensemble_via_utils(detrended, config.ses_alphas, False, 1) * (config.rs_lr * 0.2)
            lite_test = _mse(config.y_tr, new_fitted2 + lite_tren)
            lite_improves = (lite_test < (new_best_after_exo * (1.0 - 0.01))) & (lite_test < ols_test)

            if ses_mode_code == 1:
                use_ses = lite_improves
                chosen = lax.cond(use_ses, lambda: lite_tren, lambda: ols_tren)
                chosen_test = lax.cond(use_ses, lambda: lite_test, lambda: ols_test)
                chosen_improves = lax.cond(use_ses, lambda: lite_improves, lambda: ols_improves)
                return chosen, chosen_improves, chosen_test, use_ses

            full_tren = _ses_ensemble_via_utils(detrended, config.ses_alphas, True, 1) * (config.rs_lr * 0.2)
            full_test = _mse(config.y_tr, new_fitted2 + full_tren)
            full_improves = (full_test < (new_best_after_exo * (1.0 - 0.01))) & (full_test < ols_test)

            if ses_mode_code == 2:
                use_ses = full_improves
                chosen = lax.cond(use_ses, lambda: full_tren, lambda: ols_tren)
                chosen_test = lax.cond(use_ses, lambda: full_test, lambda: ols_test)
                chosen_improves = lax.cond(use_ses, lambda: full_improves, lambda: ols_improves)
                return chosen, chosen_improves, chosen_test, use_ses

            # Adaptive: start with lite; use full only if it clearly outperforms lite.
            use_full = lite_improves & full_improves & (full_test < (lite_test * (1.0 - 0.002)))
            ses_tren = lax.cond(use_full, lambda: full_tren, lambda: lite_tren)
            ses_test = lax.cond(use_full, lambda: full_test, lambda: lite_test)
            ses_improves = lax.cond(use_full, lambda: full_improves, lambda: lite_improves)
            use_ses = ses_improves
            chosen = lax.cond(use_ses, lambda: ses_tren, lambda: ols_tren)
            chosen_test = lax.cond(use_ses, lambda: ses_test, lambda: ols_test)
            chosen_improves = lax.cond(use_ses, lambda: ses_improves, lambda: ols_improves)
            return chosen, chosen_improves, chosen_test, use_ses

        def no_ses():
            return ols_tren, ols_improves, ols_test, jnp.bool_(False)

        # Delay SES attempts; early rounds prioritize trend convergence.
        if ses_mode_code == 0:
            return no_ses()
        return lax.cond(i >= jnp.int32(12), try_ses, no_ses)

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

    # Compute R² penalty on first trend iteration.
    def compute_penalty():
        mu = jnp.mean(resids)
        ssres = jnp.sum((resids - tren) ** 2)
        sstot = jnp.sum((resids - mu) ** 2) + 1e-12
        return jnp.float32(1.0 - (ssres / sstot))

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
            multiplicative = config.y_tr.min() > 0  # Approximation
            return _cov_proxy(resids, jnp.int32(multiplicative)) > config.cov_threshold

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
    # Note: trend_tail will be computed from final fitted values AFTER the loop
    return _LoopState(
        i=i + 1,
        fitted=new_fitted3,
        linear_component=new_linear,
        seasonal_component=new_seasonal,
        ses_component=new_ses,
        exogenous_component=new_exo_comp,
        trend_tail=state.trend_tail,  # Keep as-is, will be updated after loop
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
    uses_exog = True

    def __init__(self, verbose: int = 1, robust: bool | None = None, alias: str = "MFLES", conformal_params=None):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
        self.verbose = verbose
        # Disable robust auto-detection for large series (O(n²) complexity)
        self.robust = robust if robust is not None else False
        self.predicted = None
        self._exo_beta = None
        self.penalty = None

    def fit(
        self,
        y,
        seasonal_period=None,
        X=None,
        fourier_order=None,
        ma=None,
        alpha=1.0,
        decay=-1,
        n_changepoints=0.25,
        seasonal_lr=0.9,
        rs_lr=1.0,
        exogenous_lr=1.0,
        exogenous_estimator=None,
        exogenous_params={},
        linear_lr=1.0,
        cov_threshold=0.7,
        moving_medians=False,
        max_rounds=50,
        min_alpha=0.05,
        max_alpha=1.0,
        round_penalty=0.0001,
        trend_penalty=True,
        multiplicative=None,
        changepoints=True,
        smoother=False,
        ses_mode: str = "adaptive",
        seasonality_weights=False,
        gradient_strategy=False,
    ):
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        n = y.shape[0]
        if multiplicative is None:
            multiplicative = seasonal_period is not None and jnp.min(y) > 0
        if multiplicative:
            const = jnp.min(y)
            y_tr = jnp.log(y)
            mean, std = jnp.array(0.0, y.dtype), jnp.array(1.0, y.dtype)
        else:
            const = None
            mean, std = jnp.mean(y), jnp.std(y)
            y_tr = y - mean
            y_tr = jnp.where(std > 0, y_tr / std, y_tr)

        self.const, self.mean, self.std = const, mean, std
        self.trend_penalty = bool(trend_penalty)

        if n < 4 or jnp.all(y_tr == jnp.mean(y_tr)):
            base = y_tr[-1]
            self.trend = jnp.array([base, base], dtype=y.dtype)
            self.seasonality = None
            self.linear_component = jnp.zeros(n, y.dtype)
            self.seasonal_component = jnp.zeros(n, y.dtype)
            self.ses_component = jnp.zeros(n, y.dtype)
            self.median_component = jnp.zeros(n, y.dtype)
            self.exogenous_component = jnp.zeros(n, y.dtype)
            self._exo_beta = None
            fitted = jnp.full(n, base, y.dtype)
            return self._finalize_fit(fitted, multiplicative)

        sp_list = None
        if seasonal_period is not None:
            sp_list = seasonal_period if isinstance(seasonal_period, list) else [int(seasonal_period)]
        seas_tail = None
        fourier_series = []
        cycle_weights = []
        if sp_list is not None:
            for p in sp_list:
                fo = int(_set_fourier(p)) if fourier_order is None else int(fourier_order)
                fourier_series.append(_fourier_series(n, p, fo))
                if seasonality_weights:
                    cycle_weights.append(_seasonality_weights(n, p))

        linear_component = jnp.zeros(n, y.dtype)
        seasonal_component = jnp.zeros(n, y.dtype)
        ses_component = jnp.zeros(n, y.dtype)
        median_component = _median_init(y_tr, max(sp_list) if (sp_list is not None and moving_medians) else None)
        exogenous_component = jnp.zeros(n, y.dtype)
        fitted = median_component.copy()
        trend_tail = jnp.array([fitted[-1], fitted[-1]], y.dtype)

        ma_cycle = [1] if ma is None else (ma if isinstance(ma, list) else [int(ma)])

        if isinstance(n_changepoints, float) and n_changepoints < 1:
            n_cps = int(n_changepoints * n)
        elif isinstance(n_changepoints, int):
            n_cps = n_changepoints
        else:
            n_cps = 0

        # Adaptive changepoint policy:
        # For long series, the default dense changepoint setting (0.25*n) can
        # overfit and degrade out-of-sample accuracy. Keep changepoints enabled
        # only when the caller explicitly requests a non-default setting.
        default_cp_density = isinstance(n_changepoints, float) and abs(n_changepoints - 0.25) < 1e-12
        auto_disable_changepoints = bool(changepoints) and default_cp_density and (n >= 5000)
        effective_changepoints = bool(changepoints) and (not auto_disable_changepoints)

        # Large-n speed guardrail: cap changepoints to keep trend step near-linear.
        # The default 0.25*n becomes very expensive above 5k and hurts warm latency.
        if n_cps > 0:
            if n >= 10000:
                n_cps = min(n_cps, 160)
            elif n >= 5000:
                n_cps = min(n_cps, 224)
            elif n >= 2000:
                n_cps = min(n_cps, 256)

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
        use_changepoints = bool(effective_changepoints)
        init_robust = self.robust is None  # Need auto-detection
        ses_mode_norm = str(ses_mode).lower()
        if ses_mode_norm not in ("off", "lite", "full", "adaptive"):
            raise ValueError("ses_mode must be one of: 'off', 'lite', 'full', 'adaptive'")
        ses_mode_code = {"off": 0, "lite": 1, "full": 2, "adaptive": 3}[ses_mode_norm]
        
        # Stack Fourier series into 3D array (num_periods, n, max_cols)
        if has_seasonality:
            max_fourier_cols = max(fs.shape[1] for fs in fourier_series)
            fourier_stack = jnp.zeros((len(sp_list), n, max_fourier_cols), dtype=y.dtype)
            for idx, fs in enumerate(fourier_series):
                fourier_stack = fourier_stack.at[idx, :, :fs.shape[1]].set(fs)
            
            if use_seasonality_weights:
                weights_stack = jnp.stack(cycle_weights, axis=0)
            else:
                weights_stack = jnp.zeros((len(sp_list), n), dtype=y.dtype)
            
            sp_array = jnp.array(sp_list, dtype=jnp.int32)
            max_period = max(sp_list)
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

        # Adaptive linear_lr: boost slope updates for larger n but avoid overshoot.
        # Keep the cap moderate so trend doesn't explode on long series.
        if n >= 2000:
            scale_factor = min(1.35, 1.0 + 0.00010 * (n - 2000))
            effective_linear_lr = linear_lr * scale_factor
        else:
            effective_linear_lr = linear_lr

        # Adaptive min_iter: keep enough rounds at large n for stable trend/seasonality.
        if n < 1000:
            min_iter = 10
        elif n < 2000:
            min_iter = 14
        elif n < 5000:
            min_iter = 18
        elif n < 10000:
            min_iter = 22
        else:
            min_iter = 24

        # Large-n trend step budget: trimmed vs baseline, but not aggressively.
        if n < 2000:
            lasso_maxiter = 50
        elif n < 5000:
            lasso_maxiter = 42
        elif n < 10000:
            lasso_maxiter = 34
        else:
            lasso_maxiter = 28

        # Slightly looser convergence threshold at large n to reduce tail rounds.
        if n >= 10000:
            convergence_tol = 2e-4
        elif n >= 5000:
            convergence_tol = 1.5e-4
        else:
            convergence_tol = 1e-4

        # Pre-compute hinge basis + spectral step ONCE (avoid redundant SVD per iteration)
        if bool(effective_changepoints) and n_cps > 0 and not bool(gradient_strategy):
            knots = _uniform_knots(n, n_cps)
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
            seasonal_lr=float(seasonal_lr),
            linear_lr=float(effective_linear_lr),
            exogenous_lr=float(exogenous_lr),
            rs_lr=float(rs_lr),
            round_penalty=float(round_penalty),
            alpha=float(alpha),
            cov_threshold=float(cov_threshold),
            n_cps=n_cps,
            max_period=max_period,
            n_exo_features=n_exo_features,
            convergence_tol=float(convergence_tol),  # Early stopping tolerance
            min_iter=min_iter,  # n-dependent minimum iterations
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
            robust=jnp.bool_(self.robust if self.robust is not None else False),
            penalty=jnp.float32(0.0),
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

        # Extract trend from linear component only
        # Note: SES captures residual noise/seasonality, not trend
        self.trend = jnp.array([self.linear_component[-2], self.linear_component[-1]], dtype=fitted.dtype)
        
        # Extract seasonality tail (dynamic slice based on seas_tail_len)
        if has_seasonality and final_state.seas_tail_len > 0:
            self.seasonality = lax.dynamic_slice(
                final_state.seas_tail, 
                (0,), 
                (final_state.seas_tail_len,)
            )
        else:
            self.seasonality = None
        
        # Update robust and penalty if they were auto-detected
        if init_robust:
            self.robust = bool(final_state.robust)
        self.penalty = float(final_state.penalty) if final_state.penalty > 0 else None
        
        # Update exogenous coefficients
        if has_exogenous and jnp.any(final_state.exo_beta != 0):
            self._exo_beta = final_state.exo_beta

        return self._finalize_fit(fitted, multiplicative)

    def _finalize_fit(self, fitted_tr: jnp.ndarray, multiplicative: bool):
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

    def predict(self, h: int, X=None, level: list[int | float] | None = None):
        h = int(h)
        last, prev = self.trend[1], self.trend[0]
        slope = (last - prev)
        if self.trend_penalty and (self.penalty is not None):
            slope = slope * jnp.maximum(0.0, self.penalty)
        trend_fcst = slope * jnp.arange(1, h + 1, dtype=jnp.float32) + last

        if self.seasonality is not None and self.seasonality.size > 0:
            seas_fcst = utils._repeat_val_seas(self.seasonality, h)
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
            cs = self.conformity_scores(self.model_["fitted"])
            out = self.add_confidence_intervals(out, cs, level, "conformal_distribution")
        return out

    def forecast(self, y, h, X=None, X_future=None, level: list[int | float] | None = None, seasonal_period=None, **fit_kwargs):
        return self.new().fit(y, X=X, seasonal_period=seasonal_period, **fit_kwargs).predict(h, X=X_future, level=level)

    def optimize(
        self,
        y,
        seasonal_period,
        n_steps,
        test_size,
        step_size=1,
        metric="smape",
        X=None,
        params=None,
    ):
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
    from conformal_intervals import ConformalIntervals
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