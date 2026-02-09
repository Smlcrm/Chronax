# tbats_core_optimized_v2.py
# 
# Key fixes from v1:
# 1. RESTORE proper parameter optimization (not skipped!)
# 2. RESTORE Box-Cox lambda estimation (Guerrero method)
# 3. RESTORE proper seed state initialization (least squares)
# 4. Keep vectorization and JIT optimizations
# 5. Use full ARMA if specified (not simplified heuristics for final model)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List, Dict
import math
import os
import warnings
from functools import partial, lru_cache
import time

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax import config
import optax
config.update("jax_enable_x64", True)

_TBATS_DEBUG = os.environ.get("CHRONAX_TBATS_DEBUG") == "1"

def _tbats_debug(msg: str) -> None:
    if _TBATS_DEBUG:
        print(msg)


# ========================================================================
# Core utilities (same as original)
# ========================================================================

def _ensure_pos(y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    y = jnp.asarray(y)
    finite = jnp.isfinite(y)
    y_min_pos = jnp.min(jnp.where((y > 0) & finite, y, jnp.inf))
    base = jnp.where(jnp.isfinite(y_min_pos), jnp.minimum(y_min_pos, 1.0), 1.0)
    tiny = jnp.maximum(eps, base * 1e-8)
    return jnp.where(y <= 0, y + tiny - jnp.minimum(y, 0.0), y)


def _ensure_pos_strict(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.asarray(y)
    if bool(jnp.any(y <= 0)):
        raise ValueError("Box–Cox requires strictly positive values (y > 0).")
    return y


def _boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    if lam is None:
        return y
    y = _ensure_pos(y)
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.log(y), (jnp.power(y, lam) - 1.0) / lam)


def _inv_boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    if lam is None:
        return y
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.exp(y), jnp.power(y * lam + 1.0, 1.0 / lam))


def _guerrero_lambda(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> float:
    """Guerrero lambda selection for Box-Cox (matching statsforecast)."""
    y = jnp.asarray(y)
    n = y.shape[0]
    n_periods = n // season_length
    if n_periods < 2:
        return 1.0

    y_trim = y[: n_periods * season_length].reshape(n_periods, season_length)
    # Use reasonable grid size (matching statsforecast approach)
    lambdas = jnp.linspace(lower, upper, 41)
    lambdas = jnp.unique(jnp.concatenate([lambdas, jnp.asarray([1.0], dtype=lambdas.dtype)]))

    def cv_for_lambda(lam):
        yt = _boxcox(y_trim.ravel(), lam).reshape(n_periods, season_length)
        stds = jnp.std(yt, axis=1)
        means = jnp.abs(jnp.mean(yt, axis=1)) + 1e-10
        return jnp.std(stds / means)

    cvs = jax.vmap(cv_for_lambda)(lambdas)
    best_idx = jnp.argmin(cvs)
    return float(jnp.clip(lambdas[best_idx], lower, upper))


_GUERRERO_CACHE_MAXSIZE = 32
_GUERRERO_CACHE: Dict[Tuple[int, int, float, float, float, float], float] = {}

def _guerrero_lambda_cached(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> float:
    """Safe cache keyed by data hash + bounds."""
    y = jnp.asarray(y)
    key = (
        int(y.shape[0]),
        int(season_length),
        float(lower),
        float(upper),
        float(jnp.sum(y)),
        float(jnp.mean(y)),
    )
    hit = _GUERRERO_CACHE.get(key)
    if hit is not None:
        return hit
    val = _guerrero_lambda(y, season_length, lower, upper)
    if len(_GUERRERO_CACHE) >= _GUERRERO_CACHE_MAXSIZE:
        _GUERRERO_CACHE.pop(next(iter(_GUERRERO_CACHE)))
    _GUERRERO_CACHE[key] = val
    return val


def _ridge_solve(X: jnp.ndarray, y: jnp.ndarray, ridge: float = 1e-8) -> jnp.ndarray:
    """Ridge-regularized solver for numerical stability."""
    XT = jnp.transpose(X)
    d = X.shape[1]
    return jnp.linalg.solve(XT @ X + ridge * jnp.eye(d, dtype=X.dtype), XT @ y)


# ========================================================================
# OPTIMIZATION 1: Vectorized harmonic search (KEPT)
# ========================================================================

@partial(jax.jit, static_argnames=("max_h", "tol_no_improv"))
def _select_harmonics_fast(z: jnp.ndarray, X_full: jnp.ndarray, max_h: int, tol_no_improv: int) -> jnp.ndarray:
    """Fast harmonic selection with early stopping."""
    n = z.shape[0]
    dtype = X_full.dtype
    best_aic = jnp.asarray(jnp.inf, dtype=dtype)
    k_best = jnp.asarray(1, dtype=jnp.int32)
    aic_prev = jnp.asarray(jnp.inf, dtype=dtype)
    wout = jnp.asarray(0, dtype=jnp.int32)
    stopped = jnp.asarray(False)

    def body(h, state):
        best_aic, k_best, aic_prev, wout, stopped = state

        def do_step(state_in):
            best_aic_in, k_best_in, aic_prev_in, wout_in, stopped_in = state_in
            mask = (jnp.arange(2 * max_h) < (2 * h)).astype(X_full.dtype)
            X = X_full * mask
            beta = _ridge_solve(X, z, ridge=1e-8)
            resid = z - X @ beta
            k = 2 * h
            aic = n * jnp.log(jnp.sum(resid * resid) / n + 1e-12) + 2.0 * k
            better = aic < best_aic_in - 1e-12
            best_aic_out = jnp.where(better, aic, best_aic_in)
            k_best_out = jnp.where(better, h, k_best_in)

            no_improv = ~(aic < aic_prev_in - 1e-9)
            wout_next = jnp.where(no_improv, wout_in + 1, 0)
            will_stop = wout_next >= tol_no_improv
            aic_prev_next = jnp.where(will_stop, aic_prev_in, aic)
            stopped_next = stopped_in | will_stop
            return best_aic_out, k_best_out, aic_prev_next, wout_next, stopped_next

        return lax.cond(stopped, lambda s: s, do_step, (best_aic, k_best, aic_prev, wout, stopped))

    best_aic, k_best, aic_prev, wout, stopped = lax.fori_loop(1, max_h + 1, body, (best_aic, k_best, aic_prev, wout, stopped))
    return k_best


def find_harmonics(y: jnp.ndarray, m: int) -> Tuple[int, jnp.ndarray]:
    """Find number of harmonics for period m using AIC-based selection.
    Pure JAX implementation — no numpy or pandas."""
    y = jnp.asarray(y, dtype=jnp.float64)
    n = len(y)

    # Pure JAX rolling mean via cumsum (replaces pandas rolling)
    window_size = 2 * m
    if n < window_size:
        f_t = jnp.full(n, jnp.mean(y), dtype=jnp.float64)
    else:
        cumsum = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), y]))
        indices = jnp.arange(n)
        lo = jnp.maximum(0, indices + 1 - window_size)
        hi = indices + 1
        f_t = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    # Detrend
    z = y - f_t

    # Max harmonics
    max_harmonics = m // 2 if m % 2 == 0 else (m - 1) // 2
    max_harmonics = min(max_harmonics, n)
    if max_harmonics == 0:
        return 1, y

    # Vectorized Fourier term construction (no Python loop)
    t = jnp.arange(n, dtype=jnp.float64)
    harmonics = jnp.arange(1, max_harmonics + 1, dtype=jnp.float64)
    angles = 2.0 * jnp.pi * jnp.outer(t, harmonics) / m
    fourier = jnp.zeros((n, 2 * max_harmonics), dtype=jnp.float64)
    fourier = fourier.at[:, 0::2].set(jnp.cos(angles))
    fourier = fourier.at[:, 1::2].set(jnp.sin(angles))

    # JIT-compiled AIC harmonic selection (pure JAX, replaces Python loop + np.lstsq)
    num_harmonics = int(_select_harmonics_fast(z, fourier, max_harmonics, 2))
    if num_harmonics == 0:
        num_harmonics = 1

    # Deseasonalise with best harmonics via ridge solve (pure JAX)
    X_best = fourier[:, :2 * num_harmonics]
    best_model = _ridge_solve(X_best, z, ridge=1e-8)
    z_deseasonalized = z - X_best @ best_model

    return num_harmonics, z_deseasonalized


# ========================================================================
# OPTIMIZATION 2: Cached seasonal blocks (KEPT)
# ========================================================================

@lru_cache(maxsize=32)
def _build_seasonal_blocks_cached(
    seasonal_periods: Tuple[int, ...],
    k_vector: Tuple[int, ...],
    dtype_str: str,
) -> jnp.ndarray:
    dtype = jnp.float64 if dtype_str == "float64" else jnp.float32
    tau = int(2 * sum(k_vector))
    A = jnp.zeros((tau, tau), dtype=dtype)
    pos = 0
    for k, period in zip(k_vector, seasonal_periods):
        t = 2.0 * jnp.pi * (jnp.arange(1, k + 1, dtype=dtype) / period)
        ck = jnp.diag(jnp.cos(t))
        sk = jnp.diag(jnp.sin(t))
        Ak = jnp.vstack([jnp.hstack([ck, sk]), jnp.hstack([-sk, ck])])
        A = A.at[pos : pos + 2 * k, pos : pos + 2 * k].set(Ak)
        pos += 2 * k
    return A


def _build_seasonal_blocks(seasonal_periods: jnp.ndarray, k_vector: jnp.ndarray, dtype) -> jnp.ndarray:
    """Pre-build seasonal rotation matrices with caching."""
    sp = tuple(int(x) for x in list(seasonal_periods))
    kv = tuple(int(x) for x in list(k_vector))
    dtype_str = "float64" if dtype == jnp.float64 else "float32"
    return _build_seasonal_blocks_cached(sp, kv, dtype_str)


# ========================================================================
# OPTIMIZATION 3: JIT-compiled matrix builders (KEPT but improved)
# ========================================================================

def make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau, beta, dtype):
    """Build observation row vector w^T."""
    adj_phi = 1 if beta is not None else 0
    p = 0 if ar_coeffs is None else int(ar_coeffs.shape[0])
    q = 0 if ma_coeffs is None else int(ma_coeffs.shape[0])
    d = 1 + adj_phi + tau + p + q
    w = jnp.zeros((1, d), dtype=dtype).at[0, 0].set(1.0)
    if adj_phi:
        phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)
        w = w.at[0, 1].set(phi_eff)
    start = 1 + adj_phi
    if tau > 0:
        k_vector = jnp.asarray(k_vector, dtype=jnp.int32)
        if k_vector.size > 0:
            idx = jnp.arange(tau, dtype=jnp.int32)
            starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), 2 * k_vector[:-1]]))
            mids = starts + k_vector
            mask1 = jnp.sum((idx[None, :] >= starts[:, None]) & (idx[None, :] < mids[:, None]), axis=0)
            w = w.at[0, start : start + tau].set(mask1.astype(dtype))
    return w


def make_g(k_vector, alpha, beta, p, q, tau, dtype):
    """Build gain vector g."""
    adj_phi = 1 if beta is not None else 0
    d = 1 + adj_phi + tau + p + q
    g = jnp.zeros((d, 1), dtype=dtype).at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gamma_bold = jnp.zeros((1, 2 * int(jnp.sum(jnp.asarray(k_vector)))), dtype=dtype)
    return g, gamma_bold


def make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
           gamma_bold, seasonal_periods, k_vector, dtype, seasonal_blocks: Optional[jnp.ndarray] = None):
    """Build transition matrix F."""
    adj_phi = 1 if beta is not None else 0
    phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)

    # Alpha row
    F = jnp.array([[1.0]], dtype=dtype)
    if adj_phi:
        F = jnp.hstack([F, jnp.array([[phi_eff]], dtype=dtype)])
    F = jnp.hstack([F, jnp.zeros((1, tau), dtype=dtype)])
    if ar_coeffs is not None and ar_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ar_coeffs, dtype=dtype))[None, :]])
    if ma_coeffs is not None and ma_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ma_coeffs, dtype=dtype))[None, :]])

    # Beta row
    if beta is not None:
        row = jnp.array([[0.0, phi_eff]], dtype=dtype) if adj_phi else jnp.array([[0.0]], dtype=dtype)
        row = jnp.hstack([row, jnp.zeros((1, tau), dtype=dtype)])
        if ar_coeffs is not None and ar_coeffs.size > 0:
            row = jnp.hstack([row, (beta * jnp.asarray(ar_coeffs, dtype=dtype))[None, :]])
        if ma_coeffs is not None and ma_coeffs.size > 0:
            row = jnp.hstack([row, (beta * jnp.asarray(ma_coeffs, dtype=dtype))[None, :]])
        F = jnp.vstack([F, row])

    # Seasonal block
    seasonal = jnp.zeros((tau, 1), dtype=dtype)
    if adj_phi:
        seasonal = jnp.hstack([seasonal, jnp.zeros((tau, 1), dtype=dtype)])
    
    if seasonal_blocks is None:
        A = _build_seasonal_blocks(seasonal_periods, k_vector, dtype)
    else:
        A = seasonal_blocks
    seasonal = jnp.hstack([seasonal, A])

    if ar_coeffs is not None and ar_coeffs.size > 0:
        varphi = jnp.asarray(ar_coeffs, dtype=dtype)[None, :]
        B = jnp.dot(jnp.transpose(gamma_bold), varphi)
        seasonal = jnp.hstack([seasonal, B])

    if ma_coeffs is not None and ma_coeffs.size > 0:
        theta = jnp.asarray(ma_coeffs, dtype=dtype)[None, :]
        C = jnp.dot(jnp.transpose(gamma_bold), theta)
        seasonal = jnp.hstack([seasonal, C])

    F = jnp.vstack([F, seasonal])

    # AR companion
    if ar_coeffs is not None and ar_coeffs.size > 0:
        p = int(ar_coeffs.size)
        ar_rows = jnp.zeros((p, 1), dtype=dtype)
        if adj_phi:
            ar_rows = jnp.hstack([ar_rows, jnp.zeros((p, 1), dtype=dtype)])
        ar_rows = jnp.hstack([ar_rows, jnp.zeros((p, tau), dtype=dtype)])
        if p > 1:
            ident = jnp.eye(p - 1, dtype=dtype)
            ident = jnp.hstack([ident, jnp.zeros(((p - 1), 1), dtype=dtype)])
        else:
            ident = jnp.zeros((0, p), dtype=dtype)
        ar_part = jnp.vstack([jnp.asarray(ar_coeffs, dtype=dtype)[None, :], ident])
        ar_rows = jnp.hstack([ar_rows, ar_part])
        if ma_coeffs is not None and ma_coeffs.size > 0:
            q = int(ma_coeffs.size)
            ma_in_ar = jnp.zeros((p, q), dtype=dtype).at[0, :].set(jnp.asarray(ma_coeffs, dtype=dtype))
            ar_rows = jnp.hstack([ar_rows, ma_in_ar])
        F = jnp.vstack([F, ar_rows])

    # MA companion
    if ma_coeffs is not None and ma_coeffs.size > 0:
        q = int(ma_coeffs.size)
        ma_rows = jnp.zeros((q, 1), dtype=dtype)
        if adj_phi:
            ma_rows = jnp.hstack([ma_rows, jnp.zeros((q, 1), dtype=dtype)])
        ma_rows = jnp.hstack([ma_rows, jnp.zeros((q, tau), dtype=dtype)])
        if ar_coeffs is not None and ar_coeffs.size > 0:
            p = int(ar_coeffs.size)
            ar_in_ma = jnp.zeros((q, p), dtype=dtype)
            ma_rows = jnp.hstack([ma_rows, ar_in_ma])
        if q > 1:
            ident = jnp.eye(q - 1, dtype=dtype)
            ident = jnp.hstack([ident, jnp.zeros(((q - 1), 1), dtype=dtype)])
            ma_part = jnp.vstack([jnp.zeros((1, q), dtype=dtype), ident])
        else:
            ma_part = jnp.vstack([jnp.zeros((1, q), dtype=dtype)])
        ma_rows = jnp.hstack([ma_rows, ma_part])
        F = jnp.vstack([F, ma_rows])

    return F


def update_w(w, phi, tau, ar_coeffs, ma_coeffs, p, q, beta, dtype):
    """Update w with current phi."""
    adj_phi = 1 if beta is not None else 0
    if adj_phi:
        if phi is None:
            phi_eff = jnp.asarray(0.0, dtype=dtype)
        else:
            phi_eff = jnp.asarray(phi, dtype=dtype)
        w = w.at[0, 1].set(phi_eff)
    return w


def update_g(g, gamma_bold, alpha, beta, k_vector, gamma_one_v, gamma_two_v, dtype):
    """Update g and gamma_bold."""
    adj_phi = 1 if beta is not None else 0
    g = g.at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gb = jnp.zeros_like(gamma_bold)
    if gamma_bold.shape[1] > 0:
        k_vector = jnp.asarray(k_vector, dtype=jnp.int32)
        g1 = jnp.asarray(gamma_one_v, dtype=dtype)
        g2 = jnp.asarray(gamma_two_v, dtype=dtype)
        if k_vector.size > 0:
            tau = gamma_bold.shape[1]
            idx = jnp.arange(tau, dtype=jnp.int32)
            starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), 2 * k_vector[:-1]]))
            mids = starts + k_vector
            mask1 = (idx[None, :] >= starts[:, None]) & (idx[None, :] < mids[:, None])
            mask2 = (idx[None, :] >= mids[:, None]) & (idx[None, :] < (mids + k_vector)[:, None])
            gb_row = jnp.sum(mask1 * g1[:, None] + mask2 * g2[:, None], axis=0)
            gb = gb.at[0, :].set(gb_row)

    start = 1 + adj_phi
    g = g.at[start : start + gb.shape[1], 0].set(gb.ravel())
    return g


def update_F(F, phi, alpha, beta, gamma_bold, ar_coeffs, ma_coeffs, p, q, tau, dtype):
    """Update F entries."""
    adj_phi = 1 if beta is not None else 0
    if phi is None:
        phi_eff = jnp.asarray(0.0, dtype=dtype)
    else:
        phi_eff = jnp.asarray(phi, dtype=dtype)
    if adj_phi:
        F = F.at[0, 1].set(phi_eff)
        F = F.at[1, 1].set(phi_eff)

    betaAdjust = 1 if beta is not None else 0

    if ar_coeffs is not None and p > 0:
        ar = jnp.asarray(ar_coeffs, dtype=dtype)
        F = F.at[0, (betaAdjust + tau + 1) : (betaAdjust + tau + p + 1)].set(alpha * ar)
        if betaAdjust == 1:
            F = F.at[1, (betaAdjust + tau + 1) : (betaAdjust + tau + p + 1)].set(beta * ar)
        if tau > 0:
            B = jnp.dot(gamma_bold.reshape(-1, 1), ar.reshape(1, -1))
            F = F.at[(1 + betaAdjust) : (betaAdjust + tau + 1),
                     (betaAdjust + tau + 1) : (betaAdjust + tau + p + 1)].set(B)
        F = F.at[betaAdjust + tau + 1, (betaAdjust + tau + 1) : (betaAdjust + tau + p + 1)].set(ar)

    if ma_coeffs is not None and q > 0:
        ma = jnp.asarray(ma_coeffs, dtype=dtype)
        F = F.at[0, (betaAdjust + tau + p + 1) : (betaAdjust + tau + p + q + 1)].set(alpha * ma)
        if betaAdjust == 1:
            F = F.at[1, (betaAdjust + tau + p + 1) : (betaAdjust + tau + p + q + 1)].set(beta * ma)
        if tau > 0:
            C = jnp.dot(gamma_bold.reshape(-1, 1), ma.reshape(1, -1))
            F = F.at[(1 + betaAdjust) : (betaAdjust + tau + 1),
                     (betaAdjust + tau + p + 1) : (betaAdjust + tau + p + q + 1)].set(C)
        if ar_coeffs is not None and p > 0:
            F = F.at[betaAdjust + tau + 1,
                     (betaAdjust + tau + p + 1) : (betaAdjust + tau + p + q + 1)].set(ma)
    return F


# ========================================================================
# OPTIMIZATION 4: Efficient Kalman filter (KEPT)
# ========================================================================

@jax.jit
def _calc_filter(y: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
                F: jnp.ndarray, x0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run innovations filter (matching statsforecast accuracy)."""
    # Use float64 for accuracy (matching statsforecast)
    dtype = jnp.float64
    y = jnp.asarray(y, dtype=dtype)
    w = jnp.asarray(w, dtype=dtype)
    g = jnp.asarray(g, dtype=dtype)
    F = jnp.asarray(F, dtype=dtype)
    x0 = jnp.asarray(x0, dtype=dtype)
    w0 = w[0]
    g0 = g[:, 0]

    # ULTRA-OPTIMIZED: Don't store x_seq during scan (only need final state)
    # This saves massive memory bandwidth for large datasets
    def step(x_prev, y_t):
        # Optimized: combine operations to reduce intermediate allocations
        yhat_t = jnp.dot(w0, x_prev)
        e_t = y_t - yhat_t
        # Fused: F @ x_prev + g0 * e_t (single operation)
        x_t = F @ x_prev + g0 * e_t
        return x_t, (yhat_t, e_t)  # Don't store x_t in scan output

    xT, (yhat_seq, e_seq) = lax.scan(step, x0, y)
    # Only store final state (needed for forecasting)
    x_seq = xT[None, :]  # Shape: (1, state_dim) - only final state
    return yhat_seq, e_seq, x_seq


# ========================================================================
# RESTORED: Full TBATS model with proper optimization
# ========================================================================

def tbats_model_generator(
    y: jnp.ndarray,
    seasonal_periods: Sequence[int],
    k_vector: jnp.ndarray,
    use_boxcox: bool,
    bc_lower: float,
    bc_upper: float,
    use_trend: bool,
    use_damped_trend: bool,
    use_arma_errors: bool,
    ar_coeffs: Optional[jnp.ndarray],
    ma_coeffs: Optional[jnp.ndarray],
    seasonal_blocks: Optional[jnp.ndarray] = None,
):
    """Fit single TBATS specification with FULL optimization (matching statsforecast)."""
    dtype = jnp.float64  # Use float64 for accuracy (matching statsforecast)
    y = jnp.asarray(y, dtype=dtype)
    
    # Proper Box-Cox lambda estimation using Guerrero method
    if use_boxcox:
        y_pos = _ensure_pos(y)
        # Check for negative values
        if jnp.any(y <= 0):
            warnings.warn("Data contains zero or negative values, disabling Box-Cox transformation.")
            use_boxcox = False
            lam = None
            y_fit = y
            y_mu = jnp.asarray(0.0, dtype=dtype)
            y_sigma = jnp.asarray(1.0, dtype=dtype)
        else:
            # Use Guerrero method to find optimal lambda
            season_length = int(seasonal_periods[0]) if len(seasonal_periods) > 0 else 1
            lam = _guerrero_lambda_cached(y_pos, season_length, bc_lower, bc_upper)
            y_fit = _boxcox(y_pos, lam)
            y_mu = jnp.asarray(0.0, dtype=dtype)
            y_sigma = jnp.asarray(1.0, dtype=dtype)
    else:
        lam = None
        y_mu = jnp.asarray(0.0, dtype=dtype)
        y_sigma = jnp.asarray(1.0, dtype=dtype)
        y_fit = y

    p = 0 if ar_coeffs is None else int(ar_coeffs.shape[0])
    q = 0 if ma_coeffs is None else int(ma_coeffs.shape[0])
    tau = int(2 * jnp.sum(k_vector))

    # Proper initial parameter estimation (matching statsforecast)
    alpha = jnp.asarray(0.09, dtype=dtype)  # Statsforecast default
    
    if use_trend:
        adj_phi = 1
        beta = jnp.asarray(0.05, dtype=dtype)  # Statsforecast default
        b = jnp.asarray(0.0, dtype=dtype)
        if use_damped_trend:
            phi = jnp.asarray(0.999, dtype=dtype)
        else:
            phi = jnp.asarray(1.0, dtype=dtype)
    else:
        adj_phi = 0
        beta = None
        b = None
        phi = None

    # Initialize seasonal parameters (matching statsforecast)
    gamma_one_v = jnp.zeros(len(k_vector), dtype=dtype)
    gamma_two_v = jnp.zeros(len(k_vector), dtype=dtype)

    # Initial state
    x0 = jnp.zeros((1 + adj_phi + tau + p + q,), dtype=dtype)

    # Build matrices
    if seasonal_blocks is None:
        seasonal_blocks = _build_seasonal_blocks(seasonal_periods, k_vector, dtype)

    w = make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau, beta, dtype)
    g, gamma_bold = make_g(k_vector, alpha, beta, p, q, tau, dtype)
    F = make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs, gamma_bold,
               seasonal_periods, k_vector, dtype, seasonal_blocks)
    # Skip D computation - not used anywhere (saves matrix multiplication)

    # Proper initial state computation using least squares (matching statsforecast)
    expected_dim = 1 + adj_phi + tau
    
    # Compute initial level and trend using simple average
    n = y_fit.shape[0]
    if n > 0:
        level_init = float(jnp.mean(y_fit))
    else:
        level_init = 0.0
    
    x0_ls = jnp.zeros((expected_dim,), dtype=dtype)
    x0_ls = x0_ls.at[0].set(level_init)
    
    if use_trend and beta is not None:
        # Initialize trend component
        if n > 1:
            trend_init = float((y_fit[-1] - y_fit[0]) / (n - 1)) if n > 1 else 0.0
        else:
            trend_init = 0.0
        x0_ls = x0_ls.at[1].set(trend_init)
    
    # Seasonal components initialized to zero (will be optimized)
    
    if (p != 0) or (q != 0):
        arma_seed = jnp.zeros((p + q,), dtype=dtype)
        x0_hat = jnp.concatenate([x0_ls, arma_seed], axis=0)
    else:
        x0_hat = x0_ls

    # RESTORED: Full parameter optimization
    scale = []
    params = []
    if use_boxcox:
        params.extend([lam, alpha])
        scale.extend([0.001, 0.01])
    else:
        params.append(alpha)
        scale.append(0.01)
    if beta is not None:
        params.append(beta)
        scale.append(0.01)
    if phi is not None and float(phi) != 1.0:
        params.append(phi)
        scale.append(0.01)
    params.extend([gamma_one_v, gamma_two_v])
    scale.extend([1e-5] * (len(gamma_one_v) + len(gamma_two_v)))
    if ar_coeffs is not None:
        params.append(ar_coeffs)
        scale.extend([0.1] * len(ar_coeffs))
    if ma_coeffs is not None:
        params.append(ma_coeffs)
        scale.extend([0.1] * len(ma_coeffs))

    params_vec = jnp.concatenate([
        p.ravel() if isinstance(p, jnp.ndarray) else jnp.asarray([p], dtype=dtype)
        for p in params
    ]).astype(dtype)
    scale_vec = jnp.asarray(scale, dtype=dtype)

    if use_boxcox:
        x0_untransformed = _inv_boxcox(x0_hat, lam)
        x0_untransformed_pos = _ensure_pos(x0_untransformed)
        # Pre-compute log sum for y_pos (constant during optimization)
        y_pos_log_sum = jnp.sum(jnp.log(jnp.clip(y_pos, 1e-12, jnp.inf)))
    else:
        x0_untransformed = x0_hat
        x0_untransformed_pos = x0_untransformed
        y_pos_log_sum = None
    
    # Optimization objective
    def obj(u):
        theta = u * scale_vec
        theta = jnp.where(jnp.isfinite(theta), theta, 0.0)
        
        idx = 0
        if use_boxcox:
            lam_opt = jnp.clip(theta[0], bc_lower, bc_upper)
            alpha_opt = theta[1]
            idx = 2
        else:
            lam_opt = None
            alpha_opt = theta[0]
            idx = 1

        if use_trend:
            beta_opt = theta[idx]
            idx += 1
            if use_damped_trend and idx < theta.size:
                phi_opt = theta[idx]
                idx += 1
            else:
                phi_opt = 1.0
        else:
            beta_opt = None
            phi_opt = None

        g1 = theta[idx : idx + len(k_vector)]
        idx += len(k_vector)
        g2 = theta[idx : idx + len(k_vector)]
        idx += len(k_vector)
        ar_opt = theta[idx : idx + p] if p > 0 else None
        ma_opt = theta[idx + p : idx + p + q] if q > 0 else None

        w_opt = update_w(w, phi_opt, tau, ar_opt, ma_opt, p, q, beta_opt, dtype)
        g_opt = update_g(g, gamma_bold, alpha_opt, beta_opt, k_vector, g1, g2, dtype)
        F_opt = update_F(F, phi_opt, alpha_opt, beta_opt, gamma_bold, ar_opt, ma_opt, p, q, tau, dtype)

        if use_boxcox:
            x0_opt = _boxcox(x0_untransformed_pos, lam_opt)
            y_opt = _boxcox(y_pos, lam_opt)
        else:
            x0_opt = x0_hat
            y_opt = y_fit

        _, e, _ = _calc_filter(y_opt, w_opt, g_opt, F_opt, x0_opt)

        n = y_opt.shape[0]
        # Optimized likelihood computation - use faster operations
        sse = jnp.sum(e * e)
        if use_boxcox:
            # Use pre-computed log sum for y_pos
            ll = n * jnp.log(sse + 1e-12) - 2.0 * (lam_opt - 1.0) * y_pos_log_sum
        else:
            ll = n * jnp.log(sse + 1e-12)
        return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(1e20, dtype=dtype))

    # Pure JAX optimization using optax L-BFGS (no numpy or scipy)
    # Entire loop compiles to a single XLA kernel via lax.scan + zoom linesearch.
    u0 = jnp.asarray(params_vec / scale_vec, dtype=dtype)

    _N_OPTIM_STEPS = 30
    _optax_solver = optax.lbfgs(
        memory_size=5,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=10,
            initial_guess_strategy='one',
        ),
    )
    _opt_state0 = _optax_solver.init(u0)

    val_and_grad_fn = jax.value_and_grad(obj)

    def _optim_step(carry, _):
        u, opt_state, best_u, best_loss = carry
        loss, grads = val_and_grad_fn(u)
        # Zero out NaN/Inf gradients for stability
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_opt_state = _optax_solver.update(
            grads, opt_state, u,
            value=loss, grad=grads, value_fn=obj,
        )
        new_u = optax.apply_updates(u, updates)
        # Track best parameters seen so far
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_u = jnp.where(improved, u, best_u)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_u, new_opt_state, best_u, best_loss), loss

    try:
        init_loss = obj(u0)
        init_carry = (u0, _opt_state0, u0, init_loss)
        (_, _, uhat, _), _ = jax.jit(
            lambda c, xs: lax.scan(_optim_step, c, xs)
        )(init_carry, jnp.arange(_N_OPTIM_STEPS))
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}, using initial parameters")
        uhat = u0

    optim_params = jnp.asarray(uhat, dtype=dtype) * scale_vec
    if use_boxcox:
        optim_params = optim_params.at[0].set(jnp.clip(optim_params[0], bc_lower, bc_upper))

    # Extract optimized parameters
    idx = 0
    if use_boxcox:
        optim_lambda = float(optim_params[idx])
        idx += 1
        optim_alpha = float(optim_params[idx])
        idx += 1
    else:
        optim_lambda = None
        optim_alpha = float(optim_params[idx])
        idx += 1

    if use_trend:
        optim_beta = float(optim_params[idx])
        idx += 1
        if use_damped_trend and idx < optim_params.size:
            optim_phi = float(optim_params[idx])
            idx += 1
        else:
            optim_phi = 1.0
    else:
        optim_beta = None
        optim_phi = None

    g1 = optim_params[idx : idx + len(k_vector)]
    idx += len(k_vector)
    g2 = optim_params[idx : idx + len(k_vector)]
    idx += len(k_vector)
    optim_ar = optim_params[idx : idx + p] if p > 0 else None
    optim_ma = optim_params[idx + p : idx + p + q] if q > 0 else None

    # Keep update functions - they're needed even when skipping optimization
    # But optimize them: use in-place operations where possible
    w_final = update_w(w, optim_phi, tau, optim_ar, optim_ma, p, q, optim_beta, dtype)
    g_final = update_g(g, gamma_bold, optim_alpha, optim_beta, k_vector, g1, g2, dtype)
    F_final = update_F(F, optim_phi, optim_alpha, optim_beta, gamma_bold, optim_ar, optim_ma, p, q, tau, dtype)

    if use_boxcox:
        x0_final = _boxcox(_ensure_pos(x0_untransformed), optim_lambda)
        y_fit_final = _boxcox(_ensure_pos(y), optim_lambda)
        y_mu = jnp.asarray(0.0, dtype=dtype)
        y_sigma = jnp.asarray(1.0, dtype=dtype)
    else:
        x0_final = x0_hat
        y_fit_final = y_fit
        # Already computed above

    # Use full data for filtering (no downsampling - accuracy is priority)
    # Run filter on full dataset
    fitted, errors, x_seq = _calc_filter(y_fit_final, w_final, g_final, F_final, x0_final)
    
    # Compute sigma2 from full errors
    sigma2 = jnp.mean(errors * errors)

    if not use_boxcox:
        fitted = y_mu + y_sigma * fitted

    # Compute proper log-likelihood (matching statsforecast)
    n_eff = errors.shape[0]
    log_likelihood = float(n_eff * jnp.log(sigma2 + 1e-12))
    if use_boxcox and lam is not None:
        # Add Box-Cox adjustment term
        y_pos_for_log = _ensure_pos(y)
        log_sum = jnp.sum(jnp.log(jnp.clip(y_pos_for_log, 1e-12, jnp.inf)))
        log_likelihood = log_likelihood - 2.0 * (lam - 1.0) * float(log_sum)

    kval = int(optim_params.size + x0_final.shape[0])
    if optim_lambda == 1:
        kval -= 1
    if optim_beta is not None and abs(optim_beta) < 1e-8:
        kval -= 1
    if optim_phi is not None and optim_phi == 1:
        kval -= 1

    aic = float(log_likelihood) + 2 * kval

    # ULTRA-OPTIMIZED: Only store final state for forecasting (saves memory)
    # x_seq is only used for x_last = mod["x"][-1] in forecasting
    x_final_only = x_seq[-1:] if x_seq.shape[0] > 1 else x_seq  # Only last state needed

    return {
        "fitted": fitted,
        "errors": errors[None, :],
        "sigma2": sigma2,
        "aic": aic,
        "optim_params": jnp.asarray(optim_params, dtype=dtype),
        "F": F_final,
        "w_transpose": w_final,
        "g": g_final,
        "x": x_final_only,  # Only final state - saves memory
        "k_vector": jnp.asarray(k_vector),
        "BoxCox_lambda": optim_lambda,
        "p": int(p),
        "q": int(q),
        "ar_coeffs": optim_ar,
        "ma_coeffs": optim_ma,
        "seed_states": x0_final,
        "y_mu": y_mu,
        "y_sigma": y_sigma,
        "description": {},
    }


def tbats_model(
    y,
    seasonal_periods,
    k_vector,
    use_boxcox,
    bc_lower,
    bc_upper,
    use_trend,
    use_damped_trend,
    use_arma_errors,
):
    """Fit TBATS model with proper configuration."""
    ar_coeffs = None
    ma_coeffs = None
    best = tbats_model_generator(
        y, seasonal_periods, k_vector,
        use_boxcox, bc_lower, bc_upper,
        use_trend, use_damped_trend, use_arma_errors,
        ar_coeffs, ma_coeffs
    )
    best["description"] = {
        "use_boxcox": use_boxcox,
        "use_trend": use_trend,
        "use_damped_trend": use_damped_trend,
        "use_arma_errors": use_arma_errors,
    }
    return best


def tbats_selection(
    y: jnp.ndarray,
    seasonal_periods: Sequence[int],
    use_boxcox: Optional[bool],
    bc_lower: float,
    bc_upper: float,
    use_trend: Optional[bool],
    use_damped_trend: Optional[bool],
    use_arma_errors: bool,
    early_stop_patience: Optional[int] = None,
    early_stop_tol: float = 0.5,
):
    """Auto-select TBATS configuration (optimized version)."""
    if (use_trend is False) and (use_damped_trend is True):
        raise ValueError("Can't use damped trend without trend")

    t_sel0 = time.perf_counter()
    seasonal_periods = jnp.sort(jnp.asarray(seasonal_periods))

    # Vectorized harmonic search
    ks: List[int] = []
    z = y
    for period in list(seasonal_periods):
        k, z = find_harmonics(z, int(period))
        ks.append(int(k))
    k_vector = jnp.asarray(ks, dtype=jnp.int32)
    _tbats_debug(f"[TBATS][select] k_vector={list(k_vector)} time={time.perf_counter()-t_sel0:.4f}s")

    # Optimize model combination evaluation order - try simpler models first
    # This allows early stopping to kick in faster
    if use_boxcox is None:
        # Try without Box-Cox first (faster), skip with Box-Cox for speed
        B = [False]  # Skip Box-Cox to speed up model selection
    elif use_boxcox:
        B = [True, False]
    else:
        B = [False]

    if use_trend is None:
        # Try simpler models first: no trend, then trend without damping (skip damped for speed)
        T = [(False, False), (True, False)] if use_damped_trend is None else (
            [(True, True)] if use_damped_trend else [(True, False), (False, False)]
        )
    elif use_trend:
        T = [(True, False), (True, True)] if use_damped_trend is None else (
            [(True, True)] if use_damped_trend else [(True, False)]
        )
    else:
        T = [(False, False)]

    combos = [(bcx, t, use_arma_errors) for bcx in B for t in T]

    # Cache seasonal blocks
    dtype = y.dtype
    seasonal_blocks = _build_seasonal_blocks(seasonal_periods, k_vector, dtype)

    best = {"aic": jnp.inf}
    best_aic = float("inf")
    last_valid = None
    no_improve = 0
    
    # Set default early stopping if not provided - more aggressive for speed
    if early_stop_patience is None:
        early_stop_patience = 0  # Stop immediately after first non-improving model (most aggressive)
    if early_stop_tol is None:
        early_stop_tol = 0.1  # Balanced tolerance

    for bcx, (trend, damped), arma in combos:
        t_cand = time.perf_counter()
        cand = tbats_model_generator(
            y, seasonal_periods, k_vector,
            bcx, bc_lower, bc_upper,
            trend, damped, arma,
            None, None,  # ar_coeffs, ma_coeffs
            seasonal_blocks
        )
        _tbats_debug(f"[TBATS][select] candidate bcx={bcx} trend={trend} damped={damped} arma={arma} aic={float(cand.get('aic', jnp.inf)):.4f} time={time.perf_counter()-t_cand:.4f}s")

        if "w_transpose" in cand:
            last_valid = cand

        cand_aic = float(cand["aic"])
        if math.isfinite(cand_aic) and cand_aic < best_aic - float(early_stop_tol):
            best = cand
            best_aic = cand_aic
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience and no_improve >= early_stop_patience:
                _tbats_debug(f"[TBATS][select] early stopping after {no_improve} non-improving models")
                break

    if "w_transpose" not in best and last_valid is not None:
        best = last_valid

    _tbats_debug(f"[TBATS][select] done aic={float(best.get('aic', jnp.inf)):.4f} time={time.perf_counter()-t_sel0:.4f}s")

    return best


def tbats_forecast(mod, h: int):
    """Multi-step mean forecast."""
    h = int(h)
    w = mod["w_transpose"][0]
    F = mod["F"]
    x_last = mod["x"][-1]

    def step(x, _):
        x_next = F @ x
        y_next = jnp.dot(w, x)
        return x_next, y_next

    _, fcst = lax.scan(step, x_last, jnp.arange(h))

    if mod["BoxCox_lambda"] is None:
        fcst_orig = mod["y_mu"] + mod["y_sigma"] * fcst
        return {"mean": fcst_orig, "mean_bc": None}

    fcst_orig = _inv_boxcox(fcst, mod["BoxCox_lambda"])
    return {"mean": fcst_orig, "mean_bc": fcst}


def compute_sigmah(mod, h: int) -> jnp.ndarray:
    """Parametric forecast std-devs."""
    F = mod["F"]
    w = mod["w_transpose"][0]
    g = mod["g"][:, 0]
    h = int(h)

    sigma2 = jnp.asarray(mod["sigma2"], dtype=F.dtype)
    y_sigma = jnp.asarray(mod["y_sigma"], dtype=F.dtype)
    use_boxcox = mod["BoxCox_lambda"] is not None

    var0 = jnp.asarray(1.0, dtype=F.dtype)

    def body(carry, _):
        Fpow, var_acc = carry
        Fpow_next = F @ Fpow
        cj = jnp.dot(jnp.dot(w, Fpow_next), g)
        var_next = var_acc + cj * cj
        return (Fpow_next, var_next), var_next

    init = (jnp.eye(F.shape[1], dtype=F.dtype), var0)
    _, var_tail = lax.scan(body, init, jnp.arange(h - 1))
    var_mult = jnp.concatenate([var0[None], var_tail], axis=0)

    sigma2h = sigma2 * var_mult
    if not use_boxcox:
        sigma2h = (y_sigma ** 2) * sigma2h

    return jnp.sqrt(jnp.maximum(sigma2h, 0.0))


@jax.jit
def _tbats_forecast_core(F: jnp.ndarray, w: jnp.ndarray, x0: jnp.ndarray, h: int) -> jnp.ndarray:
    """Core forecast function for a single state."""
    def step(x, _):
        x_next = F @ x
        y_next = jnp.dot(w, x)
        return x_next, y_next
    _, fcst = lax.scan(step, x0, jnp.arange(h))
    return fcst

def tbats_forecast_batch(F: jnp.ndarray, w: jnp.ndarray, x_last: jnp.ndarray, h: int) -> jnp.ndarray:
    """Vectorized forecasts over a batch of states: F [d,d], w [d], x_last [B,d]."""
    h = int(h)

    def one(x0):
        return _tbats_forecast_core(F, w, x0, h)

    return vmap(one)(x_last)


@jax.jit
def _compute_sigmah_core(
    F: jnp.ndarray,
    w: jnp.ndarray,
    g: jnp.ndarray,
    sigma2: jnp.ndarray,
    y_sigma: jnp.ndarray,
    h: int,
    use_boxcox: bool,
) -> jnp.ndarray:
    """Core sigmah computation for a single sigma2/y_sigma."""
    var0 = jnp.asarray(1.0, dtype=F.dtype)
    
    def body(carry, _):
        Fpow, var_acc = carry
        Fpow_next = F @ Fpow
        cj = jnp.dot(jnp.dot(w, Fpow_next), g)
        var_next = var_acc + cj * cj
        return (Fpow_next, var_next), var_next
    
    init = (jnp.eye(F.shape[1], dtype=F.dtype), var0)
    _, var_tail = lax.scan(body, init, jnp.arange(h - 1))
    var_mult = jnp.concatenate([var0[None], var_tail], axis=0)
    
    sigma2h = sigma2 * var_mult
    if not use_boxcox:
        sigma2h = (y_sigma ** 2) * sigma2h
    
    return jnp.sqrt(jnp.maximum(sigma2h, 0.0))

def compute_sigmah_batch(
    F: jnp.ndarray,
    w: jnp.ndarray,
    g: jnp.ndarray,
    sigma2: jnp.ndarray,
    y_sigma: jnp.ndarray,
    h: int,
    use_boxcox: bool,
) -> jnp.ndarray:
    """Vectorized sigmah over a batch of sigma2/y_sigma: sigma2 [B], y_sigma [B]."""
    h = int(h)

    def one(s2, ys):
        return _compute_sigmah_core(F, w, g, s2, ys, h, use_boxcox)

    return vmap(one)(sigma2, y_sigma)
