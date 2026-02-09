"""
tbats_core.py — TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) core.

Pure-JAX implementation with optax L-BFGS optimisation.  All heavy paths are
JIT-compiled and the module-level solver constant enables XLA cache reuse
across warm calls.

Public API
----------
- find_harmonics        : AIC-based harmonic count selection
- tbats_model_generator : fit a single TBATS specification
- tbats_model           : convenience wrapper (no ARMA)
- tbats_selection       : auto-select the best TBATS configuration
- tbats_forecast        : multi-step mean forecast
- compute_sigmah        : parametric forecast standard deviations
- tbats_forecast_batch  : vectorised forecast over a batch of states
- compute_sigmah_batch  : vectorised sigmah over a batch
"""

from __future__ import annotations

import math
import os
import time
import warnings
from functools import lru_cache, partial
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import config, lax, vmap

config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# Configuration & Debug
# ═══════════════════════════════════════════════════════════════════════

_TBATS_DEBUG = os.environ.get("CHRONAX_TBATS_DEBUG") == "1"


def _tbats_debug(msg: str) -> None:
    """Print *msg* when ``CHRONAX_TBATS_DEBUG=1``."""
    if _TBATS_DEBUG:
        print(msg)


# ═══════════════════════════════════════════════════════════════════════
# Box-Cox Utilities
# ═══════════════════════════════════════════════════════════════════════

def _ensure_pos(y: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Shift *y* so every element is strictly positive (JAX-safe)."""
    y = jnp.asarray(y)
    finite = jnp.isfinite(y)
    y_min_pos = jnp.min(jnp.where((y > 0) & finite, y, jnp.inf))
    base = jnp.where(jnp.isfinite(y_min_pos), jnp.minimum(y_min_pos, 1.0), 1.0)
    tiny = jnp.maximum(eps, base * 1e-8)
    return jnp.where(y <= 0, y + tiny - jnp.minimum(y, 0.0), y)


def _ensure_pos_strict(y: jnp.ndarray) -> jnp.ndarray:
    """Raise if *y* contains non-positive values (for user-facing checks)."""
    y = jnp.asarray(y)
    if bool(jnp.any(y <= 0)):
        raise ValueError("Box–Cox requires strictly positive values (y > 0).")
    return y


def _boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    """Box-Cox transform (applies ``_ensure_pos`` first)."""
    if lam is None:
        return y
    y = _ensure_pos(y)
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.log(y), (jnp.power(y, lam) - 1.0) / lam)


def _boxcox_raw(y: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """Box-Cox transform **without** ``_ensure_pos`` — use when *y* is already positive."""
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.log(y), (jnp.power(y, lam) - 1.0) / lam)


def _inv_boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    """Inverse Box-Cox transform."""
    if lam is None:
        return y
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.exp(y), jnp.power(y * lam + 1.0, 1.0 / lam))


# ── Guerrero lambda selection ──────────────────────────────────────────

def _guerrero_lambda(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> float:
    """Select Box-Cox lambda via the Guerrero CV method."""
    y = jnp.asarray(y)
    n = y.shape[0]
    n_periods = n // season_length
    if n_periods < 2:
        return 1.0

    y_trim = y[: n_periods * season_length].reshape(n_periods, season_length)
    lambdas = jnp.linspace(lower, upper, 41)
    lambdas = jnp.unique(jnp.concatenate([lambdas, jnp.asarray([1.0], dtype=lambdas.dtype)]))

    def cv_for_lambda(lam: jnp.ndarray) -> jnp.ndarray:
        # _boxcox_raw is safe here: caller passes already-positive y_pos
        yt = _boxcox_raw(y_trim.ravel(), lam).reshape(n_periods, season_length)
        stds = jnp.std(yt, axis=1)
        means = jnp.abs(jnp.mean(yt, axis=1)) + 1e-10
        return jnp.std(stds / means)

    cvs = vmap(cv_for_lambda)(lambdas)
    best_idx = jnp.argmin(cvs)
    return float(jnp.clip(lambdas[best_idx], lower, upper))


_GUERRERO_CACHE_MAXSIZE = 32
_GUERRERO_CACHE: Dict[Tuple[int, int, float, float, float, float], float] = {}


def _guerrero_lambda_cached(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> float:
    """Cached wrapper around ``_guerrero_lambda``."""
    y = jnp.asarray(y)
    key = (int(y.shape[0]), int(season_length), float(lower), float(upper),
           float(jnp.sum(y)), float(jnp.mean(y)))
    hit = _GUERRERO_CACHE.get(key)
    if hit is not None:
        return hit
    val = _guerrero_lambda(y, season_length, lower, upper)
    if len(_GUERRERO_CACHE) >= _GUERRERO_CACHE_MAXSIZE:
        _GUERRERO_CACHE.pop(next(iter(_GUERRERO_CACHE)))
    _GUERRERO_CACHE[key] = val
    return val


# ═══════════════════════════════════════════════════════════════════════
# Numeric Utilities
# ═══════════════════════════════════════════════════════════════════════

def _ridge_solve(X: jnp.ndarray, y: jnp.ndarray, ridge: float = 1e-8) -> jnp.ndarray:
    """Ridge-regularised least-squares solve: ``(X^T X + ridge·I)^{-1} X^T y``."""
    XT = jnp.transpose(X)
    d = X.shape[1]
    return jnp.linalg.solve(XT @ X + ridge * jnp.eye(d, dtype=X.dtype), XT @ y)


# ═══════════════════════════════════════════════════════════════════════
# Harmonic Selection
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("max_h", "tol_no_improv"))
def _select_harmonics_fast(
    z: jnp.ndarray, X_full: jnp.ndarray, max_h: int, tol_no_improv: int,
) -> jnp.ndarray:
    """JIT-compiled AIC-based harmonic count selector with early stopping."""
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

    state = lax.fori_loop(1, max_h + 1, body, (best_aic, k_best, aic_prev, wout, stopped))
    return state[1]  # k_best


_HARMONICS_CACHE: Dict[Tuple, Tuple[int, jnp.ndarray]] = {}
_HARMONICS_CACHE_MAXSIZE = 32


def find_harmonics(y: jnp.ndarray, m: int) -> Tuple[int, jnp.ndarray]:
    """Find optimal number of harmonics for period *m* using AIC.

    Results are cached by ``(n, m, sum, std)`` so warm runs are free.

    Returns
    -------
    (k, z_deseasonalised) : int, jnp.ndarray
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = len(y)
    
    # Cache lookup
    cache_key = (n, int(m), float(jnp.sum(y)), float(jnp.std(y)))
    cached = _HARMONICS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Rolling mean via cumsum (replaces pandas)
    window_size = 2 * m
    if n < window_size:
        f_t = jnp.full(n, jnp.mean(y), dtype=jnp.float64)
    else:
        cumsum = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), y]))
        indices = jnp.arange(n)
        lo = jnp.maximum(0, indices + 1 - window_size)
        hi = indices + 1
        f_t = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    z = y - f_t  # detrend
    
    # Determine max harmonics
    max_harmonics = m // 2 if m % 2 == 0 else (m - 1) // 2
    max_harmonics = min(max_harmonics, n)

    if max_harmonics == 0:
        result = (1, y)
        _harmonics_cache_put(cache_key, result)
        return result
    
    # Vectorised Fourier terms
    t = jnp.arange(n, dtype=jnp.float64)
    harmonics = jnp.arange(1, max_harmonics + 1, dtype=jnp.float64)
    angles = 2.0 * jnp.pi * jnp.outer(t, harmonics) / m
    fourier = jnp.zeros((n, 2 * max_harmonics), dtype=jnp.float64)
    fourier = fourier.at[:, 0::2].set(jnp.cos(angles))
    fourier = fourier.at[:, 1::2].set(jnp.sin(angles))

    # AIC selection (JIT-compiled)
    num_harmonics = max(1, int(_select_harmonics_fast(z, fourier, max_harmonics, 2)))

    # Deseasonalise with the chosen harmonics
    X_best = fourier[:, : 2 * num_harmonics]
    z_deseasonalised = z - X_best @ _ridge_solve(X_best, z, ridge=1e-8)

    result = (num_harmonics, z_deseasonalised)
    _harmonics_cache_put(cache_key, result)
    return result


def _harmonics_cache_put(key: Tuple, value: Tuple[int, jnp.ndarray]) -> None:
    """Bounded LRU-style insert into the harmonics cache."""
    if len(_HARMONICS_CACHE) >= _HARMONICS_CACHE_MAXSIZE:
        _HARMONICS_CACHE.pop(next(iter(_HARMONICS_CACHE)))
    _HARMONICS_CACHE[key] = value


# ═══════════════════════════════════════════════════════════════════════
# Seasonal Block Construction
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=32)
def _build_seasonal_blocks_cached(
    seasonal_periods: Tuple[int, ...],
    k_vector: Tuple[int, ...],
    dtype_str: str,
) -> jnp.ndarray:
    """Build and cache the block-diagonal trigonometric rotation matrix."""
    dtype = jnp.float64 if dtype_str == "float64" else jnp.float32
    tau = int(2 * sum(k_vector))
    A = jnp.zeros((tau, tau), dtype=dtype)
    pos = 0
    for k, period in zip(k_vector, seasonal_periods):
        t = 2.0 * jnp.pi * (jnp.arange(1, k + 1, dtype=dtype) / period)
        ck = jnp.diag(jnp.cos(t))
        sk = jnp.diag(jnp.sin(t))
        Ak = jnp.vstack([jnp.hstack([ck, sk]), jnp.hstack([-sk, ck])])
        A = A.at[pos: pos + 2 * k, pos: pos + 2 * k].set(Ak)
        pos += 2 * k
    return A


def _build_seasonal_blocks(
    seasonal_periods: jnp.ndarray, k_vector: jnp.ndarray, dtype,
) -> jnp.ndarray:
    """Public entry: converts arrays → tuples then delegates to the cached builder."""
    sp = tuple(int(x) for x in list(seasonal_periods))
    kv = tuple(int(x) for x in list(k_vector))
    dtype_str = "float64" if dtype == jnp.float64 else "float32"
    return _build_seasonal_blocks_cached(sp, kv, dtype_str)


# ═══════════════════════════════════════════════════════════════════════
# State-Space Matrix Builders
# ═══════════════════════════════════════════════════════════════════════

# ── Initial construction ───────────────────────────────────────────────

def make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau: int, beta, dtype) -> jnp.ndarray:
    """Build the observation row-vector w^T  (shape ``(1, d)``)."""
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
        kv = jnp.asarray(k_vector, dtype=jnp.int32)
        if kv.size > 0:
            idx = jnp.arange(tau, dtype=jnp.int32)
            starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), 2 * kv[:-1]]))
            mids = starts + kv
            mask = jnp.sum((idx[None, :] >= starts[:, None]) & (idx[None, :] < mids[:, None]), axis=0)
            w = w.at[0, start: start + tau].set(mask.astype(dtype))
    return w


def make_g(k_vector, alpha, beta, p: int, q: int, tau: int, dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build the gain vector g  (shape ``(d, 1)``) and the gamma-bold row."""
    adj_phi = 1 if beta is not None else 0
    d = 1 + adj_phi + tau + p + q

    g = jnp.zeros((d, 1), dtype=dtype).at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gamma_bold = jnp.zeros((1, 2 * int(jnp.sum(jnp.asarray(k_vector)))), dtype=dtype)
    return g, gamma_bold


def make_F(
    phi, tau: int, alpha, beta, ar_coeffs, ma_coeffs,
    gamma_bold, seasonal_periods, k_vector, dtype,
    seasonal_blocks: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Build the transition matrix F."""
    adj_phi = 1 if beta is not None else 0
    phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)

    # Level row
    F = jnp.array([[1.0]], dtype=dtype)
    if adj_phi:
        F = jnp.hstack([F, jnp.array([[phi_eff]], dtype=dtype)])
    F = jnp.hstack([F, jnp.zeros((1, tau), dtype=dtype)])
    if ar_coeffs is not None and ar_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ar_coeffs, dtype=dtype))[None, :]])
    if ma_coeffs is not None and ma_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ma_coeffs, dtype=dtype))[None, :]])

    # Trend row
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
    
    A = seasonal_blocks if seasonal_blocks is not None else _build_seasonal_blocks(seasonal_periods, k_vector, dtype)
    seasonal = jnp.hstack([seasonal, A])

    if ar_coeffs is not None and ar_coeffs.size > 0:
        B = jnp.dot(jnp.transpose(gamma_bold), jnp.asarray(ar_coeffs, dtype=dtype)[None, :])
        seasonal = jnp.hstack([seasonal, B])
    if ma_coeffs is not None and ma_coeffs.size > 0:
        C = jnp.dot(jnp.transpose(gamma_bold), jnp.asarray(ma_coeffs, dtype=dtype)[None, :])
        seasonal = jnp.hstack([seasonal, C])

    F = jnp.vstack([F, seasonal])

    # AR companion rows
    if ar_coeffs is not None and ar_coeffs.size > 0:
        p = int(ar_coeffs.size)
        ar_rows = jnp.zeros((p, 1), dtype=dtype)
        if adj_phi:
            ar_rows = jnp.hstack([ar_rows, jnp.zeros((p, 1), dtype=dtype)])
        ar_rows = jnp.hstack([ar_rows, jnp.zeros((p, tau), dtype=dtype)])
        if p > 1:
            ident = jnp.hstack([jnp.eye(p - 1, dtype=dtype), jnp.zeros((p - 1, 1), dtype=dtype)])
        else:
            ident = jnp.zeros((0, p), dtype=dtype)
        ar_part = jnp.vstack([jnp.asarray(ar_coeffs, dtype=dtype)[None, :], ident])
        ar_rows = jnp.hstack([ar_rows, ar_part])
        if ma_coeffs is not None and ma_coeffs.size > 0:
            q = int(ma_coeffs.size)
            ma_in_ar = jnp.zeros((p, q), dtype=dtype).at[0, :].set(jnp.asarray(ma_coeffs, dtype=dtype))
            ar_rows = jnp.hstack([ar_rows, ma_in_ar])
        F = jnp.vstack([F, ar_rows])

    # MA companion rows
    if ma_coeffs is not None and ma_coeffs.size > 0:
        q = int(ma_coeffs.size)
        ma_rows = jnp.zeros((q, 1), dtype=dtype)
        if adj_phi:
            ma_rows = jnp.hstack([ma_rows, jnp.zeros((q, 1), dtype=dtype)])
        ma_rows = jnp.hstack([ma_rows, jnp.zeros((q, tau), dtype=dtype)])
        if ar_coeffs is not None and ar_coeffs.size > 0:
            ma_rows = jnp.hstack([ma_rows, jnp.zeros((q, int(ar_coeffs.size)), dtype=dtype)])
        if q > 1:
            ident = jnp.hstack([jnp.eye(q - 1, dtype=dtype), jnp.zeros((q - 1, 1), dtype=dtype)])
            ma_part = jnp.vstack([jnp.zeros((1, q), dtype=dtype), ident])
        else:
            ma_part = jnp.zeros((1, q), dtype=dtype)
        ma_rows = jnp.hstack([ma_rows, ma_part])
        F = jnp.vstack([F, ma_rows])

    return F


# ── In-place updates (used during optimisation) ───────────────────────

def update_w(w, phi, tau: int, ar_coeffs, ma_coeffs, p: int, q: int, beta, dtype) -> jnp.ndarray:
    """Return *w* with the phi entry updated."""
    adj_phi = 1 if beta is not None else 0
    if adj_phi:
        phi_eff = jnp.asarray(0.0 if phi is None else phi, dtype=dtype)
        w = w.at[0, 1].set(phi_eff)
    return w


def update_g(g, gamma_bold, alpha, beta, k_vector, gamma_one_v, gamma_two_v, dtype) -> jnp.ndarray:
    """Return *g* with alpha, beta and seasonal gains updated."""
    adj_phi = 1 if beta is not None else 0
    g = g.at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gb = jnp.zeros_like(gamma_bold)
    if gamma_bold.shape[1] > 0:
        kv = jnp.asarray(k_vector, dtype=jnp.int32)
        g1 = jnp.asarray(gamma_one_v, dtype=dtype)
        g2 = jnp.asarray(gamma_two_v, dtype=dtype)
        if kv.size > 0:
            tau = gamma_bold.shape[1]
            idx = jnp.arange(tau, dtype=jnp.int32)
            starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), 2 * kv[:-1]]))
            mids = starts + kv
            mask1 = (idx[None, :] >= starts[:, None]) & (idx[None, :] < mids[:, None])
            mask2 = (idx[None, :] >= mids[:, None]) & (idx[None, :] < (mids + kv)[:, None])
            gb_row = jnp.sum(mask1 * g1[:, None] + mask2 * g2[:, None], axis=0)
            gb = gb.at[0, :].set(gb_row)

    start = 1 + adj_phi
    g = g.at[start: start + gb.shape[1], 0].set(gb.ravel())
    return g


def update_F(F, phi, alpha, beta, gamma_bold, ar_coeffs, ma_coeffs, p: int, q: int, tau: int, dtype) -> jnp.ndarray:
    """Return *F* with phi, alpha, beta and ARMA entries updated."""
    phi_eff = jnp.asarray(0.0 if phi is None else phi, dtype=dtype)
    adj = 1 if beta is not None else 0

    if adj:
        F = F.at[0, 1].set(phi_eff)
        F = F.at[1, 1].set(phi_eff)

    if ar_coeffs is not None and p > 0:
        ar = jnp.asarray(ar_coeffs, dtype=dtype)
        F = F.at[0, adj + tau + 1: adj + tau + p + 1].set(alpha * ar)
        if adj:
            F = F.at[1, adj + tau + 1: adj + tau + p + 1].set(beta * ar)
        if tau > 0:
            B = jnp.dot(gamma_bold.reshape(-1, 1), ar.reshape(1, -1))
            F = F.at[1 + adj: adj + tau + 1, adj + tau + 1: adj + tau + p + 1].set(B)
        F = F.at[adj + tau + 1, adj + tau + 1: adj + tau + p + 1].set(ar)

    if ma_coeffs is not None and q > 0:
        ma = jnp.asarray(ma_coeffs, dtype=dtype)
        F = F.at[0, adj + tau + p + 1: adj + tau + p + q + 1].set(alpha * ma)
        if adj:
            F = F.at[1, adj + tau + p + 1: adj + tau + p + q + 1].set(beta * ma)
        if tau > 0:
            C = jnp.dot(gamma_bold.reshape(-1, 1), ma.reshape(1, -1))
            F = F.at[1 + adj: adj + tau + 1, adj + tau + p + 1: adj + tau + p + q + 1].set(C)
        if ar_coeffs is not None and p > 0:
            F = F.at[adj + tau + 1, adj + tau + p + 1: adj + tau + p + q + 1].set(ma)
    return F


# ═══════════════════════════════════════════════════════════════════════
# Innovations (Kalman) Filter
# ═══════════════════════════════════════════════════════════════════════

def _calc_filter_impl(
    y: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
    F: jnp.ndarray, x0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Innovations filter (shared implementation).

    Returns ``(fitted, errors, x_final)`` where ``x_final`` has shape ``(1, d)``.
    """
    w0 = w[0]
    g0 = g[:, 0]

    def step(x_prev, y_t):
        yhat_t = jnp.dot(w0, x_prev)
        e_t = y_t - yhat_t
        x_t = F @ x_prev + g0 * e_t
        return x_t, (yhat_t, e_t)

    xT, (yhat_seq, e_seq) = lax.scan(step, x0, y)
    return yhat_seq, e_seq, xT[None, :]


@jax.jit
def _calc_filter(
    y: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
    F: jnp.ndarray, x0: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled innovations filter (casts all inputs to float64)."""
    dtype = jnp.float64
    return _calc_filter_impl(
        jnp.asarray(y, dtype=dtype), jnp.asarray(w, dtype=dtype),
        jnp.asarray(g, dtype=dtype), jnp.asarray(F, dtype=dtype),
        jnp.asarray(x0, dtype=dtype),
    )


# Alias: use inside an outer JIT context (avoids nested-jit overhead)
_calc_filter_core = _calc_filter_impl


# ═══════════════════════════════════════════════════════════════════════
# L-BFGS Parameter Optimisation
# ═══════════════════════════════════════════════════════════════════════

_N_OPTIM_STEPS = 30

_TBATS_SOLVER = optax.lbfgs(
    memory_size=5,
    linesearch=optax.scale_by_zoom_linesearch(
        max_linesearch_steps=10,
        initial_guess_strategy="one",
    ),
)


@partial(jax.jit, static_argnames=(
    "use_boxcox", "use_trend", "use_damped_trend", "p", "q", "tau", "n_k",
))
def _run_lbfgs_optim(
    u0: jnp.ndarray, scale_vec: jnp.ndarray,
    w: jnp.ndarray, g: jnp.ndarray, F: jnp.ndarray,
    gamma_bold: jnp.ndarray, k_vector_arr: jnp.ndarray,
    x0_hat: jnp.ndarray, x0_utp: jnp.ndarray,
    y_fit: jnp.ndarray, y_pos: jnp.ndarray, y_pos_log_sum: jnp.ndarray,
    bc_lower: jnp.ndarray, bc_upper: jnp.ndarray,
    # ── static args (hashed, not traced) ──
    use_boxcox: bool, use_trend: bool, use_damped_trend: bool,
    p: int, q: int, tau: int, n_k: int,
) -> jnp.ndarray:
    """Top-level JIT-cached L-BFGS optimisation of the TBATS log-likelihood.

    Being a module-level function (not a closure) lets JAX cache the compiled
    XLA kernel across warm runs when the static args and array shapes match.
    """
    dtype = jnp.float64

    def obj(u: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.where(jnp.isfinite(u * scale_vec), u * scale_vec, 0.0)

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
            beta_opt = theta[idx]; idx += 1
            phi_opt = (theta[idx] if use_damped_trend and idx < theta.size else 1.0)
            if use_damped_trend and idx < theta.size:
                idx += 1
        else:
            beta_opt = None
            phi_opt = None

        g1 = theta[idx: idx + n_k]; idx += n_k
        g2 = theta[idx: idx + n_k]; idx += n_k
        ar_opt = theta[idx: idx + p] if p > 0 else None
        ma_opt = theta[idx + p: idx + p + q] if q > 0 else None

        w_opt = update_w(w, phi_opt, tau, ar_opt, ma_opt, p, q, beta_opt, dtype)
        g_opt = update_g(g, gamma_bold, alpha_opt, beta_opt, k_vector_arr, g1, g2, dtype)
        F_opt = update_F(F, phi_opt, alpha_opt, beta_opt, gamma_bold, ar_opt, ma_opt, p, q, tau, dtype)

        if use_boxcox:
            x0_opt = _boxcox_raw(x0_utp, lam_opt)
            y_opt = _boxcox_raw(y_pos, lam_opt)
        else:
            x0_opt = x0_hat
            y_opt = y_fit

        _, e, _ = _calc_filter_core(y_opt, w_opt, g_opt, F_opt, x0_opt)

        n = y_opt.shape[0]
        sse = jnp.sum(e * e)
        ll = n * jnp.log(sse + 1e-12)
        if use_boxcox:
            ll = ll - 2.0 * (lam_opt - 1.0) * y_pos_log_sum
        return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(1e20, dtype=dtype))

    opt_state0 = _TBATS_SOLVER.init(u0)
    val_and_grad_fn = jax.value_and_grad(obj)

    def _optim_step(carry, _):
        u, opt_state, best_u, best_loss = carry
        loss, grads = val_and_grad_fn(u)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_opt_state = _TBATS_SOLVER.update(
            grads, opt_state, u, value=loss, grad=grads, value_fn=obj,
        )
        new_u = optax.apply_updates(u, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_u = jnp.where(improved, u, best_u)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_u, new_opt_state, best_u, best_loss), None

    init_loss = obj(u0)
    init_carry = (u0, opt_state0, u0, init_loss)
    (_, _, best_u, _), _ = lax.scan(_optim_step, init_carry, jnp.arange(_N_OPTIM_STEPS))
    return best_u


# ═══════════════════════════════════════════════════════════════════════
# Model Fitting
# ═══════════════════════════════════════════════════════════════════════

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
) -> Dict:
    """Fit a single TBATS specification (Box-Cox + optimisation + filter)."""
    dtype = jnp.float64
    y = jnp.asarray(y, dtype=dtype)
    
    # ── Box-Cox lambda estimation (Guerrero method) ────────────────────
    if use_boxcox:
        y_pos = _ensure_pos(y)
        if jnp.any(y <= 0):
            warnings.warn("Data contains zero/negative values; disabling Box-Cox.")
            use_boxcox = False
            lam = None
            y_fit = y
        else:
            season_length = int(seasonal_periods[0]) if len(seasonal_periods) > 0 else 1
            lam = _guerrero_lambda_cached(y_pos, season_length, bc_lower, bc_upper)
            y_fit = _boxcox(y_pos, lam)
    else:
        lam = None
        y_fit = y

        y_mu = jnp.asarray(0.0, dtype=dtype)
        y_sigma = jnp.asarray(1.0, dtype=dtype)

    p = 0 if ar_coeffs is None else int(ar_coeffs.shape[0])
    q = 0 if ma_coeffs is None else int(ma_coeffs.shape[0])
    tau = int(2 * jnp.sum(k_vector))

    # ── Initial parameters ─────────────────────────────────────────────
    alpha = jnp.asarray(0.09, dtype=dtype)
    
    if use_trend:
        adj_phi = 1
        beta = jnp.asarray(0.05, dtype=dtype)
        phi = jnp.asarray(0.999 if use_damped_trend else 1.0, dtype=dtype)
    else:
        adj_phi = 0
        beta = None
        phi = None

    gamma_one_v = jnp.zeros(len(k_vector), dtype=dtype)
    gamma_two_v = jnp.zeros(len(k_vector), dtype=dtype)

    # ── Build state-space matrices ─────────────────────────────────────
    if seasonal_blocks is None:
        seasonal_blocks = _build_seasonal_blocks(seasonal_periods, k_vector, dtype)

    w = make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau, beta, dtype)
    g, gamma_bold = make_g(k_vector, alpha, beta, p, q, tau, dtype)
    F = make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
               gamma_bold, seasonal_periods, k_vector, dtype, seasonal_blocks)

    # ── Seed state (level + trend + seasonal + ARMA) ───────────────────
    n = y_fit.shape[0]
    level_init = float(jnp.mean(y_fit)) if n > 0 else 0.0
    
    x0_ls = jnp.zeros((1 + adj_phi + tau,), dtype=dtype)
    x0_ls = x0_ls.at[0].set(level_init)
    
    if use_trend and beta is not None and n > 1:
        trend_init = float((y_fit[-1] - y_fit[0]) / (n - 1))
        x0_ls = x0_ls.at[1].set(trend_init)
    
    x0_hat = jnp.concatenate([x0_ls, jnp.zeros(p + q, dtype=dtype)]) if (p or q) else x0_ls

    # ── Pack initial parameter vector + scales ─────────────────────────
    params: list = []
    scale: list = []
    if use_boxcox:
        params.extend([lam, alpha]); scale.extend([0.001, 0.01])
    else:
        params.append(alpha); scale.append(0.01)
    if beta is not None:
        params.append(beta); scale.append(0.01)
    if phi is not None and float(phi) != 1.0:
        params.append(phi); scale.append(0.01)
    params.extend([gamma_one_v, gamma_two_v])
    scale.extend([1e-5] * (len(gamma_one_v) + len(gamma_two_v)))
    if ar_coeffs is not None:
        params.append(ar_coeffs); scale.extend([0.1] * len(ar_coeffs))
    if ma_coeffs is not None:
        params.append(ma_coeffs); scale.extend([0.1] * len(ma_coeffs))

    params_vec = jnp.concatenate([
        _p.ravel() if isinstance(_p, jnp.ndarray) else jnp.asarray([_p], dtype=dtype)
        for _p in params
    ]).astype(dtype)
    scale_vec = jnp.asarray(scale, dtype=dtype)

    # ── Pre-compute quantities for optimiser ───────────────────────────
    if use_boxcox:
        x0_untransformed_pos = _ensure_pos(_inv_boxcox(x0_hat, lam))
        y_pos_log_sum = jnp.sum(jnp.log(jnp.clip(y_pos, 1e-12, jnp.inf)))
    else:
        x0_untransformed_pos = x0_hat
        y_pos_log_sum = jnp.asarray(0.0, dtype=dtype)

    u0 = jnp.asarray(params_vec / scale_vec, dtype=dtype)
    n_k = len(k_vector)

    # Dummies keep JIT signatures consistent regardless of Box-Cox branch
    _y_pos = y_pos if use_boxcox else y_fit
    _y_fit = y_fit if not use_boxcox else y_pos

    try:
        uhat = _run_lbfgs_optim(
            u0, scale_vec, w, g, F, gamma_bold, k_vector,
            x0_hat, x0_untransformed_pos, _y_fit, _y_pos, y_pos_log_sum,
            jnp.asarray(bc_lower, dtype=dtype), jnp.asarray(bc_upper, dtype=dtype),
            use_boxcox=use_boxcox, use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            p=p, q=q, tau=tau, n_k=n_k,
        )
    except Exception as exc:
        warnings.warn(f"Optimisation failed ({exc}); using initial parameters")
        uhat = u0

    # ── Unpack optimised parameters ────────────────────────────────────
    optim_params = jnp.asarray(uhat, dtype=dtype) * scale_vec
    if use_boxcox:
        optim_params = optim_params.at[0].set(jnp.clip(optim_params[0], bc_lower, bc_upper))

    idx = 0
    if use_boxcox:
        optim_lambda = float(optim_params[idx]); idx += 1
        optim_alpha = float(optim_params[idx]); idx += 1
    else:
        optim_lambda = None
        optim_alpha = float(optim_params[idx]); idx += 1

    if use_trend:
        optim_beta = float(optim_params[idx]); idx += 1
        if use_damped_trend and idx < optim_params.size:
            optim_phi = float(optim_params[idx]); idx += 1
        else:
            optim_phi = 1.0
    else:
        optim_beta = None
        optim_phi = None

    g1 = optim_params[idx: idx + len(k_vector)]; idx += len(k_vector)
    g2 = optim_params[idx: idx + len(k_vector)]; idx += len(k_vector)
    optim_ar = optim_params[idx: idx + p] if p > 0 else None
    optim_ma = optim_params[idx + p: idx + p + q] if q > 0 else None

    # ── Rebuild final matrices & run filter ────────────────────────────
    w_final = update_w(w, optim_phi, tau, optim_ar, optim_ma, p, q, optim_beta, dtype)
    g_final = update_g(g, gamma_bold, optim_alpha, optim_beta, k_vector, g1, g2, dtype)
    F_final = update_F(F, optim_phi, optim_alpha, optim_beta, gamma_bold,
                       optim_ar, optim_ma, p, q, tau, dtype)

    if use_boxcox:
        x0_final = _boxcox_raw(x0_untransformed_pos, optim_lambda)
        y_fit_final = _boxcox_raw(y_pos, optim_lambda)
    else:
        x0_final = x0_hat
        y_fit_final = y_fit

    fitted, errors, x_seq = _calc_filter(y_fit_final, w_final, g_final, F_final, x0_final)
    sigma2 = jnp.mean(errors * errors)

    # ── Log-likelihood & AIC ───────────────────────────────────────────
    n_eff = errors.shape[0]
    log_likelihood = float(n_eff * jnp.log(sigma2 + 1e-12))
    if use_boxcox and lam is not None:
        log_likelihood -= 2.0 * (lam - 1.0) * float(y_pos_log_sum)

    kval = int(optim_params.size + x0_final.shape[0])
    if optim_lambda == 1:
        kval -= 1
    if optim_beta is not None and abs(optim_beta) < 1e-8:
        kval -= 1
    if optim_phi is not None and optim_phi == 1:
        kval -= 1

    aic = float(log_likelihood) + 2 * kval

    return {
        "fitted": fitted,
        "errors": errors[None, :],
        "sigma2": sigma2,
        "aic": aic,
        "optim_params": jnp.asarray(optim_params, dtype=dtype),
        "F": F_final,
        "w_transpose": w_final,
        "g": g_final,
        "x": x_seq,             # shape (1, d) — final state only
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
    y, seasonal_periods, k_vector,
    use_boxcox, bc_lower, bc_upper,
    use_trend, use_damped_trend, use_arma_errors,
) -> Dict:
    """Convenience wrapper: fit a single TBATS spec with no ARMA."""
    best = tbats_model_generator(
        y, seasonal_periods, k_vector,
        use_boxcox, bc_lower, bc_upper,
        use_trend, use_damped_trend, use_arma_errors,
        ar_coeffs=None, ma_coeffs=None,
    )
    best["description"] = {
        "use_boxcox": use_boxcox,
        "use_trend": use_trend,
        "use_damped_trend": use_damped_trend,
        "use_arma_errors": use_arma_errors,
    }
    return best


# ═══════════════════════════════════════════════════════════════════════
# Model Selection
# ═══════════════════════════════════════════════════════════════════════

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
) -> Dict:
    """Auto-select the best TBATS configuration via AIC comparison."""
    if use_trend is False and use_damped_trend is True:
        raise ValueError("Cannot use damped trend without trend")

    t_sel0 = time.perf_counter()
    seasonal_periods = jnp.sort(jnp.asarray(seasonal_periods))

    # ── Harmonic search per season ─────────────────────────────────────
    ks: List[int] = []
    z = y
    for period in list(seasonal_periods):
        k, z = find_harmonics(z, int(period))
        ks.append(int(k))
    k_vector = jnp.asarray(ks, dtype=jnp.int32)
    _tbats_debug(f"[TBATS][select] k_vector={list(k_vector)} "
                 f"time={time.perf_counter() - t_sel0:.4f}s")

    # ── Candidate grid ─────────────────────────────────────────────────
    if use_boxcox is None:
        B = [False]
    elif use_boxcox:
        B = [False, True]  # try simpler first; selection keeps best AIC
    else:
        B = [False]

    if use_trend is None:
        T = ([(False, False), (True, False)] if use_damped_trend is None
             else ([(True, True)] if use_damped_trend else [(True, False), (False, False)]))
    elif use_trend:
        T = ([(True, False), (True, True)] if use_damped_trend is None
             else ([(True, True)] if use_damped_trend else [(True, False)]))
    else:
        T = [(False, False)]

    combos = [(bcx, t, use_arma_errors) for bcx in B for t in T]

    # Pre-build seasonal blocks (shared across all candidates)
    seasonal_blocks = _build_seasonal_blocks(seasonal_periods, k_vector, y.dtype)

    # ── Evaluate candidates ────────────────────────────────────────────
    if early_stop_patience is None:
        early_stop_patience = 0
    if early_stop_tol is None:
        early_stop_tol = 0.1

    best: Dict = {"aic": jnp.inf}
    best_aic = float("inf")
    last_valid = None
    no_improve = 0
    
    for ci, (bcx, (trend, damped), arma) in enumerate(combos):
        # Fast exit: patience-0 and we already have a valid model
        if ci > 0 and early_stop_patience == 0 and math.isfinite(best_aic):
            _tbats_debug(f"[TBATS][select] patience-0 fast exit after {ci} candidate(s)")
            break

        t_cand = time.perf_counter()
        cand = tbats_model_generator(
            y, seasonal_periods, k_vector,
            bcx, bc_lower, bc_upper,
            trend, damped, arma,
            None, None,  # ar_coeffs, ma_coeffs
            seasonal_blocks,
        )
        _tbats_debug(
            f"[TBATS][select] bcx={bcx} trend={trend} damped={damped} arma={arma} "
            f"aic={float(cand.get('aic', jnp.inf)):.4f} "
            f"time={time.perf_counter() - t_cand:.4f}s"
        )

        if "w_transpose" in cand:
            last_valid = cand

        cand_aic = float(cand["aic"])
        if math.isfinite(cand_aic) and cand_aic < best_aic - float(early_stop_tol):
            best = cand
            best_aic = cand_aic
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience is not None and no_improve > early_stop_patience:
                _tbats_debug(f"[TBATS][select] early stopping after {no_improve} non-improving models")
                break

    if "w_transpose" not in best and last_valid is not None:
        best = last_valid

    _tbats_debug(f"[TBATS][select] done aic={float(best.get('aic', jnp.inf)):.4f} "
                 f"time={time.perf_counter() - t_sel0:.4f}s")
    return best


# ═══════════════════════════════════════════════════════════════════════
# Forecasting
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("h",))
def _tbats_forecast_core(F: jnp.ndarray, w: jnp.ndarray, x0: jnp.ndarray, h: int) -> jnp.ndarray:
    """JIT-compiled multi-step forecast for a single initial state."""
    def step(x, _):
        y_next = jnp.dot(w, x)
        x_next = F @ x
        return x_next, y_next
    _, fcst = lax.scan(step, x0, jnp.arange(h))
    return fcst


def tbats_forecast(mod: Dict, h: int) -> Dict[str, jnp.ndarray]:
    """Multi-step mean forecast from a fitted model dictionary."""
    h = int(h)
    w = mod["w_transpose"][0]
    F = mod["F"]
    x_last = mod["x"][-1]

    fcst = _tbats_forecast_core(F, w, x_last, h)

    if mod["BoxCox_lambda"] is None:
        return {"mean": mod["y_mu"] + mod["y_sigma"] * fcst, "mean_bc": None}

    return {"mean": _inv_boxcox(fcst, mod["BoxCox_lambda"]), "mean_bc": fcst}


# ── Prediction intervals ──────────────────────────────────────────────

@partial(jax.jit, static_argnames=("h", "use_boxcox"))
def _compute_sigmah_core(
    F: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
    sigma2: jnp.ndarray, y_sigma: jnp.ndarray,
    h: int, use_boxcox: bool,
) -> jnp.ndarray:
    """JIT-compiled parametric forecast standard deviations."""
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


def compute_sigmah(mod: Dict, h: int) -> jnp.ndarray:
    """Parametric forecast standard deviations from a fitted model dictionary."""
    F = mod["F"]
    w = mod["w_transpose"][0]
    g = mod["g"][:, 0]
    sigma2 = jnp.asarray(mod["sigma2"], dtype=F.dtype)
    y_sigma = jnp.asarray(mod["y_sigma"], dtype=F.dtype)
    use_boxcox = mod["BoxCox_lambda"] is not None
    return _compute_sigmah_core(F, w, g, sigma2, y_sigma, int(h), use_boxcox)


# ═══════════════════════════════════════════════════════════════════════
# Batch Operations (for multi-series workloads)
# ═══════════════════════════════════════════════════════════════════════

def tbats_forecast_batch(
    F: jnp.ndarray, w: jnp.ndarray, x_last: jnp.ndarray, h: int,
) -> jnp.ndarray:
    """Vectorised forecasts: ``x_last`` has shape ``(B, d)``."""
    h = int(h)
    return vmap(lambda x0: _tbats_forecast_core(F, w, x0, h))(x_last)


def compute_sigmah_batch(
    F: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
    sigma2: jnp.ndarray, y_sigma: jnp.ndarray,
    h: int, use_boxcox: bool,
) -> jnp.ndarray:
    """Vectorised sigmah: ``sigma2`` and ``y_sigma`` have shape ``(B,)``."""
    h = int(h)
    return vmap(lambda s2, ys: _compute_sigmah_core(F, w, g, s2, ys, h, use_boxcox))(sigma2, y_sigma)
