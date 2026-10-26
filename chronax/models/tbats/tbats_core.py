"""
tbats_core.py — TBATS (Trigonometric, Box-Cox, ARMA, Trend, Seasonal) core.

Pure-JAX implementation with optax L-BFGS optimisation.  All heavy paths are
JIT-compiled and the module-level solver constant enables XLA cache reuse
across warm calls.

vmap-native design: every data-dependent quantity (Box-Cox lambda, harmonic
counts, AIC selection) stays a traced ``jnp`` value end-to-end.  Because the
selected harmonic count would otherwise determine the state-space DIMENSION,
the state space is built at the static, config-derived maximum harmonic count
(``k_vector_max``) and the traced selected count enters only through gain
masks — inactive harmonics get zero gain and zero seed, so their states stay
exactly 0 and the padded model is numerically identical to the sliced one.

Public API
----------
- find_harmonics        : AIC-based harmonic count selection (traced count)
- tbats_model_generator : fit a single TBATS specification
- tbats_model           : convenience wrapper (no ARMA)
- tbats_selection       : fit the candidate grid, argmin-select by AIC
- tbats_forecast        : multi-step mean forecast
- compute_sigmah        : parametric forecast standard deviations
- tbats_forecast_batch  : vectorised forecast over a batch of states
- compute_sigmah_batch  : vectorised sigmah over a batch
"""

from __future__ import annotations

import os
import time
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

def _guerrero_lambda(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> jnp.ndarray:
    """Select Box-Cox lambda via the Guerrero CV method.

    Returns a ``jnp`` scalar so the selection traces under vmap (the candidate
    lambda grid is config-derived and therefore concrete even under trace).
    """
    y = jnp.asarray(y)
    n = y.shape[0]
    n_periods = n // season_length
    if n_periods < 2:  # static: shape // config
        return jnp.asarray(1.0, dtype=y.dtype)

    y_trim = y[: n_periods * season_length].reshape(n_periods, season_length)
    lambdas = jnp.linspace(lower, upper, 41)
    lambdas = jnp.unique(jnp.concatenate([lambdas, jnp.asarray([1.0], dtype=lambdas.dtype)]))

    def cv_for_lambda(lam: jnp.ndarray) -> jnp.ndarray:
        """Compute the Guerrero coefficient-of-variation score for one lambda."""
        # _boxcox_raw is safe here: caller passes already-positive y_pos
        yt = _boxcox_raw(y_trim.ravel(), lam).reshape(n_periods, season_length)
        stds = jnp.std(yt, axis=1)
        means = jnp.abs(jnp.mean(yt, axis=1)) + 1e-10
        return jnp.std(stds / means)

    cvs = vmap(cv_for_lambda)(lambdas)
    best_idx = jnp.argmin(cvs)
    return jnp.clip(lambdas[best_idx], lower, upper).astype(y.dtype)


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

    def body(h: int, state: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        """Evaluate one harmonic count candidate and update the running best."""
        best_aic, k_best, aic_prev, wout, stopped = state

        def do_step(state_in: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
            """Run the AIC computation for an active harmonic-count candidate."""
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


def _max_harmonics(m: int, n: int) -> int:
    """Static (config + shape derived) harmonic-count ceiling for period *m*.

    Mirrors the bound used inside ``find_harmonics``; ``tbats_selection`` uses
    it to size the padded state space, so the two must stay in lockstep.
    """
    mh = m // 2 if m % 2 == 0 else (m - 1) // 2
    return min(mh, n)


def find_harmonics(y: jnp.ndarray, m: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find optimal number of harmonics for period *m* using AIC.

    vmap-native: *k* is returned as a traced ``jnp`` int32 scalar in
    ``[1, _max_harmonics(m, n)]`` and the deseasonalisation uses column
    masking instead of a data-dependent slice (with ridge regularisation the
    masked solve gives exactly zero betas for masked columns, so it equals
    the sliced solve).

    Returns
    -------
    (k, z_deseasonalised) : jnp int32 scalar, jnp.ndarray
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = len(y)

    # Rolling mean via cumsum; positions before a full window use the
    # expanding mean (the reference's min_periods=1 behaviour), which also
    # covers n < 2m with no special case.
    window_size = 2 * m
    cumsum = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), y]))
    indices = jnp.arange(n)
    lo = jnp.maximum(0, indices + 1 - window_size)
    hi = indices + 1
    f_t = (cumsum[hi] - cumsum[lo]) / (hi - lo)

    z = y - f_t  # detrend

    max_harmonics = _max_harmonics(m, n)
    if max_harmonics == 0:  # static (m < 2)
        return jnp.asarray(1, dtype=jnp.int32), y

    # Vectorised Fourier terms
    t = jnp.arange(n, dtype=jnp.float64)
    harmonics = jnp.arange(1, max_harmonics + 1, dtype=jnp.float64)
    angles = 2.0 * jnp.pi * jnp.outer(t, harmonics) / m
    fourier = jnp.zeros((n, 2 * max_harmonics), dtype=jnp.float64)
    fourier = fourier.at[:, 0::2].set(jnp.cos(angles))
    fourier = fourier.at[:, 1::2].set(jnp.sin(angles))

    # AIC selection (JIT-compiled, traced result)
    num_harmonics = jnp.maximum(
        _select_harmonics_fast(z, fourier, max_harmonics, 2), 1
    ).astype(jnp.int32)

    # Deseasonalise: betas are fit on the detrended z, but the returned
    # series is y minus the SEASONAL fit only (trend intact — the next
    # period's search re-detrends with its own rolling mean).
    col_mask = (jnp.arange(2 * max_harmonics) < 2 * num_harmonics).astype(fourier.dtype)
    X_best = fourier * col_mask
    z_deseasonalised = y - X_best @ _ridge_solve(X_best, z, ridge=1e-8)

    return num_harmonics, z_deseasonalised


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
    seasonal_periods: jnp.ndarray, k_vector: jnp.ndarray, dtype: Any,
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

def make_w(
    phi: Optional[float],
    k_vector: jnp.ndarray,
    ar_coeffs: Optional[jnp.ndarray],
    ma_coeffs: Optional[jnp.ndarray],
    tau: int,
    beta: Optional[float],
    dtype: Any,
) -> jnp.ndarray:
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


def make_g(
    k_vector: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: Optional[jnp.ndarray],
    p: int,
    q: int,
    tau: int,
    dtype: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build the gain vector g  (shape ``(d, 1)``) and the gamma-bold row."""
    adj_phi = 1 if beta is not None else 0
    d = 1 + adj_phi + tau + p + q

    g = jnp.zeros((d, 1), dtype=dtype).at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gamma_bold = jnp.zeros((1, 2 * int(jnp.sum(jnp.asarray(k_vector)))), dtype=dtype)
    return g, gamma_bold


def make_F(
    phi: Optional[float],
    tau: int,
    alpha: jnp.ndarray,
    beta: Optional[jnp.ndarray],
    ar_coeffs: Optional[jnp.ndarray],
    ma_coeffs: Optional[jnp.ndarray],
    gamma_bold: jnp.ndarray,
    seasonal_periods: Sequence[int] | jnp.ndarray,
    k_vector: jnp.ndarray,
    dtype: Any,
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

def update_w(
    w: jnp.ndarray,
    phi: Optional[float],
    tau: int,
    ar_coeffs: Optional[jnp.ndarray],
    ma_coeffs: Optional[jnp.ndarray],
    p: int,
    q: int,
    beta: Optional[jnp.ndarray],
    dtype: Any,
) -> jnp.ndarray:
    """Return *w* with the phi entry updated."""
    adj_phi = 1 if beta is not None else 0
    if adj_phi:
        phi_eff = jnp.asarray(0.0 if phi is None else phi, dtype=dtype)
        w = w.at[0, 1].set(phi_eff)
    return w


def update_g(
    g: jnp.ndarray,
    gamma_bold: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: Optional[jnp.ndarray],
    k_vector: jnp.ndarray,
    gamma_one_v: jnp.ndarray,
    gamma_two_v: jnp.ndarray,
    dtype: Any,
    k_vector_max: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Return *g* with alpha, beta and seasonal gains updated.

    ``k_vector_max`` (when given) fixes the per-season block LAYOUT of the
    padded state space, while the possibly-traced ``k_vector`` bounds the
    ACTIVE gain positions inside each block — harmonics beyond ``k_vector``
    keep zero gain, so their states never activate.  When omitted, layout ==
    active (the unpadded legacy behaviour).
    """
    adj_phi = 1 if beta is not None else 0
    g = g.at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gb = jnp.zeros_like(gamma_bold)
    if gamma_bold.shape[1] > 0:
        kv = jnp.asarray(k_vector, dtype=jnp.int32)
        kv_layout = jnp.asarray(
            k_vector if k_vector_max is None else k_vector_max, dtype=jnp.int32
        )
        g1 = jnp.asarray(gamma_one_v, dtype=dtype)
        g2 = jnp.asarray(gamma_two_v, dtype=dtype)
        if kv.size > 0:
            tau = gamma_bold.shape[1]
            idx = jnp.arange(tau, dtype=jnp.int32)
            starts = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), 2 * kv_layout[:-1]]))
            mids = starts + kv_layout  # sin-block start (layout)
            mask1 = (idx[None, :] >= starts[:, None]) & (idx[None, :] < (starts + kv)[:, None])
            mask2 = (idx[None, :] >= mids[:, None]) & (idx[None, :] < (mids + kv)[:, None])
            gb_row = jnp.sum(mask1 * g1[:, None] + mask2 * g2[:, None], axis=0)
            gb = gb.at[0, :].set(gb_row)

    start = 1 + adj_phi
    g = g.at[start: start + gb.shape[1], 0].set(gb.ravel())
    return g


def update_F(
    F: jnp.ndarray,
    phi: Optional[float],
    alpha: jnp.ndarray,
    beta: Optional[jnp.ndarray],
    gamma_bold: jnp.ndarray,
    ar_coeffs: Optional[jnp.ndarray],
    ma_coeffs: Optional[jnp.ndarray],
    p: int,
    q: int,
    tau: int,
    dtype: Any,
) -> jnp.ndarray:
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

    def step(x_prev: jnp.ndarray, y_t: jnp.ndarray) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """Advance the innovations filter by one observation."""
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

# Convergence-gated L-BFGS: run at least _MIN_OPTIM_STEPS, then continue while the
# per-observation best-loss improvement stays above _OPTIM_FTOL, stopping after
# _OPTIM_PATIENCE consecutive stalled steps or at the _MAX_OPTIM_STEPS cap. The
# A fixed-step scan leaves TBATS badly under-converged on trend-heavy and long
# series, while the gate charges each cell only the steps it needs — fast-m cells
# stop early, large-m cells run to the cap. The PATIENCE counter is load-bearing: a
# bare "improvement < tol" test stops on a single L-BFGS line-search misfire
# (best_loss flat for one step) and returns a barely-optimized result. The FTOL is
# scaled
# by n because the objective is n·log(sse). Under conformity_scores' vmap the loop
# runs until every CV window stalls.
_MIN_OPTIM_STEPS = 30
_MAX_OPTIM_STEPS = 150
_OPTIM_PATIENCE = 10
_OPTIM_FTOL = 1e-6

_TBATS_SOLVER = optax.lbfgs(
    memory_size=5,
    linesearch=optax.scale_by_zoom_linesearch(
        max_linesearch_steps=10,
        initial_guess_strategy="one",
    ),
)


@partial(jax.jit, static_argnames=(
    "use_boxcox", "use_trend", "use_damped_trend", "p", "q", "tau", "n_k", "kvmax",
))
def _run_lbfgs_optim(
    u0: jnp.ndarray, scale_vec: jnp.ndarray,
    w: jnp.ndarray, g: jnp.ndarray, F: jnp.ndarray,
    gamma_bold: jnp.ndarray, k_vector_arr: jnp.ndarray,
    x0_hat: jnp.ndarray, x0_utp: jnp.ndarray,
    y_fit: jnp.ndarray, y_pos: jnp.ndarray, y_pos_log_sum: jnp.ndarray,
    bc_disabled: jnp.ndarray,
    bc_lower: jnp.ndarray, bc_upper: jnp.ndarray,
    # ── static args (hashed, not traced; all config/shape-derived) ──
    use_boxcox: bool, use_trend: bool, use_damped_trend: bool,
    p: int, q: int, tau: int, n_k: int, kvmax: Tuple[int, ...],
) -> jnp.ndarray:
    """Top-level JIT-cached L-BFGS optimisation of the TBATS log-likelihood.

    Being a module-level function (not a closure) lets JAX cache the compiled
    XLA kernel across warm runs when the static args and array shapes match.
    """
    dtype = jnp.float64
    kvmax_arr = jnp.asarray(kvmax, dtype=jnp.int32)

    def obj(u: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the TBATS objective for one unconstrained parameter vector."""
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
            # Damping is admissible only in [0.8, 1] (the reference's box);
            # clip projects the unconstrained coordinate into it.
            phi_opt = (jnp.clip(theta[idx], 0.8, 1.0)
                       if use_damped_trend and idx < theta.size else 1.0)
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
        g_opt = update_g(g, gamma_bold, alpha_opt, beta_opt, k_vector_arr, g1, g2, dtype,
                         k_vector_max=kvmax_arr)
        F_opt = update_F(F, phi_opt, alpha_opt, beta_opt, gamma_bold, ar_opt, ma_opt, p, q, tau, dtype)

        if use_boxcox:
            # bc_disabled (non-positive data): fit the RAW series with the raw
            # seed. The transform input is swapped to a safe constant on the
            # disabled lane so the UNSELECTED branch cannot inject NaN into
            # gradients through the where; on the enabled lane a lambda that
            # makes a seed component non-transformable yields a NaN -> 1e20
            # objective and the optimizer backs away from that region.
            x0_safe = jnp.where(bc_disabled, jnp.ones_like(x0_utp), x0_utp)
            x0_opt = jnp.where(bc_disabled, x0_hat, _boxcox_raw(x0_safe, lam_opt))
            y_opt = jnp.where(bc_disabled, y_fit, _boxcox_raw(y_pos, lam_opt))
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
    inf = jnp.asarray(jnp.inf, dtype=dtype)
    n_obs = y_fit.shape[0]

    def _cond(carry: tuple) -> jnp.ndarray:
        """Continue while under the cap AND (below the floor OR not yet stalled)."""
        _, _, _, _, stall, it = carry
        return (it < _MAX_OPTIM_STEPS) & (
            (it < _MIN_OPTIM_STEPS) | (stall < _OPTIM_PATIENCE)
        )

    def _optim_step(carry: tuple) -> tuple:
        """One L-BFGS update; track the best iterate + a consecutive-stall counter."""
        u, opt_state, best_u, best_loss, stall, it = carry
        loss, grads = val_and_grad_fn(u)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_opt_state = _TBATS_SOLVER.update(
            grads, opt_state, u, value=loss, grad=grads, value_fn=obj,
        )
        new_u = optax.apply_updates(u, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        new_best_u = jnp.where(improved, u, best_u)
        new_best_loss = jnp.where(improved, loss, best_loss)
        # A step improving best_loss by < _OPTIM_FTOL·n (per-obs LL) is a stall;
        # PATIENCE consecutive stalls ⇒ converged. best_loss=inf on step 0 ⇒ the
        # first real evaluation always counts as improvement (resets the counter).
        improved_enough = (best_loss - new_best_loss) > (_OPTIM_FTOL * n_obs)
        new_stall = jnp.where(improved_enough, jnp.int32(0), stall + jnp.int32(1))
        return (new_u, new_opt_state, new_best_u, new_best_loss, new_stall, it + 1)

    it0 = jnp.asarray(0, dtype=jnp.int32)
    stall0 = jnp.asarray(0, dtype=jnp.int32)
    init_carry = (u0, opt_state0, u0, inf, stall0, it0)
    _, _, best_u, _, _, _ = lax.while_loop(_cond, _optim_step, init_carry)
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
    k_vector_max: Optional[Sequence[int]] = None,
) -> Dict:
    """Fit a single TBATS specification (Box-Cox + optimisation + filter).

    ``k_vector`` may be a traced array (per-season selected harmonic counts)
    ONLY when ``k_vector_max`` — the static, config-derived per-season layout
    ceiling — is also given; the state space is then built at ``k_vector_max``
    and ``k_vector`` acts purely through gain masks.  Without ``k_vector_max``,
    ``k_vector`` must be concrete and defines the layout itself (legacy
    unpadded behaviour, used when callers fix the harmonic counts).
    """
    dtype = jnp.float64
    y = jnp.asarray(y, dtype=dtype)

    if k_vector_max is None:
        kvmax = tuple(int(x) for x in list(k_vector))  # requires concrete k_vector
    else:
        kvmax = tuple(int(x) for x in list(k_vector_max))
    kvmax_arr = jnp.asarray(kvmax, dtype=jnp.int32)

    # Defaults for y_mu, y_sigma (always initialized)
    y_mu = jnp.asarray(0.0, dtype=dtype)
    y_sigma = jnp.asarray(1.0, dtype=dtype)

    # ── Box-Cox lambda estimation (Guerrero method) ────────────────────
    # Non-positive data disables the transform lax-natively (independently per
    # vmap batch member): the filter then runs on the RAW series, and only the
    # REPORTED lambda becomes the NaN sentinel that downstream transforms read
    # as "identity" — the candidate stays a valid, honestly-optimized model.
    if use_boxcox:
        y_pos = _ensure_pos(y)
        bc_disabled = jnp.any(y <= 0)
        season_length = int(seasonal_periods[0]) if len(seasonal_periods) > 0 else 1
        lam_candidate = _guerrero_lambda(y_pos, season_length, bc_lower, bc_upper)
        # Finite stand-in keeps the packed parameter vector (and the L-BFGS
        # state built from it) NaN-free; the sentinel is applied to the
        # OUTPUT lambda only, after unpacking.
        lam_init = jnp.where(bc_disabled, jnp.asarray(1.0, dtype=dtype), lam_candidate)
        y_fit = jnp.where(bc_disabled, y, _boxcox(y_pos, lam_candidate))
    else:
        lam_init = None
        y_fit = y
        bc_disabled = jnp.asarray(False)

    p = 0 if ar_coeffs is None else int(ar_coeffs.shape[0])
    q = 0 if ma_coeffs is None else int(ma_coeffs.shape[0])
    tau = int(2 * sum(kvmax))  # static: padded layout dimension

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

    gamma_one_v = jnp.zeros(len(kvmax), dtype=dtype)
    gamma_two_v = jnp.zeros(len(kvmax), dtype=dtype)

    # ── Build state-space matrices (layout = static kvmax padding) ─────
    if seasonal_blocks is None:
        seasonal_blocks = _build_seasonal_blocks(seasonal_periods, kvmax_arr, dtype)

    w = make_w(phi, kvmax_arr, ar_coeffs, ma_coeffs, tau, beta, dtype)
    g, gamma_bold = make_g(kvmax_arr, alpha, beta, p, q, tau, dtype)
    F = make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
               gamma_bold, seasonal_periods, kvmax_arr, dtype, seasonal_blocks)

    # ── Seed state (level + trend + seasonal + ARMA) ───────────────────
    n = y_fit.shape[0]
    level_init = jnp.mean(y_fit) if n > 0 else jnp.asarray(0.0, dtype=dtype)
    
    x0_ls = jnp.zeros((1 + adj_phi + tau,), dtype=dtype)
    x0_ls = x0_ls.at[0].set(level_init)
    
    if use_trend and beta is not None and n > 1:
        trend_init = (y_fit[-1] - y_fit[0]) / (n - 1)
        x0_ls = x0_ls.at[1].set(trend_init)
    
    x0_hat = jnp.concatenate([x0_ls, jnp.zeros(p + q, dtype=dtype)]) if (p or q) else x0_ls

    # ── Pack initial parameter vector + scales ─────────────────────────
    params: list = []
    scale: list = []
    if use_boxcox:
        params.extend([lam_init, alpha]); scale.extend([0.001, 0.01])
    else:
        params.append(alpha); scale.append(0.01)
    if beta is not None:
        params.append(beta); scale.append(0.01)
    if use_trend and use_damped_trend:  # phi == 0.999 exactly when damped (config-static)
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
        # Raw roundtrip: at lam_init this is exactly the pre-transform seed
        # (a negative trend slope stays negative). A lambda that later makes
        # any seed component non-transformable yields a NaN objective, which
        # the finite-guard scores 1e20 — the optimizer avoids that region
        # rather than fitting a sign-flipped seed.
        x0_untransformed_pos = _inv_boxcox(x0_hat, lam_init)
        y_pos_log_sum = jnp.where(
            bc_disabled,
            jnp.asarray(0.0, dtype=dtype),
            jnp.sum(jnp.log(jnp.clip(y_pos, 1e-12, jnp.inf))),
        )
    else:
        x0_untransformed_pos = x0_hat
        y_pos_log_sum = jnp.asarray(0.0, dtype=dtype)

    u0 = jnp.asarray(params_vec / scale_vec, dtype=dtype)
    n_k = len(kvmax)
    kv_active = jnp.asarray(k_vector, dtype=jnp.int32)

    # Dummy keeps JIT signatures consistent regardless of Box-Cox branch;
    # y_fit is always passed for real — it is the bc_disabled fallback series.
    _y_pos = y_pos if use_boxcox else y_fit

    uhat = _run_lbfgs_optim(
        u0, scale_vec, w, g, F, gamma_bold, kv_active,
        x0_hat, x0_untransformed_pos, y_fit, _y_pos, y_pos_log_sum, bc_disabled,
        jnp.asarray(bc_lower, dtype=dtype), jnp.asarray(bc_upper, dtype=dtype),
        use_boxcox=use_boxcox, use_trend=use_trend,
        use_damped_trend=use_damped_trend,
        p=p, q=q, tau=tau, n_k=n_k, kvmax=kvmax,
    )
    # Fallback to initial params if optimization produced non-finite values
    uhat = jnp.where(jnp.all(jnp.isfinite(uhat)), uhat, u0)

    # ── Unpack optimised parameters ────────────────────────────────────
    optim_params = jnp.asarray(uhat, dtype=dtype) * scale_vec
    if use_boxcox:
        optim_params = optim_params.at[0].set(jnp.clip(optim_params[0], bc_lower, bc_upper))

    # ── Unpack optimised parameters (keep as JAX scalars for vmap) ─────
    idx = 0
    if use_boxcox:
        # optim_lambda_raw (finite) drives the final transforms; the REPORTED
        # lambda carries the NaN sentinel when the transform was disabled —
        # downstream (tbats_forecast, _bc_original_scale) reads NaN as identity.
        optim_lambda_raw = optim_params[idx]; idx += 1
        optim_lambda = jnp.where(bc_disabled, jnp.asarray(jnp.nan, dtype=dtype), optim_lambda_raw)
        optim_alpha = optim_params[idx]; idx += 1
    else:
        optim_lambda = None
        optim_alpha = optim_params[idx]; idx += 1

    if use_trend:
        optim_beta = optim_params[idx]; idx += 1
        if use_damped_trend and idx < optim_params.size:
            optim_phi = jnp.clip(optim_params[idx], 0.8, 1.0); idx += 1
        else:
            optim_phi = jnp.asarray(1.0, dtype=dtype)
    else:
        optim_beta = None
        optim_phi = None

    g1 = optim_params[idx: idx + n_k]; idx += n_k
    g2 = optim_params[idx: idx + n_k]; idx += n_k
    optim_ar = optim_params[idx: idx + p] if p > 0 else None
    optim_ma = optim_params[idx + p: idx + p + q] if q > 0 else None

    # ── Rebuild final matrices & run filter ────────────────────────────
    w_final = update_w(w, optim_phi, tau, optim_ar, optim_ma, p, q, optim_beta, dtype)
    g_final = update_g(g, gamma_bold, optim_alpha, optim_beta, kv_active, g1, g2, dtype,
                       k_vector_max=kvmax_arr)
    F_final = update_F(F, optim_phi, optim_alpha, optim_beta, gamma_bold,
                       optim_ar, optim_ma, p, q, tau, dtype)

    if use_boxcox:
        x0_safe_f = jnp.where(bc_disabled, jnp.ones_like(x0_untransformed_pos),
                              x0_untransformed_pos)
        x0_final = jnp.where(bc_disabled, x0_hat, _boxcox_raw(x0_safe_f, optim_lambda_raw))
        y_fit_final = jnp.where(bc_disabled, y, _boxcox_raw(y_pos, optim_lambda_raw))
    else:
        x0_final = x0_hat
        y_fit_final = y_fit

    fitted, errors, x_seq = _calc_filter(y_fit_final, w_final, g_final, F_final, x0_final)
    sigma2 = jnp.mean(errors * errors)

    # ── Log-likelihood & AIC (vmap-safe: all JAX ops) ─────────────────
    n_eff = errors.shape[0]
    log_likelihood = n_eff * jnp.log(sigma2 + 1e-12)
    if use_boxcox:
        # y_pos_log_sum is already zeroed when the transform is disabled; the
        # where is belt-and-braces. lam_candidate (the Guerrero-init lambda,
        # not the optimized one) is retained deliberately: the optimized-lambda
        # form is SF-faithful but the A/B showed it regresses airline/co2-class
        # holdout selection (IC != holdout) — parked, documented divergence.
        bc_adjustment = 2.0 * (lam_candidate - 1.0) * y_pos_log_sum
        log_likelihood = log_likelihood - jnp.where(bc_disabled, 0.0, bc_adjustment)

    # Seed-state dof counts the ACTIVE (possibly traced) harmonics, not the
    # padded layout — keeps AIC identical to the unpadded model.
    n_seed_states = 1 + adj_phi + 2 * jnp.sum(kv_active) + p + q
    kval = optim_params.size + n_seed_states
    if use_boxcox and optim_lambda is not None:
        kval = kval - jnp.where(optim_lambda == 1, 1, 0)
    if optim_beta is not None:
        kval = kval - jnp.where(jnp.abs(optim_beta) < 1e-8, 1, 0)
    if optim_phi is not None:
        kval = kval - jnp.where(optim_phi == 1, 1, 0)

    aic = log_likelihood + 2 * kval

    return {
        "fitted": fitted,
        "errors": errors[None, :],
        "sigma2": sigma2,
        "aic": aic,
        "optim_params": jnp.asarray(optim_params, dtype=dtype),
        "F": F_final,
        "w_transpose": w_final,
        "g": g_final,
        "x": x_seq,
        "k_vector": kv_active,
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
    y: jnp.ndarray,
    seasonal_periods: Sequence[int],
    k_vector: jnp.ndarray,
    use_boxcox: bool,
    bc_lower: float,
    bc_upper: float,
    use_trend: bool,
    use_damped_trend: bool,
    use_arma_errors: bool,
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

def _trend_grid(use_trend, use_damped_trend):
    """Trend/damped candidate combinations. None on either axis means "search
    both". ⚠ The default (both None) deliberately OMITS the (True, True)
    damped candidate: A/B on the benchmark showed adding it regresses
    airline/synth-multi holdout (chronax's damped fit underperforms SF's
    there) while its AIC wins selection — the IC != holdout class. The
    damped candidate is only searched when the caller asks for it."""
    if use_trend is None:
        if use_damped_trend is None:
            return [(False, False), (True, False)]
        if use_damped_trend:
            return [(True, True)]
        return [(True, False), (False, False)]
    if use_trend:
        if use_damped_trend is None:
            return [(True, False), (True, True)]
        if use_damped_trend:
            return [(True, True)]
        return [(True, False)]
    return [(False, False)]


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
    k_vector: Optional[jnp.ndarray] = None,
) -> Dict:
    """Fit the TBATS candidate grid and argmin-select the best config by AIC.

    vmap-native: the candidate grid is a static config loop, per-candidate
    AICs stay traced scalars, invalid candidates (NaN AIC) are masked to
    +inf, and the winner index comes from ``jnp.argmin``.  Box-Cox candidates
    on non-positive data degrade lax-natively to an untransformed fit whose
    reported lambda is a NaN sentinel (identity for downstream transforms).
    Candidates can differ in state dimension (trend on/off), so the returned
    dict carries the full ``candidates`` list plus the traced ``best`` index;
    ``tbats_forecast``/``compute_sigmah`` forecast every candidate and
    row-select the winner (the AutoCES pattern).  Leaves whose shapes agree
    across all candidates (scalars and n-shaped ones always do) are also
    exposed at the top level, winner-selected, for direct inspection.

    ``use_boxcox`` follows the StatsForecast/R convention: ``None`` tries both
    off and on, ``True`` forces on, ``False`` forces off.

    ``early_stop_patience``/``early_stop_tol`` are accepted for backward
    compatibility and ignored (early stopping is not vmap-compatible).
    ``k_vector`` (concrete values only) skips the per-season harmonic search
    and fixes the layout to exactly those counts.
    """
    if use_trend is False and use_damped_trend is True:
        raise ValueError("Cannot use damped trend without trend")

    t_sel0 = time.perf_counter()
    periods = sorted(int(p) for p in list(jnp.atleast_1d(jnp.asarray(seasonal_periods))))
    n = jnp.asarray(y).shape[0]

    # ── Harmonic search per season (traced counts; static kvmax layout) ─
    if k_vector is None:
        ks: List[jnp.ndarray] = []
        z = y
        for period in periods:
            k, z = find_harmonics(z, period)
            ks.append(k)
        kv = jnp.stack([jnp.asarray(k, dtype=jnp.int32) for k in ks])
        kvmax = tuple(max(1, _max_harmonics(m, n)) for m in periods)
    else:
        kv = jnp.asarray(k_vector, dtype=jnp.int32)
        kvmax = tuple(int(x) for x in list(k_vector))  # concrete-only legacy path
    if _TBATS_DEBUG:
        _tbats_debug(f"[TBATS][select] k_vector={kv} kvmax={kvmax} "
                     f"time={time.perf_counter() - t_sel0:.4f}s")

    # ── Candidate grid (static config; SF/R-convention Box-Cox mapping) ─
    if use_boxcox is None:
        B = [False, True]
    elif use_boxcox:
        B = [True]
    else:
        B = [False]

    T = _trend_grid(use_trend, use_damped_trend)

    combos = [(bcx, t, use_arma_errors) for bcx in B for t in T]

    # Pre-build seasonal blocks once (shared across candidates), in the
    # generator's dtype — building them in y.dtype skewed the selection-path
    # AIC against direct float64 fits.
    seasonal_blocks = _build_seasonal_blocks(
        periods, jnp.asarray(kvmax, dtype=jnp.int32), jnp.float64
    )

    # ── Evaluate all candidates (static config loop — vmap-compatible) ──
    candidates = []
    for bcx, (trend, damped), arma in combos:
        t_cand = time.perf_counter()
        cand = tbats_model_generator(
            y, periods, kv,
            bcx, bc_lower, bc_upper,
            trend, damped, arma,
            None, None,  # ar_coeffs, ma_coeffs
            seasonal_blocks,
            k_vector_max=kvmax,
        )
        if _TBATS_DEBUG:
            _tbats_debug(
                f"[TBATS][select] bcx={bcx} trend={trend} damped={damped} arma={arma} "
                f"time={time.perf_counter() - t_cand:.4f}s"
            )
        candidates.append(cand)

    # ── argmin selection; NaN AICs masked to +inf ───────────────────────
    # Box-Cox candidates degrade lax-natively on non-positive data (fitted on
    # the raw series; only their REPORTED lambda is the NaN sentinel), so they
    # remain valid competitors — the extra lambda dof costs them +2 AIC vs
    # their non-Box-Cox twin, which then wins the auto grid deterministically.
    aics = jnp.stack([jnp.asarray(c["aic"]) for c in candidates])
    invalid = jnp.isnan(aics)
    best = jnp.argmin(jnp.where(invalid, jnp.inf, aics))
    # Raising on all-invalid is impossible under trace; expose a flag instead
    # (argmin over all-inf silently picks index 0).
    valid = (~invalid).any()

    def _stack_take(key: str) -> jnp.ndarray:
        return jnp.take(jnp.stack([jnp.asarray(c[key]) for c in candidates]), best, axis=0)

    out: Dict[str, Any] = {
        "candidates": candidates,
        "combos": tuple(combos),
        "best": best,
        "valid": valid,
        "k_vector": kv,
        # ARMA orders are currently never searched by selection; these mirror
        # the winner's fields and are identical across candidates.
        "p": candidates[0]["p"],
        "q": candidates[0]["q"],
        "ar_coeffs": candidates[0]["ar_coeffs"],
        "ma_coeffs": candidates[0]["ma_coeffs"],
        "y_mu": candidates[0]["y_mu"],
        "y_sigma": candidates[0]["y_sigma"],
        "description": {},
    }

    # BoxCox_lambda: None when no candidate uses Box-Cox (legacy convention);
    # otherwise the winner's lambda, where NaN means the winner either does
    # not use Box-Cox or its lambda degraded on non-positive data.
    if any(bcx for bcx, _t, _a in combos):
        lams = jnp.stack([
            jnp.asarray(c["BoxCox_lambda"])
            if combos[i][0] else jnp.asarray(jnp.nan, dtype=jnp.float64)
            for i, c in enumerate(candidates)
        ])
        out["BoxCox_lambda"] = jnp.take(lams, best, axis=0)
    else:
        out["BoxCox_lambda"] = None

    # Winner-selected leaves, exposed only when shapes agree across candidates
    # (state-space matrices drop out when the grid mixes trend on/off).
    for key in ("aic", "sigma2", "fitted", "errors",
                "optim_params", "F", "w_transpose", "g", "x", "seed_states"):
        shapes = {jnp.asarray(c[key]).shape for c in candidates}
        if len(shapes) == 1:
            out[key] = _stack_take(key)

    _tbats_debug(f"[TBATS][select] done "
                 f"time={time.perf_counter() - t_sel0:.4f}s")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Forecasting
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("h",))
def _tbats_forecast_core(F: jnp.ndarray, w: jnp.ndarray, x0: jnp.ndarray, h: int) -> jnp.ndarray:
    """JIT-compiled multi-step forecast for a single initial state."""
    def step(x: jnp.ndarray, _: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Advance the TBATS state one forecast step ahead."""
        y_next = jnp.dot(w, x)
        x_next = F @ x
        return x_next, y_next
    _, fcst = lax.scan(step, x0, jnp.arange(h))
    return fcst


def tbats_forecast(mod: Dict, h: int) -> Dict[str, jnp.ndarray]:
    """Multi-step mean forecast from a fitted model dictionary.

    Accepts either a single fitted candidate (``tbats_model_generator``
    output) or a ``tbats_selection`` result: for the latter, every candidate
    is forecast with its own static config and the winner's row is selected
    with ``jnp.take`` — candidate state dimensions may differ, so the traced
    ``best`` index can never be dispatched on directly.
    """
    h = int(h)
    if "candidates" in mod:  # structural (key presence), trace-safe dispatch
        outs = [tbats_forecast(c, h) for c in mod["candidates"]]
        res = {"mean": jnp.take(jnp.stack([o["mean"] for o in outs]), mod["best"], axis=0)}
        if all(o["mean_bc"] is not None for o in outs):
            res["mean_bc"] = jnp.take(
                jnp.stack([o["mean_bc"] for o in outs]), mod["best"], axis=0
            )
        else:
            res["mean_bc"] = None
        return res

    w = mod["w_transpose"][0]
    F = mod["F"]
    x_last = mod["x"][-1]

    fcst = _tbats_forecast_core(F, w, x_last, h)

    lam = mod["BoxCox_lambda"]
    if lam is None:
        return {"mean": mod["y_mu"] + mod["y_sigma"] * fcst, "mean_bc": None}

    # NaN sentinel (Box-Cox degraded on non-positive data) -> identity
    mean = jnp.where(jnp.isnan(lam), fcst, _inv_boxcox(fcst, lam))
    return {"mean": mean, "mean_bc": fcst}


# ── Prediction intervals ──────────────────────────────────────────────

@partial(jax.jit, static_argnames=("h", "use_boxcox"))
def _compute_sigmah_core(
    F: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray,
    sigma2: jnp.ndarray, y_sigma: jnp.ndarray,
    h: int, use_boxcox: bool,
) -> jnp.ndarray:
    """JIT-compiled parametric forecast standard deviations."""
    var0 = jnp.asarray(1.0, dtype=F.dtype)

    def body(carry: tuple[jnp.ndarray, jnp.ndarray], _: jnp.ndarray) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Accumulate one additional forecast-variance step.

        c_j = w' F^(j-1) g (De Livera et al. 2011): the j-th tail term reads
        the CURRENT power of F (starting at F^0 = I), then advances it.
        """
        Fpow, var_acc = carry
        cj = jnp.dot(jnp.dot(w, Fpow), g)
        var_next = var_acc + cj * cj
        return (F @ Fpow, var_next), var_next

    init = (jnp.eye(F.shape[1], dtype=F.dtype), var0)
    _, var_tail = lax.scan(body, init, jnp.arange(h - 1))
    var_mult = jnp.concatenate([var0[None], var_tail], axis=0)

    sigma2h = sigma2 * var_mult
    if not use_boxcox:
        sigma2h = (y_sigma ** 2) * sigma2h
    return jnp.sqrt(jnp.maximum(sigma2h, 0.0))


def compute_sigmah(mod: Dict, h: int) -> jnp.ndarray:
    """Parametric forecast standard deviations from a fitted model dictionary.

    Like ``tbats_forecast``, accepts a single candidate or a
    ``tbats_selection`` result (per-candidate sigmah, winner row selected).
    """
    if "candidates" in mod:
        rows = [compute_sigmah(c, int(h)) for c in mod["candidates"]]
        return jnp.take(jnp.stack(rows), mod["best"], axis=0)

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
