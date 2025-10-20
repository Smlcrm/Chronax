# tbats_core.py
# Pure-JAX TBATS core (state-space builders, filter, likelihood, selection & forecast)
# Keeps float32 throughout. Parameter names aligned to caller: bc_lower / bc_upper.
# CORRECTED VERSION with bug fixes

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import jax
import jax.numpy as jnp
from jax import lax

"""
TBATS (Trigonometric, Box–Cox, ARMA, Trend, Seasonal) — Pure JAX core

Differences vs StatsForecast reference:
- Pure JAX math; CPU/GPU/TPU friendly except the small Nelder–Mead wrapper
  (host NumPy loop calling a JAX function).
- Forecasts and fitted values are always returned on the *original data scale*.
  When Box–Cox is enabled we inverse-transform via λ; when disabled we de-standardize.
- Box–Cox λ is selected via a Guerrero-style grid search (coefficient of variation
  over seasonal slices) rather than a continuous optimizer.
- The state update uses a light "innovations" style filter with fixed gain vector g
  (as per TBATS formulation); not a full Kalman gain recomputation.
- ARMA error terms are plumbed through the state-space builders, but this module
  does not auto-identify (p, q). (The higher-level wrapper may decide to try ARMA.)

Limitations / notes:
- Nelder–Mead runs on the host (NumPy loop). It’s robust but not JIT’d; for very
  large runs consider swapping in a JAX-compatible optimizer.
- Guerrero λ search uses a coarse grid (25 points); increase if you need more precision.
- No exogenous regressors; no missing-value imputation (NaNs should be filtered upstream).
- Parameter admissibility checks are relaxed compared to the original R/StatsForecast
  implementations (no root checks for AR/MA polynomials here).
- This is a compact educational/production-lite core. Heavy-duty production may
  want stronger constraints, multiple random restarts, and richer diagnostics.
"""

# -----------------------
# Small Box–Cox utilities
# -----------------------

def _boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    """Apply Box–Cox transform to `y`.

    Args:
        y: 1D array of strictly positive values.
        lam: Box–Cox λ (None = no transform). Values |λ|<1e-8 use log(y).

    Returns:
        Transformed array with same dtype as `y`.

    Notes:
        - This version treats λ≈0 as log(y) for numerical stability.
        - Caller is responsible for ensuring y>0 when λ is used.
    """
    if lam is None:
        return y
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.log(y), (jnp.power(y, lam) - 1.0) / lam)

def _inv_boxcox(y: jnp.ndarray, lam: Optional[float]) -> jnp.ndarray:
    """Inverse Box–Cox transform.

    Args:
        y: Box–Cox-domain values.
        lam: Box–Cox λ (None = identity). Values |λ|<1e-8 use exp(y).

    Returns:
        Inverse-transformed values.
    """
    if lam is None:
        return y
    lam = jnp.asarray(lam, dtype=y.dtype)
    return jnp.where(jnp.abs(lam) < 1e-8, jnp.exp(y), jnp.power(y * lam + 1.0, 1.0 / lam))

def _guerrero_lambda(y: jnp.ndarray, season_length: int, lower: float, upper: float) -> float:
    """Select λ using a Guerrero-style criterion (grid search).

    We split the series into complete seasonal blocks and choose λ that minimizes
    the coefficient of variation across block-wise standard deviations.

    Args:
        y: 1D series (must be positive if Box–Cox is to be used).
        season_length: dominant season length used to slice blocks.
        lower, upper: bounds for λ search (inclusive).

    Returns:
        Scalar λ in [lower, upper]. Falls back to 0.5 if <2 seasonal blocks.

    Differences vs StatsForecast:
        - Uses a fixed grid of 25 candidates; StatsForecast may use a different
          approach/optimizer.

    Limitations:
        - Coarse grid; tune density if more precision is required.
    """
    n = y.shape[0]
    n_periods = n // season_length
    if n_periods < 2:
        return 0.5  # default fallback

    y_trim = y[: n_periods * season_length].reshape(n_periods, season_length)
    lambdas = jnp.linspace(lower, upper, 25)

    def cv_for_lambda(lam):
        yt = _boxcox(y_trim.ravel(), lam).reshape(n_periods, season_length)
        stds = jnp.std(yt, axis=1)
        means = jnp.abs(jnp.mean(yt, axis=1)) + 1e-10
        return jnp.std(stds / means)

    cvs = jnp.stack([cv_for_lambda(l) for l in lambdas])
    best_idx = jnp.argmin(cvs)
    return float(jnp.clip(lambdas[best_idx], lower, upper))


# -------------------------------------
# Harmonics finder (simple JAX version)
# -------------------------------------

def find_harmonics(y: jnp.ndarray, m: int) -> Tuple[int, jnp.ndarray]:
    """Estimate number of Fourier harmonics for period `m`.

    Approach:
        1) Detrend via a 2*m moving average.
        2) Fit increasing numbers of sin/cos harmonics by least-squares.
        3) Choose the smallest AIC; stop early after 2 non-improvements.

    Args:
        y: 1D series.
        m: seasonal period (int).

    Returns:
        (k_best, residual_z): number of harmonics and residuals after removing
        the selected Fourier terms from the detrended series.

    Differences vs StatsForecast:
        - Fully JAX (no pandas rolling). Edge padding for the moving average.
        - Conservative cap for `max_h` to avoid rank issues on short series.

    Limitations:
        - Heuristic early stopping and AIC scoring on LS residuals.
    """
    y = jnp.asarray(y)
    n = y.shape[0]
    dtype = y.dtype

    # 2*m moving average (left-padded edge to avoid pandas dep)
    window = max(2, 2 * m)
    kernel = jnp.ones((window,), dtype=dtype) / float(window)
    y_pad = jnp.pad(y, (window - 1, 0), mode="edge")
    f_t = jnp.convolve(y_pad, kernel, mode="valid").astype(dtype)

    z = y - f_t  # detrended

    max_h = (m // 2) if (m % 2 == 0) else ((m - 1) // 2)
    max_h = int(min(max_h, max(1, n // 2)))  # cap for rank stability
    if max_h <= 0:
        return 1, y  # force at least one harmonic

    t = jnp.arange(n, dtype=dtype)

    def design(h):
        cols = []
        for i in range(1, h + 1):
            ang = 2.0 * jnp.pi * i * t / jnp.asarray(m, dtype=dtype)
            cols.append(jnp.cos(ang))
            cols.append(jnp.sin(ang))
        return jnp.stack(cols, axis=1)

    best_aic = jnp.inf
    k_best = 1
    aic_prev = jnp.inf
    wout = 0
    tol_no_improv = 2

    for h in range(1, max_h + 1):
        X = design(h)
        beta, *_ = jnp.linalg.lstsq(X, z, rcond=None)
        resid = z - X @ beta
        k = beta.shape[0]
        aic = n * jnp.log(jnp.sum(resid * resid) / n + 1e-12) + 2.0 * k
        better = bool(aic < best_aic - 1e-12)
        best_aic = jnp.where(better, aic, best_aic)
        k_best = int(jnp.where(better, h, k_best))
        if not bool(aic < aic_prev - 1e-9):
            wout += 1
            if wout >= tol_no_improv:
                break
        else:
            wout = 0
        aic_prev = aic

    X_best = design(k_best)
    beta_best, *_ = jnp.linalg.lstsq(X_best, z, rcond=None)  # regress residuals, not y
    z_res = z - X_best @ beta_best
    return k_best, z_res


# ---------------------------------------
# Builders for w, g, F with consistent d
# ---------------------------------------

def _pq(ar_coeffs: Optional[jnp.ndarray], ma_coeffs: Optional[jnp.ndarray]) -> Tuple[int, int]:
    """Utility: return (p, q) from possibly-None AR/MA arrays."""
    p = 0 if ar_coeffs is None else int(ar_coeffs.shape[0])
    q = 0 if ma_coeffs is None else int(ma_coeffs.shape[0])
    return p, q

def make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau, beta, dtype):
    """Build the observation row vector `w^T` (shape 1×d).

    Layout (conceptual): [level, (trend if used), seasonal states..., AR states..., MA states...]

    Args:
        phi: damping factor or None.
        k_vector: number of harmonics per seasonal period (array-like).
        ar_coeffs, ma_coeffs: optional AR/MA coeff arrays.
        tau: total seasonal state dimension (sum over 2*k).
        beta: trend smoothing parameter or None (controls presence of trend state).
        dtype: JAX dtype.

    Returns:
        w: shape (1, d).
    """
    adj_phi = 1 if beta is not None else 0
    p, q = _pq(ar_coeffs, ma_coeffs)
    d = 1 + adj_phi + tau + p + q
    w = jnp.zeros((1, d), dtype=dtype).at[0, 0].set(1.0)
    if adj_phi:
        phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)
        w = w.at[0, 1].set(phi_eff)
    pos = 0
    start = 1 + adj_phi
    for k in k_vector:
        k = int(k)
        w = w.at[0, start + pos : start + pos + k].set(1.0)
        pos += 2 * k
    return w

def make_g(k_vector, alpha, beta, p, q, tau, dtype):
    """Build the gain vector `g` and seasonal weights `gamma_bold`.

    Args:
        k_vector: harmonics per seasonal period.
        alpha: level smoothing.
        beta: trend smoothing or None.
        p, q: AR and MA orders.
        tau: seasonal state dimension.
        dtype: JAX dtype.

    Returns:
        g: (d,1) update vector.
        gamma_bold: (1, tau) block that populates the seasonal rows in `g` and `F`.
    """
    adj_phi = 1 if beta is not None else 0
    d = 1 + adj_phi + tau + p + q
    g = jnp.zeros((d, 1), dtype=dtype).at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gamma_bold = jnp.ones((1, 2 * int(jnp.sum(jnp.asarray(k_vector)))), dtype=dtype)
    start = 1 + adj_phi
    end = start + gamma_bold.shape[1]
    g = g.at[start:end, 0].set(gamma_bold.ravel())
    if p != 0:
        g = g.at[end + 0, 0].set(1.0)
    if q != 0:
        g = g.at[end + p, 0].set(1.0)
    return g, gamma_bold

def make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs,
           gamma_bold, seasonal_periods, k_vector, dtype):
    """Build the transition matrix `F` (shape d×d).

    Blocks:
        - Level/trend (with optional damping).
        - Seasonal block as stacked rotation matrices for each harmonic.
        - AR/MA companion sub-blocks (if present), coupled via gamma_bold.

    Args:
        phi, tau, alpha, beta, ar_coeffs, ma_coeffs, gamma_bold, seasonal_periods, k_vector, dtype

    Returns:
        F: system transition matrix.
    """
    adj_phi = 1 if beta is not None else 0
    phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)

    # alpha row
    F = jnp.array([[1.0]], dtype=dtype)
    if adj_phi:
        F = jnp.hstack([F, jnp.array([[phi_eff]], dtype=dtype)])
    F = jnp.hstack([F, jnp.zeros((1, tau), dtype=dtype)])
    if ar_coeffs is not None and ar_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ar_coeffs, dtype=dtype))[None, :]])
    if ma_coeffs is not None and ma_coeffs.size > 0:
        F = jnp.hstack([F, (alpha * jnp.asarray(ma_coeffs, dtype=dtype))[None, :]])

    # beta row
    if beta is not None:
        row = jnp.array([[0.0, phi_eff]], dtype=dtype) if adj_phi else jnp.array([[0.0]], dtype=dtype)
        row = jnp.hstack([row, jnp.zeros((1, tau), dtype=dtype)])
        if ar_coeffs is not None and ar_coeffs.size > 0:
            row = jnp.hstack([row, (beta * jnp.asarray(ar_coeffs, dtype=dtype))[None, :]])
        if ma_coeffs is not None and ma_coeffs.size > 0:
            row = jnp.hstack([row, (beta * jnp.asarray(ma_coeffs, dtype=dtype))[None, :]])
        F = jnp.vstack([F, row])

    # seasonal block
    seasonal = jnp.zeros((tau, 1), dtype=dtype)
    if adj_phi:
        seasonal = jnp.hstack([seasonal, jnp.zeros((tau, 1), dtype=dtype)])
    A = jnp.zeros((tau, tau), dtype=dtype)
    pos = 0
    for k, period in zip(k_vector, seasonal_periods):
        k = int(k)
        t = 2.0 * jnp.pi * (jnp.arange(1, k + 1, dtype=dtype) / jnp.asarray(period, dtype=dtype))
        ck = jnp.diag(jnp.cos(t))
        sk = jnp.diag(jnp.sin(t))
        top = jnp.hstack([ck, sk])
        bot = jnp.hstack([-sk, ck])
        Ak = jnp.vstack([top, bot])
        A = A.at[pos : pos + 2 * k, pos : pos + 2 * k].set(Ak)
        pos += 2 * k
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


# ----------------------------
# Update helpers (in-place-ish)
# ----------------------------

def update_w(w, phi, tau, ar_coeffs, ma_coeffs, p, q, beta, dtype):
    """Update `w` in place-ish with the current `phi` (trend presence controlled by `beta`)."""
    adj_phi = 1 if beta is not None else 0
    if adj_phi:
        phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)
        w = w.at[0, 1].set(phi_eff)
    return w

def update_g(g, gamma_bold, alpha, beta, k_vector, gamma_one_v, gamma_two_v, dtype):
    """Update `g` and `gamma_bold` given smoothing params and per-season gamma weights."""
    adj_phi = 1 if beta is not None else 0
    g = g.at[0, 0].set(alpha)
    if beta is not None:
        g = g.at[1, 0].set(beta)

    gb = jnp.ones_like(gamma_bold)
    endPos = 0
    for k, g1, g2 in zip(k_vector, gamma_one_v, gamma_two_v):
        k = int(k)
        gb = gb.at[0, endPos : endPos + k].set(g1)
        gb = gb.at[0, endPos + k : endPos + 2 * k].set(g2)
        endPos += 2 * k

    start = 1 + adj_phi
    g = g.at[start : start + gb.shape[1], 0].set(gb.ravel())
    return g, gb

def update_F(F, phi, alpha, beta, gamma_bold, ar_coeffs, ma_coeffs, p, q, tau, dtype):
    """Update `F` entries dependent on current params (phi, alpha/beta, AR/MA, seasonal coupling)."""
    adj_phi = 1 if beta is not None else 0
    phi_eff = 0.0 if (beta is not None and phi is None) else (float(phi) if phi is not None else 1.0)
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


# -------------------------
# Kalman-like simple filter
# -------------------------

def _calc_filter(y: jnp.ndarray, w: jnp.ndarray, g: jnp.ndarray, F: jnp.ndarray, x0: jnp.ndarray
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the simple innovations filter.

    Args:
        y: observations on model scale (standardized or Box–Cox domain).
        w, g, F: state-space components.
        x0: initial state vector.

    Returns:
        yhat_seq: one-step-ahead predictions.
        e_seq: residuals (y - yhat).
        x_seq: filtered states.

    Notes:
        - This uses the fixed-gain TBATS update (not a Kalman gain recomputation).
    """
    dtype = y.dtype
    y = jnp.asarray(y, dtype=dtype)
    w = jnp.asarray(w, dtype=dtype)
    g = jnp.asarray(g, dtype=dtype)
    F = jnp.asarray(F, dtype=dtype)
    x0 = jnp.asarray(x0, dtype=dtype)

    def step(x_prev, y_t):
        yhat_t = (w @ x_prev)[0]
        e_t = y_t - yhat_t
        x_t = F @ x_prev + (g[:, 0] * e_t)
        return x_t, (yhat_t, e_t, x_t)

    xT, (yhat_seq, e_seq, x_seq) = lax.scan(step, x0, y)
    return yhat_seq, e_seq, x_seq


# ------------------------
# Likelihood & optimizer
# ------------------------

def negative_loglikelihood(params: jnp.ndarray,
                           use_boxcox: bool,
                           use_trend: bool,
                           use_damped_trend: bool,
                           use_arma_errors: bool,
                           y: jnp.ndarray,
                           y_trans_init: jnp.ndarray,
                           seasonal_periods: jnp.ndarray,
                           k_vector: jnp.ndarray,
                           tau: int,
                           w_template: jnp.ndarray,
                           F_template: jnp.ndarray,
                           g_template: jnp.ndarray,
                           gamma_template: jnp.ndarray,
                           x0_template: jnp.ndarray,
                           x0_untransformed: jnp.ndarray,
                           bc_lower: float,
                           bc_upper: float,
                           p: int,
                           q: int,
                           scale: jnp.ndarray) -> jnp.ndarray:
    """Negative log-likelihood (up to constants) for the TBATS state-space.

    Parameters are unpacked in the same order used by `tbats_model_generator`.
    When Box–Cox is off, the series is standardized (float64 moments), so the
    criterion is invariant to affine rescaling.

    Returns:
        Scalar loss to minimize.

    Differences vs StatsForecast:
        - Uses standardized series (no BC) with double-precision moments for
          numerical stability and scale-invariance.
        - Box–Cox path adds the Jacobian term sum(log y) as usual.
    """
    dtype = y.dtype
    params = params * scale

    idx = 0
    if use_boxcox:
        lam = params[idx]; idx += 1
        alpha = params[idx]; idx += 1
    else:
        lam = None
        alpha = params[idx]; idx += 1

    if use_trend:
        beta = params[idx]; idx += 1
        if use_damped_trend:
            phi = params[idx]; idx += 1
        else:
            phi = 1.0
    else:
        beta = None
        phi = None

    g1 = params[idx : idx + len(k_vector)]; idx += len(k_vector)
    g2 = params[idx : idx + len(k_vector)]; idx += len(k_vector)
    ar = None; ma = None
    if use_arma_errors:
        if p != 0 and q != 0:
            ar = params[idx : idx + p]; ma = params[idx + p : idx + p + q]
        elif p != 0:
            ar = params[idx : idx + p]
        elif q != 0:
            ma = params[idx : idx + q]

    w = update_w(w_template, phi, tau, ar, ma, p, q, beta, dtype)
    g, gamma_bold = update_g(g_template, gamma_template, alpha, beta, k_vector, g1, g2, dtype)
    F = update_F(F_template, phi, alpha, beta, gamma_bold, ar, ma, p, q, tau, dtype)

    if use_boxcox:
        x0 = _boxcox(x0_untransformed, lam)
        y_fit = _boxcox(y, lam)
        y_for_ll = y
    else:
        y64 = y.astype(jnp.float64)
        mu64 = y64.mean()
        sig64 = y64.std() + 1e-12
        y_fit64 = (y64 - mu64) / sig64
        x0 = x0_template
        y_fit = y_fit64.astype(y.dtype)
        y_for_ll = y_fit  # constant terms drop out

    _, e, _ = _calc_filter(y_fit, w, g, F, x0)
    n = y_fit.shape[0]
    if use_boxcox:
        ll = n * jnp.log(jnp.nansum(e * e) + 1e-12) - 2.0 * (lam - 1.0) * jnp.nansum(jnp.log(jnp.clip(y_for_ll, 1e-12)))
    else:
        ll = n * jnp.log(jnp.nansum(e * e) + 1e-12)
    return ll


def nelder_mead_minimize(f, x0, maxiter: int = 2000, tol: float = 1e-6, step: float = 0.1):
    """Very small Nelder–Mead wrapper (host NumPy loop calling a JAX function).

    Args:
        f: callable taking 1D jnp.ndarray → scalar loss (JAX-compatible).
        x0: initial parameter vector (array-like).
        maxiter, tol, step: standard NM knobs.

    Returns:
        Best parameter vector (NumPy array).

    Limitations:
        - Runs on host; not JIT’d.
        - No bound constraints; use scaling and admissibility inside `f`.
        - For tough surfaces consider multiple restarts or a different optimizer.
    """
    # lightweight NM in host NumPy; f is JAX-callable
    import numpy as _np
    x0 = _np.asarray(x0, dtype=float)
    n = x0.size
    simplex = _np.vstack([x0] + [x0 + step * _np.eye(n)[i] for i in range(n)])

    def f_np(x): return float(f(jnp.asarray(x, dtype=jnp.float32)))
    values = _np.array([f_np(v) for v in simplex], dtype=float)

    def order():
        idx = _np.argsort(values)
        return simplex[idx], values[idx]

    it = 0
    alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5
    while it < maxiter:
        simplex, values = order()
        if _np.std(values) < tol:
            break
        best = simplex[0]
        worst = simplex[-1]
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - worst)
        fr = f_np(xr)
        if values[0] <= fr < values[-2]:
            simplex[-1] = xr; values[-1] = fr
        elif fr < values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f_np(xe)
            if fe < fr:
                simplex[-1] = xe; values[-1] = fe
            else:
                simplex[-1] = xr; values[-1] = fr
        else:
            xc = centroid + rho * (worst - centroid)
            fc = f_np(xc)
            if fc < values[-1]:
                simplex[-1] = xc; values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = f_np(simplex[i])
        it += 1
    simplex, values = order()
    return _np.asarray(simplex[0], dtype=float)


# ---------------------------------------
# Model generator, selection and forecast
# ---------------------------------------

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
):
    """Fit a single TBATS specification (given k_vector and feature flags).

    Args:
        y: observed series (original scale).
        seasonal_periods: iterable of periods (e.g., [7, 365]).
        k_vector: number of harmonics per period (same length as seasonal_periods).
        use_boxcox, bc_lower, bc_upper: Box–Cox controls.
        use_trend, use_damped_trend: trend components.
        use_arma_errors: whether ARMA error subspace is used (p/q inferred upstream).
        ar_coeffs, ma_coeffs: optional fixed AR/MA coeffs.

    Returns:
        Dict with fitted fields, state-space matrices, parameters, AIC, and
        auxiliaries (mu/sigma for de-standardization).

    Notes:
        - If Box–Cox is disabled, `y` is standardized before fitting to gain
          affine equivariance. All outputs are mapped back to original scale.
    """
    dtype = y.dtype
    seasonal_periods = jnp.asarray(seasonal_periods, dtype=jnp.int32)
    p, q = _pq(ar_coeffs, ma_coeffs)
    tau = int(2 * int(jnp.sum(k_vector)))

    # Standardization if no Box-Cox
    if use_boxcox:
        lam = _guerrero_lambda(y, int(seasonal_periods.max()), bc_lower, bc_upper)
        y_fit = _boxcox(y, lam)
        y_mu = jnp.asarray(0.0, dtype=dtype)
        y_sigma = jnp.asarray(1.0, dtype=dtype)
    else:
        lam = None
        y64 = y.astype(jnp.float64)
        mu64 = y64.mean()
        sig64 = y64.std() + 1e-12
        y_mu = jnp.asarray(mu64, dtype=jnp.float64)
        y_sigma = jnp.asarray(sig64, dtype=jnp.float64)
        y_fit = ((y64 - mu64) / sig64).astype(dtype)

    alpha = jnp.asarray(0.09, dtype=dtype)
    if use_trend:
        beta = jnp.asarray(0.05, dtype=dtype)
        phi = jnp.asarray(0.999 if use_damped_trend else 1.0, dtype=dtype)
    else:
        beta = None; phi = None

    gamma_one_v = jnp.zeros((len(k_vector),), dtype=dtype)
    gamma_two_v = jnp.zeros((len(k_vector),), dtype=dtype)

    adj_phi = 1 if beta is not None else 0
    d = 1 + adj_phi + tau + p + q
    x0 = jnp.zeros((d,), dtype=dtype)

    w = make_w(phi, k_vector, ar_coeffs, ma_coeffs, tau, beta, dtype)
    g, gamma_bold = make_g(k_vector, alpha, beta, p, q, tau, dtype)
    F = make_F(phi, tau, alpha, beta, ar_coeffs, ma_coeffs, gamma_bold,
               seasonal_periods, k_vector, dtype)

    D = F - jnp.dot(g, w)
    steps = y.shape[0]
    w_tilde = jnp.zeros((steps, w.shape[1]), dtype=dtype).at[0, :].set(w[0])

    def step_w(prev, _):
        nxt = jnp.dot(prev, D)
        return nxt, nxt

    _, w_seq = lax.scan(step_w, w[0], jnp.arange(steps - 1))
    w_tilde = w_tilde.at[1:, :].set(w_seq)

    if p != 0 or q != 0:
        end_cut = w_tilde.shape[1]
        start_cut = end_cut - (p + q)
        cols = jnp.arange(0, start_cut)
        w_tilde = w_tilde[:, cols]

    _, e0, _ = _calc_filter(y_fit, w, g, F, jnp.zeros_like(x0))
    E = e0.reshape((e0.shape[0], 1))
    x0_ls, *_ = jnp.linalg.lstsq(w_tilde, E, rcond=None)
    x0_ls = x0_ls.ravel()

    expected_dim = 1 + adj_phi + tau
    if x0_ls.shape[0] < expected_dim:
        x0_ls = jnp.pad(x0_ls, (0, expected_dim - x0_ls.shape[0]))

    if (p != 0) or (q != 0):
        arma_seed = jnp.zeros((p + q,), dtype=dtype)
        x0_hat = jnp.concatenate([x0_ls, arma_seed], axis=0)
    else:
        x0_hat = x0_ls

    scale = []
    params = []
    if use_boxcox:
        params += [jnp.asarray(lam, dtype=dtype), alpha]; scale += [0.001, 0.01]
    else:
        params += [alpha]; scale += [0.01]
    if beta is not None:
        params += [beta]; scale += [0.01]
        if use_damped_trend and phi is not None and float(phi) != 1.0:
            params += [phi]; scale += [0.01]
    params += [gamma_one_v, gamma_two_v]; scale += ([1e-5] * (gamma_one_v.size + gamma_two_v.size))
    if ar_coeffs is not None and p > 0:
        params += [ar_coeffs]; scale += ([0.1] * p)
    if ma_coeffs is not None and q > 0:
        params += [ma_coeffs]; scale += ([0.1] * q)

    params = jnp.concatenate([p_.ravel() if isinstance(p_, jnp.ndarray) else jnp.asarray([p_], dtype=dtype)
                              for p_ in params]).astype(dtype)
    scale = jnp.asarray(scale, dtype=dtype)
    x0_untransformed = x0_hat if lam is None else _inv_boxcox(x0_hat, lam)

    def obj(par):
        return negative_loglikelihood(
            par,
            use_boxcox=use_boxcox,
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            use_arma_errors=use_arma_errors,
            y=y,
            y_trans_init=y_fit,
            seasonal_periods=seasonal_periods,
            k_vector=jnp.asarray(k_vector),
            tau=tau,
            w_template=w,
            F_template=F,
            g_template=g,
            gamma_template=gamma_bold,
            x0_template=x0_hat,
            x0_untransformed=x0_untransformed,
            bc_lower=bc_lower,
            bc_upper=bc_upper,
            p=p,
            q=q,
            scale=scale,
        )

    params_hat = nelder_mead_minimize(obj, params, maxiter=400 * (params.size + 1), tol=1e-6, step=0.1)
    optim_params = jnp.asarray(params_hat, dtype=dtype) * scale

    idx = 0
    if use_boxcox:
        optim_lambda = float(optim_params[idx]); idx += 1
        optim_alpha  = float(optim_params[idx]); idx += 1
    else:
        optim_lambda = None
        optim_alpha  = float(optim_params[idx]); idx += 1

    if use_trend:
        optim_beta = float(optim_params[idx]); idx += 1
        if use_damped_trend:
            optim_phi = float(optim_params[idx]) if idx < optim_params.size else 1.0; idx += (1 if idx < optim_params.size else 0)
        else:
            optim_phi = 1.0
    else:
        optim_beta = None; optim_phi = None

    g1 = optim_params[idx : idx + len(k_vector)]; idx += len(k_vector)
    g2 = optim_params[idx : idx + len(k_vector)]; idx += len(k_vector)
    optim_ar = None; optim_ma = None
    if use_arma_errors:
        p_int = int(p); q_int = int(q)
        if p_int != 0 and q_int != 0:
            optim_ar = optim_params[idx : idx + p_int]; optim_ma = optim_params[idx + p_int : idx + p_int + q_int]
        elif p_int != 0:
            optim_ar = optim_params[idx : idx + p_int]
        elif q_int != 0:
            optim_ma = optim_params[idx : idx + q_int]

    w_final = update_w(w, optim_phi, tau, optim_ar, optim_ma, int(p), int(q), optim_beta, dtype)
    g_final, gamma_bold_final = update_g(g, gamma_bold, optim_alpha, optim_beta, k_vector, g1, g2, dtype)
    F_final = update_F(F, optim_phi, optim_alpha, optim_beta, gamma_bold_final, optim_ar, optim_ma, int(p), int(q), tau, dtype)

    if use_boxcox:
        x0_final = _boxcox(x0_untransformed, optim_lambda)
        y_fit_final = _boxcox(y, optim_lambda)
    else:
        x0_final = x0_hat
        y_fit_final = y_fit

    fitted, errors, x_seq = _calc_filter(y_fit_final, w_final, g_final, F_final, x0_final)

    # destandardize fitted if needed
    if not use_boxcox:
        fitted = y_mu + y_sigma * fitted

    sigma2 = jnp.sum(errors * errors) / y_fit_final.shape[0]
    loglike = obj(optim_params / scale)

    kval = int(optim_params.size + x0_final.shape[0])
    if optim_lambda == 1:
        kval -= 1
    if optim_beta is not None and abs(optim_beta) < 1e-8:
        kval -= 1
    if (optim_phi is not None) and (optim_phi == 1.0):
        kval -= 1
    aic = float(loglike) + 2 * kval

    return {
        "fitted": fitted,                  # original scale
        "errors": errors[None, :],         # std/boxcox scale
        "sigma2": sigma2,                  # variance in std/boxcox scale
        "aic": aic,
        "optim_params": jnp.asarray(optim_params, dtype=dtype),
        "F": F_final,
        "w_transpose": w_final,            # shape (1, d)
        "g": g_final,
        "x": x_seq,
        "k_vector": jnp.asarray(k_vector),
        "BoxCox_lambda": optim_lambda,
        "p": int(p),
        "q": int(q),
        "seed_states": x0_final,
        "y_mu": y_mu,                      # for de-standardizing forecasts
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
):
    """Fit a **single** TBATS configuration with the provided k_vector.

    This is a thin wrapper over `tbats_model_generator`, mirroring the StatsForecast
    entry point when k_vector is already chosen.

    Returns:
        Model dict with the same fields as `tbats_model_generator`.
    """
    ar_coeffs = None
    ma_coeffs = None
    best = tbats_model_generator(
        y, seasonal_periods, k_vector, use_boxcox, bc_lower, bc_upper,
        use_trend, use_damped_trend, use_arma_errors, ar_coeffs, ma_coeffs
    )
    best["description"] = {
        "use_boxcox": use_boxcox,
        "use_trend": use_trend,
        "use_damped_trend": use_damped_trend,
        "use_arma_errors": False,
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
):
    """Auto-select core TBATS options over a small grid, including harmonics.

    Args:
        y: 1D series.
        seasonal_periods: list/array of seasonal periods.
        use_boxcox: True/False/None (None → try both).
        use_trend: True/False/None.
        use_damped_trend: True/False/None (ignored if trend=False).
        use_arma_errors: pass-through flag (this core does not auto-ID p,q).

    Returns:
        Best model dict by AIC.

    Differences vs StatsForecast:
        - Same spirit but a smaller, explicit grid for clarity; harmonics are
          chosen with `find_harmonics` per period.
    """
    if (use_trend is False) and (use_damped_trend is True):
        raise ValueError("Can't use damped trend without trend")

    seasonal_periods = jnp.sort(jnp.asarray(seasonal_periods))

    ks: List[int] = []
    z = y
    for period in list(seasonal_periods):
        k, z = find_harmonics(z, int(period))
        ks.append(int(k))
    k_vector = jnp.asarray(ks, dtype=jnp.int32)

    B = [True, False] if use_boxcox is None else [use_boxcox]
    if use_trend is None:
        if use_damped_trend is None:
            T = [(True, True), (True, False), (False, False)]
        elif use_damped_trend:
            T = [(True, True)]
        else:
            T = [(True, False), (False, False)]
    elif use_trend:
        if use_damped_trend is None:
            T = [(True, True), (True, False)]
        elif use_damped_trend:
            T = [(True, True)]
        else:
            T = [(True, False)]
    else:
        T = [(False, False)]

    combos = [(bcx, t, use_arma_errors) for bcx in B for t in T]

    best = {"aic": jnp.inf}
    for bcx, (trend, damped), arma in combos:
        cand = tbats_model(
            y, seasonal_periods, k_vector,
            bcx, bc_lower, bc_upper,
            trend, damped, arma
        )
        if cand["aic"] < best.get("aic", jnp.inf):
            best = cand

    return best


def tbats_forecast(mod, h: int):
    """Multi-step mean forecast from a fitted TBATS model.

    Args:
        mod: model dict from `tbats_model_generator`/`tbats_selection`.
        h: forecast horizon (int).

    Returns:
        {"mean": ndarray of length h} on the **original data scale**.

    Notes:
        - Internally we step the state on the model scale, then:
          * if Box–Cox was disabled → de-standardize by `y_mu`/`y_sigma`;
          * if enabled → inverse Box–Cox via `λ`.
    """
    dtype = mod["F"].dtype
    h = int(h)
    fcst = jnp.zeros((h,), dtype=dtype)
    xx = jnp.zeros((h, mod["x"].shape[1]), dtype=dtype)
    w = mod["w_transpose"][0]  # shape (d,)

    # Forecast on model scale (standardized if Box-Cox disabled, Box-Cox domain if enabled)
    fcst = fcst.at[0].set(jnp.dot(w, mod["x"][-1]))
    xx = xx.at[0].set(jnp.dot(mod["F"], mod["x"][-1]))

    def step(prev_x, _):
        nxt_x = jnp.dot(mod["F"], prev_x)
        yhat = jnp.dot(w, prev_x)
        return nxt_x, (nxt_x, yhat)

    if h > 1:
        _, (xx_tail, yhat_tail) = lax.scan(step, xx[0], jnp.arange(h - 1))
        xx = xx.at[1:].set(xx_tail)
        fcst = fcst.at[1:].set(yhat_tail)

    # --- always return forecasts on the original data scale ---
    if mod["BoxCox_lambda"] is None:
        fcst_orig = mod["y_mu"] + mod["y_sigma"] * fcst
    else:
        fcst_orig = _inv_boxcox(fcst, mod["BoxCox_lambda"])

    return {"mean": fcst_orig}


def compute_sigmah(mod, h: int) -> jnp.ndarray:
    """Parametric forecast std-devs σ_h for horizons 1..h.

    We accumulate var multipliers as:
        var_mult[0] = 1
        var_mult[j] = var_mult[j-1] + (w^T F^j g)^2

    Then:
        sigma2h = sigma2 * var_mult
        sigmah   = sqrt(sigma2h)

    Args:
        mod: fitted model dict (expects "F", "w_transpose", "g", "sigma2",
             "BoxCox_lambda", and (y_mu, y_sigma) for de-standardization).
        h: horizon.

    Returns:
        σ_h on the **original data scale**.

    Differences vs StatsForecast:
        - Explicitly rescales σ_h back to original units when Box–Cox is off
          (multiplying by y_sigma). With Box–Cox on, we keep the linearized
          variance from the model space (standard TBATS treatment).
    """
    F = mod["F"]; w = mod["w_transpose"][0]; g = mod["g"][:, 0]
    dtype = F.dtype
    h = int(h)

    # j = 0 term: conventionally set to 1.0 (obs noise)
    var0 = jnp.asarray(1.0, dtype=dtype)

    if h == 1:
        sigma2h = mod["sigma2"] * var0
        if mod["BoxCox_lambda"] is None:
            sigma2h = (mod["y_sigma"] ** 2) * sigma2h
        return jnp.sqrt(jnp.maximum(sigma2h, 0.0))

    # carry (Fpow, var_acc) with Fpow starting at I for j=0
    def body(carry, _):
        Fpow, var_acc, j = carry
        # advance to F^{j} -> F^{j+1}
        Fpow_next = jnp.dot(Fpow, F)           # now F^{j+1}
        cj = jnp.dot(jnp.dot(w, Fpow_next), g) # c_{j+1}
        var_next = var_acc + cj * cj
        return (Fpow_next, var_next, j + 1), var_next

    init = (jnp.eye(F.shape[1], dtype=dtype), var0, jnp.asarray(0, dtype=jnp.int32))
    _, var_tail = lax.scan(body, init, jnp.arange(h - 1))
    # var_tail contains var_mult[1],...,var_mult[h-1]
    var_mult = jnp.concatenate([var0[None], var_tail], axis=0)

    sigma2h = mod["sigma2"] * var_mult
    if mod["BoxCox_lambda"] is None:
        sigma2h = (mod["y_sigma"] ** 2) * sigma2h
    return jnp.sqrt(jnp.maximum(sigma2h, 0.0))
