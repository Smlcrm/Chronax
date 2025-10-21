# ets_srcv2.py — Logic matched to NumPy/SciPy's Nelder–Mead optimizer
from __future__ import annotations
"""
ETS core simulation and a SciPy-like Nelder–Mead optimizer in JAX.

This module implements the state evolution, forecasting, loss rollout, and an
in-module Nelder–Mead optimizer for Exponential Smoothing / ETS models using
JAX arrays and control flow. The numerical update/forecast logic mirrors
canonical NumPy-style implementations (e.g., statsmodels conventions), while
the optimizer strives to match SciPy's Nelder–Mead behavior as closely as
practical in pure Python/JAX without calling SciPy.

Key pieces:
- `update`   : single-timestep ETS state update (level/trend/season).
- `forecast` : h-step-ahead forecast given current states.
- `_calc_roll` / `calc_full` / `calc` : loss (likelihood/MSE/etc.) rollout
  across a time series, with rolling-horizon MSE tracking.
- `optimize` : a SciPy-like Nelder–Mead implementation (reflection/expansion/
  contraction/shrink) with the same initial simplex rule and a box-penalty for
  bounds.

Design notes:
- JIT: Computational kernels (`update`, `forecast`, `_calc_roll`, `calc_full`,
  `calc`) are `@jax.jit`-compiled with static arguments for model structure.
- Parity: The logic follows NumPy/SciPy conventions for multiplicative corner
  cases (e.g., phi≈1.0, division-by-near-zero guards, multiplicative-season
  positivity checks, and seasonal balancing slots).

Limitations of `optimize` vs SciPy are documented in its docstring.
"""

from enum import Enum
from typing import NamedTuple, Tuple
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
#import optimistix as optx

# ---------------------------
# Constants
# ---------------------------
HUGE_N: float = 1e10
"""A very large sentinel value used to avoid division by ~0 in multiplicative cases."""

NA: float = -99999.0
"""Legacy sentinel value; retained for parity with upstream behavior."""

TOL: float = 1e-10
"""Small tolerance used for near-zero/near-one checks (e.g., phi≈1.0)."""


class Component(Enum):
    """
    ETS component type.

    Values
    ------
    Nothing : No component present.
    Additive : Additive dynamics for trend/season/error.
    Multiplicative : Multiplicative dynamics for trend/season/error.
    """
    Nothing = 0
    Additive = 1
    Multiplicative = 2


class Criterion(Enum):
    """
    Optimization objective to minimize.

    Values
    ------
    Likelihood : Gaussian log-likelihood proxy used by ETS implementations.
    MSE        : One-step mean squared error.
    AMSE       : Average MSE across horizons up to `n_mse` (capped at 30).
    Sigma      : Mean of squared residuals (variance proxy).
    MAE        : Mean absolute error.
    """
    Likelihood = 0
    MSE = 1
    AMSE = 2
    Sigma = 3
    MAE = 4


class OptimResult(NamedTuple):
    """
    Result of the Nelder–Mead optimization.

    Attributes
    ----------
    success : bool
        Whether termination was successful under the convergence test.
    status : int
        0 for success; 2 if maximum iterations exceeded (SciPy-style codes).
    message : str
        Human-readable status message.
    x : jnp.ndarray
        Best-found parameter vector.
    fun : float
        Objective value at `x`.
    nit : int
        Number of iterations performed.
    nfev : int
        Number of objective evaluations performed.
    """
    success: bool
    status: int
    message: str
    x: jnp.ndarray
    fun: float
    nit: int
    nfev: int


# ---------------------------
# Core state update (NumPy logic)
# ---------------------------
@partial(jax.jit, static_argnames=("trend", "season", "m"))
def update(
    s: jnp.ndarray,
    l: jnp.float64,
    b: jnp.float64,
    old_l: jnp.float64,
    old_b: jnp.float64,
    old_s: jnp.ndarray,
    m: int,
    trend: Component,
    season: Component,
    alpha: jnp.float64,
    beta: jnp.float64,
    gamma: jnp.float64,
    phi: jnp.float64,
    y: jnp.float64,
) -> Tuple[jnp.float64, jnp.float64, jnp.ndarray]:
    """
    One-step ETS state update.

    Mirrors NumPy/statsmodels-like formulas for level `l`, trend `b`, and
    seasonal vector `s` given previous states (`old_l`, `old_b`, `old_s`), the
    current observation `y`, and smoothing parameters.

    Parameters
    ----------
    s, l, b : current seasonal vector, level, trend (updated in-place conceptually)
    old_l, old_b, old_s : previous states used for update
    m : int
        Season length; `max(m, 1)` is used in callers.
    trend, season : Component
        Structural flags controlling additive/multiplicative behavior.
    alpha, beta, gamma, phi : jnp.float64
        Smoothing parameters (beta/phi used only if trend present; gamma only if season present).
    y : jnp.float64
        Current observation.

    Returns
    -------
    l_new, b_new, s_new : updated states
    """
    # Multiplicative trend helper: branch on |phi-1|<TOL
    def mul_trend_case(phi, old_l, old_b):
        cond = jnp.abs(phi - 1.0) < TOL
        phi_b_local = jnp.where(cond, old_b, old_b ** phi)
        q_local = jnp.where(cond, old_l * old_b, old_l * phi_b_local)
        return phi_b_local, q_local

    # q (pre-smoothing level) and phi_b depend on trend
    if trend == Component.Nothing:
        q = old_l
        phi_b = jnp.asarray(0.0, jnp.float64)
    elif trend == Component.Additive:
        phi_b = phi * old_b
        q = old_l + phi_b
    else:  # Multiplicative
        phi_b, q = mul_trend_case(phi, old_l, old_b)

    # seasonally adjusted observation p
    if season == Component.Nothing:
        p = y
    elif season == Component.Additive:
        p = y - old_s[m - 1]
    else:  # Multiplicative
        denom = old_s[m - 1]
        p = jnp.where(jnp.abs(denom) < TOL, jnp.asarray(HUGE_N, jnp.float64), y / denom)

    # new level
    l = q + alpha * (p - q)

    # new growth (if trend present) — NumPy uses (beta/alpha) without alpha guard
    if trend != Component.Nothing:
        if trend == Component.Additive:
            r = l - old_l
        else:
            r = jnp.where(jnp.abs(old_l) < TOL, jnp.asarray(HUGE_N, jnp.float64), l / old_l)
        b = phi_b + (beta / alpha) * (r - phi_b)

    # new seasonal (if present)
    if season != Component.Nothing:
        if season == Component.Additive:
            t = y - q
        else:
            t = jnp.where(jnp.abs(q) < TOL, jnp.asarray(HUGE_N, jnp.float64), y / q)
        s0 = old_s[m - 1] + gamma * (t - old_s[m - 1])
        s = s.at[0].set(s0)
        if m > 1:
            s = s.at[1:m].set(old_s[0:m - 1])

    return l, b, s


# ---------------------------
# h-step forecast (NumPy logic) — phistar fixed
# ---------------------------
@partial(jax.jit, static_argnames=("trend", "season", "m"))
def forecast(
    f: jnp.ndarray,
    l: jnp.float64,
    b: jnp.float64,
    s: jnp.ndarray,
    m: int,
    trend: Component,
    season: Component,
    phi: jnp.float64,
    h: jnp.int64,
) -> jnp.ndarray:
    """
    Multi-step ETS forecast from a given state.

    Builds `h` forecasts into `f` using additive/multiplicative trend/season
    conventions, with special handling when `phi ≈ 1`.

    Parameters
    ----------
    f : jnp.ndarray
        Preallocated buffer (length ≥ h) to hold forecasts.
    l, b, s : jnp.float64, jnp.float64, jnp.ndarray
        Current level, trend, and seasonal states.
    m : int
        Season length.
    trend, season : Component
        Structural flags for trend and seasonality.
    phi : jnp.float64
        Trend damping parameter.
    h : jnp.int64
        Forecast horizon.

    Returns
    -------
    f : jnp.ndarray
        Same buffer with indices [0..h-1] filled.
    """
    def body(i, carry):
        f, phistar = carry

        # Update phistar to sum_{k=1}^{i+1} phi^k, with special case phi≈1
        incr = jnp.where(jnp.abs(phi - 1.0) < TOL, 1.0, phi ** (i + 1))
        phistar = phistar + incr

        # Trend contribution
        def fi_mul():
            # NumPy behavior: if b < 0 => NaN, else l * (b ** phistar)
            return jnp.where(b < 0.0, jnp.asarray(jnp.nan, jnp.float64), l * (b ** phistar))

        fi = jnp.where(
            trend == Component.Nothing, l,
            jnp.where(trend == Component.Additive, l + phistar * b, fi_mul())
        )

        # Seasonal contribution, j = (m - 1 - i) % m
        j_idx = (m - 1 - i) % m
        fi = jnp.where(
            season == Component.Additive, fi + s[j_idx],
            jnp.where(season == Component.Multiplicative, fi * s[j_idx], fi)
        )

        f = f.at[i].set(fi)
        return (f, phistar)

    phistar0 = jnp.asarray(0.0, jnp.float64)
    f, _ = lax.fori_loop(0, h, body, (f, phistar0))
    return f


# ---------------------------
# Likelihood + error rollout (NumPy logic) — scan/fori_loop, dynamic slices fixed
# ---------------------------
@partial(jax.jit, static_argnames=("error", "trend", "season", "n_mse", "m"))
def _calc_roll(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: jnp.float64,
    beta: jnp.float64,
    gamma: jnp.float64,
    phi: jnp.float64,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.float64]:
    """
    Core time-series rollout computing residuals and objective pieces.

    This function iterates through `y`, producing one-step residuals `e`,
    rolling multi-horizon MSEs `a_mse[0:k]`, and an ETS likelihood-like score.

    Parameters
    ----------
    x : jnp.ndarray
        Flat buffer holding (n+1) state snapshots; the first state slice must
        be initialized by caller. This function writes subsequent slices.
    e : jnp.ndarray
        Residual buffer of length n (zero-initialized by caller).
    a_mse : jnp.ndarray
        Rolling MSE buffer of length ≥ min(n_mse, 30).
    n_mse : int
        Maximum horizon for AMSE tracking (capped at 30).
    y : jnp.ndarray
        Observations (length n).
    error, trend, season : Component
        Model structure flags.
    alpha, beta, gamma, phi : jnp.float64
        Smoothing parameters.
    m : int
        Season length.

    Returns
    -------
    x_local : jnp.ndarray
        Updated state buffer with (n+1) snapshots packed.
    e_local : jnp.ndarray
        Residuals for each time step.
    a_local : jnp.ndarray
        Rolling horizon MSE estimates.
    lik : jnp.float64
        Objective component for `Criterion.Likelihood`.
    """
    n = y.shape[0]
    n_s = max(m, 24)
    m_eff = max(m, 1)
    n_mse_eff = min(n_mse, 30)

    n_states = m_eff * int(season != Component.Nothing) + int(trend != Component.Nothing) + 1

    # Unpack initial states from x
    l = x[0]
    b = jnp.array(0.0, jnp.float64)
    if trend != Component.Nothing:
        b = x[1]

    s_vec = jnp.zeros(n_s, dtype=jnp.float64)
    if season != Component.Nothing:
        start0 = 1 + int(trend != Component.Nothing)
        s_vec = s_vec.at[:m_eff].set(x[start0:start0 + m_eff])

    # Work buffers
    x_local = x
    e_local = e.at[:].set(0.0)
    a_local = a_mse.at[:].set(0.0)
    denom = jnp.zeros(30, dtype=jnp.float64)
    lik = jnp.array(0.0, jnp.float64)
    lik2 = jnp.array(0.0, jnp.float64)

    # f-buffer of fixed max size 30; we only use first n_mse_eff entries
    f_buf = jnp.zeros(30, dtype=jnp.float64)

    def step(carry, y_i):
        (i, x_local, e_local, a_local, denom, l, b, s_vec, lik, lik2, f_buf) = carry

        old_l = l
        old_b = jnp.where(trend != Component.Nothing, b, jnp.asarray(0.0, jnp.float64))
        old_s = jnp.zeros_like(s_vec).at[:m_eff].set(s_vec[:m_eff])

        # forecasts up to n_mse_eff
        f_buf = forecast(
            f_buf, old_l, old_b, old_s, m_eff, trend, season, phi, jnp.int64(n_mse_eff)
        )

        # residual for this step
        f0 = f_buf[0]
        if error == Component.Additive:
            ei = y_i - f0
        else:
            f0_denom = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
            ei = (y_i - f0) / f0_denom

        e_local = e_local.at[i].set(ei)

        # rolling MSEs over horizon (bounded by sequence end)
        def mse_body(j, a_denom):
            a_local, denom = a_denom
            ij = i + j
            cond = ij < n
            denom_j_new = denom[j] + jnp.where(cond, 1.0, 0.0)
            tmp = jnp.where(cond, y[ij] - f_buf[j], 0.0)
            a_num = a_local[j] * (denom[j] - 1.0) + tmp * tmp
            a_new = jnp.where(denom_j_new > 0.0, a_num / denom_j_new, a_local[j])
            a_local = a_local.at[j].set(a_new)
            denom = denom.at[j].set(denom_j_new)
            return (a_local, denom)

        a_local, denom = lax.fori_loop(0, n_mse_eff, mse_body, (a_local, denom))

        # state update with current observation
        l, b, s_vec = update(
            s_vec, l, b, old_l, old_b, old_s, m_eff,
            trend, season, alpha, beta, gamma, phi, y_i
        )

        # store back states for time i+1
        base = n_states * (i + 1)

        # l at base
        x_local = lax.dynamic_update_slice(x_local, jnp.asarray([l]), (base,))
        # b at base+1 if trend present
        if trend != Component.Nothing:
            x_local = lax.dynamic_update_slice(x_local, jnp.asarray([b]), (base + 1,))

        # seasonal block at dynamic start if season present
        if season != Component.Nothing:
            start = base + 1 + int(trend != Component.Nothing)
            x_local = lax.dynamic_update_slice(x_local, s_vec[:m_eff], (start,))

        # accumulate likelihood bits
        lik = lik + ei * ei
        val = jnp.abs(f0)
        log_arg = jnp.where(val > 0.0, val, val + 1e-8)
        lik2 = lik2 + jnp.log(log_arg)

        i = i + 1
        return (i, x_local, e_local, a_local, denom, l, b, s_vec, lik, lik2, f_buf), None

    init_carry = (jnp.array(0, jnp.int32), x_local, e_local, a_local, denom, l, b, s_vec, lik, lik2, f_buf)
    (i_out, x_local, e_local, a_local, denom, l, b, s_vec, lik, lik2, f_buf), _ = lax.scan(
        step, init_carry, y
    )

    n_f64 = jnp.asarray(n, dtype=jnp.float64)
    lik_log_arg = jnp.where(lik > 0.0, lik, lik + 1e-8)
    lik = n_f64 * jnp.log(lik_log_arg)
    lik = jnp.where(error == Component.Multiplicative, lik + 2.0 * lik2, lik)

    return x_local, e_local, a_local, lik


@partial(jax.jit, static_argnames=("error", "trend", "season", "n_mse", "m"))
def calc_full(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: jnp.float64,
    beta: jnp.float64,
    gamma: jnp.float64,
    phi: jnp.float64,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.float64]:
    """
    Convenience wrapper that performs a full rollout and returns states.

    Parameters
    ----------
    x, e, a_mse : jnp.ndarray
        Working buffers (see `_calc_roll`). `x` must contain the initial state
        in its leading slice; this function arranges storage for (n+1) slices.
    n_mse : int
        Horizon cap for rolling MSE metrics (≤30).
    y : jnp.ndarray
        Observations.
    error, trend, season : Component
        Model structure flags.
    alpha, beta, gamma, phi : jnp.float64
        Smoothing parameters.
    m : int
        Season length.

    Returns
    -------
    a_work : jnp.ndarray
        Rolling MSEs.
    e_work : jnp.ndarray
        One-step residuals.
    states : jnp.ndarray
        Reshaped state snapshots of shape (n+1, n_states).
    lik : jnp.float64
        Likelihood-style objective value.
    """
    n = y.shape[0]
    m_eff = max(m, 1)
    n_states = m_eff * int(season != Component.Nothing) + int(trend != Component.Nothing) + 1

    add_season_balancer = int(season != Component.Nothing)
    P = n_states + add_season_balancer

    x_work = jnp.zeros(P * (n + 1), dtype=jnp.float64)
    e_work = jnp.zeros_like(e)
    a_work = jnp.zeros_like(a_mse)
    x_work = x_work.at[:n_states].set(x[:n_states])  # initial states

    x_work, e_work, a_work, lik = _calc_roll(
        x_work, e_work, a_work, n_mse, y,
        error, trend, season, alpha, beta, gamma, phi, m
    )

    # reshape only the true state block, excluding the balancing slot
    states = x_work[: (n + 1) * n_states].reshape((n + 1, n_states))
    return a_work, e_work, states, lik


@partial(jax.jit, static_argnames=("error", "trend", "season", "n_mse", "m"))
def calc(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: jnp.float64,
    beta: jnp.float64,
    gamma: jnp.float64,
    phi: jnp.float64,
    m: int,
) -> jnp.float64:
    """
    Compute only the scalar objective (e.g., likelihood) for given parameters.

    A thin wrapper over `calc_full` that discards intermediate arrays and
    returns the likelihood-style scalar used by `Criterion.Likelihood`.

    Returns
    -------
    lik : jnp.float64
        Objective value for the given parameters and data.
    """
    _, _, _, lik = calc_full(x, e, a_mse, n_mse, y, error, trend, season, alpha, beta, gamma, phi, m)
    return lik


# ---------------------------
# Objective 
# ---------------------------
def _objective_from_params(
    p: jnp.ndarray,
    y: jnp.ndarray,
    n_state: int,
    error: Component,
    trend: Component,
    season: Component,
    opt_crit: Criterion,
    n_mse: int,
    m: int,
    opt_alpha: bool,
    opt_beta: bool,
    opt_gamma: bool,
    opt_phi: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
) -> jnp.float64:
    """
    Assemble smoothing+state parameters from a flat vector and evaluate the objective.

    This unpacks `(alpha, beta, gamma, phi)` (respecting which are being
    optimized) and the initial state vector (the trailing `n_state` elements
    of `p`), applies seasonal balancing (for multiplicative or additive cases
    consistent with NumPy parity), runs `_calc_roll`, and computes the scalar
    objective per `opt_crit`.

    Parameters
    ----------
    p : jnp.ndarray
        Concatenated parameter vector: first the free smoothing params (in the
        order alpha, beta, gamma, phi for those flagged as `opt_*`), then the
        `n_state` initial state entries.
    y : jnp.ndarray
        Observations.
    n_state : int
        Length of the initial state vector.
    error, trend, season : Component
        Model structure flags.
    opt_crit : Criterion
        Objective to minimize.
    n_mse, m : int
        Rolling-MSE horizon cap and season length.
    opt_alpha, opt_beta, opt_gamma, opt_phi : bool
        Which smoothing parameters are free in `p`.
    alpha, beta, gamma, phi : float
        Fixed values for smoothing parameters that are not optimized.

    Returns
    -------
    obj_val : jnp.float64
        Scalar objective value with +inf for invalid setups.
    """
    # Unpack smoothing parameters like NumPy
    j = 0
    a  = jnp.asarray(alpha, jnp.float64)
    b  = jnp.asarray(beta,  jnp.float64)
    g  = jnp.asarray(gamma, jnp.float64)
    ph = jnp.asarray(phi,   jnp.float64)
    if opt_alpha:
        a = p[j]; j += 1
    if opt_beta:
        b = p[j]; j += 1
    if opt_gamma:
        g = p[j]; j += 1
    if opt_phi:
        ph = p[j]; j += 1

    n_params = p.size
    n = y.size

    add_season_balancer = int(season != Component.Nothing)
    P = n_state + add_season_balancer
    state = jnp.zeros(P * (n + 1), dtype=jnp.float64)

    # head states from tail of p
    head = p[n_params - n_state : n_params]
    state = state.at[:n_state].set(head)

    # seasonal balancing term (NumPy parity)
    if season != Component.Nothing:
        start = 1 + int(trend != Component.Nothing)
        s_sum = jnp.sum(state[start:n_state])
        target = m * int(season == Component.Multiplicative)
        state = state.at[n_state].set(jnp.asarray(target, jnp.float64) - s_sum)

    # EXACT NumPy parity: multiplicative => check ALL seasonal entries INCLUDING balancing slot
    if season != Component.Nothing:
        start = 1 + int(trend != Component.Nothing)
        seasonal_block_including_balance = state[start:]
    else:
        seasonal_block_including_balance = jnp.zeros(0, dtype=jnp.float64)

    neg_any = jnp.any(seasonal_block_including_balance < 0.0) if season == Component.Multiplicative else jnp.array(False)
    cond_neg = jnp.logical_and(jnp.array(season == Component.Multiplicative), neg_any)

    # Compute rollout
    a_mse = jnp.zeros(30, dtype=jnp.float64)
    e     = jnp.zeros(n,  dtype=jnp.float64)
    _, e, a_mse, lik = _calc_roll(state, e, a_mse, n_mse, y, error, trend, season, a, b, g, ph, m)

    # NumPy post-processing
    lik = jnp.maximum(lik, jnp.asarray(-1e10, jnp.float64))
    bad = jnp.logical_or(jnp.isnan(lik), jnp.abs(lik + 99999.0) < 1e-7)

    k = min(n_mse, 30)
    obj_val = jnp.select(
        [
            opt_crit == Criterion.Likelihood,
            opt_crit == Criterion.MSE,
            opt_crit == Criterion.AMSE,
            opt_crit == Criterion.Sigma,
            opt_crit == Criterion.MAE,
        ],
        [lik, a_mse[0], jnp.mean(a_mse[:k]), jnp.mean(e * e), jnp.mean(jnp.abs(e))],
        default=lik,
    )

    # IMPORTANT: invalids must be +inf for a minimizer
    obj_val = jnp.where(bad, jnp.asarray(jnp.inf, jnp.float64), obj_val)
    # keep multiplicative-season negativity as +inf (reject)
    obj_val = jnp.where(cond_neg, jnp.asarray(jnp.inf, jnp.float64), obj_val)

    return obj_val


# ---------------------------
# Optimizer (matches SciPy's Nelder-Mead behavior closely)
# ---------------------------
def optimize(
    x0: jnp.ndarray,
    y: jnp.ndarray,
    n_state: int,
    error: Component,
    trend: Component,
    season: Component,
    opt_crit: Criterion,
    n_mse: int,
    m: int,
    opt_alpha: bool,
    opt_beta: bool,
    opt_gamma: bool,
    opt_phi: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    tol_std: float,
    max_iter: int,
    adaptive: bool,
) -> OptimResult:
    """
    Nelder–Mead (SciPy-like) direct-search optimizer with box-penalty bounds.

    This reimplements the core SciPy `method="Nelder-Mead"` loop:
    - Initial simplex built with `nonzdelt=0.05` and `zdelt=0.00025`.
    - Classic coefficients: rho=1 (reflect), chi=2 (expand), psi=0.5 (contract),
      sigma=0.5 (shrink).
    - Convergence test uses a simple "fatol"-style check: `max(f) - min(f) <= tol_std`.
    - Bounds are enforced via a large quadratic penalty on violations.

    Parameters
    ----------
    x0 : jnp.ndarray
        Initial parameter vector (free smooth params + initial states).
    y : jnp.ndarray
        Observations.
    n_state : int
        Length of the initial state portion at the tail of `x0`.
    error, trend, season : Component
        Model structure flags.
    opt_crit : Criterion
        Objective to minimize.
    n_mse, m : int
        Rolling-MSE horizon cap and season length.
    opt_alpha, opt_beta, opt_gamma, opt_phi : bool
        Which smoothing parameters are free and included in `x0`.
    alpha, beta, gamma, phi : float
        Fixed smoothing parameters for those not optimized.
    lower, upper : jnp.ndarray
        Elementwise (soft) bounds. Violations incur a 1e6 * L2 penalty.
    tol_std : float
        Function-value range tolerance for termination (fatol-like).
    max_iter : int
        Maximum number of Nelder–Mead iterations.
    adaptive : bool
        Accepted for API parity; classic (non-adaptive) coefficients are used.

    Returns
    -------
    OptimResult
        Tuple-like result with fields: success, status (0 ok, 2 max iters),
        message, x (best params), fun, nit, nfev.

    Notes
    -----
    **Intended Parity with SciPy**
    - Initial simplex construction and the reflection/expansion/contraction/
      shrink decisions follow SciPy behavior closely for *deterministic* parity.
    - We terminate on a fatol-style condition (`max(f)-min(f) <= tol_std`),
      which typically aligns with using `fatol` in SciPy.

    **Limitations vs SciPy's Nelder–Mead**
    - No `xatol` (vertex spread) termination: only a function-range (fatol-like)
      test is used.
    - No `callback`, `return_all`, or `maxfev` controls; `nfev` is tracked but
      stopping is by `max_iter` only.
    - Bounds are *soft* via a quadratic penalty (1e6 * ||violation||^2); SciPy
      NM itself is unconstrained, so exact projection/box-simplex logic is not used.
    - `adaptive=True` is accepted but ignored; coefficients remain classic
      (rho=1, chi=2, psi=0.5, sigma=0.5) to keep behavior predictable.
    - The objective is JAX-based, but the NM loop uses Python control flow;
      it is **not** `jit`-compiled end-to-end (the inner objective calculations
      are JITed). This mirrors SciPy's imperative loop style.
    - Numerical defensive guards (e.g., `TOL`, `HUGE_N`, multiplicative-season
      positivity) follow NumPy/statsmodels parity; edge cases may still differ
      slightly from SciPy due to floating-point ordering and JAX evaluation.

    If you need *exact* SciPy semantics (adaptive coefficients, xatol/fatol,
    callback, strict `maxfev`, etc.), use `scipy.optimize.minimize` directly.
    """
    x0    = jnp.asarray(x0, dtype=jnp.float64)
    y     = jnp.asarray(y,  dtype=jnp.float64)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    # Core objective (identical to before)
    def _core_obj(p: jnp.ndarray) -> jnp.float64:
        return _objective_from_params(
            p, y, n_state, error, trend, season, opt_crit, n_mse, m,
            opt_alpha, opt_beta, opt_gamma, opt_phi, alpha, beta, gamma, phi
        )

    # Box penalty (identical to before)
    def boxed_objective(p: jnp.ndarray) -> jnp.float64:
        vio_low = jnp.maximum(0.0, lower - p)
        vio_up  = jnp.maximum(0.0, p - upper)
        vio = jnp.dot(vio_low, vio_low) + jnp.dot(vio_up, vio_up)
        penalty = 1e6 * vio
        core = _core_obj(p)
        return jnp.where(vio > 0.0, penalty, core)

    # ------- Build initial simplex exactly like SciPy -------
    N = int(x0.shape[0])
    sim = jnp.zeros((N + 1, N), dtype=jnp.float64)
    sim = sim.at[0].set(x0)
    for k in range(N):
        y_k = jnp.copy(x0)
        if float(y_k[k]) != 0.0:
            y_k = y_k.at[k].set((1.0 + 0.05) * y_k[k])  # nonzdelt=0.05
        else:
            y_k = y_k.at[k].set(0.00025)                # zdelt=0.00025
        sim = sim.at[k + 1].set(y_k)

    # Evaluate initial simplex
    fsim = jnp.array([boxed_objective(sim[i]) for i in range(N + 1)], dtype=jnp.float64)

    # Sort by function value
    order = jnp.argsort(fsim)
    sim = sim[order]
    fsim = fsim[order]

    # ------- NM hyperparameters (SciPy defaults) -------
    rho = 1.0     # reflection
    chi = 2.0     # expansion
    psi = 0.5     # contraction
    sigma = 0.5   # shrink

    # Adaptive NM changes chi/psi/sigma; we keep classic constants for parity
    # with SciPy's non-adaptive default. (Adaptive flag accepted but unused.)

    # ------- Iterate -------
    nit = 0
    nfev = int(N + 1)

    def eval_point(p):
        nonlocal nfev
        nfev += 1
        return float(boxed_objective(p))

    while nit < max_iter:
        # Convergence test on function range (fatol-like)
        fmax = float(fsim[-1])
        fmin = float(fsim[0])
        if (fmax - fmin) <= tol_std:
            break

        # Centroid of all but worst
        x_bar = jnp.mean(sim[:-1], axis=0)

        # Reflection
        xr = x_bar + rho * (x_bar - sim[-1])
        fr = eval_point(xr)

        if fr < float(fsim[0]):
            # Expansion
            xe = x_bar + chi * (xr - x_bar)
            fe = eval_point(xe)
            if fe < fr:
                sim = sim.at[-1].set(xe); fsim = fsim.at[-1].set(fe)
            else:
                sim = sim.at[-1].set(xr); fsim = fsim.at[-1].set(fr)
        else:
            if fr < float(fsim[-2]):
                # Accept reflection between best and second-worst
                sim = sim.at[-1].set(xr); fsim = fsim.at[-1].set(fr)
            else:
                # Contraction
                if fr < float(fsim[-1]):
                    # Outside contraction
                    xc = x_bar + psi * (xr - x_bar)
                else:
                    # Inside contraction
                    xc = x_bar - psi * (x_bar - sim[-1])
                fc = eval_point(xc)
                if fc <= min(fr, float(fsim[-1])):
                    sim = sim.at[-1].set(xc); fsim = fsim.at[-1].set(fc)
                else:
                    # Shrink
                    x0_best = sim[0]
                    new_rows = [x0_best + sigma * (sim[i] - x0_best) for i in range(1, N + 1)]
                    new_rows = jnp.stack(new_rows, axis=0)
                    # Evaluate shrunk points
                    new_vals = jnp.array([eval_point(new_rows[i]) for i in range(N)], dtype=jnp.float64)
                    sim = sim.at[1:].set(new_rows)
                    fsim = fsim.at[1:].set(new_vals)

        # Re-sort
        order = jnp.argsort(fsim)
        sim = sim[order]
        fsim = fsim[order]

        nit += 1

    p_star = jnp.asarray(sim[0], dtype=jnp.float64)
    f_star = float(fsim[0])
    success = bool((f_star == f_star) and jnp.isfinite(p_star).all().item())  # finite & not NaN

    # SciPy status mapping: 0=ok, 2=max iter
    status = 0 if success and (nit < max_iter) else 2
    msg = "ok" if status == 0 else "Maximum number of iterations has been exceeded."

    return OptimResult(
        success=success,
        status=status,
        message=msg,
        x=p_star,
        fun=f_star,
        nit=nit,
        nfev=nfev,
    )


__all__ = [
    "HUGE_N","NA","TOL","Component","Criterion","OptimResult",
    "update","forecast","calc_full","calc","optimize",
]
