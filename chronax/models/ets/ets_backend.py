"""ETS core simulation and optax-based parameter optimization in JAX.

This module provides the low-level building blocks for Exponential Smoothing
(ETS) models.  It is consumed by ``ets_functions.py`` (model selection) and
``auto_ets.py`` (public API).

Key components
--------------
* **State primitives** — :func:`update` (one-step ETS recursion) and
  :func:`forecast` (multi-step ahead prediction from a given state).
* **Rollout helpers** — :func:`calc_full` and :func:`calc` iterate through
  an entire series, producing residuals, AMSE, and a likelihood-style scalar.
* **Optimizer** — :func:`optimize_bfgs_smoothing` runs a two-phase strategy:

  - *Phase 1*: Adam warm-up (first-order, ~15–30 steps in a Python loop).
  - *Phase 2*: L-BFGS refinement (second-order, 30 steps via ``lax.scan``
    compiled into a single XLA kernel).

Only ``optax`` is used for optimization — no ``jaxopt`` or ``scipy``
dependency.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache, partial
from typing import Any, Callable, NamedTuple, Tuple
import os

import jax
jax.config.update("jax_enable_x64", True)  # ETS needs float64 precision
import jax.numpy as jnp
from jax import lax
import optax  # Adam + L-BFGS optimizers and zoom line-search


def _init_jax_compilation_cache() -> None:
    """Set up a local on-disk XLA compilation cache to avoid redundant recompiles."""
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir


_init_jax_compilation_cache()

# ---------------------------
# Constants
# ---------------------------
HUGE_N: float = 1e10
"""Sentinel replacing near-zero denominators in multiplicative ETS formulas."""

NA: float = -99999.0
"""Legacy sentinel for missing / invalid values; retained for upstream parity."""

TOL: float = 1e-10
"""Near-zero guard for conditional branches (e.g. |φ − 1| < TOL)."""

EPS: float = 1e-3
"""Clipping margin used by the *legacy* sigmoid parameter transform."""

EPS_PURE: float = 2e-2
"""Clipping margin used by the *pure* sigmoid parameter transform."""

PHI_LOWER: float = 0.8
"""Default lower bound for the trend-damping parameter φ."""

PHI_UPPER: float = 0.98
"""Default upper bound for the trend-damping parameter φ."""


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
    Result of the Optax-based optimization.

    Attributes
    ----------
    success : jnp.ndarray
        Traced boolean scalar — whether the best parameters and loss are finite.
        (A jnp value, not a Python bool: the optimizer must stay traceable
        under vmap, so no concretizing casts are allowed here.)
    status : int
        Always 0 (kept for API compatibility).
    message : str
        Human-readable status message.
    x : jnp.ndarray
        Best-found parameter vector.
    fun : jnp.ndarray
        Objective value at `x` (traced scalar).
    nit : int
        Number of iterations performed (static).
    nfev : int
        Number of objective evaluations performed (static).
    """
    success: jnp.ndarray
    status: int
    message: str
    x: jnp.ndarray
    fun: jnp.ndarray
    nit: int
    nfev: int


@lru_cache(maxsize=128)
def _get_objective_closure(
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
    clip_multiplicative_errors: bool,
    pure_sigmoid: bool,
    opt_init_state: bool,
    n_state: int,
) -> Callable[..., jnp.float64]:
    """Return a cached objective closure for a specific ETS model structure.

    JAX only traces / compiles the objective once per unique combination of
    structural flags (error, trend, season, which parameters are free, etc.).
    The returned callable accepts *data*-level arrays and produces a scalar
    loss — ready for ``jax.value_and_grad``.

    Parameters
    ----------
    error, trend, season : Component
        Model structure flags (baked into the closure at creation time).
    opt_crit : Criterion
        Which loss metric to minimise (likelihood, MSE, …).
    n_mse : int
        AMSE horizon cap (≤ 30).
    m : int
        Seasonal period.
    opt_alpha, opt_beta, opt_gamma, opt_phi : bool
        Which smoothing parameters are free (True) vs. fixed (False).
    clip_multiplicative_errors : bool
        Whether to clamp multiplicative residuals to ``[-2, 2]``.
    pure_sigmoid : bool
        Use the cleaner independent sigmoid parameterisation.
    opt_init_state : bool
        If ``True``, the tail of ``p`` holds optimisable initial states.
    n_state : int
        Number of initial-state parameters appended to ``p``.

    Returns
    -------
    Callable
        ``_obj(p, y, init_state, n_obs, alpha, beta, gamma, phi, lower,
        upper) -> jnp.float64`` — the scalar objective.
    """
    # Inner callable receives data-level arrays; all structure config is baked in.
    def _obj(
        p: jnp.ndarray,
        y: jnp.ndarray,
        init_state: jnp.ndarray,
        n_obs: int,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ) -> jnp.float64:
        """Evaluate the scalar ETS objective for a fixed structural configuration."""
        return _objective_smoothing_only(
            p,
            y,
            init_state,
            error,
            trend,
            season,
            opt_crit,
            n_mse,
            m,
            n_obs,
            opt_alpha,
            opt_beta,
            opt_gamma,
            opt_phi,
            alpha,
            beta,
            gamma,
            phi,
            lower,
            upper,
            clip_multiplicative_errors,
            pure_sigmoid,
            opt_init_state,
            n_state,
        )

    return _obj


# ---------------------------------------------------------------------------
# Integer aliases for Component values — used inside ``lax.select`` branches
# where Python-level enum comparison is not possible.
# ---------------------------------------------------------------------------
_COMP_NOTHING: int = int(Component.Nothing.value)
_COMP_ADD: int = int(Component.Additive.value)
_COMP_MUL: int = int(Component.Multiplicative.value)


@partial(jax.jit, static_argnames=("has_trend", "has_season", "m"))
def _unpack_state(
    state: jnp.ndarray,
    has_trend: bool,
    has_season: bool,
    m: int,
) -> Tuple[jnp.float64, jnp.float64, jnp.ndarray]:
    """Unpack level, trend, and seasonal vector from a packed ETS state array.

    Parameters
    ----------
    state : jnp.ndarray
        Flat state vector ``[level (, trend) (, season_1 … season_m)]``.
    has_trend, has_season : bool
        Whether the model contains a trend / seasonal component.
    m : int
        Seasonal period (used only when ``has_season`` is True).

    Returns
    -------
    l : jnp.float64
        Current level.
    b : jnp.float64
        Current trend (``0.0`` if ``has_trend`` is False).
    s_vec : jnp.ndarray
        Seasonal vector of length ``max(m, 24)`` (zero-padded when no season).
    """
    n_s = max(m, 24)
    l = state[0]
    b = state[1] if has_trend else jnp.asarray(0.0, dtype=state.dtype)
    s_vec = jnp.zeros((n_s,), dtype=state.dtype)
    if has_season:
        start = 1 + int(has_trend)
        s_vec = s_vec.at[:m].set(state[start:start + m])
    return l, b, s_vec


@partial(jax.jit, static_argnames=("has_trend", "has_season", "m"))
def _pack_state_row(
    l: jnp.float64,
    b: jnp.float64,
    s_vec: jnp.ndarray,
    has_trend: bool,
    has_season: bool,
    m: int,
) -> jnp.ndarray:
    """Pack level, trend, and seasonal values into a single flat state row.

    Parameters
    ----------
    l : jnp.float64
        Level value.
    b : jnp.float64
        Trend value (ignored when ``has_trend`` is False).
    s_vec : jnp.ndarray
        Seasonal vector (first ``m`` entries are used).
    has_trend, has_season : bool
        Whether the model has a trend / seasonal component.
    m : int
        Seasonal period.

    Returns
    -------
    jnp.ndarray
        Flat state row of length ``m * has_season + has_trend + 1``.
    """
    dtype = jnp.asarray(l).dtype
    n_states = m * int(has_season) + int(has_trend) + 1
    row = jnp.zeros((n_states,), dtype=dtype)
    row = row.at[0].set(l)
    if has_trend:
        row = row.at[1].set(b)
    if has_season:
        start = 1 + int(has_trend)
        row = row.at[start:start + m].set(s_vec[:m])
    return row


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
    error: int,
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
    def mul_trend_case(
        phi: jnp.float64, old_l: jnp.float64, old_b: jnp.float64
    ) -> Tuple[jnp.float64, jnp.float64]:
        """Compute the multiplicative-trend carry-over terms."""
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

    # Additive-error update path
    e_add = p - q
    l_add = q + alpha * e_add
    b_add = b
    if trend != Component.Nothing:
        if trend == Component.Additive:
            r_add = l_add - old_l
        else:
            r_add = jnp.where(jnp.abs(old_l) < TOL, jnp.asarray(HUGE_N, jnp.float64), l_add / old_l)
        b_add = phi_b + (beta / alpha) * (r_add - phi_b)
    s_add = s
    if season != Component.Nothing:
        if season == Component.Additive:
            t_add = y - q
        else:
            t_add = jnp.where(jnp.abs(q) < TOL, jnp.asarray(HUGE_N, jnp.float64), y / q)
        s0_add = old_s[m - 1] + gamma * (t_add - old_s[m - 1])
        s_add = s_add.at[0].set(s0_add)
        if m > 1:
            s_add = s_add.at[1:m].set(old_s[0:m - 1])

    # Multiplicative-error update path
    q_safe = jnp.where(jnp.abs(q) < TOL, jnp.asarray(HUGE_N, jnp.float64), q)
    e_mul = p / q_safe - 1.0
    l_mul = q * (1.0 + alpha * e_mul)
    b_mul = b
    if trend != Component.Nothing:
        if trend == Component.Additive:
            b_mul = phi_b + beta * q * e_mul
        else:
            b_mul = phi_b * (1.0 + beta * e_mul)
    s_mul = s
    if season != Component.Nothing:
        if season == Component.Additive:
            s0_mul = old_s[m - 1] + gamma * q * e_mul
        else:
            s0_mul = old_s[m - 1] * (1.0 + gamma * e_mul)
        s_mul = s_mul.at[0].set(s0_mul)
        if m > 1:
            s_mul = s_mul.at[1:m].set(old_s[0:m - 1])

    add_flag = error == _COMP_ADD
    l = lax.select(add_flag, l_add, l_mul)
    b = lax.select(add_flag, b_add, b_mul)
    s = lax.select(add_flag, s_add, s_mul)

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
    steps = jnp.arange(f.shape[0], dtype=jnp.int32)
    active = steps < h
    k = steps + 1
    phi_is_one = jnp.abs(phi - 1.0) < TOL
    # phistar_k = sum_{i=1..k} phi^i
    phistar = jnp.where(
        phi_is_one,
        k.astype(jnp.float64),
        phi * (1.0 - phi ** k.astype(jnp.float64)) / (1.0 - phi),
    )

    if trend == Component.Nothing:
        base = jnp.full((f.shape[0],), l, dtype=jnp.float64)
    elif trend == Component.Additive:
        base = l + phistar * b
    else:
        base = jnp.where(
            b < 0.0,
            jnp.full((f.shape[0],), jnp.nan, dtype=jnp.float64),
            l * (b ** phistar),
        )

    if season != Component.Nothing:
        j_idx = (m - 1 - steps) % m
        seas = s[j_idx]
        if season == Component.Additive:
            base = base + seas
        else:
            base = base * seas

    return jnp.where(active, base, f)


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
    e_out : jnp.ndarray
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

    err_val = int(error.value)
    has_trend = trend != Component.Nothing
    has_season = season != Component.Nothing
    n_states = m_eff * int(has_season) + int(has_trend) + 1

    # Unpack initial states from x
    l, b, s_vec = _unpack_state(
        x[:n_states], has_trend=has_trend, has_season=has_season, m=m_eff
    )

    # Work buffers
    x_local = x
    a_local = jnp.zeros_like(a_mse)
    denom = jnp.zeros(30, dtype=jnp.float64)
    lik = jnp.array(0.0, jnp.float64)
    lik2 = jnp.array(0.0, jnp.float64)

    # f-buffer of fixed max size 30; we only use first n_mse_eff entries
    f_buf = jnp.zeros(30, dtype=jnp.float64)
    # Maintain a compact (n+1, n_states) view for cheaper per-step updates
    x_states = x_local[: (n + 1) * n_states].reshape((n + 1, n_states))

    def step(carry: tuple[Any, ...], y_i: jnp.float64) -> tuple[tuple[Any, ...], jnp.float64]:
        """Advance the full-history rollout by one observation."""
        (i, x_states, a_local, denom, l, b, s_vec, lik, lik2, f_buf) = carry

        old_l = l
        if has_trend:
            old_b = b
        else:
            old_b = jnp.asarray(0.0, jnp.float64)
        old_s = s_vec

        # forecasts up to n_mse_eff
        f_buf = forecast(
            f_buf, old_l, old_b, old_s, m_eff, trend, season, phi, jnp.int64(n_mse_eff)
        )

        # residual for this step
        f0 = f_buf[0]
        ei_add = y_i - f0
        f0_denom = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
        ei_mul = (y_i - f0) / f0_denom
        ei = jnp.where(error == Component.Additive, ei_add, ei_mul)

        # Vectorized rolling MSE update over horizons [0, n_mse_eff)
        if n_mse_eff > 0:
            js = jnp.arange(n_mse_eff, dtype=jnp.int32)
            ij = i + js
            cond = ij < n
            ij_safe = jnp.minimum(ij, n - 1)
            y_fut = y[ij_safe]
            tmp = jnp.where(cond, y_fut - f_buf[:n_mse_eff], 0.0)

            denom_slice = denom[:n_mse_eff]
            a_slice = a_local[:n_mse_eff]
            denom_new = denom_slice + cond.astype(jnp.float64)
            a_num = a_slice * (denom_slice - 1.0) + tmp * tmp
            a_new = jnp.where(denom_new > 0.0, a_num / denom_new, a_slice)

            denom = denom.at[:n_mse_eff].set(denom_new)
            a_local = a_local.at[:n_mse_eff].set(a_new)

        # state update with current observation
        l, b, s_vec = update(
            s_vec, l, b, old_l, old_b, old_s, m_eff,
            trend, season, err_val, alpha, beta, gamma, phi, y_i
        )

        # store back states for time i+1
        row = _pack_state_row(
            l, b, s_vec, has_trend=has_trend, has_season=has_season, m=m_eff
        )
        x_states = x_states.at[i + 1].set(row)

        # accumulate likelihood bits
        lik = lik + ei * ei
        val = jnp.abs(f0)
        log_arg = jnp.where(val > 0.0, val, val + 1e-8)
        lik2 = lik2 + jnp.log(log_arg)

        i = i + 1
        return (i, x_states, a_local, denom, l, b, s_vec, lik, lik2, f_buf), ei

    init_carry = (jnp.array(0, jnp.int32), x_states, a_local, denom, l, b, s_vec, lik, lik2, f_buf)
    (i_out, x_states, a_local, denom, l, b, s_vec, lik, lik2, f_buf), e_out = lax.scan(
        step, init_carry, y
    )
    x_local = x_local.at[: (n + 1) * n_states].set(x_states.reshape(-1))

    n_f64 = jnp.asarray(n, dtype=jnp.float64)
    sse = jnp.where(lik > 0.0, lik, lik + 1e-8)
    sigma2 = sse / n_f64
    base = n_f64 * (jnp.log(2.0 * jnp.pi) + 1.0 + jnp.log(sigma2))
    lik = jnp.where(error == Component.Multiplicative, base + 2.0 * lik2, base)

    return x_local, e_out, a_local, lik


@partial(jax.jit, static_argnames=("error", "trend", "season", "n_mse", "m", "clip_multiplicative_errors", "fcst_h"))
def _calc_roll_nohist(
    state0: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    n_obs: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: jnp.float64,
    beta: jnp.float64,
    gamma: jnp.float64,
    phi: jnp.float64,
    m: int,
    clip_multiplicative_errors: bool = True,
    fcst_h: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.float64]:
    """Rollout computing residuals and objective *without* storing state history.

    This mirrors :func:`_calc_roll` but skips all state-history writes,
    making it cheaper for model-selection passes where only the scalar
    objective is needed.

    Parameters
    ----------
    state0 : jnp.ndarray
        Initial state vector (level [+ trend] [+ m seasonal entries]).
    e : jnp.ndarray
        Pre-allocated residual buffer (length ``n``).
    a_mse : jnp.ndarray
        Rolling MSE buffer (length ≥ ``min(n_mse, 30)``).
    n_mse : int
        Maximum AMSE horizon (capped at 30).
    y : jnp.ndarray
        Observations (length ``n``).
    n_obs : jnp.ndarray
        Number of *active* observations (``≤ n``; allows padded series).
    error, trend, season : Component
        Model structure flags.
    alpha, beta, gamma, phi : jnp.float64
        Smoothing parameters.
    m : int
        Seasonal period.
    clip_multiplicative_errors : bool
        Clamp multiplicative residuals to ``[-2, 2]``.
    fcst_h : int
        Per-step forecast horizon used for AMSE (``1`` for pure likelihood).

    Returns
    -------
    e_out : jnp.ndarray
        Residuals for each time step.
    a_local : jnp.ndarray
        Rolling horizon MSE estimates.
    lik : jnp.float64
        Likelihood-style objective value.
    """
    n = y.shape[0]
    n_eff = jnp.minimum(n_obs, n)
    n_s = max(m, 24)
    m_eff = max(m, 1)
    n_mse_eff = min(n_mse, 30)

    err_val = int(error.value)
    has_trend = trend != Component.Nothing
    has_season = season != Component.Nothing
    n_states = m_eff * int(has_season) + int(has_trend) + 1

    # Unpack initial states
    l, b, s_vec = _unpack_state(
        state0[:n_states], has_trend=has_trend, has_season=has_season, m=m_eff
    )

    # Work buffers
    a_local = jnp.zeros_like(a_mse)
    denom = jnp.zeros(30, dtype=jnp.float64)
    lik = jnp.array(0.0, jnp.float64)
    lik2 = jnp.array(0.0, jnp.float64)
    f_buf = jnp.zeros(30, dtype=jnp.float64)

    def step(carry: tuple[Any, ...], y_i: jnp.float64) -> tuple[tuple[Any, ...], jnp.float64]:
        """Advance the no-history rollout by one observation."""
        (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf) = carry
        active = i < n_eff

        def do_active(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], jnp.float64]:
            """Process an active observation in the padded rollout."""
            (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf, y_i) = args
            old_l = l
            if has_trend:
                old_b = b
            else:
                old_b = jnp.asarray(0.0, jnp.float64)
            old_s = s_vec

            # forecasts up to requested horizon
            f_buf = forecast(
                f_buf, old_l, old_b, old_s, m_eff, trend, season, phi, jnp.int64(fcst_h)
            )

            # residual for this step
            f0_raw = f_buf[0]
            f0 = jnp.where(
                clip_multiplicative_errors and (error == Component.Multiplicative),
                jnp.maximum(f0_raw, 1e-6),
                f0_raw,
            )
            ei_add = y_i - f0
            f0_denom = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
            ei_mul = (y_i - f0) / f0_denom
            ei_mul = jnp.where(
                clip_multiplicative_errors,
                jnp.clip(ei_mul, -2.0, 2.0),
                ei_mul,
            )
            ei = jnp.where(error == Component.Additive, ei_add, ei_mul)

            # Vectorized rolling MSE update over horizons [0, n_mse_eff)
            if n_mse_eff > 0:
                js = jnp.arange(n_mse_eff, dtype=jnp.int32)
                ij = i + js
                cond = ij < n_eff
                ij_safe = jnp.minimum(ij, n - 1)
                y_fut = y[ij_safe]
                tmp = jnp.where(cond, y_fut - f_buf[:n_mse_eff], 0.0)

                denom_slice = denom[:n_mse_eff]
                a_slice = a_local[:n_mse_eff]
                denom_new = denom_slice + cond.astype(jnp.float64)
                a_num = a_slice * (denom_slice - 1.0) + tmp * tmp
                a_new = jnp.where(denom_new > 0.0, a_num / denom_new, a_slice)

                denom = denom.at[:n_mse_eff].set(denom_new)
                a_local = a_local.at[:n_mse_eff].set(a_new)

            # state update with current observation
            l, b, s_vec = update(
                s_vec, l, b, old_l, old_b, old_s, m_eff,
                trend, season, err_val, alpha, beta, gamma, phi, y_i
            )

            # accumulate likelihood bits
            lik = lik + ei * ei
            val = jnp.abs(f0)
            log_arg = jnp.where(val > 0.0, val, val + 1e-8)
            lik2 = lik2 + jnp.log(log_arg)

            i = i + 1
            return (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf), ei

        def do_inactive(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], jnp.float64]:
            """Skip padded observations once the active series is exhausted."""
            (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf, _) = args
            i = i + 1
            return (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf), jnp.asarray(0.0, dtype=jnp.float64)

        carry_out, e_out = lax.cond(
            active,
            do_active,
            do_inactive,
            (i, a_local, denom, l, b, s_vec, lik, lik2, f_buf, y_i),
        )
        return carry_out, e_out

    init_carry = (jnp.array(0, jnp.int32), a_local, denom, l, b, s_vec, lik, lik2, f_buf)
    (i_out, a_local, denom, l, b, s_vec, lik, lik2, f_buf), e_out = lax.scan(
        step, init_carry, y
    )

    n_f64 = jnp.asarray(n_eff, dtype=jnp.float64)
    sse = jnp.where(lik > 0.0, lik, lik + 1e-8)
    sigma2 = sse / n_f64
    base = n_f64 * (jnp.log(2.0 * jnp.pi) + 1.0 + jnp.log(sigma2))
    lik = jnp.where(error == Component.Multiplicative, base + 2.0 * lik2, base)

    return e_out, a_local, lik


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


def _transform_smoothing_params(
    p: jnp.ndarray,
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
    pure_sigmoid: bool = False,
) -> Tuple[jnp.float64, jnp.float64, jnp.float64, jnp.float64]:
    """Map an unconstrained parameter vector to valid (α, β, γ, φ) via sigmoid.

    Two modes are supported:

    * **Legacy mode** (``pure_sigmoid=False``): scaled sigmoid with safety
      clipping to ``[eps, 1-eps]`` and additional ``[PHI_LOWER, PHI_UPPER]``
      clipping for φ.
    * **Pure-sigmoid mode** (``pure_sigmoid=True``): each parameter is
      independently mapped through ``sigmoid → scale → shift``, producing
      cleaner gradients.

    Parameters
    ----------
    p : jnp.ndarray
        Unconstrained optimiser variables (length depends on which params
        are free).
    opt_alpha, opt_beta, opt_gamma, opt_phi : bool
        Which smoothing parameters are being optimised.
    alpha, beta, gamma, phi : float
        Current / default values for fixed parameters.
    lower, upper : jnp.ndarray
        Box-constraint bounds (length 4).
    pure_sigmoid : bool
        Select the parameterisation mode (see above).

    Returns
    -------
    tuple[jnp.float64, jnp.float64, jnp.float64, jnp.float64]
        ``(alpha, beta, gamma, phi)`` in the constrained domain.
    """
    idx = 0
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    # --- Pure-sigmoid mode: cleaner independent mapping per param ---
    if pure_sigmoid:
        if opt_alpha:
            alpha = jax.nn.sigmoid(p[idx])
            alpha = EPS_PURE + (1.0 - 2.0 * EPS_PURE) * alpha
            idx += 1
        if opt_beta:
            beta = jax.nn.sigmoid(p[idx])
            beta = alpha * beta
            idx += 1
        if opt_gamma:
            gamma = jax.nn.sigmoid(p[idx])
            gamma = (1.0 - alpha) * gamma
            idx += 1
        if opt_phi:
            pphi = jax.nn.sigmoid(p[idx])
            pphi = EPS_PURE + (1.0 - 2.0 * EPS_PURE) * pphi
            phi = lower[3] + (upper[3] - lower[3]) * pphi
        return (
            jnp.asarray(alpha, jnp.float64),
            jnp.asarray(beta, jnp.float64),
            jnp.asarray(gamma, jnp.float64),
            jnp.asarray(phi, jnp.float64),
        )

    # --- Legacy sigmoid mode: scaled sigmoid + safety clipping ---
    if opt_alpha:
        a = jax.nn.sigmoid(p[idx] * 0.1)
        alpha = lower[0] + (upper[0] - lower[0]) * a
        idx += 1
    if opt_beta:
        b = jax.nn.sigmoid(p[idx])
        beta = alpha * b
        idx += 1
    if opt_gamma:
        g = jax.nn.sigmoid(p[idx])
        gamma = (1.0 - alpha) * g
        idx += 1
    if opt_phi:
        pphi = jax.nn.sigmoid(p[idx])
        phi = lower[3] + (upper[3] - lower[3]) * pphi
    eps = 1e-4
    alpha = jnp.clip(alpha, eps, 1.0 - eps)
    beta = jnp.clip(beta, eps, 1.0 - eps)
    gamma = jnp.clip(gamma, eps, 1.0 - eps)
    phi = jnp.clip(phi, PHI_LOWER, PHI_UPPER)
    return (
        jnp.asarray(alpha, jnp.float64),
        jnp.asarray(beta, jnp.float64),
        jnp.asarray(gamma, jnp.float64),
        jnp.asarray(phi, jnp.float64),
    )


@partial(jax.jit, static_argnames=("error", "trend", "season", "opt_crit", "n_mse", "m", "opt_alpha", "opt_beta", "opt_gamma", "opt_phi", "clip_multiplicative_errors", "pure_sigmoid", "opt_init_state", "n_state"))
def _objective_smoothing_only(
    p: jnp.ndarray,
    y: jnp.ndarray,
    init_state: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    opt_crit: Criterion,
    n_mse: int,
    m: int,
    n_obs: int,
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
    clip_multiplicative_errors: bool = True,
    pure_sigmoid: bool = False,
    opt_init_state: bool = False,
    n_state: int = 0,
) -> jnp.float64:
    """Innermost objective: transform params → run no-history rollout → scalar loss.

    This is the function that ``jax.value_and_grad`` differentiates through.
    It:

    1. Transforms unconstrained ``p`` → constrained (α, β, γ, φ) via
       :func:`_transform_smoothing_params`.
    2. Optionally splits initial-state parameters from the tail of ``p``.
    3. Runs :func:`_calc_roll_nohist` to obtain residuals, AMSE, and
       likelihood in a single forward pass (no state-history storage).
    4. Returns the scalar objective selected by ``opt_crit`` (likelihood,
       MSE, AMSE, σ², or MAE).

    Parameters
    ----------
    p : jnp.ndarray
        Unconstrained parameter vector (free smoothing params, optionally
        followed by initial-state params).
    y : jnp.ndarray
        Observed time series.
    init_state : jnp.ndarray
        Initial ETS state (may be overridden when ``opt_init_state`` is True).
    error, trend, season : Component
        Model structure flags (static for JIT).
    opt_crit : Criterion
        Which loss to return.
    n_mse, m, n_obs : int
        AMSE horizon, seasonal period, and active observation count.
    opt_alpha … opt_phi : bool
        Which parameters are optimised.
    alpha … phi : float
        Default / fixed values for non-optimised parameters.
    lower, upper : jnp.ndarray
        Box-constraint bounds.
    clip_multiplicative_errors : bool
        Clamp multiplicative residuals to ``[-2, 2]``.
    pure_sigmoid : bool
        Parameterisation mode.
    opt_init_state : bool
        Whether the tail of ``p`` holds initial-state params.
    n_state : int
        Count of initial-state params appended to ``p``.

    Returns
    -------
    jnp.float64
        Scalar loss value.
    """
    # Optionally split off initial-state params from the tail of p.
    if opt_init_state and n_state > 0:
        p_smooth = p[:-n_state]
        init_state = p[-n_state:]
    else:
        p_smooth = p
    alpha, beta, gamma, phi = _transform_smoothing_params(
        p_smooth, opt_alpha, opt_beta, opt_gamma, opt_phi,
        alpha, beta, gamma, phi, lower, upper, pure_sigmoid
    )

    # Run the lightweight (no state-history) rollout to get residuals + likelihood.
    e = jnp.zeros_like(y, dtype=jnp.float64)
    a_mse = jnp.zeros((n_mse,), dtype=jnp.float64)
    n_obs_arr = jnp.asarray(n_obs, dtype=jnp.int32)
    e, a_mse, lik = _calc_roll_nohist(
        init_state,
        e,
        a_mse,
        n_mse,
        y,
        n_obs_arr,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        m,
        clip_multiplicative_errors=clip_multiplicative_errors,
        fcst_h=1 if opt_crit == Criterion.Likelihood else n_mse,
    )

    # Select the appropriate scalar objective from the rollout outputs.
    is_lik = opt_crit == Criterion.Likelihood
    is_mse = opt_crit == Criterion.MSE
    is_amse = opt_crit == Criterion.AMSE
    is_sigma = opt_crit == Criterion.Sigma

    objective = jnp.where(
        is_lik,
        lik,
        jnp.where(
            is_mse,
            a_mse[0],
            jnp.where(
                is_amse,
                jnp.mean(a_mse),
                jnp.where(is_sigma, jnp.mean(e * e), jnp.mean(jnp.abs(e))),
            ),
        ),
    )

    return objective




_LBFGS_STEPS = 30
"""Fixed L-BFGS refinement budget (quasi-Newton steps after the Adam warm-up)."""


@lru_cache(maxsize=256)
def _get_optimizer_runner(
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
    clip_multiplicative_errors: bool,
    pure_sigmoid: bool,
    opt_init_state: bool,
    n_state: int,
    n_obs: int,
    adam_steps: int,
    adam_lr: float,
) -> Callable[..., Tuple[jnp.ndarray, jnp.ndarray]]:
    """Build (and cache) the jitted two-phase scan optimizer for one static config.

    All arguments are *config/shape-static* (Component enums, ints, bools,
    the learning rate). Caching on them keeps the returned jitted callable's
    identity stable across calls, so XLA compiles once per (structure, shape)
    instead of re-tracing freshly-created closures on every fit.

    The returned callable takes only data-level arrays
    ``(x0, y, init_state, alpha, beta, gamma, phi, lower, upper)`` and returns
    ``(best_params, best_loss)`` as traced values — the whole optimizer is a
    pair of ``lax.scan`` loops (Adam warm-up, then L-BFGS refinement) with
    best-iterate tracking in the carry, so it is natively traceable under
    ``jax.vmap`` (no host loop, no ``device_get``, no try/except).
    """
    core_obj = _get_objective_closure(
        error, trend, season, opt_crit, n_mse, m,
        opt_alpha, opt_beta, opt_gamma, opt_phi,
        clip_multiplicative_errors, pure_sigmoid, opt_init_state, n_state,
    )

    def _run(
        x0: jnp.ndarray,
        y: jnp.ndarray,
        init_state: jnp.ndarray,
        alpha: jnp.float64,
        beta: jnp.float64,
        gamma: jnp.float64,
        phi: jnp.float64,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        def _obj(p: jnp.ndarray) -> jnp.float64:
            return core_obj(p, y, init_state, n_obs, alpha, beta, gamma, phi, lower, upper)

        vg_fn = jax.value_and_grad(_obj)

        # Both phases share one step shape: evaluate at p, update, and track
        # the best *evaluated* point (never the post-update point, whose loss
        # is unknown — storing new_p with p's loss mislabels the optimum).
        # A NaN/Inf loss can never become the best, so a diverging line
        # search or exploding rollout degrades gracefully to the best finite
        # iterate instead of needing a try/except escape hatch.
        def _tracked_step(opt_update):
            def step(carry, _):
                p, opt_state, best_p, best_loss = carry
                loss, grads = vg_fn(p)
                grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
                updates, opt_state = opt_update(grads, opt_state, p, loss)
                new_p = optax.apply_updates(p, updates)
                improved = jnp.isfinite(loss) & (loss < best_loss)
                best_p = jnp.where(improved, p, best_p)
                best_loss = jnp.where(improved, loss, best_loss)
                return (new_p, opt_state, best_p, best_loss), None
            return step

        inf = jnp.asarray(jnp.inf, dtype=jnp.float64)

        # ── Phase 1: Adam warm-up ─────────────────────────────────────
        adam = optax.adam(adam_lr)

        def _adam_update(grads, state, p, loss):
            return adam.update(grads, state, p)

        if adam_steps > 0:
            (_, _, adam_p, adam_loss), _ = lax.scan(
                _tracked_step(_adam_update),
                (x0, adam.init(x0), x0, inf),
                None,
                length=adam_steps,
            )
        else:
            adam_p, adam_loss = x0, inf

        # ── Phase 2: L-BFGS refinement, seeded from Adam's best point ─
        lbfgs = optax.lbfgs(
            memory_size=8,
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=15,
                initial_guess_strategy="one",
            ),
        )

        def _lbfgs_update(grads, state, p, loss):
            return lbfgs.update(grads, state, p, value=loss, grad=grads, value_fn=_obj)

        (_, _, best_p, best_loss), _ = lax.scan(
            _tracked_step(_lbfgs_update),
            (adam_p, lbfgs.init(adam_p), adam_p, adam_loss),
            None,
            length=_LBFGS_STEPS,
        )
        return best_p, best_loss

    return jax.jit(_run)


def optimize_bfgs_smoothing(
    x0: jnp.ndarray,
    y: jnp.ndarray,
    init_state: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    opt_crit: Criterion,
    n_mse: int,
    m: int,
    n_obs: int,
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
    steps: int,
    lr: float,
    clip_norm: float,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    adaptive_tol: bool = True,
    is_final_model: bool = False,
    clip_multiplicative_errors: bool = True,
    pure_sigmoid: bool = False,
    opt_init_state: bool = False,
    n_state: int = 0,
) -> OptimResult:
    """Two-phase optax optimiser: Adam warm-up → L-BFGS refinement.

    Finds the unconstrained parameter vector that minimises the ETS objective
    selected by *opt_crit*.  The search proceeds in two stages:

    1. **Adam warm-up** — a short burst of momentum-based first-order steps
       (15–30 iterations depending on series length) that moves ``x0`` into
       a reasonable basin.
    2. **L-BFGS refinement** — 30 quasi-Newton steps.

    Both phases run as ``lax.scan`` loops with best-iterate tracking in the
    carry, so the whole optimizer is natively traceable under ``jax.vmap``
    (the conformity_scores CV path). The best parameter vector seen across
    *both* phases is returned.

    Parameters
    ----------
    x0 : jnp.ndarray
        Initial unconstrained parameter vector.
    y : jnp.ndarray
        Observed time series (float64).
    init_state : jnp.ndarray
        Initial ETS state (level, trend, seasonal).
    error, trend, season : Component
        Model structure flags.
    opt_crit : Criterion
        Which loss to minimise (likelihood, MSE, AMSE, σ², MAE).
    n_mse : int
        Horizon cap for rolling MSE (≤ 30).
    m : int
        Seasonal period.
    n_obs : int
        Active observation count (may be ``< len(y)`` if padded).
    opt_alpha, opt_beta, opt_gamma, opt_phi : bool
        ``True`` → optimise the corresponding smoothing parameter;
        ``False`` → keep it fixed at the supplied value.
    alpha, beta, gamma, phi : float
        Current / default values for the four smoothing parameters.
    lower, upper : jnp.ndarray
        Box-constraint bounds for smoothing parameters (length 4).
    steps : int
        Total iteration budget (Adam uses a fraction of this).
    lr : float
        Base learning rate for the Adam warm-up phase.
    clip_norm : float
        *(Unused — kept for caller API compatibility.)*
    early_stop_patience : int
        *(Unused — ``lax.scan`` runs a fixed number of iterations; host-side
        early stopping would require concretizing the traced loss.)*
    early_stop_min_delta : float
        *(Unused — kept for API compatibility.)*
    adaptive_tol : bool
        *(Unused — kept for API compatibility.)*
    is_final_model : bool
        *(Unused — kept for API compatibility.)*
    clip_multiplicative_errors : bool
        Clamp multiplicative residuals to ``[-2, 2]``.
    pure_sigmoid : bool
        Use the cleaner independent sigmoid parameterisation.
    opt_init_state : bool
        If ``True``, the tail of ``x0`` holds optimisable initial states.
    n_state : int
        Number of initial-state parameters appended to ``x0``.

    Returns
    -------
    OptimResult
        Named tuple with fields ``success``, ``status``, ``message``,
        ``x`` (best params), ``fun`` (best loss), ``nit``, ``nfev``.
        ``x``/``fun``/``success`` are jnp values (traced under vmap).
    """
    # ── Input normalisation ───────────────────────────────────────────
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    init_state = jnp.asarray(init_state, dtype=jnp.float64)
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    # Adam budget: static, derived from *shape* (n_obs) and config only.
    # Step counts are inversely scaled with series length because each grad
    # eval is O(n) in the ETS rollout.
    if int(n_obs) <= 200:
        adam_steps = min(int(steps), 30)
        adam_lr = float(lr) if lr is not None else 5e-2
    elif int(n_obs) <= 1000:
        adam_steps = min(int(steps), 25)
        adam_lr = float(lr) if lr is not None else 3e-2
    else:
        adam_steps = min(int(steps), 15)
        adam_lr = float(lr) if lr is not None else 2e-2

    runner = _get_optimizer_runner(
        error, trend, season, opt_crit, int(n_mse), int(m),
        bool(opt_alpha), bool(opt_beta), bool(opt_gamma), bool(opt_phi),
        bool(clip_multiplicative_errors), bool(pure_sigmoid),
        bool(opt_init_state), int(n_state),
        int(n_obs), adam_steps, adam_lr,
    )
    best_params, best_loss = runner(x0, y, init_state, alpha, beta, gamma, phi, lower, upper)

    nit = adam_steps + _LBFGS_STEPS
    success = jnp.isfinite(best_params).all() & jnp.isfinite(best_loss)
    return OptimResult(
        success=success,
        status=0,
        message="ok",
        x=best_params,
        fun=best_loss,
        nit=nit,
        nfev=nit,
    )


# Public API surface
__all__ = [
    "HUGE_N", "NA", "TOL",                    # constants
    "Component", "Criterion", "OptimResult",   # enums / result type
    "update", "forecast",                      # ETS recursion primitives
    "calc_full", "calc",                       # rollout helpers
    "optimize_bfgs_smoothing",                 # optimizer entry-point
]
