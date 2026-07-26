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

  - *Stage 1*: Adam warm-up (first-order, ~15–30 steps in a Python loop).
  - *Stage 2*: L-BFGS refinement (second-order, 30 steps via ``lax.scan``
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

from chronax.utils import adaptive_iterate  # convergence-based step runner


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
    last_state : jnp.ndarray
        Final packed state row (length ``n_states``) — the one row the
        post-fit consumers actually read.
    """
    n = y.shape[0]
    n_eff = jnp.minimum(n_obs, n)
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

    # Final state snapshot: the post-fit rollout consumers (pegelsresid_C ->
    # forecast seeding / native intervals / the stacked CV refit) only ever read the
    # LAST state row, so materializing an (n+1)-row history via a per-step scatter
    # would be dead weight. Packing the final carry here costs one row.
    last_state = _pack_state_row(
        l, b, s_vec, has_trend=has_trend, has_season=has_season, m=m_eff
    )

    return e_out, a_local, lik, last_state


def _make_step_core(
    error: Component,
    trend: Component,
    season: Component,
    m_eff: int,
    clip_multiplicative_errors: bool,
) -> Callable[..., tuple]:
    """Scalar per-observation core of the likelihood rollout.

    Returns ``core(L, B, Sm, y_i, alpha, beta, gamma, phi)`` computing one
    observation's forecast/error terms plus the state update — the exact
    expressions of ``_calc_roll_lik``'s step, but taking the ONLY seasonal
    entry the math reads (``Sm = old_s[m-1]``) as a scalar. Valid because the
    ETS recurrence's nonlinear math never reads ``old_s[0:m-1]``: ``update()``
    consumes ``old_s[m-1]`` for ``p``/``s0'`` and entries ``0..m-2`` only via the
    pure shift copy; the one-step forecast reads ``s[m-1]``. The state update is
    delegated to the SHARED :func:`update` on a synthesized seasonal vector
    ``zeros(m).at[m-1].set(Sm)`` — when season is active, ``update`` overwrites
    all ``m`` output entries, so the dummy zeros cannot leak; ``l'``, ``b'``,
    ``s0'`` are bit-identical to the full-vector call.

    Used by BOTH the custom-adjoint forward scan and its backward's local
    ``jax.vjp`` (``_get_roll_scan_custom``), so forward/backward math cannot
    drift. Outputs: ``(l', b', s0', ei)`` for additive error, plus ``lg`` (the
    log|f0| likelihood term) for multiplicative error.
    """
    has_season = season != Component.Nothing
    err_val = int(error.value)
    is_mul_error = error == Component.Multiplicative

    def core(
        L: jnp.float64,
        B: jnp.float64,
        Sm: jnp.float64,
        y_i: jnp.float64,
        alpha: jnp.float64,
        beta: jnp.float64,
        gamma: jnp.float64,
        phi: jnp.float64,
    ) -> tuple:
        one = jnp.asarray(1.0, jnp.float64)
        # forecast()'s lane 0 (k=1), expression-for-expression (see
        # _calc_roll_lik._forecast1, kept verbatim for the unrouted branch).
        if trend == Component.Nothing:
            base = L
        else:
            phi_is_one = jnp.abs(phi - 1.0) < TOL
            phistar = jnp.where(phi_is_one, one, phi * (1.0 - phi ** one) / (1.0 - phi))
            if trend == Component.Additive:
                base = L + phistar * B
            else:
                base = jnp.where(
                    B < 0.0,
                    jnp.asarray(jnp.nan, jnp.float64),
                    L * (B ** phistar),
                )
        if has_season:
            base = base + Sm if season == Component.Additive else base * Sm

        f0_raw = base
        if clip_multiplicative_errors and is_mul_error:
            f0 = jnp.maximum(f0_raw, 1e-6)
        else:
            f0 = f0_raw
        if error == Component.Additive:
            ei = y_i - f0
        else:
            f0_denom = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
            ei = (y_i - f0) / f0_denom
            if clip_multiplicative_errors:
                ei = jnp.clip(ei, -2.0, 2.0)

        if has_season:
            s_dummy = jnp.zeros((m_eff,), dtype=jnp.float64).at[m_eff - 1].set(Sm)
        else:
            # season-N: update() never reads old_s and applies no seasonal
            # writes — skip the scatter (it was the only one in the m=1
            # backward body and a fusion hazard there).
            s_dummy = jnp.zeros((m_eff,), dtype=jnp.float64)
        l_new, b_new, s_new = update(
            s_dummy, L, B, L, B, s_dummy, m_eff,
            trend, season, err_val, alpha, beta, gamma, phi, y_i
        )
        s0_new = s_new[0] if has_season else jnp.asarray(0.0, jnp.float64)

        if is_mul_error:
            val = jnp.abs(f0)
            log_arg = jnp.where(val > 0.0, val, val + 1e-8)
            lg = jnp.log(log_arg)
            return l_new, b_new, s0_new, ei, lg
        return l_new, b_new, s0_new, ei

    return core


def _make_step_partials(
    error: Component,
    trend: Component,
    season: Component,
    clip_multiplicative_errors: bool,
) -> Callable[..., tuple]:
    """Analytic per-step VJP of :func:`_make_step_core`'s math (v2 adjoint).

    Returns ``partials(L, B, Sm, y_i, alpha, beta, gamma, phi, ct_l, ct_b,
    ct_s0, ct_e, ct_lg) -> (L̄, B̄, S̄m, ȳ, ᾱ, β̄, γ̄, φ̄)`` — the hand-written
    transpose of one observation's forecast/error/update chain, live path
    only (the ``jax.vjp``-of-core v1 measured ~1.0x the checkpointed AD it
    replaced on m=1 M-error cells: same recompute + dead error-path
    transposes + select/scatter machinery; this straight-line form is what
    fuses). For additive error ``ct_e`` is the CONSTANT lik-cotangent and the
    ``2*ei`` factor is applied internally; for multiplicative error ``ct_e``
    is the per-step ``eis`` cotangent and ``ct_lg`` the ``lgs`` one.

    Derivative discipline (AD-parity, NaN semantics included): every forward
    ``where(c, f, g)`` transposes as gate-the-cotangent (``where(c, ct, 0)``
    into ``f``, ``where(c, 0, ct)`` into ``g``) times the RAW un-protected
    partial expressions; ``maximum``/``clip`` boundaries use the 3-way
    select with 0.5 at exact ties (JAX's convention). Verified against AD of
    the identical forward by ``autoets_adjoint_oracle.py`` (all routed forms
    x param points x padded windows, NaN-aware).
    """
    has_season = season != Component.Nothing
    is_mul_error = error == Component.Multiplicative

    def partials(L, B, Sm, y_i, alpha, beta, gamma, phi,
                 ct_l, ct_b, ct_s0, ct_e, ct_lg):
        one = jnp.asarray(1.0, jnp.float64)
        zero = jnp.asarray(0.0, jnp.float64)
        ct_L = zero; ct_B = zero; ct_Sm = zero; ct_y = zero
        ct_al = zero; ct_be = zero; ct_ga = zero; ct_ph = zero

        # ---- forward intermediates (live path only, verbatim expressions) ----
        if trend == Component.Nothing:
            base_t = L
        else:
            cond1 = jnp.abs(phi - 1.0) < TOL
            phistar = jnp.where(cond1, one, phi * (1.0 - phi ** one) / (1.0 - phi))
            # Raw AD-form derivative of the phistar quotient EXPRESSION
            # (NOT the simplified mathematical value 1): NaN at the exact
            # 0/0 point phi=1, matching what AD produces through the
            # gated-cotangent-times-raw-partial route.
            one_m_phi = 1.0 - phi
            dphistar = ((1.0 - 2.0 * phi) * one_m_phi + phi * one_m_phi) / (
                one_m_phi * one_m_phi
            )
            if trend == Component.Additive:
                base_t = L + phistar * B
            else:
                neg = B < 0.0
                pw = B ** phistar
                base_t = jnp.where(neg, jnp.asarray(jnp.nan, jnp.float64), L * pw)
        if has_season:
            f0_raw = base_t + Sm if season == Component.Additive else base_t * Sm
        else:
            f0_raw = base_t
        clip_f0 = clip_multiplicative_errors and is_mul_error
        f0 = jnp.maximum(f0_raw, 1e-6) if clip_f0 else f0_raw

        if error == Component.Additive:
            ei = y_i - f0
        else:
            d = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
            e_raw = (y_i - f0) / d
            ei = jnp.clip(e_raw, -2.0, 2.0) if clip_multiplicative_errors else e_raw

        # update() intermediates (live error path)
        if trend == Component.Nothing:
            q = L
            phi_b = zero
        elif trend == Component.Additive:
            phi_b = phi * B
            q = L + phi_b
        else:
            cond1u = jnp.abs(phi - 1.0) < TOL
            pwu = B ** phi
            phi_b = jnp.where(cond1u, B, pwu)
            q = jnp.where(cond1u, L * B, L * phi_b)
        if season == Component.Nothing:
            p = y_i
        elif season == Component.Additive:
            p = y_i - Sm
        else:
            sm_small = jnp.abs(Sm) < TOL
            p = jnp.where(sm_small, jnp.asarray(HUGE_N, jnp.float64), y_i / Sm)

        if is_mul_error:
            q_small = jnp.abs(q) < TOL
            q_safe = jnp.where(q_small, jnp.asarray(HUGE_N, jnp.float64), q)
            e_mul = p / q_safe - 1.0
        else:
            e_add = p - q
            l_new = q + alpha * e_add  # needed by the r-chain below

        # ---- transpose: error/likelihood emits -> f0 ----
        if error == Component.Additive:
            # lik contribution: ct_e is the constant lik-cotangent; d lik/d ei_i = 2 ei
            ct_ei = 2.0 * ei * ct_e
            ct_y = ct_y + ct_ei
            ct_f0 = -ct_ei
        else:
            ct_ei = ct_e
            if clip_multiplicative_errors:
                t_lo = jnp.maximum(e_raw, -2.0)
                d_lo = jnp.where(e_raw > -2.0, one, jnp.where(e_raw < -2.0, zero, 0.5 * one))
                d_hi = jnp.where(t_lo < 2.0, one, jnp.where(t_lo > 2.0, zero, 0.5 * one))
                ct_eraw = ct_ei * d_hi * d_lo
            else:
                ct_eraw = ct_ei
            ct_y = ct_y + ct_eraw / d
            # d = f0 (+TOL): derivative 1 on both where-branches
            ct_f0 = ct_eraw * (-(one / d) - (y_i - f0) / (d * d))
            # lg = log(where(|f0|>0, |f0|, |f0|+1e-8)); both branches d=1
            val = jnp.abs(f0)
            log_arg = jnp.where(val > 0.0, val, val + 1e-8)
            ct_f0 = ct_f0 + ct_lg * jnp.sign(f0) / log_arg

        if clip_f0:
            d_max = jnp.where(f0_raw > 1e-6, one, jnp.where(f0_raw < 1e-6, zero, 0.5 * one))
            ct_f0raw = ct_f0 * d_max
        else:
            ct_f0raw = ct_f0

        # ---- transpose: forecast -> (L, B, Sm, phi) ----
        if has_season:
            if season == Component.Additive:
                ct_base = ct_f0raw
                ct_Sm = ct_Sm + ct_f0raw
            else:
                ct_base = ct_f0raw * Sm
                ct_Sm = ct_Sm + ct_f0raw * base_t
        else:
            ct_base = ct_f0raw
        if trend == Component.Nothing:
            ct_L = ct_L + ct_base
        elif trend == Component.Additive:
            ct_L = ct_L + ct_base
            ct_B = ct_B + ct_base * phistar
            ct_ph = ct_ph + jnp.where(cond1, zero, ct_base * B) * dphistar
        else:
            ct_prod = jnp.where(neg, zero, ct_base)  # NaN branch is a constant
            ct_L = ct_L + ct_prod * pw
            ct_pw = ct_prod * L
            ct_B = ct_B + ct_pw * phistar * B ** (phistar - 1.0)
            ct_ph = ct_ph + jnp.where(cond1, zero, ct_pw * pw * jnp.log(B)) * dphistar

        # ---- transpose: state update -> inputs ----
        ct_q = zero
        ct_phib = zero
        if is_mul_error:
            u = 1.0 + alpha * e_mul
            ct_emul = zero
            # l' = q * u
            ct_q = ct_q + ct_l * u
            ct_al = ct_al + ct_l * q * e_mul
            ct_emul = ct_emul + ct_l * q * alpha
            # b'
            if trend == Component.Additive:
                ct_phib = ct_phib + ct_b
                ct_be = ct_be + ct_b * q * e_mul
                ct_q = ct_q + ct_b * beta * e_mul
                ct_emul = ct_emul + ct_b * beta * q
            elif trend == Component.Multiplicative:
                w = 1.0 + beta * e_mul
                ct_phib = ct_phib + ct_b * w
                ct_be = ct_be + ct_b * phi_b * e_mul
                ct_emul = ct_emul + ct_b * phi_b * beta
            else:
                ct_B = ct_B + ct_b  # b passthrough
            # s0'
            if has_season:
                if season == Component.Additive:
                    ct_Sm = ct_Sm + ct_s0
                    ct_ga = ct_ga + ct_s0 * q * e_mul
                    ct_q = ct_q + ct_s0 * gamma * e_mul
                    ct_emul = ct_emul + ct_s0 * gamma * q
                else:
                    v = 1.0 + gamma * e_mul
                    ct_Sm = ct_Sm + ct_s0 * v
                    ct_ga = ct_ga + ct_s0 * Sm * e_mul
                    ct_emul = ct_emul + ct_s0 * Sm * gamma
            # e_mul = p / q_safe - 1
            ct_p = ct_emul / q_safe
            ct_qsafe = ct_emul * (-p / (q_safe * q_safe))
            ct_q = ct_q + jnp.where(q_small, zero, ct_qsafe)
        else:
            # additive-error update; b'-chain consumes l', so total l'-cotangent
            # = incoming ct_l + the r-chain contribution (order matters).
            ct_p = zero
            if trend == Component.Nothing:
                ct_B = ct_B + ct_b
                ct_lp = ct_l
            else:
                ratio = beta / alpha
                if trend == Component.Additive:
                    r = l_new - L
                else:
                    l_small = jnp.abs(L) < TOL
                    r = jnp.where(l_small, jnp.asarray(HUGE_N, jnp.float64), l_new / L)
                ct_phib = ct_phib + ct_b * (1.0 - ratio)
                ct_be = ct_be + ct_b * (r - phi_b) / alpha
                ct_al = ct_al + ct_b * (r - phi_b) * (-beta / (alpha * alpha))
                ct_r = ct_b * ratio
                if trend == Component.Additive:
                    ct_lp = ct_l + ct_r
                    ct_L = ct_L - ct_r
                else:
                    ct_re = jnp.where(l_small, zero, ct_r)
                    ct_lp = ct_l + ct_re / L
                    ct_L = ct_L + ct_re * (-l_new / (L * L))
            # s0' = Sm + gamma * (t - Sm)
            if has_season:
                ct_Sm = ct_Sm + ct_s0 * (1.0 - gamma)
                if season == Component.Additive:
                    t_v = y_i - q
                else:
                    t_v = jnp.where(jnp.abs(q) < TOL, jnp.asarray(HUGE_N, jnp.float64), y_i / q)
                ct_ga = ct_ga + ct_s0 * (t_v - Sm)
                ct_t = ct_s0 * gamma
                if season == Component.Additive:
                    ct_y = ct_y + ct_t
                    ct_q = ct_q - ct_t
                else:
                    q_small_t = jnp.abs(q) < TOL
                    ct_te = jnp.where(q_small_t, zero, ct_t)
                    ct_y = ct_y + ct_te / q
                    ct_q = ct_q + ct_te * (-y_i / (q * q))
            # l' = q + alpha * e_add
            ct_q = ct_q + ct_lp
            ct_al = ct_al + ct_lp * e_add
            ct_eadd = ct_lp * alpha
            # e_add = p - q
            ct_p = ct_p + ct_eadd
            ct_q = ct_q - ct_eadd

        # ---- transpose: p -> (y, Sm) ----
        if season == Component.Nothing:
            ct_y = ct_y + ct_p
        elif season == Component.Additive:
            ct_y = ct_y + ct_p
            ct_Sm = ct_Sm - ct_p
        else:
            ct_pe = jnp.where(sm_small, zero, ct_p)
            ct_y = ct_y + ct_pe / Sm
            ct_Sm = ct_Sm + ct_pe * (-y_i / (Sm * Sm))

        # ---- transpose: q / phi_b -> (L, B, phi) ----
        if trend == Component.Nothing:
            ct_L = ct_L + ct_q
        elif trend == Component.Additive:
            ct_L = ct_L + ct_q
            ct_phib = ct_phib + ct_q
            ct_ph = ct_ph + ct_phib * B
            ct_B = ct_B + ct_phib * phi
        else:
            ct_LB = jnp.where(cond1u, ct_q, zero)
            ct_Lphib = jnp.where(cond1u, zero, ct_q)
            ct_L = ct_L + ct_LB * B + ct_Lphib * phi_b
            ct_B = ct_B + ct_LB * L
            ct_phib = ct_phib + ct_Lphib * L
            ct_Bdir = jnp.where(cond1u, ct_phib, zero)
            ct_pow = jnp.where(cond1u, zero, ct_phib)
            ct_B = ct_B + ct_Bdir + ct_pow * phi * B ** (phi - 1.0)
            ct_ph = ct_ph + ct_pow * pwu * jnp.log(B)

        return ct_L, ct_B, ct_Sm, ct_y, ct_al, ct_be, ct_ga, ct_ph

    return partials


@lru_cache(maxsize=512)
def _get_roll_scan_custom(
    error: Component,
    trend: Component,
    season: Component,
    m_eff: int,
    clip_multiplicative_errors: bool,
    n: int,
) -> Callable[..., Any]:
    """Custom-adjoint likelihood-rollout scan for one static ETS config.

    Returns ``roll(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi)`` — the
    scan section of :func:`_calc_roll_lik` wrapped in ``jax.custom_vjp`` with a
    hand-written reverse-scan backward. Routed for
    multiplicative-error forms (all trend x season, output ``(eis, lgs)``
    per-step arrays) and additive-error SEASONAL forms (output the in-carry
    ``lik`` scalar, preserving today's bit-exact summation order); the
    additive non-seasonal branch stays on plain AD in the caller.

    Forward: today's step math verbatim (via :func:`_make_step_core` + the
    real seasonal shift + ``jnp.where`` masking), additionally emitting the
    PRE-update states ``(old_l, old_b, old_s[m-1])`` per step — the 3-scalar
    trajectory the backward reads (no ``(n, m)`` storage; the seasonal shift
    is linear so its adjoint needs no values). ``jax.checkpoint`` is gone from
    routed branches: nothing differentiates through this forward anymore.

    Backward: ONE reverse ``lax.scan`` carrying ``(l̄, b̄, s̄, ᾱ, β̄, γ̄, φ̄)``;
    per step the local input cotangents come from ``jax.vjp`` of the shared
    core at the saved states (traced once at scan-trace time — straight-line
    primal-recompute + transpose ops in the body), and the structural adjoints
    are wired by hand: masking ``z̄_i = J^T(a-gated z̄_{i+1}) + where(a, 0,
    z̄_{i+1})`` (AD's exact select-transpose form), seasonal-shift cotangent
    left-shift with the core's Sm-cotangent entering slot ``m-1``, parameter
    accumulation in-carry, ``ȳ`` emitted as ys. The masking and shift adjoints are
    verified against AD to ~4e-15 relative on M-error forms and bitwise on A-error
    forms, including padded windows and the
    ``jit∘vmap∘while∘value_and_grad`` composition.

    ``n_eff`` rides as an f64 scalar (comparisons ``arange(n) < n_eff_f``
    inside) so the backward returns an ordinary zero cotangent for it instead
    of int/float0 plumbing. Note the production CV path feeds a STATIC n_obs
    (edge-filled windows; shape-derived int in ets_functions.py:~1135) — the
    runtime mask is kept because it is strictly more general and free.

    Cache key = config + shapes only (``n = y.shape[0]``), per the repo's
    static-args-from-config rule; the cached identity also keeps repeat traces
    from rebuilding the custom_vjp wrapper.
    """
    is_mul_error = error == Component.Multiplicative
    has_season = season != Component.Nothing
    core = _make_step_core(error, trend, season, m_eff, clip_multiplicative_errors)
    partials = _make_step_partials(error, trend, season, clip_multiplicative_errors)

    def _fwd_impl(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi):
        idx = jnp.arange(n, dtype=jnp.float64)

        def fwd_step(carry, xs):
            y_i, i_f = xs
            if is_mul_error:
                l, b, s_vec = carry
            else:
                l, b, s_vec, lik = carry
            active = i_f < n_eff_f
            L, Bv, Sm = l, b, s_vec[m_eff - 1]

            outs = core(L, Bv, Sm, y_i, alpha, beta, gamma, phi)
            if is_mul_error:
                l_new, b_new, s0_new, ei, lg = outs
            else:
                l_new, b_new, s0_new, ei = outs

            if has_season:
                s_full = s_vec.at[0].set(s0_new)
                if m_eff > 1:
                    s_full = s_full.at[1:m_eff].set(s_vec[0:m_eff - 1])
                s_next = jnp.where(active, s_full, s_vec)
            else:
                s_next = s_vec
            l_next = jnp.where(active, l_new, l)
            b_next = jnp.where(active, b_new, b)

            if is_mul_error:
                ei_out = jnp.where(active, ei, 0.0)
                lg_out = jnp.where(active, lg, 0.0)
                return (l_next, b_next, s_next), (L, Bv, Sm, ei_out, lg_out)
            lik_new = lik + ei * ei
            lik_next = jnp.where(active, lik_new, lik)
            return (l_next, b_next, s_next, lik_next), (L, Bv, Sm)

        if is_mul_error:
            init = (l0, b0, s0)
            _, (l_hist, b_hist, sm_hist, eis, lgs) = lax.scan(fwd_step, init, (y, idx))
            primal = (eis, lgs)
        else:
            init = (l0, b0, s0, jnp.asarray(0.0, dtype=jnp.float64))
            (_, _, _, lik), (l_hist, b_hist, sm_hist) = lax.scan(fwd_step, init, (y, idx))
            primal = lik
        resid = (l_hist, b_hist, sm_hist, y, n_eff_f, alpha, beta, gamma, phi)
        return primal, resid

    @jax.custom_vjp
    def roll(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi):
        return _fwd_impl(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi)[0]

    def roll_fwd(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi):
        return _fwd_impl(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi)

    def roll_bwd(resid, cot):
        l_hist, b_hist, sm_hist, y, n_eff_f, alpha, beta, gamma, phi = resid
        idx = jnp.arange(n, dtype=jnp.float64)
        zero = jnp.asarray(0.0, dtype=jnp.float64)

        if is_mul_error:
            g_eis, g_lgs = cot
            xs = (l_hist, b_hist, sm_hist, y, idx, g_eis, g_lgs)
        else:
            g_lik = cot
            xs = (l_hist, b_hist, sm_hist, y, idx)

        s_ct0 = jnp.zeros((m_eff if has_season else 1,), dtype=jnp.float64)
        init = (zero, zero, s_ct0, zero, zero, zero, zero)

        def bwd_step(carry, xs_i):
            l_ct, b_ct, s_ct, al_ct, be_ct, ga_ct, ph_ct = carry
            if is_mul_error:
                L, Bv, Sm, y_i, i_f, g_ei_i, g_lg_i = xs_i
            else:
                L, Bv, Sm, y_i, i_f = xs_i
            a = i_f < n_eff_f

            # a-gated output cotangents into the analytic step transpose (the
            # masked forward's select-transpose routes exactly these; inactive
            # steps contribute zero through the step and identity through the
            # carry).
            g_l = jnp.where(a, l_ct, zero)
            g_b = jnp.where(a, b_ct, zero)
            g_s0 = jnp.where(a, s_ct[0], zero) if has_season else zero
            if is_mul_error:
                ct_e = jnp.where(a, g_ei_i, zero)
                ct_lg = jnp.where(a, g_lg_i, zero)
            else:
                # lik = sum of active ei^2 in-carry; the where(a, lik+ei^2, lik)
                # chain is identity in lik on both branches, so the incoming
                # lik-cotangent is the CONSTANT g_lik at every step (gated;
                # the 2*ei factor is applied inside `partials`, matching AD's
                # transpose order exactly).
                ct_e = jnp.where(a, g_lik, zero)
                ct_lg = zero
            L_ct, B_ct, Sm_ct, y_ct, aal, abe, aga, aph = partials(
                L, Bv, Sm, y_i, alpha, beta, gamma, phi,
                g_l, g_b, g_s0, ct_e, ct_lg,
            )

            l_ct_new = L_ct + jnp.where(a, zero, l_ct)
            b_ct_new = B_ct + jnp.where(a, zero, b_ct)
            if has_season:
                # forward s' = [s0', S[0..m-2]] under the active mask:
                # cotangents left-shift; the tail slot receives the core's Sm
                # cotangent (which carries the p/f0/s0' chains, a-gated via
                # the input cotangents above).
                gated_next = jnp.where(a, s_ct, zero)
                shift_ct = jnp.concatenate(
                    [gated_next[1:], jnp.zeros((1,), dtype=jnp.float64)]
                )
                s_ct_new = shift_ct.at[m_eff - 1].add(Sm_ct) + jnp.where(a, zero, s_ct)
            else:
                s_ct_new = s_ct
            new_carry = (
                l_ct_new, b_ct_new, s_ct_new,
                al_ct + aal, be_ct + abe, ga_ct + aga, ph_ct + aph,
            )
            return new_carry, y_ct

        (l0_ct, b0_ct, s0_ct, al_ct, be_ct, ga_ct, ph_ct), y_bar = lax.scan(
            bwd_step, init, xs, reverse=True
        )
        return (
            l0_ct, b0_ct, s0_ct, y_bar,
            jnp.zeros_like(n_eff_f), al_ct, be_ct, ga_ct, ph_ct,
        )

    roll.defvjp(roll_fwd, roll_bwd)
    # Test hooks: the same forward WITHOUT the custom vjp (adjoint oracle
    # compares the hand backward against plain AD of the identical forward —
    # benchmarks/scripts/speed_probes/autoets_adjoint_oracle.py) and the raw
    # backward (standalone timing/decomposition probes).
    roll._fwd_impl = _fwd_impl
    roll._bwd = roll_bwd
    return roll


@partial(jax.jit, static_argnames=("error", "trend", "season", "m", "clip_multiplicative_errors"))
def _calc_roll_lik(
    state0: jnp.ndarray,
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
) -> jnp.float64:
    """Likelihood-only rollout: the optimizer-objective hot path.

    The ``Criterion.Likelihood`` slice of :func:`_calc_roll_nohist`, rebuilt
    lean because this scan body is what ``jax.value_and_grad`` differentiates
    ~40x per candidate per fit, making it the AutoETS warm bottleneck. Differences
    are shape/structure
    only, never math: a scalar one-step forecast replaces the (30,)-lane
    masked ``forecast`` buffer (the likelihood consumes only ``f[0]``); the
    AMSE machinery (``a_local``/``denom``/``f_buf`` carries and their per-step
    ``y`` gathers) is dropped — it never feeds the likelihood; the seasonal
    state rides at its true ``(m,)`` width instead of ``max(m, 24)``; and the
    padded-window mask is a ``jnp.where`` on the carry instead of a
    ``lax.cond`` whose both-branch residuals the VJP must store. Every
    surviving expression replicates the general rollout verbatim (same guard
    order); only the M-error accumulation ORDER differs (ys-emitted terms reduced by
    ``jnp.sum`` — see the in-step comment). Against ``_calc_roll_nohist`` the A-error
    objective is bit-exact bar occasional single-digit-ULP XLA fusion
    re-association, M-error values agree to ~1e-14 relative (the summation
    reassociation) and gradients to ~1e-13, so optimizer endpoints can drift at the
    ~1e-10 level without changing selection.
    """
    n = y.shape[0]
    n_eff = jnp.minimum(n_obs, n)
    m_eff = max(m, 1)

    has_trend = trend != Component.Nothing
    has_season = season != Component.Nothing
    err_val = int(error.value)
    is_mul_error = error == Component.Multiplicative

    l0 = state0[0]
    b0 = state0[1] if has_trend else jnp.asarray(0.0, dtype=jnp.float64)
    if has_season:
        start = 1 + int(has_trend)
        s0 = state0[start:start + m_eff]
    else:
        s0 = jnp.zeros((1,), dtype=jnp.float64)

    def _forecast1(l: jnp.float64, b: jnp.float64, s: jnp.ndarray) -> jnp.float64:
        # forecast()'s lane 0 (k=1), expression-for-expression: lane-wise
        # elementwise semantics make the scalar result bit-match f_buf[0].
        one = jnp.asarray(1.0, jnp.float64)
        if trend == Component.Nothing:
            base = l
        else:
            phi_is_one = jnp.abs(phi - 1.0) < TOL
            phistar = jnp.where(phi_is_one, one, phi * (1.0 - phi ** one) / (1.0 - phi))
            if trend == Component.Additive:
                base = l + phistar * b
            else:
                base = jnp.where(
                    b < 0.0,
                    jnp.asarray(jnp.nan, jnp.float64),
                    l * (b ** phistar),
                )
        if has_season:
            seas = s[m_eff - 1]  # forecast()'s j_idx = (m-1-0) % m at step 0
            base = base + seas if season == Component.Additive else base * seas
        return base

    def step(carry: tuple[Any, ...], y_i: jnp.float64) -> tuple[tuple[Any, ...], Any]:
        """Advance the likelihood rollout by one observation."""
        if is_mul_error:
            i, l, b, s_vec = carry
            lik = lik2 = None
        else:
            i, l, b, s_vec, lik, lik2 = carry
        active = i < n_eff

        old_l = l
        old_b = b if has_trend else jnp.asarray(0.0, jnp.float64)
        old_s = s_vec

        f0_raw = _forecast1(old_l, old_b, old_s)
        if clip_multiplicative_errors and is_mul_error:
            f0 = jnp.maximum(f0_raw, 1e-6)
        else:
            f0 = f0_raw
        if error == Component.Additive:
            ei = y_i - f0
        else:
            f0_denom = jnp.where(jnp.abs(f0) >= TOL, f0, f0 + TOL)
            ei = (y_i - f0) / f0_denom
            if clip_multiplicative_errors:
                ei = jnp.clip(ei, -2.0, 2.0)

        l_new, b_new, s_new = update(
            s_vec, l, b, old_l, old_b, old_s, m_eff,
            trend, season, err_val, alpha, beta, gamma, phi, y_i
        )

        l = jnp.where(active, l_new, l)
        b = jnp.where(active, b_new, b)
        s_vec = jnp.where(active, s_new, s_vec)
        if is_mul_error:
            # M-error accumulators leave the carry: chaining the div/clip/log-
            # derived running sums through the carry breaks backward-scan
            # fusion (measured 4.4x on the vg for M-trend specs); the per-step
            # terms are emitted as ys and reduced outside the loop instead.
            # Cost: summation reassociation — the M-error likelihood drifts by a few
            # ULP relative to the sequential order.
            # The A-error branch below keeps the in-carry form and stays
            # bit-exact vs the general rollout.
            val = jnp.abs(f0)
            log_arg = jnp.where(val > 0.0, val, val + 1e-8)
            ei_out = jnp.where(active, ei, 0.0)
            lg_out = jnp.where(active, jnp.log(log_arg), 0.0)
            return (i + 1, l, b, s_vec), (ei_out, lg_out)
        lik_new = lik + ei * ei
        lik = jnp.where(active, lik_new, lik)
        return (i + 1, l, b, s_vec, lik, lik2), None

    # Gradient routing (static branch): the expensive AD paths — M-error (all
    # forms) and A-error SEASONAL — go through the hand-written custom-vjp
    # adjoint (`_get_roll_scan_custom`, session-2 levers 1-2): forward math
    # verbatim (values unchanged; the sums below keep today's order), backward
    # = one lean reverse scan over the saved 3-scalar state trajectory instead
    # of AD-through-scan (which cost 10-59x a forward even checkpointed). The
    # A-error NON-seasonal step is ~linear with near-zero AD residuals and
    # stays on plain AD — its objective values are the bit-exact-prized path.
    n_eff_f = n_eff.astype(jnp.float64)
    if is_mul_error:
        roll = _get_roll_scan_custom(
            error, trend, season, m_eff, clip_multiplicative_errors, n
        )
        eis, lgs = roll(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi)
        lik = jnp.sum(eis * eis)
        lik2 = jnp.sum(lgs)
    elif has_season:
        roll = _get_roll_scan_custom(
            error, trend, season, m_eff, clip_multiplicative_errors, n
        )
        lik = roll(l0, b0, s0, y, n_eff_f, alpha, beta, gamma, phi)
        lik2 = jnp.asarray(0.0, jnp.float64)
    else:
        init_carry = (
            jnp.asarray(0, jnp.int32), l0, b0, s0,
            jnp.asarray(0.0, jnp.float64), jnp.asarray(0.0, jnp.float64),
        )
        (_, _, _, _, lik, lik2), _ = lax.scan(step, init_carry, y)

    n_f64 = jnp.asarray(n_eff, dtype=jnp.float64)
    sse = jnp.where(lik > 0.0, lik, lik + 1e-8)
    sigma2 = sse / n_f64
    base = n_f64 * (jnp.log(2.0 * jnp.pi) + 1.0 + jnp.log(sigma2))
    if is_mul_error:
        return base + 2.0 * lik2
    return base


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


@partial(jax.jit, static_argnames=("opt_alpha", "opt_beta", "opt_gamma", "opt_phi", "pure_sigmoid"))
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

    Jitted with config-only statics (session-2 lever 3, S2 pattern): the
    eager post-fit unpacking call dispatched ~40 tiny ops per candidate; the
    in-trace optimizer call site (``_objective_smoothing_only``) passes the
    same Python-bool statics and simply inlines.

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

    # Static fast path for the criterion every auto/fixed ETS fit actually
    # optimises: the general rollout below threads the AMSE machinery (dead
    # weight for the likelihood) through the differentiated scan carry, which
    # dominates AutoETS's warm cost.
    # Bit-identical values; opt_crit is trace-static so no lax branching.
    if opt_crit == Criterion.Likelihood:
        return _calc_roll_lik(
            init_state,
            y,
            jnp.asarray(n_obs, dtype=jnp.int32),
            error,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            m,
            clip_multiplicative_errors=clip_multiplicative_errors,
        )

    # Run the lightweight (no state-history) rollout to get residuals + likelihood.
    e = jnp.zeros_like(y, dtype=jnp.float64)
    a_mse = jnp.zeros((n_mse,), dtype=jnp.float64)
    n_obs_arr = jnp.asarray(n_obs, dtype=jnp.int32)
    e, a_mse, lik, _ = _calc_roll_nohist(
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
"""L-BFGS refinement budget CAP (quasi-Newton steps after the Adam warm-up).
`adaptive_iterate` stops earlier once the loss plateaus; hard problems (e.g. co2,
trend-heavy long series) still run the full 30."""

_LBFGS_RTOL = 1e-4
"""Relative-loss-improvement plateau threshold for the adaptive L-BFGS refinement
Tuned so fast-converging candidates stop early while trend-heavy series keep
running to `_LBFGS_STEPS`, which is the accuracy guard.
Larger ⇒ stops sooner (accuracy risk); smaller ⇒ closer to the fixed-30 budget."""

_LBFGS_PATIENCE = 2
"""Consecutive plateau iterations required before the adaptive L-BFGS stops. Higher
values ride through a temporary flat spot (a plateau-then-drop loss landscape, e.g.
seasonal-AR) at the cost of a few extra steps on truly-converged candidates."""


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

        def _plain_step(opt_update):
            # `step(carry) -> (carry', loss)` for adaptive_iterate: evaluate the
            # loss at the current point, then advance. adaptive_iterate does the
            # best-evaluated-point tracking + plateau stop that _tracked_step does
            # inline for the fixed Adam scan — same NaN-graceful semantics (a
            # non-finite loss never becomes the best).
            def step(carry):
                p, opt_state = carry
                loss, grads = vg_fn(p)
                grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
                updates, opt_state = opt_update(grads, opt_state, p, loss)
                new_p = optax.apply_updates(p, updates)
                return (new_p, opt_state), loss
            return step

        inf = jnp.asarray(jnp.inf, dtype=jnp.float64)

        # ── Phase 1: Adam warm-up ─────────────────────────────────────
        adam = optax.adam(adam_lr)

        def _adam_update(grads, state, p, loss):
            return adam.update(grads, state, p)

        if adam_steps > 0:
            (_, _, adam_p, _), _ = lax.scan(
                _tracked_step(_adam_update),
                (x0, adam.init(x0), x0, inf),
                None,
                length=adam_steps,
            )
        else:
            adam_p = x0
        # adam's best loss is not threaded to L-BFGS: adaptive_iterate re-evaluates
        # loss(adam_p) as its probe-step seed, which is the same value.

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

        # Adaptive L-BFGS: stop once the loss plateaus (rel improvement < rtol for
        # 2 steps), capped at _LBFGS_STEPS. Eager (one candidate at a time) this
        # stops fast-converging candidates early — the P6 warm speedup; under the
        # CV vmap it runs to the slowest lane's stop, capped — never more than the
        # old fixed scan. best_p seeds from Adam's best (params_of(init)=adam_p).
        best_p, best_loss, _ = adaptive_iterate(
            _plain_step(_lbfgs_update),
            (adam_p, lbfgs.init(adam_p)),
            lambda c: c[0],
            max_steps=_LBFGS_STEPS,
            rtol=_LBFGS_RTOL,
            patience=_LBFGS_PATIENCE,
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

    # Iteration CAP, not the actual count: the adaptive L-BFGS may plateau-stop
    # before _LBFGS_STEPS. nit is reported metadata only (not consumed by the
    # math/vmap path); the true count stays inside the jitted runner.
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
