# ets_src.py (JAX version using SciPy Nelder–Mead, plus calc_full)
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

# ---------------------------
# Constants (match C++ values)
# ---------------------------
HUGE_N: float = 1e10
NA: float = -99999.0
TOL: float = 1e-10


# ---------------------------
# Enums (mirror the C++ enums)
# ---------------------------
class Component(Enum):
    Nothing = 0
    Additive = 1
    Multiplicative = 2


class Criterion(Enum):
    Likelihood = 0
    MSE = 1
    AMSE = 2
    Sigma = 3
    MAE = 4


# -----------------------------------
# Optimizer return (POD-like) struct
# -----------------------------------
class OptimResult(NamedTuple):
    success: bool
    status: int
    message: str
    x: jnp.ndarray
    fun: float
    nit: int
    nfev: int


# ---------------------------
# Core state update
# ---------------------------
def update(
    s: jnp.ndarray,
    l: float,
    b: float,
    old_l: float,
    old_b: float,
    old_s: jnp.ndarray,
    m: int,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    y: float,
) -> Tuple[float, float, jnp.ndarray]:
    # new level (q) depends on trend
    if trend == Component.Nothing:
        q = old_l
        phi_b = 0.0
    elif trend == Component.Additive:
        phi_b = phi * old_b
        q = old_l + phi_b
    else:  # Multiplicative
        if abs(phi - 1.0) < TOL:
            phi_b = old_b
            q = old_l * old_b
        else:
            phi_b = old_b**phi
            q = old_l * phi_b

    # seasonally adjusted observation p
    if season == Component.Nothing:
        p = y
    elif season == Component.Additive:
        p = y - float(old_s[m - 1])
    else:  # Multiplicative
        if abs(float(old_s[m - 1])) < TOL:
            p = HUGE_N
        else:
            p = y / float(old_s[m - 1])

    # new level
    l = q + alpha * (p - q)

    # new growth (if trend present)
    if trend != Component.Nothing:
        if trend == Component.Additive:
            r = l - old_l
        else:
            if abs(old_l) < TOL:
                r = HUGE_N
            else:
                r = l / old_l
        b = phi_b + (beta / alpha) * (r - phi_b)

    # new seasonal (if present)
    if season != Component.Nothing:
        if season == Component.Additive:
            t = y - q
        else:
            if abs(q) < TOL:
                t = HUGE_N
            else:
                t = y / q
        s = s.at[0].set(float(old_s[m - 1]) + gamma * (t - float(old_s[m - 1])))
        if m > 1:
            s = s.at[1:m].set(old_s[0 : m - 1])

    return float(l), float(b), s


# ---------------------------
# h-step forecast
# ---------------------------
def forecast(
    f: jnp.ndarray,
    l: float,
    b: float,
    s: jnp.ndarray,
    m: int,
    trend: Component,
    season: Component,
    phi: float,
    h: int,
) -> jnp.ndarray:                   # ⬅️ return type
    phistar = phi
    for i in range(h):
        if trend == Component.Nothing:
            fi = l
        elif trend == Component.Additive:
            fi = l + phistar * b
        else:
            fi = jnp.nan if (b < 0) else l * (b**phistar)

        j = m - 1 - i
        while j < 0:
            j += m

        if season == Component.Additive:
            fi = fi + float(s[j])
        elif season == Component.Multiplicative:
            fi = fi * float(s[j])

        f = f.at[i].set(float(fi))

        if i < h - 1:
            if abs(phi - 1.0) < TOL:
                phistar += 1.0
            else:
                phistar += phi ** (i + 1)

    return f                         # ⬅️ critical



# ---------------------------
# Likelihood + error rollout
# ---------------------------
def _calc_roll(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    m: int,
) -> float:
    n = y.shape[0]
    n_s = max(m, 24)
    m_eff = max(m, 1)
    n_mse_eff = min(n_mse, 30)

    n_states = m_eff * (season != Component.Nothing) + (trend != Component.Nothing) + 1
    # initial states in x[0:n_states]
    l = float(x[0])
    b = float(x[1]) if trend != Component.Nothing else 0.0

    s_vec = jnp.zeros(n_s, dtype=jnp.float64)
    if season != Component.Nothing:
        start = 1 + int(trend != Component.Nothing)
        s_vec = s_vec.at[:m_eff].set(x[start : start + m_eff])

    a_mse = a_mse.at[:n_mse_eff].set(0.0)
    old_s = jnp.zeros(n_s, dtype=jnp.float64)
    denom = jnp.zeros(30, dtype=jnp.float64)
    f = jnp.zeros(30, dtype=jnp.float64)
    old_b = 0.0
    lik = 0.0
    lik2 = 0.0

    for i in range(int(n)):
        old_l = l
        if trend != Component.Nothing:
            old_b = b
        if season != Component.Nothing:
            old_s = old_s.at[:m_eff].set(s_vec[:m_eff])

        f = forecast(f, old_l, old_b, old_s, m_eff, trend, season, phi, n_mse_eff)

        if abs(float(f[0]) - NA) < TOL:
            return NA

        if error == Component.Additive:
            e = e.at[i].set(float(y[i]) - float(f[0]))
        else:
            f0 = float(f[0]) if abs(float(f[0])) >= TOL else float(f[0]) + TOL
            e = e.at[i].set((float(y[i]) - float(f[0])) / f0)

        for j in range(n_mse_eff):
            if i + j < n:
                denom = denom.at[j].set(float(denom[j] + 1.0))
                tmp = float(y[i + j]) - float(f[j])
                a_mse = a_mse.at[j].set((float(a_mse[j]) * (float(denom[j]) - 1.0) + tmp * tmp) / float(denom[j]))

        # IMPORTANT: capture updated seasonal state from update()
        l, b, s_vec = update(
            s_vec, l, b, old_l, old_b, old_s, m_eff, trend, season, alpha, beta, gamma, phi, float(y[i])
        )

        x = x.at[n_states * (i + 1)].set(l)
        if trend != Component.Nothing:
            x = x.at[n_states * (i + 1) + 1].set(b)
        if season != Component.Nothing:
            start = n_states * (i + 1) + 1 + int(trend != Component.Nothing)
            x = x.at[start : start + m_eff].set(s_vec[:m_eff])

        lik += float(e[i]) * float(e[i])
        val = abs(float(f[0]))
        lik2 += math.log(val if val > 0.0 else val + 1e-8)

    n_float = float(n)
    lik = n_float * math.log(lik if lik > 0.0 else lik + 1e-8)
    if error == Component.Multiplicative:
        lik += 2.0 * lik2
    return float(lik)


# --- full-buffer-returning API (for callers that need e/amse/states) ---
def calc_full(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    m: int,
):
    n = y.shape[0]
    m_eff = max(m, 1)
    n_states = m_eff * (season != Component.Nothing) + (trend != Component.Nothing) + 1

    # Build a clean work buffer
    x_work = jnp.zeros_like(x)    # <= instead of jnp.array(x, ...)
    e_work = jnp.zeros_like(e)
    a_work = jnp.zeros_like(a_mse)

    # Copy just the *initial* state from the head of x
    x_work = x_work.at[:n_states].set(x[:n_states])

    lik = _calc_roll(
        x_work, e_work, a_work, n_mse, y,
        error, trend, season, alpha, beta, gamma, phi, m
    )

    states = x_work.reshape((n + 1, n_states))
    return a_work, e_work, states, float(lik)


# Keep old scalar API for optimizer
def calc(
    x: jnp.ndarray,
    e: jnp.ndarray,
    a_mse: jnp.ndarray,
    n_mse: int,
    y: jnp.ndarray,
    error: Component,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    m: int,
) -> float:
    _, _, _, lik = calc_full(x, e, a_mse, n_mse, y, error, trend, season, alpha, beta, gamma, phi, m)
    return float(lik)


# ---------------------------------------
# SciPy Nelder–Mead optimizer (with box)
# ---------------------------------------
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
    adaptive: bool,  # ignored by SciPy NM
) -> OptimResult:
    """
    Nelder–Mead via SciPy. Bounds are enforced by projection + soft penalty
    since NM ignores hard bounds.
    """
    # Host (NumPy) copies
    x0_np    = np.asarray(jax.device_get(x0), dtype=np.float64)
    y_np     = np.asarray(jax.device_get(y),  dtype=np.float64)
    lower_np = np.asarray(jax.device_get(lower), dtype=np.float64)
    upper_np = np.asarray(jax.device_get(upper), dtype=np.float64)

    PENALTY_W = 1e6  # penalty weight for box violations

    def _objective_function(vec_np: np.ndarray) -> float:
        params = jnp.asarray(vec_np, dtype=jnp.float64)
        Y      = jnp.asarray(y_np,    dtype=jnp.float64)

        j = 0
        a = alpha
        b = beta
        g = gamma
        p = phi
        if opt_alpha:
            a = float(params[j]); j += 1
        if opt_beta:
            b = float(params[j]); j += 1
        if opt_gamma:
            g = float(params[j]); j += 1
        if opt_phi:
            p = float(params[j]); j += 1

        n_params = int(params.size)
        n        = int(Y.size)
        add_season_balancer = int(season != Component.Nothing)

        state = jnp.zeros((n_state + add_season_balancer) * (n + 1), dtype=jnp.float64)
        head  = params[n_params - n_state : n_params]
        state = state.at[:n_state].set(head)

        if season != Component.Nothing:
            start = 1 + int(trend != Component.Nothing)
            s_sum = jnp.sum(state[start:n_state])
            state = state.at[n_state].set(float(m * (season == Component.Multiplicative)) - float(s_sum))
            if season == Component.Multiplicative and float(jnp.min(state[start:])) < 0.0:
                return float(np.inf)

        a_mse = jnp.zeros(30, dtype=jnp.float64)
        e     = jnp.zeros(n,  dtype=jnp.float64)

        lik = _calc_roll(state, e, a_mse, n_mse, Y, error, trend, season, a, b, g, p, m)
        lik = max(lik, -1e10)
        if math.isnan(lik) or abs(lik + 99999.0) < 1e-7:
            lik = -float('inf')

        if opt_crit == Criterion.Likelihood:
            obj_val = lik
        elif opt_crit == Criterion.MSE:
            obj_val = float(a_mse[0])
        elif opt_crit == Criterion.AMSE:
            obj_val = float(jnp.mean(a_mse[: min(n_mse, 30)]))
        elif opt_crit == Criterion.Sigma:
            obj_val = float(jnp.mean(e * e))
        else:  # Criterion.MAE
            obj_val = float(jnp.mean(jnp.abs(e)))
        return float(obj_val)

    # Project-to-box + soft penalty wrapper
    def _penalized_objective(vec_np: np.ndarray) -> float:
        clipped = np.clip(vec_np, lower_np, upper_np)
        diff    = vec_np - clipped
        penalty = PENALTY_W * float(np.dot(diff, diff))
        val     = _objective_function(clipped)
        if not np.isfinite(val):
            val = 1e300
        return val + penalty

    options = {
        "maxiter": int(max_iter),
        "fatol":   float(tol_std),  # function tolerance (≈ your tol_std)
        "xatol":   1e-9,
        "disp":    False,
    }

    res = minimize(
        fun=_penalized_objective,
        x0=x0_np,
        method="Nelder-Mead",
        options=options,
    )

    return OptimResult(
        success=bool(res.success),
        status=int(getattr(res, "status", 0)),
        message=str(getattr(res, "message", "")),
        x=jnp.asarray(res.x, dtype=jnp.float64),
        fun=float(res.fun),
        nit=int(getattr(res, "nit", -1)),
        nfev=int(getattr(res, "nfev", -1)),
    )


__all__ = [
    "HUGE_N",
    "NA",
    "TOL",
    "Component",
    "Criterion",
    "OptimResult",
    "update",
    "forecast",
    "calc_full",
    "calc",
    "optimize",
]
