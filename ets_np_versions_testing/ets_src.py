# ets_src.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    # SciPy provides the built-in Nelder–Mead optimizer
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    raise ImportError(
        "ets.optimize() requires SciPy. Please `pip install scipy`."
    ) from e


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
    x: NDArray[np.float64]
    fun: float
    nit: int
    nfev: int


# ---------------------------
# Core state update
# ---------------------------
def update(
    s: NDArray[np.float64],
    l: float,
    b: float,
    old_l: float,
    old_b: float,
    old_s: NDArray[np.float64],
    m: int,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    y: float,
) -> Tuple[float, float]:
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
        p = y - old_s[m - 1]
    else:  # Multiplicative
        if abs(old_s[m - 1]) < TOL:
            p = HUGE_N
        else:
            p = y / old_s[m - 1]

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
        s[0] = old_s[m - 1] + gamma * (t - old_s[m - 1])
        # rotate the previous seasonal states
        if m > 1:
            s[1:m] = old_s[0 : m - 1]

    return l, b


# ---------------------------
# h-step forecast
# ---------------------------
def forecast(
    f: NDArray[np.float64],
    l: float,
    b: float,
    s: NDArray[np.float64],
    m: int,
    trend: Component,
    season: Component,
    phi: float,
    h: int,
) -> None:
    phistar = phi
    for i in range(h):
        if trend == Component.Nothing:
            fi = l
        elif trend == Component.Additive:
            fi = l + phistar * b
        else:
            if b < 0:
                fi = np.nan
            else:
                fi = l * (b**phistar)

        j = m - 1 - i
        while j < 0:
            j += m

        if season == Component.Additive:
            fi = fi + s[j]
        elif season == Component.Multiplicative:
            fi = fi * s[j]

        f[i] = fi

        if i < h - 1:
            if abs(phi - 1.0) < TOL:
                phistar += 1.0
            else:
                phistar += phi ** (i + 1)


# ---------------------------
# Likelihood + error rollout
# ---------------------------
def _calc_roll(
    x: NDArray[np.float64],
    e: NDArray[np.float64],
    a_mse: NDArray[np.float64],
    n_mse: int,
    y: NDArray[np.float64],
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

    s_vec = np.zeros(n_s, dtype=float)
    if season != Component.Nothing:
        start = 1 + (trend != Component.Nothing)
        s_vec[:m_eff] = x[start : start + m_eff]

    a_mse[:n_mse_eff] = 0.0
    old_s = np.zeros(n_s, dtype=float)
    denom = np.zeros(30, dtype=float)
    f = np.zeros(30, dtype=float)
    old_b = 0.0
    lik = 0.0
    lik2 = 0.0

    for i in range(n):
        old_l = l
        if trend != Component.Nothing:
            old_b = b
        if season != Component.Nothing:
            old_s[:m_eff] = s_vec[:m_eff]

        # one-step and up-to-n_mse forecasts
        forecast(f, old_l, old_b, old_s, m_eff, trend, season, phi, n_mse_eff)

        if abs(f[0] - NA) < TOL:
            return NA

        if error == Component.Additive:
            e[i] = y[i] - f[0]
        else:
            f0 = f[0] if abs(f[0]) >= TOL else f[0] + TOL
            e[i] = (y[i] - f[0]) / f0

        for j in range(n_mse_eff):
            if i + j < n:
                denom[j] += 1.0
                tmp = y[i + j] - f[j]
                a_mse[j] = (a_mse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j]

        # state update
        l, b = update(s_vec, l, b, old_l, old_b, old_s, m_eff, trend, season, alpha, beta, gamma, phi, float(y[i]))

        # store back the states into x
        x[n_states * (i + 1)] = l
        if trend != Component.Nothing:
            x[n_states * (i + 1) + 1] = b
        if season != Component.Nothing:
            start = n_states * (i + 1) + 1 + (trend != Component.Nothing)
            x[start : start + m_eff] = s_vec[:m_eff]

        lik += e[i] * e[i]
        val = abs(f[0])
        lik2 += np.log(val if val > 0.0 else val + 1e-8)

    n_float = float(n)
    lik = n_float * np.log(lik if lik > 0.0 else lik + 1e-8)
    if error == Component.Multiplicative:
        lik += 2.0 * lik2
    return float(lik)


def calc(
    x: NDArray[np.float64],
    e: NDArray[np.float64],
    a_mse: NDArray[np.float64],
    n_mse: int,
    y: NDArray[np.float64],
    error: Component,
    trend: Component,
    season: Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    m: int,
) -> float:
    return _calc_roll(x, e, a_mse, n_mse, y, error, trend, season, alpha, beta, gamma, phi, m)


# ---------------------------
# Objective (keeps signature)
# ---------------------------
def _objective_function(
    params: NDArray[np.float64],
    y: NDArray[np.float64],
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
) -> float:
    j = 0
    if opt_alpha:
        alpha = float(params[j]); j += 1
    if opt_beta:
        beta = float(params[j]); j += 1
    if opt_gamma:
        gamma = float(params[j]); j += 1
    if opt_phi:
        phi = float(params[j]); j += 1

    n_params = params.size
    n = y.size

    p = n_state + (season != Component.Nothing)
    state = np.zeros(p * (n + 1), dtype=float)

    # copy last n_state elements of params into the head of state
    state[:n_state] = params[n_params - n_state : n_params]

    if season != Component.Nothing:
        start = 1 + (trend != Component.Nothing)
        # extra balancing state term
        s_sum = state[start:n_state].sum()
        state[n_state] = float(m * (season == Component.Multiplicative)) - s_sum

        if season == Component.Multiplicative and np.min(state[start:]) < 0.0:
            return float(np.inf)

    a_mse = np.zeros(30, dtype=float)
    e = np.zeros(n, dtype=float)

    lik = _calc_roll(state, e, a_mse, n_mse, y, error, trend, season, alpha, beta, gamma, phi, m)
    lik = max(lik, -1e10)

    if np.isnan(lik) or abs(lik + 99999.0) < 1e-7:
        lik = -float(np.inf)

    if opt_crit == Criterion.Likelihood:
        obj_val = lik
    elif opt_crit == Criterion.MSE:
        obj_val = a_mse[0]
    elif opt_crit == Criterion.AMSE:
        obj_val = float(np.mean(a_mse[: min(n_mse, 30)]))
    elif opt_crit == Criterion.Sigma:
        obj_val = float(np.mean(e * e))
    else:  # MAE
        obj_val = float(np.mean(np.abs(e)))
    return obj_val


# ---------------------------------------
# Nelder–Mead optimizer with box penalty
# ---------------------------------------
def optimize(
    x0: NDArray[np.float64],
    y: NDArray[np.float64],
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
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    tol_std: float,
    max_iter: int,
    adaptive: bool,
) -> OptimResult:
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Penalty if outside [lower, upper] because SciPy Nelder–Mead does not support bounds
    def boxed_objective(p):
        # hard penalty for violating bounds
        vio_low = np.maximum(0.0, lower - p)
        vio_up = np.maximum(0.0, p - upper)
        vio = float(np.dot(vio_low, vio_low) + np.dot(vio_up, vio_up))
        if vio > 0.0:
            # Scale penalty by a large factor to keep simplex within the box
            return 1e6 * vio

        return _objective_function(
            p, y, n_state, error, trend, season, opt_crit, n_mse, m,
            opt_alpha, opt_beta, opt_gamma, opt_phi, alpha, beta, gamma, phi
        )

    res = minimize(
        boxed_objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": int(max_iter),
            # Map tol_std to function tolerance; NM also has xatol but we keep one knob
            "fatol": float(tol_std),
            "adaptive": bool(adaptive),
        },
    )

    return OptimResult(
        success=bool(res.success),
        status=int(getattr(res, "status", 1)),
        message=str(res.message),
        x=np.asarray(res.x, dtype=float),
        fun=float(res.fun),
        nit=int(getattr(res, "nit", 0)),
        nfev=int(getattr(res, "nfev", 0)),
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
    "calc",
    "optimize",
]
