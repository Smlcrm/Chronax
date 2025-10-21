# ets.py (JAX version)
"""
High-level ETS (Exponential Smoothing) model API in JAX.

This module wires together:
- low-level ETS kernels and a SciPy-like Nelder–Mead optimizer (from `_ets`),
- parameter initialization and admissibility checks,
- state initialization via simple regression/Fourier or seasonal decomposition,
- full model selection / fitting (AICc, etc.),
- forecasting and analytical/simulation-based prediction intervals.

The goal is functional parity with NumPy/statsmodels-style ETS flows while
keeping kernels JIT-friendly and deterministic.
"""

__all__ = ['ets_f']

import math
from typing import Dict, Any
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrand
from statsmodels.tsa.seasonal import seasonal_decompose

import ets_backend as _ets
from utils import _calculate_intervals, results

# Global variables
_smalno = jnp.finfo(float).eps
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98


def etssimulate(
    x: jnp.ndarray,
    m: int,
    error: _ets.Component,
    trend: _ets.Component,
    season: _ets.Component,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    h: int,
    y: jnp.ndarray,
    e: jnp.ndarray,
) -> None:
    """
    Simulate h-step future sample paths from a given ETS state.

    Parameters
    ----------
    x : jnp.ndarray
        Initial state vector (level [+ trend] [+ m seasonal entries]).
    m : int
        Seasonal period (>=1). If `season != Nothing`, must be <= 24 here.
    error, trend, season : _ets.Component
        ETS structure: additive/multiplicative or absent.
    alpha, beta, gamma, phi : float
        Smoothing parameters (beta/phi used only if trend present; gamma only if seasonal).
    h : int
        Forecast horizon to simulate.
    y : jnp.ndarray
        Preallocated output array of length h; filled with simulated values.
    e : jnp.ndarray
        Innovations (length h) already drawn (e.g., from N(0, sigma)).

    Notes
    -----
    This is a *stateful* helper used by interval simulation for Class 4/5
    models; it mirrors the kernel `_ets.update` and `_ets.forecast` behavior.
    """
    oldb = 0.0
    olds = jnp.zeros(24)
    s = jnp.zeros(24)
    f = jnp.zeros(10)
    if m > 24 and season != _ets.Component.Nothing:
        return
    elif m < 1:
        m = 1
    # Copy initial state components
    l = float(x[0])
    if trend != _ets.Component.Nothing:
        b = float(x[1])
    else:
        b = 0.0
    if season != _ets.Component.Nothing:
        # x offset = 1 + (trend != Nothing)
        off = 1 + int(trend != _ets.Component.Nothing)
        for j in range(m):
            s = s.at[j].set(float(x[off + j]))

    for i in range(h):
        # Copy previous state
        oldl = l
        if trend != _ets.Component.Nothing:
            oldb = b
        if season != _ets.Component.Nothing:
            olds = olds.at[:m].set(s[:m])

        # one step forecast
        f = _ets.forecast(
            f,
            oldl,
            oldb,
            olds,
            m,
            trend,
            season,
            phi,
            1,
        )
        if math.fabs(float(f[0]) - _ets.NA) < _ets.TOL:
            y = y.at[0].set(_ets.NA)
            return
        if error == _ets.Component.Additive:
            y = y.at[i].set(float(f[0]) + float(e[i]))
        else:
            y = y.at[i].set(float(f[0]) * (1.0 + float(e[i])))

        # Update state
        l, b, s = _ets.update(
            s,
            l,
            b,
            oldl,
            oldb,
            olds,
            m,
            trend,
            season,
            alpha,
            beta,
            gamma,
            phi,
            float(y[i]),
        )


def etsforecast(
    x: jnp.ndarray,
    m: int,
    trend: _ets.Component,
    season: _ets.Component,
    phi: float,
    h: int,
    f: jnp.ndarray,
) -> jnp.ndarray:
    """
    Produce h-step-ahead forecasts from a state snapshot.

    Parameters
    ----------
    x : jnp.ndarray
        State vector (level [+ trend] [+ m seasonal]).
    m : int
        Seasonal period (>=1).
    trend, season : _ets.Component
        Structural flags for trend/seasonality.
    phi : float
        Damping parameter (ignored if no trend).
    h : int
        Number of steps to forecast.
    f : jnp.ndarray
        Optional preallocated buffer (length h); created if None/wrong shape.

    Returns
    -------
    jnp.ndarray
        Forecasts of shape (h,).
    """
    if m < 1:
        m = 1
    dt = x.dtype

    l = float(x[0])
    has_trend = (trend != _ets.Component.Nothing)
    b = float(x[1]) if has_trend else 0.0

    if season != _ets.Component.Nothing:
        start = 1 + int(has_trend)
        s = jnp.asarray(x[start:start + m], dtype=dt)
    else:
        s = jnp.zeros((m,), dtype=dt)

    if (f is None) or (getattr(f, "shape", ()) != (h,)):
        f = jnp.zeros((h,), dtype=dt)

    f = _ets.forecast(
        f=f,
        l=l,
        b=b,
        s=s,
        m=int(m),
        trend=trend,
        season=season,
        phi=float(phi),
        h=int(h),
    )
    return f


def initparam(
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    trendtype: str,
    seasontype: str,
    damped: bool,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    m: int,
    bounds: str,
):
    """
    Initialize (and lightly sanitize) smoothing parameters and bounds.

    Mirrors R/NumPy-style heuristics to pick starting values when any of
    alpha/beta/gamma/phi are NaN, and adjusts bounds to respect typical
    constraints (e.g., beta <= alpha, gamma <= 1 - alpha).

    Parameters
    ----------
    alpha, beta, gamma, phi : float
        Optional user-provided starting values (use NaN to auto-init).
    trendtype, seasontype : {"N","A","M"}
        Structure flags as strings for convenience.
    damped : bool
        Whether a damped trend is considered.
    lower, upper : jnp.ndarray
        4-element arrays of lower/upper bound suggestions.
    m : int
        Seasonal period.
    bounds : {"both","usual","admissible"}
        Bound mode (admissible relaxes early to allow search to start).

    Returns
    -------
    (dict, jnp.ndarray, jnp.ndarray)
        Dict of possibly-updated {alpha,beta,gamma,phi}, and the (possibly
        clipped) lower/upper arrays actually used.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    if bounds == "admissible":
        lower = lower.at[:3].set(0.0)
        upper = upper.at[:3].set(1e-3)
    elif jnp.any(lower > upper):
        raise Exception("Inconsistent parameter boundaries")

    # select alpha
    if math.isnan(alpha):
        alpha = float(lower[0] + 0.2 * (upper[0] - lower[0]) / m)
        if alpha > 1 or alpha < 0:
            alpha = float(lower[0] + 2e-3)
    
    # select beta
    if trendtype != "N" and math.isnan(beta):
        upper = upper.at[1].set(float(jnp.minimum(upper[1], alpha)))
        beta = float(lower[1] + 0.1 * (upper[1] - lower[1]))
        if beta < 0 or beta > alpha:
            beta = alpha - 1e-3
    
    # select gamma
    if seasontype != "N" and math.isnan(gamma):
        upper = upper.at[2].set(float(jnp.minimum(upper[2], 1 - alpha)))
        gamma = float(lower[2] + 0.05 * (upper[2] - lower[2]))
        if gamma < 0 or gamma > 1 - alpha:
            gamma = 1 - alpha - 1e-3
    
    # select phi
    if damped and math.isnan(phi):
        phi = float(lower[3] + 0.99 * (upper[3] - lower[3]))
        if phi < 0 or phi > 1:
            phi = float(upper[3] - 1e-3)
    
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "phi": phi}, lower, upper


def _polyroots_power_basis(coeff_power_inc: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the roots of a polynomial in power basis with increasing coefficients.

    Parameters
    ----------
    coeff_power_inc : jnp.ndarray
        Coefficients [c0, c1, ..., cN] representing P(x)=c0+c1 x+...+cN x^N.

    Returns
    -------
    jnp.ndarray (complex128)
        Eigenvalues of the companion matrix (the polynomial roots).
    """
    c = jnp.asarray(coeff_power_inc, dtype=jnp.float64)
    n = c.shape[0] - 1
    if n <= 0:
        return jnp.array([], dtype=jnp.complex128)
    # Highest degree coefficient
    a_n = c[-1]
    # Handle degenerate
    if jnp.isclose(a_n, 0.0):
        # trim trailing zeros
        idx = jnp.where(jnp.abs(c[::-1]) > 0)[0]
        if idx.size == 0:
            return jnp.array([], dtype=jnp.complex128)
        k = int(idx[0])
        c = c[: c.shape[0] - k]
        n = c.shape[0] - 1
        if n <= 0:
            return jnp.array([], dtype=jnp.complex128)
        a_n = c[-1]
    # normalize to monic
    c_monic = c / a_n
    # Companion matrix for x^n + c_{n-1} x^{n-1} + ... + c0
    # Build with JAX
    C = jnp.zeros((n, n), dtype=jnp.float64)
    C = C.at[1:, :-1].set(jnp.eye(n - 1, dtype=jnp.float64))
    C = C.at[0, :].set(-c_monic[:-1][::-1])
    # eigenvalues
    evals = jnp.linalg.eigvals(C.astype(jnp.complex128))
    return evals


def admissible(alpha: float, beta: float, gamma: float, phi: float, m: int):
    """
    Check ETS smoothing parameters against standard admissibility conditions.

    Includes classical ETS constraints and a seasonal stability check via the
    characteristic polynomial’s roots (|root| <= 1).

    Returns
    -------
    bool
        True if parameter tuple passes admissibility checks.
    """
    # Mirror original admissibility tests, including characteristic-equation root check
    if math.isnan(phi):
        phi = 1.0
    if phi < 0.0 or phi > 1 + 1e-8:
        return False
    if math.isnan(gamma):
        # non-seasonal
        if alpha < 1 - 1 / phi or alpha > 1 + 1 / phi:
            return False
        if not math.isnan(beta):
            if beta < alpha * (phi - 1) or beta > (1 + phi) * (2 - alpha):
                return False
    elif m > 1:  # seasonal model
        if math.isnan(beta):
            beta = 0.0
        if gamma < max(1 - 1 / phi - alpha, 0) or gamma > 1 + 1 / phi - alpha:
            return False
        if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m):
            return False
        if beta < -(1 - phi) * (gamma / m + alpha):
            return False
        # Characteristic-equation check via companion-matrix roots (JAX)
        P = jnp.full(2 + m - 2 + 2, jnp.nan)
        P = P.at[:2].set(jnp.array([phi * (1 - alpha - gamma), alpha + beta - alpha * phi + gamma - 1.0]))
        mid_len = m - 2
        if mid_len > 0:
            P = P.at[2 : (m - 2 + 2)].set(jnp.repeat(alpha + beta - alpha * phi, mid_len))
        P = P.at[(m - 2 + 2) :].set(jnp.array([alpha + beta - phi, 1.0]))
        roots = _polyroots_power_basis(P)  # complex
        mod = jnp.abs(roots)
        max_mod = float(jnp.max(mod))
        if max_mod > 1 + 1e-10:
            return False
    return True


def check_param(
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    bounds: str,
    m: int,
):
    """
    Validate smoothing parameters against box bounds and (optionally) admissibility.

    Parameters
    ----------
    alpha, beta, gamma, phi : float
        Candidate smoothing parameters (NaN for unused, e.g., when no season).
    lower, upper : jnp.ndarray
        Elementwise lower/upper bounds (length 4).
    bounds : {"both","usual","admissible"}
        If not "admissible", enforce box bounds; if not "usual", enforce ETS admissibility.
    m : int
        Seasonal period.

    Returns
    -------
    bool
        True if parameters are within range and admissible per `bounds`.
    """
    lower = jnp.asarray(lower, dtype=jnp.float64)
    upper = jnp.asarray(upper, dtype=jnp.float64)

    if bounds != "admissible":
        if not math.isnan(alpha):
            if alpha < float(lower[0]) or alpha > float(upper[0]):
                return False
        if not math.isnan(beta):
            if beta < float(lower[1]) or beta > alpha or beta > float(upper[1]):
                return False
        if not math.isnan(phi):
            if phi < float(lower[3]) or phi > float(upper[3]):
                return False
        if not math.isnan(gamma):
            if gamma < float(lower[2]) or gamma > 1 - alpha or gamma > float(upper[2]):
                return False
    if bounds != "usual":
        if not admissible(alpha, beta, gamma, phi, m):
            return False
    return True


def fourier(x, period, K, h=None):
    """
    Build a simple Fourier design matrix for seasonality.

    Parameters
    ----------
    x : array-like
        Input series (used only for length alignment).
    period : list[int]
        Seasonal periods to include (e.g., [m]).
    K : list[int]
        Number of harmonics per period.
    h : Optional[int]
        If provided, build the matrix for the *future* h steps; else fit window.

    Returns
    -------
    jnp.ndarray
        Matrix with sin/cos columns for selected harmonics, with degenerate
        sinpi=0 columns removed.
    """
    n = len(x)
    if h is None:
        times = jnp.arange(1, n + 1, dtype=jnp.float64)
    else:
        times = jnp.arange(n + 1, n + h + 1, dtype=jnp.float64)

    len_p = sum(K)
    p = jnp.full((len_p,), jnp.nan)
    idx = 0
    for j, per in enumerate(period):
        if K[j] > 0:
            vals = jnp.arange(1, K[j] + 1, dtype=jnp.float64) / float(per)
            p = p.at[idx : idx + K[j]].set(vals)
            idx += K[j]
    p = jnp.unique(p)
    # Remove columns where sinpi=0
    k = jnp.abs(2 * p - jnp.round(2 * p)) > _smalno
    X = jnp.full((times.shape[0], 2 * p.shape[0]), jnp.nan)
    for j in range(p.shape[0]):
        if bool(k[j]):
            X = X.at[:, 2 * j - 1].set(jnp.sin(2 * jnp.pi * p[j] * times))
        X = X.at[:, 2 * j].set(jnp.cos(2 * jnp.pi * p[j] * times))
    colmask = ~jnp.isnan(jnp.sum(X, axis=0))
    X = X[:, colmask]
    return X


def initstate(y, m, trendtype, seasontype):
    """
    Initialize ETS states (level [+ trend] [+ seasonal]) from data.

    Strategy
    --------
    - If seasonal and `len(y) < 3m`: fit a small Fourier regression to extract
      rough seasonality; else use `seasonal_decompose` (statsmodels).
    - Deseasonalize (if needed), then fit OLS for intercept/slope over the
      first `max(10, 2m)` points to seed level/trend, with multiplicative
      reparametrization safeguards.

    Returns
    -------
    jnp.ndarray
        Concatenated initial state vector.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    if seasontype != "N":
        if n < 4:
            raise ValueError("You've got to be joking (not enough data).")
        elif n < 3 * m:  # fit simple Fourier model
            fouriery = fourier(y, [m], [1])
            X_fourier = jnp.full((n, 4), jnp.nan)
            X_fourier = X_fourier.at[:, 0].set(1.0)
            X_fourier = X_fourier.at[:, 1].set(jnp.arange(1, n + 1, dtype=jnp.float64))
            X_fourier = X_fourier.at[:, 2:4].set(fouriery)
            # JAX lstsq
            coefs, *_ = jnp.linalg.lstsq(X_fourier, y, rcond=-1.0)
            if seasontype == "A":
                y_d = {"seasonal": y - coefs[0] - coefs[1] * X_fourier[:, 1]}
            else:
                if not float(jnp.min(y)) > 0:
                    raise Exception(
                        "Multiplicative seasonality is not appropriate for zero and negative values"
                    )
                y_d = {"seasonal": y / (coefs[0] + coefs[1] * X_fourier[:, 1])}
        else:
            # Decomposition (statsmodels), then convert to jnp
            sd = seasonal_decompose(
                jnp.array(y), period=m, model="additive" if seasontype == "A" else "multiplicative"
            )
            y_d = {"seasonal": jnp.asarray(sd.seasonal, dtype=jnp.float64)}
        init_seas = y_d["seasonal"][1:m][::-1]
        if seasontype == "A":
            y_sa = y - y_d["seasonal"]
        else:
            init_seas = jnp.clip(init_seas, a_min=1e-2)
            if float(jnp.sum(init_seas)) > m:
                init_seas = init_seas / jnp.sum(init_seas + 1e-2)
            y_sa = y / jnp.clip(y_d["seasonal"], a_min=1e-2)
    else:
        m = 1
        init_seas = jnp.array([], dtype=jnp.float64)
        y_sa = y

    maxn = min(max(10, 2 * m), int(y_sa.shape[0]))
    if trendtype == "N":
        l0 = float(jnp.mean(y_sa[:maxn]))
        return jnp.concatenate([jnp.array([l0]), init_seas])
    else:
        X = jnp.full((n, 2), jnp.nan)
        X = X.at[:, 0].set(1.0)
        X = X.at[:, 1].set(jnp.arange(1, n + 1, dtype=jnp.float64))
        (l, b), *_ = jnp.linalg.lstsq(X[:maxn], y_sa[:maxn], rcond=-1.0)
        l = float(l); b = float(b)
        if trendtype == "A":
            l0 = l
            b0 = b
            if abs(l0 + b0) < 1e-8:
                l0 = l0 * (1 + 1e-3)
                b0 = b0 * (1 - 1e-3)
        else:
            l0 = l + b
            if abs(l0) < 1e-8:
                l0 = 1e-7
            b0 = (l + 2 * b) / l0
            div = b0 if not math.isclose(b0, 0.0, abs_tol=1e-8) else 1e-8
            l0 = l0 / div
            if abs(b0) > 1e10:
                b0 = math.copysign(1e10, b0)
            if l0 < 1e-8 or b0 < 1e-8:
                l0 = max(float(y_sa[0]), 1e-3)
                div2 = float(y_sa[0]) if not math.isclose(float(y_sa[0]), 0.0, abs_tol=1e-8) else 1e-8
                b0 = max(float(y_sa[1]) / div2, 1e-3)
        return jnp.concatenate([jnp.array([l0, b0]), init_seas])


def switch(x: str) -> _ets.Component:
    """
    Map string flags {"N","A","M"} to `_ets.Component` enum.
    """
    if x == "N":
        return _ets.Component.Nothing
    if x == "A":
        return _ets.Component.Additive
    if x == "M":
        return _ets.Component.Multiplicative
    raise ValueError(f"Unknown component {x}")


def switch_criterion(x: str) -> _ets.Criterion:
    """
    Map objective string to `_ets.Criterion` enum.

    Accepted values: {"lik","mse","amse","sigma","mae"}.
    """
    if x == "lik":
        return _ets.Criterion.Likelihood
    if x == "mse":
        return _ets.Criterion.MSE
    if x == "amse":
        return _ets.Criterion.AMSE
    if x == "sigma":
        return _ets.Criterion.Sigma
    if x == "mae":
        return _ets.Criterion.MAE
    raise ValueError(f"Unknown crtierion {x}")


def pegelsresid_C(
    y: jnp.ndarray,
    m: int,
    init_state: jnp.ndarray,
    errortype: str,
    trendtype: str,
    seasontype: str,
    damped: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    nmse: int,
):
    """
    Roll out ETS residuals, AMSE, and likelihood from an initialized state.

    This is the high-level wrapper around `_ets.calc_full` that:
    - prepares buffers,
    - enforces structural simplifications (e.g., set beta/gamma/phi when absent),
    - reshapes the packed state history,
    - returns `(amse, residuals, states, lik)`.

    Returns
    -------
    (jnp.ndarray, jnp.ndarray, jnp.ndarray, float)
        AMSE vector (length nmse), residuals (length n), state snapshots
        ((n+1) x n_state), and the likelihood-style scalar.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    p = int(init_state.shape[0])
    n = int(y.shape[0])

    x = jnp.full((p * (n + 1),), jnp.nan)
    x = x.at[:p].set(jnp.asarray(init_state, dtype=jnp.float64))
    e = jnp.full_like(y, jnp.nan)

    if not damped:
        phi = 1.0
    if trendtype == "N":
        beta = 0.0
    if seasontype == "N":
        gamma = 0.0
    amse = jnp.full((nmse,), jnp.nan)

    amse, e, x, lik = _ets.calc_full(
        x,
        e,
        amse,
        nmse,
        y,
        switch(errortype),
        switch(trendtype),
        switch(seasontype),
        float(alpha),
        float(beta),
        float(gamma),
        float(phi),
        int(m),
    )
    xs = x.reshape((n + 1, p))
    if not jnp.isnan(lik):
        if float(jnp.abs(lik + 99999)) < _ets.TOL:
            lik = jnp.nan
    return amse, e, xs, float(lik)


def optimize_ets_target_fn(
    x0,
    par,
    y,
    nstate,
    errortype,
    trendtype,
    seasontype,
    damped,
    par_noopt,
    lowerb,
    upperb,
    opt_crit,
    nmse,
    bounds,
    m,
    pnames,
    pnames2,
):
    """
    Build and solve the ETS optimization problem via `_ets.optimize`.

    Parameters
    ----------
    x0 : array
        Initial parameter vector = free smoothing params (subset of alpha/beta/gamma/phi)
        followed by the initial state vector.
    par : dict
        Only the optimizable parameters (non-NaN) from initparam.
    y : array
        Observations.
    nstate : int
        Length of the state vector.
    errortype, trendtype, seasontype : str
        Structure flags ("A","M","N").
    damped : bool
        Whether trend is damped.
    par_noopt : dict
        Original user-provided params (to freeze any non-NaN).
    lowerb, upperb : array
        Box bounds for the *free* prefix of x0 (smoothing params + states).
    opt_crit : str
        Objective ("lik","mse","amse","sigma","mae").
    nmse : int
        Horizon for AMSE.
    bounds : str
        Bound mode ("both","usual","admissible").
    m : int
        Seasonal period.
    pnames, pnames2 : iterable
        Keys for printing/debug purposes.

    Returns
    -------
    results(...)
        A small namedtuple-like object compatible with your test harness.
    """
    alpha = par_noopt["alpha"] if math.isnan(par["alpha"]) else par["alpha"]
    if math.isnan(alpha):
        raise ValueError("alpha problem!")
    if trendtype != "N":
        beta = par_noopt["beta"] if math.isnan(par["beta"]) else par["beta"]
        if math.isnan(beta):
            raise ValueError("beta problem!")
    else:
        beta = jnp.nan
    if seasontype != "N":
        gamma = par_noopt["gamma"] if math.isnan(par["gamma"]) else par["gamma"]
        if math.isnan(gamma):
            raise ValueError("gamma problem!")
    else:
        m = 1
        gamma = jnp.nan
    if damped:
        phi = par_noopt["phi"] if math.isnan(par["phi"]) else par["phi"]
        if math.isnan(phi):
            raise ValueError("phi problem!")
    else:
        phi = jnp.nan

    optAlpha = not math.isnan(float(alpha))
    optBeta = not math.isnan(float(beta))
    optGamma = not math.isnan(float(gamma))
    optPhi = not math.isnan(float(phi))

    if not math.isnan(par_noopt["alpha"]):
        optAlpha = False
    if not math.isnan(par_noopt["beta"]):
        optBeta = False
    if not math.isnan(par_noopt["gamma"]):
        optGamma = False
    if not math.isnan(par_noopt["phi"]):
        optPhi = False

    if not damped:
        phi = 1.0
    if trendtype == "N":
        beta = 0.0
    if seasontype == "N":
        gamma = 0.0

    print("Optimizing parameters:", {k: par[k] for k in pnames}, "with fixed params:", {k: par_noopt[k] for k in pnames2})
    opt_res = _ets.optimize(
        jnp.asarray(x0, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
        int(nstate),
        switch(errortype),
        switch(trendtype),
        switch(seasontype),
        switch_criterion(opt_crit),
        int(nmse),
        int(m),
        bool(optAlpha),
        bool(optBeta),
        bool(optGamma),
        bool(optPhi),
        float(alpha),
        float(beta),
        float(gamma),
        float(phi),
        jnp.asarray(lowerb, dtype=jnp.float64),
        jnp.asarray(upperb, dtype=jnp.float64),
        1e-4,
        1_000,
        True,
    )
    print("Optimization result:", opt_res)
    return results(
        x=jnp.asarray(opt_res.x),
        fn=float(opt_res.fun),
        nit=int(opt_res.nit),
        simplex=jnp.empty((0,), dtype=jnp.asarray(opt_res.x).dtype),
    )


def etsmodel(
    y: jnp.ndarray,
    m: int,
    errortype: str,
    trendtype: str,
    seasontype: str,
    damped: bool,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    opt_crit: str,
    nmse: int,
    bounds: str,
    maxit: int = 2_000,
    control=None,
    seed=None,
    trace: bool = False,
):
    """
    Fit a *single* ETS structure to the data and return a stats-like result dict.

    This function:
    1) initializes or accepts smoothing params via `initparam`,
    2) checks ranges/admissibility via `check_param`,
    3) builds initial states via `initstate`,
    4) optimizes smoothing+state vector with `_ets.optimize`,
    5) re-runs the rollout to compute likelihood, residuals, fitted values,
    6) computes information criteria (AIC, BIC, AICc), sigma2, and aggregates
       outputs in a dictionary.

    Returns
    -------
    dict
        Modeled fields include:
        - "loglik", "aic", "bic", "aicc", "mse", "amse", "sigma2"
        - "fit" (opt result), "residuals", "fitted"
        - "components" (e.g., "AAd" flags), "m", "nstate", "states"
        - "par" (alpha,beta,gamma,phi + initial states), "n_params"
    """
    if seasontype == "N":
        m = 1

    par_, lower, upper = initparam(
        alpha, beta, gamma, phi, trendtype, seasontype, damped,
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        m, bounds
    )
    par_noopt = dict(alpha=alpha, beta=beta, gamma=gamma, phi=phi)

    alpha = par_["alpha"] if not math.isnan(par_["alpha"]) else alpha
    beta = par_["beta"] if not math.isnan(par_["beta"]) else beta
    gamma = par_["gamma"] if not math.isnan(par_["gamma"]) else gamma
    phi = par_["phi"] if not math.isnan(par_["phi"]) else phi

    if not check_param(alpha, beta, gamma, phi,
                       jnp.asarray(lower, dtype=jnp.float64),
                       jnp.asarray(upper, dtype=jnp.float64),
                       bounds, m):
        raise Exception("Parameters out of range")

    # initialize state
    init_state = initstate(jnp.asarray(y, dtype=jnp.float64), m, trendtype, seasontype)
    nstate = int(init_state.shape[0])

    par_clean = {key: val for key, val in par_.items() if not math.isnan(val)}
    par_vec = jnp.full((len(par_clean) + nstate,), jnp.nan, dtype=jnp.float64)
    if len(par_clean) > 0:
        par_vec = par_vec.at[: len(par_clean)].set(jnp.asarray(list(par_clean.values()), dtype=jnp.float64))
    par_vec = par_vec.at[len(par_clean):].set(init_state)

    lower_ = jnp.full_like(par_vec, -jnp.inf)
    upper_ = jnp.full_like(par_vec, jnp.inf)
    j = 0
    for i, pr in enumerate(["alpha", "beta", "gamma", "phi"]):
        if pr in par_clean.keys():
            lower_ = lower_.at[j].set(float(lower[i]))
            upper_ = upper_.at[j].set(float(upper[i]))
            j += 1
    lower_box = lower_
    upper_box = upper_

    np_ = int(par_vec.shape[0])
    if np_ >= len(y) - 1:
        return dict(
            aic=jnp.inf,
            bic=jnp.inf,
            aicc=jnp.inf,
            mse=jnp.inf,
            amse=jnp.inf,
            fit=None,
            par=par_vec,
            states=init_state,
        )
    print("Number of parameters to estimate:", np_)
    fred = optimize_ets_target_fn(
        x0=par_vec,
        par=par_clean,
        y=jnp.asarray(y, dtype=jnp.float64),
        nstate=nstate,
        errortype=errortype,
        trendtype=trendtype,
        seasontype=seasontype,
        damped=damped,
        par_noopt=par_noopt,
        lowerb=lower_box,
        upperb=upper_box,
        opt_crit=opt_crit,
        nmse=nmse,
        bounds=bounds,
        m=m,
        pnames=par_clean.keys(),
        pnames2=par_noopt.keys(),
    )
    fit_par = jnp.asarray(fred.x)
    init_state_fit = fit_par[-nstate:]
    if seasontype != "N":
        tail = m * (seasontype == "M") - jnp.sum(init_state_fit[(1 + (trendtype != "N")) : nstate])
        init_state_fit = jnp.hstack([init_state_fit, jnp.array([tail], dtype=jnp.float64)])

    j = 0
    if not jnp.isnan(fit_par[j]):
        alpha = float(fit_par[j]); j += 1
    if trendtype != "N":
        if not jnp.isnan(fit_par[j]):
            beta = float(fit_par[j])
        j += 1
    if seasontype != "N":
        if not jnp.isnan(fit_par[j]):
            gamma = float(fit_par[j])
        j += 1
    if damped:
        if not jnp.isnan(fit_par[j]):
            phi = float(fit_par[j])

    amse, e, states, lik = pegelsresid_C(
        y=jnp.asarray(y, dtype=jnp.float64),
        m=m,
        init_state=init_state_fit,
        errortype=errortype,
        trendtype=trendtype,
        seasontype=seasontype,
        damped=damped,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        phi=phi,
        nmse=nmse,
    )
    print("Final likelihood:", lik)
    np_ = np_ + 1
    ny = len(y)
    aic = float(lik) + 2 * np_
    bic = float(lik) + math.log(ny) * np_
    if ny - np_ - 1 != 0.0:
        aicc = aic + 2 * np_ * (np_ + 1) / (ny - np_ - 1)
    else:
        aicc = jnp.inf

    mse = float(amse[0])
    amse_mean = float(jnp.mean(amse))

    fit_par_full = jnp.concatenate([jnp.array([alpha, beta, gamma, phi], dtype=jnp.float64), init_state_fit])
    if errortype == "A":
        fits = jnp.asarray(y) - e
    else:
        # protect e == -1
        aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
        fits = jnp.asarray(y) / (1.0 + aux_e)

    sq_e = e * e
    finites = ~jnp.isinf(sq_e)
    sigma2 = float(jnp.sum(sq_e[finites]) / (ny - np_ - 1))

    return dict(
        loglik=-0.5 * float(lik),
        aic=aic,
        bic=bic,
        aicc=float(aicc),
        mse=mse,
        amse=amse_mean,
        fit=fred,
        residuals=e,
        components=f"{errortype}{trendtype}{seasontype}{'D' if damped else 'N'}",
        m=m,
        nstate=nstate,
        fitted=fits,
        states=states,
        par=fit_par_full,
        sigma2=sigma2,
        n_params=np_,
    )


def is_constant(x):
    """
    Quick check for a constant series.

    Returns
    -------
    bool
        True if all entries equal the first entry.
    """
    x = jnp.asarray(x)
    return bool(jnp.all(x[0] == x))


def ets_f(
    y,
    m,
    model="ZZZ",
    damped=None,
    alpha=None,
    beta=None,
    gamma=None,
    phi=None,
    additive_only=None,
    blambda=None,
    biasadj=None,
    lower=None,
    upper=None,
    opt_crit="lik",
    nmse=3,
    bounds="both",
    ic="aicc",
    restrict=True,
    allow_multiplicative_trend=False,
    use_initial_values=False,
    maxit=2_000,
):
    """
    Top-level ETS interface: model selection + fitting.

    Parameters
    ----------
    y : array-like
        Time series (float64).
    m : int
        Seasonal period.
    model : str or dict
        - If str like "ZZZ", expands to grids:
          error in {"A","M"}, trend in {"N","A"[,"M" if allowed]}, season in {"N","A","M"},
          damped in {True, False} if None.
        - If dict (a previously-fitted object), forecast forward using stored params.
    damped : Optional[bool]
        If None, both damped and undamped are tried for trend != "N".
    alpha, beta, gamma, phi : Optional[float]
        Provide to fix values (set the others to NaN to optimize).
    additive_only : Optional[bool]
        If True, forbids multiplicative forms.
    blambda, biasadj : unused
        Not implemented here (Box-Cox/bias adjustment).
    lower, upper : Optional[array]
        Bounds for (alpha, beta, gamma, phi); defaults applied if None.
    opt_crit : {"lik","mse","amse","sigma","mae"}
        Optimization criterion.
    nmse : int
        Horizon for AMSE tracking (1..30).
    bounds : {"both","usual","admissible"}
        Bound/admissibility behavior.
    ic : {"aicc","aic","bic"}
        Information criterion used to pick the best among candidates.
    restrict : bool
        Apply standard ETS combination restrictions (forbid certain mixes).
    allow_multiplicative_trend : bool
        If True and model="ZZZ", includes "M" trend in the grid.
    use_initial_values : bool
        Reserved; not used here.
    maxit : int
        Maximum iterations budget (passed to optimizer).

    Returns
    -------
    dict
        Best fitted model dictionary (see `etsmodel` return schema) with an
        added "method" key like "ETS(A,Ad,M)".

    Notes
    -----
    - When `model` is a dict, this function acts as a *forward* function (no re-fit).
    - When `y` is constant, it falls back to ANN with alpha≈1 to mimic
      standard implementations.
    """
    y = jnp.asarray(y, dtype=jnp.float64)

    if alpha is None:
        alpha = jnp.nan
    if beta is None:
        beta = jnp.nan
    if gamma is None:
        gamma = jnp.nan
    if phi is None:
        phi = jnp.nan
    if blambda is not None:
        raise NotImplementedError("`blambda` not None")
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    if lower is None:
        lower = jnp.array([0.0001, 0.0001, 0.0001, _PHI_LOWER], dtype=jnp.float64)
    else:
        lower = jnp.asarray(lower, dtype=jnp.float64)
    if upper is None:
        upper = jnp.array([0.9999, 0.9999, 0.9999, _PHI_UPPER], dtype=jnp.float64)
    else:
        upper = jnp.asarray(upper, dtype=jnp.float64)
    if jnp.any(upper < lower):
        raise ValueError("Lower limits must be less than upper limits")

    if is_constant(y):
        return etsmodel(
            y=y,
            m=m,
            errortype="A",
            trendtype="N",
            seasontype="N",
            alpha=0.9999,
            beta=beta,
            gamma=gamma,
            phi=phi,
            damped=False,
            lower=lower,
            upper=upper,
            opt_crit=opt_crit,
            nmse=nmse,
            bounds=bounds,
            maxit=maxit,
        )

    if isinstance(model, dict):
        m = model["m"]
        errortype, trendtype, seasontype = model["components"][:3]
        damped = model["components"][3] != "N"
        alpha, beta, gamma, phi = model["par"][:4]
        init_state = model["par"][4:]
        amse, e, states, lik = pegelsresid_C(
            y=y,
            m=m,
            init_state=init_state,
            errortype=errortype,
            trendtype=trendtype,
            seasontype=seasontype,
            damped=damped,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            phi=phi,
            nmse=nmse,
        )
        fred = model["fit"]
        nstate = len(init_state)
        np_ = model["n_params"] - 1
        np_ = np_ + 1
        ny = len(y)
        aic = float(lik) + 2 * np_
        bic = float(lik) + math.log(ny) * np_
        if ny - np_ - 1 != 0.0:
            aicc = aic + 2 * np_ * (np_ + 1) / (ny - np_ - 1)
        else:
            aicc = jnp.inf

        mse = float(amse[0])
        amse_mean = float(jnp.mean(amse))

        fit_par = jnp.concatenate([jnp.array([alpha, beta, gamma, phi], dtype=jnp.float64), init_state])
        if errortype == "A":
            fits = y - e
        else:
            aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
            fits = y / (1 + aux_e)
        sq_e = e * e
        finites = ~jnp.isinf(sq_e)
        sigma2 = float(jnp.sum(sq_e[finites]) / (ny - np_ - 1))

        return dict(
            loglik=-0.5 * float(lik),
            aic=aic,
            bic=bic,
            aicc=float(aicc),
            mse=mse,
            amse=amse_mean,
            fit=fred,
            residuals=e,
            components=f"{errortype}{trendtype}{seasontype}{'D' if damped else 'N'}",
            m=m,
            nstate=nstate,
            fitted=fits,
            states=states,
            par=fit_par,
            sigma2=sigma2,
            n_params=np_,
        )

    errortype, trendtype, seasontype = model
    print("Assessing model",errortype, trendtype, seasontype)
    if errortype not in ["M", "A", "Z"]:
        raise ValueError("Invalid error type")
    if trendtype not in ["N", "A", "M", "Z"]:
        raise ValueError("Invalid trend type")
    if seasontype not in ["N", "A", "M", "Z"]:
        raise ValueError("Invalid season type")
    if m < 1 or len(y) <= m:
        seasontype = "N"
    if m == 1:
        if seasontype in ("A", "M"):
            raise ValueError("Nonseasonal data")
        else:
            seasontype = "N"
    if restrict:
        if (
            (errortype == "A" and (trendtype == "M" or seasontype == "M"))
            or (errortype == "M" and trendtype == "M" and seasontype == "A")
            or (
                additive_only
                and (errortype == "M" or trendtype == "M" or seasontype == "M")
            )
        ):
            raise ValueError("Forbidden model combination")
    data_positive = float(jnp.min(y)) > 0
    if (not data_positive) and errortype == "M":
        raise ValueError("Inappropriate model for data with negative or zero values")
    if damped is not None:
        if damped and trendtype == "N":
            ValueError("Forbidden model combination")
    n = len(y)
    npars = 2
    if trendtype in ["A", "M"]:
        npars += 2
    if seasontype in ["A", "M"]:
        npars += 2
    if damped is not None:
        npars += int(bool(damped))
    if n <= npars + 4:
        raise NotImplementedError("tiny datasets")

    if errortype == "Z":
        errortype = ["A", "M"]
    if trendtype == "Z":
        trendtype = ["N", "A"] + (["M"] if allow_multiplicative_trend else [])
    if seasontype == "Z":
        seasontype = ["N", "A", "M"]
    if damped is None:
        damped = [True, False]
    else:
        damped = [damped]

    best_ic = jnp.inf
    best = None
    print("Fitting models:", "error type", errortype, "trend type", trendtype, "season type", seasontype, "damped", damped)
    for etype in errortype:
        for ttype in trendtype:
            for stype in seasontype:
                for dtype in damped:
                    if ttype == "N" and dtype:
                        continue
                    if restrict:
                        if etype == "A" and (ttype == "M" or stype == "M"):
                            continue
                        if etype == "M" and ttype == "M" and stype == "A":
                            continue
                        if additive_only and (
                            etype == "M" or ttype == "M" or stype == "M"
                        ):
                            continue
                    if (not data_positive) and etype == "M":
                        continue
                    if (not data_positive) and stype == "M":
                        continue
                    if stype != "N" and m == 1:
                        continue
                    fit = etsmodel(
                        y,
                        m,
                        etype,
                        ttype,
                        stype,
                        dtype,
                        alpha,
                        beta,
                        gamma,
                        phi,
                        lower=lower,
                        upper=upper,
                        opt_crit=opt_crit,
                        nmse=nmse,
                        bounds=bounds,
                        maxit=maxit,
                    )
                    fit_ic = fit[ic]
                    if not math.isnan(float(fit_ic)):
                        if float(fit_ic) < float(best_ic):
                            best = fit
                            best_ic = fit_ic
                            best_e = etype
                            best_t = ttype
                            best_s = stype
                            best_d = dtype
    if best is None or jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")
    best["method"] = f"ETS({best_e},{best_t}{'d' if best_d else ''},{best_s})"
    print("Selected model:", best["method"], "with", ic,",", "AICc:", best["aicc"])
    return best


def pegelsfcast_C(h, obj, npaths=None, level=None, bootstrap=None):
    """
    One-step call to produce the mean forecast path from a fitted model dict.

    Parameters
    ----------
    h : int
        Horizon.
    obj : dict
        Fitted model dictionary from `etsmodel` / `ets_f`.
    npaths, level, bootstrap : unused
        Present for interface parity.

    Returns
    -------
    jnp.ndarray
        Mean forecast of length h.
    """
    states = jnp.asarray(obj["states"][-1, :], dtype=jnp.float64)
    etype, ttype, stype = [switch(comp) for comp in obj["components"][:3]]
    phi = 1.0 if obj["components"][3] == "N" else float(obj["par"][3])
    m = int(obj["m"])

    # allocate and CAPTURE the result of etsforecast
    forecast = jnp.full((h,), jnp.nan, dtype=states.dtype)
    forecast = etsforecast(
        x=states, m=m, trend=ttype, season=stype, phi=phi, h=h, f=forecast
    )
    return forecast


def _compute_sigmah(pf, h, sigma, cvals):
    """
    Helper for multiplicative-error variance recursion used in intervals.

    Parameters
    ----------
    pf : jnp.ndarray
        Point forecasts (length h).
    h : int
        Horizon.
    sigma : float
        Innovation variance estimate (sigma^2).
    cvals : jnp.ndarray
        Coefficients per step capturing linearization terms.

    Returns
    -------
    jnp.ndarray
        sigma_h (length h) used to scale interval widths.
    """
    theta = jnp.full((h,), jnp.nan)
    theta = theta.at[0].set(pf[0] ** 2)

    for k in range(1, h):
        sum_val = jnp.dot(cvals[:k] ** 2, theta[:k][::-1])
        theta = theta.at[k].set(pf[k] ** 2 + sigma * sum_val)

    return (1 + sigma) * theta - pf**2


def _class3models(
    h,
    sigma,
    last_state,
    season_length,
    error,
    trend,
    seasonality,
    damped,
    alpha,
    beta,
    gamma,
    phi,
):
    """
    Analytical variance for Class 3 (multiplicative seasonality) ETS cases.

    Returns
    -------
    jnp.ndarray
        Variance per horizon step (length h).
    """
    damped_val = (damped != "N")
    p = last_state.shape[0]

    H1 = jnp.array([[1, 1]], dtype=jnp.float64) if trend != "N" else jnp.array([[1]], dtype=jnp.float64)
    H2 = jnp.concatenate([jnp.zeros(season_length - 1), jnp.array([1.0])]).reshape(1, season_length)

    if trend == "N":
        F1 = jnp.array(1.0)
        G1 = jnp.array(alpha)
    else:
        f1 = phi if damped_val else 1.0
        F1 = jnp.array([[1.0, 1.0], [0.0, f1]], dtype=jnp.float64)
        G1 = jnp.array([[alpha, alpha], [beta, beta]], dtype=jnp.float64)

    f2_top = jnp.concatenate([jnp.zeros(season_length - 1), jnp.array([1.0])]).reshape(1, season_length)
    f2_bottom = jnp.c_[jnp.identity(season_length - 1), jnp.zeros((season_length - 1, 1))]
    F2 = jnp.r_[f2_top, f2_bottom]

    G2 = jnp.zeros((season_length, season_length))
    G2 = G2.at[0, season_length - 1].set(gamma)

    Mh = jnp.matmul(
        last_state[0 : (p - season_length)].reshape(-1, 1),
        last_state[(p - season_length) : p].reshape(1, season_length),
    )
    Vh = jnp.zeros((Mh.size, Mh.size))
    H21 = jnp.kron(H2, H1)
    F21 = jnp.kron(F2, F1)
    G21 = jnp.kron(G2, G1)
    K = jnp.kron(G2, F1) + jnp.kron(F2, G1)
    mu = jnp.zeros(h)
    var = jnp.zeros(h)

    for i in range(h):
        mu = mu.at[i].set((H1 @ (Mh @ H2.T)).item())
        var = var.at[i].set(((1 + sigma) * (H21 @ (Vh @ H21.T))).item() + sigma * float(mu[i] ** 2))
        vecMh = Mh.flatten()
        exp1 = F21 @ (Vh @ F21.T)
        exp2 = F21 @ (Vh @ G21.T)
        exp3 = G21 @ (Vh @ F21.T)
        exp4 = K @ ((Vh + (vecMh * vecMh.reshape(vecMh.shape[0], 1))) @ K.T)
        exp5 = (sigma * G21) @ ((3 * Vh + 2 * vecMh * vecMh.reshape(vecMh.shape[0], 1)) @ G21.T)
        Vh = exp1 + sigma * (exp2 + exp3 + exp4 + exp5)

    if trend == "N":
        Mh = F1 * (Mh @ F2.T) + G1 * (Mh @ G2.T) * sigma
    else:
        Mh = F1 @ (Mh @ F2.T) + G1 @ (Mh @ G2.T) * sigma

    return var


def _compute_pred_intervals(model: Dict[str, Any], forecasts: Dict[str, jnp.ndarray], h: int, level):
    """
    Compute prediction intervals for ETS forecasts.

    Uses closed-form expressions for Classes 1–3 where available, and falls
    back to simulation for Classes 4–5 (mixtures/multiplicative trickier cases).

    Parameters
    ----------
    model : dict
        Fitted model dictionary (from `etsmodel` / `ets_f`).
    forecasts : dict
        Must include "mean" forecast path.
    h : int
        Horizon.
    level : list[int]
        Confidence levels, e.g., [80, 95].

    Returns
    -------
    dict
        Keys: "lo-<level>", "hi-<level>" arrays aligned with horizon.
    """
    sigma = float(model["sigma2"])
    season_length = int(model["m"])
    pf = forecasts["mean"]

    model_type = model["components"]
    steps = jnp.arange(1, h + 1)
    hm = jnp.floor((h - 1) / season_length)
    last_state = jnp.asarray(model["states"][-1])

    # error, trend, and seasonality type
    error = model_type[0]
    trend = model_type[1]
    seasonality = model_type[2]
    damped = model_type[3]

    # parameters
    alpha = float(model["par"][0])
    beta = float(model["par"][1])
    gamma = float(model["par"][2])
    phi = float(model["par"][3])

    exp1 = alpha**2 + alpha * beta * steps + (1 / 6) * beta**2 * steps * (2 * steps - 1)
    exp2 = (beta * phi * steps) / (1 - phi) ** 2
    exp3 = 2 * alpha * (1 - phi) + beta * phi
    exp4 = (beta * phi * (1 - phi**steps)) / ((1 - phi) ** 2 * (1 - phi**2))
    exp5 = 2 * alpha * (1 - phi**2) + beta * phi * (1 + 2 * phi - phi**steps)

    compute_intervals = True
    # Class 1 models
    if error == "A" and trend == "N" and seasonality == "N" and damped == "N":
        sigmah = 1 + alpha**2 * (steps - 1)
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "N" and damped == "N":
        sigmah = 1 + (steps - 1) * exp1
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "N" and damped == "D":
        sigmah = 1 + alpha**2 * (steps - 1) + exp2 * exp3 - exp4 * exp5
        sigmah = sigma * sigmah

    elif error == "A" and trend == "N" and seasonality == "A" and damped == "N":
        sigmah = 1 + alpha**2 * (steps - 1) + gamma * hm * (2 * alpha + gamma)
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "A" and damped == "N":
        exp6 = 2 * alpha + gamma + beta * season_length * (hm + 1)
        sigmah = 1 + (steps - 1) * exp1 + gamma * hm * exp6
        sigmah = sigma * sigmah

    elif error == "A" and trend == "A" and seasonality == "A" and damped == "D":
        exp7 = (2 * beta * gamma * phi) / ((1 - phi) * (1 - phi**season_length))
        exp8 = hm * (1 - phi**season_length) - (phi**season_length) * (1 - phi ** (season_length * hm))
        sigmah = 1 + alpha**2 * (steps - 1) + exp2 * exp3 - exp4 * exp5 + gamma * hm * (2 * alpha + gamma) + exp7 * exp8
        sigmah = sigma * sigmah

    # Class 2 models
    elif error == "M" and trend == "N" and seasonality == "N" and damped == "N":
        cvals = jnp.full((h,), alpha)
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "N" and damped == "N":
        cvals = alpha + beta * steps
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "N" and damped == "D":
        cvals = jnp.full((h,), jnp.nan)
        def sum_phi_k(k):
            j = jnp.arange(1, k + 1)
            return jnp.sum(phi**j)
        cvals = cvals.at[:].set(alpha + beta * jnp.array([sum_phi_k(k) for k in range(1, h + 1)]))
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "N" and seasonality == "A" and damped == "N":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        cvals = alpha + gamma * dvals
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "A" and damped == "N":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        cvals = alpha * beta * steps + gamma * dvals
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and trend == "A" and seasonality == "A" and damped == "D":
        dvals = jnp.where(jnp.arange(1, h + 1) % season_length == 0, 1.0, 0.0)
        def sum_phi_k(k):
            j = jnp.arange(1, k + 1)
            return jnp.sum(phi**j)
        cvals = jnp.array([alpha + beta * sum_phi_k(k) + gamma * dvals[k - 1] for k in range(1, h + 1)], dtype=jnp.float64)
        sigmah = _compute_sigmah(pf, h, sigma, cvals)

    elif error == "M" and seasonality == "M":
        sigmah = _class3models(
            h, sigma, last_state, season_length, error, trend, seasonality, damped, alpha, beta, gamma, phi
        )

    else:
        # Classes 4 and 5 models: simulation approach
        compute_intervals = False
        nsim = 5000
        y_path = jnp.zeros((nsim, h), dtype=jnp.float64)

        # Fill NaNs as in original
        beta_sim = 0.0 if math.isnan(beta) else beta
        gamma_sim = 0.0 if math.isnan(gamma) else gamma
        phi_sim = 0.0 if math.isnan(phi) else phi

        key = jrand.PRNGKey(1)
        e = jrand.normal(key, shape=(nsim, h)) * math.sqrt(sigma)

        def run_sim(k):
            yhat = jnp.zeros((h,), dtype=jnp.float64)
            etssimulate(
                last_state,
                season_length,
                switch(error),
                switch(trend),
                switch(seasonality),
                alpha,
                beta_sim,
                gamma_sim,
                phi_sim,
                h,
                yhat,
                e[k],
            )
            return yhat

        y_path = jnp.stack([run_sim(k) for k in range(nsim)], axis=0)

        lower_q = 0.5 - jnp.asarray(level) / 200.0
        upper_q = 0.5 + jnp.asarray(level) / 200.0
        lower = jnp.quantile(y_path, lower_q.reshape(-1, 1), axis=0)
        upper = jnp.quantile(y_path, upper_q.reshape(-1, 1), axis=0)
        pi = {
            **{f"lo-{int(lv)}": lower[i] for i, lv in enumerate(level)},
            **{f"hi-{int(lv)}": upper[i] for i, lv in enumerate(level)},
        }

    if compute_intervals:
        pi = _calculate_intervals(forecasts, level=level, h=h, sigmah=jnp.sqrt(sigmah))

    return pi


def forecast_ets(obj, h, level=None):
    """
    Convenience wrapper: produce forecasts (and optional PI) from fitted model.

    Parameters
    ----------
    obj : dict
        Fitted model dictionary returned by `ets_f`/`etsmodel`.
    h : int
        Horizon.
    level : Optional[list[int]]
        Confidence levels (e.g., [80, 95]) for prediction intervals.

    Returns
    -------
    dict
        Keys: "mean", "residuals", "fitted", and optionally "lo-XX"/"hi-XX".
    """
    fcst = pegelsfcast_C(h, obj)
    out = {"mean": fcst}
    out["residuals"] = obj["residuals"]
    out["fitted"] = obj["fitted"]
    if level is not None:
        pi = _compute_pred_intervals(model=obj, forecasts=out, level=level, h=h)
        out = {**out, **pi}
    return out


def forward_ets(fitted_model, y):
    """
    Reuse a previously fitted ETS model dictionary to roll forward on new data.

    Parameters
    ----------
    fitted_model : dict
        Output of `ets_f` / `etsmodel` with learned params & states.
    y : array-like
        New time series segment to append/continue the fit.

    Returns
    -------
    dict
        New fitted model dict (same schema), reusing structure & params.
    """
    return ets_f(y=y, m=fitted_model["m"], model=fitted_model)
