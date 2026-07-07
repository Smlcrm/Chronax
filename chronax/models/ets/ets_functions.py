"""High-level ETS (Exponential Smoothing) model-selection and fitting API in JAX.

This module wires together the following components into a single entry-point
(:func:`ets_f`) that performs automatic or fixed-specification ETS fitting:

* **Low-level kernels** — state update, forecast, and rollout routines from
  :mod:`ets_backend` (imported as ``_ets``).
* **Parameter initialisation** — :func:`initparam` picks sensible starting
  values for (α, β, γ, φ) and adjusts box bounds.
* **State initialisation** — :func:`initstate` seeds level / trend / seasonal
  states via OLS regression, Fourier decomposition, or seasonal averages.
* **Admissibility** — :func:`check_param` and :func:`admissible` enforce
  classical ETS constraints and characteristic-polynomial root checks.
* **Model selection** — :func:`ets_f` fits every candidate
  (error, trend, season, damped) tuple via :func:`etsmodel` and selects the
  winner by AICc (or AIC / BIC) with a NaN/validity-masked ``jnp.argmin``.
* **Forecasting** — :func:`forecast_ets` / :func:`pegelsfcast_C` produce
  mean paths, while :func:`_compute_pred_intervals` adds analytical or
  simulation-based prediction intervals.

vmap contract (CLAUDE.md §1 rule 2): ``ets_f`` and the point-forecast path
(:func:`ets_point_forecast`, :func:`ets_forward_stacked`) are natively
traceable under ``jax.vmap`` — candidate structure is *config/shape-static*
(a Python loop over static specs), while the winner is a traced
``jnp.argmin`` index resolved with ``jnp.take``. Only the *native-interval*
machinery (:func:`_compute_pred_intervals`, :func:`ets_winner_view` with
more than one candidate) is eager-only, because interval formulas dispatch
on the winner's class string; the conformal path never needs it.
"""

__all__ = [
    'ets_f',
    'ets_point_forecast',
    'ets_winner_view',
    'ets_forward_stacked',
    'forecast_ets',
    'forward_ets',
]

import math
from functools import partial
from typing import Dict, Any, Optional
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

from . import ets_backend as _ets
from chronax.utils import _calculate_intervals, results

# Global variables
_smalno = jnp.finfo(float).eps
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98
_AICC_RATIO_THRESHOLD = 40.0
EPS = 1e-3
EPS_PURE = 2e-2


def _aicc(aic: float, n: int, k: int) -> float:
    """Small-sample corrected AIC (AICc) with extra penalty for tiny datasets.

    Parameters
    ----------
    aic : float
        Uncorrected AIC value.
    n : int
        Number of observations.
    k : int
        Number of free parameters (including σ²).

    Returns
    -------
    float
        AICc value; ``inf`` when the correction denominator is non-positive.
    """
    if k <= 0:
        return aic
    if (n / k) < _AICC_RATIO_THRESHOLD:
        denom = n - k - 1
        if denom <= 0:
            return float(jnp.inf)
        base_aicc = aic + 2.0 * k * (k + 1) / denom
    else:
        base_aicc = aic
    if n < 150:
        extra_penalty = k * (150 - n) / 150.0
        base_aicc = base_aicc + extra_penalty
    return base_aicc


def _prepare_init_state(
    y: jnp.ndarray, m: int, trendtype: str, seasontype: str
) -> jnp.ndarray:
    """
    Build initial state vector (level [+ trend] [+ season]) and append the
    seasonal balancing term for multiplicative seasonality.
    """
    init_state = initstate(jnp.asarray(y, dtype=jnp.float64), m, trendtype, seasontype)
    if seasontype == "N":
        return init_state
    nstate = int(init_state.shape[0])
    start = 1 + int(trendtype != "N")
    tail = m * (seasontype == "M") - jnp.sum(init_state[start:nstate])
    return jnp.hstack([init_state, jnp.array([tail], dtype=jnp.float64)])


def _compute_ic(lik: jnp.ndarray, n: int, k: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute AIC, BIC, and AICc from the likelihood-style scalar *lik*.

    *lik* may be a traced jnp scalar (the fit runs under vmap in the
    conformal CV path); *n* and *k* are shape/config-static ints, so all
    branching happens on statics and the returned ICs stay traced.

    Parameters
    ----------
    lik : jnp.ndarray
        Gaussian log-likelihood proxy (as returned by the rollout).
    n : int
        Number of observations.
    k : int
        Number of free parameters (including σ²).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ``(AIC, BIC, AICc)`` as jnp scalars.
    """
    aic = lik + 2.0 * k
    bic = lik + math.log(n) * k
    aicc = _aicc(aic=aic, n=n, k=k)
    return aic, bic, aicc


def _inv_sigmoid_scaled(val: float, low: float, high: float) -> float:
    """Inverse of the legacy scaled-sigmoid transform: constrained → unconstrained."""
    z = (val - low) / (high - low)
    z = jnp.clip(z, 1e-4, 1.0 - 1e-4)
    return float(jnp.log(z / (1.0 - z)))


@partial(jax.jit, static_argnames=("m", "error", "trend", "season", "h"))
def _etssimulate_jit(
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
    e: jnp.ndarray,
) -> jnp.ndarray:
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
    if m > 24 and season != _ets.Component.Nothing:
        return jnp.zeros((h,), dtype=x.dtype)
    elif m < 1:
        m = 1
    dt = x.dtype
    oldb = jnp.asarray(0.0, dtype=dt)
    # Copy initial state components
    l = jnp.asarray(x[0], dtype=dt)
    if trend != _ets.Component.Nothing:
        b = jnp.asarray(x[1], dtype=dt)
    else:
        b = jnp.asarray(0.0, dtype=dt)
    if season != _ets.Component.Nothing:
        # x offset = 1 + (trend != Nothing)
        off = 1 + int(trend != _ets.Component.Nothing)
        s = jnp.asarray(x[off : off + m], dtype=dt)
    else:
        s = jnp.zeros((m,), dtype=dt)

    y = jnp.zeros((h,), dtype=dt)

    def step(i: int, carry: tuple[Any, ...]) -> tuple[Any, ...]:
        """Advance one simulated step for a single ETS sample path."""
        l, b, s, y, alive = carry

        def do_alive(carry_inner: tuple[Any, ...]) -> tuple[Any, ...]:
            """Update the simulated state while the path remains valid."""
            l_i, b_i, s_i, y_i, _ = carry_inner
            oldl = l_i
            oldb = b_i
            olds = s_i
            f = _ets.forecast(
                jnp.zeros((1,), dtype=dt),
                oldl,
                oldb,
                olds,
                m,
                trend,
                season,
                jnp.asarray(phi, dtype=dt),
                1,
            )
            invalid = jnp.abs(f[0] - _ets.NA) < _ets.TOL

            def on_invalid(args: tuple[Any, ...]) -> tuple[Any, ...]:
                """Mark the simulated path as invalid and stop future updates."""
                l_j, b_j, s_j, y_j = args
                y_j = y_j.at[0].set(_ets.NA)
                return l_j, b_j, s_j, y_j, False

            def on_valid(args: tuple[Any, ...]) -> tuple[Any, ...]:
                """Apply the normal simulation update for a valid forecast step."""
                l_j, b_j, s_j, y_j = args
                if error == _ets.Component.Additive:
                    y_val = f[0] + e[i]
                else:
                    y_val = f[0] * (1.0 + e[i])
                y_j = y_j.at[i].set(y_val)
                l_new, b_new, s_new = _ets.update(
                    s_j,
                    l_j,
                    b_j,
                    oldl,
                    oldb,
                    olds,
                    m,
                    trend,
                    season,
                    int(error.value),
                    jnp.asarray(alpha, dtype=dt),
                    jnp.asarray(beta, dtype=dt),
                    jnp.asarray(gamma, dtype=dt),
                    jnp.asarray(phi, dtype=dt),
                    y_val,
                )
                return l_new, b_new, s_new, y_j, True

            return lax.cond(invalid, on_invalid, on_valid, (l_i, b_i, s_i, y_i))

        def do_dead(carry_inner: tuple[Any, ...]) -> tuple[Any, ...]:
            """Keep the simulated path unchanged after it has terminated."""
            return carry_inner

        return lax.cond(alive, do_alive, do_dead, carry)

    l, b, s, y, _ = lax.fori_loop(0, h, step, (l, b, s, y, True))
    return y


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
) -> jnp.ndarray:
    """Simulate *h*-step future sample paths from a given ETS state.

    Thin convenience wrapper around the JIT-compiled
    :func:`_etssimulate_jit`; the ``y`` argument is unused (output is
    returned from the inner function).

    Parameters
    ----------
    x : jnp.ndarray
        State vector (level [+ trend] [+ seasonal]).
    m : int
        Seasonal period.
    error, trend, season : _ets.Component
        Model structure flags.
    alpha, beta, gamma, phi : float
        Smoothing parameters.
    h : int
        Forecast horizon.
    y : jnp.ndarray
        Pre-allocated output buffer (*unused* — kept for API parity).
    e : jnp.ndarray
        Innovation draws of length ``h`` (e.g. from ``N(0, σ)``).

    Returns
    -------
    jnp.ndarray
        Simulated future path of length ``h``.
    """
    y = _etssimulate_jit(
        x,
        m,
        error,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        h,
        e,
    )
    return y


@partial(jax.jit, static_argnames=("m", "trend", "season", "h"))
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

    l = jnp.asarray(x[0], dtype=dt)
    has_trend = (trend != _ets.Component.Nothing)
    b = jnp.asarray(x[1], dtype=dt) if has_trend else jnp.asarray(0.0, dtype=dt)

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
        phi=jnp.asarray(phi, dtype=dt),
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
) -> tuple[dict[str, float], jnp.ndarray, jnp.ndarray]:
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
        upper = upper.at[:3].set(1.0)
    elif jnp.any(lower > upper):
        raise Exception("Inconsistent parameter boundaries")

    # select alpha
    if math.isnan(alpha):
        alpha = 0.1 if m > 1 else 0.2
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
        gamma = 0.05
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


def admissible(alpha: float, beta: float, gamma: float, phi: float, m: int) -> bool:
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
) -> bool:
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


def _admissible_root_ok(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    m: int,
) -> jnp.ndarray:
    """Seasonal-model stability check via companion-matrix eigenvalues (traced).

    jnp mirror of the characteristic-polynomial root test inside
    :func:`admissible`. The polynomial is monic by construction (leading
    coefficient 1.0), so no trailing-zero trimming is needed and the
    companion matrix has a static shape — safe under jit/vmap.
    """
    P = jnp.zeros((m + 2,), dtype=jnp.float64)
    P = P.at[0].set(phi * (1.0 - alpha - gamma))
    P = P.at[1].set(alpha + beta - alpha * phi + gamma - 1.0)
    if m - 2 > 0:
        P = P.at[2:m].set(alpha + beta - alpha * phi)
    P = P.at[m].set(alpha + beta - phi)
    P = P.at[m + 1].set(1.0)
    n_deg = m + 1
    C = jnp.zeros((n_deg, n_deg), dtype=jnp.float64)
    C = C.at[1:, :-1].set(jnp.eye(n_deg - 1, dtype=jnp.float64))
    C = C.at[0, :].set(-P[:-1][::-1])
    roots = jnp.linalg.eigvals(C.astype(jnp.complex128))
    return jnp.max(jnp.abs(roots)) <= 1.0 + 1e-10


def _valid_params_jnp(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    gamma: jnp.ndarray,
    phi: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    bounds: str,
    m: int,
    damped: bool,
    has_beta: bool,
    has_gamma: bool,
) -> jnp.ndarray:
    """Traced mirror of :func:`check_param` + :func:`admissible` for post-fit params.

    The eager originals branch with ``math.isnan``/``float()`` on parameter
    *values*, which cannot run on optimizer output under vmap. Here the
    presence of each parameter is a *static* fact of the model structure
    (``damped``/``has_beta``/``has_gamma``), so all Python branching is on
    config and the returned validity is a traced boolean scalar. NaN
    parameters fail every comparison, so a diverged fit is invalid for free.
    """
    one = jnp.asarray(True)
    phi_e = phi if damped else jnp.asarray(1.0, dtype=jnp.float64)
    ok = one
    if bounds != "admissible":
        ok = ok & (alpha >= lower[0]) & (alpha <= upper[0])
        if has_beta:
            ok = ok & (beta >= lower[1]) & (beta <= alpha) & (beta <= upper[1])
        if damped:
            ok = ok & (phi_e >= lower[3]) & (phi_e <= upper[3])
        if has_gamma:
            ok = ok & (gamma >= lower[2]) & (gamma <= 1.0 - alpha) & (gamma <= upper[2])
    if bounds != "usual":
        ok = ok & (phi_e >= 0.0) & (phi_e <= 1.0 + 1e-8)
        if not has_gamma:
            ok = ok & (alpha >= 1.0 - 1.0 / phi_e) & (alpha <= 1.0 + 1.0 / phi_e)
            if has_beta:
                ok = ok & (beta >= alpha * (phi_e - 1.0)) & (beta <= (1.0 + phi_e) * (2.0 - alpha))
        elif m > 1:
            beta_e = beta if has_beta else jnp.asarray(0.0, dtype=jnp.float64)
            ok = ok & (gamma >= jnp.maximum(1.0 - 1.0 / phi_e - alpha, 0.0))
            ok = ok & (gamma <= 1.0 + 1.0 / phi_e - alpha)
            ok = ok & (alpha >= 1.0 - 1.0 / phi_e - gamma * (1.0 - m + phi_e + phi_e * m) / (2.0 * phi_e * m))
            ok = ok & (beta_e >= -(1.0 - phi_e) * (gamma / m + alpha))
            ok = ok & _admissible_root_ok(alpha, beta_e, gamma, phi_e, m)
    return ok


@partial(jax.jit, static_argnames=("m", "multiplicative"))
def _seasonal_decompose_jax(
    y: jnp.ndarray,
    m: int,
    multiplicative: bool,
) -> Dict[str, jnp.ndarray]:
    """Lightweight JAX-native seasonal decomposition for ETS state initialisation.

    Computes a centred moving-average trend proxy and derives seasonal
    patterns from the de-trended residuals (additive) or ratios
    (multiplicative).

    Parameters
    ----------
    y : jnp.ndarray
        Observed series.
    m : int
        Seasonal period.
    multiplicative : bool
        If ``True``, decompose as ``y / trend``; otherwise ``y − trend``.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``{"seasonal": …, "trend": …}`` arrays aligned with ``y``.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = y.shape[0]
    m_eff = max(int(m), 1)

    # Centered moving-average trend proxy
    w = jnp.ones((m_eff,), dtype=jnp.float64) / float(m_eff)
    trend = jnp.convolve(y, w, mode="same")
    if (m_eff % 2) == 0:
        trend = jnp.convolve(trend, jnp.array([0.5, 0.5], dtype=jnp.float64), mode="same")

    if multiplicative:
        trend_safe = jnp.where(jnp.abs(trend) > 1e-8, trend, 1e-8)
        detrended = y / trend_safe
    else:
        detrended = y - trend

    n_periods = n // m_eff
    n_full = n_periods * m_eff

    def _seasonal_from_matrix() -> jnp.ndarray:
        """Compute a repeated seasonal template from full seasonal blocks."""
        mat = detrended[:n_full].reshape((n_periods, m_eff))
        pat = jnp.mean(mat, axis=0)
        rep = jnp.tile(pat, n // m_eff + 1)[:n]
        if multiplicative:
            return rep / jnp.where(jnp.abs(jnp.mean(rep)) > 1e-8, jnp.mean(rep), 1.0)
        return rep - jnp.mean(rep)

    seasonal = lax.cond(
        n_periods >= 1,
        lambda _: _seasonal_from_matrix(),
        lambda _: jnp.ones((n,), dtype=jnp.float64) if multiplicative else jnp.zeros((n,), dtype=jnp.float64),
        operand=None,
    )
    return {"seasonal": seasonal, "trend": trend}


def initstate(y: jnp.ndarray, m: int, trendtype: str, seasontype: str) -> jnp.ndarray:
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
        elif n < 3 * m:  # small-n fallback: simple seasonal means
            idx = jnp.arange(n, dtype=jnp.int32) % int(m)
            sums = jnp.zeros((m,), dtype=jnp.float64).at[idx].add(y)
            counts = jnp.zeros((m,), dtype=jnp.float64).at[idx].add(1.0)
            pattern = sums / jnp.maximum(counts, 1.0)
            seasonal = jnp.take(pattern, idx)
            if seasontype == "A":
                seasonal = seasonal - jnp.mean(seasonal)
                y_d = {"seasonal": seasonal}
            else:
                # Multiplicative seasonality on zero/negative data cannot raise
                # under trace; the guarded divide keeps values finite and the
                # positivity mask in ets_f() invalidates such candidates.
                seasonal = seasonal / jnp.maximum(jnp.mean(seasonal), 1e-8)
                y_d = {"seasonal": seasonal}
        else:
            y_d = _seasonal_decompose_jax(y, m, seasontype == "M")
        init_seas = y_d["seasonal"][1:m][::-1]
        if n < 5 * m:
            if n < 2 * m:
                shrinkage = 0.5
            else:
                shrinkage = 0.5 + 0.5 * (n / (5.0 * m))
            if seasontype == "A":
                init_seas = init_seas * shrinkage
            else:
                init_seas = 1.0 + (init_seas - 1.0) * shrinkage
        if seasontype == "A":
            y_sa = y - y_d["seasonal"]
        else:
            init_seas = jnp.clip(init_seas, a_min=1e-2)
            init_seas = jnp.where(
                jnp.sum(init_seas) > m,
                init_seas / jnp.sum(init_seas + 1e-2),
                init_seas,
            )
            y_sa = y / jnp.clip(y_d["seasonal"], a_min=1e-2)
    else:
        m = 1
        init_seas = jnp.array([], dtype=jnp.float64)
        y_sa = y

    maxn = min(max(10, 2 * m), int(y_sa.shape[0]))
    if trendtype == "N":
        l0 = jnp.mean(y_sa[:maxn])
        return jnp.concatenate([l0[None], init_seas])
    else:
        X = jnp.full((n, 2), jnp.nan)
        X = X.at[:, 0].set(1.0)
        X = X.at[:, 1].set(jnp.arange(1, n + 1, dtype=jnp.float64))
        (l, b), *_ = jnp.linalg.lstsq(X[:maxn], y_sa[:maxn], rcond=-1.0)
        # The remaining safeguards branch on regression *values* (traced under
        # vmap) — expressed as jnp.where chains that preserve the original
        # sequential-mutation order exactly.
        if trendtype == "A":
            degenerate = jnp.abs(l + b) < 1e-8
            l0 = jnp.where(degenerate, l * (1.0 + 1e-3), l)
            b0 = jnp.where(degenerate, b * (1.0 - 1e-3), b)
        else:
            l0 = l + b
            l0 = jnp.where(jnp.abs(l0) < 1e-8, 1e-7, l0)
            b0 = (l + 2.0 * b) / l0
            div = jnp.where(jnp.abs(b0) <= 1e-8, 1e-8, b0)
            l0 = l0 / div
            b0 = jnp.where(jnp.abs(b0) > 1e10, jnp.sign(b0) * 1e10, b0)
            fallback = (l0 < 1e-8) | (b0 < 1e-8)
            div2 = jnp.where(jnp.abs(y_sa[0]) <= 1e-8, 1e-8, y_sa[0])
            l0 = jnp.where(fallback, jnp.maximum(y_sa[0], 1e-3), l0)
            b0 = jnp.where(fallback, jnp.maximum(y_sa[1] / div2, 1e-3), b0)
        return jnp.concatenate([jnp.stack([l0, b0]), init_seas])


def switch(x: str) -> _ets.Component:
    """Map a single-character component string to the :class:`ets_backend.Component` enum.

    Parameters
    ----------
    x : str
        One of ``"N"`` (Nothing), ``"A"`` (Additive), ``"M"`` (Multiplicative).

    Returns
    -------
    _ets.Component

    Raises
    ------
    ValueError
        If *x* is not a recognised flag.
    """
    if x == "N":
        return _ets.Component.Nothing
    if x == "A":
        return _ets.Component.Additive
    if x == "M":
        return _ets.Component.Multiplicative
    raise ValueError(f"Unknown component {x}")


def switch_criterion(x: str) -> _ets.Criterion:
    """Map an objective string to the :class:`ets_backend.Criterion` enum.

    Parameters
    ----------
    x : str
        One of ``"lik"``, ``"mse"``, ``"amse"``, ``"sigma"``, ``"mae"``.

    Returns
    -------
    _ets.Criterion

    Raises
    ------
    ValueError
        If *x* is not a recognised objective name.
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
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
        jnp.asarray(alpha, dtype=jnp.float64),
        jnp.asarray(beta, dtype=jnp.float64),
        jnp.asarray(gamma, dtype=jnp.float64),
        jnp.asarray(phi, dtype=jnp.float64),
        int(m),
    )
    xs = x.reshape((n + 1, p))
    # Legacy NA sentinel (-99999) → NaN, branch-free: if lik is already NaN the
    # comparison is False and NaN propagates unchanged (same as the old
    # eager double-if, but traceable under vmap).
    lik = jnp.where(jnp.abs(lik + 99999.0) < _ets.TOL, jnp.nan, lik)
    return amse, e, xs, lik


def optimize_ets_target_fn(
    x0: Any,
    par: dict[str, float],
    y: jnp.ndarray,
    init_state: jnp.ndarray,
    errortype: str,
    trendtype: str,
    seasontype: str,
    damped: bool,
    par_noopt: dict[str, float],
    lowerb: jnp.ndarray,
    upperb: jnp.ndarray,
    opt_crit: str,
    nmse: int,
    bounds: str,
    m: int,
    pnames: Any,
    pnames2: Any,
    pad_to: Optional[int] = None,
    bucket_size: Optional[int] = None,
    maxit: int = 1_000,
    optax_steps: int = 300,
    optax_lr: float = 1e-2,
    optax_clip: float = 1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    adaptive_tol: bool = True,
    is_final_model: bool = False,
    clip_multiplicative_errors: bool = True,
    pure_sigmoid: bool = False,
    opt_init_state: bool = False,
    init_state_opt: jnp.ndarray | None = None,
) -> Any:
    """
    Build and solve the ETS optimization problem via `_ets.optimize_bfgs_smoothing`.

    Parameters
    ----------
    x0 : array
        Initial vector of free smoothing parameters.
    par : dict
        Only the optimizable parameters (non-NaN) from initparam.
    y : array
        Observations.
    init_state : array
        Fixed initial state vector.
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
    par_alpha = par.get("alpha", jnp.nan)
    alpha = par_noopt["alpha"] if math.isnan(par_alpha) else par_alpha
    if math.isnan(alpha):
        raise ValueError("alpha problem!")
    if trendtype != "N":
        par_beta = par.get("beta", jnp.nan)
        beta = par_noopt["beta"] if math.isnan(par_beta) else par_beta
        if math.isnan(beta):
            raise ValueError("beta problem!")
    else:
        beta = jnp.nan
    if seasontype != "N":
        par_gamma = par.get("gamma", jnp.nan)
        gamma = par_noopt["gamma"] if math.isnan(par_gamma) else par_gamma
        if math.isnan(gamma):
            raise ValueError("gamma problem!")
    else:
        m = 1
        gamma = jnp.nan
    if damped:
        par_phi = par.get("phi", jnp.nan)
        phi = par_noopt["phi"] if math.isnan(par_phi) else par_phi
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

    def _pad_for_jit(
        y_arr: jnp.ndarray, pad_to_val: Optional[int], bucket_size_val: Optional[int]
    ) -> tuple[jnp.ndarray, int]:
        """Pad the series for shape-stable JIT execution when requested."""
        n_obs = int(y_arr.shape[0])
        if pad_to_val is None and bucket_size_val is None:
            return y_arr, n_obs
        if pad_to_val is None:
            if bucket_size_val is None or bucket_size_val <= 0:
                return y_arr, n_obs
            pad_to_val = int(math.ceil(n_obs / bucket_size_val) * bucket_size_val)
        if pad_to_val <= n_obs:
            return y_arr, n_obs
        pad_len = pad_to_val - n_obs
        pad_val = y_arr[-1]
        pad = jnp.full((pad_len,), pad_val, dtype=y_arr.dtype)
        return jnp.concatenate([y_arr, pad]), n_obs

    y_opt, n_obs = _pad_for_jit(jnp.asarray(y, dtype=jnp.float64), pad_to, bucket_size)
    opt_steps = int(maxit) if optax_steps is None else int(optax_steps)
    if opt_init_state:
        if init_state_opt is None:
            raise ValueError("init_state_opt required when opt_init_state=True")
        x0_full = jnp.concatenate([jnp.asarray(x0, dtype=jnp.float64), jnp.asarray(init_state_opt, dtype=jnp.float64)])
        n_state = int(jnp.asarray(init_state_opt).shape[0])
    else:
        x0_full = jnp.asarray(x0, dtype=jnp.float64)
        n_state = 0
    opt_res = _ets.optimize_bfgs_smoothing(
        x0_full,
        y_opt,
        jnp.asarray(init_state, dtype=jnp.float64),
        switch(errortype),
        switch(trendtype),
        switch(seasontype),
        switch_criterion(opt_crit),
        int(nmse),
        int(m),
        n_obs,
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
        opt_steps,
        float(optax_lr),
        float(optax_clip),
        early_stop_patience,
        early_stop_min_delta,
        adaptive_tol,
        is_final_model,
        clip_multiplicative_errors,
        pure_sigmoid,
        opt_init_state,
        n_state,
    )
    return results(
        x=jnp.asarray(opt_res.x),
        fn=jnp.asarray(opt_res.fun),
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
    optax_steps: int | None = 300,
    optax_lr: float = 1e-2,
    optax_clip: float = 1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    adaptive_tol: bool = True,
    is_final_model: bool = False,
    control: Any = None,
    seed: Any = None,
    trace: bool = False,
    pad_to: Optional[int] = None,
    bucket_size: Optional[int] = None,
    stabilize: bool = True,
    pure_sigmoid: bool = False,
    init_state_override: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a *single* ETS specification to the data and return a result dict.

    Pipeline
    --------
    1. Initialise / sanitise smoothing parameters via :func:`initparam`.
    2. Check ranges and admissibility of the *starting* values via
       :func:`check_param` (config-level, eager).
    3. Build initial states via :func:`initstate`.
    4. Optimise the smoothing vector with
       :func:`ets_backend.optimize_bfgs_smoothing` (scan-based, vmap-safe).
    5. Re-run the full rollout to obtain likelihood, residuals, and fitted
       values.
    6. Compute information criteria (AIC, BIC, AICc), σ², and a traced
       ``valid`` flag (:func:`_valid_params_jnp` on the *fitted* params).

    The model structure (errortype/trendtype/seasontype/damped, m, bounds)
    is static config; everything derived from ``y`` stays a traced jnp value,
    so this whole function is traceable under ``jax.vmap`` (the conformal CV
    path). Instead of returning early with ``inf`` ICs when fitted parameters
    are out of range (untraceable), the ICs are real and the ``valid`` flag
    marks the fit for masking at selection time.

    Parameters
    ----------
    y : jnp.ndarray
        Observed time series.
    m : int
        Seasonal period.
    errortype, trendtype, seasontype : str
        Fixed structure flags (``"A"`` / ``"M"`` / ``"N"``).
    damped : bool
        Whether the trend is damped.
    alpha, beta, gamma, phi : float
        Starting / fixed smoothing parameters (``NaN`` → optimise).
    lower, upper : jnp.ndarray
        Box-constraint bounds (length 4).
    opt_crit : str
        Optimisation objective (``"lik"`` / ``"mse"`` / ``"amse"`` /
        ``"sigma"`` / ``"mae"``).
    nmse : int
        AMSE horizon (1–30).
    bounds : str
        Bound mode (``"both"`` / ``"usual"`` / ``"admissible"``).
    maxit : int
        Maximum iterations passed to the optimiser.
    optax_steps : int | None
        Explicit number of optax steps (``None`` → use *maxit*).

    Returns
    -------
    dict
        Keys: ``loglik``, ``aic``, ``bic``, ``aicc``, ``mse``, ``amse``,
        ``sigma2``, ``fit``, ``residuals``, ``fitted``, ``components``,
        ``m``, ``nstate``, ``states``, ``par``, ``n_params``, ``valid``.

    Raises
    ------
    ValueError
        If the series is too short to estimate this specification
        (a shape/config-static condition, so safe under trace).
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

    # initialize state (closed-form)
    if init_state_override is None:
        init_state_full = _prepare_init_state(y, m, trendtype, seasontype)
    else:
        init_state_full = jnp.asarray(init_state_override, dtype=jnp.float64)
    nstate = int(init_state_full.shape[0])
    opt_init_state = bool(pure_sigmoid and (not stabilize))

    opt_names = [
        name
        for name in ["alpha", "beta", "gamma", "phi"]
        if (not math.isnan(par_[name])) and math.isnan(par_noopt[name])
    ]
    par_clean = {name: par_[name] for name in opt_names}
    par_vec = jnp.full((len(opt_names),), jnp.nan, dtype=jnp.float64)
    if len(opt_names) > 0:
        par_vec = par_vec.at[: len(opt_names)].set(jnp.asarray([par_[name] for name in opt_names], dtype=jnp.float64))

    lower_box = jnp.asarray(lower, dtype=jnp.float64)
    upper_box = jnp.asarray(upper, dtype=jnp.float64)

    np_ = int(par_vec.shape[0])
    n_state_opt = int(init_state_full.shape[0]) if opt_init_state else 0
    np_eff = np_ + n_state_opt
    if np_eff >= len(y) - 1:
        # Shape/config-static condition — raising is trace-safe and replaces
        # the old inf-IC early return (whose shape differed from a real fit).
        raise ValueError(
            f"Not enough observations ({len(y)}) to estimate the "
            f"ETS({errortype},{trendtype},{seasontype}) specification."
        )
    if np_ > 0:
        x0_unconstrained = []
        for name in opt_names:
            if name == "alpha":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[0]), float(upper[0])))
            elif name == "beta":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    beta_cap = float(jnp.minimum(upper[1], par_["alpha"]))
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[1]), beta_cap))
            elif name == "gamma":
                if pure_sigmoid:
                    val = float(jnp.clip(par_[name], 1e-6, 1.0 - 1e-6))
                    x0_unconstrained.append(float(jnp.log(val / (1.0 - val))))
                else:
                    gamma_cap = float(jnp.minimum(upper[2], 1.0 - par_["alpha"]))
                    x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[2]), gamma_cap))
            elif name == "phi":
                x0_unconstrained.append(_inv_sigmoid_scaled(par_[name], float(lower[3]), float(upper[3])))
        par_vec = jnp.asarray(x0_unconstrained, dtype=jnp.float64)
        fred = optimize_ets_target_fn(
            x0=par_vec,
            par=par_clean,
            y=jnp.asarray(y, dtype=jnp.float64),
            init_state=init_state_full,
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
            pad_to=pad_to,
            bucket_size=bucket_size,
            maxit=maxit,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            adaptive_tol=adaptive_tol,
            is_final_model=is_final_model,
            clip_multiplicative_errors=stabilize,
            pure_sigmoid=pure_sigmoid,
            opt_init_state=opt_init_state,
            init_state_opt=init_state_full if opt_init_state else None,
        )
        fit_par = jnp.asarray(fred.x)
    else:
        fred = results(x=jnp.asarray(par_vec), fn=jnp.nan, nit=0, simplex=jnp.empty((0,), dtype=jnp.float64))
        fit_par = jnp.asarray(par_vec)

    init_state_fit = init_state_full
    nstate = int(init_state_fit.shape[0])

    valid = jnp.asarray(True)
    if np_ > 0:
        if opt_init_state and n_state_opt > 0:
            fit_par_smooth = fit_par[:-n_state_opt]
            init_state_fit = fit_par[-n_state_opt:]
        else:
            fit_par_smooth = fit_par
        # Decode the (traced) optimizer output with the backend's jnp-native
        # transform; the fitted params stay traced end-to-end.
        alpha, beta, gamma, phi = _ets._transform_smoothing_params(
            fit_par_smooth,
            "alpha" in opt_names, "beta" in opt_names,
            "gamma" in opt_names, "phi" in opt_names,
            alpha, beta, gamma, phi, lower, upper, pure_sigmoid,
        )
        valid = _valid_params_jnp(
            jnp.asarray(alpha, dtype=jnp.float64),
            jnp.asarray(beta, dtype=jnp.float64),
            jnp.asarray(gamma, dtype=jnp.float64),
            jnp.asarray(phi, dtype=jnp.float64),
            lower, upper, bounds, m,
            damped=bool(damped),
            has_beta=(trendtype != "N"),
            has_gamma=(seasontype != "N"),
        )

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
    k = np_ + n_state_opt + 1
    ny = len(y)
    aic, bic, aicc = _compute_ic(lik, ny, k)

    mse = amse[0]
    amse_mean = jnp.mean(amse)

    fit_par_full = jnp.concatenate([
        jnp.stack([
            jnp.asarray(alpha, dtype=jnp.float64),
            jnp.asarray(beta, dtype=jnp.float64),
            jnp.asarray(gamma, dtype=jnp.float64),
            jnp.asarray(phi, dtype=jnp.float64),
        ]),
        init_state_fit,
    ])

    if errortype == "A":
        fits = jnp.asarray(y) - e
    else:
        # protect e == -1
        aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
        fits = jnp.asarray(y) / (1.0 + aux_e)

    sq_e = e * e
    # Exclude infs from the sum but let NaN propagate (matches the old eager
    # boolean-mask indexing, which cannot trace: masked shapes are dynamic).
    sigma2 = jnp.sum(jnp.where(jnp.isinf(sq_e), 0.0, sq_e)) / (ny - k - 1)

    return dict(
        loglik=-0.5 * lik,
        aic=aic,
        bic=bic,
        aicc=aicc,
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
        n_params=k,
        valid=valid,
    )


def ets_f(
    y: jnp.ndarray,
    m: int,
    model: str | dict[str, Any] = "ZZZ",
    damped: Optional[bool] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    phi: Optional[float] = None,
    additive_only: Optional[bool] = None,
    blambda: Any = None,
    biasadj: Any = None,
    lower: Optional[jnp.ndarray] = None,
    upper: Optional[jnp.ndarray] = None,
    opt_crit: str = "lik",
    nmse: int = 3,
    bounds: str = "both",
    ic: str = "aicc",
    restrict: bool = True,
    allow_multiplicative_trend: bool = False,
    use_initial_values: bool = False,
    maxit: int = 2_000,
    optax_steps: Optional[int] = None,
    optax_lr: float = 1e-2,
    optax_clip: float = 1.0,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-6,
    allow_extended_iterations: bool = False,
    adaptive_tol: bool = True,
    pad_to: Optional[int] = None,
    bucket_size: Optional[int] = None,
) -> dict[str, Any]:
    """Top-level ETS entry-point: automatic model selection **and** fitting.

    When *model* is a three-character string (e.g. ``"ZZZ"``), every ``"Z"``
    is expanded into a grid of candidate component types. The grid is
    *config/shape-static*: every candidate is fitted via :func:`etsmodel`
    (a static Python loop) and the winner is a NaN/validity-masked
    ``jnp.argmin`` over the chosen information criterion — a traced index,
    never a concretized value. This makes the whole function traceable under
    ``jax.vmap`` (the conformity_scores CV path).

    When *model* is a ``dict`` (a previously-fitted classic dict), this
    function acts as a **forward** step — rolling the stored parameters over
    new data without re-optimisation (see :func:`forward_ets`).

    Parameters
    ----------
    y : array-like
        Time series (cast to float64 internally).
    m : int
        Seasonal period (``1`` for non-seasonal data). Config-only: array
        shapes derive from it, so it is never inferred from the data.
    model : str | dict
        ``"ZZZ"`` for full auto-selection, a fixed spec like ``"AAN"``
        for a single fit, or a previously-fitted dict for the forward path.
    damped : bool | None
        ``None`` → try both damped and undamped; ``True`` / ``False`` →
        fix the choice.
    alpha, beta, gamma, phi : float | None
        ``None`` → optimise; provide a float to fix the value.
    additive_only : bool | None
        If ``True``, forbid all multiplicative component types.
    blambda, biasadj
        *(Not implemented — Box-Cox / bias-adjustment placeholders.)*
    lower, upper : array-like | None
        Box bounds for ``[α, β, γ, φ]``; sensible defaults applied when
        ``None``.
    opt_crit : str
        Optimisation objective (``"lik"`` / ``"mse"`` / ``"amse"`` /
        ``"sigma"`` / ``"mae"``).
    nmse : int
        AMSE tracking horizon (1–30).
    bounds : str
        Bound mode (``"both"`` / ``"usual"`` / ``"admissible"``).
    ic : str
        Information criterion for model selection (``"aicc"`` / ``"aic"``
        / ``"bic"``).
    restrict : bool
        Apply standard ETS combination restrictions (e.g. forbid MMA).
    allow_multiplicative_trend : bool
        Include ``"M"`` trend in the candidate grid.
    use_initial_values : bool
        *(Reserved — not used.)*
    maxit : int
        Maximum iteration budget passed to the optimiser.
    optax_steps : int | None
        Explicit optax step count (``None`` → a static table keyed on the
        series *length*, which is shape-static under vmap).
    early_stop_patience, early_stop_min_delta, allow_extended_iterations,
    adaptive_tol
        *(Inert — retained for API compatibility. Early stopping and
        data-adaptive iteration budgets require concretizing traced values;
        the optimizer instead runs a fixed budget with monotone best-iterate
        tracking, so extra steps can never make the fit worse.)*

    Returns
    -------
    dict
        A **stacked** model dictionary (see Notes) for string *model* input,
        or a classic single-model dictionary for dict input (forward path).

    Raises
    ------
    ValueError
        For invalid model strings, forbidden combinations, or inconsistent
        bounds (all config-static conditions).

    Notes
    -----
    The stacked dictionary contains, with ``C`` candidates and state width
    ``S = max(nstate)``:

    * static metadata — ``m``, ``n``, ``ic_name``, ``candidates``
      (tuple of ``(error, trend, season, damped)``), ``cand_m`` /
      ``cand_nstate`` / ``cand_k`` (per-candidate ints);
    * per-candidate stacks (axis 0 = candidate) — ``cand_par`` ``(C, 4)``,
      ``cand_init_state`` / ``cand_last_state`` ``(C, S)`` (zero-padded),
      ``cand_fitted`` / ``cand_residuals`` ``(C, n)``, ``cand_sigma2`` /
      ``cand_loglik`` / ``cand_aic`` / ``cand_bic`` / ``cand_aicc`` /
      ``cand_mse`` / ``cand_amse`` / ``cand_valid`` ``(C,)``, and ``ic``
      (criterion values masked to ``inf`` where invalid);
    * selection — ``best`` (traced argmin index), ``valid`` (traced "any
      candidate finite" flag);
    * winner conveniences under the classic keys (``jnp.take``-selected,
      traced) — ``fitted``, ``residuals``, ``sigma2``, ``loglik``, ``aic``,
      ``bic``, ``aicc``, ``mse``, ``amse``, ``par`` (4,), ``n_params``.

    Use :func:`ets_point_forecast` for traceable point forecasts,
    :func:`ets_winner_view` for an eager classic-dict view of the winner
    (needed by the native-interval machinery), and
    :func:`ets_forward_stacked` / :func:`forward_ets` to re-roll on new data.

    Multiplicative-component candidates are *masked out* (IC → ``inf``) when
    the data are not strictly positive, instead of raising — a raise on a
    data value cannot trace. A fixed multiplicative spec on non-positive
    data therefore yields NaN forecasts rather than an exception.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    n = len(y)
    season_forced = m > 1 and n < 2 * m
    if season_forced:
        m = 1
    auto_model = isinstance(model, str) and ("Z" in model)
    errortype = trendtype = seasontype = None
    if isinstance(model, str):
        errortype, trendtype, seasontype = model
        if season_forced:
            seasontype = "N"
    if optax_steps is None:
        # Static iteration table keyed on series *length* (shape-static under
        # vmap). The old data-adaptive estimator concretized traced values.
        if n < 100:
            optax_steps = 50
        elif n < 200:
            optax_steps = 75
        elif n < 500:
            optax_steps = 100
        elif n < 2000:
            optax_steps = 120
        else:
            optax_steps = 100

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
    if bounds == "both":
        bounds = "admissible"
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
        aic, bic, aicc = _compute_ic(lik, ny, np_)

        mse = amse[0]
        amse_mean = jnp.mean(amse)

        fit_par = jnp.concatenate([jnp.asarray(model["par"][:4], dtype=jnp.float64), init_state])
        if errortype == "A":
            fits = y - e
        else:
            aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
            fits = y / (1 + aux_e)
        sq_e = e * e
        sigma2 = jnp.sum(jnp.where(jnp.isinf(sq_e), 0.0, sq_e)) / (ny - np_ - 1)

        return dict(
            loglik=-0.5 * lik,
            aic=aic,
            bic=bic,
            aicc=aicc,
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

    # errortype, trendtype, seasontype already parsed above (early parse block)
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
    if damped is not None:
        if damped and trendtype == "N":
            raise ValueError("Forbidden model combination")
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

    cycles = n / max(m, 1)
    # Expand Z components into their full candidate lists; all pruning below
    # is config/shape-static. Data-dependent gates (positivity) are applied
    # as traced IC masks after fitting, never by shrinking the static grid.
    if errortype == "Z":
        errortype = ["A", "M"]
    else:
        errortype = [errortype]
    if trendtype == "Z":
        trendtype = ["N", "A"] + (["M"] if allow_multiplicative_trend else [])
    else:
        trendtype = [trendtype]
    if seasontype == "Z":
        seasontype = ["N", "A", "M"]
    else:
        seasontype = [seasontype]
    prefer_damped_small_n = auto_model and n < 200
    force_additive_error = auto_model and n < 80
    if damped is None:
        damped_opts = [True, False]
    else:
        damped_opts = [bool(damped)]

    candidates: list[tuple[str, str, str, bool]] = []

    def _maybe_add_candidate(etype: str, ttype: str, stype: str, dtype: bool) -> None:
        """Append an admissible candidate spec to the static search grid."""
        if restrict:
            # Additive error with any multiplicative component is numerically
            # unstable (forecast::ets forbids A+M-trend and A+M-season under
            # restrict; the legacy hand-built grid never generated them, so
            # the full static grid must exclude them here — matches the
            # fixed-spec "Forbidden model combination" raise above).
            if etype == "A" and (ttype == "M" or stype == "M"):
                return
            if etype == "M" and ttype == "M" and stype == "A":
                return
            if additive_only and (etype == "M" or ttype == "M" or stype == "M"):
                return
        if stype != "N" and m == 1:
            return
        if ttype == "N" and dtype:
            return
        if auto_model:
            # Shape-static pruning heuristics — auto grid only; a user-fixed
            # spec must never be silently dropped by a heuristic.
            if force_additive_error and etype == "M":
                return
            if etype == "M" and cycles < 10:
                return
            if stype == "M" and cycles < 10:
                return
            if prefer_damped_small_n and ttype != "N" and (not dtype):
                return
        candidates.append((etype, ttype, stype, dtype))

    for etype in errortype:
        for ttype in trendtype:
            for stype in seasontype:
                for dtype in damped_opts:
                    _maybe_add_candidate(etype, ttype, stype, dtype)

    if auto_model:
        priority = ["ANN", "AAN", "ANA", "AAA", "MAN", "MNM", "MAM", "MMM"]
        priority_index = {k: i for i, k in enumerate(priority)}

        def _rank(c: tuple[str, str, str, bool]) -> tuple[float, str]:
            """Rank candidate ETS specifications for deterministic ordering."""
            et, tt, st, dt = c
            key = f"{et}{tt}{st}"
            base = priority_index.get(key, len(priority))
            damp_penalty = 0.5 if dt else 0.0
            return (base + damp_penalty, key)

        # Deterministic order (argmin tie-break follows priority) and a cap
        # that bounds per-shape compile cost for very large grids.
        candidates.sort(key=_rank)
        candidates = candidates[:12]

    if not candidates:
        raise ValueError("No admissible ETS model found")

    # ── Fit every candidate (static loop) and select by masked argmin ──
    init_state_cache: dict[tuple[str, str], jnp.ndarray] = {}
    fits: list[dict[str, Any]] = []
    for etype, ttype, stype, dtype in candidates:
        init_key = (ttype, stype)
        if init_key not in init_state_cache:
            init_state_cache[init_key] = _prepare_init_state(y, m, ttype, stype)
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
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            optax_clip=optax_clip,
            pad_to=pad_to,
            bucket_size=bucket_size,
            stabilize=True,
            pure_sigmoid=auto_model,
            init_state_override=init_state_cache[init_key],
        )
        fits.append(fit)

    data_positive = jnp.min(y) > 0
    ic_rows = []
    for spec, fit in zip(candidates, fits):
        ok = fit["valid"]
        if spec[0] == "M" or spec[2] == "M":
            ok = ok & data_positive
        ic_val = jnp.asarray(fit[ic], dtype=jnp.float64)
        ic_rows.append(jnp.where(ok & jnp.isfinite(ic_val), ic_val, jnp.inf))
    ic_vec = jnp.stack(ic_rows)
    best = jnp.argmin(ic_vec)
    # Raising on "no admissible model" is impossible under trace; expose a
    # traced flag instead (argmin over all-inf picks index 0).
    valid_any = jnp.isfinite(ic_vec).any()

    cand_nstate = tuple(int(f["nstate"]) for f in fits)
    S = max(cand_nstate)

    def _pad_state(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.pad(v, (0, S - v.shape[0]))

    def _stack(key: str) -> jnp.ndarray:
        return jnp.stack([jnp.asarray(f[key], dtype=jnp.float64) for f in fits])

    def _take(stack: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(stack, best, axis=0)

    cand_fitted = _stack("fitted")
    cand_residuals = _stack("residuals")
    cand_sigma2 = _stack("sigma2")
    cand_loglik = _stack("loglik")
    cand_aic = _stack("aic")
    cand_bic = _stack("bic")
    cand_aicc = _stack("aicc")
    cand_mse = _stack("mse")
    cand_amse = _stack("amse")
    cand_k = jnp.asarray([f["n_params"] for f in fits])
    cand_par = jnp.stack([f["par"][:4] for f in fits])
    cand_init_state = jnp.stack([_pad_state(f["par"][4:]) for f in fits])
    cand_last_state = jnp.stack([_pad_state(f["states"][-1, :]) for f in fits])

    return dict(
        # static metadata
        m=m,
        n=n,
        ic_name=ic,
        candidates=tuple(candidates),
        cand_m=tuple(int(f["m"]) for f in fits),
        cand_nstate=cand_nstate,
        cand_k=tuple(int(f["n_params"]) for f in fits),
        # per-candidate stacks
        cand_par=cand_par,
        cand_init_state=cand_init_state,
        cand_last_state=cand_last_state,
        cand_fitted=cand_fitted,
        cand_residuals=cand_residuals,
        cand_sigma2=cand_sigma2,
        cand_loglik=cand_loglik,
        cand_aic=cand_aic,
        cand_bic=cand_bic,
        cand_aicc=cand_aicc,
        cand_mse=cand_mse,
        cand_amse=cand_amse,
        cand_valid=jnp.stack([f["valid"] for f in fits]),
        ic=ic_vec,
        # selection
        best=best,
        valid=valid_any,
        # winner conveniences under the classic keys (traced take-selection)
        fitted=_take(cand_fitted),
        residuals=_take(cand_residuals),
        sigma2=_take(cand_sigma2),
        loglik=_take(cand_loglik),
        aic=_take(cand_aic),
        bic=_take(cand_bic),
        aicc=_take(cand_aicc),
        mse=_take(cand_mse),
        amse=_take(cand_amse),
        par=_take(cand_par),
        n_params=_take(cand_k),
    )


def ets_point_forecast(mod: dict[str, Any], h: int) -> jnp.ndarray:
    """Traceable point forecasts from a stacked :func:`ets_f` result.

    ``etsforecast`` needs *static* structure flags, so the traced winner
    index cannot be dispatched on directly. Each candidate is forecast with
    its own static config (a Python loop over config) and the winning row is
    selected with ``jnp.take`` — the AutoCES ``_forecast_candidates`` pattern.

    Parameters
    ----------
    mod : dict
        Stacked model dictionary from :func:`ets_f` (string-model input) or
        :func:`ets_forward_stacked`.
    h : int
        Forecast horizon (static).

    Returns
    -------
    jnp.ndarray
        Point forecasts of shape ``(h,)`` for the IC-selected candidate.
    """
    fcsts = []
    for i, (etype, ttype, stype, dtype) in enumerate(mod["candidates"]):
        nst = mod["cand_nstate"][i]
        m_i = mod["cand_m"][i]
        x = mod["cand_last_state"][i, :nst]
        phi_i = mod["cand_par"][i, 3] if dtype else 1.0
        f = jnp.full((h,), jnp.nan, dtype=jnp.float64)
        f = etsforecast(
            x=x, m=m_i, trend=switch(ttype), season=switch(stype),
            phi=phi_i, h=h, f=f,
        )
        fcsts.append(f)
    return jnp.take(jnp.stack(fcsts), mod["best"], axis=0)


def ets_winner_view(mod: dict[str, Any]) -> dict[str, Any]:
    """Materialize the winning candidate of a stacked fit as a classic dict.

    With a single candidate the winner index is static and this stays fully
    traceable. With multiple candidates the traced ``best`` index must be
    concretized (``int(...)``) to recover the winner's *strings* (components,
    method) — an **eager-only** operation, acceptable because its only
    consumers are the native-interval formulas and display paths, which run
    on concrete post-fit values (the conformal CV path never needs strings).

    Returns
    -------
    dict
        Classic model dictionary compatible with :func:`forecast_ets`,
        :func:`pegelsfcast_C`, :func:`_compute_pred_intervals`, and the
        dict-input forward path of :func:`ets_f`.
    """
    n_cand = len(mod["candidates"])
    b = 0 if n_cand == 1 else int(mod["best"])
    etype, ttype, stype, dtype = mod["candidates"][b]
    nst = mod["cand_nstate"][b]
    par4 = mod["cand_par"][b]
    return dict(
        loglik=mod["cand_loglik"][b],
        aic=mod["cand_aic"][b],
        bic=mod["cand_bic"][b],
        aicc=mod["cand_aicc"][b],
        mse=mod["cand_mse"][b],
        amse=mod["cand_amse"][b],
        fit=results(x=par4, fn=jnp.nan, nit=0, simplex=jnp.empty((0,), dtype=jnp.float64)),
        residuals=mod["cand_residuals"][b],
        fitted=mod["cand_fitted"][b],
        components=f"{etype}{ttype}{stype}{'D' if dtype else 'N'}",
        m=mod["cand_m"][b],
        nstate=nst,
        states=mod["cand_last_state"][b, :nst][None, :],
        par=jnp.concatenate([par4, mod["cand_init_state"][b, :nst]]),
        sigma2=mod["cand_sigma2"][b],
        n_params=mod["cand_k"][b],
        method=f"ETS({etype},{ttype}{'d' if dtype else ''},{stype})",
    )


def ets_forward_stacked(mod: dict[str, Any], y: jnp.ndarray) -> dict[str, Any]:
    """Re-roll a stacked fit's stored parameters over *new* data (traceable).

    Mirrors the dict-input forward path of :func:`ets_f`, but for every
    candidate: each candidate's stored smoothing parameters and fit-time
    initial state are rolled over the new series with :func:`pegelsresid_C`
    (no re-optimisation), and the *fit-time* winner index is kept.

    Parameters
    ----------
    mod : dict
        Stacked model dictionary from :func:`ets_f`.
    y : array-like
        New time series.

    Returns
    -------
    dict
        A stacked dictionary with the same schema as :func:`ets_f`'s.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    ny = len(y)
    S = mod["cand_init_state"].shape[1]

    def _pad_state(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.pad(v, (0, S - v.shape[0]))

    rolled: dict[str, list] = {k: [] for k in (
        "fitted", "residuals", "sigma2", "loglik", "aic", "bic", "aicc",
        "mse", "amse", "last_state",
    )}
    for i, (etype, ttype, stype, dtype) in enumerate(mod["candidates"]):
        nst = mod["cand_nstate"][i]
        m_i = mod["cand_m"][i]
        k_i = mod["cand_k"][i]
        par4 = mod["cand_par"][i]
        amse, e, states, lik = pegelsresid_C(
            y=y,
            m=m_i,
            init_state=mod["cand_init_state"][i, :nst],
            errortype=etype,
            trendtype=ttype,
            seasontype=stype,
            damped=dtype,
            alpha=par4[0],
            beta=par4[1],
            gamma=par4[2],
            phi=par4[3],
            nmse=3,
        )
        aic, bic, aicc = _compute_ic(lik, ny, k_i)
        if etype == "A":
            fits_i = y - e
        else:
            aux_e = jnp.where(e == -1.0, -1 + 1e-3, e)
            fits_i = y / (1 + aux_e)
        sq_e = e * e
        sigma2_i = jnp.sum(jnp.where(jnp.isinf(sq_e), 0.0, sq_e)) / (ny - k_i - 1)
        rolled["fitted"].append(fits_i)
        rolled["residuals"].append(e)
        rolled["sigma2"].append(sigma2_i)
        rolled["loglik"].append(-0.5 * lik)
        rolled["aic"].append(aic)
        rolled["bic"].append(bic)
        rolled["aicc"].append(aicc)
        rolled["mse"].append(amse[0])
        rolled["amse"].append(jnp.mean(amse))
        rolled["last_state"].append(_pad_state(states[-1, :]))

    best = mod["best"]

    def _stackrow(key: str) -> jnp.ndarray:
        return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in rolled[key]])

    def _take(stack: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(stack, best, axis=0)

    cand_fitted = _stackrow("fitted")
    cand_residuals = _stackrow("residuals")
    cand_sigma2 = _stackrow("sigma2")
    cand_loglik = _stackrow("loglik")
    cand_aic = _stackrow("aic")
    cand_bic = _stackrow("bic")
    cand_aicc = _stackrow("aicc")
    cand_mse = _stackrow("mse")
    cand_amse = _stackrow("amse")
    cand_k = jnp.asarray(list(mod["cand_k"]))

    return dict(
        m=mod["m"],
        n=ny,
        ic_name=mod["ic_name"],
        candidates=mod["candidates"],
        cand_m=mod["cand_m"],
        cand_nstate=mod["cand_nstate"],
        cand_k=mod["cand_k"],
        cand_par=mod["cand_par"],
        cand_init_state=mod["cand_init_state"],
        cand_last_state=jnp.stack(rolled["last_state"]),
        cand_fitted=cand_fitted,
        cand_residuals=cand_residuals,
        cand_sigma2=cand_sigma2,
        cand_loglik=cand_loglik,
        cand_aic=cand_aic,
        cand_bic=cand_bic,
        cand_aicc=cand_aicc,
        cand_mse=cand_mse,
        cand_amse=cand_amse,
        cand_valid=mod["cand_valid"],
        ic=mod["ic"],
        best=best,
        valid=mod["valid"],
        fitted=_take(cand_fitted),
        residuals=_take(cand_residuals),
        sigma2=_take(cand_sigma2),
        loglik=_take(cand_loglik),
        aic=_take(cand_aic),
        bic=_take(cand_bic),
        aicc=_take(cand_aicc),
        mse=_take(cand_mse),
        amse=_take(cand_amse),
        par=jnp.take(mod["cand_par"], best, axis=0),
        n_params=_take(cand_k),
    )


def pegelsfcast_C(
    h: int,
    obj: dict[str, Any],
    npaths: Optional[int] = None,
    level: Optional[list[int]] = None,
    bootstrap: Optional[bool] = None,
) -> jnp.ndarray:
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
    # Damping is a static fact of the components string; par[3] may be traced.
    phi = 1.0 if obj["components"][3] == "N" else obj["par"][3]
    m = int(obj["m"])

    # allocate and CAPTURE the result of etsforecast
    forecast = jnp.full((h,), jnp.nan, dtype=states.dtype)
    forecast = etsforecast(
        x=states, m=m, trend=ttype, season=stype, phi=phi, h=h, f=forecast
    )
    return forecast


@partial(jax.jit, static_argnames=("h",))
def _compute_sigmah(
    pf: jnp.ndarray, h: int, sigma: float, cvals: jnp.ndarray
) -> jnp.ndarray:
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
    theta = jnp.zeros((h,), dtype=pf.dtype)
    theta = theta.at[0].set(pf[0] ** 2)
    idx = jnp.arange(h)

    def body(k: int, theta_acc: jnp.ndarray) -> jnp.ndarray:
        """Advance the multiplicative-error variance recursion.

        Computes sum_{j<k} cvals[j]^2 * theta[k-1-j] with a masked
        full-length gather — the loop index k is traced inside fori_loop,
        so dynamic slices (cvals[:k], theta_acc[:k][::-1]) cannot compile
        (slice bounds must be static under jit).
        """
        rev = theta_acc[jnp.clip(k - 1 - idx, 0, h - 1)]
        sum_val = jnp.sum(jnp.where(idx < k, cvals[idx] ** 2 * rev, 0.0))
        return theta_acc.at[k].set(pf[k] ** 2 + sigma * sum_val)

    theta = lax.fori_loop(1, h, body, theta)
    return (1 + sigma) * theta - pf**2


@partial(jax.jit, static_argnames=("h", "season_length", "error", "trend", "seasonality", "damped"))
def _class3models(
    h: int,
    sigma: float,
    last_state: jnp.ndarray,
    season_length: int,
    error: str,
    trend: str,
    seasonality: str,
    damped: str,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
) -> jnp.ndarray:
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
    mu = jnp.zeros(h, dtype=last_state.dtype)
    var = jnp.zeros(h, dtype=last_state.dtype)
    vecMh = Mh.flatten()
    vecMh_col = vecMh.reshape(vecMh.shape[0], 1)

    def body(i: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Advance the analytical variance recursion for one horizon step."""
        Vh_acc, mu_acc, var_acc = carry
        mu_i = jnp.squeeze(H1 @ (Mh @ H2.T))
        var_i = jnp.squeeze((1 + sigma) * (H21 @ (Vh_acc @ H21.T))) + sigma * (mu_i ** 2)
        mu_acc = mu_acc.at[i].set(mu_i)
        var_acc = var_acc.at[i].set(var_i)
        exp1 = F21 @ (Vh_acc @ F21.T)
        exp2 = F21 @ (Vh_acc @ G21.T)
        exp3 = G21 @ (Vh_acc @ F21.T)
        exp4 = K @ ((Vh_acc + (vecMh * vecMh_col)) @ K.T)
        exp5 = (sigma * G21) @ ((3 * Vh_acc + 2 * vecMh * vecMh_col) @ G21.T)
        Vh_next = exp1 + sigma * (exp2 + exp3 + exp4 + exp5)
        return (Vh_next, mu_acc, var_acc)

    Vh, mu, var = lax.fori_loop(0, h, body, (Vh, mu, var))

    if trend == "N":
        Mh = F1 * (Mh @ F2.T) + G1 * (Mh @ G2.T) * sigma
    else:
        Mh = F1 @ (Mh @ F2.T) + G1 @ (Mh @ G2.T) * sigma

    return var


def _compute_pred_intervals(
    model: Dict[str, Any],
    forecasts: Dict[str, jnp.ndarray],
    h: int,
    level: list[int],
) -> dict[str, jnp.ndarray]:
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
        phi_powers = phi ** jnp.arange(1, h + 1, dtype=jnp.float64)
        cvals = cvals.at[:].set(alpha + beta * jnp.cumsum(phi_powers))
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
        phi_powers = phi ** jnp.arange(1, h + 1, dtype=jnp.float64)
        cvals = alpha + beta * jnp.cumsum(phi_powers) + gamma * dvals
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

        def run_sim(e_k: jnp.ndarray) -> jnp.ndarray:
            """Simulate one future path for fallback interval estimation."""
            return _etssimulate_jit(
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
                e_k,
            )

        y_path = jax.vmap(run_sim)(e)

        lower_q = 0.5 - jnp.asarray(level) / 200.0
        upper_q = 0.5 + jnp.asarray(level) / 200.0
        # q must be rank <= 1 for jnp.quantile; the result is already
        # (len(level), h), so per-level row indexing below stands.
        lower = jnp.quantile(y_path, lower_q, axis=0)
        upper = jnp.quantile(y_path, upper_q, axis=0)
        pi = {
            **{f"lo-{int(lv)}": lower[i] for i, lv in enumerate(level)},
            **{f"hi-{int(lv)}": upper[i] for i, lv in enumerate(level)},
        }

    if compute_intervals:
        pi = _calculate_intervals(forecasts, level=level, h=h, sigmah=jnp.sqrt(sigmah))

    return pi


def forecast_ets(
    obj: dict[str, Any], h: int, level: Optional[list[int]] = None
) -> dict[str, jnp.ndarray]:
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


def forward_ets(fitted_model: dict, y: jnp.ndarray) -> dict:
    """Roll a previously fitted ETS model forward on *new* data.

    The stored model structure and smoothing parameters are reused — no
    re-optimisation takes place. Stacked dictionaries (from string-model
    :func:`ets_f`) are re-rolled per candidate via
    :func:`ets_forward_stacked`; classic single-model dictionaries go
    through the dict-input branch of :func:`ets_f`. Both paths are
    traceable under vmap.

    Parameters
    ----------
    fitted_model : dict
        Output of :func:`ets_f` (stacked) or a classic dict from
        :func:`etsmodel` / :func:`ets_winner_view`.
    y : array-like
        New time series segment.

    Returns
    -------
    dict
        Fresh model dict (same schema as the input's) with residuals,
        fitted values, and states recomputed on *y*.
    """
    if "candidates" in fitted_model:
        return ets_forward_stacked(fitted_model, y)
    return ets_f(y=y, m=fitted_model["m"], model=fitted_model)
