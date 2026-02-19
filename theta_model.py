"""
Theta Model — JAX-accelerated implementation.

Implements the Standard Theta Method (STM), Optimized Theta Method (OTM),
Dynamic Standard Theta Method (DSTM), and Dynamic Optimized Theta Method (DOTM)
with full JAX JIT compilation and two-phase ADAM + L-BFGS optimization.

The Theta method decomposes a time series into an SES (simple exponential smoothing)
component and a linear trend correction:
    forecast[i] = SES_level[i-1] + (1 - 1/theta) * trend_correction(i)
where:
    trend_correction(i) = An*(1-alpha)^i + Bn*(1-(1-alpha)^(i+1))/alpha

State vector: [level, mean_y, An, Bn, mu] (5 elements)

References:
    Jose A. Fiorucci et al. (2016). "Models for optimising the theta method and
    their relationship to state space models". International Journal of Forecasting.
"""
import functools
import math

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax import lax
from jax.scipy.stats import norm
from enum import IntEnum
from typing import Optional, List

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
from utils import (
    ensure_float,
    _add_fitted_pi,
    _store_cs,
    _add_conformal_intervals,
    _add_predict_conformal_intervals,
    _conformal_method,
    is_constant,
    seasonal_decompose,
    acf,
    _repeat_val_seas,
    _seasonal_naive,
    _jax_norm_ppf,
)


# =============================================================================
# Constants
# =============================================================================

class ModelType(IntEnum):
    STM = 1
    OTM = 2
    DSTM = 3
    DOTM = 4

_EPSILON = 1e-10
_EPS_PARAM = 0.001
_ADAM_STEPS = 30
_ADAM_LR = 0.02
_LBFGS_STEPS = 20

__all__ = ['Theta', 'AutoTheta', 'ModelType']


# =============================================================================
# Core Math — Module-level JIT'd functions
# =============================================================================

@functools.partial(jax.jit, static_argnums=(1,))
def _init_state(y, model_type, initial_smoothed, alpha, theta):
    """Initialize the 5-element state vector.

    State: [level, mean_y, An, Bn, mu]

    For STM/OTM: An, Bn come from OLS linear regression on full series.
    For DSTM/DOTM: An = y[0], Bn = 0.
    """
    n = y.shape[0]
    n_f = jnp.asarray(n, dtype=y.dtype)

    # OLS regression for STM/OTM
    y_mean = jnp.mean(y)
    weighted_avg = jnp.dot(y, jnp.arange(1, n + 1, dtype=y.dtype)) / n_f
    Bn_static = 6.0 * (2.0 * weighted_avg - (n_f + 1.0) * y_mean) / (n_f ** 2 - 1.0)
    An_static = y_mean - (n_f + 1.0) * Bn_static / 2.0

    # Dynamic initialization for DSTM/DOTM
    An_dynamic = y[0]
    Bn_dynamic = jnp.asarray(0.0, dtype=y.dtype)

    is_dynamic = (model_type == ModelType.DSTM) | (model_type == ModelType.DOTM)

    An = jnp.where(is_dynamic, An_dynamic, An_static)
    Bn = jnp.where(is_dynamic, Bn_dynamic, Bn_static)

    level = alpha * y[0] + (1.0 - alpha) * initial_smoothed
    mu_static = initial_smoothed + (1.0 - 1.0 / theta) * (An_static + Bn_static)
    mu_dynamic = y[0]
    mu = jnp.where(is_dynamic, mu_dynamic, mu_static)

    return jnp.stack([level, y[0], An, Bn, mu])


def _make_step_fn(model_type_val):
    """Return (step_fn, forecast_step_fn) for the given model type.

    Step functions are designed for lax.scan:
        carry = (level, mean_y, An, Bn, alpha, theta, step_idx)
        input = y_t (observation)
        output = (e_t, mu_t)

    Forecast step functions use mu instead of observation.
    """
    is_dynamic = model_type_val in (ModelType.DSTM, ModelType.DOTM)

    def step_fn(carry, y_t):
        level, mean_y, An, Bn, alpha, theta, i = carry

        # Theta forecast (mu)
        decay = (1.0 - alpha) ** i
        decay_next = (1.0 - alpha) ** (i + 1.0)
        trend_correction = An * decay + Bn * (1.0 - decay_next) / jnp.maximum(alpha, _EPSILON)
        mu = level + (1.0 - 1.0 / theta) * trend_correction

        # Residual
        e_t = y_t - mu

        # SES level update
        new_level = alpha * y_t + (1.0 - alpha) * level

        # Running mean update
        new_mean_y = (i * mean_y + y_t) / (i + 1.0)

        if is_dynamic:
            new_Bn = ((i - 1.0) * Bn + 6.0 * (y_t - mean_y) / (i + 1.0)) / (i + 2.0)
            new_An = new_mean_y - new_Bn * (i + 2.0) / 2.0
        else:
            new_An = An
            new_Bn = Bn

        new_carry = (new_level, new_mean_y, new_An, new_Bn, alpha, theta, i + 1.0)
        return new_carry, (e_t, mu)

    def forecast_step_fn(carry, _unused):
        level, mean_y, An, Bn, alpha, theta, i = carry

        # Theta forecast (mu)
        decay = (1.0 - alpha) ** i
        decay_next = (1.0 - alpha) ** (i + 1.0)
        trend_correction = An * decay + Bn * (1.0 - decay_next) / jnp.maximum(alpha, _EPSILON)
        mu = level + (1.0 - 1.0 / theta) * trend_correction

        # SES level update with mu as "observation"
        new_level = alpha * mu + (1.0 - alpha) * level

        # Running mean update
        new_mean_y = (i * mean_y + mu) / (i + 1.0)

        if is_dynamic:
            new_Bn = ((i - 1.0) * Bn + 6.0 * (mu - mean_y) / (i + 1.0)) / (i + 2.0)
            new_An = new_mean_y - new_Bn * (i + 2.0) / 2.0
        else:
            new_An = An
            new_Bn = Bn

        new_carry = (new_level, new_mean_y, new_An, new_Bn, alpha, theta, i + 1.0)
        return new_carry, mu

    return step_fn, forecast_step_fn


# Pre-build step functions for each model type (static dispatch)
_STEP_FNS = {}
_FORECAST_STEP_FNS = {}
for _mt in ModelType:
    _sfn, _fsfn = _make_step_fn(_mt)
    _STEP_FNS[int(_mt)] = _sfn
    _FORECAST_STEP_FNS[int(_mt)] = _fsfn


@functools.partial(jax.jit, static_argnums=(1, 5))
def _pegels_resid(y, model_type, initial_smoothed, alpha, theta, nmse):
    """Compute Theta model residuals and MSE using lax.scan.

    Returns: (amse, residuals, states, mse)
    - amse: average multi-step-ahead MSE (nmse,)
    - residuals: one-step-ahead errors (n,)
    - states: state vectors at each step (n, 5)
    - mse: sum(e[3:]^2) / max(mean(|y|), eps) — the optimization objective
    """
    n = y.shape[0]
    dtype = y.dtype

    # Initialize state
    state0 = _init_state(y, model_type, initial_smoothed, alpha, theta)
    level0, mean_y0, An0, Bn0, mu0 = state0

    # First residual
    e0 = y[0] - mu0

    # Select step function (static dispatch via model_type as static arg)
    step_fn = _STEP_FNS[int(model_type)]

    # Build carry: (level, mean_y, An, Bn, alpha, theta, step_idx)
    init_carry = (level0, mean_y0, An0, Bn0, alpha, theta, jnp.asarray(1.0, dtype=dtype))

    # Run lax.scan over y[1:]
    final_carry, (residuals_rest, mus_rest) = lax.scan(step_fn, init_carry, y[1:])

    # Concatenate first step
    residuals = jnp.concatenate([e0[None], residuals_rest])
    mus = jnp.concatenate([mu0[None], mus_rest])

    # Build states array (n, 5) — record level and mu at each step
    # For simplicity, we record the key info needed for forecasting
    final_level, final_mean_y, final_An, final_Bn, _, _, final_i = final_carry
    states = jnp.stack([
        jnp.concatenate([level0[None], jnp.zeros(n - 1, dtype=dtype)]),
        jnp.concatenate([mean_y0[None], jnp.zeros(n - 1, dtype=dtype)]),
        jnp.concatenate([An0[None], jnp.zeros(n - 1, dtype=dtype)]),
        jnp.concatenate([Bn0[None], jnp.zeros(n - 1, dtype=dtype)]),
        mus,
    ], axis=-1).reshape(n, 5)

    # Store final state in last row
    states = states.at[-1].set(jnp.stack([final_level, final_mean_y, final_An, final_Bn, mus[-1]]))

    # MSE: sum(e[3:]^2) / max(mean(|y|), eps) — matches statsforecast
    mean_abs_y = jnp.maximum(jnp.mean(jnp.abs(y)), _EPSILON)
    mse = jnp.sum(residuals[3:] ** 2) / mean_abs_y

    # AMSE (multi-step ahead): simplified single-step for now
    amse = jnp.zeros(nmse, dtype=dtype)

    return amse, residuals, states, mse


# =============================================================================
# Optimizer — ADAM + L-BFGS (following Holt-Winters pattern)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(1, 6, 7, 8, 9))
def _jit_optimize_theta(y, model_type, x0, init_level, init_alpha, init_theta,
                        opt_level, opt_alpha, opt_theta, nmse):
    """JIT'd ADAM + L-BFGS optimization for Theta model parameters.

    Parameters are reparameterized via sigmoid to enforce bounds:
    - initial_smoothed: unconstrained (scaled by y_std)
    - alpha: [0.1, 0.99]
    - theta: [1.0, 100.0]

    Compiled once per (model_type, opt_level, opt_alpha, opt_theta, nmse,
    y.shape, y.dtype), then cached.
    """
    dtype = y.dtype
    eps_param = jnp.asarray(_EPS_PARAM, dtype=dtype)
    eps_val = jnp.asarray(_EPSILON, dtype=dtype)

    # Scaling
    y_std = jnp.maximum(jnp.std(y), eps_val)
    y_mean = jnp.mean(y)

    # Bounds for sigmoid reparameterization
    alpha_lo = jnp.asarray(0.1, dtype=dtype)
    alpha_hi = jnp.asarray(0.99, dtype=dtype)
    theta_lo = jnp.asarray(1.0, dtype=dtype)
    theta_hi = jnp.asarray(100.0, dtype=dtype)

    def _to_constrained(p):
        """Map unconstrained parameters to bounded space."""
        idx = 0
        if opt_level:
            level = y_mean + y_std * p[idx]
            idx += 1
        else:
            level = init_level

        if opt_alpha:
            alpha = alpha_lo + (alpha_hi - alpha_lo) * jax.nn.sigmoid(p[idx])
            idx += 1
        else:
            alpha = init_alpha

        if opt_theta:
            theta = theta_lo + (theta_hi - theta_lo) * jax.nn.sigmoid(p[idx])
            idx += 1
        else:
            theta = init_theta

        return level, alpha, theta

    # Initialize unconstrained parameters
    n_params = int(opt_level) + int(opt_alpha) + int(opt_theta)
    init_unconstrained = jnp.zeros(max(n_params, 1), dtype=dtype)

    def loss_fn(p_unconstrained):
        p_unconstrained = jnp.asarray(p_unconstrained, dtype=dtype)
        level, alpha, theta = _to_constrained(p_unconstrained)
        _, _, _, mse = _pegels_resid(y, model_type, level, alpha, theta, nmse)
        return mse

    # Phase 1: ADAM warm-up
    vg_fn = jax.value_and_grad(loss_fn)
    adam_opt = optax.adam(_ADAM_LR)
    adam_state = adam_opt.init(init_unconstrained)

    def _adam_step(carry, _):
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = adam_opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_p = jnp.where(improved, new_p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    inf_val = jnp.asarray(float('inf'), dtype=dtype)
    adam_carry = (init_unconstrained, adam_state, init_unconstrained, inf_val)
    (_, _, adam_best_p, adam_best_loss), _ = lax.scan(
        _adam_step, adam_carry, jnp.arange(_ADAM_STEPS),
    )

    # Phase 2: L-BFGS refinement
    lbfgs_solver = optax.lbfgs(
        memory_size=8,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=8,
            initial_guess_strategy="one",
        ),
    )
    lbfgs_state = lbfgs_solver.init(adam_best_p)

    def _lbfgs_step(carry, _):
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = lbfgs_solver.update(
            grads, state, p,
            value=loss, grad=grads, value_fn=loss_fn,
        )
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_p = jnp.where(improved, p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    lbfgs_carry = (adam_best_p, lbfgs_state, adam_best_p, adam_best_loss)
    (_, _, lbfgs_best_p, lbfgs_best_loss), _ = lax.scan(
        _lbfgs_step, lbfgs_carry, jnp.arange(_LBFGS_STEPS),
    )

    # Pick best between ADAM and L-BFGS
    use_lbfgs = jnp.isfinite(lbfgs_best_loss) & (lbfgs_best_loss < adam_best_loss)
    best_p = jnp.where(use_lbfgs, lbfgs_best_p, adam_best_p)

    # Extract final constrained parameters
    level, alpha, theta = _to_constrained(best_p)

    return level, alpha, theta


# =============================================================================
# Parameter Initialization (Pure Python)
# =============================================================================

def _initparamtheta(initial_smoothed, alpha, theta, y, model_type):
    """Determine initial values and optimization flags per model type.

    STM/DSTM: theta fixed at 2.0, optimize level and alpha.
    OTM/DOTM: optimize level, alpha, and theta.
    """
    if model_type in (ModelType.STM, ModelType.DSTM):
        if initial_smoothed is None or (isinstance(initial_smoothed, float) and math.isnan(initial_smoothed)):
            initial_smoothed = float(y[0]) / 2.0
            opt_level = True
        else:
            opt_level = False
        if alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
            alpha = 0.5
            opt_alpha = True
        else:
            opt_alpha = False
        theta = 2.0
        opt_theta = False
    else:
        if initial_smoothed is None or (isinstance(initial_smoothed, float) and math.isnan(initial_smoothed)):
            initial_smoothed = float(y[0]) / 2.0
            opt_level = True
        else:
            opt_level = False
        if alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
            alpha = 0.5
            opt_alpha = True
        else:
            opt_alpha = False
        if theta is None or (isinstance(theta, float) and math.isnan(theta)):
            theta = 2.0
            opt_theta = True
        else:
            opt_theta = False

    return {
        "initial_smoothed": float(initial_smoothed),
        "alpha": float(alpha),
        "theta": float(theta),
        "opt_level": opt_level,
        "opt_alpha": opt_alpha,
        "opt_theta": opt_theta,
    }


# =============================================================================
# Optimization Wrapper
# =============================================================================

def _run_theta_optimization(y, model_type, par, nmse=3):
    """Run optimization and return fitted model dict.

    Handles dtype conversion, optimizer dispatch, and result packaging.
    """
    opt_level = par["opt_level"]
    opt_alpha = par["opt_alpha"]
    opt_theta = par["opt_theta"]
    init_level = par["initial_smoothed"]
    init_alpha = par["alpha"]
    init_theta = par["theta"]

    # Build initial parameter vector
    x0_list = []
    if opt_level:
        x0_list.append(0.0)  # unconstrained, centered at 0
    if opt_alpha:
        x0_list.append(0.0)  # sigmoid(0) = 0.5 → alpha ≈ 0.545
    if opt_theta:
        x0_list.append(0.0)  # sigmoid(0) = 0.5 → theta ≈ 50.5
    if not x0_list:
        x0_list.append(0.0)  # dummy

    y_jax = jnp.asarray(y, dtype=jnp.float32)
    x0 = jnp.array(x0_list, dtype=jnp.float32)

    # Run optimization
    opt_level_val, opt_alpha_val, opt_theta_val = _jit_optimize_theta(
        y_jax, int(model_type), x0,
        jnp.asarray(init_level, dtype=jnp.float32),
        jnp.asarray(init_alpha, dtype=jnp.float32),
        jnp.asarray(init_theta, dtype=jnp.float32),
        opt_level, opt_alpha, opt_theta, nmse,
    )

    # Update parameters with optimized values
    level = float(opt_level_val)
    alpha = float(opt_alpha_val)
    theta = float(opt_theta_val)

    # Final evaluation with optimized parameters
    amse, residuals, states, mse = _pegels_resid(
        y_jax, int(model_type),
        jnp.asarray(level, dtype=jnp.float32),
        jnp.asarray(alpha, dtype=jnp.float32),
        jnp.asarray(theta, dtype=jnp.float32),
        nmse,
    )

    return {
        "mse": float(mse),
        "amse": amse,
        "residuals": residuals,
        "states": states,
        "par": {
            "initial_smoothed": level,
            "alpha": alpha,
            "theta": theta,
        },
        "n": len(y),
        "modeltype": model_type.name if isinstance(model_type, ModelType) else ModelType(model_type).name,
        "mean_y": float(jnp.mean(y_jax)),
        "m": 1,  # set by caller
    }


# =============================================================================
# Forecast
# =============================================================================

@functools.partial(jax.jit, static_argnums=(1, 4, 5))
def _forecast_theta(last_state, model_type, alpha, theta, n, h):
    """Generate h-step-ahead forecasts using lax.scan.

    Args:
        last_state: Final state vector [level, mean_y, An, Bn, mu] (5,)
        model_type: ModelType int value (static)
        alpha, theta: Model parameters
        n: Number of observations in training set (static)
        h: Forecast horizon (static)
    """
    dtype = last_state.dtype
    level, mean_y, An, Bn, _ = last_state

    forecast_step_fn = _FORECAST_STEP_FNS[int(model_type)]

    # Build carry: (level, mean_y, An, Bn, alpha, theta, step_idx)
    init_carry = (level, mean_y, An, Bn, alpha, theta, jnp.asarray(n, dtype=dtype))

    # Scan over h forecast steps
    _, forecasts = lax.scan(forecast_step_fn, init_carry, jnp.arange(h, dtype=dtype))

    return forecasts


# =============================================================================
# Prediction Interval Samples
# =============================================================================

@functools.partial(jax.jit, static_argnums=(6, 7))
def _compute_pi_samples(last_state, alpha, theta, sigma, n, mean_y, h, n_samples, seed=0):
    """Monte Carlo prediction interval samples.

    Generates n_samples stochastic forecast paths of length h.
    Each path: deterministic mu + N(0, sigma) noise, then update states.

    Returns: (h, n_samples) array of sample forecasts.
    """
    dtype = last_state.dtype
    level, meany, An, Bn, _ = last_state

    key = jrandom.PRNGKey(seed)
    samples = jnp.zeros((h, n_samples), dtype=dtype)

    # Pre-cast constants to dtype to avoid float64 promotion
    one = jnp.asarray(1.0, dtype=dtype)
    six = jnp.asarray(6.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    eps_c = jnp.asarray(_EPSILON, dtype=dtype)

    def body_fn(j, val):
        smoothed, my, A, B, samp, rng = val
        i = n + j

        # Deterministic mu
        i_f = jnp.asarray(i, dtype=dtype)
        decay = (one - alpha) ** i_f
        decay_next = (one - alpha) ** (i_f + one)
        trend_correction = A * decay + B * (one - decay_next) / jnp.maximum(alpha, eps_c)
        mu = smoothed + (one - one / theta) * trend_correction

        # Random noise
        rng, subkey = jrandom.split(rng)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=dtype) * sigma

        # Sample
        s = mu + eps

        # Update states with sample mean (cast to maintain dtype)
        s_mean = jnp.mean(s).astype(dtype)
        new_smoothed = (alpha * s_mean + (one - alpha) * smoothed).astype(dtype)
        new_my = ((i_f * my + s_mean) / (i_f + one)).astype(dtype)
        new_B = (((i_f - one) * B + six * (s_mean - my) / (i_f + one)) / (i_f + two)).astype(dtype)
        new_A = (new_my - new_B * (i_f + two) / two).astype(dtype)

        samp = samp.at[j].set(s)
        return new_smoothed, new_my, new_A, new_B, samp, rng

    smoothed, my, A, B, samples, _ = lax.fori_loop(
        0, h, body_fn,
        (jnp.asarray(level, dtype=dtype),
         jnp.asarray(mean_y, dtype=dtype),
         jnp.asarray(An, dtype=dtype),
         jnp.asarray(Bn, dtype=dtype),
         samples, key),
    )

    return samples


# =============================================================================
# Full Pipeline Functions
# =============================================================================

def _fit_theta_model(y, m, modeltype_str, initial_smoothed=None, alpha=None,
                     theta=None, nmse=3):
    """Fit a single theta model variant.

    Args:
        y: Time series (JAX array)
        m: Season length
        modeltype_str: One of "STM", "OTM", "DSTM", "DOTM"
        initial_smoothed, alpha, theta: Initial parameters (None = optimize)
        nmse: Number of MSE steps

    Returns:
        Model dict with fitted parameters, residuals, states, mse.
    """
    model_type = ModelType[modeltype_str]
    par = _initparamtheta(initial_smoothed, alpha, theta, y, model_type)
    result = _run_theta_optimization(y, model_type, par, nmse=nmse)
    result["m"] = m
    return result


def _forecast_from_model(obj, h, level=None):
    """Generate forecasts from a fitted theta model dict.

    Args:
        obj: Fitted model dict from _fit_theta_model
        h: Forecast horizon
        level: List of confidence levels (0-100), or None

    Returns:
        Dict with 'mean' and optional 'lo-{lv}', 'hi-{lv}' keys.
    """
    n = obj["n"]
    states = obj["states"]
    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = ModelType[obj["modeltype"]]

    # Get last state
    last_state = states[-1]

    # Generate forecasts
    forecasts = _forecast_theta(
        last_state, int(model_type),
        jnp.asarray(alpha, dtype=jnp.float32),
        jnp.asarray(theta, dtype=jnp.float32),
        n, h,
    )

    res = {"mean": forecasts}

    # Prediction intervals via Monte Carlo
    if level is not None:
        sigma = jnp.std(obj["residuals"][3:], ddof=1)
        samples = _compute_pi_samples(
            last_state,
            jnp.asarray(alpha, dtype=jnp.float32),
            jnp.asarray(theta, dtype=jnp.float32),
            sigma,
            n,
            jnp.asarray(obj["mean_y"], dtype=jnp.float32),
            h, 200,
        )

        for lv in level:
            min_q = (100 - lv) / 200.0
            max_q = min_q + lv / 100.0
            res[f"lo-{lv}"] = jnp.quantile(samples, min_q, axis=1)
            res[f"hi-{lv}"] = jnp.quantile(samples, max_q, axis=1)

    # Recompose if seasonal decomposition was applied
    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res


def _auto_theta(y, m, model=None, initial_smoothed=None, alpha=None,
                theta=None, nmse=3, decomposition_type="multiplicative"):
    """Auto-select best theta model variant.

    Tests all 4 model types (or a single specified one), selects by MSE.
    Handles seasonal decomposition if data has significant seasonality.

    Args:
        y: Time series (JAX array)
        m: Season length
        model: None (auto-select) or one of "STM", "OTM", "DSTM", "DOTM"
        decomposition_type: "multiplicative" or "additive"

    Returns:
        Model dict for the best model.
    """
    # Constant series shortcut
    if is_constant(y):
        return _fit_theta_model(
            y, m, "STM", initial_smoothed=float(jnp.mean(y)) / 2.0,
            alpha=0.5, theta=2.0, nmse=nmse,
        )

    # Seasonal decomposition test
    decompose = False
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m)[1:]
        stat = jnp.sqrt((1.0 + 2.0 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = bool(jnp.abs(r[-1]) / stat > norm.ppf(0.95))

    y_decompose = None
    seas_forecast = None
    data_positive = bool(jnp.min(y) > 0)

    if decompose:
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m)['seasonal']
        if decomposition_type == "multiplicative" and bool(jnp.any(y_decompose < 0.01)):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m)['seasonal']
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(y=y_decompose, h=m, season_length=m, fitted=False)

    # Validate model type
    if model is not None and model not in ("STM", "OTM", "DSTM", "DOTM"):
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    if n <= 3:
        raise NotImplementedError("Series too short (n <= 3)")

    # Model selection
    if model is None:
        model_types = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        model_types = [model]

    best_model = None
    best_mse = float('inf')
    for mtype in model_types:
        fit = _fit_theta_model(
            y, m, mtype, initial_smoothed=initial_smoothed,
            alpha=alpha, theta=theta, nmse=nmse,
        )
        fit_mse = fit["mse"]
        if not jnp.isnan(fit_mse) and fit_mse < best_mse:
            best_model = fit
            best_mse = fit_mse

    if best_model is None:
        raise Exception("No model able to be fitted")

    # Attach seasonal decomposition info
    if decompose:
        if decomposition_type == "multiplicative":
            best_model["residuals"] = best_model["residuals"] * y_decompose
        else:
            best_model["residuals"] = best_model["residuals"] + y_decompose
        best_model["decompose"] = True
        best_model["decomposition_type"] = decomposition_type
        best_model["seas_forecast"] = dict(seas_forecast)

    return best_model


def _forward_theta(fitted_model, y):
    """Re-fit using the same model type and pre-fitted parameters."""
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return _auto_theta(
        y=y, m=m, model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha, theta=theta,
    )


# =============================================================================
# AutoTheta Class
# =============================================================================

class AutoTheta(BaseForecaster):
    r"""AutoTheta model.

    Automatically selects the best Theta (Standard Theta Model ('STM'),
    Optimized Theta Model ('OTM'), Dynamic Standard Theta Model ('DSTM'),
    Dynamic Optimized Theta Model ('DOTM')) model using MSE.

    Args:
        season_length (int, default=1): Number of observations per unit of time.
        decomposition_type (str, default="multiplicative"): Seasonal decomposition
            type, 'multiplicative' or 'additive'.
        model (Optional[str], optional): Controlling Theta Model. By default
            searches the best model.
        alias (str, default="AutoTheta"): Custom name of the model.
        prediction_intervals (Optional[ConformalIntervals], optional): Information
            to compute conformal prediction intervals.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = None,
        alias: str = "AutoTheta",
        prediction_intervals: Optional[ConformalIntervals] = None,
        conformal_params: Optional[ConformalIntervals] = None,
    ):
        self.season_length = season_length
        self.decomposition_type = decomposition_type
        self.model = model
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        if conformal_params is None:
            self.conformal_params = ConformalIntervals()
        else:
            if not isinstance(conformal_params, ConformalIntervals):
                raise TypeError("conformal_params must be a ConformalIntervals object.")
            self.conformal_params = conformal_params

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None):
        r"""Fit the AutoTheta model.

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            X (jnp.ndarray, optional): Optional exogenous of shape (t, n_x).

        Returns:
            AutoTheta: Fitted model.
        """
        y = ensure_float(y)
        self.model_ = _auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        self.model_["fitted"] = y - self.model_["residuals"]
        _store_cs(self, y, X)
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted AutoTheta.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray, optional): Optional exogenous of shape (h, n_x).
            level (List[float], optional): Confidence levels (0-100) for
                prediction intervals.

        Returns:
            dict: Dictionary with 'mean' and optional interval keys.
        """
        fcst = _forecast_from_model(self.model_, h=h, level=level)
        if self.prediction_intervals is not None and level is not None:
            fcst = _add_predict_conformal_intervals(self, fcst, level)
        return fcst

    def predict_in_sample(self, level: Optional[List[int]] = None):
        r"""Access fitted AutoTheta insample predictions.

        Args:
            level (List[float], optional): Confidence levels (0-100) for
                prediction intervals.

        Returns:
            dict: Dictionary with 'fitted' and optional interval keys.
        """
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            se = jnp.std(self.model_["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Memory-efficient AutoTheta predictions.

        Fits and forecasts without storing model state.

        Args:
            y (jnp.ndarray): Clean time series of shape (t,).
            h (int): Forecast horizon.
            level (List[float], optional): Confidence levels (0-100).
            fitted (bool, default=False): Whether to return insample predictions.

        Returns:
            dict: Forecast results.
        """
        y = ensure_float(y)
        mod = _auto_theta(
            y=y,
            m=self.season_length,
            model=self.model,
            decomposition_type=self.decomposition_type,
        )
        res = _forecast_from_model(mod, h, level=level)

        if self.prediction_intervals is not None and level is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)

        if fitted:
            res["fitted"] = y - mod["residuals"]

        if level is not None and fitted:
            se = jnp.std(mod["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)

        return res

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Apply fitted AutoTheta model to a new time series.

        Uses the model type and parameters from the original fit.

        Args:
            y (jnp.ndarray): Clean time series of shape (n,).
            h (int): Forecast horizon.
            level (List[float], optional): Confidence levels (0-100).
            fitted (bool, default=False): Whether to return insample predictions.

        Returns:
            dict: Forecast results.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = _forward_theta(self.model_, y=y)
        res = _forecast_from_model(mod, h, level=level)
        if self.prediction_intervals is not None and level is not None:
            res = _add_conformal_intervals(self, fcst=res, y=y, X=X, level=level)
        if fitted:
            res["fitted"] = y - mod["residuals"]
        if level is not None and fitted:
            se = jnp.std(mod["residuals"][3:], ddof=1)
            res = _add_fitted_pi(res=res, se=se, level=level)
        return res


class Theta(AutoTheta):
    r"""Standard Theta Method (STM).

    A simplified version of AutoTheta that always uses the Standard Theta Model.

    Args:
        season_length (int): Number of observations per unit of time. Default 1.
        decomposition_type (str): Seasonal decomposition type. Default 'multiplicative'.
        alias (str): Custom name of the model. Default 'Theta'.
        prediction_intervals (Optional[ConformalIntervals]): Conformal prediction
            interval parameters. Default None.
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        alias: str = "Theta",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ):
        super().__init__(
            season_length=season_length,
            model="STM",
            decomposition_type=decomposition_type,
            alias=alias,
            prediction_intervals=prediction_intervals,
        )
