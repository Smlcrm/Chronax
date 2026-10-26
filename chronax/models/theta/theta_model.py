"""
Theta Model — JAX-accelerated implementation.

Implements the Standard Theta Method (STM), Optimized Theta Method (OTM),
Dynamic Standard Theta Method (DSTM), and Dynamic Optimized Theta Method (DOTM)
with full JAX JIT compilation and two-phase ADAM + L-BFGS optimization.

This module contains the core computational engine (state initialization,
step functions, optimization, forecasting, Monte Carlo PI sampling).
The ``AutoTheta`` and ``Theta`` classes live in ``auto_theta.py`` and are
re-exported here for backward compatibility.

Features:
- Four model variants: STM, OTM, DSTM, DOTM
- Additive and multiplicative seasonal decomposition
- Monte Carlo prediction intervals
- JAX JIT compilation for performance
- Two-phase ADAM + L-BFGS optimization

Implementation Notes:
- State vector: [level, mean_y, An, Bn, mu] (5 elements)
- forecast[i] = SES_level[i-1] + (1 - 1/theta) * trend_correction(i)
- trend_correction(i) = An*(1-alpha)^i + Bn*(1-(1-alpha)^(i+1))/alpha
- Step functions pre-built per model type for static JIT dispatch

References:
    Jose A. Fiorucci et al. (2016). "Models for optimising the theta method and
    their relationship to state space models". International Journal of Forecasting.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax import lax
from jax import jit
from chronax.utils import (
    is_constant,
    seasonal_decompose,
    acf,
    _repeat_val_seas,
    _seasonal_naive,
    _jax_norm_ppf,
)
from chronax.models.base_forecaster import BaseForecaster
from typing import Optional, List
from chronax.utils import ConformalIntervals



# =============================================================================
# Constants
# =============================================================================

class ModelType:
    """Integer constants and name maps for Theta model variants.

    Attributes
    ----------
    STM : int
        Standard Theta Method (theta fixed at 2.0).
    OTM : int
        Optimized Theta Method (theta optimized).
    DSTM : int
        Dynamic Standard Theta Method (dynamic An/Bn, theta fixed at 2.0).
    DOTM : int
        Dynamic Optimized Theta Method (dynamic An/Bn, theta optimized).
    """

    STM = 1
    OTM = 2
    DSTM = 3
    DOTM = 4
    _name_map = {1: "STM", 2: "OTM", 3: "DSTM", 4: "DOTM"}
    _from_name = {"STM": 1, "OTM": 2, "DSTM": 3, "DOTM": 4}
    _all = (1, 2, 3, 4)

_EPSILON = 1e-10
_ADAM_STEPS = 30
_ADAM_LR = 0.02
_LBFGS_STEPS = 20

__all__ = ['Theta', 'AutoTheta', 'ModelType']


# =============================================================================
# Core Math — Module-level JIT'd functions
# =============================================================================

def _init_state(y: jnp.ndarray, model_type: int, initial_smoothed: float,
                alpha: float, theta: float) -> jnp.ndarray:
    """Initialize the 5-element state vector [level, mean_y, An, Bn, mu].

    For STM/OTM: An, Bn come from OLS linear regression on full series.
    For DSTM/DOTM: An = y[0], Bn = 0.

    Parameters
    ----------
    y : jnp.ndarray
        Time series of shape (n,).
    model_type : int
        ModelType constant (static arg for JIT).
    initial_smoothed : float
        Initial smoothed level value.
    alpha : float
        Smoothing parameter.
    theta : float
        Theta parameter.

    Returns
    -------
    jnp.ndarray
        State vector of shape (5,): [level, mean_y, An, Bn, mu].
    """
    n = y.shape[0]

    # OLS regression for STM/OTM
    y_mean = jnp.mean(y)
    weighted_avg = jnp.dot(y, jnp.arange(1, n + 1, dtype=y.dtype)) / n
    Bn_static = 6.0 * (2.0 * weighted_avg - (n + 1.0) * y_mean) / (n ** 2 - 1.0)
    An_static = y_mean - (n + 1.0) * Bn_static / 2.0

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

_init_state = jax.jit(_init_state, static_argnums=(1,))


def _make_step_fn(model_type_val: int) -> tuple[callable, callable]:
    """Return (step_fn, forecast_step_fn) for the given model type.

    Step functions are designed for ``lax.scan``. Forecast step functions
    use mu instead of the observation for level updates.

    Parameters
    ----------
    model_type_val : int
        ModelType constant.

    Returns
    -------
    tuple[callable, callable]
        step_fn : Fit step: carry = (level, mean_y, An, Bn, alpha, theta,
        step_idx), input = y_t, output = (e_t, mu_t).
        forecast_step_fn : Forecast step: same carry, input unused,
        output = mu.
    """
    is_dynamic = model_type_val in (ModelType.DSTM, ModelType.DOTM)

    def step_fn(carry: tuple, y_t: jnp.ndarray) -> tuple[tuple, tuple]:
        """Single fit step for lax.scan.

        Parameters
        ----------
        carry : tuple
            (level, mean_y, An, Bn, alpha, theta, step_idx).
        y_t : jnp.ndarray
            Observation at time t.

        Returns
        -------
        tuple[tuple, tuple]
            Updated carry and (residual, fitted_value).
        """
        level, mean_y, An, Bn, alpha, theta, i = carry

        decay = (1.0 - alpha) ** i
        decay_next = (1.0 - alpha) ** (i + 1.0)
        trend_correction = An * decay + Bn * (1.0 - decay_next) / jnp.maximum(alpha, _EPSILON)
        mu = level + (1.0 - 1.0 / theta) * trend_correction
        e_t = y_t - mu
        new_level = alpha * y_t + (1.0 - alpha) * level
        new_mean_y = (i * mean_y + y_t) / (i + 1.0)

        if is_dynamic:
            new_Bn = ((i - 1.0) * Bn + 6.0 * (y_t - mean_y) / (i + 1.0)) / (i + 2.0)
            new_An = new_mean_y - new_Bn * (i + 2.0) / 2.0
        else:
            new_An = An
            new_Bn = Bn

        new_carry = (new_level, new_mean_y, new_An, new_Bn, alpha, theta, i + 1.0)
        return new_carry, (e_t, mu)

    def forecast_step_fn(carry: tuple, _unused: jnp.ndarray) -> tuple[tuple, jnp.ndarray]:
        """Single forecast step for lax.scan.

        Uses mu (the forecast) as the pseudo-observation for SES level updates.

        Parameters
        ----------
        carry : tuple
            (level, mean_y, An, Bn, alpha, theta, step_idx).
        _unused : jnp.ndarray
            Ignored scan input.

        Returns
        -------
        tuple[tuple, jnp.ndarray]
            Updated carry and forecast value mu.
        """
        level, mean_y, An, Bn, alpha, theta, i = carry

        decay = (1.0 - alpha) ** i
        decay_next = (1.0 - alpha) ** (i + 1.0)
        trend_correction = An * decay + Bn * (1.0 - decay_next) / jnp.maximum(alpha, _EPSILON)
        mu = level + (1.0 - 1.0 / theta) * trend_correction

        # SES level update with mu as "observation"
        new_level = alpha * mu + (1.0 - alpha) * level
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
for _mt in ModelType._all:
    _sfn, _fsfn = _make_step_fn(_mt)
    _STEP_FNS[int(_mt)] = _sfn
    _FORECAST_STEP_FNS[int(_mt)] = _fsfn


def _pegels_resid(y: jnp.ndarray, model_type: int, initial_smoothed: jnp.ndarray,
                  alpha: jnp.ndarray, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Theta model residuals and MSE using ``lax.scan``.

    Parameters
    ----------
    y : jnp.ndarray
        Time series of shape (n,).
    model_type : int
        ModelType constant (static arg for JIT).
    initial_smoothed : jnp.ndarray
        Initial smoothed level value.
    alpha : jnp.ndarray
        Smoothing parameter.
    theta : jnp.ndarray
        Theta parameter.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        residuals : One-step-ahead errors of shape (n,).
        final_state : Final state vector of shape (5,): [level, mean_y, An, Bn, mu].
        mse : Scalar optimization objective: sum(e[3:]^2) / max(mean(|y|), eps).
    """
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

    # Concatenate first residual with scan output
    residuals = jnp.concatenate([e0[None], residuals_rest])

    # Final state: only last fitted value is needed (for forecasting init)
    final_level, final_mean_y, final_An, final_Bn, _, _, _ = final_carry
    final_state = jnp.stack([final_level, final_mean_y, final_An, final_Bn, mus_rest[-1]])

    # MSE objective: skip first 3 residuals (initialization transient),
    # normalize by mean(|y|) for scale-invariant optimization.
    mean_abs_y = jnp.maximum(jnp.mean(jnp.abs(y)), _EPSILON)
    mse = jnp.sum(residuals[3:] ** 2) / mean_abs_y

    return residuals, final_state, mse

_pegels_resid = jax.jit(_pegels_resid, static_argnums=(1,))


# =============================================================================
# Optimizer — ADAM + L-BFGS (following Holt-Winters pattern)
# =============================================================================

def _jit_optimize_theta(y: jnp.ndarray, model_type: int, x0: jnp.ndarray,
                        init_level: jnp.ndarray, init_alpha: jnp.ndarray, init_theta: jnp.ndarray,
                        opt_level: bool, opt_alpha: bool, opt_theta: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JIT-compiled ADAM + L-BFGS optimization for Theta model parameters.

    Parameters are reparameterized via sigmoid to enforce bounds.
    Compiled once per (model_type, opt_level, opt_alpha, opt_theta,
    y.shape, y.dtype), then cached.

    Parameters
    ----------
    y : jnp.ndarray
        Time series of shape (n,).
    model_type : int
        ModelType constant (static arg).
    x0 : jnp.ndarray
        Initial unconstrained parameter vector.
    init_level : jnp.ndarray
        Initial smoothed level (used when opt_level is False).
    init_alpha : jnp.ndarray
        Initial alpha (used when opt_alpha is False).
    init_theta : jnp.ndarray
        Initial theta (used when opt_theta is False).
    opt_level : bool
        Whether to optimize the level (static arg).
    opt_alpha : bool
        Whether to optimize alpha (static arg).
    opt_theta : bool
        Whether to optimize theta (static arg).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        level : Optimized initial smoothed level.
        alpha : Optimized smoothing parameter in [0.1, 0.99].
        theta : Optimized theta parameter in [1.0, 100.0].
    """
    dtype = y.dtype

    # Scaling
    y_std = jnp.maximum(jnp.std(y), _EPSILON)
    y_mean = jnp.mean(y)

    # Bounds for sigmoid reparameterization
    alpha_lo = jnp.asarray(0.1, dtype=dtype)
    alpha_hi = jnp.asarray(0.99, dtype=dtype)
    theta_lo = jnp.asarray(1.0, dtype=dtype)
    theta_hi = jnp.asarray(100.0, dtype=dtype)

    def _to_constrained(p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Map unconstrained parameters to bounded space.

        Parameters
        ----------
        p : jnp.ndarray
            Unconstrained parameter vector.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (level, alpha, theta) in their constrained ranges.
        """
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

    def loss_fn(p_unconstrained: jnp.ndarray) -> jnp.ndarray:
        """Compute MSE loss from unconstrained parameters.

        Parameters
        ----------
        p_unconstrained : jnp.ndarray
            Unconstrained parameter vector.

        Returns
        -------
        jnp.ndarray
            Scalar MSE loss.
        """
        p_unconstrained = jnp.asarray(p_unconstrained, dtype=dtype)
        level, alpha, theta = _to_constrained(p_unconstrained)
        _, _, mse = _pegels_resid(y, model_type, level, alpha, theta)
        return mse

    # Phase 1: ADAM warm-up — essential for navigating the flat, ridged
    # MSE landscape where level/alpha/theta trade off.  L-BFGS alone
    # (including jaxopt.LBFGSB) gets trapped in local minima; Adam's
    # momentum-based exploration finds the basin, then L-BFGS refines.
    vg_fn = jax.value_and_grad(loss_fn)
    adam_opt = optax.adam(_ADAM_LR)
    adam_state = adam_opt.init(x0)

    def _adam_step(carry: tuple, _: jnp.ndarray) -> tuple[tuple, None]:
        """Single ADAM optimization step for lax.scan.

        Parameters
        ----------
        carry : tuple
            (params, optimizer_state, best_params, best_loss).
        _ : jnp.ndarray
            Unused scan input.

        Returns
        -------
        tuple[tuple, None]
            Updated carry and None.
        """
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = adam_opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        # Track the EVALUATED point p (where `loss` was measured), not the
        # post-update new_p — best_loss corresponds to p, so storing new_p
        # mismatches the pair and seeds L-BFGS from the wrong iterate. Matches
        # the L-BFGS phase below and the ETS backend's convention.
        best_p = jnp.where(improved, p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    inf_val = jnp.asarray(float('inf'), dtype=dtype)
    adam_carry = (x0, adam_state, x0, inf_val)
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

    def _lbfgs_step(carry: tuple, _: jnp.ndarray) -> tuple[tuple, None]:
        """Single L-BFGS optimization step for lax.scan.

        Parameters
        ----------
        carry : tuple
            (params, optimizer_state, best_params, best_loss).
        _ : jnp.ndarray
            Unused scan input.

        Returns
        -------
        tuple[tuple, None]
            Updated carry and None.
        """
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

_jit_optimize_theta = jax.jit(_jit_optimize_theta, static_argnums=(1, 6, 7, 8))


# =============================================================================
# Parameter Initialization (Pure Python)
# =============================================================================

def _initparamtheta(initial_smoothed: float | None, alpha: float | None,
                    theta: float | None, y: jnp.ndarray, model_type: int) -> dict:
    """Determine initial values and optimization flags per model type.

    STM/DSTM: theta fixed at 2.0, optimize level and alpha.
    OTM/DOTM: optimize level, alpha, and theta.

    Parameters
    ----------
    initial_smoothed : float or None
        Initial smoothed level. None triggers optimization.
    alpha : float or None
        Smoothing parameter. None triggers optimization.
    theta : float or None
        Theta parameter. None triggers optimization (OTM/DOTM only).
    y : jnp.ndarray
        Time series (used for default initial_smoothed = y[0]/2).
    model_type : int
        ModelType constant.

    Returns
    -------
    dict
        Keys: 'initial_smoothed', 'alpha', 'theta' (JAX float32 scalars),
        'opt_level', 'opt_alpha', 'opt_theta' (bool flags).
    """
    if initial_smoothed is None:
        initial_smoothed = y[0] / 2.0
        opt_level = True
    else:
        opt_level = False

    if alpha is None:
        alpha = 0.5
        opt_alpha = True
    else:
        opt_alpha = False

    if model_type in (ModelType.STM, ModelType.DSTM):
        theta = 2.0
        opt_theta = False
    else:
        if theta is None:
            theta = 2.0
            opt_theta = True
        else:
            opt_theta = False

    return {
        "initial_smoothed": jnp.asarray(initial_smoothed, dtype=jnp.float32),
        "alpha": jnp.asarray(alpha, dtype=jnp.float32),
        "theta": jnp.asarray(theta, dtype=jnp.float32),
        "opt_level": opt_level,
        "opt_alpha": opt_alpha,
        "opt_theta": opt_theta,
    }


# =============================================================================
# Optimization Wrapper
# =============================================================================

def _run_theta_optimization(y: jnp.ndarray, model_type: int, par: dict) -> dict:
    """Run optimization and return fitted model dict.

    Handles dtype conversion, optimizer dispatch, and result packaging.

    Parameters
    ----------
    y : jnp.ndarray
        Time series.
    model_type : int
        ModelType constant.
    par : dict
        From ``_initparamtheta``: initial values and optimization flags.

    Returns
    -------
    dict
        Keys: 'mse' (JAX scalar), 'residuals' (ndarray),
        'final_state' (ndarray of shape (5,)), 'par' (dict with
        'initial_smoothed', 'alpha', 'theta' as JAX scalars), 'n' (int),
        'modeltype' (str), 'mean_y' (JAX scalar), 'm' (int).
    """
    opt_level = par["opt_level"]
    opt_alpha = par["opt_alpha"]
    opt_theta = par["opt_theta"]
    init_level = par["initial_smoothed"]
    init_alpha = par["alpha"]
    init_theta = par["theta"]

    y_jax = jnp.asarray(y, dtype=jnp.float32)

    # Early return when no parameters need optimization
    if not opt_level and not opt_alpha and not opt_theta:
        residuals, final_state, mse = _pegels_resid(
            y_jax, int(model_type),
            jnp.asarray(init_level, dtype=jnp.float32),
            jnp.asarray(init_alpha, dtype=jnp.float32),
            jnp.asarray(init_theta, dtype=jnp.float32),
        )
        return {
            "mse": mse,
            "residuals": residuals,
            "final_state": final_state,
            "par": {"initial_smoothed": jnp.asarray(init_level, dtype=jnp.float32),
                    "alpha": jnp.asarray(init_alpha, dtype=jnp.float32),
                    "theta": jnp.asarray(init_theta, dtype=jnp.float32)},
            "n": len(y),
            "modeltype": ModelType._name_map[int(model_type)],
            "mean_y": jnp.mean(y_jax),
            "m": 1,
        }

    # Build initial parameter vector
    x0_list = []
    if opt_level:
        x0_list.append(0.0)  # unconstrained, centered at 0
    if opt_alpha:
        x0_list.append(0.0)  # sigmoid(0) = 0.5 → alpha ≈ 0.545
    if opt_theta:
        x0_list.append(-4.595119850134589)  # logit((2-1)/(100-1)) → theta ≈ 2.0

    x0 = jnp.array(x0_list, dtype=jnp.float32)

    # Run optimization
    opt_level_val, opt_alpha_val, opt_theta_val = _jit_optimize_theta(
        y_jax, int(model_type), x0,
        jnp.asarray(init_level, dtype=jnp.float32),
        jnp.asarray(init_alpha, dtype=jnp.float32),
        jnp.asarray(init_theta, dtype=jnp.float32),
        opt_level, opt_alpha, opt_theta,
    )

    # Final evaluation with optimized parameters
    residuals, final_state, mse = _pegels_resid(
        y_jax, int(model_type),
        opt_level_val, opt_alpha_val, opt_theta_val,
    )

    return {
        "mse": mse,
        "residuals": residuals,
        "final_state": final_state,
        "par": {
            "initial_smoothed": opt_level_val,
            "alpha": opt_alpha_val,
            "theta": opt_theta_val,
        },
        "n": len(y),
        "modeltype": ModelType._name_map[int(model_type)],
        "mean_y": jnp.mean(y_jax),
        "m": 1,  # set by caller
    }


# =============================================================================
# Forecast
# =============================================================================

def _forecast_theta(last_state: jnp.ndarray, model_type: int, alpha: jnp.ndarray,
                    theta: jnp.ndarray, n: int, h: int) -> jnp.ndarray:
    """Generate h-step-ahead forecasts using ``lax.scan``.

    Parameters
    ----------
    last_state : jnp.ndarray
        Final state vector of shape (5,): [level, mean_y, An, Bn, mu].
    model_type : int
        ModelType constant (static arg).
    alpha : jnp.ndarray
        Smoothing parameter.
    theta : jnp.ndarray
        Theta parameter.
    n : int
        Number of observations in training set (static arg).
    h : int
        Forecast horizon (static arg).

    Returns
    -------
    jnp.ndarray
        Point forecasts of shape (h,).
    """
    dtype = last_state.dtype
    level, mean_y, An, Bn, _ = last_state

    forecast_step_fn = _FORECAST_STEP_FNS[int(model_type)]

    # Build carry: (level, mean_y, An, Bn, alpha, theta, step_idx)
    init_carry = (level, mean_y, An, Bn, alpha, theta, jnp.asarray(n, dtype=dtype))

    # Scan over h forecast steps
    _, forecasts = lax.scan(forecast_step_fn, init_carry, jnp.arange(h, dtype=dtype))

    return forecasts

_forecast_theta = jax.jit(_forecast_theta, static_argnums=(1, 4, 5))


# =============================================================================
# Prediction Interval Samples
# =============================================================================

def _compute_pi_samples(last_state: jnp.ndarray, model_type: int, alpha: jnp.ndarray,
                        theta: jnp.ndarray, sigma: jnp.ndarray, n: int, mean_y: jnp.ndarray,
                        h: int, n_samples: int, seed: int = 0) -> jnp.ndarray:
    """Generate Monte Carlo prediction interval samples.

    Produces ``n_samples`` stochastic forecast paths of length ``h``.
    Each path adds N(0, sigma) noise to the deterministic forecast
    and evolves its own independent level/trend state.

    For static models (STM, OTM), An and Bn remain constant — they are
    derived from OLS on the full training series. Only dynamic models
    (DSTM, DOTM) update these coefficients during simulation.

    Parameters
    ----------
    last_state : jnp.ndarray
        Final state vector of shape (5,): [level, mean_y, An, Bn, mu].
    model_type : int
        ModelType constant (static arg for JIT).
    alpha : jnp.ndarray
        Smoothing parameter.
    theta : jnp.ndarray
        Theta parameter.
    sigma : jnp.ndarray
        Residual standard deviation.
    n : int
        Number of observations in training set.
    mean_y : jnp.ndarray
        Mean of the training series.
    h : int
        Forecast horizon (static arg).
    n_samples : int
        Number of Monte Carlo paths (static arg).
    seed : int, default 0
        PRNG seed.

    Returns
    -------
    jnp.ndarray
        Sample forecasts of shape (h, n_samples).
    """
    dtype = last_state.dtype
    level, _, An, Bn, _ = last_state
    is_dynamic = (model_type == ModelType.DSTM) | (model_type == ModelType.DOTM)

    key = jrandom.PRNGKey(seed)
    samples = jnp.zeros((h, n_samples), dtype=dtype)

    # Pre-cast constants and inputs to dtype to avoid float64 promotion
    sigma = jnp.asarray(sigma, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    six = jnp.asarray(6.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)
    eps_c = jnp.asarray(_EPSILON, dtype=dtype)

    def body_fn(j: int, val: tuple) -> tuple:
        """Single MC simulation step for lax.fori_loop.

        Parameters
        ----------
        j : int
            Forecast step index (0-based).
        val : tuple
            (smoothed, mean_y_acc, A, B, samples_array, prng_key),
            all arrays of shape (n_samples,) except samples (h, n_samples)
            and key (scalar).

        Returns
        -------
        tuple
            Updated val with new state and samples[j] filled.
        """
        smoothed, mean_y_acc, A, B, samp, rng = val
        i = n + j

        i_f = jnp.asarray(i, dtype=dtype)
        decay = (one - alpha) ** i_f
        decay_next = (one - alpha) ** (i_f + one)
        trend_correction = A * decay + B * (one - decay_next) / jnp.maximum(alpha, eps_c)
        mu = smoothed + (one - one / theta) * trend_correction

        rng, subkey = jrandom.split(rng)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=dtype) * sigma
        s = mu + eps  # (n_samples,)

        # Per-path state updates — each sample path evolves independently
        new_smoothed = alpha * s + (one - alpha) * smoothed  # (n_samples,)
        new_mean_y = (i_f * mean_y_acc + s) / (i_f + one)   # (n_samples,)

        # Only update An/Bn for dynamic models; static models keep OLS values
        if is_dynamic:
            new_B = ((i_f - one) * B + six * (s - mean_y_acc) / (i_f + one)) / (i_f + two)
            new_A = new_mean_y - new_B * (i_f + two) / two
        else:
            new_A = A
            new_B = B

        samp = samp.at[j].set(s)
        return new_smoothed, new_mean_y, new_A, new_B, samp, rng

    _, _, _, _, samples, _ = lax.fori_loop(
        0, h, body_fn,
        (jnp.broadcast_to(jnp.asarray(level, dtype=dtype), (n_samples,)),
         jnp.broadcast_to(jnp.asarray(mean_y, dtype=dtype), (n_samples,)),
         jnp.broadcast_to(jnp.asarray(An, dtype=dtype), (n_samples,)),
         jnp.broadcast_to(jnp.asarray(Bn, dtype=dtype), (n_samples,)),
         samples, key),
    )

    return samples

_compute_pi_samples = jax.jit(_compute_pi_samples, static_argnums=(1, 7, 8))


# =============================================================================
# Full Pipeline Functions
# =============================================================================

def _fit_theta_model(y: jnp.ndarray, m: int, modeltype_str: str,
                     initial_smoothed: float | None = None, alpha: float | None = None,
                     theta: float | None = None) -> dict:
    """Fit a single theta model variant.

    Parameters
    ----------
    y : jnp.ndarray
        Time series.
    m : int
        Season length.
    modeltype_str : str
        One of 'STM', 'OTM', 'DSTM', 'DOTM'.
    initial_smoothed : float or None, default None
        Initial smoothed level. None triggers optimization.
    alpha : float or None, default None
        Smoothing parameter. None triggers optimization.
    theta : float or None, default None
        Theta parameter. None triggers optimization (OTM/DOTM only).

    Returns
    -------
    dict
        Fitted model dict from ``_run_theta_optimization`` with 'm' set.
    """
    model_type = ModelType._from_name[modeltype_str]
    par = _initparamtheta(initial_smoothed, alpha, theta, y, model_type)
    result = _run_theta_optimization(y, model_type, par)
    result["m"] = m
    return result


def _forecast_from_model(obj: dict, h: int, level: list | None = None, n_samples: int = 200) -> dict:
    """Generate forecasts from a fitted theta model dict.

    Parameters
    ----------
    obj : dict
        Fitted model dict from ``_fit_theta_model``.
    h : int
        Forecast horizon.
    level : list of float or None, default None
        Confidence levels (0-100) for prediction intervals.
    n_samples : int, default 200
        Number of Monte Carlo samples for prediction intervals.

    Returns
    -------
    dict
        Keys: 'mean' (ndarray of shape (h,)), and optionally
        'lo-{lv}', 'hi-{lv}' for each level.
    """
    n = obj["n"]
    last_state = obj["final_state"]
    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]

    # Use stored int model_type if available (vmap-safe), fall back to string lookup
    if "model_type_int" in obj:
        model_type_int = obj["model_type_int"]
    else:
        model_type_int = ModelType._from_name[obj["modeltype"]]

    # Generate forecasts — handle traced model_type_int under vmap
    alpha_f32 = jnp.asarray(alpha, dtype=jnp.float32)
    theta_f32 = jnp.asarray(theta, dtype=jnp.float32)
    if isinstance(model_type_int, (jnp.ndarray, jax.Array)):
        # Under vmap: compute all model types and select
        # ModelType values are 1-4 (STM=1, OTM=2, DSTM=3, DOTM=4)
        all_forecasts = jnp.stack([
            _forecast_theta(last_state, mt, alpha_f32, theta_f32, n, h)
            for mt in ModelType._all  # (1, 2, 3, 4)
        ])
        forecasts = all_forecasts[model_type_int - 1]  # convert 1-indexed to 0-indexed
    else:
        forecasts = _forecast_theta(
            last_state, int(model_type_int), alpha_f32, theta_f32, n, h,
        )

    res = {"mean": forecasts}

    # Prediction intervals via Monte Carlo (not called under vmap — level is None)
    if level is not None:
        sigma = jnp.std(obj["residuals"][3:], ddof=1)
        # Eager path (level is not None ⇒ not under the CV vmap): read the
        # winning variant's int so DSTM/DOTM sample their own dynamics.
        mt_int = int(model_type_int)
        samples = _compute_pi_samples(
            last_state,
            mt_int,
            alpha_f32,
            theta_f32,
            sigma,
            n,
            jnp.asarray(obj["mean_y"], dtype=jnp.float32),
            h, n_samples,
        )

        for lv in level:
            min_q = (100 - lv) / 200.0
            max_q = min_q + lv / 100.0
            res[f"lo-{lv}"] = jnp.quantile(samples, min_q, axis=1)
            res[f"hi-{lv}"] = jnp.quantile(samples, max_q, axis=1)

    # Recompose if seasonal decomposition was applied (vmap-safe: traced boolean)
    should_recompose = obj.get("decompose", False)
    if isinstance(should_recompose, (jnp.ndarray, jax.Array)) or should_recompose:
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        is_additive = obj.get("decomposition_type_is_additive", jnp.bool_(True))
        for key in list(res.keys()):
            recomp_add = res[key] + seas_forecast
            recomp_mult = res[key] * seas_forecast
            recomposed = jnp.where(is_additive, recomp_add, recomp_mult)
            res[key] = jnp.where(should_recompose, recomposed, res[key])

    return res


def _auto_theta(y: jnp.ndarray, m: int, model: str | None = None,
                initial_smoothed: float | None = None, alpha: float | None = None,
                theta: float | None = None, decomposition_type: str = "multiplicative") -> dict:
    """Auto-select best theta model variant.

    Tests all 4 model types (or a single specified one), selects by MSE.
    Handles seasonal decomposition if data has significant seasonality.

    vmap-compatible: all data-dependent control flow uses JAX primitives.

    Parameters
    ----------
    y : jnp.ndarray
        Time series.
    m : int
        Season length.
    model : str or None, default None
        One of 'STM', 'OTM', 'DSTM', 'DOTM', or None for auto-selection.
    initial_smoothed : float or None, default None
        Initial smoothed level.
    alpha : float or None, default None
        Smoothing parameter.
    theta : float or None, default None
        Theta parameter.
    decomposition_type : str, default 'multiplicative'
        Seasonal decomposition type: 'multiplicative' or 'additive'.

    Returns
    -------
    dict
        Fitted model dict for the best model variant.

    Raises
    ------
    ValueError
        If ``model`` is not a valid model type string.
    NotImplementedError
        If the series has 3 or fewer observations.
    """
    # Validate model type (static check — OK under vmap)
    if model is not None and model not in ("STM", "OTM", "DSTM", "DOTM"):
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)  # static under vmap
    if n <= 3:
        raise NotImplementedError("Series too short (n <= 3)")

    # --- Constant series model (always computed, selected via jnp.where at end) ---
    y_is_constant = is_constant(y)
    const_model = _fit_theta_model(
        y, m, "STM", initial_smoothed=jnp.mean(y) / 2.0,
        alpha=0.5, theta=2.0,
    )

    # --- Seasonal decomposition (traced booleans, precompute both paths) ---
    if m >= 4 and n >= 2 * m:  # static shape checks
        r = acf(y, nlags=m)[1:]
        stat = jnp.sqrt((1.0 + 2.0 * jnp.sum(r[:-1] ** 2)) / n)
        should_decompose = jnp.abs(r[-1]) / stat > _jax_norm_ppf(0.95)
    else:
        should_decompose = jnp.bool_(False)

    data_positive = jnp.min(y) > 0

    # Always compute additive decomposition
    if m >= 4 and n >= 2 * m:  # static guard to avoid errors on short series
        y_dec_add = seasonal_decompose(y, model="additive", period=m)['seasonal']

        if decomposition_type == "multiplicative":  # static string from __init__
            y_dec_mult = seasonal_decompose(y, model="multiplicative", period=m)['seasonal']
            use_additive = ~data_positive | jnp.any(y_dec_mult < 0.01)
            y_decompose = jnp.where(use_additive, y_dec_add, y_dec_mult)
            is_additive = use_additive
        else:
            y_decompose = y_dec_add
            is_additive = jnp.bool_(True)

        y_deseasonalized = jnp.where(is_additive, y - y_decompose, y / y_decompose)
        y_final = jnp.where(should_decompose, y_deseasonalized, y)
        seas_forecast = _seasonal_naive(y=y_decompose, h=m, season_length=m, fitted=False)
    else:
        y_final = y
        y_decompose = jnp.zeros_like(y)
        is_additive = jnp.bool_(True)
        seas_forecast = {"mean": jnp.zeros(m, dtype=y.dtype)}

    # --- Model selection (loop unrolls at trace time since model_types is static) ---
    model_types = ["STM", "OTM", "DSTM", "DOTM"] if model is None else [model]
    fits = [_fit_theta_model(y_final, m, mt, initial_smoothed=initial_smoothed,
            alpha=alpha, theta=theta) for mt in model_types]

    # Select best by MSE using JAX ops (no Python if on traced values)
    mses = jnp.array([f["mse"] for f in fits])
    mses_safe = jnp.where(jnp.isnan(mses), jnp.inf, mses)
    best_idx = jnp.argmin(mses_safe)

    # Select best model dict — stack numeric/array values, index by best_idx
    ref = fits[0]
    if len(fits) > 1:
        best_model = {}
        for k in ref:
            v = ref[k]
            if isinstance(v, dict):
                # Handle nested dicts (e.g., "par")
                best_model[k] = {
                    pk: jnp.stack([f[k][pk] for f in fits])[best_idx]
                    for pk in v
                }
            elif isinstance(v, (jnp.ndarray, jax.Array)):
                best_model[k] = jnp.stack([f[k] for f in fits])[best_idx]
            else:
                # Static values (ints, strings) — same across fits
                best_model[k] = v
        # Store all model type ints for downstream forecast dispatch
        model_type_ints = jnp.array([ModelType._from_name[mt] for mt in model_types])
        best_model["model_type_int"] = model_type_ints[best_idx]
    else:
        best_model = fits[0]
        best_model["model_type_int"] = jnp.int32(ModelType._from_name[model_types[0]])

    # --- Apply decomposition info (always present for consistent pytree structure) ---
    best_model["decompose"] = should_decompose
    best_model["decomposition_type_is_additive"] = is_additive
    best_model["seas_forecast"] = seas_forecast
    # Recompose residuals conditionally
    residuals_add = best_model["residuals"] + y_decompose
    residuals_mult = best_model["residuals"] * y_decompose
    residuals_recomposed = jnp.where(is_additive, residuals_add, residuals_mult)
    best_model["residuals"] = jnp.where(should_decompose, residuals_recomposed, best_model["residuals"])

    # --- Choose constant model vs selected model ---
    def select_model(use_const, const_mod, sel_mod):
        result = {}
        for k in sel_mod:
            if isinstance(sel_mod[k], (jnp.ndarray, jax.Array)):
                result[k] = jnp.where(use_const, const_mod.get(k, sel_mod[k]), sel_mod[k])
            else:
                result[k] = sel_mod[k]
        return result

    # Ensure const_model has the same keys
    const_model["decompose"] = jnp.bool_(False)
    const_model["decomposition_type_is_additive"] = jnp.bool_(True)
    const_model["seas_forecast"] = seas_forecast
    const_model["model_type_int"] = jnp.int32(ModelType._from_name["STM"])

    best_model = select_model(y_is_constant, const_model, best_model)

    return best_model


def _forward_theta(fitted_model: dict, y: jnp.ndarray) -> dict:
    """Re-fit using the same model type and pre-fitted parameters.

    Parameters
    ----------
    fitted_model : dict
        Previously fitted model dict.
    y : jnp.ndarray
        New time series.

    Returns
    -------
    dict
        Fitted model dict for the new series.
    """
    m = fitted_model["m"]
    # The stored "modeltype" string is fits[0]'s ("STM") after an auto merge;
    # the winning variant is carried by model_type_int. forward() is eager.
    if "model_type_int" in fitted_model:
        model = ModelType._name_map[int(fitted_model["model_type_int"])]
    else:
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
# Backward-compatible re-exports (AutoTheta and Theta live in auto_theta.py)
# =============================================================================

def __getattr__(name: str):
    """Lazy import of AutoTheta and Theta from auto_theta to avoid circular imports.

    Parameters
    ----------
    name : str
        Attribute name being accessed.

    Returns
    -------
    type
        The requested class.

    Raises
    ------
    AttributeError
        If the attribute is not found.
    """
    if name in ("AutoTheta", "Theta"):
        from chronax.models.theta.auto_theta import AutoTheta, Theta  # noqa: F811
        if name == "AutoTheta":
            return AutoTheta
        return Theta
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
