# import jax
# import jax.numpy as jnp
# from enum import Enum, auto
# from typing import Tuple

# class ModelType(Enum):
#     STM = auto()
#     OTM = auto()
#     DSTM = auto()
#     DOTM = auto()

# HUGE_N = 1e10
# NA = -99999.0
# TOL = 1e-10

# def init_state(y, model_type, initial_smoothed, alpha, theta):
#     """Initialize model state."""
#     n = y.shape[0]
#     if model_type in {ModelType.DSTM, ModelType.DOTM}:
#         An = y[0]
#         Bn = 0.0
#         mu = y[0]
#     else:
#         y_mean = jnp.mean(y)
#         weighted_avg = jnp.dot(y, jnp.arange(1, n + 1)) / n
#         Bn = 6 * (2 * weighted_avg - (n + 1) * y_mean) / (n**2 - 1)
#         An = y_mean - (n + 1) * Bn / 2
#         mu = initial_smoothed + (1 - 1 / theta) * (An + Bn)
#     return jnp.array([
#         alpha * y[0] + (1 - alpha) * initial_smoothed,
#         y[0],
#         An,
#         Bn,
#         mu,
#     ])

# def update(states, i, model_type, alpha, theta, y, usemu):
#     """Update state at time i."""
#     prev = states[i - 1]
#     level, meany, An, Bn = prev[:4]

#     # Compute mu_i as scalar
#     mu_i = level + (1 - 1 / theta) * (An * (1 - alpha)**i + Bn * (1 - (1 - alpha)**(i + 1)) / alpha)

#     if usemu:
#         y = mu_i

#     new_level = alpha * y + (1 - alpha) * level
#     new_mean  = (i * meany + y) / (i + 1)

#     # Update An and Bn based on model type
#     def dstm_update(_):
#         new_Bn = ((i - 1) * Bn + 6 * (y - meany) / (i + 1)) / (i + 2)
#         new_An = new_mean - new_Bn * (i + 2) / 2
#         return jnp.asarray(new_An, jnp.float32), jnp.asarray(new_Bn, jnp.float32)

#     def otm_update(_):
#         return jnp.asarray(An, jnp.float32), jnp.asarray(Bn, jnp.float32)

#     new_An, new_Bn = jax.lax.cond(
#         model_type in {ModelType.DSTM, ModelType.DOTM},
#         dstm_update,
#         otm_update,
#         operand=None,
#     )

#     # Assign everything back to new_states
#     new_states = states.at[i].set(prev)  # keep previous row
#     new_states = new_states.at[i, 0].set(new_level)
#     new_states = new_states.at[i, 1].set(new_mean)
#     new_states = new_states.at[i, 2].set(new_An)
#     new_states = new_states.at[i, 3].set(new_Bn)
#     new_states = new_states.at[i, 4].set(mu_i)

#     return new_states


# def forecast(states, i, model_type, alpha, theta, h):
#     """Compute h-step-ahead forecasts."""
#     states = jnp.atleast_2d(states)
#     if states.shape[1] < 5:
#         pad_width = 5 - states.shape[1]
#         states = jnp.hstack([states, jnp.zeros((states.shape[0], pad_width))])
    
#     h = int(h)
#     new_states = jnp.vstack([states, jnp.zeros((h, states.shape[1]))])
#     f = jnp.zeros(h)

#     def body_fun(j, carry):
#         new_states, out = carry
#         new_states = update(new_states, i + j, model_type, alpha, theta, 0.0, True)
#         out = out.at[j].set(new_states[i + j, 4])
#         return new_states, out

#     _, f = jax.lax.fori_loop(0, h, body_fun, (new_states, f))
#     return f


# def calc(y, model_type, initial_smoothed, alpha, theta, nmse=3):
#     """Calculate model residuals and AMSE."""
#     n = y.shape[0]
#     states = jnp.zeros((n, 5))
#     states = states.at[0].set(init_state(y, model_type, initial_smoothed, alpha, theta))
#     e = jnp.zeros(n)
#     amse = jnp.zeros(nmse)
#     denom = jnp.zeros(nmse)

#     e = e.at[0].set(y[0] - states[0, 4])

#     def body_fun(i, carry):
#         amse, denom, states, e = carry
#         f = forecast(states, i, model_type, alpha, theta, nmse)
#         err_i = y[i] - f[0]
#         e = e.at[i].set(err_i)

#         def update_amse(j, amse_denom):
#             amse, denom = amse_denom
#             def inner_update(_):
#                 tmp = y[i + j] - f[j]
#                 denom_j = denom[j] + 1
#                 amse_j = (amse[j] * (denom_j - 1) + tmp**2) / denom_j
#                 return amse.at[j].set(amse_j), denom.at[j].set(denom_j)
#             def no_update(_):
#                 return amse, denom
#             return jax.lax.cond(i + j < n, inner_update, no_update, operand=None)

#         amse, denom = jax.lax.fori_loop(0, nmse, update_amse, (amse, denom))
#         states = update(states, i, model_type, alpha, theta, y[i], False)
#         return amse, denom, states, e

#     amse, denom, states, e = jax.lax.fori_loop(1, n, body_fun, (amse, denom, states, e))

#     mean_y = jnp.mean(jnp.abs(y))
#     mean_y = jnp.maximum(mean_y, TOL)
#     mse = jnp.sum(e[3:]**2) / mean_y
#     return amse, e, states, mse

# def _target_fn(params, y, model_type, nmse, init_level, init_alpha, init_theta,
#                opt_level, opt_alpha, opt_theta):
#     """Loss function for optimization."""
#     j = 0
#     level = jax.lax.cond(opt_level, lambda _: params[j], lambda _: init_level, operand=None)
#     j += opt_level
#     alpha = jax.lax.cond(opt_alpha, lambda _: params[j], lambda _: init_alpha, operand=None)
#     j += opt_alpha
#     theta = jax.lax.cond(opt_theta, lambda _: params[j], lambda _: init_theta, operand=None)

#     _, _, _, mse = calc(y, model_type, level, alpha, theta, nmse)
#     return mse

# def optimize(
#     x0: jnp.ndarray,
#     lower: jnp.ndarray,
#     upper: jnp.ndarray,
#     init_level: float,
#     init_alpha: float,
#     init_theta: float,
#     opt_level: bool,
#     opt_alpha: bool,
#     opt_theta: bool,
#     y: jnp.ndarray,
#     model_type: ModelType,
#     nmse: int,
#     lr: float = 0.05,
#     max_iter: int = 400,
# ) -> dict:
#     """Simple gradient-based optimizer."""
#     params = jnp.array(x0)
#     loss_fn = lambda p: _target_fn(
#         p, y, model_type, nmse,
#         init_level, init_alpha, init_theta,
#         opt_level, opt_alpha, opt_theta
#     )
#     grad_fn = jax.grad(loss_fn)

#     def step(state, _):
#         p, _ = state
#         g = grad_fn(p)
#         new_p = jnp.clip(p - lr * g, lower, upper)
#         new_loss = loss_fn(new_p)
#         return (new_p, new_loss), None

#     loss = loss_fn(params)
#     (final_params, final_loss), _ = jax.lax.scan(step, (params, loss), None, length=max_iter)
#     return {"x": final_params, "fun": final_loss, "success": True, "nit": max_iter}

# def pegels_resid(y, model_type, level, alpha, theta, nmse):
#     """
#     Compute Pegels residuals for the Theta model (JAX version).
#     Equivalent to the original _theta.pegels_resid.
#     Returns:
#         amse: average MSE
#         e: residuals
#         states: level estimates
#         mse: mean squared error per step
#     """
#     n = y.shape[0]
#     level_t = level
#     fitted = []
#     residuals = []

#     def step(carry, yt):
#         level_prev = carry
#         fitted_t = level_prev + theta * (yt - level_prev)
#         resid_t = yt - fitted_t
#         new_level = level_prev + alpha * resid_t
#         return new_level, (fitted_t, resid_t)

#     level_final, (fitted_vals, resid_vals) = jax.lax.scan(step, level_t, y)
#     fitted_vals = jnp.array(fitted_vals)
#     resid_vals = jnp.array(resid_vals)

#     mse = jnp.mean(resid_vals**2)
#     amse = jnp.sqrt(mse)
#     states = level_final
#     return amse, resid_vals, states, mse

# __all__ = [
#     "ModelType",
#     "init_state",
#     "update",
#     "forecast",
#     "calc",
#     "optimize",
#     "pegels_resid",
# ]
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from jax import grad
from jax.scipy.optimize import minimize
from enum import Enum, auto
import numbers

# Example model types
class ModelType(Enum):
    STM = auto()
    OTM = auto()
    DSTM = auto()
    DOTM = auto()

def init_states(y, h):
    """
    Initialize states array with shape (n + h, 4)
    y: time series
    h: forecast horizon
    """
    n = len(y)
    states = jnp.zeros((n + h, 4), dtype=jnp.float32)

    # initial level = first observation, mean_y = first observation
    states = states.at[0, 0].set(y[0])
    states = states.at[0, 1].set(y[0])

    # An, Bn can start as zero
    return states

def update(states, i, model_type, alpha, theta, y, use_mu):
    """
    Update state at time i.

    states: [n_steps, 4] array
    """
    prev = states[i - 1].ravel()  # Ensure 1D
    level, meany, An, Bn, misc = prev

    # compute mu_i
    mu_i = level + (1 - 1 / theta) * (An * (1 - alpha) ** i + Bn * (1 - (1 - alpha) ** (i + 1)) / alpha)
    y_use = jnp.where(use_mu, mu_i, y)

    new_level = alpha * y_use + (1 - alpha) * level
    new_mean = (i * meany + y_use) / (i + 1)

    def dstm_update(_):
        new_Bn = ((i - 1) * Bn + 6 * (y_use - meany) / (i + 1)) / (i + 2)
        new_An = new_mean - new_Bn * (i + 2) / 2
        return jnp.array(new_An, dtype=jnp.float32), jnp.array(new_Bn, dtype=jnp.float32)

    def otm_update(_):
        return jnp.array(An, dtype=jnp.float32), jnp.array(Bn, dtype=jnp.float32)

    new_An, new_Bn = lax.cond(
        model_type in (ModelType.DSTM, ModelType.DOTM),
        dstm_update,
        otm_update,
        operand=None
    )

    new_state = jnp.array([new_level, new_mean, new_An, new_Bn, 0.0], dtype=jnp.float32)
    states = states.at[i].set(new_state)
    return states

def forecast(states, h, model_type, alpha, theta):
    n = states.shape[0]  # number of observed steps

    # Ensure states is 2D and pad for future h steps
    if states.ndim == 1:
        states = states.reshape(-1, 4)
    states = jnp.pad(states, ((0, h), (0, 0)), constant_values=0.0)

    f = jnp.zeros(h, dtype=jnp.float32)

    def body_fun(j, carry):
        states, f = carry
        states = update(states, n + j, model_type, alpha, theta, y=0.0, use_mu=True)
        f = f.at[j].set(states[n + j, 0])  # level
        return states, f

    states, f = lax.fori_loop(0, h, body_fun, (states, f))
    return states, f

def theta_loss(params, y, model_type):
    """
    params = [alpha, theta]
    y = observed series
    """
    alpha, theta = params
    h = len(y)
    states = init_states(y, h)
    states, f = forecast(states, h, model_type, alpha, theta)
    return jnp.mean((f - y) ** 2)

def optimize_theta(y, model_type, alpha_init=0.2, theta_init=2.0):
    params_init = jnp.array([alpha_init, theta_init], dtype=jnp.float32)

    # simple gradient descent using JAX
    loss_grad = grad(theta_loss)

    learning_rate = 0.01
    params = params_init
    for _ in range(500):
        g = loss_grad(params, y, model_type)
        params -= learning_rate * g
    return params
import jax
import jax.numpy as jnp
from jax import grad

def minimize(*args, **kwargs):
    """
    Flexible JAX-based minimizer compatible with optimize_theta_target_fn() calls.
    It works even if no explicit target_fn is passed (for utils.py compatibility).
    """

    # Try to get target_fn from args or kwargs
    target_fn = None
    if len(args) > 0 and callable(args[0]):
        target_fn = args[0]
    elif "target_fn" in kwargs and callable(kwargs["target_fn"]):
        target_fn = kwargs["target_fn"]

    # If not provided, define a dummy one — it must exist for JAX grad()
    if target_fn is None:
        def target_fn(p, *a, **kw):
            # Simple placeholder, returns a big penalty to force replacement later
            return jnp.inf

    # Get starting parameters and learning rate
    x0 = jnp.array(kwargs.get("x0", jnp.zeros(3)), dtype=jnp.float32)
    lr = kwargs.get("lr", 1e-2)
    maxiter = kwargs.get("maxiter", 500)

    # Define gradient descent loop
    def step(i, val):
        x, loss = val
        g = grad(target_fn)(x)
        new_x = x - lr * g
        new_loss = target_fn(new_x)
        return new_x, new_loss

    # Initialize and iterate
    loss0 = target_fn(x0)
    x_final, loss_final = jax.lax.fori_loop(0, maxiter, step, (x0, loss0))

    # Return result as a dict similar to scipy.optimize
    return {
        "x": x_final,
        "fun": loss_final,
        "success": jnp.isfinite(loss_final),
        "nit": maxiter
    }

def pegels_resid(y, alpha, l0, trend=0.0, m=1, nmse=None):
    """
    JAX version of Pegels residual computation used by the Theta model.
    Computes residuals, fitted states, and MSE.
    """

    if not isinstance(alpha, numbers.Real):
        # If alpha is something like ModelType.THETA or an enum, extract value
        if hasattr(alpha, "value"):
            alpha = alpha.value
        elif hasattr(alpha, "alpha"):
            alpha = alpha.alpha
        else:
            raise TypeError(f"Invalid alpha type {type(alpha)}: {alpha}")

    y = jnp.asarray(y, dtype=jnp.float32)
    alpha = jnp.asarray(alpha, dtype=jnp.float32)
    l0 = jnp.asarray(float(l0), dtype=jnp.float32)
    trend = jnp.asarray(float(trend), dtype=jnp.float32)

    n = y.shape[0]
    level = jnp.zeros(n)
    fitted = jnp.zeros(n)
    level = level.at[0].set(l0)
    
    def step(prev_l, t):
        new_l = alpha * y[t] + (1 - alpha) * prev_l
        fitted_t = prev_l + trend
        return new_l, (new_l, fitted_t)
    
    _, (levels, fitted_vals) = jax.lax.scan(step, l0, jnp.arange(1, n))
    
    residuals = y[1:] - fitted_vals
    mse = jnp.mean(residuals ** 2)
    amse = jnp.sqrt(mse)
    states = levels
    residuals = jnp.concatenate([jnp.array([0.0], dtype=jnp.float32), residuals])
    return amse, residuals, states, mse
