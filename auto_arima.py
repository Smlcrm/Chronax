from __future__ import annotations
import math
import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple, Union,NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
# import jaxopt
import jax.scipy.optimize
from stl import stl_decompose
Array = jnp.ndarray

from base_forecaster import BaseForecaster 

from utils import _quantiles
# =============================================================================
# LOW-LEVEL TRANSFORMS (from arima.cpp)
# =============================================================================

def partrans(p: int, raw: Array) -> Array:
    """
    Transform unconstrained parameters to stationary AR coefficients.
    
    Uses tanh transform followed by Levinson-Durbin recursion.
    Optimization: Inner loop vectorized for JAX efficiency.
    
    Args:
        p: Number of AR coefficients (static int)
        raw: Unconstrained parameters (shape (p,) or larger)
    
    Returns:
        new: Stationary AR coefficients (shape (p,))
    """
    if p == 0:
        # Return empty array with correct float type
        return jnp.array([], dtype=jnp.float64)
    
    # Ensure inputs are correct type and sliced to p
    # In JAX, slicing with static 'p' is preferred
    raw = jnp.asarray(raw, dtype=jnp.float64)[:p]
    
    # 1. Tanh transform: maps to (-1, 1) -> Partial Autocorrelations
    new = jnp.tanh(raw)
    
    # 2. Levinson-Durbin recursion
    # We iterate p-1 times. Since p is usually small and static, 
    # Python loop unrolling is efficient here.
    for j in range(1, p):
        # 'a' is the reflection coefficient for this step (new[j])
        a = new[j]
        
        # Snapshot the previous coefficients required for the update
        # We need indices 0 to j-1
        prev_coeffs = new[:j]
        
        # Vectorized update:
        # work[k] = new[k] - a * new[j - k - 1]
        # In vector form: new[:j] -= a * reverse(new[:j])
        update = a * prev_coeffs[::-1]
        
        # Apply update
        new = new.at[:j].add(-update)
        
    return new

def invpartrans(p: int, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse transform: stationary AR coefficients to unconstrained parameters.
    
    Reverse Levinson-Durbin recursion.
    Optimization: Inner loop vectorized.
    
    Args:
        p: Number of AR coefficients
        phi: Stationary AR coefficients (shape (p,))
    
    Returns:
        out: Unconstrained parameters (shape (p,)) ready for arctanh
    """
    if p == 0:
        return jnp.array([], dtype=jnp.float64)
    
    # Ensure input is float64 and sliced
    phi = jnp.asarray(phi, dtype=jnp.float64)[:p]
    
    # We work on 'out' in place (conceptually)
    out = phi
    
    # Iterate backwards from the last coefficient down to the second
    # Python range: p-1, p-2, ..., 1
    for j in range(p - 1, 0, -1):
        # The last element 'a' at this step is the reflection coefficient (PACF value)
        a = out[j]
        
        # In the forward step we did: new = old - a * old_reversed
        # Here we invert that: old = (new + a * new_reversed) / (1 - a^2)
        denom = 1.0 - a * a
        
        # 1. Slice the relevant section
        current_slice = out[:j]
        
        # 2. Vectorized calculation
        # Matches: (out[k] + a * out[j - k - 1]) / denom
        update = (current_slice + a * current_slice[::-1]) / denom
        
        # 3. Update state
        out = out.at[:j].set(update)
    
    # Final transform to unconstrained space (-inf, inf)
    # Clip serves as numerical safety for values near +/- 1.0
    out = jnp.arctanh(jnp.clip(out, -0.999999, 0.999999))
    
    return out

def arima_transpar(params: Array, arma: Tuple[int, ...], trans: bool) -> Tuple[Array, Array]:
    """
    Transform parameters and expand seasonal components.
    
    Equivalent to arima.cpp arima_transpar().
    Optimization: Vectorized interaction terms + Static ARMA handling.
    
    Args:
        params: Input parameters (phi, theta, seasonal_phi, seasonal_theta)
        arma: Tuple of ints (mp, mq, msp, msq, ns, d, D)
        trans: Whether to apply stationarity transform (partrans)
    
    Returns:
        (phi, theta): Expanded AR and MA coefficient vectors
    """
    # 1. UNPACK DIRECTLY (Do not convert to jnp.array)
    # This keeps these variables as static Python integers.
    mp, mq, msp, msq, ns, d, D = arma
    
    # 2. Setup
    p = mp + ns * msp 
    q = mq + ns * msq 
    
    # Ensure params is float64
    params = jnp.asarray(params, dtype=jnp.float64)
    params_work = params
    
    # 3. Transform (Stationarity)
    if trans:
        if mp > 0:
            params_work = params_work.at[:mp].set(partrans(mp, params[:mp]))
        v = mp + mq
        if msp > 0:
            params_work = params_work.at[v:v+msp].set(partrans(msp, params[v:v+msp]))
            
    phi = jnp.zeros(p, dtype=jnp.float64)
    theta = jnp.zeros(q, dtype=jnp.float64)
    
    # 4. Expansion (Vectorized Logic)
    if ns > 0 and (msp > 0 or msq > 0):
        # --- AR ---
        phi = phi.at[:mp].set(params_work[:mp])
        for j in range(msp):
            seas_param = params_work[mp + mq + j]
            idx_main = (j + 1) * ns - 1
            phi = phi.at[idx_main].add(seas_param)
            
            if mp > 0:
                indices = (j + 1) * ns + jnp.arange(mp)
                updates = -params_work[:mp] * seas_param
                phi = phi.at[indices].add(updates)
        
        # --- MA ---
        theta = theta.at[:mq].set(params_work[mp:mp+mq])
        for j in range(msq):
            seas_param = params_work[mp + mq + msp + j]
            idx_main = (j + 1) * ns - 1
            theta = theta.at[idx_main].add(seas_param)
            
            if mq > 0:
                indices = (j + 1) * ns + jnp.arange(mq)
                updates = params_work[mp : mp + mq] * seas_param
                theta = theta.at[indices].add(updates)
    else:
        # Fast path
        phi = phi.at[:mp].set(params_work[:mp])
        theta = theta.at[:mq].set(params_work[mp:mp+mq])
    
    return phi, theta


def arima_undopars(x: Array, arma: Tuple[int, ...]) -> Array:
    """
    Undo parameter transform (apply partrans).
    
    Transforms unconstrained parameters (optimizer space) into 
    constrained stationary coefficients (model space).
    
    Equivalent to arima.cpp arima_undopars().
    
    Args:
        x: Unconstrained parameters (1D array)
        arma: Tuple of ints (mp, mq, msp, msq, ns, d, D)
        
    Returns:
        out: Transformed (stationary) parameters
    """
    # 1. Unpack Tuple (Static Integers for JAX)
    mp, mq, msp, msq, ns, d, D = arma
    
    # 2. Ensure float64
    x = jnp.asarray(x, dtype=jnp.float64)
    
    # Initialize output (start with copy of input)
    # Note: In JAX 'out = x' is just a reference, but .at[].set creates the modified copy.
    out = x
    
    # 3. Transform Non-Seasonal AR part
    if mp > 0:
        # Calls the optimized partrans function we defined earlier
        out = out.at[:mp].set(partrans(mp, x[:mp]))
    
    # 4. Transform Seasonal AR part
    # Seasonal AR parameters are stored after Non-Seasonal AR (mp) and MA (mq)
    v = mp + mq
    if msp > 0:
        out = out.at[v:v+msp].set(partrans(msp, x[v:v+msp]))
    
    return out

# =============================================================================
# DIFFERENCING
# =============================================================================

@partial(jax.jit, static_argnames=['lag', 'differences'])
def diff(x: Array, lag: int, differences: int) -> Array:
    """
    Compute differenced series.
    
    Optimized JAX implementation using direct slicing.
    JIT-compiled with static shape handling.
    
    Args:
        x: Input series
        lag: Lag for differencing (Must be static int)
        differences: Number of times to difference (Must be static int)
    
    Returns:
        Differenced series (shorter than input by lag * differences)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    
    # Check for no-op (Static check)
    if differences < 1:
        return x

    # Iterative differencing
    # JAX unrolls this loop because 'differences' is marked static
    for _ in range(differences):
        # x[lag:]  -> The series shifted forward
        # x[:-lag] -> The series shifted backward
        # Direct subtraction slices off the "NaN" regions automatically
        x = x[lag:] - x[:-lag]
        
    return x

# =============================================================================
# 3. GETQ0_WEB (Doubling Algorithm)
# =============================================================================

@partial(jax.jit, static_argnames=['arma'])
def getQ0(phi: Array, theta: Array, arma: Tuple[int, ...]) -> Array:
    """
    Computes Initial State Covariance P solving P = F P F' + V
    using the Doubling Algorithm (Smith 1968).
    
    This uses the Standard State Space representation (Jones/Pearlman).
    
    Args:
        phi: AR coefficients
        theta: MA coefficients
        arma: (p, q, ...) tuple
        
    Returns:
        P: Initial State Covariance Matrix (r x r)
    """
    # 1. Setup shapes
    p = phi.shape[0]
    q = theta.shape[0]
    r = max(p, q + 1)
    
    phi = jnp.asarray(phi, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)

    # 2. State Transition Matrix F (Standard Companion Form)
    # x_{t+1} = F x_t + G e_{t+1}
    F = jnp.zeros((r, r), dtype=jnp.float64)
    if p > 0: 
        F = F.at[0, :p].set(phi)
    if r > 1: 
        F = F.at[jnp.arange(r - 1) + 1, jnp.arange(r - 1)].set(1.0)
        
    # 3. Process Noise Covariance V = G * G^T
    # G = [1, theta_1, theta_2, ...]
    G = jnp.zeros(r, dtype=jnp.float64)
    G = G.at[0].set(1.0)
    if q > 0: 
        G = G.at[1:q+1].set(theta)
        
    V = jnp.outer(G, G)
    
    # 4. Doubling Algorithm (Iterative Lyapunov Solver)
    # Converges to machine precision in ~16 iterations.
    def body(_, val):
        P, F_mat = val
        # P_new = P + F * P * F.T
        P_new = P + F_mat @ P @ F_mat.T
        # F_new = F * F
        F_new = F_mat @ F_mat
        return (P_new, F_new)
    
    # Initial P = V
    P_final, _ = jax.lax.fori_loop(0, 16, body, (V, F))
    
    return P_final

# =============================================================================
# CSS ESTIMATION
# =============================================================================

@partial(jax.jit, static_argnames=['arma'])
def arima_css(y: Array, arma: Tuple[int, ...], phi: Array, theta: Array) -> Tuple[float, Array]:
    """
    Compute CSS (Conditional Sum of Squares) for ARIMA.
    
    Optimized JAX implementation using:
    1. Vectorized differencing
    2. Parallel convolution for AR terms
    3. jax.lax.scan for recursive MA terms
    
    Args:
        y: Time series
        arma: Tuple of (p, q, P, Q, m, d, D) ints. (MUST be a tuple for JIT)
        phi: AR coefficients (expanded)
        theta: MA coefficients (expanded)
    
    Returns:
        (sigma2, residuals): Estimated variance and residuals
    """
    # 1. Unpack Static Args
    # arma layout: [p, q, P, Q, m, d, D]
    p_ord, q_ord, P_ord, Q_ord, m, d, D = arma
    
    # Cast inputs
    y = jnp.asarray(y, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    
    p = phi.shape[0] # Expanded size
    q = theta.shape[0] # Expanded size
    n = y.shape[0]
    
    # Calculate number of conditional observations to skip
    ncond = p_ord + d + m * (P_ord + D)

    # 2. Apply Differencing
    # We maintain the array length 'n' but "zero out" the lost prefixes 
    # to match the user's specific logic (w[1:] - w[:-1]).
    w = y
    
    # Ordinary Differencing
    for _ in range(d):
        # w[t] = w[t] - w[t-1]
        # We perform this in-place-like using shifted subtraction
        w = w.at[1:].add(-w[:-1])
        w = w.at[0].set(0.0) # The first element becomes invalid (0)

    # Seasonal Differencing
    for _ in range(D):
        # w[t] = w[t] - w[t-m]
        w = w.at[m:].add(-w[:-m])
        w = w.at[:m].set(0.0) # The first m elements become invalid

    # 3. Compute AR Part (Vectorized)
    # The AR component is: Sum(phi[j] * w[t-j-1])
    # Since 'w' is fully known, we can compute the AR contribution for all 't' at once.
    resid = w # Start with the differenced series
    
    if p > 0:
        # Subtract AR terms
        # We loop over lags. Since p is usually small (<100), unrolling this is efficient.
        for j in range(p):
            # Shift w right by j+1
            w_shifted = jnp.roll(w, j + 1)
            # Zero out the wrapped-around elements (though ncond masking handles this later,
            # it's safer to be explicit)
            w_shifted = w_shifted.at[:j+1].set(0.0)
            resid = resid - phi[j] * w_shifted

    # 4. Compute MA Part (Recursive via Scan)
    # This part MUST be sequential because resid[t] depends on resid[t-1]
    # resid[t] = (w[t] - AR_part) - Sum(theta[j] * resid[t-j-1])
    
    if q > 0:
        # Prepare input: The residual containing only AR adjustments
        # We must ensure the first 'ncond' elements are 0, as per CSS definition
        scan_input = resid.at[:ncond].set(0.0)
        
        def ma_step(carry, x):
            # carry: buffer of past residuals [r_{t-1}, r_{t-2}, ..., r_{t-q}]
            # x: current partial residual (w - AR)
            
            # Predict MA effect: dot product of theta and past residuals
            ma_effect = jnp.dot(theta, carry)
            
            # Current full residual
            new_res = x - ma_effect
            
            # Update buffer: shift right, insert new_res at front
            # (Concatenate is cheap for small q)
            new_carry = jnp.concatenate([jnp.array([new_res]), carry[:-1]])
            
            return new_carry, new_res

        # Initial buffer of zeros
        init_carry = jnp.zeros(q)
        
        # Run Scan
        _, resid = jax.lax.scan(ma_step, init_carry, scan_input)
        
    else:
        # If no MA, just enforce the ncond mask
        resid = resid.at[:ncond].set(0.0)

    # 5. Compute Statistics
    # CSS = Sum of squared residuals / valid_count
    # We only consider residuals starting from ncond
    
    # Slice valid part
    valid_resid = resid[ncond:]
    
    # Check for NaNs (infinity handling)
    # StatsForecast typically returns inf if unstable
    is_finite = jnp.all(jnp.isfinite(valid_resid))
    
    ssq = jnp.sum(valid_resid ** 2)
    nu = n - ncond # Assuming all valid for CSS (unlike MLE which handles missing data differently)
    
    # Safe division
    sigma2 = jnp.where(
        (nu > 0) & is_finite,
        ssq / nu,
        jnp.inf
    )
    
    return sigma2, resid

# =============================================================================
# KALMAN FILTER
# =============================================================================

# Define a clean JAX Pytree container for the model
# This maps to the standard notation: x_t = T x_{t-1} + ...; y_t = Z x_t
class StateSpaceModel(NamedTuple):
    T: Array  # Transition Matrix (F in some texts)
    Z: Array  # Observation Matrix (H in some texts)
    V: Array  # Process Noise Covariance (Q in some texts)
    a0: Array # Initial State Mean
    P0: Array # Initial State Covariance

@partial(jax.jit, static_argnames=['arma'])
def make_arima(phi: Array, theta: Array, delta: Array, arma: Tuple[int, ...], kappa: float = 1e6) -> StateSpaceModel:
    """
    Build State-Space matrices for ARIMA using JAX-friendly structures.
    
    This prepares the model for a `jax.lax.scan` Kalman Filter (MatPalm style).
    
    Args:
        phi: AR coefficients
        theta: MA coefficients
        delta: Differencing polynomial
        arma: (p, q, ...) tuple
        kappa: Diffuse prior variance
        
    Returns:
        StateSpaceModel: A NamedTuple containing (T, Z, V, a0, P0)
    """
    # 1. Setup & Casting
    phi = jnp.asarray(phi, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    delta = jnp.asarray(delta, dtype=jnp.float64)
    
    p, q, d = phi.shape[0], theta.shape[0], delta.shape[0]
    r = max(p, q + 1)
    rd = r + d
    
    # 2. Construct Z (Observation Vector)
    # Z = [1, 0..., delta...]
    Z = jnp.zeros(rd, dtype=jnp.float64)
    Z = Z.at[0].set(1.0)
    if d > 0:
        Z = Z.at[r:].set(delta)
        
    # 3. Construct T (Transition Matrix)
    T = jnp.zeros((rd, rd), dtype=jnp.float64)
    
    # -- Stationary Block --
    if p > 0: T = T.at[:p, 0].set(phi)
    if r > 1: T = T.at[jnp.arange(r-1), jnp.arange(r-1)+1].set(1.0)
        
    # -- Non-Stationary Block --
    if d > 0:
        T = T.at[r, :].set(Z) # The "summing" junction
        if d > 1:
            idx = jnp.arange(d - 1)
            T = T.at[r + 1 + idx, r + idx].set(1.0)

    # 4. Construct V (Process Noise) via R vector
    # R = [1, theta_1, theta_2, ...]
    R = jnp.zeros(rd, dtype=jnp.float64)
    R = R.at[0].set(1.0)
    if q > 0: R = R.at[1:q+1].set(theta)
    
    V = jnp.outer(R, R)
    
    # 5. Initialization (a0, P0)
    a0 = jnp.zeros(rd, dtype=jnp.float64)
    P0 = jnp.zeros((rd, rd), dtype=jnp.float64)
    
    # Stationary Covariance (using our getQ0 logic)
    if r > 1:
        Q0 = getQ0(phi, theta, arma)
        P0 = P0.at[:r, :r].set(Q0)
    else:
        # Scalar AR(1) case fallback
        denom = 1.0 - phi[0]**2 if p > 0 else 1.0
        P0 = P0.at[0, 0].set(1.0 / denom)
        
    # Diffuse Covariance for integrated parts
    if d > 0:
        idx_d = jnp.arange(d)
        P0 = P0.at[r + idx_d, r + idx_d].set(kappa)
        
    return StateSpaceModel(T, Z, V, a0, P0)

# -----------------------------------------------------------------------------
# 3. OPTIMIZED PYTREE (arima_like_3) - PREFERRED
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['arma'])
def arima_like(y: Array, mod: StateSpaceModel, arma: Tuple[int, ...]) -> Tuple[float, float, int, Array]:
    """Optimized Kalman Filter using jax.lax.scan."""
    T, Z, V, a0, P0 = mod.T, mod.Z, mod.V, mod.a0, mod.P0
    n = y.shape[0]

    # The Scan Step: "Update Current -> Predict Next"
    # Carry is (a_prior, P_prior, ssq, sumlog, nu)
    def step_prior_carry(carry, y_t):
        a_prior, P_prior, ssq, sumlog, nu = carry
        
        # --- Update Step ---
        v = y_t - jnp.dot(Z, a_prior)
        F = jnp.dot(Z, P_prior @ Z)
        
        safe_F = jnp.where(F < 1e-9, 1e-9, F)
        M = P_prior @ Z
        K = M / safe_F
        
        # Posterior
        a_post = a_prior + K * v
        P_post = P_prior - jnp.outer(K, M)
        
        # --- Predict Step (Next Prior) ---
        a_next_prior = T @ a_post
        P_next_prior = T @ P_post @ T.T + V
        
        # --- Stats Accumulation ---
        # Accumulate only if variance is reasonable
        valid = F < 1e4
        nu_inc = jnp.where(valid, 1, 0)
        ssq_inc = jnp.where(valid, (v**2)/safe_F, 0.0)
        sumlog_inc = jnp.where(valid, jnp.log(safe_F), 0.0)
        std_resid = jnp.where(valid, v / jnp.sqrt(safe_F), 0.0)
        
        # JAX requires float return for ssq/sumlog to avoid type errors in scan
        return (a_next_prior, P_next_prior, ssq + ssq_inc, sumlog + sumlog_inc, nu + nu_inc), std_resid

    # Initial State (Prior at t=0)
    # mod.a0 is a_{1|0} (usually 0)
    # mod.P0 is P_{1|0} (from getQ0)
    init_carry = (mod.a0, mod.P0, 0.0, 0.0, 0)
    
    # Run the fast loop
    final_carry, residuals = jax.lax.scan(step_prior_carry, init_carry, y)
    
    _, _, final_ssq, final_sumlog, final_nu = final_carry
    return final_ssq, final_sumlog, final_nu, residuals


# -----------------------------------------------------------------------------
# 3. OPTIMIZED FORECAST (kalman_forecast_3)
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['n_ahead'])
def kalman_forecast(n_ahead: int, mod: StateSpaceModel) -> Tuple[Array, Array]:
    """
    Optimized Kalman Forecast using Scan.
    
    Args:
        n_ahead: Number of steps to forecast (Static int).
        mod: StateSpaceModel with the STARTING state (usually end of fit).
        
    Returns:
        (forecasts, se): Arrays of shape (n_ahead,)
    """
    T, Z, V, a_start, P_start = mod.T, mod.Z, mod.V, mod.a0, mod.P0
    
    # Define the recursive step
    def step(carry, _):
        a_curr, P_curr = carry
        
        # 1. Predict Next State (Mean)
        a_next = T @ a_curr
        
        # 2. Predict Next Covariance
        # P_{t+1} = T P_t T' + V
        P_next = V + (T @ P_curr @ T.T)
        
        # 3. Calculate Outputs
        # y_hat = Z * a_next
        forecast = jnp.dot(Z, a_next)
        
        # Variance = Z P_next Z'
        # Optimized: Vector-Matrix-Vector product is cheaper than Outer Product summation
        variance = jnp.dot(Z, P_next @ Z)
        
        return (a_next, P_next), (forecast, variance)

    # Run Scan
    # We pass 'None' as the input sequence because we just want to tick 'n_ahead' times
    # length=n_ahead ensures the loop runs correct number of times
    _, (forecasts, se) = jax.lax.scan(step, (a_start, P_start), None, length=n_ahead)
    
    return forecasts, se

# =============================================================================
# OPTIMIZATION KERNELS (Standardized)
# =============================================================================

def _compute_metrics(loglik: Array, sigma2: Array, n_obs: Array, n_params: int) -> Dict[str, Array]:
    """Compute AIC, AICc, and BIC (JIT Safe)."""
    # Cast n_obs to float for division
    n = n_obs.astype(jnp.float64)
    
    # AIC = -2*LogLik + 2*k
    aic = -2.0 * loglik + 2.0 * n_params
    
    # AICc = AIC + Correction
    # We use jnp.where to handle the "if denom > 0" check
    denom = n - n_params - 1.0
    aicc_correction = (2.0 * n_params * (n_params + 1)) / (denom + 1e-10)
    aicc = jnp.where(denom > 0, aic + aicc_correction, jnp.inf)
    
    # BIC = -2*LogLik + k*ln(n)
    bic = -2.0 * loglik + n_params * jnp.log(n)
    
    # Return JAX arrays (do not cast to float)
    return {
        "aic": aic, 
        "aicc": aicc, 
        "bic": bic,
        "loglik": loglik, 
        "sigma2": sigma2
    }

def _unpack_and_adjust(params: Array, y: Array, xreg: Optional[Array], 
                       arma: Tuple[int, ...], ncxreg: int, n_exog: int, include_mean: bool) -> Tuple[Array, Array, Array]:
    """Helper to unpack parameters and adjust series (Guaranteed Shape Fix)."""
    mp, mq, msp, msq, ns, d, D = arma
    narma_total = mp + mq + msp + msq
    
    # 1. Transform ARMA parameters
    phi, theta = arima_transpar(params[:narma_total], arma, trans=True)
    
    # 2. Extract and Pad Coefficients
    # We concatenate a dummy 0.0 to the end of coefs. 
    # This ensures that even if ncxreg is 0, the array has size 1.
    # This prevents the 'slice_sizes must be less than shape' error.
    raw_coefs = params[narma_total:]
    safe_coefs = jnp.concatenate([raw_coefs, jnp.array([0.0])])
    
    y_adj = y
    
    # --- JIT Safe Exogenous Adjustment ---
    safe_xreg = jnp.zeros((y.shape[0], 0)) if xreg is None else xreg
    
    def apply_xreg(args):
        y_in, xr, cf = args
        num_vars = xr.shape[1]
        # Slice from the padded safe_coefs
        beta = jax.lax.dynamic_slice_in_dim(cf, 0, num_vars)
        return y_in - jnp.dot(xr, beta)

    y_adj = jax.lax.cond(
        (n_exog > 0) & (xreg is not None),
        apply_xreg,
        lambda args: args[0],
        (y_adj, safe_xreg, safe_coefs)
    )
    
    # --- JIT Safe Intercept Adjustment ---
    # Because we padded safe_coefs, indexing at n_exog is ALWAYS valid.
    # If n_exog=0 and include_mean=False, it picks the dummy 0.0 we added.
    intercept_val = jax.lax.dynamic_index_in_dim(safe_coefs, n_exog, keepdims=False)
    
    # Final adjustment only if include_mean is truly active
    y_adj = y_adj - jnp.where(include_mean, intercept_val, 0.0)
            
    return y_adj, phi, theta

def _objective_css(params: Array, y: Array, xreg: Optional[Array], delta: Array,
                   arma: Tuple[int, ...], ncxreg: int, n_exog: int, include_mean: bool) -> float:
    y_adj, phi, theta = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
    sigma2, _ = arima_css(y_adj, arma, phi, theta)
    return jnp.log(sigma2 + 1e-8)

def _objective_ml(params: Array, y: Array, xreg: Optional[Array], delta: Array,
                  arma: Tuple[int, ...], ncxreg: int, n_exog: int, include_mean: bool, **kwargs) -> float:
    """
    ML Loss (Negative Log Likelihood).
    """
    y_adj, phi, theta = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
    mod = make_arima(phi, theta, delta, arma)
    ssq, sumlog, nu, _ = arima_like(y_adj, mod, arma)
    
    safe_nu = jnp.maximum(nu, 1.0)
    safe_ssq = jnp.maximum(ssq, 1e-8)
    nll = safe_nu * jnp.log(safe_ssq / safe_nu) + sumlog
    return 0.5 * nll

# =============================================================================
# OPTIMIZATION KERNEL (JAX Scipy BFGS)
# =============================================================================

@partial(jax.jit, static_argnames=['loss_fn', 'arma'])
def _fit_model_bfgs(
    init_params: Array,
    y: Array,
    xreg: Optional[Array],
    delta: Array,
    loss_fn, 
    arma: Tuple[int, ...],
    ncxreg: int, 
    n_exog: int, 
    include_mean: bool,
    maxiter: int
) -> Array:
    """
    JIT-compiled BFGS Optimization using jax.scipy.optimize.minimize.
    """
    # 1. Define the Objective Function
    # jax.scipy.optimize.minimize expects a function f(params, *args)
    # We bundle our specific arguments into the tuple format it expects later.
    def objective(p):
        return loss_fn(p, y, xreg, delta, arma, ncxreg, n_exog, include_mean)

    # 2. Run Optimization
    # method='BFGS' is robust for unconstrained optimization like ARIMA
    results = jax.scipy.optimize.minimize(
        fun=objective,
        x0=init_params,
        method='BFGS',
        options={'maxiter': maxiter}
    )
    
    # 3. Return Optimized Parameters
    return results.x


def arima_fit(
    x: Array,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Optional[Dict] = None,
    xreg: Optional[Array] = None,
    include_mean: bool = True,
    method: str = "CSS-ML",
    optim_control: Optional[Dict] = None,
) -> Dict:
    """
    Unified ARIMA Fitting Function using L-BFGS with Statistical Drift Init.
    """
    # 1. SETUP
    x = jnp.asarray(x, dtype=jnp.float64)
    if seasonal is None: seasonal = {"order": (0, 0, 0), "period": 1}
    
    p, d, q = order
    P, D, Q = seasonal["order"]
    period = seasonal.get("period", 1) or 1
    arma = (p, q, P, Q, period, d, D) 
    
    # Differencing Polynomial
    delta = jnp.array([1.0], dtype=jnp.float64)
    for _ in range(d): delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
    for _ in range(D):
        seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(period-1), jnp.array([-1.0])])
        delta = jnp.convolve(delta, seas_diff)
    delta = -delta[1:]

    # Counts
    n_exog = xreg.shape[1] if xreg is not None and xreg.ndim > 1 else (1 if xreg is not None else 0)
    ncxreg = n_exog + (1 if include_mean else 0)
    narma = p + q + P + Q
    
    # Flags for metadata (Required for the return dict)
    use_drift = include_mean and (d + D) == 1
    n_obs = x.shape[0]
    drift_coef = 0.0

    # 2. INITIALIZATION
    init_params = jnp.zeros(narma + ncxreg, dtype=jnp.float64)
    init_params = init_params + 1e-3  # Small offset to avoid zero-gradient stagnation
    
    # A. Stationary Mean (d=0, D=0)
    if include_mean and (d + D) == 0:
        init_params = init_params.at[narma + n_exog].set(jnp.nanmean(x))

    # B. Drift (d+D == 1) - WITH T-TEST
    # This logic decides if the "Trend" is real or just noise.
    if use_drift:
        # Calculate the differenced series (the "slope" at each step)
        if D == 1 and period > 1:
            dx = x[period:] - x[:-period]
        else:
            dx = x[1:] - x[:-1]
            
        # Calculate statistics
        mean_dx = jnp.nanmean(dx)
        std_dx = jnp.nanstd(dx) + 1e-10
        n_dx = dx.shape[0]
        
        # T-statistic: |mean| / standard_error
        # This tells us: "Is the mean significantly different from 0?"
        t_stat = jnp.abs(mean_dx / (std_dx / jnp.sqrt(n_dx)))
        
        # GATING LOGIC:
        # If t > 1.5, we assume there is a real trend -> Initialize with mean.
        # If t <= 1.5, we assume it's random noise -> Initialize with 0.0.
        # This prevents the model from forcing a trend on stationary data (Daily Births).
        drift_guess = jnp.where(t_stat > 1.5, mean_dx, 0.0)
        
        # Save the guess for metadata
        drift_coef = float(drift_guess)
        
        # Set the initialization
        init_params = init_params.at[narma + n_exog].set(drift_guess)

    # Optimization Settings
    if optim_control is None: optim_control = {}
    maxiter = optim_control.get("steps", 100)
    
    # 3. OPTIMIZATION LOOP
    method = method.upper()
    current_params = init_params
    
    # Helper to call the JIT kernel
    # Note: We use _fit_model_lbfgs because optax.lbfgs IS the BFGS implementation
    def run_bfgs(start_params, loss_func, steps):
        return _fit_model_bfgs(
            start_params, x, xreg, delta, loss_func,
            arma, ncxreg, n_exog, include_mean, steps
        )

    if "CSS" in method:
        # Split iterations for hybrid method
        css_iter = maxiter // 2 if method == "CSS-ML" else maxiter
        current_params = run_bfgs(current_params, _objective_css, css_iter)
        
    if "ML" in method: 
        ml_iter = maxiter // 2 if method == "CSS-ML" else maxiter
        current_params = run_bfgs(current_params, _objective_ml, ml_iter)

    # 4. POST-PROCESSING
    y_adj, phi, theta = _unpack_and_adjust(current_params, x, xreg, arma, ncxreg, n_exog, include_mean)
    mod = make_arima(phi, theta, delta, arma)
    ssq, sumlog, nu, resid = arima_like(y_adj, mod, arma)
    
    sigma2 = jnp.where(nu > 0, ssq / nu, jnp.inf)
    
    # Exact LogLikelihood
    loglik = -0.5 * (nu * jnp.log(sigma2) + sumlog + nu * jnp.log(2 * jnp.pi) + nu)
    
    n_params_total = narma + ncxreg + 1
    metrics = _compute_metrics(loglik, sigma2, nu, n_params_total)
    success = jnp.isfinite(loglik) & (sigma2 > 0)
    
    return {
        "coef": current_params,
        **metrics, 
        "model": mod,
        "residuals": resid,
        "arma": arma,
        "nobs": nu,
        "n_obs_train": n_obs,      # Required for wrapper
        "use_drift": use_drift,    # Required for wrapper
        "drift_coef": drift_coef,  # Metadata
        "success": success 
    }
# =============================================================================
# SHARED PREDICTION KERNEL (JIT)
# =============================================================================

@partial(jax.jit, static_argnames=['n_ahead', 'narma', 'n_exog'])
def _predict_core(
    mod: StateSpaceModel, 
    params: Array, 
    n_ahead: int, 
    newxreg: Array, 
    narma: int,
    n_exog: int,
    sigma2: float
) -> Tuple[Array, Array]:
    """
    Core math kernel. compiled once, used by both Single and Batch predictors.
    """
    # 1. Kalman Forecast (Stochastic Part)
    forecast_component, cov_component = kalman_forecast(n_ahead, mod)
    
    # 2. Exogenous/Mean (Deterministic Part)
    xm = jnp.zeros(n_ahead, dtype=jnp.float64)
    
    if n_exog > 0:
        # Extract coefs after the ARMA block
        # dynamic_slice ensures shape stability for JIT
        # shape: (n_exog,)
        exog_coefs = lax.dynamic_slice(params, (narma,), (n_exog,))
        
        # Matrix multiplication: X @ beta
        # newxreg shape: (n_ahead, n_exog)
        xm = jnp.dot(newxreg, exog_coefs)

    # 3. Combine & Scale
    final_pred = forecast_component + xm
    final_se = jnp.sqrt(cov_component * sigma2)
    
    return final_pred, final_se

# Batch Kernel (Vmap over batch dimension 0)
_predict_batch_kernel = jax.vmap(
    _predict_core, 
    in_axes=(
        0,    # mod: Batched StateSpaceModel (T has shape (B, r, r))
        0,    # params: Batched coefficients (B, n_params)
        None, # n_ahead: Shared horizon
        0,    # newxreg: Batched regressors (B, n_ahead, n_exog)
        None, # narma
        None, # n_exog
        0     # sigma2: Batched variance (B,)
    )
)

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_arima(
    model: Dict, 
    n_ahead: int, 
    newxreg: Optional[Array] = None,
    se_fit: bool = True
) -> Union[Array, Tuple[Array, Array]]:
    """
    Unified Prediction Function (Single + Batch).
    Automatically detects if input is a single model or a batch.
    
    Handles drift continuation: when model was fitted with drift (d+D==1 + mean),
    future time values are automatically generated for forecasting.
    """
    params = model["coef"]
    
    # Check if model uses drift
    use_drift = model.get("use_drift", False)
    drift_coef = model.get("drift_coef", 0.0)
    
    # CASE 1: SINGLE MODEL (1D Params)
    if params.ndim == 1:
        # Prepare Inputs
        mod = model["model"]
        narma = sum(model["arma"][:4])
        n_exog = params.shape[0] - narma
        
        # Prepare Exogenous (Single) - now drift is NOT in xreg
        if n_exog > 0:
            if newxreg is None:
                reg_matrix = jnp.ones((n_ahead, 1), dtype=jnp.float64)  # Intercept
            else:
                newxreg = jnp.asarray(newxreg, dtype=jnp.float64)
                if newxreg.ndim == 1: newxreg = newxreg.reshape(-1, 1)
                # Pad intercept if needed
                if newxreg.shape[1] < n_exog:
                    reg_matrix = jnp.hstack([newxreg, jnp.ones((n_ahead, 1))])
                else:
                    reg_matrix = newxreg
        else:
            reg_matrix = jnp.zeros((n_ahead, 0), dtype=jnp.float64)
            
        # Call Fast Single Kernel
        pred, se = _predict_core(mod, params, n_ahead, reg_matrix, narma, n_exog, model["sigma2"])
        
        # ADD DRIFT CUMULATIVELY: forecast[h] += drift_coef * (h+1) for h=0..n_ahead-1
        if use_drift and drift_coef != 0.0:
            drift_contribution = drift_coef * jnp.arange(1, n_ahead + 1, dtype=jnp.float64)
            pred = pred + drift_contribution
        
        return (pred, se) if se_fit else pred

    # CASE 2: BATCH MODEL (2D Params)
    else:
        # Unpack Batch
        mod = model["model"] # T is (Batch, r, r)
        sigma2 = model["sigma2"]
        narma = sum(model["arma"][:4])
        batch_size, n_total_params = params.shape
        n_exog = n_total_params - narma
        
        # Prepare Exogenous (Batch)
        if n_exog > 0:
            if newxreg is None:
                # Broadcast Intercept: (Batch, Horizon, 1)
                reg_matrix = jnp.ones((batch_size, n_ahead, 1), dtype=jnp.float64)
            else:
                xreg_arr = jnp.asarray(newxreg, dtype=jnp.float64)
                
                # SMART BROADCASTING
                # If user passed (Horizon, n_exog) but model is Batch -> Broadcast to (Batch, Horizon, n_exog)
                if xreg_arr.ndim == 2 and xreg_arr.shape[0] == n_ahead:
                    reg_matrix = jnp.broadcast_to(xreg_arr, (batch_size, n_ahead, xreg_arr.shape[1]))
                elif xreg_arr.ndim == 3:
                    reg_matrix = xreg_arr
                else:
                    raise ValueError(f"Invalid newxreg shape for batch: {xreg_arr.shape}")
        else:
            reg_matrix = jnp.zeros((batch_size, n_ahead, 0), dtype=jnp.float64)
            
        # Call Vmapped Kernel
        preds, se = _predict_batch_kernel(mod, params, n_ahead, reg_matrix, narma, n_exog, sigma2)
        return (preds, se) if se_fit else preds

# =============================================================================
# UNIT ROOT TESTS
# =============================================================================


def is_constant(x: Array, tol: float = 1e-10) -> Array:
    """
    Check if series is constant (JIT Compatible).
    
    Strategy:
    1. Avoids strict equality (==) which fails with floating point noise.
    2. Calculates range (Max - Min). If range is near 0, it is constant.
    3. Returns JAX Boolean (Tracer) to prevent 'ConcretizationTypeError'.
    """
    x = jnp.asarray(x)
    
    # Handle NaN safety: If all NaNs, range is NaN. 
    # If standard variance is 0, max == min.
    
    # We use a non-blocking peak-to-peak check
    is_flat = (jnp.max(x) - jnp.min(x)) < tol
    
    return is_flat


def kpss_test(x: jax.Array) -> float:
    """KPSS Test (JAX Optimized). Match statsmodels logic."""
    x = jnp.asarray(x)
    n = x.shape[0]
    
    # 1. Demean
    resids = x - jnp.mean(x)
    
    # 2. Lag Length
    # Use numpy to ensure static integer for JIT
    nlags = int(np.floor(3.0 * np.sqrt(n) / 13.0))
    
    # 3. Numerator (Eta)
    S = jnp.cumsum(resids)
    eta = jnp.sum(S**2) / (n**2)
    
    # 4. Denominator (HAC Variance)
    # We use jnp.correlate to compute autocovariances.
    # This avoids dynamic slicing inside loops, which breaks JIT.
    # mode='full' output shape: (2*n - 1,)
    # Index of lag 0: n - 1
    covs = jnp.correlate(resids, resids, mode='full') / n
    
    # Gamma_0
    gamma_0 = covs[n - 1]
    
    # Gamma_1 to Gamma_nlags
    # These are at indices [n, n+1, ..., n+nlags-1]
    # Since n and nlags are static integers, this slice works in JIT.
    gammas = covs[n : n + nlags]
    
    # Bartlett Weights
    lags = jnp.arange(1, nlags + 1, dtype=jnp.float64)
    weights = 1.0 - (lags / (nlags + 1.0))
    
    # HAC Variance = Gamma_0 + 2 * sum(weights * Gammas)
    hac_var = gamma_0 + 2.0 * jnp.sum(weights * gammas)
    
    # 5. Statistic & P-Value
    kpss_stat = eta / (hac_var + 1e-10)
    
    table_crit = jnp.array([0.347, 0.463, 0.574, 0.739])
    table_pval = jnp.array([0.100, 0.050, 0.025, 0.010])
    
    pval = jnp.interp(kpss_stat, table_crit, table_pval, left=0.10, right=0.01)
    
    return pval


def ndiffs(x: Array, alpha: float = 0.05, max_d: int = 2) -> int:
    """
    Determine number of differences needed for stationarity (JIT Optimized).
    
    Strategy: Unrolls the loop for d=0 and d=1 since max_d is small.
    Executes checks in parallel and selects the lowest d that satisfies condition.
    """
    x = jnp.asarray(x)
    
    # --- Check d=0 ---
    # Is raw data constant or stationary?
    is_const_0 = is_constant(x)
    pval_0 = kpss_test(x)
    # If stationary at d=0, we are done.
    is_stat_0 = is_const_0 | (pval_0 >= alpha)
    
    # --- Check d=1 ---
    # We compute this path unconditionally for JIT stability
    dx = x[1:] - x[:-1] # First difference
    
    is_const_1 = is_constant(dx)
    pval_1 = kpss_test(dx)
    # If stationary at d=1
    is_stat_1 = is_const_1 | (pval_1 >= alpha)
    
    # --- Decision Logic ---
    # Default to max_d (usually 2)
    d = max_d
    
    # If d=1 was good, set d=1
    d = jnp.where(is_stat_1, 1, d)
    
    # If d=0 was good, set d=0 (This overrides d=1 because 0 < 1)
    d = jnp.where(is_stat_0, 0, d)
    
    # Safety: If max_d was passed as 1, clamp the result
    d = jnp.minimum(d, max_d)
    
    return d.astype(int)


def nsdiffs(x: Array, period: int, max_D: int = 1, alpha: float = 0.64) -> int:
    """
    Determine number of seasonal differences (Optimized).
    
    This implementation matches statsforecast's approach:
    1. Use STL decomposition to extract seasonal component (preferred)
    2. Fall back to classical decomposition if STL fails
    3. Compute seasonal strength: 1 - Var(remainder) / Var(remainder + seasonal)
    4. If strength > alpha (default 0.64), seasonal differencing is needed
    
    Args:
        x: Input time series
        period: Seasonal period (e.g., 12 for monthly data)
        max_D: Maximum seasonal differences allowed
        alpha: Threshold for seasonal strength (default 0.64 matches statsforecast)
    
    Returns:
        D: Number of seasonal differences needed (0 to max_D)
    """
    x = jnp.asarray(x)
    n = x.shape[0]
    
    # Safety Check: If period is invalid or data too short, D=0
    if period <= 1 or n < 2 * period:
        return 0
    
    # Check if constant
    if float(jnp.max(x) - jnp.min(x)) < 1e-10:
        return 0
    
    # --- Helper: Calculate Seasonal Strength using STL ---
    def get_seasonal_strength_stl(series):
        """Use STL decomposition to compute seasonal strength."""
        try:
            # Import STL decomposition
            from stl import stl_decompose
            
            current_n = series.shape[0]
            
            # Set window parameters for STL (similar to statsmodels defaults)
            seasonal_window = max(7, 2 * period + 1) | 1  # Ensure odd
            trend_window = (2 * period + 1) | 1  # Ensure odd
            
            # Run STL decomposition: returns (seasonal, trend, remainder)
            seasonal_comp, trend_comp, remainder = stl_decompose(
                series,
                period=period,
                seasonal=seasonal_window,
                trend=trend_window,
                seasonal_deg=0,
                trend_deg=1,
                inner=1
            )
            
            # Compute seasonal strength (statsforecast formula)
            var_resid = float(jnp.var(remainder))
            var_seas_resid = float(jnp.var(remainder + seasonal_comp))
            
            # Strength = 1 - Var(Resid) / Var(Resid + Seasonal)
            strength = max(0.0, 1.0 - (var_resid / (var_seas_resid + 1e-10)))
            return strength
            
        except Exception:
            # Fallback to classical decomposition
            return get_seasonal_strength_classical(series)
    
    def get_seasonal_strength_classical(series):
        """Fallback: Classical decomposition using periodic means."""
        current_n = series.shape[0]
        
        # Linear detrend
        t = jnp.arange(current_n, dtype=jnp.float64)
        t_mean = jnp.mean(t)
        y_mean = jnp.mean(series)
        numerator = jnp.sum((t - t_mean) * (series - y_mean))
        denominator = jnp.sum((t - t_mean)**2)
        slope = numerator / (denominator + 1e-10)
        intercept = y_mean - slope * t_mean
        trend = slope * t + intercept
        series_detrended = series - trend
        
        # Classical decomposition (periodic mean)
        indices = jnp.arange(current_n) % period
        seas_sum = jax.ops.segment_sum(series_detrended, indices, num_segments=period)
        seas_count = jax.ops.segment_sum(jnp.ones_like(series_detrended), indices, num_segments=period)
        seas_means = seas_sum / (seas_count + 1e-10)
        seasonal_comp = seas_means[indices]
        remainder = series_detrended - seasonal_comp
        
        var_resid = float(jnp.var(remainder))
        var_total = float(jnp.var(remainder + seasonal_comp))
        strength = max(0.0, 1.0 - (var_resid / (var_total + 1e-10)))
        return strength
    
    # --- Main logic: iteratively check if seasonal differencing needed ---
    D = 0
    current_x = x
    
    while D < max_D:
        # Compute seasonal strength using classical decomposition
        # (Classical works better than STL for detecting seasonal patterns)
        strength = get_seasonal_strength_classical(current_x)
        
        # If strength <= alpha, no more differencing needed
        if strength <= alpha:
            break
        
        # Apply seasonal difference: x[t] - x[t - period]
        D += 1
        current_x = current_x[period:] - current_x[:-period]
        
        # Safety: check if enough data remains
        if current_x.shape[0] < 2 * period:
            break
        
        # Check if series became constant
        if float(jnp.max(current_x) - jnp.min(current_x)) < 1e-10:
            break
    
    return D

# =============================================================================
# MODEL SELECTION
# =============================================================================



class ARIMAResult(NamedTuple):
    loglik: float
    sigma2: float
    aic: float
    bic: float
    aicc: float
    ic: float
    success: bool

def myarima(
    x: jax.Array,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal_order: Tuple[int, int, int] = (0, 0, 0),
    period: int = 1,
    constant: bool = True,
    ic: str = "aic",
    method: str = "CSS-ML", # Default to Hybrid for better accuracy
    xreg: Optional[jax.Array] = None
) -> ARIMAResult:
    """
    Evaluates an ARIMA model using Exact Likelihood metrics.
    Robust, accurate, and JIT-optimized.
    """
    # 1. Fit the model using the Unified ARIMA Fitter
    # arima_fit handles the differencing (d, D) and parameter constraints.
    fit = arima_fit(
        x, 
        order=order, 
        seasonal={'order': seasonal_order, 'period': period}, 
        xreg=xreg,
        include_mean=constant, 
        method=method
    )

    # 2. Extract Exact Metrics
    # These were computed in _compute_metrics using the full Kalman Log-Likelihood
    aic_val = fit['aic']
    bic_val = fit['bic']
    aicc_val = fit['aicc']
    success = fit['success']

    # 3. Decision Logic (IC Selection)
    # Mapping string names to values using jnp.where for JIT compatibility
    chosen_ic = jnp.where(
        ic == "aic", aic_val,
        jnp.where(ic == "bic", bic_val,
        jnp.where(ic == "aicc", aicc_val, aic_val))
    )

    # 4. Global Failure Mask
    # If the optimizer failed or variance is non-positive, invalidate the model.
    # This prevents the search algorithm from selecting degenerate models.
    mask = success & jnp.isfinite(chosen_ic)
    
    return ARIMAResult(
        loglik=jnp.where(mask, fit['loglik'], -jnp.inf),
        sigma2=jnp.where(mask, fit['sigma2'], jnp.inf),
        aic=jnp.where(mask, aic_val, jnp.inf),
        bic=jnp.where(mask, bic_val, jnp.inf),
        aicc=jnp.where(mask, aicc_val, jnp.inf),
        ic=jnp.where(mask, chosen_ic, jnp.inf),
        success=mask
    )


def search_arima(
    x: Array,
    d: int = 0,
    D: int = 0,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    ic: str = "aicc",
    xreg: Optional[Array] = None,
    allow_drift: bool = True,
    allow_mean: bool = True,
    period: int = 1,
    method: str = "CSS"
) -> Dict:
    """
    Optimized Hybrid Search. 
    Python manages the grid, JAX manages the heavy math.
    """
    # Pre-asarray to avoid overhead in the loop
    x = jnp.asarray(x)
    
    allow_drift = allow_drift and (d + D) == 1
    allow_mean = allow_mean and (d + D) == 0
    max_K = 1 if (allow_drift or allow_mean) else 0
    
    actual_max_P = max_P if period > 1 else 0
    actual_max_Q = max_Q if period > 1 else 0

    best_ic = jnp.inf
    best_res = None

    # We use Python loops here. 
    # WHY? Because it allows 'p, q, P, Q' to be STATIC values.
    # This allows JAX to compile perfectly optimized kernels for each order.
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(actual_max_P + 1):
                for Q in range(actual_max_Q + 1):
                    if (p + q + P + Q) > max_order:
                        continue
                    
                    for K in range(max_K + 1):
                        # This call is JIT-compiled. 
                        # The first time an order is seen, it warms up.
                        # Subsequent calls are near-instant.
                        res = myarima(
                            x,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q),
                            period=period,
                            constant=(K == 1),
                            ic=ic,
                            method=method,
                            xreg=xreg
                        )
                        
                        # We use a simple float comparison
                        current_ic = float(res.ic)
                        if current_ic < best_ic:
                            best_ic = current_ic
                            best_res = {
                                "aic": float(res.aic),
                                "bic": float(res.bic),
                                "aicc": float(res.aicc),
                                "ic": current_ic,
                                "sigma2": float(res.sigma2),
                                "loglik": float(res.loglik),
                                "arma": (p, q, P, Q, period, d, D)
                            }

    if best_res is None:
        raise RuntimeError("No ARIMA model able to be estimated.")
        
    return best_res

def auto_arima_f(
    x: Array,
    d: Optional[int] = None,
    D: Optional[int] = None,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
    max_order: int = 5,
    max_d: int = 2,
    max_D: int = 1,
    start_p: int = 2,
    start_q: int = 2,
    start_P: int = 1,
    start_Q: int = 1,
    stationary: bool = False,
    seasonal: bool = True,
    ic: str = "aicc",
    stepwise: bool = True,
    nmodels: int = 94,
    method: str = "CSS",
    xreg: Optional[Array] = None,
    allowdrift: bool = True,
    allowmean: bool = True,
    period: int = 1,
) -> Dict:
    """
    Automatic ARIMA model selection (Optimized).
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    
    # --- Phase 1: Handle Edge Cases ---
    if is_constant(x):
        return arima_fit(x, order=(0, 0, 0), include_mean=allowmean, method=method)
    
    m = period if seasonal else 1
    
    # --- Phase 2: Determine Differencing ---
    if stationary:
        d_val = 0
        D_val = 0
    else:
        if m == 1:
            D_val = 0
            eff_max_P = 0
            eff_max_Q = 0
        else:
            D_val = D if D is not None else int(nsdiffs(x, period=m, max_D=max_D))
            eff_max_P = max_P
            eff_max_Q = max_Q
        
        if D_val > 0:
            xx = diff(x, m, D_val)
        else:
            xx = x
        
        d_val = d if d is not None else int(ndiffs(xx, max_d=max_d))
    
    # Effective limits
    if stationary or m == 1:
        eff_max_P = 0
        eff_max_Q = 0
    else:
        eff_max_P = max_P
        eff_max_Q = max_Q
    
    # --- Phase 3: Model Search ---
    
    # Helper to convert ARIMAResult to dict
    def _to_dict(res: ARIMAResult, p_: int, q_: int, P_: int, Q_: int) -> Dict:
        return {
            "aic": float(res.aic), "bic": float(res.bic), "aicc": float(res.aicc),
            "ic": float(res.ic), "sigma2": float(res.sigma2), "loglik": float(res.loglik),
            "arma": (p_, q_, P_, Q_, m, d_val, D_val), "success": bool(res.success)
        }
    
    # Full Grid Search
    if not stepwise:
        bestfit = search_arima(
            x, d=d_val, D=D_val, period=m,
            max_p=max_p, max_q=max_q, max_P=eff_max_P, max_Q=eff_max_Q,
            max_order=max_order, ic=ic, method=method, xreg=xreg,
            allow_drift=allowdrift, allow_mean=allowmean
        )
    # Stepwise Search
    else:
        p = min(start_p, max_p)
        q = min(start_q, max_q)
        P = min(start_P, eff_max_P) if m > 1 else 0
        Q = min(start_Q, eff_max_Q) if m > 1 else 0
        
        constant = (allowdrift and d_val + D_val == 1) or (allowmean and d_val + D_val == 0)
        
        res = myarima(
            x, order=(p, d_val, q), seasonal_order=(P, D_val, Q), period=m,
            constant=constant, ic=ic, method=method, xreg=xreg
        )
        bestfit = _to_dict(res, p, q, P, Q)
        
        # Try null model
        res = myarima(
            x, order=(0, d_val, 0), seasonal_order=(0, D_val, 0), period=m,
            constant=constant, ic=ic, method=method, xreg=xreg
        )
        fit = _to_dict(res, 0, 0, 0, 0)
        if fit["ic"] < bestfit["ic"]:
            bestfit = fit
            p = q = P = Q = 0
        
        k = 2
        improved = True
        while improved and k < nmodels:
            improved = False
            variations = [
                (p-1, q, P, Q), (p+1, q, P, Q),
                (p, q-1, P, Q), (p, q+1, P, Q),
                (p, q, P-1, Q), (p, q, P+1, Q),
                (p, q, P, Q-1), (p, q, P, Q+1),
            ]
            
            for new_p, new_q, new_P, new_Q in variations:
                if (0 <= new_p <= max_p and 0 <= new_q <= max_q and
                    0 <= new_P <= eff_max_P and 0 <= new_Q <= eff_max_Q and
                    new_p + new_q + new_P + new_Q <= max_order):
                    
                    res = myarima(
                        x, order=(new_p, d_val, new_q), seasonal_order=(new_P, D_val, new_Q), period=m,
                        constant=constant, ic=ic, method=method, xreg=xreg
                    )
                    fit = _to_dict(res, new_p, new_q, new_P, new_Q)
                    k += 1
                    
                    if fit["ic"] < bestfit["ic"]:
                        bestfit = fit
                        p, q, P, Q = new_p, new_q, new_P, new_Q
                        improved = True
                        break

    # --- Phase 4: Final Refit (THE FIX) ---
    # Re-run arima_fit on the best order to get coefficients and residuals
    
    final_p, final_q, final_P, final_Q, _, _, _ = bestfit["arma"]
    
    # Re-derive constant based on bestfit results or logic
    # (Simplified: assume allowdrift/allowmean logic holds for best model)
    use_constant = (allowdrift and d_val + D_val == 1) or (allowmean and d_val + D_val == 0)
    
    final_model = arima_fit(
        x, 
        order=(final_p, d_val, final_q),
        seasonal={'order': (final_P, D_val, final_Q), 'period': m},
        xreg=xreg,
        include_mean=use_constant,
        method=method
    )
    
    # Carry over the precise IC calculated during search if desired, 
    # though arima_fit recalculates it.
    
    return final_model





class AutoARIMA(BaseForecaster):
    """
    Automatic ARIMA model selection with Automatic Scaling.
    """
    uses_exog = True
    
    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        stepwise: bool = True,
        nmodels: int = 94,
        method: str = "CSS",
        allowdrift: bool = True,
        allowmean: bool = True,
        period: int = 1,
    ):
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.max_d = max_d
        self.max_D = max_D
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.stationary = stationary
        self.seasonal = seasonal
        self.ic = ic
        self.stepwise = stepwise
        self.nmodels = nmodels
        self.method = method
        self.allowdrift = allowdrift
        self.allowmean = allowmean
        self.period = period
        self.model_ = None
        
        # Scaling params
        self.mean_y = 0.0
        self.std_y = 1.0
    
    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "AutoARIMA":
        # 1. SCALING LOGIC
        y_np = np.array(y)
        self.mean_y = np.mean(y_np)
        self.std_y = np.std(y_np) + 1e-6 # Safety buffer
        
        # Normalize: (y - mean) / std
        y_scaled = (y_np - self.mean_y) / self.std_y
        y_jax = jnp.asarray(y_scaled, dtype=jnp.float64)
        
        if X is not None:
             X = jnp.asarray(X, dtype=jnp.float64)

        # --- CRITICAL CHANGE 1: SAVE HISTORY ---
        # We need the scaled history to reconstruct trends in predict()
        self.y_train_ = y_jax

        self.model_ = auto_arima_f(
            x=y_jax,
            d=self.d,
            D=self.D,
            max_p=self.max_p,
            max_q=self.max_q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_order=self.max_order,
            max_d=self.max_d,
            max_D=self.max_D,
            start_p=self.start_p,
            start_q=self.start_q,
            start_P=self.start_P,
            start_Q=self.start_Q,
            stationary=self.stationary,
            seasonal=self.seasonal,
            ic=self.ic,
            stepwise=self.stepwise,
            nmodels=self.nmodels,
            method=self.method,
            xreg=X,
            allowdrift=self.allowdrift,
            allowmean=self.allowmean,
            period=self.period,
        )
        return self
    
    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Dict[str, jnp.ndarray]:
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float64)

        # 1. Get Scaled Predictions (Differenced Scale)
        preds = predict_arima(self.model_, n_ahead=h, newxreg=X, se_fit=(level is not None))
        
        # Unpack
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # --- CRITICAL CHANGE 2: RECONSTRUCTION (Integration) ---
        p, q, P, Q, m, d, D = self.model_["arma"]
        
        # We reconstruct on CPU (numpy) because it's sequential and fast enough for h steps
        fc_reconstructed = mean_pred
        y_hist = self.y_train_ 

        # A. Undo Ordinary Differencing (Trend)
        # If d=1, the model predicts changes. We cumsum them + last value.
        for _ in range(d):
            last_val = y_hist[-1]
            fc_reconstructed = jnp.cumsum(fc_reconstructed) + last_val
            # Update history for next loop (if d=2)
            y_hist = jnp.concatenate([y_hist, fc_reconstructed])

        # B. Undo Seasonal Differencing (Seasonality)
        # If D=1, forecast[t] = forecast[t-m] + prediction[t]
        for _ in range(D):
            history_buffer = list(y_hist)
            new_forecast = []
            for k in range(h):
                prev = history_buffer[-m] # Value m steps ago
                val = prev + fc_reconstructed[k]
                new_forecast.append(val)
                history_buffer.append(val)
            fc_reconstructed = jnp.array(new_forecast)
            y_hist = jnp.concatenate([y_hist, fc_reconstructed])

        # C. Scale Standard Errors (Approximate)
        # If we integrated, uncertainty grows.
        if se_pred is not None:
            if d + D > 0:
                se_scaled = jnp.sqrt(jnp.cumsum(se_pred**2))
            else:
                se_scaled = se_pred
        
        # 3. INVERSE TRANSFORM (Unscale)
        result = {}
        result["mean"] = fc_reconstructed * self.std_y + self.mean_y
        
        if level is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                std_dev = se_scaled * self.std_y
                result[f"lo-{lv}"] = result["mean"] - q * std_dev
                result[f"hi-{lv}"] = result["mean"] + q * std_dev

        return result

    def summary(self) -> str:
        if self.model_ is None: return "Model not fitted"
        p, q, P, Q, m, d, D = self.model_["arma"]
        return f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}] | AICc: {self.model_.get('aicc', 0.0):.4f}"

class ARIMA(BaseForecaster):
    """
    Fixed ARIMA model wrapper with Automatic Scaling.
    """
    uses_exog = True
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        period: int = 1,
        include_mean: bool = True,
        method: str = "CSS",
        alias: str = "ARIMA",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.period = period
        self.include_mean = include_mean
        self.method = method
        self.alias = alias
        self.model_ = None
        
        # Scaling
        self.mean_y = 0.0
        self.std_y = 1.0

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "ARIMA":
        # 1. SCALING LOGIC
        y_np = np.array(y)
        self.mean_y = np.mean(y_np)
        self.std_y = np.std(y_np) + 1e-6
        
        y_scaled = (y_np - self.mean_y) / self.std_y
        y_jax = jnp.asarray(y_scaled, dtype=jnp.float64)
        
        if X is not None: X = jnp.asarray(X, dtype=jnp.float64)

        # --- SAVE HISTORY ---
        self.y_train_ = y_jax

        seasonal = {'order': self.seasonal_order, 'period': self.period}
        
        self.model_ = arima_fit(
            y_jax, order=self.order, seasonal=seasonal,
            include_mean=self.include_mean, method=self.method, xreg=X
        )
        
        # Inject arma tuple
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        self.model_['arma'] = (p, q, P, Q, self.period, d, D)
        
        return self

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level=None) -> Dict[str, jnp.ndarray]:
        if self.model_ is None: raise RuntimeError("Model not fitted.")
        if X is not None: X = jnp.asarray(X, dtype=jnp.float64)
        
        preds = predict_arima(self.model_, n_ahead=h, newxreg=X, se_fit=(level is not None))
        
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # --- RECONSTRUCTION (Copy of AutoARIMA logic) ---
        p, q, P, Q, m, d, D = self.model_["arma"]
        fc_reconstructed = mean_pred
        y_hist = self.y_train_ 

        for _ in range(d):
            last_val = y_hist[-1]
            fc_reconstructed = jnp.cumsum(fc_reconstructed) + last_val
            y_hist = jnp.concatenate([y_hist, fc_reconstructed])

        for _ in range(D):
            history_buffer = list(y_hist)
            new_forecast = []
            for k in range(h):
                prev = history_buffer[-m]
                val = prev + fc_reconstructed[k]
                new_forecast.append(val)
                history_buffer.append(val)
            fc_reconstructed = jnp.array(new_forecast)
            y_hist = jnp.concatenate([y_hist, fc_reconstructed])

        if se_pred is not None:
            if d + D > 0:
                se_scaled = jnp.sqrt(jnp.cumsum(se_pred**2))
            else:
                se_scaled = se_pred

        result = {}
        result["mean"] = fc_reconstructed * self.std_y + self.mean_y
            
        if level is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                std_dev = se_scaled * self.std_y
                result[f"lo-{lv}"] = result["mean"] - q * std_dev
                result[f"hi-{lv}"] = result["mean"] + q * std_dev
            
        return result