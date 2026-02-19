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
# AUTOMATIC PERIOD DETECTION
# =============================================================================

def detect_period(y: np.ndarray, max_period: Optional[int] = None) -> int:
    """
    Detect the dominant seasonal period from a time series using ACF peaks.
    Returns 1 if no significant seasonality is found.
    
    Uses a strict validation: the candidate period must have strong ACF,
    and harmonics (2*period, 3*period) should also show elevated ACF.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if max_period is None:
        max_period = min(n // 4, 200)
    if max_period < 2:
        return 1
    
    # Detrend via first differencing
    dy = np.diff(y)
    
    # Compute ACF via FFT
    dy_centered = dy - dy.mean()
    fft_len = 1
    while fft_len < 2 * len(dy_centered):
        fft_len *= 2
    fft_vals = np.fft.rfft(dy_centered, n=fft_len)
    acf_full = np.fft.irfft(fft_vals * np.conj(fft_vals), n=fft_len)
    acf_full = acf_full[:len(dy_centered)]
    if acf_full[0] > 0:
        acf_full = acf_full / acf_full[0]
    
    # Strict threshold: require correlation above noise floor
    threshold = max(0.1, 3.0 / np.sqrt(n))
    
    # Find candidate peaks
    candidates = []
    for lag in range(2, min(max_period + 1, len(acf_full) - 1)):
        val = acf_full[lag]
        if val > threshold:
            left = acf_full[lag - 1]
            right = acf_full[lag + 1]
            if val >= left and val >= right:
                candidates.append((lag, val))
    
    if not candidates:
        return 1
    
    # Score candidates: check harmonic consistency
    best_lag = 1
    best_score = 0.0
    
    for lag, acf_val in candidates:
        score = acf_val
        # Check harmonics (2x, 3x period)
        for mult in [2, 3]:
            harm_lag = lag * mult
            if harm_lag < len(acf_full):
                harm_val = acf_full[harm_lag]
                if harm_val > threshold * 0.5:
                    score += harm_val * 0.3
        
        if score > best_score:
            best_score = score
            best_lag = lag
    
    # Final check: reject if the peak is too weak
    if best_score < threshold * 1.5:
        return 1
    
    return best_lag


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

    # 3. Compute AR Part (Vectorized via Convolution)
    # The AR component is: resid[t] = w[t] - phi[0]*w[t-1] - phi[1]*w[t-2] - ...
    # This is equivalent to convolving w with the AR filter [1, -phi_0, -phi_1, ...]
    resid = w  # Start with the differenced series
    
    if p > 0:
        # Build AR filter polynomial: [1, -phi_0, -phi_1, ..., -phi_{p-1}]
        ar_filter = jnp.concatenate([jnp.array([1.0], dtype=jnp.float64), -phi])
        # Convolve and take only the first n elements (causal filter)
        resid = jnp.convolve(w, ar_filter)[:n]

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
    """Optimized Kalman Filter using jax.lax.scan. Returns (ssq, sumlog, nu, residuals)."""
    (a_final, P_final, ssq, sumlog, nu), residuals, _ = _kalman_filter_core(y, mod)
    return ssq, sumlog, nu, residuals


def _kalman_filter_core(y: Array, mod: StateSpaceModel):
    """
    Core Kalman filter. Returns full final carry, standardized residuals,
    and raw innovations. Used by arima_like (for metrics) and for
    innovation correction in forecasting.
    """
    T, Z, V, a0, P0 = mod.T, mod.Z, mod.V, mod.a0, mod.P0

    def step_prior_carry(carry, y_t):
        a_prior, P_prior, ssq, sumlog, nu = carry
        
        v = y_t - jnp.dot(Z, a_prior)
        F = jnp.dot(Z, P_prior @ Z)
        
        safe_F = jnp.where(F < 1e-9, 1e-9, F)
        M = P_prior @ Z
        K = M / safe_F
        
        a_post = a_prior + K * v
        P_post = P_prior - jnp.outer(K, M)
        
        a_next_prior = T @ a_post
        P_next_prior = T @ P_post @ T.T + V
        
        valid = F < 1e4
        nu_inc = jnp.where(valid, 1, 0)
        ssq_inc = jnp.where(valid, (v**2)/safe_F, 0.0)
        sumlog_inc = jnp.where(valid, jnp.log(safe_F), 0.0)
        std_resid = jnp.where(valid, v / jnp.sqrt(safe_F), 0.0)
        
        return (a_next_prior, P_next_prior, ssq + ssq_inc, sumlog + sumlog_inc, nu + nu_inc), (std_resid, v)

    init_carry = (mod.a0, mod.P0, 0.0, 0.0, 0)
    final_carry, (std_resids, innovations) = jax.lax.scan(step_prior_carry, init_carry, y)
    return final_carry, std_resids, innovations


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

    # B. Drift (d+D == 1)
    # Always initialize drift with the mean of the differenced series.
    # The optimizer will shrink it toward zero if the trend is not significant.
    if use_drift:
        if D == 1 and period > 1:
            dx = x[period:] - x[:-period]
        else:
            dx = x[1:] - x[:-1]
        
        mean_dx = jnp.nanmean(dx)
        init_params = init_params.at[narma + n_exog].set(mean_dx)

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
    (_, _, ssq, sumlog, nu), resid, innovations = _kalman_filter_core(y_adj, mod)
    
    sigma2 = jnp.where(nu > 0, ssq / nu, jnp.inf)
    
    # Exact LogLikelihood
    loglik = -0.5 * (nu * jnp.log(sigma2) + sumlog + nu * jnp.log(2 * jnp.pi) + nu)
    
    n_params_total = narma + ncxreg + 1
    metrics = _compute_metrics(loglik, sigma2, nu, n_params_total)
    success = jnp.isfinite(loglik) & (sigma2 > 0)
    
    # Extract the optimized drift from fitted params (not the initial guess)
    if use_drift:
        drift_coef = float(current_params[narma + n_exog])
    
    return {
        "coef": current_params,
        **metrics, 
        "model": mod,
        "residuals": resid,
        "innovations": innovations,
        "y_adj": y_adj,
        "delta": delta,
        "arma": arma,
        "nobs": nu,
        "n_obs_train": n_obs,
        "use_drift": use_drift,
        "drift_coef": drift_coef,
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
# SHARED RECONSTRUCTION: Undo differencing using the delta polynomial
# =============================================================================

def _reconstruct_forecast(
    raw_pred: Array,
    y_train: Array,
    arma: Tuple[int, ...],
    h: int,
) -> Array:
    """
    Undo all differencing (ordinary + seasonal) using the delta polynomial
    as an inverse filter. This correctly handles d>0, D>0, and combined cases.
    
    The delta polynomial encodes the combined differencing:
        delta = convolve([1,-1]^d, [1,0,...,-1]^D)
    stored as -delta[1:] in make_arima. We use the *positive* differencing
    coefficients for the inverse filter.
    
    Inverse filter: y[n+k] = raw[k] + sum(diffc[j] * y[n+k-1-j])
    where diffc are the positive differencing coefficients excluding the
    leading 1.
    """
    p, q, P, Q, m, d, D = arma
    
    if d + D == 0:
        return raw_pred
    
    # Rebuild the positive differencing polynomial coefficients
    # diffc = coefficients of (1-B)^d * (1-B^m)^D, excluding the leading 1
    poly = np.array([1.0])
    for _ in range(d):
        poly = np.convolve(poly, [1.0, -1.0])
    for _ in range(D):
        seas = np.zeros(m + 1)
        seas[0] = 1.0
        seas[m] = -1.0
        poly = np.convolve(poly, seas)
    
    # diffc = -poly[1:] (positive form for inverse filter)
    diffc = -poly[1:]
    nd = len(diffc)
    
    # Build extended series: last nd values of training + h forecast slots
    tail = np.array(y_train[-nd:], dtype=np.float64) if nd <= len(y_train) else np.array(y_train, dtype=np.float64)
    raw_np = np.array(raw_pred, dtype=np.float64)
    
    # Extend with forecast values
    extended = np.concatenate([tail, np.zeros(h)])
    offset = len(tail)
    
    for k in range(h):
        val = raw_np[k]
        for j in range(min(nd, offset + k)):
            val += diffc[j] * extended[offset + k - 1 - j]
        extended[offset + k] = val
    
    return jnp.array(extended[offset:])


# =============================================================================
# FUSED FORECAST KERNEL (Single XLA Dispatch: BFGS params -> forecasts)
# =============================================================================

@partial(jax.jit, static_argnames=['arma', 'ncxreg', 'n_exog', 'include_mean', 'n_ahead'])
def _fused_forecast_kernel(
    params: Array,
    y: Array,
    delta: Array,
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
    n_ahead: int,
) -> Array:
    """
    Fused kernel: params -> phi/theta -> state-space -> forecast.
    Skips all metric computation (AIC, BIC, sigma2, loglik, residuals).
    Single XLA dispatch eliminates multiple Python-to-XLA roundtrips.
    """
    # 1. Unpack params to phi, theta
    y_adj, phi, theta = _unpack_and_adjust(params, y, None, arma, ncxreg, n_exog, include_mean)

    # 2. Build state-space model (fresh initial state a0, P0)
    mod = make_arima(phi, theta, delta, arma)

    # 3. Structural Kalman forecast from initial state
    forecasts, _ = kalman_forecast(n_ahead, mod)

    # 4. Add deterministic component (intercept/drift)
    narma_total = sum(arma[:4])
    if ncxreg > 0:
        reg_matrix = jnp.ones((n_ahead, ncxreg), dtype=jnp.float64)
        exog_coefs = lax.dynamic_slice(params, (narma_total,), (ncxreg,))
        xm = jnp.dot(reg_matrix, exog_coefs)
        forecasts = forecasts + xm

    return forecasts


@partial(jax.jit, static_argnames=['arma', 'ncxreg', 'n_exog', 'include_mean', 'n_ahead'])
def _forecast_from_params(
    params: Array,
    y: Array,
    delta: Array,
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
    n_ahead: int,
) -> Array:
    """
    Fully fused forecast kernel: params → forecast in a single XLA dispatch.
    
    Combines all post-BFGS steps:
    1. Unpack params → phi, theta (once)
    2. Build state-space model (once)  
    3. Run Kalman filter on training data → innovations
    4. Kalman forecast from trained state
    5. MA innovation correction
    6. Deterministic component (intercept/drift)
    
    This replaces 5+ separate function calls that previously required
    multiple XLA dispatches and redundant computation.
    """
    narma_total = sum(arma[:4])
    
    # 1. Unpack params to phi, theta, adjusted y (ONCE)
    y_adj, phi, theta = _unpack_and_adjust(params, y, None, arma, ncxreg, n_exog, include_mean)
    
    # 2. Build state-space model (ONCE)
    mod = make_arima(phi, theta, delta, arma)
    
    # 3. Structural Kalman forecast from initial state (fresh a0, P0)
    # This matches the original approach: forecast from a0=zeros, not trained state
    forecasts, _ = kalman_forecast(n_ahead, mod)
    
    # 4. Run Kalman filter on training data ONLY to extract innovations
    # for MA correction (the filter state itself is not used for forecasting)
    _, _, innovations = _kalman_filter_core(y_adj, mod)
    
    # 5. MA innovation correction (inline vectorized)
    q_total = theta.shape[0]
    n_train = innovations.shape[0]
    # Apply MA correction if model has MA terms (q_total is static, known at trace time)
    if q_total > 0:
        # q_total (from arma) is always << n_train for valid models
        rev_tail = jax.lax.dynamic_slice(innovations, (n_train - q_total,), (q_total,))[::-1]
        corr_vals = jnp.zeros(n_ahead, dtype=jnp.float64)
        for k in range(min(n_ahead, q_total)):
            corr_vals = corr_vals.at[k].set(
                jnp.dot(theta[k:], rev_tail[:q_total - k])
            )
        forecasts = forecasts + corr_vals
    
    # 6. Deterministic component (intercept/drift)
    if ncxreg > 0:
        reg_matrix = jnp.ones((n_ahead, ncxreg), dtype=jnp.float64)
        exog_coefs = lax.dynamic_slice(params, (narma_total,), (ncxreg,))
        xm = jnp.dot(reg_matrix, exog_coefs)
        forecasts = forecasts + xm
    
    return forecasts


def _ma_innovation_correction(
    raw_pred: Array,
    theta_expanded: Array,
    innovations: Array,
    n_ahead: int,
) -> Array:
    """
    Add MA innovation correction to structural Kalman forecast.
    The Kalman forecast from a0=zeros misses the decaying MA contribution
    from recent innovations. For forecast step k, past innovations weighted
    by the expanded MA polynomial are added.
    
    Vectorized: for step k, corr[k] = sum_{j=k}^{q-1} theta[j] * innov[n-1-(j-k)]
    which equals dot(theta[k:], innov[n-1:n-1-(q-k):-1]) = dot(theta[k:], reversed tail)
    """
    q_total = theta_expanded.shape[0]
    if q_total == 0:
        return raw_pred
    
    n_train = innovations.shape[0]
    # Reverse the last q_total innovations: [innov[n-1], innov[n-2], ..., innov[n-q]]
    tail_len = min(q_total, n_train)
    rev_tail = innovations[n_train - tail_len:][::-1]  # shape (tail_len,)
    # Pad if needed (when n_train < q_total)
    if tail_len < q_total:
        rev_tail = jnp.concatenate([rev_tail, jnp.zeros(q_total - tail_len, dtype=jnp.float64)])
    
    # For step k: corr[k] = dot(theta[k:], rev_tail[:q_total-k])
    # This is a correlation/convolution operation
    corr = jnp.correlate(rev_tail, theta_expanded, mode='full')
    # The correlation output at index q_total-1+k gives sum_{j} theta[j] * rev_tail[j+k]
    # We want sum_{j=k}^{q-1} theta[j] * rev_tail[j-k] = correlate result at offset 0..n_ahead-1
    # Actually: corr[k] = sum_j theta[j] * rev_tail[j+k] isn't right. Let me do it directly.
    # For k=0: dot(theta[0:], rev_tail[0:]) = dot(theta, rev_tail)
    # For k=1: dot(theta[1:], rev_tail[0:q-1]) 
    # This is a simple loop but vectorized per-step with slicing
    steps = min(n_ahead, q_total)
    corr_vals = jnp.zeros(n_ahead, dtype=jnp.float64)
    for k in range(steps):
        corr_vals = corr_vals.at[k].set(jnp.dot(theta_expanded[k:], rev_tail[:q_total - k]))
    return raw_pred + corr_vals


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
        mod = model["model"]
        arma = model["arma"]
        narma = sum(arma[:4])
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
            
        # Call Fast Single Kernel (structural forecast from a0=zeros)
        pred, se = _predict_core(mod, params, n_ahead, reg_matrix, narma, n_exog, model["sigma2"])
        
        # MA innovation correction from training data
        innovations = model.get("innovations", None)
        if innovations is not None:
            _, theta_expanded = arima_transpar(params[:narma], arma, trans=True)
            pred = _ma_innovation_correction(pred, theta_expanded, innovations, n_ahead)
        
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


@jax.jit
def kpss_test(x: jax.Array) -> float:
    """KPSS Test (JAX Optimized, JIT-compiled). Match statsmodels logic."""
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


@partial(jax.jit, static_argnames=['max_d'])
def ndiffs(x: Array, alpha: float = 0.05, max_d: int = 2) -> int:
    """
    Determine number of differences needed for stationarity (JIT-compiled).
    
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
    method: str = "CSS-ML"
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

def _aa_standardize(y: jnp.ndarray, eps: float = 1e-10) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standardize series to zero mean and unit variance.
    Returns (y_norm, mean, std_safe). Used internally by AutoARIMA.
    """
    mean = jnp.nanmean(y)
    std = jnp.nanstd(y)
    std_safe = jnp.where(std > eps, std, 1.0)
    return (y - mean) / std_safe, mean, std_safe


def _aa_denormalize(y_norm: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Inverse of _aa_standardize: y = y_norm * std + mean."""
    return y_norm * std + mean


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
    method: str = "CSS-ML",
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
    # Use CSS-only for fast candidate evaluation; final refit uses full method
    search_method = "CSS"
    
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
            max_order=max_order, ic=ic, method=search_method, xreg=xreg,
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
            constant=constant, ic=ic, method=search_method, xreg=xreg
        )
        bestfit = _to_dict(res, p, q, P, Q)
        
        # Try null model
        res = myarima(
            x, order=(0, d_val, 0), seasonal_order=(0, D_val, 0), period=m,
            constant=constant, ic=ic, method=search_method, xreg=xreg
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
                        constant=constant, ic=ic, method=search_method, xreg=xreg
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
    Automatic ARIMA model selection. Internally standardizes the series
    (zero mean, unit variance) by default for faster and more stable optimization.
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
        method: str = "CSS-ML",
        allowdrift: bool = True,
        allowmean: bool = True,
        period: Optional[int] = None,
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
        self._user_period = period  # None = auto-detect
        self.model_ = None
        self.standardize = True
        self._y_mean = None
        self._y_std = None
        # Cached state for fast forecast() path
        self._cached_order = None
        self._cached_seasonal_order = None
        self._cached_include_mean = None
        self._cached_delta = None
        self._cached_arma = None
        self._cached_narma = None
        self._cached_ncxreg = None
        self._cached_n_exog = None
    
    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "AutoARIMA":
        y_np = np.array(y)
        y_jax = jnp.asarray(y_np, dtype=jnp.float64)
        
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float64)

        if self.standardize:
            y_fit, self._y_mean, self._y_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            self._y_mean = jnp.array(0.0, dtype=jnp.float64)
            self._y_std = jnp.array(1.0, dtype=jnp.float64)

        self.y_train_ = y_fit

        # Auto-detect period if not specified
        if self._user_period is None:
            detected = detect_period(y_np, max_period=min(len(y_np) // 4, 24))
            self.period = detected
        else:
            self.period = self._user_period

        self.model_ = auto_arima_f(
            x=y_fit,
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
        
        # Cache the selected order for fast forecast() reuse
        p, q, P, Q, m, d, D = self.model_["arma"]
        self._cached_order = (p, d, q)
        self._cached_seasonal_order = (P, D, Q)
        self._cached_include_mean = self.model_.get("use_drift", False) or (d + D == 0 and self.allowmean)
        self._cached_arma = (p, q, P, Q, m, d, D)
        self._cached_narma = p + q + P + Q
        n_exog = 0
        self._cached_ncxreg = n_exog + (1 if self._cached_include_mean else 0)
        self._cached_n_exog = n_exog
        
        # Cache delta polynomial
        delta = jnp.array([1.0], dtype=jnp.float64)
        for _ in range(d):
            delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
        for _ in range(D):
            seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(m - 1), jnp.array([-1.0])])
            delta = jnp.convolve(delta, seas_diff)
        self._cached_delta = -delta[1:]
        
        return self
    
    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """
        Fast fit-and-predict. On first call, runs full model search. On subsequent
        calls, reuses the cached best order and runs only BFGS + fused kernel.
        """
        y_jax = jnp.asarray(np.array(y), dtype=jnp.float64)
        
        # First call: run full search to find best order, then reuse predict()
        if self._cached_order is None:
            self.fit(y_jax, X)
            return self.predict(h, X=None)

        # Subsequent calls: fast path with cached order (CSS-only for speed)
        arma = self._cached_arma
        p, q, P, Q, m, d, D = arma
        include_mean = self._cached_include_mean
        use_drift = include_mean and (d + D) == 1
        
        # Standardize current series for fast refit
        if self.standardize:
            y_fit, f_mean, f_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            f_mean = jnp.array(0.0, dtype=jnp.float64)
            f_std = jnp.array(1.0, dtype=jnp.float64)

        init_params = jnp.zeros(self._cached_narma + self._cached_ncxreg, dtype=jnp.float64) + 1e-3

        if include_mean and (d + D) == 0:
            init_params = init_params.at[self._cached_narma + self._cached_n_exog].set(jnp.nanmean(y_fit))
        
        if use_drift:
            if D == 1 and m > 1:
                dx = y_fit[m:] - y_fit[:-m]
            else:
                dx = y_fit[1:] - y_fit[:-1]
            init_params = init_params.at[self._cached_narma + self._cached_n_exog].set(jnp.nanmean(dx))
        
        current_params = _fit_model_bfgs(
            init_params, y_fit, None, self._cached_delta, _objective_css,
            self._cached_arma, self._cached_ncxreg, self._cached_n_exog, include_mean, 50
        )
        
        # Fused forecast: single XLA dispatch for params → forecast
        raw_fc = _forecast_from_params(
            current_params, y_fit, self._cached_delta,
            self._cached_arma, self._cached_ncxreg, self._cached_n_exog, include_mean, h
        )
        
        fc_norm = _reconstruct_forecast(raw_fc, y_fit, self._cached_arma, h)
        fc = _aa_denormalize(fc_norm, f_mean, f_std)
        return {"mean": fc}
    
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

        preds = predict_arima(self.model_, n_ahead=h, newxreg=X, se_fit=(level is not None))
        
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        fc_norm = _reconstruct_forecast(mean_pred, self.y_train_, self.model_["arma"], h)
        mean_orig = _aa_denormalize(fc_norm, self._y_mean, self._y_std)

        # Standard Errors
        if se_pred is not None:
            p, q, P, Q, m, d, D = self.model_["arma"]
            if d + D > 0:
                se_scaled = jnp.sqrt(jnp.cumsum(se_pred**2))
            else:
                se_scaled = se_pred
            se_orig = se_scaled * self._y_std
        
        result = {}
        result["mean"] = mean_orig
        
        if level is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                result[f"lo-{lv}"] = result["mean"] - q * se_orig
                result[f"hi-{lv}"] = result["mean"] + q * se_orig

        return result

    def summary(self) -> str:
        if self.model_ is None: return "Model not fitted"
        p, q, P, Q, m, d, D = self.model_["arma"]
        return f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}] | AICc: {self.model_.get('aicc', 0.0):.4f}"

class ARIMA(BaseForecaster):
    """
    Fixed ARIMA model wrapper.

    Internally standardizes the series (zero mean, unit variance)
    for faster and more stable optimization, and always returns
    forecasts in the original scale.
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
        standardize: bool = True,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.period = period
        self.include_mean = include_mean
        self.method = method
        self.alias = alias
        self.model_ = None
        self.standardize = standardize
        self._y_mean = None
        self._y_std = None

        # Pre-compute and cache the delta polynomial (depends only on order/period)
        p, d, q = order
        P, D, Q = seasonal_order
        delta = jnp.array([1.0], dtype=jnp.float64)
        for _ in range(d):
            delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
        for _ in range(D):
            seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(period - 1), jnp.array([-1.0])])
            delta = jnp.convolve(delta, seas_diff)
        self._delta = -delta[1:]
        self._arma = (p, q, P, Q, period, d, D)
        self._narma = p + q + P + Q
        n_exog = 0
        self._ncxreg = n_exog + (1 if include_mean else 0)
        self._n_exog = n_exog

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "ARIMA":
        y_jax = jnp.asarray(np.array(y), dtype=jnp.float64)
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float64)

        # Standardize training series if enabled
        if self.standardize:
            y_fit, self._y_mean, self._y_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            self._y_mean = jnp.array(0.0, dtype=jnp.float64)
            self._y_std = jnp.array(1.0, dtype=jnp.float64)

        self.y_train_ = y_fit

        seasonal = {'order': self.seasonal_order, 'period': self.period}
        
        self.model_ = arima_fit(
            y_fit, order=self.order, seasonal=seasonal,
            include_mean=self.include_mean, method=self.method, xreg=X
        )
        
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        self.model_['arma'] = (p, q, P, Q, self.period, d, D)
        
        return self

    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """
        Fast fit-and-predict in one shot. Uses fused JIT kernel to minimize
        Python-to-XLA dispatch overhead. No metric computation (AIC, BIC, etc.).
        
        Args:
            h: Forecast horizon
            y: Training series
            X: Exogenous regressors (not supported in fast path)
            
        Returns:
            Dict with 'mean' key containing forecast array
        """
        y_jax = jnp.asarray(np.array(y), dtype=jnp.float64)

        # Standardize input series for fast one-shot fit if enabled
        if self.standardize:
            y_fit, f_mean, f_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            f_mean = jnp.array(0.0, dtype=jnp.float64)
            f_std = jnp.array(1.0, dtype=jnp.float64)
        p, d, q = self.order
        P, D, Q = self.seasonal_order
        
        # --- BFGS optimization (same as arima_fit, but uses cached delta) ---
        use_drift = self.include_mean and (d + D) == 1
        init_params = jnp.zeros(self._narma + self._ncxreg, dtype=jnp.float64) + 1e-3
        
        if self.include_mean and (d + D) == 0:
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(y_fit))
        
        if use_drift:
            if D == 1 and self.period > 1:
                dx = y_fit[self.period:] - y_fit[:-self.period]
            else:
                dx = y_fit[1:] - y_fit[:-1]
            init_params = init_params.at[self._narma + self._n_exog].set(jnp.nanmean(dx))

        method = self.method.upper()
        current_params = init_params
        maxiter = 50  # Reduced: CSS converges fast for low-order models
        
        if "CSS" in method:
            css_iter = maxiter // 2 if method == "CSS-ML" else maxiter
            current_params = _fit_model_bfgs(
                current_params, y_fit, None, self._delta, _objective_css,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, css_iter
            )
        if "ML" in method:
            ml_iter = maxiter // 2 if method == "CSS-ML" else maxiter
            current_params = _fit_model_bfgs(
                current_params, y_fit, None, self._delta, _objective_ml,
                self._arma, self._ncxreg, self._n_exog, self.include_mean, ml_iter
            )

        # --- Fused forecast: single XLA dispatch for params → forecast ---
        raw_fc = _forecast_from_params(
            current_params, y_fit, self._delta,
            self._arma, self._ncxreg, self._n_exog, self.include_mean, h
        )

        # --- Reconstruction using delta polynomial inverse filter ---
        fc_norm = _reconstruct_forecast(raw_fc, y_fit, self._arma, h)
        fc = _aa_denormalize(fc_norm, f_mean, f_std)

        return {"mean": fc}

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level=None) -> Dict[str, jnp.ndarray]:
        if self.model_ is None: raise RuntimeError("Model not fitted.")
        if X is not None: X = jnp.asarray(X, dtype=jnp.float64)
        
        preds = predict_arima(self.model_, n_ahead=h, newxreg=X, se_fit=(level is not None))
        
        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # Reconstruct in standardized space
        fc_norm = _reconstruct_forecast(mean_pred, self.y_train_, self.model_["arma"], h)
        # Map back to original scale if standardization was used
        if self.standardize:
            mean_orig = _aa_denormalize(fc_norm, self._y_mean, self._y_std)
        else:
            mean_orig = fc_norm

        if se_pred is not None:
            p, q, P, Q, m, d, D = self.model_["arma"]
            if d + D > 0:
                se_scaled = jnp.sqrt(jnp.cumsum(se_pred**2))
            else:
                se_scaled = se_pred
            se_orig = se_scaled * (self._y_std if self.standardize else 1.0)
        else:
            se_orig = None

        result = {}
        result["mean"] = mean_orig
            
        if level is not None and se_orig is not None:
            if isinstance(level, int): level = (level,)
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                q = z_scores[i]
                result[f"lo-{lv}"] = result["mean"] - q * se_orig
                result[f"hi-{lv}"] = result["mean"] + q * se_orig
            
        return result