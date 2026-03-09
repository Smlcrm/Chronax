"""
File: auto_arima.py

High-level Purpose:
    Implements ARIMA and AutoARIMA model estimation, model-order search,
    state-space construction, Kalman likelihood evaluation, and forecasting
    kernels optimized for JAX execution.

Problem Solved:
    Provides an end-to-end automatic ARIMA workflow that can infer differencing,
    explore candidate orders, fit selected models, and generate forecasts with
    optional uncertainty while maintaining high numerical throughput.

Architectural Role:
    Serves as the statistical core for ARIMA-family models in the forecasting
    stack, powering higher-level wrappers and benchmark pathways with reusable
    low-level transforms, objective functions, and prediction kernels.

Major Classes/Functions:
    - `AutoARIMA`: High-level automatic order-selection estimator.
    - `ARIMA`: Fixed-order estimator backed by shared optimization kernels.
    - `arima_fit`, `predict_arima`, `auto_arima_f`: Core fit/predict/search APIs.
    - Supporting transforms and Kalman/filtering helpers for ARIMA internals.

External Dependencies:
    - `numpy`
    - `jax`, `jax.numpy`, `jax.scipy.optimize`
    - `optax`
    - Internal modules: `stl`, `base_forecaster`, `utils`

Expected Inputs and Outputs:
    - Input: numeric time-series arrays, model-order constraints, seasonal
      metadata, and optional exogenous regressors.
    - Output: typed result structures and model dictionaries containing
      coefficients, information criteria, residual diagnostics, and forecasts.

Example:
    >>> import jax.numpy as jnp
    >>> from auto_arima import AutoARIMA
    >>> model = AutoARIMA(seasonal=False, stepwise=True)
    >>> model.fit(jnp.array([10.0, 12.0, 11.0, 13.0, 14.0]))
    >>> model.predict(h=2)["mean"].shape
    (2,)

Assumptions:
    - Input arrays are finite numeric sequences suitable for float64 casting.
    - JAX execution environment is available for compiled kernels.
    - Seasonal period and order bounds are consistent with available history.

Side Effects:
    - Compiles JAX kernels on first invocation.
    - Updates model instance state during fit/search operations.

Author:
    Auto-documented
Date:
    2026-02-21
"""

from __future__ import annotations
import math
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
# import jaxopt
import jax.scipy.optimize
from ..stl import stl_decompose
Array = jnp.ndarray

from chronax.models.base_forecaster import BaseForecaster

from chronax.utils import _quantiles


# =============================================================================
# AUTOMATIC PERIOD DETECTION
# =============================================================================

def detect_period(y: np.ndarray, max_period: Optional[int] = None) -> int:
    """
    Detect the dominant seasonal period from a time series using ACF peaks.

    Detailed Description:
        Detrends the series via first differencing, computes the
        autocorrelation function (ACF) using FFT, and identifies lag peaks
        above a noise-floor threshold. Candidates are scored by requiring
        elevated ACF at harmonics (2*period, 3*period) to reduce spurious
        picks. Returns 1 if no significant seasonality is found or if
        max_period is too small. Used by AutoARIMA when period is None to
        set the seasonal cycle before order search.

    Args:
        y (np.ndarray): Univariate time series; will be cast to float64.
        max_period (Optional[int]): Maximum period to consider. If None,
            set to min(n // 4, 200). If < 2, returns 1.

    Returns:
        int: Detected seasonal period (>= 1). 1 means no seasonality detected.

    Raises:
        None. Short or invalid inputs yield return value 1.

    Side Effects:
        None. Pure function; does not modify y.

    Example:
        >>> detect_period(np.array([1., 2., 1., 2., 1., 2.]), max_period=10)
        2

    Notes:
        Role: Drives automatic seasonal specification in AutoARIMA and
        ensures the search space (P, Q, period) is data-appropriate.
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
    Apply lag-differencing one or more times to produce a stationary-like series.

    Detailed Description:
        For each of `differences` steps, replaces x with x[lag:] - x[:-lag],
        so the output length is len(x) - lag * differences. Used for ordinary
        differencing (lag=1, differences=d) and seasonal differencing
        (lag=period, differences=D) in arima_css, auto_arima_f (seasonal
        differencing for nsdiffs output), and elsewhere. JIT-compiled with
        static lag and differences so the loop is unrolled. No padding; result
        is strictly shorter than input when differences >= 1.

    Args:
        x (Array): Input series; converted to float64.
        lag (int): Lag for each difference step (e.g. 1 for (1-B), m for (1-B^m)); must be static.
        differences (int): Number of times to apply the lag difference; must be static.

    Returns:
        Array: Differenced series; length len(x) - lag * differences when differences >= 1.

    Raises:
        None. differences < 1 returns x unchanged.

    Side Effects:
        None. Pure function.

    Example:
        >>> dx = diff(y, 1, 1); d12y = diff(y, 12, 1)

    Notes:
        Role: Shared differencing primitive for CSS, seasonal detection, and
        model fitting; static args keep JIT efficient.
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
    Compute the initial state covariance P for the ARIMA state-space model.

    Detailed Description:
        Solves the Lyapunov equation P = F P F' + V where F is the companion
        transition matrix and V = G G' is the process noise covariance. Uses
        the doubling algorithm (iterating P := P + F P F', F := F^2) which
        converges in a small number of steps for stable F. The result is the
        unconditional covariance of the state at t=0, used as P0 in make_arima
        and the Kalman filter. Standard state-space representation
        (Jones/Pearlman style). JIT-compiled with static arma.

    Args:
        phi (Array): AR coefficients (expanded).
        theta (Array): MA coefficients (expanded).
        arma (Tuple[int, ...]): (p, q, P, Q, m, d, D) for static_argnames.

    Returns:
        Array: Initial state covariance matrix P, shape (r, r) with r = max(p, q+1).

    Raises:
        None. Unstable F can yield large P.

    Side Effects:
        None. Pure function.

    Example:
        >>> P0 = getQ0(phi, theta, arma)

    Notes:
        Role: Provides P0 for make_arima and Kalman filter; doubling avoids
        dense linear system solves.
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
    def body(_, val: Tuple[Array, Array]) -> Tuple[Array, Array]:
        """
        One step of the doubling algorithm for the Lyapunov equation P = F P F' + V.

        Detailed Description:
            Updates (P, F) to (P + F P F', F^2). Repeated application
            converges to the unique solution P for stable F. Used inside
            getQ0 to compute the initial state covariance for the ARIMA
            state-space model without solving a dense linear system.

        Args:
            _: Unused (fori_loop step index).
            val (Tuple[Array, Array]): (P, F) current covariance and transition matrix.

        Returns:
            Tuple[Array, Array]: (P_new, F_new), the new carry for fori_loop.

        Notes:
            Role: Inner kernel of getQ0; must be JAX-pure for JIT.
        """
        P, F_mat = val
        # P_new = P + F * P * F.T
        P_new = P + F_mat @ P @ F_mat.T
        # F_new = F * F
        F_new = F_mat @ F_mat
        return (P_new, F_new)

    # Initial carry (V, F); fori_loop returns final carry (P_final, F_final)
    P_final, _ = jax.lax.fori_loop(0, 16, body, (V, F))
    
    return P_final

# =============================================================================
# CSS ESTIMATION
# =============================================================================

@partial(jax.jit, static_argnames=['arma'])
def arima_css(y: Array, arma: Tuple[int, ...], phi: Array, theta: Array) -> Tuple[float, Array]:
    """
    Compute conditional sum-of-squares residual variance and residuals for ARIMA.

    Detailed Description:
        Applies ordinary and seasonal differencing to y, then computes
        residuals from the AR part (vectorized convolution) and the MA part
        (sequential recursion via jax.lax.scan). The first ncond observations
        are treated as conditioning; sigma2 is the sum of squared residuals
        over the valid range divided by (n - ncond). Used by _objective_css
        for fast CSS objective evaluation and by arima_fit when method is CSS
        or the first phase of CSS-ML. Returns inf variance when residuals are
        non-finite or unstable.

    Args:
        y (Array): Time series (possibly after mean/exog adjustment).
        arma (Tuple[int, ...]): (p, q, P, Q, m, d, D); must be tuple for JIT.
        phi (Array): Expanded AR coefficients.
        theta (Array): Expanded MA coefficients.

    Returns:
        Tuple[float, Array]: (sigma2, residuals). sigma2 is scalar variance;
            residuals is the full-length residual array.

    Raises:
        None. Unstable models yield sigma2=inf.

    Side Effects:
        None. Pure JAX; JIT-compiled.

    Example:
        >>> sigma2, resid = arima_css(y_adj, arma, phi, theta)

    Notes:
        Role: Core CSS estimation kernel; fast alternative to Kalman-based
        likelihood for initialization or CSS-only fitting.
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
        
        def ma_step(carry: Array, x: Array) -> Tuple[Array, Array]:
            """
            Single MA recursion step for conditional sum-of-squares residuals.

            Detailed Description:
                Given the current partial residual (w - AR part) at time t and
                a buffer of the last q residuals, computes the full residual
                as x - dot(theta, carry) and returns the updated buffer (new
                residual at front, shift right) plus the new residual for
                scan output. Implements the recursive MA equation used in
                arima_css for JIT-friendly residual computation.

            Args:
                carry (Array): Length-q buffer of past residuals [r_{t-1}, ..., r_{t-q}].
                x (Array): Current partial residual (w[t] minus AR contribution).

            Returns:
                Tuple[Array, Array]: (new_carry, new_residual) for jax.lax.scan.

            Notes:
                Role: Inner step of arima_css MA recursion; pure for JIT.
            """
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
    """
    StateSpaceModel

    Description:
        Immutable container describing the ARIMA state-space representation used
        by Kalman filtering and forecasting kernels.

    Attributes:
        T (Array): State transition matrix.
        Z (Array): Observation vector/matrix.
        V (Array): Process noise covariance matrix.
        a0 (Array): Initial state mean.
        P0 (Array): Initial state covariance.

    Args:
        T (Array): Transition dynamics.
        Z (Array): Observation mapping.
        V (Array): Process covariance.
        a0 (Array): Initial state location.
        P0 (Array): Initial uncertainty.

    Methods:
        _asdict(): Convert fields to mapping.
        _replace(): Return copy with updated fields.

    Returns:
        Carries all linear-Gaussian model matrices needed by Kalman routines.

    Example:
        >>> # Typically returned by make_arima(...)
        >>> isinstance(make_arima(jnp.array([]), jnp.array([]), jnp.array([]), (0,0,0,0,1,0,0)), StateSpaceModel)
        True

    Notes:
        Being a NamedTuple, this structure is immutable and JAX-pytree-friendly.
    """
    T: Array  # Transition Matrix (F in some texts)
    Z: Array  # Observation Matrix (H in some texts)
    V: Array  # Process Noise Covariance (Q in some texts)
    a0: Array # Initial State Mean
    P0: Array # Initial State Covariance

@partial(jax.jit, static_argnames=['arma'])
def make_arima(phi: Array, theta: Array, delta: Array, arma: Tuple[int, ...], kappa: float = 1e6) -> StateSpaceModel:
    """
    Build the state-space representation (T, Z, V, a0, P0) for ARIMA.

    Detailed Description:
        Constructs the companion-form transition matrix T from phi, observation
        vector Z, process noise covariance V from theta, zero initial state a0,
        and initial covariance P0 from getQ0 (or diffuse kappa when
        appropriate). The result is a StateSpaceModel used by _kalman_filter_core
        and kalman_forecast. All arrays are JAX-friendly for jax.lax.scan-based
        Kalman filtering and forecasting. JIT-compiled with static arma.

    Args:
        phi (Array): AR coefficients (expanded to full length).
        theta (Array): MA coefficients (expanded).
        delta (Array): Differencing polynomial (-delta[1:] from convolution).
        arma (Tuple[int, ...]): (p, q, P, Q, m, d, D) for static_argnames.
        kappa (float): Diffuse prior variance for initial state; default 1e6.

    Returns:
        StateSpaceModel: NamedTuple (T, Z, V, a0, P0) for Kalman routines.

    Raises:
        None. Invalid coefficients can produce non-stationary T.

    Side Effects:
        None. Pure function.

    Example:
        >>> mod = make_arima(phi, theta, delta, arma)

    Notes:
        Role: Single place that turns AR/MA/delta into state-space; used by
        arima_like, _forecast_from_params, and predict_arima.
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
    """
    Run the Kalman filter and return sufficient statistics for log-likelihood and residuals.

    Detailed Description:
        Calls _kalman_filter_core to run the prior-form Kalman filter over the
        observed series y using the given state-space model. Extracts and
        returns the sum of squared prediction errors (ssq), the sum of log
        innovation variances (sumlog), the number of valid observations (nu),
        and the standardized residuals. These quantities are used to compute
        the exact Gaussian log-likelihood and information criteria (AIC, BIC,
        AICc) in arima_fit and _objective_ml. JIT-compiled with static arma.

    Args:
        y (Array): Observed univariate series (possibly after differencing and
            mean/exogenous adjustment).
        mod (StateSpaceModel): State-space matrices (T, Z, V, a0, P0) from make_arima.
        arma (Tuple[int, ...]): (p, q, P, Q, m, d, D) for static_argnames.

    Returns:
        Tuple[float, float, int, Array]: (ssq, sumlog, nu, residuals). ssq and
            sumlog are scalars; nu is int; residuals is the 1D array of
            standardized one-step prediction errors.

    Raises:
        None. Non-finite inputs can produce non-finite outputs.

    Side Effects:
        None. Pure JAX computation.

    Example:
        >>> ssq, sumlog, nu, resid = arima_like(y_adj, mod, arma)

    Notes:
        Role: Core likelihood evaluation for ML estimation and model
        selection; used by _objective_ml and arima_fit post-processing.
    """
    (a_final, P_final, ssq, sumlog, nu), residuals, _ = _kalman_filter_core(y, mod)
    return ssq, sumlog, nu, residuals


def _kalman_filter_core(
    y: Array,
    mod: StateSpaceModel,
) -> Tuple[
    Tuple[Array, Array, Array, Array, Array],
    Array,
    Array,
]:
    """
    Run the prior-form Kalman filter and return final state, standardized residuals, and innovations.

    Detailed Description:
        Iterates over the observation sequence y, at each step computing the
        prediction error (innovation), its variance (F), the Kalman gain (K),
        and the posterior state and covariance. Accumulates sum of squared
        standardized errors and sum of log(F) for likelihood computation.
        Returns the final (a, P, ssq, sumlog, nu) carry, the array of
        standardized residuals, and the array of raw innovations. The
        innovations are used by _forecast_from_params and predict_arima for
        MA correction; arima_like uses only the sufficient statistics.

    Args:
        y (Array): Observed series, length n.
        mod (StateSpaceModel): State-space model (T, Z, V, a0, P0).

    Returns:
        Tuple of (final_carry, std_residuals, innovations). final_carry is
        (a, P, ssq, sumlog, nu); std_residuals and innovations are 1D arrays
        of length n.

    Raises:
        None.

    Side Effects:
        None. Pure jax.lax.scan over y.

    Example:
        >>> (a, P, ssq, sumlog, nu), std_resid, innov = _kalman_filter_core(y, mod)

    Notes:
        Role: Single Kalman implementation used for both likelihood
        (arima_like) and innovation extraction (forecasting); keeps
        numerical behavior consistent.
    """
    T, Z, V, a0, P0 = mod.T, mod.Z, mod.V, mod.a0, mod.P0

    def step_prior_carry(
        carry: Tuple[Array, Array, float, float, int],
        y_t: Array,
    ) -> Tuple[Tuple[Array, Array, Array, Array, Array], Tuple[Array, Array]]:
        """
        Single time-step update of the prior-form Kalman filter.

        Detailed Description:
            Computes innovation v = y_t - Z @ a_prior, forecast variance F = Z P Z',
            Kalman gain K, then posterior state and covariance and next prior
            (a_next = T @ a_post, P_next = T P_post T' + V). Accumulates ssq
            and sumlog only when F is finite and below a large threshold.
            Returns updated carry and (std_residual, innovation) for scan.

        Args:
            carry: (a_prior, P_prior, ssq, sumlog, nu).
            y_t: Current scalar observation.

        Returns:
            New carry and (standardized_residual, raw_innovation).

        Notes:
            Role: Inner step of _kalman_filter_core; must be JAX-pure.
        """
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
    Produce multi-step-ahead point forecasts and forecast standard errors from the state-space model.

    Detailed Description:
        Starting from the model's initial state (a0, P0) in mod, iterates
        n_ahead steps: at each step computes the one-step-ahead forecast
        (Z @ a) and its variance (Z P Z'), then updates state to (T @ a, T P T' + V).
        Returns the array of point forecasts and the array of forecast standard
        errors (square roots of the variances). Used by _predict_core,
        _forecast_from_params, and _fused_forecast_kernel. The starting state
        is typically a0/P0 from make_arima (structural forecast) rather than
        the filtered state at end of sample; MA correction is applied
        separately when needed.

    Args:
        n_ahead (int): Number of steps to forecast; must be static for JIT.
        mod (StateSpaceModel): Model with T, Z, V and starting state (a0, P0).

    Returns:
        Tuple[Array, Array]: (forecasts, se), each of shape (n_ahead,). forecasts
            are point predictions; se are forecast standard errors.

    Raises:
        None.

    Side Effects:
        None. Implemented with jax.lax.scan.

    Example:
        >>> fc, se = kalman_forecast(12, mod)

    Notes:
        Role: Core deterministic forecast from state-space; combined with
        MA innovation correction and deterministic terms for full ARIMA forecast.
    """
    T, Z, V, a_start, P_start = mod.T, mod.Z, mod.V, mod.a0, mod.P0
    
    # Define the recursive step
    def step(
        carry: Tuple[Array, Array],
        _: None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """
        Advance state and covariance one step and compute one-step-ahead forecast and variance.

        Detailed Description:
            Applies the transition a_next = T @ a_curr, P_next = T @ P_curr @ T' + V,
            then computes forecast = Z @ a_next and variance = Z @ P_next @ Z'.
            Returns the new (a_next, P_next) as carry and (forecast, variance) as
            output for jax.lax.scan in kalman_forecast.

        Args:
            carry: (a_curr, P_curr) current state mean and covariance.
            _: Unused (scan over None with length n_ahead).

        Returns:
            New carry (a_next, P_next) and output (forecast, variance).

        Notes:
            Role: Inner step of kalman_forecast; JAX-pure for JIT.
        """
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
    """
    Compute AIC, AICc, BIC, log-likelihood, and residual variance from sufficient statistics.

    Detailed Description:
        Takes the exact log-likelihood (loglik), residual variance (sigma2),
        effective number of observations (n_obs), and number of parameters
        (n_params) and returns AIC = -2*loglik + 2*k, AICc (with small-sample
        correction when n > k+1), and BIC = -2*loglik + k*ln(n). All operations
        use JAX primitives so the result is JIT-safe and can be returned in
        the arima_fit dictionary. Used only after a successful fit to populate
        model selection metrics.

    Args:
        loglik (Array): Scalar log-likelihood from Kalman filter.
        sigma2 (Array): Scalar residual variance (innovation variance).
        n_obs (Array): Scalar effective number of observations (e.g. nu from arima_like).
        n_params (int): Total number of estimated parameters (ARMA + regression + variance).

    Returns:
        Dict[str, Array]: Keys "aic", "aicc", "bic", "loglik", "sigma2"; values
            are JAX arrays (scalars) for downstream use in arima_fit.

    Raises:
        None. Division-by-zero guarded with 1e-10; denom <= 0 yields inf for AICc.

    Side Effects:
        None. Pure function.

    Example:
        >>> metrics = _compute_metrics(loglik, sigma2, nu, n_params)

    Notes:
        Role: Central place for information-criteria computation so
        arima_fit and model selection use consistent formulas.
    """
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

def _unpack_and_adjust(
    params: Array,
    y: Array,
    xreg: Optional[Array],
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
) -> Tuple[Array, Array, Array]:
    """
    Unpack ARMA and regression parameters and adjust the series for mean and exogenous terms.

    Detailed Description:
        Transforms the flat parameter vector into (phi, theta) via arima_transpar
        and extracts regression coefficients (exogenous and intercept). Pads
        the coefficient vector with a dummy zero so that JIT-safe indexing
        (dynamic_slice, dynamic_index_in_dim) never fails when ncxreg is 0.
        Subtracts the exogenous contribution (X @ beta) and the intercept
        from y when applicable, using jax.lax.cond for JIT. Returns the
        adjusted series and the expanded phi, theta for use in arima_css or
        make_arima. Used by _objective_css, _objective_ml, _forecast_from_params,
        and _fused_forecast_kernel.

    Args:
        params (Array): Full parameter vector (ARMA + ncxreg).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous matrix or None.
        arma (Tuple[int, ...]): (mp, mq, msp, msq, ns, d, D).
        ncxreg (int): Number of regression coefficients (exog + intercept).
        n_exog (int): Number of exogenous columns only.
        include_mean (bool): Whether to subtract intercept.

    Returns:
        Tuple[Array, Array, Array]: (y_adj, phi, theta). y_adj is the series
            after subtracting X@beta and optionally the intercept; phi and
            theta are the expanded AR and MA coefficient arrays.

    Raises:
        None. Shape fixes ensure no index errors inside JIT.

    Side Effects:
        None. Pure function.

    Example:
        >>> y_adj, phi, theta = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, True)

    Notes:
        Role: Single unpacking and adjustment path for objectives and
        forecast kernels; guarantees shape stability for JIT.
    """
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
    
    def apply_xreg(args: Tuple[Array, Array, Array]) -> Array:
        """
        Subtract the exogenous linear combination (X @ beta) from the response.

        Detailed Description:
            Slices the coefficient vector to the number of exogenous columns,
            computes the fitted exogenous contribution as xr @ beta, and
            returns y_in - xr @ beta. Used inside _unpack_and_adjust via
            jax.lax.cond when n_exog > 0 and xreg is not None, so that the
            objective and forecast kernels see mean-adjusted and exog-adjusted
            series. JIT-safe because safe_coefs is padded and dynamic_slice
            is used.

        Args:
            args (Tuple[Array, Array, Array]): (y_in, xr, cf) — response,
                exogenous matrix, and padded coefficient vector.

        Returns:
            Array: y_in - xr @ beta, same shape as y_in.

        Notes:
            Role: Inner helper for _unpack_and_adjust; keeps cond branch pure.
        """
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

def _objective_css(
    params: Array,
    y: Array,
    xreg: Optional[Array],
    delta: Array,
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
) -> float:
    """
    Evaluate the conditional sum-of-squares (CSS) objective for the optimizer.

    Detailed Description:
        Unpacks params into adjusted series and (phi, theta), then computes
        the CSS fit via arima_css (conditional sum of squared residuals and
        their variance sigma2). Returns log(sigma2 + 1e-8) as the scalar
        objective to minimize; minimizing this is equivalent to minimizing
        sigma2. Used as the first-stage or sole objective in _fit_model_bfgs
        when method is "CSS" or "CSS-ML". Fast because it avoids the Kalman
        filter; less efficient for small samples than ML.

    Args:
        params (Array): Full parameter vector (ARMA + regression).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous regressors or None.
        delta (Array): Differencing polynomial (unused in CSS path but required by signature).
        arma (Tuple[int, ...]): ARMA structure.
        ncxreg (int): Number of regression coefficients.
        n_exog (int): Number of exogenous columns.
        include_mean (bool): Whether intercept is included.

    Returns:
        float: log(sigma2 + 1e-8) where sigma2 is the CSS residual variance.
            Minimized by BFGS in _fit_model_bfgs.

    Raises:
        None. Non-finite sigma2 yields inf via log.

    Side Effects:
        None. Pure function; used inside JIT-compiled _fit_model_bfgs.

    Example:
        >>> loss = _objective_css(params, y, None, delta, arma, ncxreg, n_exog, True)

    Notes:
        Role: CSS branch of ARIMA estimation; provides fast initial fit for
        hybrid CSS-ML or CSS-only estimation.
    """
    y_adj, phi, theta = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
    sigma2, _ = arima_css(y_adj, arma, phi, theta)
    return jnp.log(sigma2 + 1e-8)

def _objective_ml(
    params: Array,
    y: Array,
    xreg: Optional[Array],
    delta: Array,
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
    **kwargs: Any,
) -> float:
    """
    Evaluate the negative log-likelihood (ML objective) for the optimizer.

    Detailed Description:
        Unpacks params into adjusted series and (phi, theta), builds the
        state-space model with make_arima, and runs the Kalman filter via
        arima_like to get sufficient statistics (ssq, sumlog, nu). Computes
        the exact Gaussian log-likelihood and returns one-half of the
        negative log-likelihood as the scalar to minimize. Used as the
        second-stage or sole objective in _fit_model_bfgs when method is
        "ML" or "CSS-ML". More statistically efficient than CSS but costlier
        per evaluation.

    Args:
        params (Array): Full parameter vector (ARMA + regression).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous regressors or None.
        delta (Array): Differencing polynomial for make_arima.
        arma (Tuple[int, ...]): ARMA structure.
        ncxreg (int): Number of regression coefficients.
        n_exog (int): Number of exogenous columns.
        include_mean (bool): Whether intercept is included.
        **kwargs (Any): Ignored; allows uniform callable signature with _objective_css.

    Returns:
        float: 0.5 * nll where nll is the negative log-likelihood. Minimized
            by BFGS in _fit_model_bfgs.

    Raises:
        None. Unstable models can yield inf or non-finite nll.

    Side Effects:
        None. Pure function; used inside JIT-compiled _fit_model_bfgs.

    Example:
        >>> loss = _objective_ml(params, y, None, delta, arma, ncxreg, n_exog, True)

    Notes:
        Role: ML branch of ARIMA estimation; used for final fit quality and
        in hybrid CSS-ML after CSS warm start.
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
    loss_fn: Callable[[Array], float],
    arma: Tuple[int, ...],
    ncxreg: int, 
    n_exog: int, 
    include_mean: bool,
    maxiter: int
) -> Array:
    """
    Minimize the ARIMA objective (CSS or ML) using JAX's BFGS optimizer.

    Detailed Description:
        Wraps the provided loss_fn (e.g. _objective_css or _objective_ml) with
        the current (y, xreg, delta, arma, ncxreg, n_exog, include_mean) so
        that jax.scipy.optimize.minimize sees a single-argument objective.
        Runs BFGS from init_params for up to maxiter iterations and returns
        the optimized parameter vector. Used by arima_fit (and by the fast
        forecast path in arima.py) to obtain fitted coefficients. JIT-compiled
        with static loss_fn and arma for efficiency.

    Args:
        init_params (Array): Initial parameter vector (ARMA + regression).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous matrix or None.
        delta (Array): Differencing polynomial.
        loss_fn (Callable[[Array], float]): Scalar objective; called with
            (p, y, xreg, delta, arma, ncxreg, n_exog, include_mean) inside objective.
        arma (Tuple[int, ...]): ARMA structure (static for JIT).
        ncxreg (int): Number of regression coefficients.
        n_exog (int): Number of exogenous columns.
        include_mean (bool): Whether intercept is included.
        maxiter (int): Maximum BFGS iterations.

    Returns:
        Array: Optimized parameter vector, same shape as init_params.

    Raises:
        None. Optimizer may converge to a local minimum or hit maxiter; caller
        should check model success/metrics.

    Side Effects:
        None. Pure optimization; no mutation of inputs.

    Example:
        >>> opt_params = _fit_model_bfgs(init, y, None, delta, _objective_css, arma, ncxreg, n_exog, True, 100)

    Notes:
        Role: Single optimization entry point for ARIMA fitting; used in
        arima_fit and in the one-shot forecast path in ARIMA.forecast.
    """
    # 1. Define the Objective Function
    # jax.scipy.optimize.minimize expects a function f(params, *args)
    # We bundle our specific arguments into the tuple format it expects later.
    def objective(p: Array) -> float:
        """
        Closure that fixes (y, xreg, delta, arma, ncxreg, n_exog, include_mean) for the optimizer.

        Detailed Description:
            jax.scipy.optimize.minimize expects a function f(x) -> scalar.
            This closure captures the current training data and options and
            calls loss_fn(p, y, xreg, delta, arma, ncxreg, n_exog, include_mean)
            so that BFGS only varies p.

        Args:
            p (Array): Current parameter vector (optimizer variable).

        Returns:
            float: Objective value (e.g. log(sigma2) for CSS or 0.5*nll for ML).

        Notes:
            Role: Adapter between minimize() and the multi-argument loss_fn.
        """
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
    seasonal: Optional[Dict[str, Any]] = None,
    xreg: Optional[Array] = None,
    include_mean: bool = True,
    method: str = "CSS-ML",
    optim_control: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit an ARIMA model to a univariate series and return coefficients, metrics, and diagnostics.

    Detailed Description:
        Sets up the ARIMA structure from order and seasonal, builds the
        differencing polynomial (delta), and initializes parameters (with
        optional drift/mean init from the series). Runs a one or two-phase
        optimization (CSS then ML when method is "CSS-ML", or a single phase
        for "CSS" or "ML"). Post-processes the optimal parameters to compute
        residuals and innovations via the Kalman filter, then exact
        log-likelihood and information criteria (AIC, AICc, BIC). Returns a
        dictionary with "coef", "model" (StateSpaceModel), "residuals",
        "innovations", "sigma2", "loglik", "aic", "aicc", "bic", "arma",
        "delta", "use_drift", "drift_coef", "success", and related keys. Used
        by the ARIMA class in arima.py and by auto_arima_f for the final refit.

    Args:
        x (Array): Training series; converted to float64.
        order (Tuple[int, int, int]): Non-seasonal (p, d, q).
        seasonal (Optional[Dict[str, Any]]): "order" (P, D, Q) and "period" (m); default (0,0,0), period 1.
        xreg (Optional[Array]): Exogenous regressors; optional.
        include_mean (bool): Whether to include intercept/drift.
        method (str): "CSS", "ML", or "CSS-ML" for optimization path.
        optim_control (Optional[Dict[str, Any]]): Optional "steps" (maxiter) for optimizer.

    Returns:
        Dict[str, Any]: Fitted model dict with coef, model, residuals, innovations,
            sigma2, loglik, aic, aicc, bic, arma, delta, nobs, use_drift, drift_coef, success.

    Raises:
        None. Optimizer failure can yield non-finite metrics; success flag indicates validity.

    Side Effects:
        None. Does not mutate x or seasonal.

    Example:
        >>> fit = arima_fit(y, order=(1, 1, 1), seasonal={"order": (0, 0, 0), "period": 1})

    Notes:
        Role: Central fitting routine for both fixed-order (arima.py) and
        automatic (auto_arima_f) ARIMA; single source of truth for likelihood and metrics.
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
    def run_bfgs(
        start_params: Array,
        loss_func: Callable[..., float],
        steps: int,
    ) -> Array:
        """Execute configured BFGS phase for current objective."""
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
    Compute n_ahead-step forecasts and standard errors from a fitted state-space model and parameters.

    Detailed Description:
        Runs the structural Kalman forecast (kalman_forecast) from the model's
        initial state to get the stochastic component, then adds the
        deterministic part (newxreg @ exog_coefs). Forecast standard errors
        are the Kalman forecast variances scaled by sigma2. Used by
        predict_arima for both single and batch models (via _predict_batch_kernel).
        Does not apply MA innovation correction; the caller adds that when
        the model has MA terms and innovations are available.

    Args:
        mod (StateSpaceModel): Fitted state-space model (T, Z, V, a0, P0).
        params (Array): Full coefficient vector (ARMA + n_exog).
        n_ahead (int): Forecast horizon (static for JIT).
        newxreg (Array): Exogenous matrix for horizon, shape (n_ahead, n_exog).
        narma (int): Number of ARMA parameters.
        n_exog (int): Number of exogenous coefficients (after ARMA block).
        sigma2 (float): Residual variance from fit.

    Returns:
        Tuple[Array, Array]: (forecasts, se), each shape (n_ahead,).

    Raises:
        None.

    Side Effects:
        None. JIT-compiled kernel.

    Notes:
        Role: Shared prediction math for single and batched ARIMA models.
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
    Integrate (undo) differencing so forecasts are on the original series scale.

    Detailed Description:
        When d>0 or D>0, the model is fitted on differenced data and Kalman
        forecasts are in differenced space. This function applies the inverse
        of the combined differencing operator: it rebuilds the positive
        differencing polynomial (1-B)^d * (1-B^m)^D and uses it as a
        recurrence y[n+k] = raw[k] + sum(diffc[j] * y[n+k-1-j]) over the last
        nd values of y_train and the h forecast slots. Correctly handles
        d>0, D>0, and combined cases. Used by predict_arima callers (e.g.
        ARIMA/AutoARIMA) after obtaining raw_pred from the core predictor.

    Args:
        raw_pred (Array): Forecasts in differenced space, length h.
        y_train (Array): Training series (before or after differencing,
            depending on caller; typically the adjusted series used for fit).
        arma (Tuple[int, ...]): (p, q, P, Q, m, d, D) to rebuild delta.
        h (int): Forecast horizon (length of raw_pred).

    Returns:
        Array: Forecasts on the original (integrated) scale, length h.

    Raises:
        None. Uses numpy for the recurrence then converts to JAX array.

    Side Effects:
        None. Does not mutate inputs.

    Example:
        >>> fc_orig = _reconstruct_forecast(raw_fc, y_adj, arma, 12)

    Notes:
        Role: Converts differenced-space predictions to level forecasts for
        integrated ARIMA models; required whenever d + D > 0.
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
    model: Dict[str, Any],
    n_ahead: int, 
    newxreg: Optional[Array] = None,
    se_fit: bool = True
) -> Union[Array, Tuple[Array, Array]]:
    """
    Produce n_ahead-step forecasts (and optionally standard errors) from a fitted ARIMA model.

    Detailed Description:
        Dispatches on whether model["coef"] is 1D (single model) or 2D (batch).
        For a single model: builds the regressor matrix (intercept and/or
        newxreg), calls _predict_core for structural forecast and SE, then
        applies MA innovation correction if the model has MA terms and
        innovations are stored. For a batch: prepares batched newxreg and
        calls _predict_batch_kernel; MA correction is not applied in the
        batch path. When the model was fitted with drift (use_drift True),
        intercept/drift is included in the regressor matrix when newxreg is
        None. Returns (pred, se) if se_fit else pred.

    Args:
        model (Dict[str, Any]): Fitted model dict from arima_fit (coef, model,
            arma, sigma2, use_drift, drift_coef, innovations, etc.).
        n_ahead (int): Forecast horizon.
        newxreg (Optional[Array]): Future exogenous values; if None and
            n_exog > 0, a constant/intercept column is used.
        se_fit (bool): If True, return (pred, se); otherwise pred only.

    Returns:
        Union[Array, Tuple[Array, Array]]: Forecasts array, or (forecasts, se)
            when se_fit is True. Single model: shapes (n_ahead,) and (n_ahead,);
            batch: (batch_size, n_ahead) and (batch_size, n_ahead).

    Raises:
        ValueError: If newxreg shape is invalid for batch mode.

    Side Effects:
        None. Does not mutate model.

    Example:
        >>> pred, se = predict_arima(fit, 12, newxreg=None, se_fit=True)

    Notes:
        Role: Public prediction API for both fixed-order and AutoARIMA
        fitted models; supports single and batch prediction.
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
    Return whether the series is effectively constant (JIT-compatible).

    Detailed Description:
        Uses range (max - min) compared to tol rather than strict equality,
        so that floating-point noise does not cause false negatives. Returns
        a JAX array (boolean scalar) so that the result can be used inside
        JIT-compiled code (e.g. ndiffs) without concretization errors. Used
        by ndiffs, nsdiffs, and auto_arima_f to skip differencing or to
        short-circuit to a constant model.

    Args:
        x (Array): Input series (any length).
        tol (float): Tolerance for range; if max(x) - min(x) < tol, treated as constant. Default 1e-10.

    Returns:
        Array: Boolean scalar (JAX array) True if range < tol, False otherwise.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> jax.jit(is_constant)(jnp.array([1.0, 1.0, 1.0]))  # True

    Notes:
        Role: Stationarity and edge-case detection in differencing and
        AutoARIMA; must remain JIT-safe.
    """
    x = jnp.asarray(x)
    
    # Handle NaN safety: If all NaNs, range is NaN. 
    # If standard variance is 0, max == min.
    
    # We use a non-blocking peak-to-peak check
    is_flat = (jnp.max(x) - jnp.min(x)) < tol
    
    return is_flat


@jax.jit
def kpss_test(x: jax.Array) -> float:
    """
    KPSS stationarity test: null is stationarity; returns approximate p-value.

    Detailed Description:
        Demeans the series, computes the KPSS statistic using cumulative
        sums and a HAC variance estimator (Bartlett weights, lag length
        floor(3*sqrt(n)/13)), then maps the statistic to an approximate
        p-value via interpolation in a small critical-value table. Logic
        matches statsmodels. Used by ndiffs to decide whether additional
        differencing is needed (low p-value suggests non-stationarity).
        JIT-compiled; returns a JAX scalar float.

    Args:
        x (jax.Array): Univariate time series.

    Returns:
        float: Approximate p-value (JAX scalar). High p-value supports
            stationarity; low p-value suggests differencing.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> pval = kpss_test(y); d = 0 if pval >= 0.05 else 1

    Notes:
        Role: Decides non-seasonal differencing order d in ndiffs and thus
        in AutoARIMA; must be JIT-safe for use inside ndiffs.
    """
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
    Determine the number of non-seasonal differences needed for stationarity.

    Detailed Description:
        Checks d=0 (raw series) and d=1 (first difference) in parallel: for
        each, computes is_constant and kpss_test; the series is considered
        stationary if it is constant or if the KPSS p-value is at least
        alpha. Returns the smallest d in {0, 1, ..., max_d} for which the
        differenced series is stationary; typically 0 or 1. Used by
        auto_arima_f when d is None to set the integration order before
        order search. JIT-compiled with static max_d.

    Args:
        x (Array): Univariate time series.
        alpha (float): Significance level for KPSS; default 0.05. Stationary if pval >= alpha.
        max_d (int): Maximum number of differences to consider; static, usually 2.

    Returns:
        int: Number of non-seasonal differences (0, 1, or max_d).

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> d = ndiffs(y, alpha=0.05, max_d=2)

    Notes:
        Role: Sets d in AutoARIMA when not user-specified; ensures
        stationarity before AR/MA order selection.
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
        """
        Compute seasonal strength via STL decomposition (preferred when available).

        Detailed Description:
            Runs STL decomposition with period and window parameters, then
            computes strength = 1 - Var(remainder) / Var(remainder + seasonal).
            Returns a value in [0, 1]; high strength suggests seasonal
            differencing is needed. On any exception, falls back to
            get_seasonal_strength_classical. Used inside nsdiffs when
            iteratively deciding seasonal differencing order.

        Args:
            series: Univariate series (JAX or numpy).

        Returns:
            float: Seasonal strength in [0, 1].

        Notes:
            Role: Primary seasonal-strength metric in nsdiffs; STL is more
            robust than classical decomposition when applicable.
        """
        try:
            # Import STL decomposition
            from ..stl import stl_decompose
            
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
        """
        Compute seasonal strength via classical (periodic-mean) decomposition.

        Detailed Description:
            Linearly detrends the series, then estimates the seasonal
            component as periodic means (one value per phase in the period).
            Computes strength = 1 - Var(remainder) / Var(remainder + seasonal).
            Used as the fallback when STL fails and as the main check in the
            nsdiffs loop for consistency. Pure JAX/segment_sum; no external
            STL dependency.

        Args:
            series: Univariate series (JAX array).

        Returns:
            float: Seasonal strength in [0, 1].

        Notes:
            Role: Fallback and primary loop metric in nsdiffs for deciding D.
        """
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
    """
    ARIMAResult

    Description:
        Immutable selection summary for a candidate ARIMA specification.

    Attributes:
        loglik (float): Exact log-likelihood for the model.
        sigma2 (float): Innovation variance estimate.
        aic (float): Akaike Information Criterion.
        bic (float): Bayesian Information Criterion.
        aicc (float): Small-sample corrected AIC.
        ic (float): Selected information criterion value.
        success (bool): Indicates finite and valid fit result.

    Args:
        loglik (float): Fitted model log-likelihood.
        sigma2 (float): Residual variance.
        aic (float): AIC value.
        bic (float): BIC value.
        aicc (float): AICc value.
        ic (float): Chosen criterion value.
        success (bool): Fit validity flag.

    Methods:
        _asdict(): Convert fields to mapping.
        _replace(): Return copy with updated fields.

    Returns:
        Encapsulates candidate-model evaluation metrics for search routines.

    Example:
        >>> ARIMAResult(-1.0, 1.0, 2.0, 3.0, 2.5, 2.0, True).success
        True

    Notes:
        Designed for light-weight transport inside Python search loops.
    """
    loglik: float
    sigma2: float
    aic: float
    bic: float
    aicc: float
    ic: float
    success: bool

def myarima(
    x: Array,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal_order: Tuple[int, int, int] = (0, 0, 0),
    period: int = 1,
    constant: bool = True,
    ic: str = "aic",
    method: str = "CSS-ML", # Default to Hybrid for better accuracy
    xreg: Optional[Array] = None,
) -> ARIMAResult:
    """
    Fit a single ARIMA specification and return information criteria and success flag.

    Detailed Description:
        Calls arima_fit with the given order, seasonal_order, period,
        constant, method, and xreg. Extracts AIC, BIC, AICc, log-likelihood,
        sigma2, and success from the fit dict. Selects the requested IC
        (aic, bic, or aicc) and returns an ARIMAResult with all metrics;
        failed or invalid fits are masked to +inf for ICs and -inf for
        loglik so that the search algorithm can reject them. Used by
        search_arima and by the stepwise path in auto_arima_f to evaluate
        each candidate model without retaining the full fit object.

    Args:
        x (Array): Training series.
        order (Tuple[int, int, int]): (p, d, q).
        seasonal_order (Tuple[int, int, int]): (P, D, Q).
        period (int): Seasonal period.
        constant (bool): Include intercept/drift.
        ic (str): "aic", "bic", or "aicc" for the chosen criterion.
        method (str): "CSS", "ML", or "CSS-ML".
        xreg (Optional[Array]): Exogenous regressors.

    Returns:
        ARIMAResult: Named tuple (loglik, sigma2, aic, bic, aicc, ic, success).
            Failed fits have inf ICs and success=False.

    Raises:
        None. Optimizer failures are reflected in success and inf ICs.

    Side Effects:
        None. Does not mutate x.

    Example:
        >>> r = myarima(y, order=(1, 0, 1), seasonal_order=(0, 0, 0), period=1, ic="aicc")

    Notes:
        Role: Single-candidate evaluator for grid and stepwise search; bridges
        arima_fit and the ARIMAResult contract expected by search_arima and auto_arima_f.
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
) -> Dict[str, Any]:
    """
    Full grid search over (p, q, P, Q) to find the best ARIMA by information criterion.

    Detailed Description:
        Iterates over all (p, q, P, Q) combinations within max_p, max_q,
        actual_max_P, actual_max_Q and total order <= max_order. For each
        combination, calls myarima (which uses JIT-compiled arima_fit) and
        keeps the result with the smallest chosen IC (aic, bic, or aicc).
        Handles drift/mean via a single constant flag when (d+D)==1 or ==0.
        Returns the best fit as a dictionary (aic, bic, aicc, ic, sigma2,
        loglik, arma). Used by auto_arima_f when stepwise=False. Python
        manages the grid; JAX handles the per-model fitting.

    Args:
        x (Array): Training series.
        d, D (int): Fixed differencing orders.
        max_p, max_q, max_P, max_Q (int): Order bounds; P/Q ignored if period<=1.
        max_order (int): p + q + P + Q <= max_order.
        ic (str): "aic", "bic", or "aicc".
        xreg (Optional[Array]): Exogenous regressors.
        allow_drift (bool): Allow drift when d+D==1.
        allow_mean (bool): Allow mean when d+D==0.
        period (int): Seasonal period.
        method (str): Fitting method for each candidate.

    Returns:
        Dict[str, Any]: Best candidate's metrics and arma tuple; used by
            auto_arima_f to refit with the full method.

    Raises:
        RuntimeError: If no model could be estimated (best_res is None).

    Side Effects:
        None. Does not mutate x.

    Example:
        >>> best = search_arima(y, d=1, D=0, max_p=2, max_q=2, period=12, ic="aicc")

    Notes:
        Role: Exhaustive order search when stepwise is disabled; complements
        the stepwise path in auto_arima_f.
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
    Standardize the series to zero mean and unit variance for numerical stability.

    Detailed Description:
        Computes nanmean and nanstd; replaces std with 1.0 when std <= eps to
        avoid division by zero or near-constant series. Returns the
        standardized series and (mean, std_safe) so that forecasts can be
        denormalized later. Used by AutoARIMA and ARIMA in fit() and
        forecast() when standardize is True. Improves optimizer behavior and
        keeps scale consistent across different series.

    Args:
        y (jnp.ndarray): Univariate series.
        eps (float): Minimum std; if std <= eps, use 1.0 instead. Default 1e-10.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: (y_norm, mean, std_safe).
            y_norm = (y - mean) / std_safe.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> y_norm, mu, sig = _aa_standardize(y)

    Notes:
        Role: Normalization for fitting and forecasting; must be paired with
        _aa_denormalize for interpretable forecasts.
    """
    mean = jnp.nanmean(y)
    std = jnp.nanstd(y)
    std_safe = jnp.where(std > eps, std, 1.0)
    return (y - mean) / std_safe, mean, std_safe


def _aa_denormalize(y_norm: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """
    Map standardized forecasts back to the original series scale.

    Detailed Description:
        Applies the inverse of _aa_standardize: y = y_norm * std + mean. Used
        after predict_arima or _reconstruct_forecast when the model was fit
        on standardized data, so that returned forecasts have the same units
        and scale as the original training series. Broadcasts if mean/std are
        scalars and y_norm is a vector.

    Args:
        y_norm (jnp.ndarray): Forecasts or values in standardized space.
        mean (jnp.ndarray): Mean used in standardization (scalar or broadcastable).
        std (jnp.ndarray): Standard deviation used (scalar or broadcastable).

    Returns:
        jnp.ndarray: Values in original scale, same shape as y_norm.

    Raises:
        None.

    Side Effects:
        None. Pure function.

    Example:
        >>> fc_orig = _aa_denormalize(fc_norm, self._y_mean, self._y_std)

    Notes:
        Role: Inverse of _aa_standardize; required for interpretable outputs
        when standardize=True in AutoARIMA/ARIMA.
    """
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
) -> Dict[str, Any]:
    """
    Automatically select and fit the best ARIMA model by information criterion.

    Detailed Description:
        Handles constant series (returns (0,0,0) fit), infers or uses seasonal
        period, determines d and D via ndiffs/nsdiffs when not fixed, then
        either runs a full grid search (stepwise=False) or a stepwise search
        (stepwise=True) over (p, q, P, Q). Each candidate is evaluated with
        myarima; the one with the best IC (aic, bic, or aicc) is refit with
        the requested method (e.g. CSS-ML) and returned as a full arima_fit
        dictionary. Used by the AutoARIMA class as the core selection and
        fitting routine. Exogenous regressors and drift/mean options are
        supported.

    Args:
        x (Array): Training series.
        d, D (Optional[int]): Optional fixed differencing orders; None to infer.
        max_p, max_q, max_P, max_Q (int): Order bounds.
        max_order (int): Total ARMA order budget.
        max_d, max_D (int): Max non-seasonal and seasonal differences.
        start_p, start_q, start_P, start_Q (int): Stepwise starting orders.
        stationary (bool): If True, force d=D=0.
        seasonal (bool): If True, allow seasonal terms and infer D.
        ic (str): "aic", "bic", or "aicc".
        stepwise (bool): If True, stepwise search; else full grid.
        nmodels (int): Max stepwise candidate count.
        method (str): "CSS", "ML", or "CSS-ML" for final refit.
        xreg (Optional[Array]): Exogenous regressors.
        allowdrift (bool): Allow drift when d+D==1.
        allowmean (bool): Allow mean when d+D==0.
        period (int): Seasonal period (e.g. 12 for monthly).

    Returns:
        Dict[str, Any]: Best fit dict from arima_fit (coef, model, residuals,
            sigma2, aic, aicc, bic, arma, etc.).

    Raises:
        RuntimeError: If no ARIMA model could be estimated (e.g. search_arima fails).

    Side Effects:
        None. Does not mutate x.

    Example:
        >>> best = auto_arima_f(y, max_p=3, max_q=3, seasonal=True, period=12)

    Notes:
        Role: Core automatic model selection used by AutoARIMA.fit(); single
        entry point for order search and final refit.
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
    def _to_dict(res: ARIMAResult, p_: int, q_: int, P_: int, Q_: int) -> Dict[str, Any]:
        """
        Convert a single ARIMAResult from myarima into a search-loop payload dict.

        Detailed Description:
            Takes the named tuple returned by myarima (aic, bic, aicc, ic,
            sigma2, loglik, success) and the current order (p_, q_, P_, Q_) and
            builds a dictionary that the stepwise/grid search loop can compare
            (e.g. by fit["ic"]) and that can be merged into the final bestfit
            structure. Used only inside auto_arima_f during stepwise and when
            comparing the null model.

        Args:
            res (ARIMAResult): Result from myarima(...).
            p_, q_, P_, Q_ (int): Current AR/MA and seasonal AR/MA orders.

        Returns:
            Dict[str, Any]: Keys aic, bic, aicc, ic, sigma2, loglik, arma, success.

        Notes:
            Role: Adapter between ARIMAResult and the search loop's dict format.
        """
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
    AutoARIMA

    Description:
        Performs automatic ARIMA model selection and fitting over configured
        search spaces, then exposes forecasting and interval prediction APIs.

    Attributes:
        uses_exog (bool): Whether exogenous features are supported.
        ``model_`` (dict[str, Any] | None): Fitted model payload after ``fit``.
        standardize (bool): Whether to normalize series before optimization.
        _cached_order (tuple[int, int, int] | None): Cached best non-seasonal order.
        _cached_seasonal_order (tuple[int, int, int] | None): Cached best seasonal order.
        _cached_delta (Array | None): Cached differencing polynomial.

    Args:
        d (int | None): Optional non-seasonal differencing override.
        D (int | None): Optional seasonal differencing override.
        max_p (int): Maximum non-seasonal AR order.
        max_q (int): Maximum non-seasonal MA order.
        max_P (int): Maximum seasonal AR order.
        max_Q (int): Maximum seasonal MA order.
        max_order (int): Maximum total ARMA order budget.
        max_d (int): Upper bound for inferred non-seasonal differencing.
        max_D (int): Upper bound for inferred seasonal differencing.
        start_p (int): Initial stepwise AR order.
        start_q (int): Initial stepwise MA order.
        start_P (int): Initial stepwise seasonal AR order.
        start_Q (int): Initial stepwise seasonal MA order.
        stationary (bool): Force stationary differencing (`d=D=0`) when true.
        seasonal (bool): Enable seasonal search behavior.
        ic (str): Information criterion for model selection.
        stepwise (bool): Enable stepwise search over full grid search.
        nmodels (int): Max number of candidate fits during stepwise search.
        method (str): Fitting objective path (`CSS`, `ML`, or hybrid).
        allowdrift (bool): Allow drift models when integration order is one.
        allowmean (bool): Allow mean term for stationary candidates.
        period (int | None): Seasonal period, or auto-detect when `None`.

    Methods:
        fit(): Search and fit the best ARIMA candidate.
        forecast(): Fast fit-and-forecast path with order caching.
        predict(): Forecast from a previously fitted model.
        summary(): Return compact model summary text.

    Returns:
        Controls automatic model selection and forecast generation.

    Example:
        >>> model = AutoARIMA(seasonal=False)
        >>> model.fit(jnp.array([1.0, 1.2, 1.5, 1.7]))
        >>> model.predict(h=2)["mean"]

    Notes:
        Stateful estimator; avoid concurrent mutation from multiple threads.
    """
    uses_exog: bool = True
    
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
    ) -> None:
        """
        Set up AutoARIMA search bounds, options, and internal caches.

        Detailed Description:
            Stores all order bounds (max_p, max_q, max_P, max_Q, max_order,
            max_d, max_D), stepwise settings (start_*, stepwise, nmodels),
            fitting options (method, ic), and flags (stationary, seasonal,
            allowdrift, allowmean). Initializes model_ to None and caches
            (_cached_order, _cached_seasonal_order, _cached_delta, etc.) for
            the fast forecast() path after the first fit. Period can be None
            for auto-detection on first fit.

        Args:
            d, D (Optional[int]): Override differencing; None to infer.
            max_p, max_q, max_P, max_Q, max_order, max_d, max_D (int): As in class docstring.
            start_p, start_q, start_P, start_Q (int): Stepwise starting orders.
            stationary, seasonal (bool): Differencing and seasonal search flags.
            ic (str): Information criterion for selection.
            stepwise (bool): Stepwise vs full grid search.
            nmodels (int): Max stepwise iterations.
            method (str): CSS, ML, or CSS-ML.
            allowdrift, allowmean (bool): Drift and mean inclusion.
            period (Optional[int]): Seasonal period or None for auto-detect.

        Returns:
            None.

        Side Effects:
            Sets instance attributes and caches; no I/O.

        Notes:
            Role: Constructor for AutoARIMA; must be called before fit or forecast.
        """
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
        self.model_: Dict[str, Any] | None = None
        self.standardize: bool = True
        self._y_mean: jnp.ndarray | None = None
        self._y_std: jnp.ndarray | None = None
        # Cached state for fast forecast() path
        self._cached_order: tuple[int, int, int] | None = None
        self._cached_seasonal_order: tuple[int, int, int] | None = None
        self._cached_include_mean: bool | None = None
        self._cached_delta: Array | None = None
        self._cached_arma: tuple[int, ...] | None = None
        self._cached_narma: int | None = None
        self._cached_ncxreg: int | None = None
        self._cached_n_exog: int | None = None
    
    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "AutoARIMA":
        """
        Fit automatic ARIMA model selection on a series.

        Detailed Description:
            Optionally standardizes the series, infers/uses seasonal period,
            executes automatic order search, and caches the winning order for
            subsequent fast forecast calls.

        Args:
            y (jnp.ndarray): Training target series.
            X (jnp.ndarray | None, optional): Optional exogenous regressors.

        Returns:
            AutoARIMA: The fitted estimator instance.

        Raises:
            RuntimeError: Propagated when no valid model can be estimated.

        Side Effects:
            Mutates fitted model state, normalization stats, and cached order
            metadata.

        Example:
            >>> model = AutoARIMA(seasonal=False)
            >>> _ = model.fit(jnp.array([10.0, 11.0, 12.0, 12.5]))

        Notes:
            The first fit may trigger JAX compilation overhead.
        """
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
    
    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None, X_future: Optional[jnp.ndarray] = None, level: Optional[list] = None, fitted: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Produce fast forecasts from history with cached-order optimization.

        Detailed Description:
            On first invocation, this method runs full automatic selection via
            `fit`. On later calls, it reuses cached orders and only runs the
            optimization/forecast kernels needed for fresh predictions.

        Args:
            h (int): Forecast horizon.
            y (jnp.ndarray): Input history series.
            X (jnp.ndarray | None, optional): Optional exogenous matrix.
            X_future (jnp.ndarray | None, optional): Future exogenous regressors (unused; included for BaseForecaster compliance). Default is None.
            level (list | None, optional): Confidence levels (unused; included for BaseForecaster compliance). Default is None.
            fitted (bool, optional): Whether to return fitted values (unused; included for BaseForecaster compliance). Default is False.

        Returns:
            dict[str, jnp.ndarray]: Forecast dictionary containing `mean`.

        Raises:
            RuntimeError: Propagated from optimizer if fitting fails.

        Side Effects:
            May update cache/state when invoked before initial `fit`.

        Example:
            >>> model = AutoARIMA(seasonal=False)
            >>> model.forecast(h=2, y=jnp.array([1.0, 2.0, 3.0]))["mean"].shape
            (2,)

        Notes:
            Fast path intentionally uses CSS objective for low latency.
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
        """
        Forecast from the fitted automatic ARIMA model.

        Detailed Description:
            Uses stored fitted model state to generate mean forecasts and, when
            confidence levels are provided, symmetric interval bounds.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None, optional): Optional future exogenous matrix.
            level (int | tuple[int, ...] | None, optional): Confidence levels.

        Returns:
            dict[str, jnp.ndarray]: Mean forecast and optional interval bounds.

        Raises:
            RuntimeError: If estimator was not fitted.

        Side Effects:
            None; consumes existing model state.

        Example:
            >>> model = AutoARIMA(seasonal=False).fit(jnp.array([1.0, 2.0, 3.0, 4.0]))
            >>> model.predict(h=2, level=95)["mean"].shape
            (2,)

        Notes:
            Integrated models accumulate uncertainty across horizons.
        """
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
        """
        Return a compact textual summary of the fitted model.

        Detailed Description:
            Builds an ARIMA order summary string with AICc when fitted, or a
            not-fitted status message otherwise.

        Args:
            None: This method takes no explicit parameters beyond `self`.

        Returns:
            str: Human-readable model summary.

        Raises:
            None.

        Side Effects:
            None.

        Example:
            >>> AutoARIMA().summary()

        Notes:
            Intended for logging and diagnostics.
        """
        if self.model_ is None: return "Model not fitted"
        p, q, P, Q, m, d, D = self.model_["arma"]
        return f"ARIMA({p},{d},{q})({P},{D},{Q})[{m}] | AICc: {self.model_.get('aicc', 0.0):.4f}"

class ARIMA(BaseForecaster):
    """
    ARIMA

    Description:
        Fixed-order ARIMA forecaster backed by shared JAX optimization kernels.

    Attributes:
        uses_exog (bool): Indicates exogenous support.
        model_ (dict[str, Any] | None): Fitted model payload.
        _delta (Array): Cached differencing polynomial.
        _arma (tuple[int, ...]): Cached ARMA metadata tuple.

    Args:
        order (tuple[int, int, int]): Non-seasonal order `(p, d, q)`.
        seasonal_order (tuple[int, int, int]): Seasonal order `(P, D, Q)`.
        period (int): Seasonal period.
        include_mean (bool): Include deterministic mean/drift term.
        method (str): Optimization method selector.
        alias (str): Friendly model label.
        standardize (bool): Normalize series during fitting.

    Methods:
        fit(): Fit model parameters.
        forecast(): One-shot fit-and-forecast.
        predict(): Forecast from fitted state.

    Returns:
        Provides mean forecasts and optional interval bands.

    Example:
        >>> model = ARIMA(order=(1, 1, 1))
        >>> model.fit(jnp.array([1.0, 2.0, 3.0, 4.0]))

    Notes:
        Stateful estimator; mutable instance attributes are not thread-safe.
    """
    uses_exog: bool = True
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        seasonal_order: Tuple[int, int, int] = (0, 0, 0),
        period: int = 1,
        include_mean: bool = True,
        method: str = "CSS",
        alias: str = "ARIMA",
        standardize: bool = True,
    ) -> None:
        """
        Set up fixed-order ARIMA and precompute differencing and ARMA metadata.

        Detailed Description:
            Stores order, seasonal_order, period, include_mean, method, and
            alias. Precomputes and caches the differencing polynomial (_delta),
            ARMA structure tuple (_arma), and parameter counts (_narma,
            _ncxreg, _n_exog) so that fit() and forecast() do not recompute
            them. Initializes model_ to None and optional standardization
            stats (_y_mean, _y_std). No fitting is performed.

        Args:
            order (Tuple[int, int, int]): (p, d, q).
            seasonal_order (Tuple[int, int, int]): (P, D, Q).
            period (int): Seasonal period.
            include_mean (bool): Include intercept/drift.
            method (str): CSS, ML, or CSS-ML.
            alias (str): Display name.
            standardize (bool): Whether to standardize series in fit/forecast.

        Returns:
            None.

        Side Effects:
            Sets instance attributes; no I/O.

        Notes:
            Role: Constructor for fixed-order ARIMA; caches enable fast
            one-shot forecast() without refitting.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.period = period
        self.include_mean = include_mean
        self.method = method
        self.alias = alias
        self.model_: Dict[str, Any] | None = None
        self.standardize = standardize
        self._y_mean: jnp.ndarray | None = None
        self._y_std: jnp.ndarray | None = None

        # Pre-compute and cache the delta polynomial (depends only on order/period)
        p, d, q = order
        P, D, Q = seasonal_order
        delta = jnp.array([1.0], dtype=jnp.float64)
        for _ in range(d):
            delta = jnp.convolve(delta, jnp.array([1.0, -1.0]))
        for _ in range(D):
            seas_diff = jnp.concatenate([jnp.array([1.0]), jnp.zeros(period - 1), jnp.array([-1.0])])
            delta = jnp.convolve(delta, seas_diff)
        self._delta: Array = -delta[1:]
        self._arma: tuple[int, ...] = (p, q, P, Q, period, d, D)
        self._narma: int = p + q + P + Q
        n_exog = 0
        self._ncxreg: int = n_exog + (1 if include_mean else 0)
        self._n_exog: int = n_exog

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "ARIMA":
        """
        Estimate ARIMA parameters and store the fitted model and training state.

        Detailed Description:
            Optionally standardizes y (and caches _y_mean, _y_std), then calls
            arima_fit with the instance's order, seasonal_order, period,
            include_mean, and method. Stores the returned dict in model_ and
            ensures model_["arma"] has the correct tuple. Saves y_fit as
            y_train_ for use in predict (e.g. for _reconstruct_forecast).
            Returns self for method chaining.

        Args:
            y (jnp.ndarray): Training target series.
            X (Optional[jnp.ndarray]): Optional exogenous regressors (same length as y).

        Returns:
            ARIMA: self, with model_ and y_train_ set.

        Raises:
            None. Optimizer failure is reflected in model_["success"].

        Side Effects:
            Mutates model_, y_train_, _y_mean, _y_std.

        Example:
            >>> model = ARIMA(order=(1, 1, 1)); model.fit(y)

        Notes:
            Role: Single fit entry point for fixed-order ARIMA; required
            before predict() and for reproducible forecast().
        """
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

    def forecast(self, h: int, y: jnp.ndarray, X: Optional[jnp.ndarray] = None, X_future: Optional[jnp.ndarray] = None, level: Optional[list] = None, fitted: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Fit the fixed-order model on the given series and return h-step forecasts in one shot.

        Detailed Description:
            Standardizes y if standardize is True, then runs BFGS (CSS and/or
            ML) using the cached _delta and _arma without building the full
            arima_fit result (no AIC/BIC/residuals). Uses _forecast_from_params
            for a single XLA dispatch from params to forecasts, then
            _reconstruct_forecast to integrate differencing and _aa_denormalize
            to map back to original scale. Exogenous X is not used in this fast
            path. Returns a dict with key "mean" containing the forecast array.
            Useful when only point forecasts are needed and fitting state is
            not retained.

        Args:
            h (int): Forecast horizon.
            y (jnp.ndarray): Training series (used only for this call).
            X (Optional[jnp.ndarray]): Exogenous regressors; not used in current fast path.
            X_future (Optional[jnp.ndarray]): Future exogenous regressors (unused; included for BaseForecaster compliance). Default is None.
            level (list | None, optional): Confidence levels (unused; included for BaseForecaster compliance). Default is None.
            fitted (bool, optional): Whether to return fitted values (unused; included for BaseForecaster compliance). Default is False.

        Returns:
            Dict[str, jnp.ndarray]: {"mean": array of shape (h,)}.

        Raises:
            None.

        Side Effects:
            None. Does not mutate instance state (no fit cache update).

        Example:
            >>> out = model.forecast(12, jnp.array([1.0, 2.0, 3.0, 4.0]))

        Notes:
            Role: One-shot fit-and-forecast for fixed-order ARIMA with
            minimal Python/XLA overhead; no model_ update.
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

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: int | tuple[int, ...] | None = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Produce h-step forecasts (and optional interval bands) from the fitted model.

        Detailed Description:
            Requires a prior fit (model_ is not None). Calls predict_arima
            with model_, n_ahead=h, newxreg=X, and se_fit=(level is not None).
            Reconstructs forecasts from differenced space via
            _reconstruct_forecast and denormalizes if standardize was used.
            When level is provided, scales standard errors for integrated
            models (d+D>0) by cumulative sum of squared SEs and builds
            symmetric intervals using _quantiles. Returns a dict with "mean"
            and optionally "lo" / "hi" keys for each level.

        Args:
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Future exogenous regressors; shape (h, n_exog).
            level (int | tuple[int, ...] | None): Confidence level(s), e.g. 90 or (80, 95).

        Returns:
            Dict[str, jnp.ndarray]: At least "mean"; if level given, "lo" and "hi" per level.

        Raises:
            RuntimeError: If the model has not been fitted.

        Side Effects:
            None. Does not mutate model_.

        Example:
            >>> preds = model.predict(h=12, level=(80, 95))

        Notes:
            Role: Primary prediction API after fit(); supports intervals via level.
        """
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