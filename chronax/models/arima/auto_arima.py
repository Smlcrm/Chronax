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
    - (The fixed-order `ARIMA` estimator lives in `arima.py`, built on this
      module's shared optimization kernels.)
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
from chronax.utils.conformal_intervals import ConformalIntervals


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

    # 2. State Transition Matrix F — must match make_arima's T on the
    # stationary block (phi in COLUMN 0, ones on the SUPERdiagonal) so the
    # solved P is the stationary covariance of the SAME representation the
    # Kalman filter runs: P = T P T' + R R'.
    F = jnp.zeros((r, r), dtype=jnp.float64)
    if p > 0:
        F = F.at[:p, 0].set(phi)
    if r > 1:
        F = F.at[jnp.arange(r - 1), jnp.arange(r - 1) + 1].set(1.0)
        
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
    
    # Ordinary Differencing: w[t] = w[t] - w[t-1]; first element invalid (0).
    # Functional concatenate form, NOT `w.at[1:].add(-w[:-1])`: that scatter
    # reads the same buffer it writes, and XLA (jax 0.6.2 CPU) miscompiles
    # the scatter chain when it fuses into the AR convolution below — the
    # jitted kernel returned different residuals than eager/vmap execution
    # (observed max Δ 0.25 on a 72-pt seasonal series), silently corrupting
    # CSS objectives, fitted params, and ICs for every d>0, p>0 model.
    for _ in range(d):
        w = jnp.concatenate([jnp.zeros(1, dtype=w.dtype), w[1:] - w[:-1]])

    # Seasonal Differencing: w[t] = w[t] - w[t-m]; first m elements invalid (0).
    for _ in range(D):
        w = jnp.concatenate([jnp.zeros(m, dtype=w.dtype), w[m:] - w[:-m]])

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
        Returns the final (a, P, ssq, sumlog, nu) carry — where (a, P) is
        the one-step-ahead prior state after the last observation, the state
        forecasts start from — plus the array of standardized residuals and
        the array of raw innovations. arima_like uses only the sufficient
        statistics.

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


# Steady-state Kalman likelihood (large-n eager ML path). The prior covariance P_t — and
# hence the Kalman gain K_t and innovation variance F_t — are data-independent (functions of
# T, Z, V, P0 only). Convergence speed is set by the MA (invertibility) root. Routed fits
# converge fast (measured: within ~40 steps), so _STEADYSTATE_BURNIN = 512 is a >10x margin;
# the _use_steadystate d+D<=1 gate keeps out the over-differenced region where the MA root
# reaches the unit circle and the gain never converges, and the exact full filter is always
# used for the forecast and reported likelihood, so the residual boundary case forecasts
# exactly regardless. The burn-in runs the full covariance filter on every optimizer
# evaluation, so k0 is a per-eval cost, not a one-time one. _STEADYSTATE_MIN_N gates the
# switch and is >= the burn-in so k0 < n on the routed path.
_STEADYSTATE_BURNIN = 512
_STEADYSTATE_MIN_N = 2048


def _use_steadystate(n_obs: int, arma: Tuple[int, ...]) -> bool:
    """Whether the eager ML objective should use the frozen-gain filter: long, non-seasonal,
    at most single-differenced series. Seasonal state memory (period m) can exceed the burn-in,
    and over-differencing (d + D >= 2) drives the MA roots to/past the invertibility boundary
    where the frozen gain never converges and the objective is unreliable; both keep the exact
    full filter. n_obs and arma are concrete at the eager call sites."""
    p, q, P, Q, m, d, D = arma
    return (n_obs > _STEADYSTATE_MIN_N) and (P == 0) and (Q == 0) and (D == 0) and (d + D <= 1)


def _kalman_filter_steadystate(
    y: Array, mod: StateSpaceModel, k0: int
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Steady-state Kalman likelihood carry for long series.

    Runs the exact prior-form filter for the first k0 observations (reusing
    _kalman_filter_core), then freezes the gain and innovation variance and
    propagates the remaining n - k0 steps with a fixed-coefficient linear
    recurrence a_{t+1} = (T - T K Z') a_t + (T K) y_t plus scalar accumulation
    of the standardized squared innovations. The per-step O(r^2..r^3)
    covariance update is skipped on the tail, which is the speed win. Two
    lax.scans and pure ops only, so it is reverse-differentiable (the ML
    optimizer differentiates this objective) and vmap-traceable.

    Exact only where the gain has converged by k0. Convergence speed is set by
    the MA (invertibility) root and slows as it nears the unit circle
    (theta -> +/-1, the over-differencing signature); past the k0 margin the
    frozen tail carries a small per-step bias, so its error GROWS with the tail
    length n - k0 — long series are the risk, not the safe case. Callers that
    need an exact result near that boundary (the forecast, the reported
    likelihood) use _kalman_filter_core directly.

    Args:
        y (Array): Observed (mean/drift-adjusted) series, length n.
        mod (StateSpaceModel): State-space model (T, Z, V, a0, P0).
        k0 (int): Static burn-in length; must satisfy k0 < n.

    Returns:
        Tuple[Array, Array, Array, Array, Array]: (a_final, P_frozen, ssq,
            sumlog, nu) — the same likelihood carry _kalman_filter_core's final
            carry holds (P_frozen is the converged prior covariance).
    """
    T, Z, V = mod.T, mod.Z, mod.V
    n = y.shape[0]

    # Phase 1 — burn-in: the exact filter over the first k0 steps.
    (a_k0, P_k0, ssq_b, sumlog_b, nu_b), _, _ = _kalman_filter_core(y[:k0], mod)

    # Freeze the converged gain / innovation variance from the prior covariance at k0.
    M_inf = P_k0 @ Z
    F_inf = jnp.dot(Z, M_inf)
    safe_F = jnp.where(F_inf < 1e-9, 1e-9, F_inf)
    K_inf = M_inf / safe_F
    logF = jnp.log(safe_F)
    A_froz = T - jnp.outer(T @ K_inf, Z)
    TK = T @ K_inf

    # Phase 2 — frozen-gain tail; no covariance update.
    def tail_step(carry: Tuple[Array, Array], y_t: Array) -> Tuple[Tuple[Array, Array], None]:
        a, ssq = carry
        v = y_t - jnp.dot(Z, a)
        a_next = A_froz @ a + TK * y_t
        return (a_next, ssq + v * v / safe_F), None

    (a_n, ssq_tail), _ = jax.lax.scan(tail_step, (a_k0, 0.0), y[k0:])
    n_tail = n - k0
    ssq = ssq_b + ssq_tail
    sumlog = sumlog_b + n_tail * logF
    nu = nu_b + n_tail
    return a_n, P_k0, ssq, sumlog, nu


# -----------------------------------------------------------------------------
# 3. FORECAST FROM FILTERED STATE
# -----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=['n_ahead'])
def _kalman_forecast_from_state(
    n_ahead: int, mod: StateSpaceModel, a_start: Array, P_start: Array
) -> Tuple[Array, Array]:
    """
    Multi-step-ahead point forecasts and variances from a filtered state.

    Detailed Description:
        Starting from a ONE-STEP-AHEAD PRIOR state (a, P) — exactly what
        _kalman_filter_core's final carry holds after filtering the training
        series — iterates n_ahead steps: reads forecast = Z @ a and
        variance = Z P Z', then advances (a, P) to (T @ a, T P T' + V).
        Read-then-advance from the prior is algebraically identical to
        R/statsforecast's advance-then-read from the last posterior state.
        Because the state carries the AR, MA, and differencing memory, the
        output is the full ARIMA forecast of the (mean/drift-adjusted) series
        on its original integration scale; only the deterministic component
        must be added back by the caller.

    Args:
        n_ahead (int): Number of steps to forecast; static for JIT.
        mod (StateSpaceModel): Model supplying T, Z, V.
        a_start (Array): One-step-ahead prior state mean, shape (rd,).
        P_start (Array): One-step-ahead prior state covariance, shape (rd, rd).

    Returns:
        Tuple[Array, Array]: (forecasts, variances), each of shape (n_ahead,).
            Variances are in innovation-variance units (multiply by sigma2 for
            forecast variances).

    Notes:
        Role: The single forecast recursion behind predict_arima and
        _forecast_from_params.
    """
    T, Z, V = mod.T, mod.Z, mod.V

    def step(
        carry: Tuple[Array, Array],
        _: None,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Read the forecast at the current prior state, then advance it."""
        a_curr, P_curr = carry
        forecast = jnp.dot(Z, a_curr)
        variance = jnp.dot(Z, P_curr @ Z)
        a_next = T @ a_curr
        P_next = V + (T @ P_curr @ T.T)
        return (a_next, P_next), (forecast, variance)

    _, (forecasts, variances) = jax.lax.scan(
        step, (a_start, P_start), None, length=n_ahead
    )
    return forecasts, variances

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


# Generosity factors for the intercept/drift box constraint. Wide
# enough to never bind on a legitimate fit, narrow relative to the runaway
# magnitudes the unconstrained LBFGS produces (drift ~ -3e5 on scale-100 data).
_DRIFT_BOUND_FACTOR = 25.0
_MEAN_BOUND_FACTOR = 3.0


def _intercept_bounds(
    y: Array, d: int, D: int, ns: int
) -> Tuple[Array, Array]:
    """Data-physical box constraint for the mean/drift coefficient.

    The mean/drift is an unconstrained parameter jointly optimized with the
    ARMA coefficients; a poorly-conditioned LBFGS step can drive it to a
    non-physical value (observed: drift = -2.96e5 on a scale-~100 d=1 series),
    which the drift ramp then compounds over the horizon into a ~1e8
    forecast. This returns a generous but finite ``(lo, hi)``
    the coefficient is clipped to, both inside the objective (via
    :func:`_unpack_and_adjust`, so the optimizer converges to the true small
    value rather than wandering off) and post-fit on the raw parameter vector
    the forecast kernel reads.

    For a **drift** (``d + D >= 1``) the coefficient is a constant on the
    ``(d, D)``-differenced series and must stay near that series' mean (a drift
    integrates over the horizon, so it has to be tight); for a **stationary
    mean** (``d + D == 0``) it must lie within a few ranges of the observed
    level. ``d``, ``D``, ``ns`` are static config ints so the differencing loops
    are static and the whole function is vmap-native.
    """
    yd = y
    for _ in range(int(d)):
        yd = yd[1:] - yd[:-1]
    for _ in range(int(D)):
        yd = yd[int(ns):] - yd[: -int(ns)]
    if (int(d) + int(D)) >= 1:
        # The drift coefficient is the PER-STEP slope of the deterministic
        # ramp; a seasonal difference spans ns steps, so the mean seasonal
        # difference corresponds to a per-step drift of mean/ns.
        per_step = ns if (int(D) == 1 and int(ns) > 1 and int(d) == 0) else 1
        center = jnp.nanmean(yd) / per_step
        scale = jnp.nanstd(yd) / per_step + jnp.abs(center) + 1e-8
        width = _DRIFT_BOUND_FACTOR * scale
        return center - width, center + width
    rng = jnp.nanmax(y) - jnp.nanmin(y) + 1e-8
    return jnp.nanmin(y) - _MEAN_BOUND_FACTOR * rng, jnp.nanmax(y) + _MEAN_BOUND_FACTOR * rng


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
        make_arima, plus the clipped intercept/drift coefficient. Used by
        _objective_css, _objective_ml, arima_fit, and _forecast_from_params.

    Args:
        params (Array): Full parameter vector (ARMA + ncxreg).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous matrix or None.
        arma (Tuple[int, ...]): (mp, mq, msp, msq, ns, d, D).
        ncxreg (int): Number of regression coefficients (exog + intercept).
        n_exog (int): Number of exogenous columns only.
        include_mean (bool): Whether to subtract intercept.

    Returns:
        Tuple[Array, Array, Array, Array]: (y_adj, phi, theta, intercept).
            y_adj is the series after subtracting X@beta and the deterministic
            component (constant mean for d+D==0, drift ramp for d+D==1,
            nothing for d+D>=2); phi and theta are the expanded AR and MA
            coefficient arrays; intercept is the clipped, include_mean-gated
            mean/drift coefficient the forecast kernels add back.

    Raises:
        None. Shape fixes ensure no index errors inside JIT.

    Side Effects:
        None. Pure function.

    Example:
        >>> y_adj, phi, theta, c = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, True)

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

    # Box-constrain the mean/drift inside the objective so the
    # optimizer converges to the true (small) value instead of a runaway (which
    # the d>0 forecast integrator amplifies to ~1e8). No-op for well-behaved
    # fits — the true optimum sits well inside the bound.
    _lo, _hi = _intercept_bounds(y, d, D, ns)
    intercept_val = jnp.clip(intercept_val, _lo, _hi)

    # Deterministic component, R semantics. d and D come from the static
    # `arma` tuple so the branch is trace-safe; `include_mean` can be a traced
    # value on the objective path, so it gates through jnp.where (when it is
    # False, ncxreg == 0 and intercept_val is the padded dummy 0.0 anyway):
    #   d+D == 0 — constant mean, subtracted as a constant;
    #   d+D == 1 — drift, subtracted as the linear ramp c*t so differencing
    #              leaves a DEMEANED series for the ARMA part to fit (a
    #              constant would be annihilated by the differencing and leave
    #              the drift contaminating the differenced series' mean);
    #   d+D >= 2 — no deterministic term (as in R's arima).
    gated_intercept = jnp.where(include_mean, intercept_val, 0.0)
    if d + D == 0:
        y_adj = y_adj - gated_intercept
    elif d + D == 1:
        t_ramp = jnp.arange(1, y.shape[0] + 1, dtype=jnp.float64)
        y_adj = y_adj - gated_intercept * t_ramp

    return y_adj, phi, theta, gated_intercept

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
    y_adj, phi, theta, _ = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
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
    y_adj, phi, theta, _ = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
    mod = make_arima(phi, theta, delta, arma)
    ssq, sumlog, nu, _ = arima_like(y_adj, mod, arma)
    
    safe_nu = jnp.maximum(nu, 1.0)
    safe_ssq = jnp.maximum(ssq, 1e-8)
    nll = safe_nu * jnp.log(safe_ssq / safe_nu) + sumlog
    return 0.5 * nll

def _objective_ml_ss(
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
    ML negative-log-likelihood via the steady-state Kalman filter (large-n eager path).

    Identical to _objective_ml but evaluates the likelihood with
    _kalman_filter_steadystate (frozen-gain tail) instead of the full filter,
    which is what makes each optimizer evaluation cheap. Matches _objective_ml
    (value and gradient) to machine precision when the MA roots sit inside the
    invertibility boundary by margin — the common case — and stays close enough
    to place the optimizer at the same optimum otherwise; near the boundary
    (theta -> +/-1) the frozen tail is approximate. Only the fitted parameters
    flow out of here: the forecast and the reported likelihood/IC are always
    recomputed with the exact full filter, so a fit near the boundary still
    forecasts and scores exactly. Same signature as _objective_ml so the fit
    kernels accept it as a drop-in loss_fn.
    """
    y_adj, phi, theta, _ = _unpack_and_adjust(params, y, xreg, arma, ncxreg, n_exog, include_mean)
    mod = make_arima(phi, theta, delta, arma)
    _, _, ssq, sumlog, nu = _kalman_filter_steadystate(y_adj, mod, _STEADYSTATE_BURNIN)

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


# Fast-path (forecast) optimizer budget — config-static scan lengths.
_FAST_LBFGS_MEMORY = 10
_FAST_LBFGS_LS_STEPS = 20
# Early-exit rule for the fast-path while_loop: stop after _FAST_LBFGS_PATIENCE
# consecutive iterations whose EVALUATED loss fails to improve the running best
# by a relative _FAST_LBFGS_FTOL (scipy L-BFGS-B factr=1e7 equivalent). maxiter
# stays the hard cap, so no run does more work than the old fixed budget.
_FAST_LBFGS_FTOL = 2.22e-9
_FAST_LBFGS_PATIENCE = 2


@partial(jax.jit, static_argnames=['loss_fn', 'arma', 'maxiter'])
def _fit_model_scan(
    init_params: Array,
    y: Array,
    xreg: Optional[Array],
    delta: Array,
    loss_fn: Callable[[Array], float],
    arma: Tuple[int, ...],
    ncxreg: int,
    n_exog: int,
    include_mean: bool,
    maxiter: int,
) -> Array:
    """
    Batch-stable fixed-budget L-BFGS for the one-shot forecast fast path.

    Detailed Description:
        Drop-in replacement for _fit_model_bfgs on the forecast fast path.
        jax.scipy's BFGS returns an inconsistent (fun, x) pair when its zoom
        line search fails (nit=1/status=3 on seasonal-MA specs like
        (1,1,1)(0,1,1)[12]), so the fast path — the code conformity_scores
        vmaps — uses this pure-lax optimizer instead (optax.lbfgs + zoom
        linesearch with best-iterate tracking): a failed search can never
        return a worse-than-init point. Same pattern as the GARCH optax
        fallback.

        The loop is a convergence-gated lax.while_loop rather than a fixed-length
        lax.scan, which would pay every one of `maxiter` O(n) CSS steps even when the
        fit converges in a fraction of them — costly on long series. Exit fires after
        _FAST_LBFGS_PATIENCE consecutive evaluated losses that fail to
        improve the best by a relative _FAST_LBFGS_FTOL; `maxiter` remains
        the hard cap, so cost is never above the old fixed budget. Per-lane
        SEMANTICS are identical between eager and vmapped runs: `done`
        latches on the PREVIOUS trip (the latching trip's own update still
        lands) and converged lanes ride along frozen while other vmap lanes
        finish; numerics agree to vmap-lowering noise (measured 4.5e-11 —
        batched reductions associate differently, the pre-existing class).
        ⚠ Because the exit test thresholds a lowering-sensitive loss, a ~ULP
        eager-vs-vmap loss difference near ftol can in principle flip one exit and
        bifurcate a trajectory — accepted repo-wide for convergence-gated
        optimizers, so endpoints are compared on accuracy, not bits.
        Non-finite losses count as
        non-improving, so a diverging run exits after PATIENCE trips and
        returns the tracked best (or init) exactly as before.

        Best-iterate tracking stores the point the loss was EVALUATED at
        (not the post-update point), and the final iterate is evaluated once
        after the scan so it also competes; a diverging run therefore never
        returns anything worse than init_params.

    Args:
        init_params (Array): Initial parameter vector (ARMA + regression).
        y (Array): Training series.
        xreg (Optional[Array]): Exogenous matrix or None.
        delta (Array): Differencing polynomial.
        loss_fn (Callable[[Array], float]): Scalar objective; called with
            (p, y, xreg, delta, arma, ncxreg, n_exog, include_mean).
        arma (Tuple[int, ...]): ARMA structure (static for JIT).
        ncxreg (int): Number of regression coefficients.
        n_exog (int): Number of exogenous columns.
        include_mean (bool): Whether intercept is included.
        maxiter (int): L-BFGS step budget (static: derives from config only).

    Returns:
        Array: Best parameter vector found, same shape as init_params.

    Raises:
        None.

    Side Effects:
        None. Pure optimization.

    Notes:
        Role: Optimizer for ARIMA.forecast / AutoARIMA.forecast fast paths —
        the code conformity_scores vmaps. arima_fit (fit/predict path) keeps
        jax.scipy BFGS; it never runs under vmap.
    """
    def objective(p: Array) -> Array:
        return loss_fn(p, y, xreg, delta, arma, ncxreg, n_exog, include_mean)

    value_and_grad_fn = jax.value_and_grad(objective)

    solver = optax.lbfgs(
        memory_size=_FAST_LBFGS_MEMORY,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=_FAST_LBFGS_LS_STEPS,
            initial_guess_strategy="one",
        ),
    )

    # ftol floored at 8*eps(dtype): a relative tolerance below the dtype's
    # resolution is unreachable and would disable the exit (the AutoCES f32
    # tolerance lesson). init_params.dtype is concrete at trace time.
    ftol = max(_FAST_LBFGS_FTOL, 8.0 * float(jnp.finfo(init_params.dtype).eps))

    def _lbfgs_cond(carry):
        _, _, _, _, it, _, done = carry
        return (it < maxiter) & jnp.logical_not(done)

    def _lbfgs_body(carry):
        params, state, best_params, best_loss, it, stall, done = carry
        loss, grads = value_and_grad_fn(params)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = solver.update(
            grads, state, params, value=loss, grad=grads, value_fn=objective)
        new_params = optax.apply_updates(params, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        # "meaningful" = improves the best by more than ftol relative. A first
        # finite loss against best_loss=inf always counts; NaN/inf losses never
        # do (so divergence accumulates stall instead of resetting it).
        denom = jnp.maximum(jnp.maximum(jnp.abs(best_loss), jnp.abs(loss)), 1.0)
        meaningful = improved & jnp.where(
            jnp.isfinite(best_loss), (best_loss - loss) / denom > ftol, True)
        new_stall = jnp.where(meaningful, 0, stall + 1)
        new_done = done | (new_stall >= _FAST_LBFGS_PATIENCE)
        new_best_params = jnp.where(improved, params, best_params)
        new_best_loss = jnp.where(improved, loss, best_loss)
        # Freeze on the PREVIOUS trip's latch: the trip that trips `done` still
        # commits its own update (eager exit happens after it), and under vmap a
        # converged lane rides along as a no-op. For a lane that instead runs out
        # of `maxiter` without latching, per-lane parity is carried by JAX's
        # batched-while lowering itself (the entire carry is re-selected on the
        # entry predicate per lane) — the freeze alone does not cover that case.
        frz = done
        return (jnp.where(frz, params, new_params),
                jax.tree_util.tree_map(
                    lambda o, n: jnp.where(frz, o, n), state, new_state),
                jnp.where(frz, best_params, new_best_params),
                jnp.where(frz, best_loss, new_best_loss),
                jnp.where(frz, it, it + 1),
                jnp.where(frz, stall, new_stall),
                new_done)

    init_loss = objective(init_params)
    safe_init_loss = jnp.where(jnp.isfinite(init_loss), init_loss, jnp.inf)
    carry0 = (init_params, solver.init(init_params), init_params, safe_init_loss,
              jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32),
              jnp.asarray(False))
    final_params, _, best_params, best_loss, _, _, _ = lax.while_loop(
        _lbfgs_cond, _lbfgs_body, carry0)

    final_loss = objective(final_params)
    take_final = jnp.isfinite(final_loss) & (final_loss < best_loss)
    return jnp.where(take_final, final_params, best_params)


# Jitted aliases for the EAGER post-processing call sites only.
# Both raw functions build their `lax.cond` / `lax.scan` bodies as fresh closures,
# so an eager call recompiles an identical jaxpr every time — 16 of the 17 compiles
# a fresh-instance `AutoARIMA.forecast` pays (one cond + one scan per candidate of
# the stepwise search), plus two per eager `ARIMA.fit()`.
#
# The RAW functions must stay in place: `_unpack_and_adjust` is also called from
# inside `_objective_css`/`_objective_ml`, which pass `ncxreg`/`n_exog`/`include_mean`
# as *traced* ints — `static_argnames` on the shared symbol raises "Non-hashable
# static arguments ... DynamicJaxprTracer". Jitting only the eager sites also leaves
# every existing compiled kernel's jaxpr byte-identical — no nested pjit inside the
# programs that contain `arima_css`, which must keep its current fusion.
_unpack_and_adjust_jit = partial(
    jax.jit, static_argnames=['arma', 'ncxreg', 'n_exog', 'include_mean']
)(_unpack_and_adjust)
_kalman_filter_core_jit = jax.jit(_kalman_filter_core)


def arima_fit(
    x: Array,
    order: Tuple[int, int, int] = (0, 0, 0),
    seasonal: Optional[Dict[str, Any]] = None,
    xreg: Optional[Array] = None,
    include_mean: bool = True,
    method: str = "CSS-ML",
    optim_control: Optional[Dict[str, Any]] = None,
    steadystate: bool = True,
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
    # Initialize the per-step drift from the mean of the differenced series;
    # a seasonal difference spans `period` steps, so its mean corresponds to
    # a per-step slope of mean/period.
    if use_drift:
        if D == 1 and period > 1:
            dx = x[period:] - x[:-period]
            per_step = period
        else:
            dx = x[1:] - x[:-1]
            per_step = 1

        mean_dx = jnp.nanmean(dx) / per_step
        init_params = init_params.at[narma + n_exog].set(mean_dx)

    # Optimization Settings
    if optim_control is None: optim_control = {}
    maxiter = optim_control.get("steps", 100)
    
    # 3. OPTIMIZATION LOOP
    method = method.upper()
    current_params = init_params
    
    # Helper to call the JIT kernel
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
        # Each phase gets the FULL budget (R gives its CSS and ML optim
        # phases independent default budgets; splitting starved both).
        current_params = run_bfgs(current_params, _objective_css, maxiter)

    if "ML" in method:
        # Long non-seasonal series use the steady-state likelihood (frozen-gain
        # tail, exact after burn-in) so each optimizer evaluation avoids the
        # full-series covariance recursion. AutoARIMA's stepwise search passes
        # steadystate=False so its candidate ICs and selected fit stay on the
        # exact objective (the frozen tail's tiny near-boundary bias must not
        # perturb order selection).
        use_ss = steadystate and _use_steadystate(n_obs, arma)
        ml_obj = _objective_ml_ss if use_ss else _objective_ml
        current_params = run_bfgs(current_params, ml_obj, maxiter)

    # The mean/drift coefficient is estimated unreliably by the ARMA optimizer — a
    # poorly-conditioned BFGS step can drive it to a wrong-sign runaway that the d>0
    # forecast ramp then amplifies over the horizon. Replace it with its
    # closed form, which is well-conditioned and what R's joint estimate converges
    # to anyway:
    #   * drift (d+D==1): per-step mean of the (seasonally-)differenced series;
    #   * stationary mean (d+D==0): the sample mean, clipped to the data range.
    # For well-behaved fits the optimizer already sits at this value, so it is a
    # no-op; it only rescues the degenerate runaways. Applied BEFORE the loglik /
    # IC computation below so order selection scores the corrected model.
    if include_mean:
        _mi = narma + n_exog
        if use_drift:
            if D == 1 and period > 1:
                _dx, _ps = x[period:] - x[:-period], period
            else:
                _dx, _ps = x[1:] - x[:-1], 1
            current_params = current_params.at[_mi].set(jnp.nanmean(_dx) / _ps)
        elif (d + D) == 0:
            _ilo, _ihi = _intercept_bounds(x, d, D, period)
            current_params = current_params.at[_mi].set(
                jnp.clip(current_params[_mi], _ilo, _ihi))
        else:
            # d+D >= 2: no deterministic term (R semantics) — pin the inert
            # slot to zero so reports cannot suggest otherwise.
            current_params = current_params.at[_mi].set(0.0)

    # 4. POST-PROCESSING (eager: use the jitted aliases — see their definition)
    y_adj, phi, theta, _ = _unpack_and_adjust_jit(current_params, x, xreg, arma, ncxreg, n_exog, include_mean)
    mod = make_arima(phi, theta, delta, arma)
    (a_final, P_final, ssq, sumlog, nu), resid, innovations = _kalman_filter_core_jit(y_adj, mod)
    
    sigma2 = jnp.where(nu > 0, ssq / nu, jnp.inf)
    
    # Exact LogLikelihood
    loglik = -0.5 * (nu * jnp.log(sigma2) + sumlog + nu * jnp.log(2 * jnp.pi) + nu)
    
    n_params_total = narma + ncxreg + 1
    metrics = _compute_metrics(loglik, sigma2, nu, n_params_total)
    success = jnp.isfinite(loglik) & (sigma2 > 0)
    
    # Extract the optimized drift from fitted params (not the initial guess).
    # Kept as a jnp scalar: float() here forced a device sync on every fit and
    # concretized under tracing; downstream use is arithmetic-only.
    if use_drift:
        drift_coef = current_params[narma + n_exog]
    
    return {
        "coef": current_params,
        **metrics,
        "model": mod,
        # One-step-ahead prior state after the last observation — the state
        # predict_arima forecasts from.
        "a_final": a_final,
        "P_final": P_final,
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
# FORECASTING
# =============================================================================

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
    Fused forecast kernel: params -> forecast in a single XLA dispatch.

    Runs the Kalman filter over the (mean/drift-adjusted) training series and
    forecasts from the END-OF-SAMPLE filtered state, so the fitted AR and
    seasonal-AR dynamics, the MA memory, and the differencing integration all
    live in the state and shape the forecast path. The deterministic
    component (constant mean for d+D==0, drift ramp for d+D==1) is added
    back at the end. Output is on the same scale as `y`.

    Returns:
        Tuple[Array, Array]: (forecasts, se) — the point forecasts and their
        standard errors (scaled by the filter's own sigma2 = ssq/nu).
    """
    _, _, _, _, _ns, d, D = arma

    # 1. Unpack params, adjust the series, build the state space (ONCE)
    y_adj, phi, theta, intercept = _unpack_and_adjust(
        params, y, None, arma, ncxreg, n_exog, include_mean
    )
    mod = make_arima(phi, theta, delta, arma)

    # 2. Filter the training series; the final carry is the one-step-ahead
    # prior state after the last observation. Always the exact full filter: the
    # frozen-gain state estimate is inaccurate near the MA invertibility boundary,
    # and this single pass is a negligible fraction of the fit cost.
    (a_final, P_final, ssq, _sumlog, nu), _, _ = _kalman_filter_core(y_adj, mod)

    # 3. Forecast from the filtered state.
    forecasts, var_units = _kalman_forecast_from_state(n_ahead, mod, a_final, P_final)
    sigma2 = jnp.where(nu > 0, ssq / nu, jnp.inf)
    se = jnp.sqrt(var_units * sigma2)

    # 4. Deterministic component (mean / drift ramp; nothing for d+D>=2).
    n = y.shape[0]
    if d + D == 0:
        forecasts = forecasts + intercept
    elif d + D == 1:
        ramp = jnp.arange(n + 1, n + n_ahead + 1, dtype=jnp.float64)
        forecasts = forecasts + intercept * ramp

    return forecasts, se


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
        Forecasts from the end-of-sample filtered state stored by arima_fit
        (model["a_final"], model["P_final"]), so the AR/seasonal-AR and MA
        dynamics and the differencing integration are all carried by the
        state; adds the deterministic component (exogenous regression,
        constant mean for d+D==0, or drift ramp for d+D==1). Standard errors
        are the forecast-variance recursion seeded from the filtered
        covariance, scaled by sigma2. Returns (pred, se) if se_fit else pred.

    Args:
        model (Dict[str, Any]): Fitted model dict from arima_fit (coef, model,
            a_final, P_final, arma, sigma2, use_drift, n_obs_train, ...).
        n_ahead (int): Forecast horizon.
        newxreg (Optional[Array]): Future exogenous values for models fitted
            with xreg; the deterministic mean/drift needs no newxreg.
        se_fit (bool): If True, return (pred, se); otherwise pred only.

    Returns:
        Union[Array, Tuple[Array, Array]]: Forecasts array of shape
            (n_ahead,), or (forecasts, se) when se_fit is True.

    Raises:
        ValueError: If newxreg has the wrong number of columns.

    Side Effects:
        None. Does not mutate model.

    Example:
        >>> pred, se = predict_arima(fit, 12, newxreg=None, se_fit=True)

    Notes:
        Role: Public prediction API for fixed-order and AutoARIMA fitted
        models.
    """
    params = model["coef"]
    mod = model["model"]
    arma = model["arma"]
    narma = sum(arma[:4])
    d, D = arma[5], arma[6]
    n_reg = params.shape[0] - narma
    use_drift = model.get("use_drift", False)

    pred, var = _kalman_forecast_from_state(
        n_ahead, mod, model["a_final"], model["P_final"]
    )

    # Deterministic component. The coefficient layout after the ARMA block is
    # [exog..., intercept] (matching _unpack_and_adjust).
    if n_reg > 0:
        n_exog_cols = n_reg - 1 if (use_drift or (d + D) == 0) else n_reg
        if n_exog_cols > 0:
            if newxreg is None:
                raise ValueError(
                    f"model was fitted with {n_exog_cols} exogenous column(s); "
                    "pass newxreg with future values to predict"
                )
            newxreg = jnp.asarray(newxreg, dtype=jnp.float64)
            if newxreg.ndim == 1:
                newxreg = newxreg.reshape(-1, 1)
            if newxreg.shape[1] != n_exog_cols:
                raise ValueError(
                    f"newxreg has {newxreg.shape[1]} column(s); model expects {n_exog_cols}"
                )
            pred = pred + jnp.dot(newxreg, params[narma:narma + n_exog_cols])
        intercept = params[narma + n_exog_cols] if (use_drift or (d + D) == 0) else None
        if use_drift:
            n_train = model["n_obs_train"]
            ramp = jnp.arange(n_train + 1, n_train + n_ahead + 1, dtype=jnp.float64)
            pred = pred + intercept * ramp
        elif (d + D) == 0 and intercept is not None:
            pred = pred + intercept

    if se_fit:
        se = jnp.sqrt(var * model["sigma2"])
        return pred, se
    return pred

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



def _min_arma_root(coef_narma: np.ndarray, arma: Tuple[int, ...]) -> float:
    """Minimum modulus over the fitted AR and MA polynomial roots.

    Expands the (unconstrained-space) ARMA coefficients to their effective
    values, trims trailing near-zero terms, and returns the smallest root
    modulus across the AR polynomial 1 - phi(z) and the MA polynomial
    1 + theta(z) (2.0 when neither has active terms). Host-side numpy — used
    only by the eager order search and final-refit admissibility check.
    """
    phi_eff, theta_eff = arima_transpar(jnp.asarray(coef_narma), arma, trans=True)
    minroot = 2.0
    for vec, sign in ((np.asarray(phi_eff), -1.0), (np.asarray(theta_eff), 1.0)):
        nz = np.abs(vec) > 1e-8
        if nz.any():
            trimmed = vec[: np.max(np.where(nz)[0]) + 1]
            roots = np.polynomial.polynomial.polyroots(np.append(1.0, sign * trimmed))
            if roots.size > 0:
                minroot = min(minroot, np.abs(roots).min())
    return float(minroot)


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
        method=method,
        steadystate=False,   # exact objective for candidate IC comparison
    )

    # 2. Candidate-ranking metrics.
    # method=="CSS" is the approximation-mode search: candidates are ranked by
    # the CSS IC (nstar*log(sigma2_css) + 2*npar, Hyndman-Khandakar) so the
    # stepwise walk visits the same neighbours as the reference search; any
    # other method ranks by the exact Kalman ICs from the fit.
    success = fit['success']
    if method.upper() == "CSS":
        arma_t = fit["arma"]
        mp_, mq_, msp_, msq_, ns_, d_, D_ = arma_t
        narma_ = mp_ + mq_ + msp_ + msq_
        ncxreg_ = int(fit["coef"].shape[0]) - narma_
        n_exog_ = max(ncxreg_ - (1 if constant else 0), 0)
        delta_ = fit["delta"]
        css_obj = _objective_css(
            fit["coef"], jnp.asarray(x, dtype=jnp.float64), xreg, delta_,
            arma_t, ncxreg_, n_exog_, constant,
        )
        sigma2_css = jnp.exp(css_obj)
        nstar = x.shape[0] - d_ - D_ * ns_
        npar = narma_ + ncxreg_ + 1
        aic_val = nstar * jnp.log(sigma2_css) + 2.0 * npar
        bic_val = aic_val + npar * (jnp.log(nstar) - 2.0)
        aicc_val = jnp.where(
            nstar - npar - 1 != 0,
            aic_val + 2.0 * npar * (npar + 1) / (nstar - npar - 1),
            jnp.inf,
        )
    else:
        aic_val = fit['aic']
        bic_val = fit['bic']
        aicc_val = fit['aicc']

    # 3. Decision Logic (IC Selection)
    # Mapping string names to values using jnp.where for JIT compatibility
    chosen_ic = jnp.where(
        ic == "aic", aic_val,
        jnp.where(ic == "bic", bic_val,
        jnp.where(ic == "aicc", aicc_val, aic_val))
    )

    # 4. Near-unit-root veto (Hyndman-Khandakar): reject candidates whose
    # fitted AR or MA polynomial has a root with modulus below 1.01 — such
    # fits sit on the (non-)stationarity/invertibility boundary and forecast
    # erratically even when their in-sample IC looks good. Host-side numpy is
    # fine: myarima only runs inside the eager order search.
    p_ord, _, q_ord = order
    P_ord, _, Q_ord = seasonal_order
    narma = p_ord + q_ord + P_ord + Q_ord
    minroot = (
        _min_arma_root(np.asarray(fit["coef"])[:narma], fit["arma"])
        if narma > 0 else 2.0
    )

    # 5. Global Failure Mask
    # If the optimizer failed, the variance is non-positive, or the roots sit
    # inside the veto band, invalidate the model so the search rejects it.
    mask = success & jnp.isfinite(chosen_ic) & (minroot >= 1.01)

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
                                "arma": (p, q, P, Q, period, d, D),
                                "constant": (K == 1)
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
        after predict_arima or _forecast_from_params when the model was fit
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
        return arima_fit(x, order=(0, 0, 0), include_mean=allowmean, method=method, steadystate=False)
    
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
    # Approximation rule (Hyndman-Khandakar): search with CSS (scored by the
    # CSS-approximation IC inside myarima) only when the series is long or
    # strongly seasonal; otherwise search with the full method and exact ICs.
    search_method = "CSS" if (x.shape[0] > 150 or m > 12) else method

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
    
    tried: list = []

    # Full Grid Search
    if not stepwise:
        bestfit = search_arima(
            x, d=d_val, D=D_val, period=m,
            max_p=max_p, max_q=max_q, max_P=eff_max_P, max_Q=eff_max_Q,
            max_order=max_order, ic=ic, method=search_method, xreg=xreg,
            allow_drift=allowdrift, allow_mean=allowmean
        )
    # Stepwise Search (Hyndman-Khandakar: 5 starting models, 17 moves per
    # round — single steps, diagonal steps, and the constant toggle — with a
    # visited-set so no candidate is ever fitted twice; the scan restarts
    # from the first move after every improvement. max_order restricts the
    # grid search only, not the stepwise walk.)
    else:
        p = min(start_p, max_p)
        q = min(start_q, max_q)
        P = min(start_P, eff_max_P) if m > 1 else 0
        Q = min(start_Q, eff_max_Q) if m > 1 else 0

        constant = (allowdrift and d_val + D_val == 1) or (allowmean and d_val + D_val == 0)
        can_toggle = constant

        visited: set = set()
        k_count = 0

        def _try_candidate(p_: int, q_: int, P_: int, Q_: int, const_: bool) -> Dict[str, Any]:
            nonlocal k_count
            visited.add((p_, q_, P_, Q_, const_))
            k_count += 1
            res_ = myarima(
                x, order=(p_, d_val, q_), seasonal_order=(P_, D_val, Q_), period=m,
                constant=const_, ic=ic, method=search_method, xreg=xreg
            )
            fit_ = _to_dict(res_, p_, q_, P_, Q_)
            fit_["constant"] = const_
            tried.append(fit_)
            return fit_

        # Starting models: user start orders, the null model, a pure-AR seed,
        # a pure-MA seed, and (when a constant is admissible) the bare
        # no-constant null.
        bestfit = _try_candidate(p, q, P, Q, constant)
        fit = _try_candidate(0, 0, 0, 0, constant)
        if fit["ic"] < bestfit["ic"]:
            bestfit = fit
            p = q = P = Q = 0
        if max_p > 0 or eff_max_P > 0:
            p_ = int(max_p > 0)
            P_ = int(m > 1 and eff_max_P > 0)
            if (p_, 0, P_, 0, constant) not in visited:
                fit = _try_candidate(p_, 0, P_, 0, constant)
                if fit["ic"] < bestfit["ic"]:
                    bestfit = fit
                    p, P, q, Q = p_, P_, 0, 0
        if max_q > 0 or eff_max_Q > 0:
            q_ = int(max_q > 0)
            Q_ = int(m > 1 and eff_max_Q > 0)
            if (0, q_, 0, Q_, constant) not in visited:
                fit = _try_candidate(0, q_, 0, Q_, constant)
                if fit["ic"] < bestfit["ic"]:
                    bestfit = fit
                    q, Q, p, P = q_, Q_, 0, 0
        if constant and (0, 0, 0, 0, False) not in visited:
            fit = _try_candidate(0, 0, 0, 0, False)
            if fit["ic"] < bestfit["ic"]:
                bestfit = fit
                p = q = P = Q = 0

        improved = True
        while improved and k_count < nmodels:
            improved = False
            moves = [
                (p, q, P - 1, Q, constant), (p, q, P, Q - 1, constant),
                (p, q, P + 1, Q, constant), (p, q, P, Q + 1, constant),
                (p, q, P - 1, Q - 1, constant), (p, q, P - 1, Q + 1, constant),
                (p, q, P + 1, Q - 1, constant), (p, q, P + 1, Q + 1, constant),
                (p - 1, q, P, Q, constant), (p, q - 1, P, Q, constant),
                (p + 1, q, P, Q, constant), (p, q + 1, P, Q, constant),
                (p - 1, q - 1, P, Q, constant), (p - 1, q + 1, P, Q, constant),
                (p + 1, q - 1, P, Q, constant), (p + 1, q + 1, P, Q, constant),
            ]
            if can_toggle:
                moves.append((p, q, P, Q, not constant))
            for new_p, new_q, new_P, new_Q, new_c in moves:
                if k_count >= nmodels:
                    break
                if not (0 <= new_p <= max_p and 0 <= new_q <= max_q and
                        0 <= new_P <= eff_max_P and 0 <= new_Q <= eff_max_Q):
                    continue
                if (new_p, new_q, new_P, new_Q, new_c) in visited:
                    continue
                fit = _try_candidate(new_p, new_q, new_P, new_Q, new_c)
                if fit["ic"] < bestfit["ic"]:
                    bestfit = fit
                    p, q, P, Q, constant = new_p, new_q, new_P, new_Q, new_c
                    improved = True
                    break

    # --- Final refit with the full method, walking the candidate ranking ---
    # The search fits candidates with the (cheaper) search method; the final
    # model is re-fit with the full method. An exact-method endpoint can land
    # on the near-unit-root veto band even when the search endpoint did not,
    # so walk the tried candidates in search-IC order and accept the first
    # whose exact refit is admissible (finite likelihood + roots outside the
    # veto band) — the same fallback statsforecast runs after an
    # approximation-mode search.
    default_constant = (allowdrift and d_val + D_val == 1) or (allowmean and d_val + D_val == 0)
    ranked = sorted(
        (f for f in tried if np.isfinite(f["ic"])), key=lambda f: f["ic"]
    )
    candidates = ranked if ranked else [bestfit]

    final_model = None
    for cand in candidates[:10]:
        c_p, c_q, c_P, c_Q, _, _, _ = cand["arma"]
        c_const = cand.get("constant", default_constant)
        fit_try = arima_fit(
            x,
            order=(c_p, d_val, c_q),
            seasonal={'order': (c_P, D_val, c_Q), 'period': m},
            xreg=xreg,
            include_mean=c_const,
            method=method,
            steadystate=False,   # exact objective for candidate IC comparison
        )
        if not bool(fit_try["success"]):
            continue
        narma_c = c_p + c_q + c_P + c_Q
        if narma_c > 0 and _min_arma_root(
            np.asarray(fit_try["coef"])[:narma_c], fit_try["arma"]
        ) < 1.01:
            continue
        final_model = fit_try
        break
    if final_model is None:
        # Every ranked candidate failed the exact-method admissibility check —
        # fall back to the search winner's spec unvetoed rather than raising.
        c_p, c_q, c_P, c_Q, _, _, _ = bestfit["arma"]
        final_model = arima_fit(
            x,
            order=(c_p, d_val, c_q),
            seasonal={'order': (c_P, D_val, c_Q), 'period': m},
            xreg=xreg,
            include_mean=bestfit.get("constant", default_constant),
            method=method,
            steadystate=False,   # exact objective for the fallback final fit
        )

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
    # Exogenous support is not wired end-to-end (the cached-order fast path
    # and the CV path ignore X), so it is not advertised; passing X raises
    # instead of silently dropping it.
    uses_exog: bool = False
    
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
        conformal_params: Optional[ConformalIntervals] = None,
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
        self.conformal_params = conformal_params
        self._cs: jnp.ndarray | None = None
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
        # fit() hosts the order search (numpy/scipy stepwise) and is eager by
        # design; the vmapped CV path never calls it directly — see
        # conformity_scores() below for how conformal CV reaches forecast().
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        if X is not None:
            raise ValueError(
                "AutoARIMA does not currently support exogenous regressors "
                "end-to-end; fit accepts only y (use auto_arima_f(xreg=...) "
                "for the low-level exogenous API)."
            )

        # Everything below stays in locals until the host search succeeds:
        # assigning traced/partial state to self earlier would leave a
        # polluted instance behind if fit is (incorrectly) reached under a
        # trace, which dies loudly in the search instead.
        if self.standardize:
            y_fit, y_mean, y_std = _aa_standardize(y_jax)
        else:
            y_fit = y_jax
            y_mean = jnp.array(0.0, dtype=jnp.float64)
            y_std = jnp.array(1.0, dtype=jnp.float64)

        # Auto-detect period if not specified
        if self._user_period is None:
            n = int(y_jax.shape[0])
            period = detect_period(np.asarray(y_jax), max_period=min(n // 4, 24))
        else:
            period = self._user_period

        model = auto_arima_f(
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
            period=period,
        )

        # Search succeeded — commit state to self.
        self._y_mean = y_mean
        self._y_std = y_std
        self.y_train_ = y_fit
        self.period = period
        self.model_ = model

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

        # Pre-compute and cache conformity scores on the training series for
        # predict() intervals (sibling convention: AutoCES/ARIMA). Runs after
        # the order caches above, so the vmapped per-window forecast takes the
        # traceable fixed-order fast path.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y_jax, X=X)
        else:
            self._cs = None

        return self

    def conformity_scores(
        self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Conformity scores with eager order selection, vmapped per-window refit.

        Detailed Description:
            The base-class implementation vmaps ``self.forecast`` over CV
            windows. AutoARIMA's order search (stepwise/grid over numpy and
            Python control flow) cannot trace, so it runs ONCE, eagerly, on
            the full series here; the vmapped ``forecast`` then re-fits the
            *parameters* of that fixed order independently per window via the
            traceable CSS fast path.

            Calibration caveat (shared with the other Auto models): the ORDER is
            selected with sight of the full series,
            including CV test windows — mildly optimistic. Parameters are
            still honestly re-fit per window, so scores vary across windows.

        Args:
            y (jnp.ndarray): Series to score on (concrete; order selection is
                a host-side search).
            X (jnp.ndarray | None, optional): Optional exogenous regressors.

        Returns:
            jnp.ndarray: ``(n_windows, h)`` conformity scores.

        Side Effects:
            First call on an unfitted estimator runs ``fit`` (caches the
            selected order) — mirroring ``forecast``'s first-call behaviour.
        """
        if self._cached_order is None:
            self.fit(y, X)
            # fit() just cached scores on exactly this y — reuse them rather
            # than paying the n_windows CV re-fits a second time.
            if self._cs is not None:
                return self._cs
        return super().conformity_scores(y, X)
    
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
            level (list | None, optional): Confidence levels (0-100) for conformal
                prediction intervals. Requires ``conformal_params``. Default is None.
            fitted (bool, optional): Whether to return fitted values (unused; the
                fast path computes forecasts only). Default is False.

        Returns:
            dict[str, jnp.ndarray]: Forecast dictionary containing `mean`, plus
            `lo-{l}`/`hi-{l}` conformal bounds when `level` is given.

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
        if X is not None or X_future is not None:
            raise ValueError(
                "AutoARIMA does not currently support exogenous regressors "
                "end-to-end; forecast accepts only y."
            )
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # First call: run full search to find best order, then reuse predict()
        fitted_this_call = False
        if self._cached_order is None:
            self.fit(y_jax, X)
            fitted_this_call = True
            res = self.predict(h, X=None)
        else:
            # Subsequent calls: fast path with cached order (CSS-only for
            # speed). This branch is what conformity_scores vmaps: every op
            # below is jnp/lax-native with config-static shapes.
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
                    per_step = m
                else:
                    dx = y_fit[1:] - y_fit[:-1]
                    per_step = 1
                init_params = init_params.at[self._cached_narma + self._cached_n_exog].set(jnp.nanmean(dx) / per_step)

            current_params = _fit_model_scan(
                init_params, y_fit, None, self._cached_delta, _objective_css,
                self._cached_arma, self._cached_ncxreg, self._cached_n_exog, include_mean, 100
            )

            # Closed-form mean/drift pin (same rescue as arima_fit).
            if include_mean:
                _mi = self._cached_narma + self._cached_n_exog
                if use_drift:
                    current_params = current_params.at[_mi].set(jnp.nanmean(dx) / per_step)
                elif (d + D) >= 2:
                    current_params = current_params.at[_mi].set(0.0)

            # Fused forecast: single XLA dispatch for params → forecast
            raw_fc, raw_se = _forecast_from_params(
                current_params, y_fit, self._cached_delta,
                self._cached_arma, self._cached_ncxreg, self._cached_n_exog, include_mean, h
            )

            # raw_fc is on the training-series scale: the state space's
            # differencing rows integrate internally.
            fc = _aa_denormalize(raw_fc, f_mean, f_std)
            res = {"mean": fc}
            res["_se"] = raw_se * (f_std if self.standardize else jnp.array(1.0, dtype=jnp.float64))

        se_arr = res.pop("_se", None)
        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is None:
            if se_arr is None:
                # First call routed through fit()+predict(); reuse predict's
                # analytic interval path directly.
                return self.predict(h, X=None, level=tuple(level))
            z_scores = _quantiles(level)
            for i, lv in enumerate(level):
                res[f"lo-{lv}"] = res["mean"] - z_scores[i] * se_arr
                res[f"hi-{lv}"] = res["mean"] + z_scores[i] * se_arr
            return res
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )
        if fitted_this_call and self._cs is not None:
            # fit() above just cached scores on exactly this y — don't pay
            # the n_windows CV re-fits a second time.
            cs = self._cs
        else:
            cs = self.conformity_scores(y=y_jax, X=X)
        return self.add_confidence_intervals(res, cs, level, self.conformal_params.method)

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
            raise ValueError(
                "AutoARIMA does not currently support exogenous regressors "
                "end-to-end; predict accepts only h."
            )

        # Conformal intervals take over whenever conformal_params is set;
        # analytic z-score intervals remain the fallback. Standard errors are
        # only computed when the analytic path will actually use them.
        use_conformal = level is not None and self.conformal_params is not None
        preds = predict_arima(
            self.model_, n_ahead=h, newxreg=X,
            se_fit=(level is not None and not use_conformal),
        )

        if isinstance(preds, tuple):
            mean_pred, se_pred = preds
        else:
            mean_pred, se_pred = preds, None

        # mean_pred is on the training-series scale (integration lives in the
        # state space).
        mean_orig = _aa_denormalize(mean_pred, self._y_mean, self._y_std)

        # Standard Errors. The state-space (make_arima) already bakes the
        # differencing into T/Z, so kalman_forecast returns the integrated-series
        # forecast SE directly; the old sqrt(cumsum(se^2)) here integrated a
        # SECOND time (double count). se_pred is already the correct per-horizon
        # SE — just rescale to the original units.
        if se_pred is not None:
            se_orig = se_pred * self._y_std
        else:
            se_orig = None

        result = {}
        result["mean"] = mean_orig

        if use_conformal:
            level = sorted([level] if isinstance(level, int) else list(level))
            if self._cs is None:
                raise ValueError(
                    "Conformity scores are not available. Fit the model (fit(...)) "
                    "with `conformal_params` set so predict() can use cached scores."
                )
            if h != self.conformal_params.h:
                raise ValueError(
                    f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                    "conformity scores cover exactly conformal_params.h steps."
                )
            return self.add_confidence_intervals(result, self._cs, level, self.conformal_params.method)

        if level is not None and se_orig is not None:
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
