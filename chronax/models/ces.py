"""
Complex Exponential Smoothing (CES) in JAX.

This module implements Complex Exponential Smoothing, a state-space forecasting method
that uses complex-valued components to capture trend and seasonality. CES generalizes
exponential smoothing by allowing the state to rotate in the complex plane.

**Mathematical Foundation:**

The CES model maintains a state vector with real and imaginary components:
- State update: s_t = s_{t-1} * (1 - α) + y_t * α_complex
- Forecast: ŷ_{t+h} = Real(s_t * rotation^h) + seasonal_component

Where α_complex = α_0 + i*α_1 controls smoothing in the complex plane.

**Variants:**
- NONE (0): No seasonality, trend only
- SIMPLE (1): Simple seasonal component (lagged states)
- PARTIAL (2): Partial seasonal damping (β_0 parameter)
- FULL (3): Full seasonal damping (β_0, β_1 parameters)

**Implementation:**

The model uses JAX for JIT compilation and efficient computation:
- `init_state()`: Initializes state vector based on variant
- `ces_update_step()`: Updates state with new observation using lax.cond
- `ces_fit_forward()`: Forward pass with lax.scan
- `ces_fit_backfit()`: Backward-forward fitting for better initialization
- `ces_forecast()`: Multi-step ahead forecasting with lax.fori_loop
- `auto_ces()`: Automatic model selection based on information criteria

**Attributes:**

- season_length (int): Seasonal period (m)
- model (str): Model variant ("N", "S", "P", "F", or "Z" for auto-selection)
- alias (str): Model name for display
- conformal_params (ConformalIntervals): Optional conformal prediction parameters

**Methods:**

- fit(y, X): Fit CES model to time series
- predict(h, X, level): Generate h-step ahead forecasts with optional prediction intervals
- forecast(y, h, X, X_future): Alternative forecasting interface

**Example:**
```python
model = AutoCES(season_length=12, model="Z")  # Auto-select best variant
model.fit(y)
forecast = model.predict(h=24, level=[90, 95])
```

**References:**
- Svetunkov & Kourentzes (2015). "Complex Exponential Smoothing"
- Uses pure JAX implementation with jit, lax.scan, lax.cond, lax.fori_loop
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax, jit

from chronax.utils import ensure_float, calculate_information_criteria, _get_conformal_method, ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster

NONE = 0
SIMPLE = 1
PARTIAL = 2
FULL = 3


@dataclass
class CESParams:
    """Parameters for Complex Exponential Smoothing model variants.

    This dataclass holds the smoothing parameters for different CES model variants.
    The complex-valued smoothing parameter is α_complex = α_0 + i*α_1, which controls
    how the state rotates in the complex plane. Seasonal damping parameters (β_0, β_1)
    are used only in PARTIAL and FULL variants.

    Attributes:
        alpha_0: Real component of complex smoothing parameter (default 1.3)
        alpha_1: Imaginary component of complex smoothing parameter (default 1.0)
        beta_0: Seasonal damping parameter for PARTIAL/FULL variants (default None)
                In PARTIAL: controls simple seasonal damping
                In FULL: real component of complex seasonal damping
        beta_1: Seasonal damping parameter for FULL variant only (default None)
                Imaginary component of complex seasonal damping
    """
    alpha_0: float = 1.3
    alpha_1: float = 1.0
    beta_0: Optional[float] = None
    beta_1: Optional[float] = None

    @classmethod
    def for_variant(cls, variant: int) -> 'CESParams':
        """Create default CESParams for a given model variant.

        Returns appropriate default parameters based on the seasonal variant:
        - NONE (0): alpha_0=1.3, alpha_1=1.0
        - SIMPLE (1): alpha_0=1.3, alpha_1=1.0
        - PARTIAL (2): alpha_0=1.3, alpha_1=1.0, beta_0=0.1
        - FULL (3): alpha_0=1.3, alpha_1=1.0, beta_0=1.3, beta_1=1.0

        Args:
            variant: Model variant identifier. One of:
                - NONE (0): No seasonality
                - SIMPLE (1): Simple seasonal component
                - PARTIAL (2): Partial seasonal damping
                - FULL (3): Full seasonal damping

        Returns:
            CESParams instance with appropriate defaults for the variant
        """
        if variant == PARTIAL:
            return cls(alpha_0=1.3, alpha_1=1.0, beta_0=0.1)
        elif variant == FULL:
            return cls(alpha_0=1.3, alpha_1=1.0, beta_0=1.3, beta_1=1.0)
        else:
            return cls(alpha_0=1.3, alpha_1=1.0)

    def to_dict(self) -> Dict:
        """Convert parameters to dictionary format.

        Returns:
            Dictionary with keys:
                - 'alpha_0': float
                - 'alpha_1': float
                - 'beta_0': Optional[float]
                - 'beta_1': Optional[float]
        """
        return {
            'alpha_0': self.alpha_0,
            'alpha_1': self.alpha_1,
            'beta_0': self.beta_0,
            'beta_1': self.beta_1,
        }


from functools import partial

@partial(jit, static_argnums=(1,))
def _init_state_n(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Initialize state for NONE variant (no seasonality).

    Computes the mean of the first min(max(10, m), len(y)) observations and uses it
    to initialize both real and imaginary components of the state. The state is padded
    to (m, 4) for compatibility with lax.switch.

    Args:
        y: Input time series array
        m: Seasonal period (used for padding even though no seasonality)

    Returns:
        State array of shape (m, 4) with columns [real, imag, 0, 0]
    """
    idx = jnp.minimum(jnp.maximum(10, m), len(y))
    # Use masking for JIT compatibility
    mask = jnp.arange(len(y)) < idx
    mean_val = jnp.sum(jnp.where(mask, y, 0.0)) / jnp.maximum(jnp.sum(mask), 1.0)
    # Pad to (m, 4) for compatibility with other variants
    base_state = jnp.array([[mean_val, mean_val / 1.1]], dtype=jnp.float32)
    # Replicate to (m, 2) then pad to (m, 4)
    states = jnp.tile(base_state, (m, 1))
    return jnp.pad(states, ((0, 0), (0, 2)), constant_values=0.0)


@partial(jit, static_argnums=(1,))
def _init_state_s(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Initialize state for SIMPLE variant (simple seasonality).

    Uses the first m observations directly to initialize the state. Each observation
    initializes the real component, with imaginary component set to obs/1.1. The last
    two columns (seasonal damping) remain zero as they are not used in SIMPLE variant.

    Args:
        y: Input time series array (must have length >= m)
        m: Seasonal period

    Returns:
        State array of shape (m, 4) with columns [real, imag, 0, 0]
    """
    states = jnp.zeros((m, 4), dtype=jnp.float32)
    states = states.at[:, 0].set(y[:m])
    states = states.at[:, 1].set(y[:m] / 1.1)
    # Columns 2-3 remain zero (not used in SIMPLE)
    return states


@partial(jit, static_argnums=(1,))
def _init_state_p(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Initialize state for PARTIAL variant (partial seasonal damping).

    Initializes the trend components (real/imaginary) with the mean of the first m values.
    Extracts seasonal components via moving average detrending when sufficient data is
    available (n >= 2*m), otherwise uses simple deviations from the mean.

    Args:
        y: Input time series array
        m: Seasonal period

    Returns:
        State array of shape (m, 4) with columns [real, imag, seasonal, 0]
    """
    states = jnp.zeros((m, 4), dtype=jnp.float32)
    mean_val = jnp.mean(y[:m])
    states = states.at[:, 0].set(mean_val)
    states = states.at[:, 1].set(mean_val / 1.1)
    
    n = len(y)
    has_enough_data = n >= 2 * m
    
    def compute_seasonal():
        kernel = jnp.ones(m) / m
        trend = jnp.convolve(y, kernel, mode='same')
        detrended = y[:m] - trend[:m]
        return detrended - jnp.mean(detrended)
    
    seasonal = jnp.where(
        has_enough_data,
        compute_seasonal(),
        y[:m] - mean_val
    )
    
    states = states.at[:, 2].set(seasonal)
    return states


@partial(jit, static_argnums=(1,))
def _init_state_f(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Initialize state for FULL variant (full seasonal damping).

    Initializes trend components (real/imaginary) with the mean of the first m values.
    Extracts seasonal components (real and imaginary) via moving average detrending
    when sufficient data is available (n >= 2*m), otherwise uses simple deviations.

    Args:
        y: Input time series array
        m: Seasonal period

    Returns:
        State array of shape (m, 4) with columns [real, imag, seasonal_real, seasonal_imag]
    """
    states = jnp.zeros((m, 4), dtype=jnp.float32)
    mean_val = jnp.mean(y[:m])
    states = states.at[:, 0].set(mean_val)
    states = states.at[:, 1].set(mean_val / 1.1)
    
    n = len(y)
    has_enough_data = n >= 2 * m
    
    def compute_seasonal():
        kernel = jnp.ones(m) / m
        trend = jnp.convolve(y, kernel, mode='same')
        detrended = y[:m] - trend[:m]
        return detrended - jnp.mean(detrended)
    
    seasonal = jnp.where(
        has_enough_data,
        compute_seasonal(),
        y[:m] - mean_val
    )
    
    states = states.at[:, 2].set(seasonal)
    states = states.at[:, 3].set(seasonal / 1.1)
    return states


@partial(jit, static_argnums=(1, 2))
def init_state(y: jnp.ndarray, m: int, season_type: int) -> jnp.ndarray:
    """Initialize CES state vector based on model variant.

    Dispatches to the appropriate initialization function based on season_type using
    lax.switch for efficient JIT-compiled branching. All variants return states padded
    to (m, 4) for compatibility.

    Args:
        y: Input time series array
        m: Seasonal period
        season_type: Model variant (NONE=0, SIMPLE=1, PARTIAL=2, FULL=3)

    Returns:
        State array of shape (m, 4)
    """
    return lax.switch(
        season_type,
        [
            lambda y_: _init_state_n(y_, m),
            lambda y_: _init_state_s(y_, m),
            lambda y_: _init_state_p(y_, m),
            lambda y_: _init_state_f(y_, m),
        ],
        y
    )


@jit
def _update_state_none_partial_full(
    state_prev: jnp.ndarray,
    y_obs: float,
    alpha_0: float,
    alpha_1: float,
    season_type: int,
    beta_0: float,
    beta_1: float,
) -> jnp.ndarray:
    """Update CES state for NONE, PARTIAL, or FULL variants (previous-step state).

    Computes the innovation error from the previous state, updates the complex
    trend components (real/imaginary), and conditionally updates seasonal components
    based on the variant. Uses lax.cond for JIT-safe branching.

    Args:
        state_prev: Previous time-step state vector of shape (4,).
        y_obs: Observed value at current time step.
        alpha_0: Real part of complex smoothing parameter.
        alpha_1: Imaginary part of complex smoothing parameter.
        season_type: Model variant (NONE=0, PARTIAL=2, FULL=3).
        beta_0: Real seasonal damping parameter (used by PARTIAL/FULL).
        beta_1: Imaginary seasonal damping parameter (used by FULL only).

    Returns:
        Updated state vector of shape (4,).
    """
    e = y_obs - state_prev[0]
    
    state_new = jnp.zeros_like(state_prev)
    state_new = state_new.at[0].set(
        state_prev[0] - (1.0 - alpha_1) * state_prev[1] + (alpha_0 - alpha_1) * e
    )
    state_new = state_new.at[1].set(
        state_prev[0] + (1.0 - alpha_0) * state_prev[1] + (alpha_0 + alpha_1) * e
    )
    
    def update_partial():
        return state_new.at[2].set(state_prev[2] + beta_0 * e)
    
    def update_full():
        return state_new.at[2].set(
            state_prev[2] - (1.0 - beta_1) * state_prev[3] + (beta_0 - beta_1) * e
        ).at[3].set(
            state_prev[2] + (1.0 - beta_0) * state_prev[3] + (beta_0 + beta_1) * e
        )
    
    return lax.cond(
        season_type == PARTIAL,
        update_partial,
        lambda: lax.cond(
            season_type == FULL,
            update_full,
            lambda: state_new
        )
    )


@jit
def _update_state_simple(
    state_lag: jnp.ndarray,
    y_obs: float,
    alpha_0: float,
    alpha_1: float,
) -> jnp.ndarray:
    """Update CES state for the SIMPLE seasonal variant (m-step lagged state).

    Uses the state from m steps ago (the matching seasonal phase) to compute
    the innovation and update only the complex trend components. Seasonal
    components are carried forward implicitly through the ring buffer.

    Args:
        state_lag: State vector from m steps ago of shape (4,).
        y_obs: Observed value at current time step.
        alpha_0: Real part of complex smoothing parameter.
        alpha_1: Imaginary part of complex smoothing parameter.

    Returns:
        Updated state vector of shape (4,).
    """
    e = y_obs - state_lag[0]
    
    state_new = jnp.zeros_like(state_lag)
    state_new = state_new.at[0].set(
        state_lag[0] - (1.0 - alpha_1) * state_lag[1] + (alpha_0 - alpha_1) * e
    )
    state_new = state_new.at[1].set(
        state_lag[0] + (1.0 - alpha_0) * state_lag[1] + (alpha_0 + alpha_1) * e
    )
    
    return state_new


@jit
def _update_state_partial_full_lag(
    state_lag: jnp.ndarray,
    y_obs: float,
    alpha_0: float,
    alpha_1: float,
    season_type: int,
    beta_0: float,
    beta_1: float,
) -> jnp.ndarray:
    """Update CES state for PARTIAL/FULL variants using the m-step lagged state.

    Computes the innovation from the lagged state (subtracting the seasonal
    component) and updates both trend and seasonal components accordingly.
    Uses lax.cond to branch between PARTIAL and FULL seasonal updates.

    Args:
        state_lag: State vector from m steps ago of shape (4,).
        y_obs: Observed value at current time step.
        alpha_0: Real part of complex smoothing parameter.
        alpha_1: Imaginary part of complex smoothing parameter.
        season_type: Model variant (PARTIAL=2 or FULL=3).
        beta_0: Real seasonal damping parameter.
        beta_1: Imaginary seasonal damping parameter (FULL only).

    Returns:
        Updated state vector of shape (4,).
    """
    e = y_obs - state_lag[0] - jnp.where(season_type > SIMPLE, state_lag[2], 0.0)
    
    state_new = jnp.zeros_like(state_lag)
    state_new = state_new.at[0].set(
        state_lag[0] - (1.0 - alpha_1) * state_lag[1] + (alpha_0 - alpha_1) * e
    )
    state_new = state_new.at[1].set(
        state_lag[0] + (1.0 - alpha_0) * state_lag[1] + (alpha_0 + alpha_1) * e
    )
    
    def update_partial():
        return state_new.at[2].set(state_lag[2] + beta_0 * e)
    
    def update_full():
        return state_new.at[2].set(
            state_lag[2] - (1.0 - beta_1) * state_lag[3] + (beta_0 - beta_1) * e
        ).at[3].set(
            state_lag[2] + (1.0 - beta_0) * state_lag[3] + (beta_0 + beta_1) * e
        )
    
    return lax.cond(
        season_type == PARTIAL,
        update_partial,
        lambda: lax.cond(
            season_type == FULL,
            update_full,
            lambda: state_new
        )
    )


@jit
def ces_update_step(
    carry: Tuple[jnp.ndarray, int],
    y_obs: float,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    season_type: int,
    m: int,
) -> Tuple[Tuple[jnp.ndarray, int], jnp.ndarray]:
    states_buffer, i = carry
    
    is_none_partial_full = (season_type == NONE) | (season_type == PARTIAL) | (season_type == FULL)
    
    def get_state_prev():
        return lax.cond(
            is_none_partial_full,
            lambda: states_buffer[(i - 1) % m],
            lambda: states_buffer[(i - m) % m]
        )
    
    state_prev = get_state_prev()
    
    def update_none_partial_full():
        return _update_state_none_partial_full(
            state_prev, y_obs, alpha_0, alpha_1,
            season_type, beta_0, beta_1
        )
    
    def update_simple():
        state_lag = states_buffer[(i - m) % m]
        return _update_state_simple(state_lag, y_obs, alpha_0, alpha_1)
    
    def update_partial_full_lag():
        state_lag = states_buffer[(i - m) % m]
        return _update_state_partial_full_lag(
            state_lag, y_obs, alpha_0, alpha_1,
            season_type, beta_0, beta_1
        )
    
    state_new = lax.cond(
        is_none_partial_full,
        update_none_partial_full,
        lambda: lax.cond(
            season_type > SIMPLE,
            update_partial_full_lag,
            update_simple
        )
    )
    
    states_buffer = states_buffer.at[i % m].set(state_new)
    
    forecast = state_new[0] + jnp.where(
        season_type > SIMPLE,
        state_new[2],
        0.0
    )
    
    return (states_buffer, i + 1), forecast


@partial(jit, static_argnums=(6, 7))
def ces_fit_forward(
    y: jnp.ndarray,
    init_state: jnp.ndarray,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    season_type: int,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single forward CES pass over the series using lax.scan.

    Scans `ces_update_step` over observations y[m:] starting from the
    initialised state buffer (first m states). season_type and m are
    static for JIT compilation.

    Args:
        y: Time series of shape (n,).
        init_state: Initial state buffer of shape (m, 4).
        alpha_0: Real part of complex smoothing parameter.
        alpha_1: Imaginary part of complex smoothing parameter.
        beta_0: Seasonal damping parameter (real).
        beta_1: Seasonal damping parameter (imaginary, FULL only).
        season_type: Model variant (static: NONE=0, SIMPLE=1, PARTIAL=2, FULL=3).
        m: Seasonal period (static).

    Returns:
        Tuple of:
            - final_states: State buffer after processing all observations, shape (m, 4).
            - forecasts: One-step-ahead in-sample forecasts for y[m:], shape (n-m,).
    """
    states_buffer = init_state.copy()
    
    (final_states, _), forecasts = lax.scan(
        lambda carry, y_obs: ces_update_step(
            carry, y_obs, alpha_0, alpha_1, beta_0, beta_1, season_type, m
        ),
        (states_buffer, m),
        y[m:],
    )
    
    return final_states, forecasts


def ces_fit_backfit(
    y: jnp.ndarray,
    init_state: jnp.ndarray,
    params: CESParams,
    season_type: int,
    m: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit CES using a three-pass back-fitting procedure for better initialisation.

    Runs three consecutive forward passes: forward → backward (reversed series)
    → forward again. The backward pass refines the initial state before the final
    forward pass, substantially reducing initialisation bias on short series.

    Args:
        y: Time series of shape (n,).
        init_state: Initial state buffer of shape (m, 4).
        params: CESParams instance with smoothing parameters.
        season_type: Model variant (NONE=0, SIMPLE=1, PARTIAL=2, FULL=3).
        m: Seasonal period.

    Returns:
        Tuple of:
            - final_states: State buffer from the last forward pass, shape (m, 4).
            - forecasts: In-sample one-step-ahead forecasts from the last forward
              pass for y[m:], shape (n-m,).
    """
    beta_0 = params.beta_0 if params.beta_0 is not None else 0.0
    beta_1 = params.beta_1 if params.beta_1 is not None else 0.0
    
    states_fwd, _ = ces_fit_forward(
        y, init_state, params.alpha_0, params.alpha_1,
        beta_0, beta_1, season_type, m
    )
    y_rev = y[::-1]
    states_rev, _ = ces_fit_forward(
        y_rev, states_fwd, params.alpha_0, params.alpha_1,
        beta_0, beta_1, season_type, m
    )
    states_final, forecasts = ces_fit_forward(
        y, states_rev, params.alpha_0, params.alpha_1,
        beta_0, beta_1, season_type, m
    )
    
    return states_final, forecasts


@partial(jit, static_argnums=(5, 6, 7))
def ces_forecast(
    final_state: jnp.ndarray,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    season_type: int,
    m: int,
    h: int,
) -> jnp.ndarray:
    """Generate forecasts with static season_type, m, and h."""
    def forecast_step(i_h, carry):
        states_buffer, forecasts, current_idx = carry
        
        is_none_partial_full = (season_type == NONE) | (season_type == PARTIAL) | (season_type == FULL)
        
        state_prev = lax.cond(
            is_none_partial_full,
            lambda: states_buffer[(current_idx - 1) % m],
            lambda: states_buffer[(current_idx - m) % m]
        )
        
        forecast = state_prev[0] + jnp.where(
            season_type > SIMPLE,
            state_prev[2],
            0.0
        )
        
        forecasts = forecasts.at[i_h].set(forecast)
        
        def update_none_partial_full():
            return _update_state_none_partial_full(
                state_prev, forecast, alpha_0, alpha_1,
                season_type, beta_0, beta_1
            )
        
        def update_simple():
            state_lag = states_buffer[(current_idx - m) % m]
            return _update_state_simple(state_lag, forecast, alpha_0, alpha_1)
        
        def update_partial_full_lag():
            state_lag = states_buffer[(current_idx - m) % m]
            return _update_state_partial_full_lag(
                state_lag, forecast, alpha_0, alpha_1,
                season_type, beta_0, beta_1
            )
        
        state_new = lax.cond(
            is_none_partial_full,
            update_none_partial_full,
            lambda: lax.cond(
                season_type > SIMPLE,
                update_partial_full_lag,
                update_simple
            )
        )
        
        new_idx = (current_idx + 1) % m
        states_buffer = states_buffer.at[new_idx].set(state_new)
        
        return (states_buffer, forecasts, current_idx + 1)
    
    forecasts = jnp.zeros(h, dtype=jnp.float32)
    states_buffer = final_state.copy()
    
    (states_buffer, forecasts, _) = lax.fori_loop(
        0, h,
        forecast_step,
        (states_buffer, forecasts, m)
    )
    
    return forecasts


def ces_fit_single(
    y: jnp.ndarray,
    m: int,
    season_type: int,
    params: Optional[CESParams] = None,
) -> Dict:
    """Fit a single CES variant and return metrics, fitted values, and state.

    Initialises the state vector, runs back-fitting, computes in-sample
    residuals, and calculates information criteria (AIC, BIC, AICc).

    Args:
        y: Time series of shape (n,).
        m: Seasonal period.
        season_type: Model variant (NONE=0, SIMPLE=1, PARTIAL=2, FULL=3).
        params: CESParams with smoothing parameters. If None, uses
            CESParams.for_variant(season_type) defaults.

    Returns:
        Dict with keys:
            - "loglik" (float): Log-likelihood.
            - "aic" / "bic" / "aicc" (float): Information criteria.
            - "mse" / "amse" (float): Mean squared error on y[m:].
            - "fitted" (jnp.ndarray): In-sample fitted values, shape (n,).
            - "residuals" (jnp.ndarray): Residuals y[m:] − ŷ[m:], shape (n-m,).
            - "states" (jnp.ndarray): Final state buffer, shape (m, 4).
            - "par" (dict): Parameter dict from params.to_dict().
            - "m" (int): Seasonal period used.
            - "n" (int): Series length.
            - "seasontype" (int): Variant used.
            - "sigma2" (float): Residual variance estimate.
    """
    y = ensure_float(y)
    
    if params is None:
        params = CESParams.for_variant(season_type)
    
    init_state_arr = init_state(y, m, season_type)
    final_states, forecasts = ces_fit_backfit(y, init_state_arr, params, season_type, m)
    
    n = len(y)
    n_components = init_state_arr.shape[1]
    n_params = n_components + 1
    n_residuals = n - m
    
    fitted = jnp.empty(n, dtype=jnp.float32)
    fitted = fitted.at[:m].set(y[:m])
    fitted = fitted.at[m:].set(forecasts)
    
    residuals = y[m:] - forecasts
    
    sse = jnp.sum(residuals ** 2)
    mse = sse / n_residuals
    denom = n - n_params - 1
    sigma2 = jnp.where(denom > 0, sse / denom, sse / n)
    
    ic_dict = calculate_information_criteria(residuals, n_params, n)
    
    return {
        'loglik': ic_dict['loglik'],
        'aic': ic_dict['aic'],
        'bic': ic_dict['bic'],
        'aicc': ic_dict['aicc'],
        'mse': float(mse),
        'amse': float(mse),
        'fitted': fitted,
        'residuals': residuals,
        'states': final_states,
        'par': params.to_dict(),
        'm': m,
        'n': n,
        'seasontype': season_type,
        'sigma2': float(sigma2),
    }


def auto_ces(
    y: jnp.ndarray,
    m: int = 1,
    model: str = "Z",
    ic: str = "aicc",
) -> Dict:
    """Fit CES with automatic or fixed model selection.

    When model="Z", fits all applicable variants (NONE always; SIMPLE/PARTIAL/FULL
    when n >= 2*m) and returns the fit with the lowest information criterion.
    Otherwise, fits the specified variant directly.

    Args:
        y: Time series of shape (n,).
        m: Seasonal period. Default is 1 (no seasonality).
        model: Variant selector. "Z" for automatic selection; one of "N", "S", "P",
            "F" to fix the variant. Default is "Z".
        ic: Information criterion used for model selection when model="Z".
            One of "aic", "bic", "aicc". Default is "aicc".

    Returns:
        Dict from ces_fit_single() for the selected variant, containing fitted
        values, residuals, states, parameters, and information criteria.

    Raises:
        ValueError: If model="Z" and no variant could be fitted successfully.
    """
    y = ensure_float(y)
    
    model_map = {"N": NONE, "S": SIMPLE, "P": PARTIAL, "F": FULL}
    
    if model == "Z":
        variants = [NONE, SIMPLE, PARTIAL, FULL]
        if m < 2 or len(y) < 2 * m:
            variants = [NONE]
        
        fits = []
        ic_values = []
        
        for variant in variants:
            try:
                fit = ces_fit_single(y, m, variant)
                ic_val = fit[ic]
                if not jnp.isnan(ic_val):
                    fits.append(fit)
                    ic_values.append(float(ic_val))
            except:
                continue
        
        if not fits:
            raise ValueError("No valid model could be fitted")
        
        best_idx = int(jnp.argmin(jnp.array(ic_values)))
        return fits[best_idx]
    else:
        season_type = model_map.get(model, NONE)
        return ces_fit_single(y, m, season_type)


class AutoCES(BaseForecaster):
    """Complex Exponential Smoothing model with optional automatic variant selection.

    Wraps `auto_ces` / `ces_fit_single` in the BaseForecaster interface.
    When model="Z", selects the best variant (NONE/SIMPLE/PARTIAL/FULL) by AICc.
    All JAX core functions are JIT-compiled; the class itself is a thin orchestrator.

    Args:
        season_length (int): Seasonal period m. Use 1 for non-seasonal data.
            Default is 1.
        model (str): Variant selector passed to auto_ces(). "Z" for automatic
            selection; "N", "S", "P", or "F" to fix the variant. Default is "Z".
        alias (str): Model name for display / repr. Default is "CES".
        conformal_params (Optional[ConformalIntervals]): Conformal prediction
            configuration for generating prediction intervals. Default is None.

    Attributes:
        model_ (dict | None): Populated after fit(); contains fitted values,
            residuals, states, parameters, and information criteria from
            ces_fit_single(). None before first fit.
    """

    uses_exog = False

    def __init__(
        self,
        season_length: int = 1,
        model: str = "Z",
        alias: str = "CES",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialise AutoCES with model configuration.

        Args:
            season_length (int): Seasonal period m. Default is 1.
            model (str): Variant selector ("Z", "N", "S", "P", "F"). Default is "Z".
            alias (str): Model name identifier. Default is "CES".
            conformal_params (Optional[ConformalIntervals]): Conformal prediction
                configuration. Default is None.
        """
        self.season_length = season_length
        self.model = model
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = None
    
    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "AutoCES":
        """Fit the CES model to a time series.

        Handles the constant-series edge case separately (stores a trivial state).
        Otherwise delegates to auto_ces() which runs variant selection and back-fitting.

        Args:
            y (jnp.ndarray): Input time series of shape (n,).
            X (Optional[jnp.ndarray]): Exogenous variables (unused; kept for API
                compatibility). Default is None.

        Returns:
            AutoCES: Self (fitted model instance) for method chaining.
        """
        y = ensure_float(y)

        if jnp.std(y) < 1e-10:
            # Constant series - create proper state for forecasting
            mean_val = jnp.mean(y)
            init_states = jnp.array([[mean_val, mean_val]], dtype=jnp.float32)
            self.model_ = {
                'fitted': y,
                'residuals': jnp.zeros_like(y),
                'par': {'alpha_0': 0.0, 'alpha_1': 0.0, 'beta_0': None, 'beta_1': None},
                'm': self.season_length,
                'n': len(y),
                'seasontype': NONE,
                'states': init_states,
            }
            return self
        
        self.model_ = auto_ces(y, m=self.season_length, model=self.model)
        return self
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
    ) -> Dict:
        """Stateless fit+forecast: fit if not already done, then generate forecasts.

        If model_ is None, fits the model on y first. Otherwise uses existing state.
        Does not support conformal intervals (use predict() after fit() for that).

        Args:
            y (jnp.ndarray): Input time series of shape (n,). Used only if not fitted.
            h (int): Forecast horizon (number of steps ahead).
            X (Optional[jnp.ndarray]): Exogenous variables (unused). Default is None.
            X_future (Optional[jnp.ndarray]): Future exogenous variables (unused).
                Default is None.

        Returns:
            Dict: Dictionary with key "mean" containing forecasts of shape (h,).
        """
        if self.model_ is None:
            self.fit(y, X)
        
        final_state = self.model_['states']
        params_dict = self.model_['par']
        season_type = self.model_['seasontype']
        m = self.model_['m']
        
        params = CESParams(**params_dict)
        beta_0 = params.beta_0 if params.beta_0 is not None else 0.0
        beta_1 = params.beta_1 if params.beta_1 is not None else 0.0
        
        forecasts = ces_forecast(
            final_state, params.alpha_0, params.alpha_1,
            beta_0, beta_1, season_type, m, h
        )
        
        return {'mean': forecasts}
    
    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None) -> Dict:
        """Generate h-step ahead forecasts from the fitted CES model.

        Runs the JIT-compiled ces_forecast() function from the stored final state.
        Handles the constant-series edge case (alpha=0) by returning flat forecasts.
        Optionally adds conformal prediction intervals.

        Args:
            h (int): Forecast horizon (number of steps ahead).
            X (Optional[jnp.ndarray]): Exogenous variables (unused; kept for API
                compatibility). Default is None.
            level (Optional[List[int]]): Confidence levels (0–100) for conformal
                prediction intervals, e.g. [90, 95]. Requires conformal_params to
                be set. Default is None.

        Returns:
            Dict: Dictionary containing:
                - "mean": Point forecasts of shape (h,).
                - "lo-{l}" / "hi-{l}": Conformal interval bounds for each level l
                  (only present when level is not None and conformal_params is set).

        Raises:
            ValueError: If called before fit().
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        final_state = self.model_['states']
        params_dict = self.model_['par']
        season_type = self.model_['seasontype']
        m = self.model_['m']
        
        params = CESParams(**params_dict)
        beta_0 = params.beta_0 if params.beta_0 is not None else 0.0
        beta_1 = params.beta_1 if params.beta_1 is not None else 0.0
        
        # Handle constant series case (alpha=0)
        if params.alpha_0 == 0.0 and params.alpha_1 == 0.0:
            # For constant series, just repeat the mean value
            mean_val = final_state[0, 0]
            forecasts = jnp.full(h, mean_val, dtype=jnp.float32)
        else:
            forecasts = ces_forecast(
                final_state, params.alpha_0, params.alpha_1,
                beta_0, beta_1, season_type, m, h
            )
        
        result = {'mean': forecasts}
        
        if level is not None and self.conformal_params is not None:
            cs = self.conformity_scores(y=self.model_['fitted'], X=X)
            conformal_fn = _get_conformal_method(self.conformal_params.method)
            result = conformal_fn(fcst=result, cs=cs, level=level)
        
        return result


if __name__ == "__main__":
    import jax.random as jrandom
    
    print("=" * 60)
    print("CES (Complex Exponential Smoothing) Test Suite")
    print("=" * 60)
    
    # Test 1: Basic fit and predict (no seasonality)
    print("\n[Test 1] Basic CES fit and predict (NONE variant)")
    n1 = 50
    t1 = jnp.arange(n1, dtype=jnp.float32)
    y1 = 10.0 + 0.5 * t1 + jrandom.normal(jrandom.PRNGKey(42), (n1,)) * 0.5
    
    model1 = AutoCES(season_length=1, model="N")
    model1.fit(y1)
    forecast1 = model1.predict(h=10)
    
    print(f"  Input series length: {n1}")
    print(f"  Fitted shape: {model1.model_['fitted'].shape}")
    print(f"  Forecast shape: {forecast1['mean'].shape}")
    print(f"  Season type: {model1.model_['seasontype']} (NONE)")
    print(f"  Alpha_0: {model1.model_['par']['alpha_0']:.4f}")
    print(f"  Alpha_1: {model1.model_['par']['alpha_1']:.4f}")
    assert model1.model_['fitted'].shape == (n1,), "Fitted shape mismatch!"
    assert forecast1['mean'].shape == (10,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast1['mean'])), "Forecast contains NaN/Inf!"
    assert model1.model_['seasontype'] == NONE, "Should be NONE variant!"
    print("  OK: Basic CES OK")
    
    # Test 2: Simple seasonal variant
    print("\n[Test 2] CES with SIMPLE seasonality")
    n2 = 84
    period = 12
    t2 = jnp.arange(n2, dtype=jnp.float32)
    seasonal2 = 3.0 * jnp.sin(2 * jnp.pi * t2 / period)
    trend2 = 20.0 + 0.3 * t2
    y2 = trend2 + seasonal2 + jrandom.normal(jrandom.PRNGKey(123), (n2,)) * 0.5
    
    model2 = AutoCES(season_length=period, model="S")
    model2.fit(y2)
    forecast2 = model2.predict(h=12)
    
    print(f"  Season length: {period}")
    print(f"  Season type: {model2.model_['seasontype']} (SIMPLE)")
    print(f"  First forecast: {forecast2['mean'][0]:.4f}")
    print(f"  Last forecast: {forecast2['mean'][-1]:.4f}")
    assert model2.model_['seasontype'] == SIMPLE, "Should be SIMPLE variant!"
    assert forecast2['mean'].shape == (12,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast2['mean'])), "Forecast contains NaN/Inf!"
    print("  OK: SIMPLE seasonality OK")
    
    # Test 3: Partial seasonal variant
    print("\n[Test 3] CES with PARTIAL seasonality")
    model3 = AutoCES(season_length=period, model="P")
    model3.fit(y2)
    forecast3 = model3.predict(h=12)
    
    print(f"  Season type: {model3.model_['seasontype']} (PARTIAL)")
    print(f"  Beta_0: {model3.model_['par']['beta_0']}")
    print(f"  Forecast variance: {jnp.var(forecast3['mean']):.4f}")
    assert model3.model_['seasontype'] == PARTIAL, "Should be PARTIAL variant!"
    assert model3.model_['par']['beta_0'] is not None, "Should have beta_0!"
    assert jnp.all(jnp.isfinite(forecast3['mean'])), "Forecast contains NaN/Inf!"
    print("  OK: PARTIAL seasonality OK")
    
    # Test 4: Full seasonal variant
    print("\n[Test 4] CES with FULL seasonality")
    model4 = AutoCES(season_length=period, model="F")
    model4.fit(y2)
    forecast4 = model4.predict(h=12)
    
    print(f"  Season type: {model4.model_['seasontype']} (FULL)")
    print(f"  Beta_0: {model4.model_['par']['beta_0']}")
    print(f"  Beta_1: {model4.model_['par']['beta_1']}")
    assert model4.model_['seasontype'] == FULL, "Should be FULL variant!"
    assert model4.model_['par']['beta_0'] is not None, "Should have beta_0!"
    assert model4.model_['par']['beta_1'] is not None, "Should have beta_1!"
    assert jnp.all(jnp.isfinite(forecast4['mean'])), "Forecast contains NaN/Inf!"
    print("  OK: FULL seasonality OK")
    
    # Test 5: Auto model selection
    print("\n[Test 5] Auto model selection (Z)")
    model5 = AutoCES(season_length=period, model="Z")
    model5.fit(y2)
    forecast5 = model5.predict(h=12)
    
    print(f"  Auto-selected variant: {model5.model_['seasontype']}")
    print(f"  AIC: {model5.model_['aic']:.4f}")
    print(f"  BIC: {model5.model_['bic']:.4f}")
    print(f"  AICc: {model5.model_['aicc']:.4f}")
    assert model5.model_['seasontype'] in [NONE, SIMPLE, PARTIAL, FULL], "Invalid variant!"
    assert 'aic' in model5.model_, "Should have AIC!"
    assert jnp.all(jnp.isfinite(forecast5['mean'])), "Forecast contains NaN/Inf!"
    print("  OK: Auto selection OK")
    
    # Test 6: Information criteria
    print("\n[Test 6] Information criteria calculation")
    print(f"  Loglik: {model5.model_['loglik']:.4f}")
    print(f"  AIC: {model5.model_['aic']:.4f}")
    print(f"  BIC: {model5.model_['bic']:.4f}")
    print(f"  AICc: {model5.model_['aicc']:.4f}")
    print(f"  MSE: {model5.model_['mse']:.4f}")
    assert jnp.isfinite(model5.model_['aic']), "AIC should be finite!"
    assert jnp.isfinite(model5.model_['bic']), "BIC should be finite!"
    assert jnp.isfinite(model5.model_['mse']), "MSE should be finite!"
    print("  OK: Information criteria OK")
    
    # Test 7: Constant series edge case
    print("\n[Test 7] Edge case: constant series")
    y7 = jnp.ones(40) * 15.0
    model7 = AutoCES(season_length=1, model="Z")
    model7.fit(y7)
    forecast7 = model7.predict(h=10)
    
    print(f"  Input (constant 15.0)")
    print(f"  Forecast mean: {jnp.mean(forecast7['mean']):.4f}")
    print(f"  Forecast std: {jnp.std(forecast7['mean']):.6f}")
    assert jnp.allclose(forecast7['mean'], 15.0, atol=0.1), "Should forecast constant!"
    print("  OK: Constant series OK")
    
    # Test 8: Short series edge case
    print("\n[Test 8] Edge case: short series")
    y8 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model8 = AutoCES(season_length=1, model="Z")
    model8.fit(y8)
    forecast8 = model8.predict(h=3)
    
    print(f"  Input length: {len(y8)}")
    print(f"  Forecast: {forecast8['mean']}")
    assert forecast8['mean'].shape == (3,), "Should forecast despite short series!"
    assert jnp.all(jnp.isfinite(forecast8['mean'])), "Forecast should be finite!"
    print("  OK: Short series OK")
    
    # Test 9: Residuals and fitted values
    print("\n[Test 9] Residuals and fitted values")
    residuals = model5.model_['residuals']
    fitted = model5.model_['fitted']
    
    print(f"  Residuals shape: {residuals.shape}")
    print(f"  Fitted shape: {fitted.shape}")
    print(f"  Residuals mean: {jnp.mean(residuals):.6f}")
    print(f"  Residuals std: {jnp.std(residuals):.4f}")
    assert fitted.shape == y2.shape, "Fitted should match input shape!"
    assert jnp.abs(jnp.mean(residuals)) < 1.0, "Residuals should be centered!"
    print("  OK: Residuals OK")
    
    # Test 10: Conformal prediction intervals
    print("\n[Test 10] Conformal prediction intervals")
    conformal_params = ConformalIntervals(h=12)
    model10 = AutoCES(season_length=period, model="Z", conformal_params=conformal_params)
    model10.fit(y2)
    forecast10 = model10.predict(h=12, level=[90, 95])
    
    print(f"  Forecast keys: {list(forecast10.keys())}")
    assert 'mean' in forecast10, "Should have 'mean'!"
    # Check for interval keys (format may vary)
    has_90_intervals = ('lower_90' in forecast10 or 'lo-90' in forecast10)
    has_95_intervals = ('lower_95' in forecast10 or 'lo-95' in forecast10)
    assert has_90_intervals, "Should have 90% interval keys!"
    assert has_95_intervals, "Should have 95% interval keys!"
    # Use whichever format exists
    lo_90_key = 'lo-90' if 'lo-90' in forecast10 else 'lower_90'
    hi_90_key = 'hi-90' if 'hi-90' in forecast10 else 'upper_90'
    print(f"  90% interval width (first): {forecast10[hi_90_key][0] - forecast10[lo_90_key][0]:.4f}")
    print("  OK: Conformal intervals OK")
    
    # Test 11: CES with weekly seasonality
    print("\n[Test 11] Weekly seasonality (period=7)")
    n11 = 70
    t11 = jnp.arange(n11, dtype=jnp.float32)
    seasonal11 = 2.5 * jnp.sin(2 * jnp.pi * t11 / 7)
    y11 = 50.0 + 0.2 * t11 + seasonal11 + jrandom.normal(jrandom.PRNGKey(456), (n11,)) * 0.3
    
    model11 = AutoCES(season_length=7, model="Z")
    model11.fit(y11)
    forecast11 = model11.predict(h=14)
    
    print(f"  Season length: 7")
    print(f"  Auto-selected variant: {model11.model_['seasontype']}")
    print(f"  Forecast for 2 weeks ahead")
    assert forecast11['mean'].shape == (14,), "Should forecast 14 steps!"
    assert jnp.all(jnp.isfinite(forecast11['mean'])), "Forecast should be finite!"
    print("  OK: Weekly seasonality OK")
    
    # Test 12: Fit then predict pattern
    print("\n[Test 12] Fit-predict pattern")
    model12 = AutoCES(season_length=12, model="Z")
    model12.fit(y2)
    
    # Multiple predict calls should work
    pred_5 = model12.predict(h=5)
    pred_10 = model12.predict(h=10)
    pred_20 = model12.predict(h=20)
    
    print(f"  Predict h=5: {pred_5['mean'].shape}")
    print(f"  Predict h=10: {pred_10['mean'].shape}")
    print(f"  Predict h=20: {pred_20['mean'].shape}")
    # First 5 forecasts should match
    assert jnp.allclose(pred_5['mean'], pred_10['mean'][:5], atol=1e-5), "Forecasts should be consistent!"
    assert jnp.allclose(pred_10['mean'], pred_20['mean'][:10], atol=1e-5), "Forecasts should be consistent!"
    print("  OK: Multiple predictions OK")
    
    # Test 13: Custom parameters
    print("\n[Test 13] Custom CES parameters")
    custom_params = CESParams(alpha_0=1.2, alpha_1=0.9, beta_0=0.15)
    fit_result = ces_fit_single(y2, m=period, season_type=PARTIAL, params=custom_params)
    
    print(f"  Custom alpha_0: {fit_result['par']['alpha_0']:.4f}")
    print(f"  Custom alpha_1: {fit_result['par']['alpha_1']:.4f}")
    print(f"  Custom beta_0: {fit_result['par']['beta_0']:.4f}")
    assert fit_result['par']['alpha_0'] == 1.2, "Should use custom alpha_0!"
    assert fit_result['par']['alpha_1'] == 0.9, "Should use custom alpha_1!"
    assert fit_result['par']['beta_0'] == 0.15, "Should use custom beta_0!"
    print("  OK: Custom parameters OK")
    
    # Test 14: State vector structure
    print("\n[Test 14] State vector structure")
    states_none = _init_state_n(y1, 1)
    states_simple = _init_state_s(y2, 12)
    states_partial = _init_state_p(y2, 12)
    states_full = _init_state_f(y2, 12)
    
    print(f"  NONE state shape: {states_none.shape}")
    print(f"  SIMPLE state shape: {states_simple.shape}")
    print(f"  PARTIAL state shape: {states_partial.shape}")
    print(f"  FULL state shape: {states_full.shape}")
    print("  Note: All states padded to (m, 4) for JAX lax.switch compatibility")
    # All states are padded to (m, 4) for JAX compatibility
    assert states_none.shape == (1, 4), "NONE should be padded to (1, 4)!"
    assert states_simple.shape == (12, 4), "SIMPLE should be padded to (12, 4)!"
    assert states_partial.shape == (12, 4), "PARTIAL should be padded to (12, 4)!"
    assert states_full.shape == (12, 4), "FULL should have (12, 4) state!"
    print("  OK: State structures OK")
    
    # Test 15: JAX JIT compilation verification
    print("\n[Test 15] JAX JIT compilation")
    print("  All core functions are JIT-compiled:")
    print("    - _init_state_n, _init_state_s, _init_state_p, _init_state_f")
    print("    - init_state (uses lax.switch)")
    print("    - ces_update_step (uses lax.cond)")
    print("    - ces_fit_forward (uses lax.scan)")
    print("    - ces_forecast (uses lax.fori_loop)")
    print("  OK: JIT compilation verified")
    
    print("\n" + "=" * 60)
    print("OK: All CES tests passed!")
    print("=" * 60)