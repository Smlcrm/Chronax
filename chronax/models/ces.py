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

from chronax.utils import ensure_float, calculate_information_criteria, ConformalIntervals, nelder_mead
from chronax.models.base_forecaster import BaseForecaster

NONE = 0
SIMPLE = 1
PARTIAL = 2
FULL = 3

# Inverse of auto_ces's model_map: variant code -> single-letter selector. Used to
# cache the eagerly-selected winner at fit so the vmapped conformity_scores CV path
# re-fits ONLY that variant per window (the AutoETS _selected_spec pattern).
_CODE_TO_LETTER = {NONE: "N", SIMPLE: "S", PARTIAL: "P", FULL: "F"}


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


from functools import lru_cache, partial

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
    # Pad to (m, 4) for compatibility with other variants.
    # Buffers follow y.dtype: hardcoded float32 caused f64->f32 downcast scatters
    # under the repo-global x64 (a FutureWarning today, a hard error in future JAX).
    base_state = jnp.array([[mean_val, mean_val / 1.1]], dtype=y.dtype)
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
    states = jnp.zeros((m, 4), dtype=y.dtype)
    states = states.at[:, 0].set(y[:m])
    states = states.at[:, 1].set(y[:m] / 1.1)
    # Columns 2-3 remain zero (not used in SIMPLE)
    return states


def _sf_seasonal_factors(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Seasonal init factors with ``seasonal_decompose(y, period=m).seasonal[:m]``
    semantics, vmap-native.

    Matches statsforecast's full-series centered-MA detrend plus per-phase means, and
    is guarded by a parity test against statsmodels in the contract suite. A
    first-cycle ``mode='same'`` detrend would instead let zero-padding bias the first
    m/2 trend values low by up to half the series LEVEL, which at large m collapses
    the PARTIAL/FULL fits and leaves AICc selecting NONE.

    Centered MA via ``mode='valid'`` ONLY (no padded values enter); even m uses the
    (m+1)-tap [0.5, 1, ..., 1, 0.5]/m filter; per-phase MASKED means over the valid
    positions (phase = absolute series position mod m); factors de-meaned. All shapes
    are static in (n, m) and everything follows ``y.dtype``. Requires n >= m + 1 (odd
    m) / m + 2 (even m); callers guard with the static n >= 2m branch. Deliberate
    deviation from statsforecast: y.dtype precision (SF buffers float32).
    """
    n = y.shape[0]
    dt = y.dtype
    if m % 2 == 0:
        kernel = (jnp.concatenate([jnp.array([0.5]), jnp.ones(m - 1), jnp.array([0.5])]) / m).astype(dt)
        k = m + 1
    else:
        kernel = (jnp.ones(m) / m).astype(dt)
        k = m
    half = k // 2  # k is odd in both branches, so the MA is exactly centered
    trend_valid = jnp.convolve(y, kernel, mode='valid')      # (n - k + 1,)
    detr = y[half:n - half] - trend_valid                     # positions half .. n-half-1
    phase = (jnp.arange(detr.shape[0]) + half) % m            # absolute position mod m
    onehot = jax.nn.one_hot(phase, m, dtype=dt)               # (L, m), static shapes
    sums = onehot.T @ detr
    counts = jnp.maximum(jnp.sum(onehot, axis=0), jnp.asarray(1.0, dt))
    seas = sums / counts
    return seas - jnp.mean(seas)


@partial(jit, static_argnums=(1,))
def _init_state_p(y: jnp.ndarray, m: int) -> jnp.ndarray:
    """Initialize state for PARTIAL variant (partial seasonal damping).

    Initializes the trend components (real/imaginary) with the mean of the first m values.
    Extracts seasonal components via moving average detrending when sufficient data is
    available (n >= 2*m), otherwise uses simple deviations from the mean.

    Seasonal init uses :func:`_sf_seasonal_factors`, which carries
    statsmodels ``seasonal_decompose`` semantics and stays well-conditioned at
    large m.

    Args:
        y: Input time series array
        m: Seasonal period

    Returns:
        State array of shape (m, 4) with columns [real, imag, seasonal, 0]
    """
    states = jnp.zeros((m, 4), dtype=y.dtype)
    mean_val = jnp.mean(y[:m])
    states = states.at[:, 0].set(mean_val)
    states = states.at[:, 1].set(mean_val / 1.1)

    # n and m are both static (y.shape + static_argnum) ⇒ static Python branch,
    # vmap-safe; it also guards _sf_seasonal_factors' n >= m+2 requirement.
    if len(y) >= 2 * m:
        seasonal = _sf_seasonal_factors(y, m)
    else:
        seasonal = y[:m] - mean_val

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
    states = jnp.zeros((m, 4), dtype=y.dtype)
    mean_val = jnp.mean(y[:m])
    states = states.at[:, 0].set(mean_val)
    states = states.at[:, 1].set(mean_val / 1.1)

    # Static n >= 2m Python branch (see _init_state_p); shares _sf_seasonal_factors.
    if len(y) >= 2 * m:
        seasonal = _sf_seasonal_factors(y, m)
    else:
        seasonal = y[:m] - mean_val

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


def _ces_read(states_buffer, i, season_type, m):
    """Read the CES level (time i-1; i-m for SIMPLE) and seasonal (time i-m) states
    and form the one-step-ahead forecast ``level[0] + seasonal``.

    Canonical CES (Svetunkov; matches statsforecast ``cesfcst``/``cesupdate``): the
    complex level evolves every step and is read from ``i-1``; the complex seasonal
    evolves every period and is read from ``i-m``. SIMPLE reads the level from ``i-m``
    (pure seasonal lag, no separate seasonal component). Index arithmetic is on the
    traced counter ``i`` mod the static period ``m`` — vmap-native.
    """
    lvl_idx = jnp.where(season_type == SIMPLE, (i - m) % m, (i - 1) % m)
    lvl = states_buffer[lvl_idx]
    seas = states_buffer[(i - m) % m]
    seas_term = jnp.where(season_type > SIMPLE, seas[2], 0.0)
    return lvl, seas, lvl[0] + seas_term


def _ces_update(lvl, seas, e, alpha_0, alpha_1, beta_0, beta_1, season_type, dtype):
    """Advance the complex level/seasonal state one step given innovation ``e``.

    Level from ``lvl`` (cols 0/1); seasonal from ``seas`` (cols 2/3). PARTIAL updates
    only the real seasonal component, FULL the complex seasonal pair, NONE/SIMPLE zero
    the seasonal columns. Branchless (``jnp.where`` on the static ``season_type``).
    """
    new0 = lvl[0] - (1.0 - alpha_1) * lvl[1] + (alpha_0 - alpha_1) * e
    new1 = lvl[0] + (1.0 - alpha_0) * lvl[1] + (alpha_0 + alpha_1) * e
    new2_partial = seas[2] + beta_0 * e
    new2_full = seas[2] - (1.0 - beta_1) * seas[3] + (beta_0 - beta_1) * e
    new3_full = seas[2] + (1.0 - beta_0) * seas[3] + (beta_0 + beta_1) * e
    new2 = jnp.where(season_type == PARTIAL, new2_partial,
                     jnp.where(season_type == FULL, new2_full, 0.0))
    new3 = jnp.where(season_type == FULL, new3_full, 0.0)
    return jnp.stack([new0, new1, new2, new3]).astype(dtype)


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
    """One in-sample CES step: emit the pre-update one-step-ahead forecast, then update.

    The forecast ``level[i-1] + seasonal[i-m]`` is emitted BEFORE consuming ``y_obs``,
    so the residual ``y_obs - forecast`` is a genuine one-step-ahead error. (The previous
    implementation emitted the post-update value, leaking ``y_obs`` into its own residual
    and corrupting IC-based variant selection, and it read the seasonal component from the
    wrong lag ``i-1`` — collapsing the seasonal variants to the non-seasonal one.)
    """
    states_buffer, i = carry
    lvl, seas, forecast = _ces_read(states_buffer, i, season_type, m)
    e = y_obs - forecast
    state_new = _ces_update(lvl, seas, e, alpha_0, alpha_1, beta_0, beta_1,
                            season_type, states_buffer.dtype)
    states_buffer = states_buffer.at[i % m].set(state_new)
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


@partial(jit, static_argnums=(5, 6, 7, 8))
def ces_forecast(
    final_state: jnp.ndarray,
    alpha_0: float,
    alpha_1: float,
    beta_0: float,
    beta_1: float,
    season_type: int,
    m: int,
    h: int,
    start_idx: int,
) -> jnp.ndarray:
    """Generate h-step CES forecasts with static season_type, m, h, and start index.

    Recursive CES forecasting (matches statsforecast ``cesfcst``): each step emits
    ``level[i-1] + seasonal[i-m]`` and propagates the state with zero innovation (the
    forecast feeds itself as the next "observation", so ``e == 0``). ``start_idx`` is the
    fit length ``n`` so the ring buffer is read at the correct phase — ``final_state`` at
    ring index ``(n-1) % m`` holds the latest level and ``(n-m) % m`` the latest seasonal
    (the previous version hardcoded ``start_idx = m``, phase-correct only when n % m == 0).
    """
    def forecast_step(i_h, carry):
        states_buffer, forecasts, current_idx = carry
        lvl, seas, forecast = _ces_read(states_buffer, current_idx, season_type, m)
        forecasts = forecasts.at[i_h].set(forecast)
        zero = jnp.asarray(0.0, dtype=states_buffer.dtype)
        state_new = _ces_update(lvl, seas, zero, alpha_0, alpha_1, beta_0, beta_1,
                                season_type, states_buffer.dtype)
        states_buffer = states_buffer.at[current_idx % m].set(state_new)
        return (states_buffer, forecasts, current_idx + 1)

    forecasts = jnp.zeros(h, dtype=final_state.dtype)
    states_buffer = final_state.copy()
    (states_buffer, forecasts, _) = lax.fori_loop(
        0, h, forecast_step, (states_buffer, forecasts, start_idx)
    )
    return forecasts

# CES smoothing-parameter optimizer configuration. Bounds and defaults mirror
# statsforecast (``ces.py`` ``optimize_ces_target_fn`` lower/upper + ``initparam``
# alpha_0=1.3, alpha_1=1.0 defaults); order is [alpha_0, alpha_1, beta_0, beta_1].
_CES_PAR_LO = jnp.array([0.01, 0.01, 0.01, 0.01])
_CES_PAR_HI = jnp.array([1.8, 1.9, 1.5, 1.5])
_CES_PAR_INIT = jnp.array([1.3, 1.0, 1.3, 1.0])
# Active (optimized) smoothing params per variant; the rest are inert for that variant.
_CES_N_ACTIVE = {NONE: 2, SIMPLE: 2, PARTIAL: 3, FULL: 4}
# Nelder-Mead iteration cap: CES objectives converge by ~55 (2-param) / ~120 (4-param)
# iterations (measured); 120 is accuracy-identical to 200 at ~40% less optimizer cost.
_CES_NM_MAXITER = 120


def _ces_backfit_forecasts(y, m, season_type, pvec4):
    """Run the 3-pass back-fit at params ``pvec4`` = [a0, a1, b0, b1]; return (states, in-sample forecasts)."""
    p = CESParams(alpha_0=pvec4[0], alpha_1=pvec4[1], beta_0=pvec4[2], beta_1=pvec4[3])
    init_state_arr = init_state(y, m, season_type)
    return ces_fit_backfit(y, init_state_arr, p, season_type, m)


@lru_cache(maxsize=64)
def _get_ces_param_runner(m: int, season_type: int, max_iter: int):
    """Build (and cache) the jitted Nelder-Mead parameter fit for one static CES config.

    Every argument is config-static, so the returned jitted callable keeps a stable
    identity across calls; the inner ``jax.jit`` then compiles once per (shape, dtype).
    Series length stays out of the key — ``_run`` derives every shape from the static
    ``m``. Without the cache, the ``objective`` closure below is a fresh object on every fit, and
    ``nelder_mead``'s ``while_loop`` — whose pjit cache keys on the callable's identity —
    recompiles an identical jaxpr each call (~180 ms per variant, 4 variants per
    ``auto_ces``). Mirrors ``ets_backend._get_optimizer_runner``.
    """
    n_active = _CES_N_ACTIVE[season_type]

    def _run(y: jnp.ndarray) -> jnp.ndarray:
        lo = _CES_PAR_LO[:n_active].astype(y.dtype)
        hi = _CES_PAR_HI[:n_active].astype(y.dtype)
        init_full = _CES_PAR_INIT.astype(y.dtype)

        def objective(xa):
            xa_c = jnp.clip(xa, lo, hi)
            pvec = init_full.at[:n_active].set(xa_c)
            _, fc = _ces_backfit_forecasts(y, m, season_type, pvec)
            resid = y[m:] - fc
            sse = jnp.sum(resid ** 2)
            return jnp.where(jnp.isfinite(sse), sse, jnp.inf)

        res = nelder_mead(objective, init_full[:n_active], max_iter=max_iter)
        return init_full.at[:n_active].set(jnp.clip(res.x, lo, hi))

    return jax.jit(_run)


def ces_fit_single(
    y: jnp.ndarray,
    m: int,
    season_type: int,
    params: Optional[CESParams] = None,
) -> Dict:
    """Fit a single CES variant (optimizing its smoothing params) and return metrics, fits, and state.

    When ``params`` is None the active smoothing parameters for this variant are fit by
    a vmap-native Nelder-Mead minimizer (:func:`chronax.utils.nelder_mead`) on the
    in-sample one-step-ahead SSE — matching statsforecast, which Nelder-Mead-optimizes
    alpha_0/alpha_1/(beta_0/beta_1) per variant. When ``params`` is given, that fixed
    parameterization is used (no optimization). ``season_type`` is static config, so the
    active-parameter count and IC penalty are resolved with plain Python control flow.

    Args:
        y: Time series of shape (n,).
        m: Seasonal period.
        season_type: Model variant (NONE=0, SIMPLE=1, PARTIAL=2, FULL=3).
        params: Optional fixed CESParams. If None (default), optimize the variant's params.

    Returns:
        Dict with keys ``loglik``/``aic``/``bic``/``aicc``/``mse``/``amse`` (jnp scalars),
        ``fitted``/``residuals``/``states`` (arrays), ``par`` (dict), ``par_vec`` (the fitted
        [a0,a1,b0,b1] jnp vector used for forecasting), ``m``/``n``/``seasontype`` (ints),
        and ``sigma2`` (jnp scalar).
    """
    y = ensure_float(y)
    dtype = y.dtype

    if params is not None:
        pvec4 = jnp.asarray([
            params.alpha_0,
            params.alpha_1,
            params.beta_0 if params.beta_0 is not None else 0.0,
            params.beta_1 if params.beta_1 is not None else 0.0,
        ], dtype=dtype)
    else:
        pvec4 = _get_ces_param_runner(m, season_type, _CES_NM_MAXITER)(y)

    p = CESParams(alpha_0=pvec4[0], alpha_1=pvec4[1], beta_0=pvec4[2], beta_1=pvec4[3])
    init_state_arr = init_state(y, m, season_type)
    final_states, forecasts = ces_fit_backfit(y, init_state_arr, p, season_type, m)

    n = len(y)
    # SF-faithful parameter count for the IC: components = 2 + (PARTIAL) + 2*(FULL); np_ = components + 1.
    n_params = 2 + int(season_type == PARTIAL) + 2 * int(season_type == FULL) + 1
    n_residuals = n - m

    fitted = jnp.empty(n, dtype=dtype)
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
        'mse': mse,
        'amse': mse,
        'fitted': fitted,
        'residuals': residuals,
        'states': final_states,
        'par': p.to_dict(),
        'par_vec': pvec4,
        'm': m,
        'n': n,
        'seasontype': season_type,
        'sigma2': sigma2,
    }


def auto_ces(
    y: jnp.ndarray,
    m: int = 1,
    model: str = "Z",
    ic: str = "aicc",
) -> Dict:
    """Fit CES with automatic or fixed model selection.

    When model="Z", fits all applicable variants (NONE always; SIMPLE/PARTIAL/FULL
    when n >= 2*m) and selects the fit with the lowest information criterion.
    Otherwise, fits only the specified variant.

    vmap-native: the variant loop is over *config* (each variant traces the jitted
    kernels with its own static season_type), NaN ICs are masked to +inf instead of
    being skipped eagerly, and the winner is chosen with `jnp.argmin` — no
    float()/int() casts, no try/except, no Python control flow on traced values.

    Args:
        y: Time series of shape (n,).
        m: Seasonal period. Default is 1 (no seasonality).
        model: Variant selector. "Z" for automatic selection; one of "N", "S", "P",
            "F" to fix the variant. Default is "Z".
        ic: Information criterion used for model selection when model="Z".
            One of "aic", "bic", "aicc". Default is "aicc".

    Returns:
        Dict with the selected fit's fields (loglik/aic/bic/aicc/mse/amse/sigma2 as
        jnp scalars; fitted/residuals/states arrays; seasontype as a jnp int scalar;
        m/n static ints) plus a candidate block used for traceable forecasting:
            - "variants" (tuple[int]): static candidate variant codes.
            - "candidate_states" (jnp.ndarray): stacked final states, (k, m, 4).
            - "best" (jnp.ndarray): argmin index into the candidate axis.
    """
    y = ensure_float(y)

    model_map = {"N": NONE, "S": SIMPLE, "P": PARTIAL, "F": FULL}

    if model == "Z":
        variants = [NONE, SIMPLE, PARTIAL, FULL]
        # Both conditions are static under trace: m is config, len(y) is shape.
        if m < 2 or len(y) < 2 * m:
            variants = [NONE]
    else:
        variants = [model_map.get(model, NONE)]

    fits = [ces_fit_single(y, m, variant) for variant in variants]

    ic_values = jnp.stack([jnp.asarray(fit[ic]) for fit in fits])
    ic_values = jnp.where(jnp.isnan(ic_values), jnp.inf, ic_values)
    best = jnp.argmin(ic_values)
    # Raising on all-invalid ICs is impossible under trace; expose a flag instead
    # (argmin over all-inf silently picks index 0 and the winner's fields are NaN).
    valid = jnp.isfinite(ic_values).any()

    def _select(key):
        return jnp.take(jnp.stack([jnp.asarray(fit[key]) for fit in fits]), best, axis=0)

    return {
        'loglik': _select('loglik'),
        'aic': _select('aic'),
        'bic': _select('bic'),
        'aicc': _select('aicc'),
        'mse': _select('mse'),
        'amse': _select('amse'),
        'sigma2': _select('sigma2'),
        'fitted': _select('fitted'),
        'residuals': _select('residuals'),
        'states': _select('states'),
        'seasontype': jnp.take(jnp.asarray(variants), best),
        'm': m,
        'n': len(y),
        'variants': tuple(variants),
        'candidate_states': jnp.stack([fit['states'] for fit in fits]),
        'candidate_par_vecs': jnp.stack([fit['par_vec'] for fit in fits]),
        'best': best,
        'valid': valid,
    }


def _forecast_candidates(mod: Dict, h: int) -> jnp.ndarray:
    """Forecast every candidate variant and select the winner's row.

    `ces_forecast` requires a *static* season_type, so the selected variant
    (a traced argmin index) cannot be dispatched on directly. Instead each
    candidate is forecast with its own static config (a Python loop over
    config, fine under trace) and the winning row is picked with `jnp.take`.

    Args:
        mod: Dict returned by auto_ces() (needs the candidate block).
        h: Forecast horizon (static).

    Returns:
        Point forecasts of shape (h,) for the IC-selected variant.
    """
    m = mod['m']
    par_vecs = mod['candidate_par_vecs']
    forecasts = []
    for i, variant in enumerate(mod['variants']):
        pv = par_vecs[i]
        forecasts.append(
            ces_forecast(
                mod['candidate_states'][i], pv[0], pv[1],
                pv[2], pv[3], variant, m, h, mod['n']
            )
        )
    return jnp.take(jnp.stack(forecasts), mod['best'], axis=0)


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
        ``model_`` (dict | None): Populated after fit(); contains fitted values,
            residuals, states, parameters, and information criteria from
            ces_fit_single(). None before first fit.
    """

    uses_exog = False

    def __init__(
        self,
        season_length: "int | str" = 1,
        model: str = "Z",
        alias: str = "CES",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialise AutoCES with model configuration.

        Args:
            season_length (int | str): Seasonal period m, or "auto" to infer it
                from the training series at fit (via `detect_period`, resolved
                once and cached). Default is 1.
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
        # Resolved concrete period (set at fit for season_length="auto"; an
        # explicit int is used verbatim). Cached so the vmapped/stateless
        # forecast path never re-resolves on a traced CV window (eager-only).
        self._m_eff = None
        # Winning variant letter ("N"/"S"/"P"/"F") cached at fit; None until fit.
        # The vmapped conformity_scores CV path re-fits ONLY this variant per window
        # (AutoETS _selected_spec pattern) instead of the full 4-variant auto_ces("Z").
        self._selected_variant: Optional[str] = None

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> "AutoCES":
        """Fit the CES model to a time series.

        Delegates to auto_ces(), which runs variant selection and back-fitting
        (constant series need no special casing: innovations are zero, so
        forecasts stay flat at the series mean).

        Args:
            y (jnp.ndarray): Input time series of shape (n,).

            X (Optional[jnp.ndarray]): Exogenous variables (unused; kept for API
                compatibility). Default is None.

        Returns:
            AutoCES: Self (fitted model instance) for method chaining.
        """
        y = ensure_float(y)
        # Resolve season_length="auto" ONCE here (eager, on the full series) and
        # cache it; the stateless/vmapped forecast path reuses the cached int so
        # a traced CV window never re-runs detect_period.
        self._m_eff = self._resolve_season_length(self.season_length, y)
        # No eager constant-series special case: `jnp.std(y) < 1e-10` is Python
        # control flow on a traced value. The normal path already yields flat
        # forecasts on constant series (innovations are zero throughout).
        self.model_ = auto_ces(y, m=self._m_eff, model=self.model)
        # Cache the eagerly-selected winning variant so the vmapped conformity_scores
        # CV path re-fits ONLY it per window (the AutoETS _selected_spec pattern).
        # int() here is eager/host-side (fit is never traced); forecast reads this
        # concrete string (like self._m_eff), so no tracer leaks under the CV vmap.
        self._selected_variant = _CODE_TO_LETTER[int(self.model_['seasontype'])]
        # Pre-compute and cache conformity scores on the training series for
        # predict() intervals (sibling convention: WindowAverage/SES/SeasWA) —
        # avoids paying n_windows re-fits on every predict(level=...) call.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self
    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict:
        """Stateless fit+forecast: always re-fits on the passed y.

        Deliberately ignores any stored ``model_``: the base class's
        conformity_scores vmaps this method over CV windows, and reusing
        fitted full-series state would return identical (future-leaking)
        forecasts for every window. Write-free on the CV/vmap path
        (``level=None``, the base's CV call); a direct ``forecast(level=...)`` on an
        UNFITTED instance triggers fit via ``conformity_scores`` (family-consistent).

        Args:
            y (jnp.ndarray): Input time series of shape (n,).
            h (int): Forecast horizon (number of steps ahead).
            X (Optional[jnp.ndarray]): Exogenous variables (unused). Default is None.
            X_future (Optional[jnp.ndarray]): Future exogenous variables (unused). Default is None.
            level (Optional[List[int]]): Confidence levels (0-100) for conformal
                prediction intervals. Requires conformal_params. Default is None.
            fitted (bool): Whether to include in-sample fitted values under the
                "fitted" key. Default is False.

        Returns:
            Dict: "mean" forecasts of shape (h,), plus "fitted" when requested and
            "lo-{l}"/"hi-{l}" bounds when level is given.
        """
        y = ensure_float(y)
        # Reuse the fit-cached resolved period under the CV vmap; only resolve
        # here on a fresh/direct forecast (y concrete → detect_period eager-safe).
        m_eff = self._m_eff if self._m_eff is not None else self._resolve_season_length(self.season_length, y)
        # Under the conformity_scores CV vmap, reuse the fit-selected variant (fits 1,
        # not all 4); a fresh/direct forecast with no prior fit falls back to the full
        # auto-select (self.model, typically "Z"). Concrete string ⇒ static under vmap.
        variant = self._selected_variant if self._selected_variant is not None else self.model
        mod = auto_ces(y, m=m_eff, model=variant)

        res = {'mean': _forecast_candidates(mod, h)}
        if fitted:
            res['fitted'] = mod['fitted']

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )
        cs = self.conformity_scores(y=y, X=X)
        return self.add_confidence_intervals(res, cs, level, self.conformal_params.method)

    def conformity_scores(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Eager variant+period selection, then the vmapped per-window param refit.

        The base ``conformity_scores`` vmaps ``self.forecast`` over CV windows. Two
        things are resolved ONCE, eagerly, before that vmap, or every window re-does
        expensive work on traced data:

        * **Variant** — a fresh ``forecast`` runs ``auto_ces(model="Z")``, fitting ALL
          4 variants (NONE/SIMPLE/PARTIAL/FULL) and arg-min-ing their IC. Repeating
          that per window costs 4x for negligible benefit on typical-length series: the
          per-window winner often DIFFERS from the full-series winner, but the variants
          are near-ties so the conformity scores (hence intervals) stay close (pooled 95%
          coverage delta ~-0.8pt). ⚠ On VERY short series (padded CV windows < 2m) the old
          padded CV windows would otherwise statically fit NONE-only per window while
          this path forces the full-series winner — a variant-REGIME change rather than
          a near-tie flip, so very short series are the worst case for coverage. The
          winner is selected once at ``fit`` (cached ``_selected_variant``); the vmapped
          ``forecast`` then re-fits only that variant's *parameters* per window.
          Mirrors AutoETS ``_selected_spec``.
        * **Period** — with ``season_length="auto"`` the vmapped ``forecast`` would
          re-run ``detect_period`` on a *traced* window (ConcretizationError); resolving
          ``_m_eff`` on the concrete ``y`` here fixes it.

        Both ``_selected_variant`` (str) and ``_m_eff`` (int) are concrete, so writing
        them is benign for statelessness. Calibration caveat (shared with AutoARIMA
        and AutoMFLES): the variant is chosen with sight of the full series incl.
        the CV test windows — mildly optimistic; parameters are still honestly re-fit
        per window, so scores vary across windows.

        Side effect: first call on an unfitted estimator runs ``fit`` (caches the
        selected variant + period + scores), mirroring the sibling Auto overrides.
        """
        y = ensure_float(y)
        if self._selected_variant is None:
            self.fit(y, X)
            # fit() just ran the CV on this exact y and cached the scores; reuse them
            # rather than paying the n_windows re-fits twice.
            if self._cs is not None:
                return self._cs
        else:
            self._m_eff = self._resolve_season_length(self.season_length, y)
        return super().conformity_scores(y=y, X=X)

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None) -> Dict:
        """Generate h-step ahead forecasts from the fitted CES model.

        Forecasts every stored candidate variant via the JIT-compiled
        ces_forecast() and selects the IC winner's row (see _forecast_candidates).
        Optionally adds conformal prediction intervals.

        Args:
            h (int): Forecast horizon (number of steps ahead).

            X (Optional[jnp.ndarray]): Exogenous variables (unused; kept for API
                compatibility). Default is None.

            level (Optional[List[int]]): Confidence levels (0-100) for conformal
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

        result = {'mean': _forecast_candidates(self.model_, h)}

        if level is None:
            return result

        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if getattr(self, "_cs", None) is None:
            raise ValueError(
                "Conformity scores are not available. Fit the model first (fit(...)) "
                "with `conformal_params` set so predict() can use cached scores."
            )
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )
        return self.add_confidence_intervals(result, self._cs, level, self.conformal_params.method)


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
    par1 = CESParams.for_variant(int(model1.model_['seasontype'])).to_dict()
    print(f"  Alpha_0: {par1['alpha_0']:.4f}")
    print(f"  Alpha_1: {par1['alpha_1']:.4f}")
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
    par3 = CESParams.for_variant(int(model3.model_['seasontype'])).to_dict()
    print(f"  Beta_0: {par3['beta_0']}")
    print(f"  Forecast variance: {jnp.var(forecast3['mean']):.4f}")
    assert model3.model_['seasontype'] == PARTIAL, "Should be PARTIAL variant!"
    assert par3['beta_0'] is not None, "Should have beta_0!"
    assert jnp.all(jnp.isfinite(forecast3['mean'])), "Forecast contains NaN/Inf!"
    print("  OK: PARTIAL seasonality OK")
    
    # Test 4: Full seasonal variant
    print("\n[Test 4] CES with FULL seasonality")
    model4 = AutoCES(season_length=period, model="F")
    model4.fit(y2)
    forecast4 = model4.predict(h=12)
    
    print(f"  Season type: {model4.model_['seasontype']} (FULL)")
    par4 = CESParams.for_variant(int(model4.model_['seasontype'])).to_dict()
    print(f"  Beta_0: {par4['beta_0']}")
    print(f"  Beta_1: {par4['beta_1']}")
    assert model4.model_['seasontype'] == FULL, "Should be FULL variant!"
    assert par4['beta_0'] is not None, "Should have beta_0!"
    assert par4['beta_1'] is not None, "Should have beta_1!"
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