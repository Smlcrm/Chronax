"""
Holt-Winters' exponential smoothing in JAX.

Triple exponential smoothing with additive or multiplicative error and
seasonality, an optional damped trend, and a two-phase Adam + L-BFGS fit.

Implementation notes:

- Sigmoid reparameterization bounds each smoothing parameter to (0, 1)
  independently (a per-parameter box, not a joint exponential smoothing (ETS)
  admissibility constraint; the fit can reach the box corners, e.g. gamma
  near 1).
- States are initialized by classical decomposition with trend estimation.
- Eight variants: AAA, AAM, MAA, MAM, each with or without damping.
- Analytical interval formulas follow Hyndman et al. (2008) and Taylor (2003).
- Seasonal constraint: additive factors sum to 0, multiplicative factors
  average to 1.
"""
import jax
import jax.numpy as jnp
from chronax import utils
import optax
from chronax.utils import ConformalIntervals
from chronax.models.base_forecaster import BaseForecaster
from jax import lax


# =============================================================================
# Constants
# =============================================================================

_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

_EPS_PARAM = 1e-4
_ADAM_STEPS = 100
_ADAM_LR = 0.05
_LBFGS_STEPS = 50
_LBFGS_MEMORY = 8
_LBFGS_LS_STEPS = 20
_EPSILON = 1e-10

__all__ = ['HoltWinters']


# =============================================================================
# Core math: module-level jitted functions
# =============================================================================

def _to_constrained(p_raw: jnp.ndarray, l0_decomp: float, b0_decomp: float,
                    s0_decomp: jnp.ndarray, l_scale: jnp.ndarray,
                    b_scale: jnp.ndarray, s_scale: jnp.ndarray,
                    is_additive_season: bool, season_length: int
                    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                               jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert unconstrained parameters to constrained space via sigmoid.

    Smoothing parameters (alpha, beta, gamma) are mapped to (eps, 1-eps)
    via sigmoid. State parameters (l0, b0, s0) are offsets from the
    decomposition-based initial estimates, scaled by data-derived factors.
    The last seasonal factor is pinned by the constraint (sum-to-0 for
    additive, mean-to-1 for multiplicative).
    """
    m = season_length
    eps = _EPS_PARAM

    alpha = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[0])
    beta  = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[1])
    gamma = eps + (1.0 - 2*eps) * jax.nn.sigmoid(p_raw[2])

    l0 = l0_decomp + p_raw[3] * l_scale
    b0 = b0_decomp + p_raw[4] * b_scale
    s_free = s0_decomp[:m-1] + p_raw[5:m+4] * s_scale

    if is_additive_season:
        s_last = -jnp.sum(s_free)
    else:
        s_last = m - jnp.sum(s_free)
    s0 = jnp.concatenate([s_free, s_last[None]])

    return alpha, beta, gamma, l0, b0, s0


def _hw_core(alpha: jnp.ndarray, beta: jnp.ndarray, gamma: jnp.ndarray,
             l0: jnp.ndarray, b0: jnp.ndarray, s0: jnp.ndarray,
             y: jnp.ndarray, phi: float,
             is_additive_season: bool, season_length: int
             ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                        jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass and MLE loss for Holt-Winters.

    Error type (A vs M) only affects prediction intervals, not state
    updates or loss, so it is not a parameter here.

    Returns (loss, fitted_vals, final_level, final_trend, final_seasonal, residuals).
    """
    m = season_length
    n = y.shape[0]

    # Additive seasonality: y_hat = (level + phi*trend) + seasonal
    def step_additive(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = level_prev + phi_trend + s_prev
        level = a * (y_t - s_prev) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t - level) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    # Multiplicative seasonality: y_hat = (level + phi*trend) * seasonal
    def step_multiplicative(carry, y_t):
        level_prev, trend_prev, seasonal_prev, a, b, g = carry
        phi_trend = phi * trend_prev
        s_prev = seasonal_prev[0]

        y_hat = (level_prev + phi_trend) * s_prev
        level = a * (y_t / jnp.maximum(s_prev, _EPSILON)) + (1 - a) * (level_prev + phi_trend)
        trend = b * (level - level_prev) + (1 - b) * phi_trend
        new_seasonal = g * (y_t / jnp.maximum(level, _EPSILON)) + (1 - g) * s_prev

        seasonal_new = jnp.roll(seasonal_prev, -1)
        seasonal_new = seasonal_new.at[-1].set(new_seasonal)
        return (level, trend, seasonal_new, a, b, g), y_hat

    step_fn = step_additive if is_additive_season else step_multiplicative

    init_carry = (l0, b0, s0, alpha, beta, gamma)
    final_carry, fitted_vals = lax.scan(step_fn, init_carry, y)
    final_level, final_trend, final_seasonal, _, _, _ = final_carry
    residuals = y - fitted_vals

    sse = jnp.sum(residuals ** 2)
    loss = n * jnp.log(jnp.maximum(sse, _EPSILON))

    return loss, fitted_vals, final_level, final_trend, final_seasonal, residuals


# =============================================================================
# Optimizer: Adam + L-BFGS
# =============================================================================

def _optimize_hw(y: jnp.ndarray, l0_decomp: float, b0_decomp: float,
                 s0_decomp: jnp.ndarray, phi: float,
                 is_additive_season: bool, season_length: int
                 ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                            jnp.ndarray, jnp.ndarray,
                            jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two-phase Adam + L-BFGS optimization for Holt-Winters.

    Parameters
    ----------
    y : jnp.ndarray
        Time series of shape (n,).
    l0_decomp : float
        Initial level from classical decomposition.
    b0_decomp : float
        Initial trend from classical decomposition.
    s0_decomp : jnp.ndarray
        Initial seasonal factors from classical decomposition, shape (m,).
    phi : float
        Damping factor (static arg). 1.0 for undamped.
    is_additive_season : bool
        Whether season type is additive (static arg).
    season_length : int
        Number of observations per season (static arg).

    Returns
    -------
    tuple
        (fitted, level, trend, seasonal, residuals, alpha, beta, gamma)
    """
    m = season_length
    dtype = y.dtype

    l_scale = jnp.std(y)
    b_scale = jnp.std(jnp.diff(y))
    if is_additive_season:
        s_scale = jnp.std(y)
    else:
        s_scale = jnp.std(y) / jnp.maximum(jnp.mean(y), _EPSILON)

    def loss_fn(p_raw: jnp.ndarray) -> jnp.ndarray:
        p_raw = jnp.asarray(p_raw, dtype=dtype)
        alpha, beta, gamma, l0, b0, s0 = _to_constrained(
            p_raw, l0_decomp, b0_decomp, s0_decomp,
            l_scale, b_scale, s_scale, is_additive_season, m)
        loss, *_ = _hw_core(alpha, beta, gamma, l0, b0, s0, y,
                            phi, is_additive_season, m)
        return loss

    vg_fn = jax.value_and_grad(loss_fn)

    # sigmoid(-2.0) ≈ 0.12, sigmoid(-5.0) ≈ 0.007
    p0_smooth = jnp.array([-2.0, -5.0, -2.0], dtype=dtype)
    p0_states = jnp.zeros(m + 1, dtype=dtype)
    p0 = jnp.concatenate([p0_smooth, p0_states])

    inf_val = jnp.asarray(float('inf'), dtype=dtype)

    # Phase 1: Adam warm-up
    adam_opt = optax.adam(_ADAM_LR)
    adam_state = adam_opt.init(p0)

    def adam_step(carry: tuple, _: None) -> tuple[tuple, None]:
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = adam_opt.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        # loss was evaluated at p, so p (not new_p) is what best_loss labels
        best_p = jnp.where(improved, p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    adam_carry = (p0, adam_state, p0, inf_val)
    (_, _, adam_best_p, adam_best_loss), _ = lax.scan(
        adam_step, adam_carry, None, length=_ADAM_STEPS)

    # Phase 2: L-BFGS refinement
    lbfgs_solver = optax.lbfgs(
        memory_size=_LBFGS_MEMORY,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=_LBFGS_LS_STEPS, initial_guess_strategy="one"))
    lbfgs_state = lbfgs_solver.init(adam_best_p)

    def lbfgs_step(carry: tuple, _: None) -> tuple[tuple, None]:
        p, state, best_p, best_loss = carry
        loss, grads = vg_fn(p)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = lbfgs_solver.update(
            grads, state, p, value=loss, grad=grads, value_fn=loss_fn)
        new_p = optax.apply_updates(p, updates)
        improved = jnp.isfinite(loss) & (loss < best_loss)
        best_p = jnp.where(improved, p, best_p)
        best_loss = jnp.where(improved, loss, best_loss)
        return (new_p, new_state, best_p, best_loss), None

    lbfgs_carry = (adam_best_p, lbfgs_state, adam_best_p, adam_best_loss)
    (_, _, lbfgs_best_p, lbfgs_best_loss), _ = lax.scan(
        lbfgs_step, lbfgs_carry, None, length=_LBFGS_STEPS)

    # Pick best
    use_lbfgs = jnp.isfinite(lbfgs_best_loss) & (lbfgs_best_loss < adam_best_loss)
    best_p = jnp.where(use_lbfgs, lbfgs_best_p, adam_best_p)

    # Final forward pass with optimal params
    alpha, beta, gamma, l0, b0, s0 = _to_constrained(
        best_p, l0_decomp, b0_decomp, s0_decomp,
        l_scale, b_scale, s_scale, is_additive_season, m)
    loss, fitted, level, trend, seasonal, residuals = _hw_core(
        alpha, beta, gamma, l0, b0, s0, y,
        phi, is_additive_season, m)

    return fitted, level, trend, seasonal, residuals, alpha, beta, gamma

_optimize_hw = jax.jit(_optimize_hw, static_argnums=(4, 5, 6))


# =============================================================================
# HoltWinters Class
# =============================================================================

class HoltWinters(BaseForecaster):
    r"""Holt-Winters' seasonal exponential smoothing.

    Fits the smoothing parameters and initial states by Adam + L-BFGS on
    ``n * log(SSE)`` (SSE: sum of squared errors). Native intervals use the
    Hyndman et al. (2008) and Taylor (2003) formulas; conformal intervals
    come from ``BaseForecaster`` when ``conformal_params`` is set.

    Parameters
    ----------
    season_length : int, default 12
        Number of observations per seasonal cycle. Must be >= 2.
    error_type : str, default 'A'
        Error type: 'A' (additive) or 'M' (multiplicative).
    season_type : str, default 'A'
        Seasonality type: 'A' (additive) or 'M' (multiplicative).
    damped : bool or None, default None
        Whether to use a damped trend. ``None`` is treated as ``False``.
    phi : float or None, default None
        Damping parameter in [0.8, 0.98]. Used only when ``damped=True``;
        ``None`` then means 0.9.
    alias : str, default 'HoltWinters'
        Custom name for the model.
    conformal_params : ConformalIntervals or None, default None
        Configuration for conformal prediction intervals.

    Attributes
    ----------
    model_ : dict
        Set by :meth:`fit`: ``fitted``, ``level``, ``trend``, ``seasonal``,
        ``residuals``, ``alpha``, ``beta``, ``gamma``, ``sigma``,
        ``sigma_rel``, ``y_train``.

    Raises
    ------
    ValueError
        If ``season_length`` is not an int >= 2; if ``error_type`` or
        ``season_type`` is not 'A'/'M'; if ``phi`` is not a number in
        [0.8, 0.98]; or if ``conformal_params`` is not a
        ``ConformalIntervals``.
    """

    @staticmethod
    def _validate_h(h: int) -> None:
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be a positive integer, got {h}")

    @staticmethod
    def _validate_level(level: list[int] | None) -> None:
        if level is not None:
            if not isinstance(level, list):
                raise ValueError(f"level must be a list or None, got {type(level).__name__}")
            if any(not isinstance(lv, (int, float)) or lv < 0 or lv > 100 for lv in level):
                raise ValueError("All level values must be numbers between 0 and 100")

    def _initialize_states(self, y: jnp.ndarray) -> tuple[float, float, jnp.ndarray]:
        """Initialize level, trend, and seasonal states via classical decomposition.

        When fewer than 2 full seasons are available, falls back to simple
        averages. Otherwise uses ordinary least squares (OLS) on per-season
        averages to estimate a per-step trend, then detrends to extract the
        seasonal pattern.

        Parameters
        ----------
        y : jnp.ndarray
            Time series of shape (n,).

        Returns
        -------
        tuple[float, float, jnp.ndarray]
            ``(l0, b0, s0)``: initial level, per-step trend, and seasonal
            factors of shape ``(season_length,)``.
        """
        m = self.season_length
        n = len(y)
        n_seasons = n // m

        # All branches below are static (shapes/config only); l0/b0 stay jnp
        # scalars, since float() on y-derived values breaks the vmapped
        # cross-validation (CV) path.
        if n_seasons < 2:
            l0 = jnp.mean(y[:m]) if n >= m else y[0]
            b0 = 0.0
            if n >= m:
                s0 = y[:m] - l0 if self.season_type == 'A' else y[:m] / jnp.maximum(l0, _EPSILON)
            else:
                s0 = jnp.zeros(m, dtype=y.dtype) if self.season_type == 'A' else jnp.ones(m, dtype=y.dtype)
            return l0, b0, s0.astype(y.dtype)

        n_complete = n_seasons * m
        y_reshaped = y[:n_complete].reshape(n_seasons, m)
        seasonal_avgs = jnp.mean(y_reshaped, axis=1)

        # OLS on seasonal averages for trend
        t = jnp.arange(n_seasons, dtype=y.dtype)
        t_mean = jnp.mean(t)
        avg_mean = jnp.mean(seasonal_avgs)
        cov_ta = jnp.sum((t - t_mean) * (seasonal_avgs - avg_mean))
        var_t = jnp.sum((t - t_mean) ** 2)
        b0_per_season = cov_ta / jnp.maximum(var_t, _EPSILON)
        l0 = avg_mean - b0_per_season * t_mean
        b0 = b0_per_season / m
        l0 = l0 - b0 * (m - 1) / 2.0

        # Detrend and compute seasonal pattern
        trend_vals = l0 + b0 * jnp.arange(n_complete, dtype=y.dtype)
        y_complete = y[:n_complete]
        if self.season_type == 'A':
            detrended = y_complete - trend_vals
        else:
            detrended = y_complete / jnp.maximum(trend_vals, _EPSILON)

        s0 = jnp.mean(detrended.reshape(n_seasons, m), axis=0)

        # Normalize: additive sums to 0, multiplicative averages to 1
        if self.season_type == 'A':
            s0 = s0 - jnp.mean(s0)
        else:
            s0 = s0 / jnp.maximum(jnp.mean(s0), _EPSILON)

        return l0, b0, s0.astype(y.dtype)

    def _get_phi(self) -> float:
        """Return the damping factor: self.phi (or 0.9) if damped, else 1.0."""
        if self.damped:
            return self.phi if self.phi is not None else 0.9
        return 1.0

    def _compute_base_variance(self, t: jnp.ndarray, alpha: float,
                               beta: float, phi: float) -> jnp.ndarray:
        """Compute cumulative variance multiplier for the trend component.

        Uses the damped-trend formula when phi < 1 (Hyndman et al. 2008,
        Ch. 6) and the simpler undamped formula otherwise.
        """
        if self.damped and phi < 0.9999:
            denom = jnp.maximum(1 - phi, _EPSILON)
            denom2 = jnp.maximum(1 - phi**2, _EPSILON)
            trend_var = (beta * phi * t) / denom**2 * (2 * alpha * denom + beta * phi) \
                       - (beta * phi * (1 - phi**t)) / (denom**2 * denom2) \
                       * (2 * alpha * denom2 + beta * phi * (1 + 2 * phi - phi**t))
            return 1 + alpha**2 * (t - 1) + trend_var
        else:
            exp1 = alpha**2 + alpha * beta * t + (1 / 6) * beta**2 * t * (2 * t - 1)
            return 1 + (t - 1) * exp1

    def _validate_forecast_inputs(self, y: jnp.ndarray, h: int,
                                  level: list[int] | None) -> jnp.ndarray:
        """Validate and convert inputs for forecast/forward methods."""
        y = utils.ensure_float(y)
        if len(y) < self.season_length:
            raise ValueError(
                f"Time series must have at least {self.season_length} observations "
                f"(season_length), got {len(y)}"
            )
        self._validate_h(h)
        self._validate_level(level)
        return y

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Fit smoothing parameters and initial states via Adam + L-BFGS.

        Returns a dict with keys ``fitted``, ``level``, ``trend``,
        ``seasonal``, ``residuals``, ``alpha``, ``beta``, ``gamma``,
        ``sigma`` and ``sigma_rel``.
        """
        l0, b0, s0 = self._initialize_states(y)
        phi = self._get_phi()
        is_additive_season = self.season_type == 'A'

        fitted, level, trend, seasonal, residuals, alpha, beta, gamma = _optimize_hw(
            y, l0, b0, s0, phi, is_additive_season, self.season_length)

        # dof follows the reference AAA convention: (m+5) free parameters
        # plus one for sigma^2 -> n - (m+6). Multiplicative-error/-season
        # interval formulas need the RELATIVE-residual sigma (dimensionless;
        # the width formula multiplies by |mean| exactly once).
        dof = len(y) - (self.season_length + 6)
        denom_f = jnp.where(jnp.abs(fitted) > 1e-10, fitted,
                            jnp.asarray(1.0, y.dtype))
        return {
            'fitted': fitted, 'level': level, 'trend': trend,
            'seasonal': seasonal, 'residuals': residuals,
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'sigma': utils.calculate_sigma(residuals, dof),
            'sigma_rel': utils.calculate_sigma(residuals / denom_f, dof),
        }

    def _generate_forecasts(self, level: jnp.ndarray, trend: jnp.ndarray,
                            seasonal: jnp.ndarray, phi: float, h: int) -> jnp.ndarray:
        """Generate h-step ahead point forecasts (vectorized)."""
        m = self.season_length
        dt = level.dtype if hasattr(level, 'dtype') else jnp.float64
        t_vals = jnp.arange(1, h + 1, dtype=dt)

        if phi == 1.0:
            trend_components = t_vals * trend
        else:
            trend_components = phi * (1 - phi**t_vals) / (1 - phi) * trend

        s_components = seasonal[jnp.arange(h) % m]

        if self.season_type == 'A':
            return level + trend_components + s_components
        else:
            return (level + trend_components) * s_components

    def _calculate_native_intervals(self, mean: jnp.ndarray, sigma: float,
                                     alpha: float, beta: float, gamma: float,
                                     phi: float, h: int) -> jnp.ndarray:
        """Calculate prediction interval width (sigmah) using analytical formulas.

        Based on Hyndman et al. (2008) and Taylor (2003).
        """
        dt = mean.dtype if hasattr(mean, 'dtype') else jnp.float64
        t = jnp.arange(1, h + 1, dtype=dt)
        # The recursion stores the CLASSICAL smoothing weights; the variance
        # formulas (Hyndman et al. 2008, class 1) are stated in the
        # innovations parametrization: beta_i = alpha*beta, gamma_i =
        # gamma*(1-alpha).
        beta_i = alpha * beta
        gamma_i = gamma * (1.0 - alpha)
        base_var = self._compute_base_variance(t, alpha, beta_i, phi)
        m = self.season_length
        hm = (t - 1) // m
        seasonal_var = gamma_i * hm * (2.0 * alpha + gamma_i + beta_i * m * (hm + 1.0))

        if self.error_type == 'A':
            if self.season_type == 'A':
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var)
            else:
                sigmah = sigma * jnp.sqrt(base_var + seasonal_var) * jnp.abs(mean)
        else:
            sigmah = sigma * jnp.sqrt(base_var + seasonal_var) * jnp.abs(mean)

        return sigmah

    def _add_interval_bounds(self, res: dict, values: jnp.ndarray,
                             sigmah: jnp.ndarray | float, level: list[int],
                             prefix: str = '') -> dict:
        """Add lo/hi prediction interval bounds to result dict.

        ``sigmah`` may be a per-step array (forecast intervals) or a
        scalar (fitted-value intervals), both broadcast correctly.
        """
        for lv in reversed(level):
            alpha_level = (100 - lv) / 100
            z = utils._jax_norm_ppf(1 - alpha_level / 2)
            res[f'{prefix}lo-{lv}'] = values - z * sigmah
            res[f'{prefix}hi-{lv}'] = values + z * sigmah
        return res

    def _compute_forecast_with_intervals(self, result: dict, phi: float, h: int,
                                          level: list[int] | None, fitted: bool,
                                          y: jnp.ndarray,
                                          X: jnp.ndarray | None) -> dict:
        """Generate forecasts and optionally add intervals and fitted values."""
        mean = self._generate_forecasts(
            result['level'], result['trend'], result['seasonal'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                # conformity_scores only calls self.forecast (never reads
                # model_), so no state needs to be swapped in or restored.
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res, cs=cs, level=level,
                    method=self.conformal_params.method)
            else:
                if len(y) - (self.season_length + 6) < 1:
                    raise ValueError(
                        "Native prediction intervals need at least "
                        f"season_length + 7 = {self.season_length + 7} "
                        "observations; pass conformal_params for shorter series."
                    )
                sig = (result['sigma']
                       if (self.error_type == 'A' and self.season_type == 'A')
                       else result['sigma_rel'])
                sigmah = self._calculate_native_intervals(
                    mean, sig, result['alpha'],
                    result['beta'], result['gamma'], phi, h)
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res, result['fitted'], result['sigma'], level, prefix='fitted-')

        return res

    def __init__(
        self,
        season_length: int = 12,
        error_type: str = 'A',
        season_type: str = 'A',
        damped: bool | None = None,
        phi: float | None = None,
        alias: str = "HoltWinters",
        conformal_params: ConformalIntervals | None = None,
    ):
        if not isinstance(season_length, int) or season_length < 2:
            raise ValueError(f"season_length must be an integer >= 2, got {season_length}")
        if error_type not in ('A', 'M'):
            raise ValueError(f"error_type must be 'A' or 'M', got '{error_type}'")
        if season_type not in ('A', 'M'):
            raise ValueError(f"season_type must be 'A' or 'M', got '{season_type}'")
        if phi is not None:
            if not isinstance(phi, (float, int)):
                raise ValueError(f"phi must be None or a number, got {type(phi).__name__}")
            phi = float(phi)
            if not _PHI_LOWER <= phi <= _PHI_UPPER:
                raise ValueError(f"phi must be in range [{_PHI_LOWER}, {_PHI_UPPER}], got {phi}")
        if conformal_params is not None and not isinstance(conformal_params, ConformalIntervals):
            raise ValueError(
                f"conformal_params must be a ConformalIntervals instance, got {type(conformal_params).__name__}"
            )

        self.season_length = season_length
        self.error_type = error_type
        self.season_type = season_type
        self.damped = damped if damped is not None else False
        self.phi = phi
        self.alias = alias
        self.conformal_params = conformal_params

    def conformity_scores(self, y: jnp.ndarray, X: jnp.ndarray | None = None):
        """Walk-forward CV conformity scores, routed by window length.

        The fit is a fixed 150-step Adam + L-BFGS whose zoom linesearch makes
        the CV vmap an anti-optimization on short windows (batched carries and
        both-branch selects), so windows of at most 512 points run
        sequentially. Longer windows amortize the optimizer scan under the
        vmap and use the base path. The 512 bound is an empirical crossover
        between the two regimes, not a mechanism bound.

        The optimizer is trajectory-sensitive: the two regimes reach equally
        valid fit endpoints whose scores differ, so the two paths agree on
        interval calibration rather than bit for bit.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``. Ignored by the model.

        Returns
        -------
        jnp.ndarray
            Signed scores, shape ``(n_windows, h)``.
        """
        if self.conformal_params is not None:
            window_n = jnp.asarray(y).size - self.conformal_params.h
            if window_n <= 512:
                return self._conformity_scores_sequential(y=y, X=X)
        return super().conformity_scores(y=y, X=X)

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> 'HoltWinters':
        r"""Fit the Holt-Winters model.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``, at least ``season_length``
            observations.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``. Ignored.

        Returns
        -------
        HoltWinters
            ``self``.

        Raises
        ------
        ValueError
            If ``y`` has fewer than ``season_length`` observations, or if a
            multiplicative error or seasonality is used with a non-positive
            value in ``y``.
        """
        y = utils.ensure_float(y)
        if len(y) < self.season_length:
            raise ValueError(
                f"Time series must have at least {self.season_length} observations "
                f"(season_length), got {len(y)}"
            )
        if (self.error_type == 'M' or self.season_type == 'M') and bool(jnp.any(y <= 0)):
            raise ValueError(
                "Multiplicative error/seasonality requires strictly positive data."
            )
        result = self._fit_parameters(y)
        result['y_train'] = y
        self.model_ = result
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None,
                level: list[int] | None = None) -> dict:
        r"""Predict with the fitted Holt-Winters model.

        Parameters
        ----------
        h : int
            Forecast horizon (positive int).
        X : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``. Only forwarded to
            :meth:`conformity_scores`; otherwise ignored.
        level : list of int or None, default None
            Confidence levels in [0, 100], e.g. ``[80, 95]``.

        Returns
        -------
        dict
            ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"``/``"hi-{lv}"`` per
            level when ``level`` is given.

        Raises
        ------
        ValueError
            If the model is not fitted, if ``h`` is not a positive int, if
            ``level`` is not a list of numbers in [0, 100], if native
            intervals are requested with fewer than ``season_length + 7``
            training observations, or if the conformal path finds the
            training series too short for 2 CV windows.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict(). Call fit() first.")
        self._validate_h(h)
        self._validate_level(level)

        phi = self._get_phi()
        mean = self._generate_forecasts(
            self.model_['level'], self.model_['trend'],
            self.model_['seasonal'], phi, h)
        res = {'mean': mean}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                cs = self.conformity_scores(y=self.model_['y_train'], X=X)
                res = self.add_confidence_intervals(
                    fcst=res, cs=cs, level=level,
                    method=self.conformal_params.method)
            else:
                n_train = self.model_['residuals'].shape[0]
                if n_train - (self.season_length + 6) < 1:
                    raise ValueError(
                        "Native prediction intervals need at least "
                        f"season_length + 7 = {self.season_length + 7} "
                        "observations; pass conformal_params for shorter series."
                    )
                sig = (self.model_['sigma']
                       if (self.error_type == 'A' and self.season_type == 'A')
                       else self.model_.get('sigma_rel', self.model_['sigma']))
                sigmah = self._calculate_native_intervals(
                    mean, sig, self.model_['alpha'],
                    self.model_['beta'], self.model_['gamma'], phi, h)
                res = self._add_interval_bounds(res, mean, sigmah, level)

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        r"""Return the in-sample fitted values of the fitted model.

        Parameters
        ----------
        level : list of int or None, default None
            Confidence levels in [0, 100].

        Returns
        -------
        dict
            ``"fitted"`` (shape ``(T,)``) plus
            ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` per level when ``level``
            is given.

        Raises
        ------
        ValueError
            If the model is not fitted or if ``level`` is not a list of
            numbers in [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict_in_sample(). Call fit() first.")
        self._validate_level(level)

        res = {'fitted': self.model_['fitted']}
        if level is not None:
            level = sorted(level)
            res = self._add_interval_bounds(
                res, self.model_['fitted'], self.model_['sigma'],
                level, prefix='fitted-')
        return res

    def forecast(self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
                 X_future: jnp.ndarray | None = None, level: list[int] | None = None,
                 fitted: bool = False) -> dict:
        r"""Stateless fit-and-predict in one call.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``, at least ``season_length``
            observations.
        h : int
            Forecast horizon (positive int).
        X : jnp.ndarray or None, default None
            In-sample exogenous regressors, shape ``(T, F)``. Only forwarded
            to :meth:`conformity_scores`; otherwise ignored.
        X_future : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``. Ignored.
        level : list of int or None, default None
            Confidence levels in [0, 100].
        fitted : bool, default False
            Whether to return in-sample fitted values under ``"fitted"``.

        Returns
        -------
        dict
            ``"mean"`` (shape ``(h,)``); ``"fitted"`` (shape ``(T,)``) when
            ``fitted``; ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level``
            is given, plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` when
            both are set.

        Raises
        ------
        ValueError
            If ``y`` has fewer than ``season_length`` observations, if ``h``
            is not a positive int, if ``level`` is not a list of numbers in
            [0, 100], if native intervals are requested with fewer than
            ``season_length + 7`` observations, or if the conformal path
            finds the series too short for 2 CV windows.
        """
        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)

    def forward(self, y: jnp.ndarray, h: int, X: jnp.ndarray | None = None,
                X_future: jnp.ndarray | None = None, level: list[int] | None = None,
                fitted: bool = False) -> dict:
        r"""Apply the fitted model's structure to a new series.

        Re-estimates the parameters on ``y``; the stored ``model_`` is left
        unchanged.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``, at least ``season_length``
            observations.
        h : int
            Forecast horizon (positive int).
        X : jnp.ndarray or None, default None
            In-sample exogenous regressors, shape ``(T, F)``. Only forwarded
            to :meth:`conformity_scores`; otherwise ignored.
        X_future : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``. Ignored.
        level : list of int or None, default None
            Confidence levels in [0, 100].
        fitted : bool, default False
            Whether to return in-sample fitted values under ``"fitted"``.

        Returns
        -------
        dict
            Same keys as :meth:`forecast`.

        Raises
        ------
        ValueError
            If the model is not fitted, plus every condition listed for
            :meth:`forecast`.
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling forward(). Call fit() first.")
        y = self._validate_forecast_inputs(y, h, level)
        result = self._fit_parameters(y)
        phi = self._get_phi()
        return self._compute_forecast_with_intervals(result, phi, h, level, fitted, y, X)
