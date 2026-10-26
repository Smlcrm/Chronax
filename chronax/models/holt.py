"""
Holt's Linear Exponential Smoothing Model (SF-faithful ETS AAN/AAdN).

This module implements Holt's method as the Hyndman *innovations* state-space
ETS(A,A,N) / ETS(A,Ad,N) model, matching ``statsforecast.Holt`` exactly:

    statsforecast `Holt` == `AutoETS(model="AAN", damped=None)` — it fits
    BOTH the undamped model (AAN, φ=1) and the damped model (AAdN,
    φ∈[0.8,0.98]) and returns the one with the lower AICc, per series.

The recursion is the innovations form (NOT the plain component form):

    ŷ_t = l_{t-1} + φ·b_{t-1}
    e_t = y_t − ŷ_t                           (additive error)
    l_t = l_{t-1} + φ·b_{t-1} + α·e_t
    b_t = φ·b_{t-1} + β·e_t

(multiplicative-error variant uses the relative error e_t=(y_t−ŷ_t)/ŷ_t and
l_t=ŷ_t·(1+α·e_t), b_t=φ·b_{t-1}+β·ŷ_t·e_t — the Hyndman MAN/MAdN form).

Objective: concentrated likelihood ``n·log(SSE)`` (``+2·Σ log|ŷ_t|`` for
multiplicative error). Optimised with the repo's vmap/jit-safe Nelder-Mead
(`chronax.utils.utils.nelder_mead`). Initial level/trend come from an
SF-style OLS warm-up (first ``min(10,n)`` points, 1-indexed) and are then
jointly optimised with (α,β[,φ]) — matching statsforecast, which appends
the initial states to its optimisation vector.

The ``damped`` constructor argument gates the candidate set:
- ``damped=False`` → AAN only (pure undamped Holt, φ≡1).
- ``damped=True``  → AAdN only (φ estimated in [0.8,0.98]).
- ``damped=None``  → fit both, branchless min-AICc select (= statsforecast).

Warm-time safety mirrors the ETS backend: a module-level ``lru_cache``'d
``jax.jit`` wrapper keyed only on static scalars, with the data passed as
traced arguments and Nelder-Mead *inside* the jit, so repeated warm calls
reuse the compiled XLA kernel.

Instance Attributes:
1. season_length: int - kept for API consistency (AAN ignores it)
2. error_type: str - 'A' (additive) or 'M' (multiplicative)
3. damped: bool - legacy attr (False unless damped=True); selection is
   driven by the raw constructor arg via _fit_mode
4. phi: float | None - optional initial damping seed (used only for AAdN)
5. alias: str
6. conformal_params: ConformalIntervals | None
7. allow_extended_iterations: bool
8. iteration_scaling: str - accepted for API compat (no-op)
9. model_: dict - fitted params (fitted, level, trend, alpha, beta, phi,
   sigma, residuals, y_train)

Class Attributes:
1. uses_exog: bool - False for Holt

Methods: __init__, fit, predict, predict_in_sample, forecast, forward.
Analytical interval formulas from Hyndman et al. (2008).
"""
import jax
# float64 internals for statsforecast-parity (lstsq init + recursion +
# n·log(SSE) on large-magnitude series). The ETS backend sets the same flag.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import lru_cache
from chronax import utils
from chronax.utils import ConformalIntervals
from chronax.utils.utils import nelder_mead
from chronax.models.base_forecaster import BaseForecaster
from jax import lax

# Damping band (identical to statsforecast AutoETS φ box).
_PHI_LOWER = 0.8
_PHI_UPPER = 0.98

# Smoothing-parameter box (sigmoid-reparametrised), matching SF [1e-4,0.9999].
_PARAM_LOWER = 1e-4
_PARAM_UPPER = 0.9999

# Optimisation init / numerics.
_INIT_ALPHA = 0.3
_INIT_BETA = 0.1
_EPSILON = 1e-10

# Nelder-Mead outer-iteration ceiling (flat). Empirically established
# (full-24 worker sweeps at ceilings 100/200/400, all with
# median(SF−CHX)≈0): (1) the repo NM does NOT meaningfully early-stop on
# the Holt 4–5D AICc surface — warm scales ~linearly with this ceiling;
# (2) more iterations yield no accuracy gain — Holt is model-limited at the
# statsforecast parity wall, not optimiser-budget-limited. So a higher or
# length-adaptive ceiling buys no accuracy and only regresses warm time;
# 100 is the warm-safe sweet spot. The
# `allow_extended_iterations=True` constructor flag is the explicit
# user escape hatch for any pathological series wanting more iters.
_MAX_ITER = 100
_MAX_ITER_EXTENDED = 400

# fit_mode codes (static; drive the candidate set).
_MODE_AAN = 0   # undamped only (φ≡1)
_MODE_AADN = 1  # damped only   (φ∈[_PHI_LOWER,_PHI_UPPER])
_MODE_AUTO = 2  # fit both, branchless min-AICc (= statsforecast Holt)

__all__ = ['Holt']


# =============================================================================
# Module-level numerics (jit/vmap-safe)
# =============================================================================

def _params_from_z(z: jnp.ndarray) -> jnp.ndarray:
    """Unconstrained z → (α, β) ∈ (_PARAM_LOWER, _PARAM_UPPER) via sigmoid."""
    span = _PARAM_UPPER - _PARAM_LOWER
    return _PARAM_LOWER + span * jax.nn.sigmoid(z)


def _z_from_params(ab: jnp.ndarray) -> jnp.ndarray:
    """Inverse of _params_from_z (logit) — for the initial simplex point."""
    span = _PARAM_UPPER - _PARAM_LOWER
    frac = (ab - _PARAM_LOWER) / span
    return jnp.log(frac / (1.0 - frac))


def _holt_recursion(y, l0, b0, alpha, beta, phi, is_additive):
    """Hyndman innovations ETS(A,A,N)/(A,Ad,N) rollout via lax.scan.

    Returns ``(fitted, l_n, b_n, e_obj)`` where ``fitted`` is the one-step
    in-sample prediction ŷ_t, ``(l_n,b_n)`` the final state, and ``e_obj``
    the per-step objective error (raw for additive, relative for
    multiplicative — its SSE is the SF concentrated-likelihood SSE).
    """
    dtype = y.dtype

    def step(carry, y_t):
        l_prev, b_prev = carry
        yhat = l_prev + phi * b_prev
        if is_additive:
            e = y_t - yhat
            l_new = yhat + alpha * e
            b_new = phi * b_prev + beta * e
            e_obj = e
        else:
            denom = jnp.where(jnp.abs(yhat) < _EPSILON,
                              jnp.asarray(_EPSILON, dtype), yhat)
            e = (y_t - yhat) / denom
            l_new = yhat * (1.0 + alpha * e)
            b_new = phi * b_prev + beta * yhat * e
            e_obj = e
        return (l_new, b_new), (yhat, e_obj)

    (l_n, b_n), (fitted, e_obj) = lax.scan(
        step, (jnp.asarray(l0, dtype), jnp.asarray(b0, dtype)), y)
    return fitted, l_n, b_n, e_obj


def _holt_objective(fitted, e_obj, n, is_additive):
    """Concentrated likelihood: n·log(SSE) [+2·Σ log|ŷ| for mult error]."""
    sse = jnp.sum(e_obj ** 2)
    base = n * jnp.log(jnp.maximum(sse, _EPSILON))
    if is_additive:
        return base, sse
    return base + 2.0 * jnp.sum(
        jnp.log(jnp.maximum(jnp.abs(fitted), _EPSILON))), sse


def _fit_one_candidate(y, l0, b0, is_additive, fit_phi, fixed_phi, n_iters):
    """Fit one ETS candidate (AAN if not fit_phi, else AAdN) via Nelder-Mead.

    Jointly optimises (α, β[, φ], l0, b0) — like statsforecast, which
    appends the initial states to the optimisation vector. (l0,b0) use a
    well-conditioned offset reparam ``l0 = l0_ols + s_l·z`` /
    ``b0 = b0_ols + s_b·z`` with ``s_l=max(std(y),ε)``,
    ``s_b=max(std(Δy),ε)`` so all NM coordinates are O(1) and z=0 recovers
    the OLS warm-up exactly (the frozen-init fit stays in the feasible
    set). Returns a result dict with a fixed pytree structure independent
    of ``fit_phi`` (so the two candidates combine branchlessly).
    """
    n = len(y)
    dtype = y.dtype
    pspan = _PHI_UPPER - _PHI_LOWER
    l0 = jnp.asarray(l0, dtype)
    b0 = jnp.asarray(b0, dtype)
    s_l = jnp.maximum(jnp.std(y), _EPSILON)
    s_b = jnp.maximum(jnp.std(jnp.diff(y)), _EPSILON) if n > 1 else jnp.asarray(1.0, dtype)

    def unpack(z):
        ab = _params_from_z(z[:2])
        if fit_phi:
            ph = _PHI_LOWER + pspan * jax.nn.sigmoid(z[2])
            zl, zb = z[3], z[4]
        else:
            ph = jnp.asarray(fixed_phi, dtype)  # AAN→1.0; damped+phi→phi
            zl, zb = z[2], z[3]
        return ab[0], ab[1], ph, l0 + s_l * zl, b0 + s_b * zb

    def loss(z):
        a, b, ph, li, bi = unpack(z)
        fitted, _, _, e_obj = _holt_recursion(y, li, bi, a, b, ph,
                                              is_additive)
        obj, _ = _holt_objective(fitted, e_obj, n, is_additive)
        return obj

    # SINGLE bounded NM from statsforecast's exact `initparam` start —
    # statsforecast does NOT seek the SSE global min (multistart/over-
    # optimisation lands lower-SSE optima that forecast far worse); it
    # runs one bounded Nelder-Mead from this init and early-stops. We
    # mimic that. SF initparam (m=1): α0=αL+0.2(αU−αL),
    # β0=βL+0.1(min(αU,α0)−βL); z-state seed = 0 (OLS warm-up); φ at the
    # band midpoint (phi_seed=0) unless an explicit phi was given.
    a0 = _PARAM_LOWER + 0.2 * (_PARAM_UPPER - _PARAM_LOWER)
    b0_ = _PARAM_LOWER + 0.1 * (min(_PARAM_UPPER, a0) - _PARAM_LOWER)
    z_ab = _z_from_params(jnp.asarray([a0, b0_], dtype))
    z_state = jnp.zeros(2, dtype)  # z=0 → (l0,b0)=OLS warm-up
    if fit_phi:
        # φ seed at the band midpoint (sigmoid(0)).
        z0 = jnp.concatenate([z_ab, jnp.zeros(1, dtype), z_state])
        k = 6  # α, β, φ, l0, b0, σ²
    else:
        z0 = jnp.concatenate([z_ab, z_state])
        k = 5  # α, β, l0, b0, σ²  (φ fixed, not estimated)

    res = nelder_mead(loss, z0, max_iter=int(n_iters),
                      x_tol=1e-7, f_tol=1e-9, initial_simplex_size=0.05)
    a, b, ph, l0f, b0f = unpack(res.x)
    fitted, l_n, b_n, e_obj = _holt_recursion(y, l0f, b0f, a, b, ph,
                                              is_additive)
    obj, sse = _holt_objective(fitted, e_obj, n, is_additive)

    # AICc on the concentrated likelihood (n·log(SSE) ≡ SF up to a constant
    # shared by both candidates, so AICc *differences* are exact).
    nf = jnp.asarray(n, dtype)
    kf = jnp.asarray(k, dtype)
    aic = obj + 2.0 * kf
    denom = nf - kf - 1.0
    aicc = jnp.where(denom > 0.0,
                     aic + 2.0 * kf * (kf + 1.0) / denom,
                     jnp.asarray(jnp.inf, dtype))

    return {
        'fitted': fitted,
        'level': l_n,
        'trend': b_n,
        'residuals': y - fitted,
        'alpha': a,
        'beta': b,
        'phi': ph,
        'sse': sse,
        'aicc': aicc,
    }


def _select_and_fit(y, l0, b0, is_additive, fit_mode, fixed_phi, n_iters):
    """Fit the candidate(s) for ``fit_mode`` and branchlessly select.

    fit_mode ∈ {_MODE_AAN, _MODE_AADN, _MODE_AUTO}.
    - AAN  : φ≡1 (pure undamped Holt).
    - AADN : if an explicit phi was given, the AAdN recursion runs AT that
      fixed φ (states and smoothing parameters are optimised under the
      damping the forecasts will use); else φ is optimised in
      [_PHI_LOWER, _PHI_UPPER].
    - AUTO : fit AAN and free-φ AAdN, keep the lower-AICc one via a pure
      ``jnp.where`` over the (structurally identical) result pytrees —
      fully vmap/jit-safe (= statsforecast Holt).
    """
    if fit_mode == _MODE_AAN:
        return _fit_one_candidate(y, l0, b0, is_additive, False,
                                  1.0, n_iters)
    if fit_mode == _MODE_AADN:
        if fixed_phi is None:
            return _fit_one_candidate(y, l0, b0, is_additive, True,
                                      1.0, n_iters)
        # Explicit fixed φ: the recursion itself runs damped at that φ.
        return _fit_one_candidate(y, l0, b0, is_additive, False,
                                  float(fixed_phi), n_iters)
    # AUTO: fit both (free-φ damped), branchless min-AICc.
    res_u = _fit_one_candidate(y, l0, b0, is_additive, False,
                               1.0, n_iters)
    res_d = _fit_one_candidate(y, l0, b0, is_additive, True,
                               1.0, n_iters)
    pick_d = res_d['aicc'] < res_u['aicc']
    return jax.tree_util.tree_map(
        lambda d, u: jnp.where(pick_d, d, u), res_d, res_u)


@lru_cache(maxsize=128)
def _get_holt_optimizer(is_additive: bool, fit_mode: int,
                        fixed_phi: float | None, n_iters: int):
    """Module-level cached jitted fitter (ETS-style warm pattern).

    Built ONCE per static config ``(is_additive, fit_mode, fixed_phi,
    n_iters)``; the series ``y`` and warm-start ``(l0,b0)`` flow as
    *traced* arguments, so repeated warm calls with the same shapes reuse
    the compiled XLA kernel (no re-trace/recompile). No ``static_argnums``
    on any traced value (φ handling is fully static).
    """
    def _run(y, l0, b0):
        return _select_and_fit(y, l0, b0, is_additive, fit_mode,
                               fixed_phi, n_iters)
    return jax.jit(_run)


class Holt(BaseForecaster):
    def conformity_scores(self, y: jnp.ndarray, X: jnp.ndarray | None = None):
        """Sequential-window CV, overriding the base vmapped path.

        Holt's optimizer fit under the base CV vmap batches every carry and lowers
        the linesearch conds to select, so the same edge-masked windows run faster
        sequentially at every scale — unlike HoltWinters, whose big-m cells amortize
        the vmap. Scores are bit-identical: Holt's small parameter vector does not
        show the trajectory sensitivity HW has. Windows, masking, and values are the
        base implementation's exactly; only the execution regime changes
        (`_conformity_scores_sequential`).
        """
        return self._conformity_scores_sequential(y=y, X=X)

    # Helper methods
    @staticmethod
    def _validate_h(h: int) -> None:
        """Validate forecast horizon parameter."""
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be a positive integer, got {h}")

    @staticmethod
    def _validate_level(level: list[int] | None) -> None:
        """Validate prediction interval level parameter."""
        if level is not None:
            if not isinstance(level, list):
                raise ValueError(f"level must be a list or None, got {type(level).__name__}")
            if any(not isinstance(lv, (int, float)) or lv < 0 or lv > 100 for lv in level):
                raise ValueError("All level values must be numbers between 0 and 100")

    def _initialize_states(self, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """SF-style OLS warm-up for (l0, b0).

        Matches statsforecast ``initstate``: regress y on a constant + a
        **1-indexed** time index over the first ``maxn = min(10, n)``
        observations (m=1 so ``max(10, 2m)=10``). Returns JAX scalars so
        the result is ``jax.vmap`` traceable; the ``n_init >= 2`` branch is
        decided at trace time from the static ``y.shape[0]``.
        """
        n = y.shape[0]
        n_init = min(10, n)

        if n_init >= 2:
            t = jnp.arange(1, n_init + 1, dtype=y.dtype)  # 1-indexed (SF)
            y_init = y[:n_init]

            t_mean = jnp.mean(t)
            y_mean = jnp.mean(y_init)

            cov_ty = jnp.sum((t - t_mean) * (y_init - y_mean))
            var_t = jnp.sum((t - t_mean) ** 2)
            b0 = cov_ty / jnp.maximum(var_t, _EPSILON)
            l0 = y_mean - b0 * t_mean
            return l0, b0

        return y[0], jnp.asarray(0.0, dtype=y.dtype)

    def _fit_parameters(self, y: jnp.ndarray) -> dict:
        """Fit ETS(AAN/AAdN) parameters and return the results dict.

        The candidate set is fixed by ``self._fit_mode`` and the NM ceiling
        by ``self._n_iters`` — both static Python ints — so the cached
        jitted fitter is selected without tracing any data, keeping the
        call ``jax.vmap``-compatible.
        """
        l0, b0 = self._initialize_states(y)
        is_additive = self.error_type == 'A'

        fit = _get_holt_optimizer(is_additive, self._fit_mode,
                                  self.phi, self._n_iters)
        result = fit(y, l0, b0)

        # σ̂ dof matches the reference AAN/AAdN convention (n − 6 undamped,
        # n − 7 damped); |φ−1|<1e-7 ⇒ undamped, branchless so vmap-safe.
        # Multiplicative error uses RELATIVE residuals (the M-error sigma is
        # dimensionless; the interval formula multiplies by |mean| once).
        near1 = jnp.abs(result['phi'] - 1.0) < 1e-7
        k = jnp.where(near1, 6, 7)
        dof = jnp.maximum(len(y) - k, 1)
        if self.error_type == 'M':
            denom_f = jnp.where(
                jnp.abs(result['fitted']) > _EPSILON,
                result['fitted'],
                jnp.asarray(1.0, y.dtype),
            )
            rel_resid = result['residuals'] / denom_f
            result['sigma'] = utils.calculate_sigma(rel_resid, dof)
        else:
            result['sigma'] = utils.calculate_sigma(result['residuals'], dof)
        return result

    def _generate_forecasts(self, level: float, trend: float, phi, h: int) -> jnp.ndarray:
        """h-step damped point forecasts (branchless in φ; vmap/jit-safe):
        ŷ_{n+k} = level + (Σ_{j=1..k} φ^j)·trend, with the φ=1 limit via
        ``where``."""
        dt = level.dtype if hasattr(level, 'dtype') else jnp.float64
        t = jnp.arange(1, h + 1, dtype=dt)
        phi = jnp.asarray(phi, dt)
        near1 = jnp.abs(phi - 1.0) < 1e-7
        denom = jnp.where(near1, jnp.asarray(1.0, dt), 1.0 - phi)
        phi_sum = jnp.where(near1, t, phi * (1.0 - phi ** t) / denom)
        return level + trend * phi_sum

    def _calculate_native_intervals(
        self,
        mean: jnp.ndarray,
        sigma: float,
        alpha: float,
        beta: float,
        phi: float,
        h: int
    ) -> jnp.ndarray:
        """Native prediction-interval width (sigmah), Hyndman et al. (2008).

        Branchless in φ: computes both the damped (AAdN/MAdN) and undamped
        (AAN/MAN) variance and selects with ``where(|φ−1|<1e-7, …)`` so a
        traced/estimated φ stays vmap/jit-safe (the damped vs undamped
        choice follows the *selected model's* φ, not ``self.damped``).
        """
        dt = mean.dtype if hasattr(mean, 'dtype') else jnp.float64
        t = jnp.arange(1, h + 1, dtype=dt)
        phi = jnp.asarray(phi, dt)
        near1 = jnp.abs(phi - 1.0) < 1e-7

        # Undamped (φ=1) variance multiplier.
        exp1 = alpha**2 + alpha * beta * t + (1.0 / 6.0) * beta**2 * t * (2.0 * t - 1.0)
        var_undamped = 1.0 + (t - 1.0) * exp1

        # Damped variance multiplier (guarded denominators).
        denom = jnp.where(near1, jnp.asarray(1.0, dt), 1.0 - phi)
        denom2 = jnp.where(near1, jnp.asarray(1.0, dt), 1.0 - phi**2)
        exp2 = (beta * phi * t) / denom ** 2
        exp3 = 2.0 * alpha * denom + beta * phi
        exp4 = (beta * phi * (1.0 - phi**t)) / (denom ** 2 * denom2)
        exp5 = 2.0 * alpha * denom2 + beta * phi * (1.0 + 2.0 * phi - phi**t)
        var_damped = 1.0 + alpha**2 * (t - 1.0) + exp2 * exp3 - exp4 * exp5

        var = jnp.where(near1, var_undamped, var_damped)
        var = jnp.maximum(var, 0.0)
        if self.error_type == 'A':
            return sigma * jnp.sqrt(var)
        return sigma * jnp.sqrt(var) * jnp.abs(mean)

    def _add_interval_bounds(
        self,
        res: dict,
        values: jnp.ndarray,
        sigmah: jnp.ndarray | float,
        level: list[int],
        prefix: str = ''
    ) -> dict:
        """Add prediction interval bounds to result dictionary."""
        for lv in reversed(level):
            alpha_level = (100 - lv) / 100
            z = utils._jax_norm_ppf(1 - alpha_level / 2)
            res[f'{prefix}lo-{lv}'] = values - z * sigmah
            res[f'{prefix}hi-{lv}'] = values + z * sigmah
        return res

    def __init__(
        self,
        season_length: int = 1,
        error_type: str = 'A',
        damped: bool | None = None,
        phi: float | None = None,
        alias: str = "Holt",
        conformal_params: ConformalIntervals | None = None,
        allow_extended_iterations: bool = False,
        iteration_scaling: str = "quadratic",
    ):
        """
        Holt's linear exponential smoothing (SF-faithful ETS AAN/AAdN).

        Parameters
        ----------
        season_length : int, default=1
            Kept for API consistency (the AAN/AAdN model ignores it).
        error_type : str, default='A'
            'A' (additive) or 'M' (multiplicative). Must be 'A' or 'M'.
        damped : bool | None, default=None
            Gates the candidate set:
            - None  → fit both AAN and AAdN, keep min-AICc (= statsforecast
              ``Holt``; this is the default and the benchmark behaviour).
            - False → AAN only (pure undamped Holt, φ≡1).
            - True  → AAdN only (φ estimated in [0.8, 0.98]).
        phi : float | None, default=None
            Optional initial damping seed (used only when a damped model is
            fitted). Must be in [0.8, 0.98] if given.
        alias : str, default="Holt"
            Custom name for the model.
        conformal_params : ConformalIntervals | None, default=None
            If None, uses native analytical prediction intervals.
        allow_extended_iterations : bool, default=False
            Raise the Nelder-Mead ceiling to 400 (default 200).
        iteration_scaling : str, default="quadratic"
            Accepted for API compatibility; no-op.

        Raises
        ------
        ValueError
            If error_type is not 'A'/'M'; if phi is not a number or is
            outside [0.8, 0.98]; if conformal_params is not a
            ConformalIntervals; if iteration_scaling is invalid.
        """
        if error_type not in ('A', 'M'):
            raise ValueError(
                f"error_type must be 'A' (additive) or 'M' (multiplicative), got '{error_type}'"
            )

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

        if iteration_scaling not in ("cubic", "quadratic"):
            raise ValueError(f"iteration_scaling must be 'cubic' or 'quadratic', got '{iteration_scaling}'")

        self.season_length = season_length
        self.error_type = error_type
        # Legacy attribute (some callers/repr read it); selection itself is
        # driven by _fit_mode derived from the *raw* damped argument so that
        # None (auto) and False (undamped-only) stay distinct.
        self.damped = damped if damped is not None else False
        self.phi = phi
        self.alias = alias
        self.conformal_params = conformal_params
        self.allow_extended_iterations = allow_extended_iterations
        self.iteration_scaling = iteration_scaling

        if damped is None:
            self._fit_mode = _MODE_AUTO   # = statsforecast Holt
        elif damped:
            self._fit_mode = _MODE_AADN
        else:
            self._fit_mode = _MODE_AAN
        # self.phi (float|None) is the fixed damping used only when
        # damped=True with an explicit phi (textbook damped). It is passed
        # as a *static* key to the cached jitted fitter.
        # Flat NM ceiling — a static Python int (lru_cache key), reused
        # across warm calls. allow_extended_iterations raises it (the user
        # escape hatch for a pathological series); see _MAX_ITER comment
        # for why flat (not adaptive) is empirically optimal.
        self._n_iters = _MAX_ITER_EXTENDED if allow_extended_iterations else _MAX_ITER

    def fit(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ) -> 'Holt':
        """Fit the Holt model to training data.

        Estimates (α, β[, φ]) and the level/trend states by minimising the
        concentrated likelihood with Nelder-Mead; selects AAN vs AAdN by
        AICc when ``damped=None``.

        Parameters
        ----------
        y : jnp.ndarray
            Training series of shape (n,). At least 2 observations.
        X : jnp.ndarray | None
            Unused (API consistency).

        Returns
        -------
        self : Holt

        Raises
        ------
        ValueError
            If y has fewer than 2 observations.
        """
        y = utils.ensure_float(y)

        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")
        if self.error_type == 'M' and bool(jnp.any(y <= 0)):
            raise ValueError("Multiplicative error requires strictly positive data.")

        result = self._fit_parameters(y)
        result['y_train'] = y  # Store for conformal prediction
        self.model_ = result
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int] | None = None,
    ) -> dict:
        """
        Predict with fitted Holt model.

        Parameters
        ----------
        h : int
            Forecast horizon (must be positive).
        X : jnp.ndarray, optional
            Unused (API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            'mean' for point predictions and 'lo-{level}'/'hi-{level}' for
            probabilistic predictions.

        Raises
        ------
        ValueError
            If model is not fitted, if h is not positive, or if level
            values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict(). Call fit() first.")

        self._validate_h(h)
        self._validate_level(level)

        phi = self.model_['phi']
        mean = self._generate_forecasts(
            self.model_['level'],
            self.model_['trend'],
            phi,
            h
        )
        res = {'mean': mean}

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                cs = self.conformity_scores(y=self.model_['y_train'], X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    self.model_['sigma'],
                    self.model_['alpha'],
                    self.model_['beta'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

        return res

    def predict_in_sample(
        self,
        level: list[int] | None = None,
    ) -> dict:
        """
        Access fitted Holt model insample predictions.

        Parameters
        ----------
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        dict
            'fitted' for point predictions and 'fitted-lo-{level}'/
            'fitted-hi-{level}' for probabilistic predictions.

        Raises
        ------
        ValueError
            If model is not fitted or if level values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling predict_in_sample(). Call fit() first.")

        self._validate_level(level)

        res = {'fitted': self.model_['fitted']}

        if level is not None:
            level = sorted(level)
            res = self._add_interval_bounds(
                res,
                self.model_['fitted'],
                self.model_['sigma'],
                level,
                prefix='fitted-'
            )

        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Memory efficient Holt predictions (stateless fit-and-predict).

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). At least 2 observations.
        h : int
            Forecast horizon (must be positive).
        X, X_future : jnp.ndarray, optional
            Unused (API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default=False
            Whether to return insample predictions.

        Returns
        -------
        dict
            'mean', optional 'fitted', and 'lo-{level}'/'hi-{level}'.

        Raises
        ------
        ValueError
            If y has fewer than 2 observations, if h is not positive, or
            if level values are outside [0, 100].
        """
        y = utils.ensure_float(y)
        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")
        self._validate_h(h)
        self._validate_level(level)
        result = self._fit_parameters(y)
        phi = result['phi']

        mean = self._generate_forecasts(result['level'], result['trend'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                temp_model = self.model_ if hasattr(self, 'model_') else None
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
                if temp_model is not None:
                    self.model_ = temp_model
                else:
                    delattr(self, 'model_')
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    result['sigma'],
                    result['alpha'],
                    result['beta'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res,
                    result['fitted'],
                    result['sigma'],
                    level,
                    prefix='fitted-'
                )

        return res

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int] | None = None,
        fitted: bool = False,
    ) -> dict:
        """
        Apply the fitted Holt model structure to a new series (re-estimates
        parameters on ``y``).

        Parameters
        ----------
        y : jnp.ndarray
            Clean time series of shape (n,). At least 2 observations.
        h : int
            Forecast horizon (must be positive).
        X, X_future : jnp.ndarray, optional
            Unused (API consistency).
        level : list[int], optional
            Confidence levels (0-100) for prediction intervals.
        fitted : bool, default=False
            Whether to return insample predictions.

        Returns
        -------
        dict
            'mean', optional 'fitted', and 'lo-{level}'/'hi-{level}'.

        Raises
        ------
        ValueError
            If model is not fitted, if y has fewer than 2 observations,
            if h is not positive, or if level values are outside [0, 100].
        """
        if not hasattr(self, 'model_'):
            raise ValueError("Model must be fitted before calling forward(). Call fit() first.")

        y = utils.ensure_float(y)
        if len(y) < 2:
            raise ValueError(f"Time series must have at least 2 observations, got {len(y)}")
        self._validate_h(h)
        self._validate_level(level)
        result = self._fit_parameters(y)
        phi = result['phi']

        mean = self._generate_forecasts(result['level'], result['trend'], phi, h)
        res = {'mean': mean}

        if fitted:
            res['fitted'] = result['fitted']

        if level is not None:
            level = sorted(level)
            if self.conformal_params is not None:
                temp_model = self.model_
                self.model_ = result
                cs = self.conformity_scores(y=y, X=X)
                res = self.add_confidence_intervals(
                    fcst=res,
                    cs=cs,
                    level=level,
                    method=self.conformal_params.method
                )
                self.model_ = temp_model
            else:
                sigmah = self._calculate_native_intervals(
                    mean,
                    result['sigma'],
                    result['alpha'],
                    result['beta'],
                    phi,
                    h
                )
                res = self._add_interval_bounds(res, mean, sigmah, level)

            if fitted:
                res = self._add_interval_bounds(
                    res,
                    result['fitted'],
                    result['sigma'],
                    level,
                    prefix='fitted-'
                )

        return res
