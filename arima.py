# -----------------------------------------------------------------------------
# ARIMA (CSS) — JAX implementation
#
# This implementation estimates ARIMA(p, d, q) via Conditional Sum-of-Squares
# (CSS). CSS minimizes the sum of squared one-step-ahead residuals computed
# from a truncated recursion starting at t = max(p, q). In contrast,
# statsmodels' statespace ARIMA uses a Kalman filter / exact Gaussian MLE.
#
# CONSEQUENCES:
# • CSS vs Kalman can yield different parameter estimates / fitted paths,
#   especially with nontrivial MA terms, short samples, or rough likelihoods.
# • We normalize the working (differenced) series for scale equivariance,
#   optimize in an unconstrained space, and map to constrained (stationary /
#   invertible) parameters via PACF transforms.
# • Fitted values are computed on the working scale, then inverted to the
#   original scale using exact discrete integration (cumulative sums with
#   appropriate “seed” finite differences).
#
# TESTING GUIDANCE:
# • When comparing to statespace (Kalman) fits, prefer shape/quality metrics
#   (correlation, relative RMSE, or MSE-to-data) on the working scale after
#   burn-in, not raw parameter equality—because the objectives differ.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from base_forecaster import BaseForecaster
from conformal_intervals import ConformalIntervals
from utils import ensure_float as _ensure_float

Array = jnp.ndarray


def _difference(y: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Compute the non-seasonal d-th forward difference Δ^d y in float64.

    Using float64 reduces round-off in subsequent inverse transforms.
    For d = 0, returns y (promoted to float64).

    Args
    ----
    y : (n,) array_like
        Original series on the observation scale.
    d : int
        Non-seasonal differencing order.

    Returns
    -------
    work : (n - d,) float64 array
        The d-th forward difference Δ^d y.
    """
    if d <= 0:
        return jnp.asarray(y, dtype=jnp.float64)
    work = jnp.asarray(y, dtype=jnp.float64)
    for _ in range(int(d)):
        work = work[1:] - work[:-1]
    return work


def _inv_difference(last_vals: Array, diffs: Array, d: int) -> Array:
    """
    Invert non-seasonal differencing *in-sample* using exact discrete integration.

    We reconstruct y[d:], given:
      • last_vals = y[:d] on the original scale (used to seed the ladder of
        finite differences at t = d - 1),
      • diffs     = Δ^d y[d:] on the working scale.

    Mathematically, Δ is the forward difference operator. Inversion applies
    cumulative sums d times, each time pre-pending the appropriate seed
    (Δ^k y at the boundary) before cumsum and then discarding the prepended seed.

    Args
    ----
    last_vals : (d,) array_like
        y[0], y[1], ..., y[d-1] on the original scale.
    diffs : (n - d,) array_like
        Δ^d y[d:], the d-th differences for in-sample times.
    d : int
        Differencing order.

    Returns
    -------
    y_rec : (n - d,) float64 array
        Reconstructed y[d:] on the original scale.
    """
    if d == 0:
        return jnp.asarray(diffs, dtype=jnp.float64)

    diffs64 = jnp.asarray(diffs, dtype=jnp.float64)
    lv64 = jnp.asarray(last_vals, dtype=jnp.float64)

    # Build seeds at t = d-1: [y[d-1], Δy[d-1], ..., Δ^{d-1}y[d-1]].
    seeds = []
    tmp = lv64
    for _ in range(d):
        seeds.append(tmp[-1])
        if tmp.shape[0] > 1:
            tmp = tmp[1:] - tmp[:-1]
        else:
            break

    y = diffs64
    # Integrate from highest order down to level 0.
    for k in range(d - 1, -1, -1):
        y = jnp.cumsum(jnp.concatenate([jnp.array([seeds[k]], dtype=y.dtype), y]))[1:]
    return y


def _inv_difference_future(y_hist: Array, diffs: Array, d: int) -> Array:
    """
    Invert differencing *out-of-sample* (future steps) by seeding with the
    ladder of finite differences at the last observed time t = n-1.

    For forecasts x = Δ^d y[n : n+h-1], we iteratively integrate down d times:
      z_d = x
      z_{r-1} = cumsum([Δ^r y_{n-1}] ⨁ z_r)[1:]   for r = d, d-1, ..., 1
    which yields y[n : n+h-1] at r = 0.

    Args
    ----
    y_hist : (n,) array_like
        Full observed series y[0..n-1] (original scale) to compute end seeds.
    diffs : (h,) array_like
        Future Δ^d y values (working scale) to invert.
    d : int
        Differencing order.

    Returns
    -------
    y_future : (h,) float32 array
        Reconstructed future y[n : n+h-1] on original scale.
    """
    if d == 0:
        last = jnp.asarray(y_hist[-1], dtype=jnp.float64)
        x = jnp.asarray(diffs, dtype=jnp.float64)
        out = last + jnp.cumsum(x)
        return out.astype(jnp.float32)

    y64 = jnp.asarray(y_hist, dtype=jnp.float64)
    x = jnp.asarray(diffs, dtype=jnp.float64)

    # Compute [Δ^0 y_{n-1}, Δ^1 y_{n-1}, ..., Δ^{d-1} y_{n-1}] robustly in float64.
    seeds = []
    cur = y64
    for k in range(d):
        if k == 0:
            seeds.append(cur[-1])
        else:
            cur = cur[1:] - cur[:-1]
            seeds.append(cur[-1])

    # Integrate from order d down to 0.
    z = x
    for r in range(d - 1, -1, -1):
        z = jnp.cumsum(jnp.concatenate([jnp.array([seeds[r]], dtype=z.dtype), z]))[1:]
    return z.astype(jnp.float32)


def _arma_residuals_css(
    w: jnp.ndarray,
    phi: jnp.ndarray,
    theta: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute CSS residuals for an ARMA(p, q) on the *working* series w.

    We form one-step-ahead predictions starting at t = m = max(p, q):
        w_hat[t] = c + sum_{i=1..p} phi[i-1] * w[t-i] + sum_{j=1..q} theta[j-1] * e[t-j]
        e[t]     = w[t] - w_hat[t]

    The first m residuals are not defined by this recursion and are dropped.
    This matches the "conditional" nature of CSS.

    Args
    ----
    w : (n,) float32 array
        Working (differenced) series.
    phi : (p,) float32 array
        AR coefficients (stationary).
    theta : (q,) float32 array
        MA coefficients (invertible).
    c : () float32 scalar
        Intercept on the working scale.

    Returns
    -------
    e_valid : (n - m,) float32 array
        Residuals from t = m .. n-1.
    """
    w = jnp.asarray(w, dtype=jnp.float32)
    phi = jnp.asarray(phi, dtype=jnp.float32)
    theta = jnp.asarray(theta, dtype=jnp.float32)
    c = jnp.asarray(c, dtype=jnp.float32)

    p = int(phi.shape[0])
    q = int(theta.shape[0])
    m = max(p, q)
    n = int(w.shape[0])

    e = jnp.zeros((n,), dtype=w.dtype)

    def body(t, e_arr):
        # AR contribution using lagged w
        ar = jnp.asarray(0.0, dtype=w.dtype)
        if p > 0:
            start = t - p
            w_win = lax.dynamic_slice(w, (start,), (p,))[::-1]
            ar = jnp.dot(phi, w_win)

        # MA contribution using lagged residuals
        ma = jnp.asarray(0.0, dtype=w.dtype)
        if q > 0:
            start = t - q
            e_win = lax.dynamic_slice(e_arr, (start,), (q,))[::-1]
            ma = jnp.dot(theta, e_win)

        pred = c + ar + ma
        new_e = w[t] - pred
        e_arr = e_arr.at[t].set(new_e)
        return e_arr

    if n > m:
        e = lax.fori_loop(m, n, body, e)

    return e[m:]


def _pacf_to_ar(pacf: Array) -> Array:
    """
    Convert partial autocorrelations to AR coefficients via Levinson–Durbin.

    If |pacf[k]| < 1 for all k, this construction guarantees a stationary AR
    polynomial (roots strictly outside the unit circle).

    Args
    ----
    pacf : (p,) array_like
        Partial autocorrelations α_1..α_p.

    Returns
    -------
    phi : (p,) float32 array
        Stationary AR coefficients.
    """
    pacf = jnp.asarray(pacf, dtype=jnp.float32)
    p = pacf.shape[0]

    if p == 0:
        return jnp.array([], dtype=jnp.float32)
    if p == 1:
        return pacf

    # Standard Levinson–Durbin recursion
    phi = jnp.zeros(p, dtype=jnp.float32)
    phi = phi.at[0].set(pacf[0])
    for k in range(1, p):
        phi_prev = phi[:k]
        phi = phi.at[k].set(pacf[k])
        phi = phi.at[:k].set(phi_prev - pacf[k] * phi_prev[::-1])
    return phi


def _pack_params(phi_u: Array, theta_u: Array, c_u: float) -> Tuple[Array, Array, float]:
    """
    Map unconstrained parameters to constrained, stable ones.

    • AR:  apply tanh to unconstrained vector, interpret as PACF, transform
            to AR coefficients via Levinson–Durbin → guarantees stationarity.
    • MA:  same trick (PACF-to-AR algebra) yields an invertible MA polynomial.
           (This works because the same recursion enforces roots outside unit circle.)
    • c:   pass-through intercept on the working scale.

    Returns
    -------
    (phi, theta, c) with phi/theta stationary/invertible by construction.
    """
    pacf_phi = jnp.tanh(phi_u)
    phi = _pacf_to_ar(pacf_phi)

    pacf_theta = jnp.tanh(theta_u)
    theta = _pacf_to_ar(pacf_theta)

    c = c_u
    return phi, theta, c


def _nll_css_gaussian(w: Array, phi_u: Array, theta_u: Array, c_u: float) -> float:
    """
    CSS negative log-likelihood under Gaussian errors:
        NLL = (n_eff/2) * [ log(2π σ^2) + 1 ]
    where σ^2 is the mean squared CSS residual and n_eff = n - max(p, q).

    We optimize over unconstrained (phi_u, theta_u, c_u), then map them to
    constrained (phi, theta, c) for residual computation.
    """
    phi, theta, c = _pack_params(phi_u, theta_u, c_u)
    e = _arma_residuals_css(w, phi, theta, c)
    n_eff = e.shape[0]
    var = jnp.mean(e * e) + 1e-12
    nll = 0.5 * n_eff * (jnp.log(2.0 * jnp.pi * var) + 1.0)
    return nll


def _adam(
    grad_f,
    init_params: Tuple[Array, Array, float],
    steps: int = 1500,
    lr: float = 5e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
):
    """
    Lightweight ADAM optimizer (pure JAX) for small parameter vectors.

    Args
    ----
    grad_f : callable
        Function returning gradients w.r.t. (phi_u, theta_u, c_u).
    init_params : tuple
        Initial (phi_u0, theta_u0, c_u0).
    steps : int
        Number of iterations.
    lr : float
        Learning rate.
    b1, b2 : float
        ADAM momentum / RMS decay rates.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    (phi_u, theta_u, c_u) : tuple
        Final unconstrained parameters.
    """
    phi_u, theta_u, c_u = init_params
    m_phi = jnp.zeros_like(phi_u)
    v_phi = jnp.zeros_like(phi_u)
    m_theta = jnp.zeros_like(theta_u)
    v_theta = jnp.zeros_like(theta_u)
    m_c = 0.0
    v_c = 0.0

    def step_fn(state, t):
        phi_u, theta_u, c_u, m_phi, v_phi, m_theta, v_theta, m_c, v_c = state
        g_phi, g_theta, g_c = grad_f(phi_u, theta_u, c_u)

        m_phi = b1 * m_phi + (1 - b1) * g_phi
        v_phi = b2 * v_phi + (1 - b2) * (g_phi * g_phi)
        m_theta = b1 * m_theta + (1 - b1) * g_theta
        v_theta = b2 * v_theta + (1 - b2) * (g_theta * g_theta)
        m_c = b1 * m_c + (1 - b1) * g_c
        v_c = b2 * v_c + (1 - b2) * (g_c * g_c)

        t_f = t + 1.0
        m_phi_hat = m_phi / (1 - b1 ** t_f)
        v_phi_hat = v_phi / (1 - b2 ** t_f)
        m_theta_hat = m_theta / (1 - b1 ** t_f)
        v_theta_hat = v_theta / (1 - b2 ** t_f)
        m_c_hat = m_c / (1 - b1 ** t_f)
        v_c_hat = v_c / (1 - b2 ** t_f)

        phi_u = phi_u - lr * m_phi_hat / (jnp.sqrt(v_phi_hat) + eps)
        theta_u = theta_u - lr * m_theta_hat / (jnp.sqrt(v_theta_hat) + eps)
        c_u = c_u - lr * m_c_hat / (jnp.sqrt(v_c_hat) + eps)

        return (phi_u, theta_u, c_u, m_phi, v_phi, m_theta, v_theta, m_c, v_c), None

    init_state = (phi_u, theta_u, c_u, m_phi, v_phi, m_theta, v_theta, m_c, v_c)
    final_state = lax.scan(step_fn, init_state, jnp.arange(steps))[0]
    phi_u, theta_u, c_u, *_ = final_state
    return phi_u, theta_u, c_u


class ARIMA(BaseForecaster):
    """
    ARIMA(p, d, q) estimated by CSS on the differenced (working) series.

    Design choices:
    • Scale equivariance: optimize on standardized working data ws = w / std(w),
      then map residual variance and intercept back to the original working scale.
    • Stability: unconstrained → constrained via PACF transforms for AR and MA.
    • Fitted values: computed on working scale after burn-in m = max(p, q), then
      inverted in-sample to the original scale with exact discrete integration.
    """
    uses_exog = False

    def __init__(
        self,
        order: Tuple[int, int, int] = (0, 0, 0),
        include_mean: bool = True,
        alias: str = "ARIMA",
        conformal_params: Optional[ConformalIntervals] = None,
        optimizer_steps: int = 1500,
        lr: float = 5e-2,
    ):
        self.p, self.d, self.q = order
        assert self.p >= 0 and self.d >= 0 and self.q >= 0
        self.include_mean = include_mean
        self.alias = alias
        self.conformal_params = conformal_params
        self.optimizer_steps = optimizer_steps
        self.lr = lr
        self.model_ = None

    def fit(self, y: Array, X: Optional[Array] = None):
        """
        Fit ARIMA(p, d, q) parameters via CSS.

        Steps
        -----
        1) Difference y by d to get working series w (float64).
        2) SPECIAL CASE (0,0,0): closed-form mean on working scale.
        3) Otherwise, standardize w → ws for scale equivariance.
        4) Optimize NLL_css(ws; phi_u, theta_u, c_u) by ADAM in unconstrained
           space, then pack to constrained (phi, theta, c_std).
        5) Compute residuals e_std(ws) and rescale intercept / σ^2 to original
           working scale.
        6) Build in-sample fitted on working scale after burn-in, invert to
           original scale with exact integration (NaN prefix length = m + d).

        Stores
        ------
        self.model_ with:
          - 'phi', 'theta', 'c', 'sigma2', 'y_last', 'w', 'fitted', 'burnin'
        """
        y = _ensure_float(y).ravel()
        n = y.shape[0]
        p, d, q = self.p, self.d, self.q
        if n <= max(1, p + q + d + 2):
            raise ValueError("Time series too short for the specified ARIMA order.")

        # 1) Differencing (original scale)
        w = _difference(y, d)

        # 2) Special-case ARMA(0,0): mean-only on working scale
        if p == 0 and q == 0 and self.include_mean:
            phi = jnp.zeros((0,), dtype=w.dtype)
            theta = jnp.zeros((0,), dtype=w.dtype)
            c = float(jnp.mean(w))

            e = _arma_residuals_css(w, phi, theta, c)  # = w - c
            sigma2 = float(jnp.mean(e * e))

            # No burn-in for p=q=0; invert directly to original scale
            w_fitted = w - jnp.concatenate([jnp.zeros((0,), dtype=w.dtype), e])
            y_fitted = _inv_difference(y[:d], w_fitted, d) if d > 0 else w_fitted

            self.model_ = {
                "order": (p, d, q),
                "phi": phi,
                "theta": theta,
                "c": c,
                "sigma2": sigma2,
                "y_last": y,
                "w": w,
                "fitted": y_fitted,
                "burnin": 0,
            }

            if self.conformal_params is not None:
                try:
                    self._cs = self.conformity_scores(y=y, X=None)
                except ValueError:
                    self._cs = jnp.zeros((1,), dtype=jnp.float32)
            else:
                self._cs = None
            return self

        # 3) Standardize working series for scale equivariance
        s = float(jnp.std(w) + 1e-12)
        ws = w / s

        # 4) Initialize and optimize CSS NLL on ws in unconstrained space
        phi_u0 = jnp.zeros((p,), dtype=ws.dtype)
        theta_u0 = jnp.zeros((q,), dtype=ws.dtype)
        c_u0 = float(jnp.clip(jnp.mean(ws), -1.0, 1.0)) if self.include_mean else 0.0

        def obj(phi_u, theta_u, c_u):
            return _nll_css_gaussian(ws, phi_u, theta_u, c_u)

        grad_f = jax.grad(lambda ph, th, c: obj(ph, th, c), argnums=(0, 1, 2))
        phi_u, theta_u, c_u = _adam(
            grad_f, (phi_u0, theta_u0, c_u0),
            steps=self.optimizer_steps, lr=self.lr
        )
        phi, theta, c_std = _pack_params(phi_u, theta_u, c_u)

        # 5) Residuals on standardized scale → rescale c and σ^2
        e_std = _arma_residuals_css(ws, phi, theta, c_std)
        c = float(c_std * s)
        sigma2 = float(jnp.mean(e_std * e_std) * (s ** 2))

        # 6) In-sample fitted on working scale with NaN prefix of length m
        m = max(p, q)
        ws_hat_valid = ws[m:] - e_std               # defined for t >= m
        ws_fitted = jnp.concatenate([jnp.full((m,), jnp.nan, dtype=ws.dtype), ws_hat_valid])
        w_fitted = ws_fitted * s

        # Invert to original scale; pad first (m+d) as NaN
        if d > 0:
            w_valid = w_fitted[m:]
            y_valid = _inv_difference(y[:d], w_valid, d)
            y_fitted = jnp.concatenate([jnp.full((m + d,), jnp.nan, dtype=y_valid.dtype), y_valid])
        else:
            y_fitted = w_fitted

        self.model_ = {
            "order": (p, d, q),
            "phi": phi,
            "theta": theta,
            "c": float(c) if self.include_mean else 0.0,
            "sigma2": sigma2,
            "y_last": y,
            "w": w,
            "fitted": y_fitted,
            "burnin": m,
        }

        if self.conformal_params is not None:
            try:
                self._cs = self.conformity_scores(y=y, X=None)
            except ValueError:
                self._cs = jnp.zeros((1,), dtype=jnp.float32)
        else:
            self._cs = None

        return self

    @staticmethod
    def _arma_mean_forecast(
        w_hist: Array,
        e_hist: Array,
        phi: Array,
        theta: Array,
        c: float,
        h: int,
    ) -> Array:
        """
        Deterministic mean forecast on working scale for ARMA(p, q).

        We roll forward h steps using:
            w_{t+1|t} = c + φ(1) w_t + ... + φ(p) w_{t+1-p} + θ(1) e_t + ... + θ(q) e_{t+1-q}
        with future residuals set to 0 by convention (mean forecast).

        Args
        ----
        w_hist : (k,) array
            Last k observations of w; we use k = max(1, p).
        e_hist : (q,) array
            Residuals ordered [e_{t}, e_{t-1}, ..., e_{t-q+1}] (most recent first).
        phi, theta, c : arrays / scalar
            Model parameters on working scale.
        h : int
            Forecast horizon on the working scale.

        Returns
        -------
        w_fcst : (h,) float32 array
            Mean forecasts Δ^d y for the next h steps.
        """
        p = phi.shape[0]
        q = theta.shape[0]

        ws = list(w_hist)
        e_buf = e_hist

        out = []
        for _ in range(h):
            ar_sum = 0.0
            if p > 0:
                ar_lags = jnp.array([ws[-(i + 1)] for i in range(p)])
                ar_sum = jnp.dot(phi, ar_lags)

            ma_sum = 0.0
            if q > 0:
                ma_sum = jnp.dot(theta, e_buf[:q])

            w_next = c + ar_sum + ma_sum
            out.append(w_next)

            ws.append(w_next)
            if q > 0:
                # shift residual buffer right and set newest (future) residual to 0
                e_buf = jnp.roll(e_buf, 1)
                e_buf = e_buf.at[0].set(0.0)

        return jnp.stack(out)

    def predict(self, h: int, X: Optional[Array] = None, level: Optional[List[int]] = None) -> Dict[str, Array]:
        """
        Predict h future observations using fitted model parameters.

        Forecast pipeline (working → original):
          1) Build w-history and residual history from stored artifacts.
          2) Roll forward mean forecasts on working scale.
          3) Invert differencing with end seeds to get original-scale forecasts.

        Returns
        -------
        {'mean': (h,) float32 array} or with intervals if conformal is enabled.
        """
        if getattr(self, "model_", None) is None:
            raise RuntimeError("ARIMA model is not fitted. Call fit(y) first.")
        if h <= 0:
            return {"mean": jnp.zeros((0,), dtype=jnp.float32)}

        p, d, q = self.model_["order"]
        phi = self.model_["phi"]
        theta = self.model_["theta"]
        c = self.model_["c"]
        y = self.model_["y_last"]
        w = self.model_["w"]

        # Residuals on differenced scale (CSS recursion)
        e_full = _arma_residuals_css(w, phi, theta, c)
        k = max(1, p)
        w_hist = w[-k:]
        if q > 0:
            e_recent = e_full[-q:] if e_full.shape[0] >= q else jnp.pad(e_full, (q - e_full.shape[0], 0))
            e_hist = jnp.flip(e_recent)  # newest first
        else:
            e_hist = jnp.zeros((0,), dtype=w.dtype)

        w_fcst = self._arma_mean_forecast(w_hist, e_hist, phi, theta, c, h=h)
        mean = _inv_difference_future(y, w_fcst, d) if d > 0 else w_fcst

        res = {"mean": mean.astype(jnp.float32)}
        if level is None:
            return res
        if self.conformal_params is None:
            raise ValueError("You must pass `conformal_params` to compute intervals.")
        if getattr(self, "_cs", None) is None:
            raise ValueError("Conformity scores are not available. Fit with conformal_params to cache them.")
        return BaseForecaster.add_confidence_intervals(
            fcst=res, cs=self._cs, level=sorted(level), method=self.conformal_params.method
        )

    def forecast(
        self,
        y: Array,
        h: int,
        X: Optional[Array] = None,
        X_future: Optional[Array] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, Array]:
        """
        Stateless variant: fit-on-the-fly and return forecasts (and optionally
        the in-sample fitted values) without mutating self.model_.

        The pipeline mirrors `fit()` + `predict()`:
          • difference y → w, standardize ws, optimize CSS NLL on ws,
            map to (phi, theta, c), and forecast on working scale;
          • invert differencing for original-scale forecast;
          • if fitted=True, also return in-sample fitted with NaN prefix of length m + d.

        Returns
        -------
        {'mean': (h,) float32, ['fitted': (n,) float32 if requested], ...}
        plus conformal intervals if configured and `level` is provided.
        """
        # Prepare data
        y = _ensure_float(y).ravel()
        p, d, q = self.p, self.d, self.q
        if y.shape[0] <= max(1, p + q + d + 2):
            raise ValueError("Time series too short for the specified ARIMA order.")

        w = _difference(y, d)

        # Special-case ARMA(0,0)
        if p == 0 and q == 0 and self.include_mean:
            phi = jnp.zeros((0,), dtype=w.dtype)
            theta = jnp.zeros((0,), dtype=w.dtype)
            c = float(jnp.mean(w))
            e_full = _arma_residuals_css(w, phi, theta, c)
            k = 1
            w_hist = w[-k:]
            e_hist = jnp.zeros((0,), dtype=w.dtype)
            w_fcst = self._arma_mean_forecast(w_hist, e_hist, phi, theta, c, h=h)
            mean = _inv_difference_future(y, w_fcst, d) if d > 0 else w_fcst
            res = {"mean": mean.astype(jnp.float32)}
            if fitted:
                w_fitted = w - jnp.concatenate([jnp.zeros((0,), dtype=w.dtype), e_full])
                y_fitted = _inv_difference(y[:d], w_fitted, d) if d > 0 else w_fitted
                res["fitted"] = y_fitted.astype(jnp.float32)
            if level is None:
                return res
            if self.conformal_params is None:
                raise ValueError("You must pass `conformal_params` to compute intervals.")
            cs = self.conformity_scores(y=y, X=None)
            return BaseForecaster.add_confidence_intervals(
                fcst=res, cs=cs, level=sorted(level), method=self.conformal_params.method
            )

        # Normalize for scale-equivariant estimation
        s = float(jnp.std(w) + 1e-12)
        ws = w / s

        # Optimize CSS NLL on ws
        phi_u0 = jnp.zeros((p,), dtype=ws.dtype)
        theta_u0 = jnp.zeros((q,), dtype=ws.dtype)
        c_u0 = float(jnp.clip(jnp.mean(ws), -1.0, 1.0)) if self.include_mean else 0.0

        def obj(phi_u, theta_u, c_u):
            return _nll_css_gaussian(ws, phi_u, theta_u, c_u)

        grad_f = jax.grad(lambda ph, th, c: obj(ph, th, c), argnums=(0, 1, 2))
        phi_u, theta_u, c_u = _adam(grad_f, (phi_u0, theta_u0, c_u0), steps=self.optimizer_steps, lr=self.lr)
        phi, theta, c_std = _pack_params(phi_u, theta_u, c_u)

        # Residuals on standardized scale; rescale intercept
        e_std = _arma_residuals_css(ws, phi, theta, c_std)
        c = float(c_std * s)

        # Build histories for forecasting on working scale
        k = max(1, p)
        w_hist = w[-k:]
        if q > 0:
            e_recent = e_std[-q:] if e_std.shape[0] >= q else jnp.pad(e_std, (q - e_std.shape[0], 0))
            e_hist = jnp.flip(e_recent)
        else:
            e_hist = jnp.zeros((0,), dtype=ws.dtype)

        w_fcst_std = self._arma_mean_forecast(w_hist / s, e_hist, phi, theta, c_std, h=h)
        w_fcst = w_fcst_std * s

        mean = _inv_difference_future(y, w_fcst, d) if d > 0 else w_fcst
        res = {"mean": mean.astype(jnp.float32)}

        if fitted:
            # In-sample fitted in the same manner as fit()
            m = max(p, q)
            w_hat_valid_std = ws[m:] - e_std
            w_fitted_std = jnp.concatenate([jnp.full((m,), jnp.nan, dtype=ws.dtype), w_hat_valid_std])
            w_fitted = w_fitted_std * s

            if d > 0:
                w_valid = w_fitted[m:]
                y_valid = _inv_difference(y[:d], w_valid, d)
                y_fitted = jnp.concatenate([jnp.full((m + d,), jnp.nan, dtype=jnp.float32), y_valid])
            else:
                y_fitted = w_fitted

            res["fitted"] = y_fitted.astype(jnp.float32)

        if level is None:
            return res

        if self.conformal_params is None:
            raise ValueError("You must pass `conformal_params` to compute intervals.")
        cs = self.conformity_scores(y=y, X=None)
        return BaseForecaster.add_confidence_intervals(
            fcst=res, cs=cs, level=sorted(level), method=self.conformal_params.method
        )
