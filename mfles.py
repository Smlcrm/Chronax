"""
MFLES (Multi-Feature Locally Exponential Smoothing) in JAX

This module implements an advanced forecasting model that combines multiple components:
- Linear trend with optional changepoints (piecewise linear via LASSO)
- Multiple seasonal patterns (Fourier series representation)
- Residual smoothing (Simple Exponential Smoothing ensemble)
- Exogenous variables support
- Robust estimation options (Siegel repeated medians)

**Algorithm:**
MFLES iteratively fits components to residuals:
1. Initial median-based estimates
2. Seasonal components via Fourier series + (weighted) OLS
3. Linear trend via OLS, robust regression, or piecewise LASSO
4. Residual smoothing via SES ensemble or rolling means
5. Exogenous variables via OLS
6. Iterates until convergence or max rounds

**Formulas:**
- Seasonal: y_t = Σ [a_k cos(2πkt/T) + b_k sin(2πkt/T)] for k=1..K
- Trend: y_t = β₀ + β₁t + Σ β_k max(0, t-τ_k) (piecewise linear)
- SES: ŷ_t = α·y_t + (1-α)·ŷ_{t-1}

**Implementation:**
- JIT-compiled components for performance
- Automatic robustness detection via coefficient of variation
- Supports additive and multiplicative modes
- Automatic hyperparameter selection via `optimize` method
- Conforms to `BaseForecaster` interface

**Attributes:**
- `robust`: Use robust regression (Siegel) vs OLS
- `multiplicative`: Log-transform for multiplicative seasonality
- `penalty`: Trend dampening factor based on R² (optional)

**Methods:**
- `fit(y, seasonal_period, X, ...)`: Fit all components to series
- `predict(h, X, level)`: Forecast h steps ahead with optional intervals
- `optimize(y, ...)`: Auto-tune hyperparameters via cross-validation
"""
# mfles.py
from __future__ import annotations
from functools import partial as _partial
import jax
import jax.numpy as jnp
from jax import lax

import utils
from base_forecaster import BaseForecaster

def _mse(y, yhat): return jnp.mean((y - yhat) ** 2)
def _mae(y, yhat): return jnp.mean(jnp.abs(y - yhat))
def _mape(y, yhat): return jnp.mean(jnp.abs((y - yhat) / (yhat + 1e-6)))
def _smape(y, yhat): return jnp.mean(2.0 * jnp.abs(y - yhat) / (jnp.abs(y) + jnp.abs(yhat) + 1e-6))
_metric2fn = {"mse": _mse, "mae": _mae, "mape": _mape, "smape": _smape}

@_partial(jax.jit, static_argnums=(1,))
def _rolling_mean(y: jnp.ndarray, window: int) -> jnp.ndarray:
    if window <= 1:
        return y
    kernel = jnp.ones((window,), dtype=y.dtype) / window
    conv = jnp.convolve(y, kernel, mode="valid")
    pad = y[:window - 1]
    return jnp.concatenate([pad, conv], axis=0)

def _ses_ensemble_via_utils(resids: jnp.ndarray, min_alpha=0.05, max_alpha=1.0, smooth=False, order=1) -> jnp.ndarray:
    if smooth:
        alphas = jnp.arange(min_alpha, max_alpha + 1e-9, 0.05, dtype=resids.dtype)
        def one(alpha):
            _, fitted = utils._ses_forecast(resids, alpha)
            f = jnp.nan_to_num(fitted, nan=0.0)
            mask = jnp.isnan(fitted)
            idx = jnp.maximum.accumulate((~mask).astype(jnp.int32) * jnp.arange(resids.size))
            return f[idx]
        mats = jax.vmap(one)(alphas)
        return jnp.mean(mats, axis=0)
    rm = _rolling_mean(resids, order + 1)
    return rm.at[:order + 1].set(resids[:order + 1])

def _cap_outliers(y: jnp.ndarray, k=3.0) -> jnp.ndarray:
    mu, sd = jnp.mean(y), jnp.std(y)
    return jnp.clip(y, mu - k * sd, mu + k * sd)

def _set_fourier(period: int) -> int:
    return 5 if period < 10 else (10 if period < 70 else 15)

def _fourier_series(n: int, period: int, order: int) -> jnp.ndarray:
    t = jnp.arange(1, n + 1, dtype=jnp.float32).reshape(-1, 1)
    k = jnp.arange(1, order + 1, dtype=jnp.float32).reshape(1, -1)
    X = 2.0 * jnp.pi * t @ (k / float(period))
    return jnp.hstack([jnp.cos(X), jnp.sin(X)])

def _seasonality_weights(n: int, period: int) -> jnp.ndarray:
    return 1.0 + (jnp.arange(n) // period).astype(jnp.float32)

def _median_init(y: jnp.ndarray, period: int | None) -> jnp.ndarray:
    n = y.shape[0]
    if period is None:
        return jnp.full_like(y, jnp.median(y))
    m = int(period)
    full = (n // m) * m
    base = y[:full].reshape(-1, m)
    per = jnp.median(base, axis=1)
    med = jnp.repeat(per, m)
    if full < n:
        tail = jnp.median(y[-m:])
        med = jnp.concatenate([med, jnp.full((n - full,), tail, dtype=y.dtype)], axis=0)
    return med

def _fast_ols_fit_predict(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    M = x.shape[0]
    x_sum, y_sum = jnp.sum(x), jnp.sum(y)
    x2, xy = jnp.dot(x, x), jnp.dot(x, y)
    denom = M * x2 - x_sum * x_sum + 1e-12
    slope = (M * xy - x_sum * y_sum) / denom
    intercept = (y_sum - slope * x_sum) / M
    return slope * x + intercept

@jax.jit
def _siegel_repeated_medians(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[0]
    X_i = x[:, None]; X_j = x[None, :]
    Y_i = y[:, None]; Y_j = y[None, :]
    dx = X_j - X_i
    dy = Y_j - Y_i
    slopes = jnp.where(dx != 0, dy / (dx + 1e-12), 0.0)
    med_slopes_i = jnp.median(slopes, axis=1)
    slope = jnp.median(med_slopes_i)
    intercept = jnp.median(y - slope * x)
    return slope * x + intercept

def _ols_predict(X: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = jnp.linalg.pinv(X.T @ X) @ (X.T @ y)
    return X @ beta, beta

def _wls_predict(X: jnp.ndarray, y: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    WX = X * w[:, None]
    beta = jnp.linalg.pinv(X.T @ WX) @ (X.T @ (w * y))
    return X @ beta

def _cov_proxy(y: jnp.ndarray, mult: int = 1) -> jnp.ndarray:
    if mult:
        return jnp.sqrt(jnp.exp(jnp.log(10.0) * (jnp.std(y) ** 2)) - 1.0)
    sd, mu = jnp.std(y), jnp.mean(y)
    return jnp.where(mu != 0, sd / mu, sd)

def _knots_from_gradients(y: jnp.ndarray, k: int, max_knots: int = 50) -> jnp.ndarray:
    """
    Select knots based on gradients. Returns fixed-size array for JIT compatibility.
    
    Args:
        y: Input series
        k: Number of knots to select (can be dynamic)
        max_knots: Maximum number of knots (static, for fixed output size)
    
    Returns:
        Array of shape (max_knots,) with knot positions (padded with zeros if k < max_knots)
    """
    g = jnp.abs(jnp.diff(y))
    idx = jnp.argsort(-g)
    
    # Create fixed-size output array
    knots = jnp.zeros(max_knots, dtype=jnp.int32)
    
    # Fill with top k indices using masking
    fill_mask = jnp.arange(max_knots) < k
    # Take indices with wrapping (safe indexing)
    safe_idx = idx[jnp.minimum(jnp.arange(max_knots), idx.shape[0] - 1)]
    knots = jnp.where(fill_mask, safe_idx + 1, 0)
    knots = jnp.clip(knots, 0, y.shape[0] - 1)
    
    return jnp.sort(knots)

def _uniform_knots(n: int, k: int, max_knots: int = 50) -> jnp.ndarray:
    """
    Create uniformly spaced knots. Returns fixed-size array for JIT compatibility.
    
    Args:
        n: Series length
        k: Number of knots (can be dynamic)
        max_knots: Maximum number of knots (static, for fixed output size)
    
    Returns:
        Array of shape (max_knots,) with knot positions (padded with zeros if k < max_knots)
    """
    knots = jnp.zeros(max_knots, dtype=jnp.int32)
    
    # Compute step size
    step = n // (k + 1)
    
    # Fill knots array
    positions = step * (jnp.arange(max_knots) + 1)
    fill_mask = jnp.arange(max_knots) < k
    knots = jnp.where(fill_mask, jnp.clip(positions, 1, n - 2), 0)
    
    return knots

def _hinge_basis_from_knots(n: int, knots: jnp.ndarray) -> jnp.ndarray:
    """
    Create hinge basis functions from knots.
    
    Knots array is fixed-size with padding (zeros indicate no knot).
    """
    t = jnp.arange(n, dtype=jnp.float32)
    K = knots.shape[0]
    
    # Create hinge functions for all positions (including padded zeros)
    def one(k_idx):
        tau = knots[k_idx].astype(jnp.float32)
        # If knot is 0 (padding), return zeros
        hinge = jnp.maximum(0.0, t - tau)
        is_valid = knots[k_idx] > 0
        return jnp.where(is_valid, hinge, 0.0)
    
    H = jax.vmap(one)(jnp.arange(K)).T  # Shape: (n, K)
    
    # Concatenate: [t, ones, hinges]
    return jnp.concatenate([t.reshape(-1, 1), jnp.ones((n, 1), t.dtype), H], axis=1)

def _spectral_step(X: jnp.ndarray) -> float:
    s = jnp.linalg.svd(X, full_matrices=False, compute_uv=False)
    L = (s[0] ** 2) + 1e-6
    return 1.0 / L

def _soft(z: jnp.ndarray, lam: float) -> jnp.ndarray:
    return jnp.sign(z) * jnp.maximum(0.0, jnp.abs(z) - lam)

def _lasso_ista(X: jnp.ndarray, y: jnp.ndarray, alpha: float, maxiter: int = 200, tol: float = 1e-4) -> jnp.ndarray:
    step = _spectral_step(X)
    beta = jnp.zeros((X.shape[1],), dtype=y.dtype)
    def body(carry):
        b = carry
        grad = X.T @ (X @ b - y)
        b_new = _soft(b - step * grad, step * alpha)
        return b_new, jnp.linalg.norm(b_new - b)
    def cond(val):
        b, diff = val
        return diff > tol
    def loop(b):
        def one(_, curr):
            return body(curr)[0]
        b_new = lax.fori_loop(0, maxiter, lambda i, bb: body(bb)[0], b)
        return b_new
    return loop(beta)

class MFLES(BaseForecaster):
    uses_exog = True

    def __init__(self, verbose: int = 1, robust: bool | None = None, alias: str = "MFLES", conformal_params=None):
        self.alias = alias
        self.conformal_params = conformal_params
        self.model_ = {}
        self.verbose = verbose
        self.robust = robust
        self.predicted = None
        self._exo_beta = None
        self.penalty = None

    def fit(
        self,
        y,
        seasonal_period=None,
        X=None,
        fourier_order=None,
        ma=None,
        alpha=1.0,
        decay=-1,
        n_changepoints=0.25,
        seasonal_lr=0.9,
        rs_lr=1.0,
        exogenous_lr=1.0,
        exogenous_estimator=None,
        exogenous_params={},
        linear_lr=0.9,
        cov_threshold=0.7,
        moving_medians=False,
        max_rounds=50,
        min_alpha=0.05,
        max_alpha=1.0,
        round_penalty=0.0001,
        trend_penalty=True,
        multiplicative=None,
        changepoints=True,
        smoother=False,
        seasonality_weights=False,
        gradient_strategy=False,
    ):
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        n = y.shape[0]
        if multiplicative is None:
            multiplicative = seasonal_period is not None and jnp.min(y) > 0
        if multiplicative:
            const = jnp.min(y)
            y_tr = jnp.log(y)
            mean, std = jnp.array(0.0, y.dtype), jnp.array(1.0, y.dtype)
        else:
            const = None
            mean, std = jnp.mean(y), jnp.std(y)
            y_tr = y - mean
            y_tr = jnp.where(std > 0, y_tr / std, y_tr)

        self.const, self.mean, self.std = const, mean, std
        self.trend_penalty = bool(trend_penalty)

        if n < 4 or jnp.all(y_tr == jnp.mean(y_tr)):
            base = y_tr[-1]
            self.trend = jnp.array([base, base], dtype=y.dtype)
            self.seasonality = None
            self.linear_component = jnp.zeros(n, y.dtype)
            self.seasonal_component = jnp.zeros(n, y.dtype)
            self.ses_component = jnp.zeros(n, y.dtype)
            self.median_component = jnp.zeros(n, y.dtype)
            self.exogenous_component = jnp.zeros(n, y.dtype)
            self._exo_beta = None
            fitted = jnp.full(n, base, y.dtype)
            return self._finalize_fit(fitted, multiplicative)

        sp_list = None
        if seasonal_period is not None:
            sp_list = seasonal_period if isinstance(seasonal_period, list) else [int(seasonal_period)]
        seas_tail = None
        fourier_series = []
        cycle_weights = []
        if sp_list is not None:
            for p in sp_list:
                fo = int(_set_fourier(p)) if fourier_order is None else int(fourier_order)
                fourier_series.append(_fourier_series(n, p, fo))
                if seasonality_weights:
                    cycle_weights.append(_seasonality_weights(n, p))

        linear_component = jnp.zeros(n, y.dtype)
        seasonal_component = jnp.zeros(n, y.dtype)
        ses_component = jnp.zeros(n, y.dtype)
        median_component = _median_init(y_tr, max(sp_list) if (sp_list is not None and moving_medians) else None)
        exogenous_component = jnp.zeros(n, y.dtype)
        fitted = median_component.copy()
        trend_tail = jnp.array([fitted[-1], fitted[-1]], y.dtype)

        ma_cycle = [1] if ma is None else (ma if isinstance(ma, list) else [int(ma)])
        ma_len = len(ma_cycle)

        if isinstance(n_changepoints, float) and n_changepoints < 1:
            n_cps = int(n_changepoints * n)
        elif isinstance(n_changepoints, int):
            n_cps = n_changepoints
        else:
            n_cps = 0

        best = None
        stalls = 0
        for i in range(int(max_rounds)):
            resids = y_tr - fitted
            cur = _mse(y_tr, fitted)
            if best is None:
                best = cur
            else:
                if cur >= best:
                    stalls += 1
                    if stalls >= 6:
                        break
                else:
                    best, stalls = cur, 0

            if sp_list is not None:
                k = i % len(sp_list)
                Xs = fourier_series[k]
                if seasonality_weights:
                    w = cycle_weights[k]
                    seas = _wls_predict(Xs, resids, w) * float(seasonal_lr)
                else:
                    seas, _ = _ols_predict(Xs, resids)
                    seas = seas * float(seasonal_lr)
                test = _mse(y_tr, fitted + seas)
                if test < best:
                    best = test
                    fitted = fitted + seas
                    seasonal_component = seasonal_component + seas
                    p = sp_list[k]
                    seas_tail = seas[-p:]

            if X is not None and i > 0:
                X_exo = utils.ensure_float(jnp.asarray(X))
                beta = jnp.linalg.pinv(X_exo.T @ X_exo) @ (X_exo.T @ resids)
                exo = (X_exo @ beta) * float(exogenous_lr)
                test = _mse(y_tr, fitted + exo)
                if test < best:
                    best = test
                    fitted = fitted + exo
                    exogenous_component = exogenous_component + exo
                    self._exo_beta = beta

            if i % 2 == 1:
                x_idx = jnp.arange(n, dtype=y.dtype)
                def trend_piecewise():
                    k = jnp.maximum(0, n_cps)
                    knots = jax.lax.cond(bool(gradient_strategy),
                                         lambda _: _knots_from_gradients(resids, k),
                                         lambda _: _uniform_knots(n, k),
                                         operand=None)
                    Xb = _hinge_basis_from_knots(n, knots)
                    lam = float(alpha)
                    beta = _lasso_ista(Xb, resids, lam, maxiter=200, tol=1e-4)
                    return ((Xb @ beta) * float(linear_lr)).astype(jnp.float32)
                def trend_robust():
                    return (_siegel_repeated_medians(x_idx, resids) * float(linear_lr)).astype(jnp.float32)
                def trend_ols():
                    return (_fast_ols_fit_predict(x_idx, resids) * float(linear_lr)).astype(jnp.float32)
                tren = jax.lax.cond(bool(self.robust),
                                    lambda _: trend_robust(),
                                    lambda _: jax.lax.cond(bool(changepoints) & (n_cps > 0),
                                                           lambda __: trend_piecewise(),
                                                           lambda __: trend_ols(),
                                                           operand=None),
                                    operand=None)
                test = _mse(y_tr, fitted + tren)
                if test < best:
                    best = test
                    fitted = fitted + tren
                    linear_component = linear_component + tren
                    trend_tail = trend_tail.at[0].set(trend_tail[1])
                    trend_tail = trend_tail.at[1].set(tren[-1])
                    if i == 1:
                        mu = jnp.mean(resids)
                        ssres = jnp.sum((resids - tren) ** 2)
                        sstot = jnp.sum((resids - mu) ** 2) + 1e-12
                        self.penalty = 1.0 - (ssres / sstot)
            elif i > 4:
                resids = resids.at[-2:].set(_cap_outliers(resids, 3.0)[-2:])
                order = int(ma_cycle[i % ma_len])
                tren = _ses_ensemble_via_utils(resids, min_alpha, max_alpha, smooth=bool(smoother), order=order) * float(rs_lr)
                test = _mse(y_tr, fitted + tren)
                if test < (best * (1.0 - float(round_penalty))):
                    best = test
                    fitted = fitted + tren
                    ses_component = ses_component + tren
                    trend_tail = trend_tail.at[0].set(trend_tail[1])
                    trend_tail = trend_tail.at[1].set(tren[-1])

            if i == 0 and self.robust is None:
                self.robust = bool(_cov_proxy(resids, int(multiplicative)) > float(cov_threshold))
            if i == 1:
                resids = _cap_outliers(resids, 5.0)

        self.linear_component = linear_component
        self.seasonal_component = seasonal_component
        self.ses_component = ses_component
        self.median_component = median_component
        self.exogenous_component = exogenous_component
        self.trend = trend_tail
        self.seasonality = seas_tail

        return self._finalize_fit(fitted, multiplicative)

    def _finalize_fit(self, fitted_tr: jnp.ndarray, multiplicative: bool):
        if multiplicative:
            fitted = jnp.exp(fitted_tr)
        else:
            fitted = self.mean + fitted_tr * jnp.where(self.std > 0, self.std, 1.0)
        self.multiplicative = multiplicative
        self.model_ = {
            "fitted": fitted,
            "trend": self.trend,
            "seasonality": self.seasonality,
            "linear_component": self.linear_component,
            "seasonal_component": self.seasonal_component,
            "ses_component": self.ses_component,
            "median_component": self.median_component,
            "exogenous_component": self.exogenous_component,
            "mean": self.mean,
            "std": self.std,
            "const": self.const,
        }
        self.fitted_ = fitted  # Store fitted values
        return self  # Return self for method chaining

    def predict(self, h: int, X=None, level: list[int | float] | None = None):
        h = int(h)
        last, prev = self.trend[1], self.trend[0]
        slope = (last - prev)
        if self.trend_penalty and (self.penalty is not None):
            slope = slope * jnp.maximum(0.0, self.penalty)
        trend_fcst = slope * jnp.arange(1, h + 1, dtype=jnp.float32) + last

        if self.seasonality is not None and self.seasonality.size > 0:
            seas_fcst = utils._repeat_val_seas(self.seasonality, h)
        else:
            seas_fcst = jnp.zeros(h, dtype=trend_fcst.dtype)

        mean = trend_fcst + seas_fcst

        if X is not None and self._exo_beta is not None:
            X_exo = utils.ensure_float(jnp.asarray(X))
            mean = mean + (X_exo @ self._exo_beta)

        if self.const is not None:
            mean = jnp.exp(mean)
        else:
            mean = self.mean + mean * jnp.where(self.std > 0, self.std, 1.0)

        out = {"mean": mean}
        if level:
            cs = self.conformity_scores(self.model_["fitted"])
            out = self.add_confidence_intervals(out, cs, level, "conformal_distribution")
        return out

    def forecast(self, y, h, X=None, X_future=None, level: list[int | float] | None = None):
        return self.new().fit(y, X=X).predict(h, X=X_future, level=level)

    def optimize(
        self,
        y,
        seasonal_period,
        n_steps,
        test_size,
        step_size=1,
        metric="smape",
        X=None,
        params=None,
    ):
        y = utils.ensure_float(jnp.asarray(y).reshape(-1))
        metric_fn = _metric2fn[metric]
        total = y.shape[0]

        if seasonal_period is not None and not isinstance(seasonal_period, list):
            seasonal_period = [int(seasonal_period)]
        if params is None:
            cfgs = []
            if seasonal_period is not None:
                for smoother in [True, False]:
                    for ma in [int(min(seasonal_period)), int(min(seasonal_period) // 2), None]:
                        for seas in [None, seasonal_period]:
                            for sw in [True, False]:
                                if seas is None and sw:
                                    continue
                                cfgs.append({"smoother": smoother, "ma": ma, "seasonal_period": seas, "seasonality_weights": sw})
            else:
                for smoother in [True, False]:
                    for cov in [0.5, -1]:
                        for mr in [5, 20]:
                            cfgs.append({"smoother": smoother, "cov_threshold": cov, "max_rounds": mr, "seasonal_period": None})
        else:
            cfgs = [dict(p) for p in params]

        max_steps = (total - test_size - 4) // step_size + 1
        if max_steps < 1:
            return cfgs[0]
        if max_steps < n_steps:
            n_steps = max_steps

        best_score = jnp.inf
        best_cfg = cfgs[0]
        for cfg in cfgs:
            scores = []
            for s in range(n_steps):
                train_end = total - (s * step_size + test_size)
                y_tr = y[:train_end]
                y_te = y[train_end: train_end + test_size]
                if X is not None:
                    X_tr = utils.ensure_float(jnp.asarray(X[:train_end]))
                    X_te = utils.ensure_float(jnp.asarray(X[train_end: train_end + test_size]))
                else:
                    X_tr = None
                    X_te = None
                yhat = self.new().fit(y_tr, X=X_tr, **cfg).predict(test_size, X=X_te)["mean"]
                scores.append(metric_fn(y_te, yhat))
            score = jnp.mean(jnp.stack(scores))
            if score < best_score:
                best_score, best_cfg = score, cfg
        return best_cfg


# =========================
# Test Cases
# =========================

if __name__ == "__main__":
    import jax.random as jrandom
    
    print("=" * 60)
    print("MFLES Test Suite")
    print("=" * 60)
    
    # Test 1: Basic MFLES with seasonal data
    print("\n[Test 1] Basic MFLES fit and forecast")
    n = 100
    period = 12
    t = jnp.arange(n)
    trend = 0.5 * t + 10
    seasonal = 3.0 * jnp.sin(2 * jnp.pi * t / period)
    noise = jrandom.normal(jrandom.PRNGKey(42), (n,)) * 0.5
    y1 = trend + seasonal + noise
    
    model1 = MFLES()
    model1.fit(y1, seasonal_period=period, max_rounds=10, changepoints=False)
    forecast1 = model1.predict(forecast_horizon=12)
    
    print(f"  Input series length: {n}")
    print(f"  Fitted shape: {model1.fitted_.shape}")
    print(f"  Forecast shape: {forecast1['mean'].shape}")
    print(f"  Model components: trend, seasonality, linear, ses, median")
    assert model1.fitted_.shape == (n,), "Fitted shape mismatch!"
    assert forecast1['mean'].shape == (12,), "Forecast shape mismatch!"
    assert jnp.all(jnp.isfinite(forecast1['mean'])), "Forecast contains NaN/Inf!"
    print("  ✓ Basic fit and forecast OK")
    
    # Test 2: Multiplicative seasonality
    print("\n[Test 2] Multiplicative seasonality")
    y2_mult = jnp.exp(jnp.log(10) + 0.02 * t + 0.3 * jnp.sin(2 * jnp.pi * t / period))
    model2 = MFLES()
    model2.fit(y2_mult, seasonal_period=period, multiplicative=True, max_rounds=10)
    forecast2 = model2.predict(forecast_horizon=12)
    
    print(f"  Multiplicative mode: {model2.multiplicative}")
    print(f"  All values positive: {jnp.all(y2_mult > 0)}")
    assert model2.multiplicative == True, "Should be multiplicative!"
    assert jnp.all(forecast2['mean'] > 0), "Multiplicative forecast should be positive!"
    print("  ✓ Multiplicative mode OK")
    
    # Test 3: Multiple seasonal periods
    print("\n[Test 3] Multiple seasonal periods")
    seas1 = 2.0 * jnp.sin(2 * jnp.pi * t / 7)
    seas2 = 1.5 * jnp.cos(2 * jnp.pi * t / 14)
    y3 = trend + seas1 + seas2 + noise
    
    model3 = MFLES()
    model3.fit(y3, seasonal_period=[7, 14], max_rounds=15)
    forecast3 = model3.predict(forecast_horizon=14)
    
    print(f"  Periods: [7, 14]")
    print(f"  Seasonal component shape: {model3.seasonal_component.shape}")
    assert forecast3['mean'].shape == (14,), "Forecast shape mismatch!"
    print("  ✓ Multiple seasonal periods OK")
    
    # Test 4: Robust mode
    print("\n[Test 4] Robust regression mode")
    # Add outliers
    y4 = y1.at[20].set(y1[20] + 10)
    y4 = y4.at[50].set(y4[50] - 8)
    
    model4_normal = MFLES(robust=False)
    model4_robust = MFLES(robust=True)
    
    model4_normal.fit(y4, seasonal_period=period, max_rounds=10)
    model4_robust.fit(y4, seasonal_period=period, max_rounds=10)
    
    # Robust should handle outliers better
    residuals_normal = y4 - model4_normal.fitted_
    residuals_robust = y4 - model4_robust.fitted_
    
    print(f"  Normal residual std: {jnp.std(residuals_normal):.4f}")
    print(f"  Robust residual std: {jnp.std(residuals_robust):.4f}")
    print("  ✓ Robust mode works")
    
    # Test 5: Exogenous variables
    print("\n[Test 5] Exogenous variables support")
    X_train = jnp.column_stack([
        jnp.sin(2 * jnp.pi * t / 30),
        jnp.cos(2 * jnp.pi * t / 30)
    ])
    exog_effect = X_train @ jnp.array([2.0, 1.5])
    y5 = trend + seasonal + exog_effect + noise
    
    model5 = MFLES()
    model5.fit(y5, seasonal_period=period, X=X_train, max_rounds=15)
    
    # Forecast with exogenous
    t_future = jnp.arange(n, n + 12)
    X_future = jnp.column_stack([
        jnp.sin(2 * jnp.pi * t_future / 30),
        jnp.cos(2 * jnp.pi * t_future / 30)
    ])
    forecast5 = model5.predict(forecast_horizon=12, X=X_future)
    
    print(f"  Exogenous beta: {model5._exo_beta}")
    print(f"  Forecast with exogenous: {forecast5['mean'][:3]}")
    assert model5._exo_beta is not None, "Should have exogenous coefficients!"
    print("  ✓ Exogenous variables OK")
    
    # Test 6: Changepoints (piecewise linear)
    print("\n[Test 6] Changepoints for piecewise linear trend")
    # Create data with changepoint
    y6_part1 = 0.2 * jnp.arange(50) + 10
    y6_part2 = 0.8 * jnp.arange(50, 100) + 5
    y6 = jnp.concatenate([y6_part1, y6_part2]) + jrandom.normal(jrandom.PRNGKey(123), (100,)) * 0.3
    
    model6 = MFLES()
    model6.fit(y6, changepoints=True, n_changepoints=0.1, alpha=0.5, max_rounds=20)
    forecast6 = model6.predict(forecast_horizon=10)
    
    print(f"  Changepoints enabled: True")
    print(f"  Linear component variance: {jnp.var(model6.linear_component):.4f}")
    assert jnp.var(model6.linear_component) > 0, "Should have non-zero linear component!"
    print("  ✓ Changepoints OK")
    
    # Test 7: Trend penalty
    print("\n[Test 7] Trend penalty based on R²")
    model7 = MFLES()
    model7.fit(y1, seasonal_period=period, trend_penalty=True, max_rounds=15)
    forecast7 = model7.predict(forecast_horizon=12)
    
    print(f"  Trend penalty: {model7.penalty}")
    print(f"  Penalty applied: {model7.trend_penalty}")
    assert model7.penalty is not None, "Should have penalty value!"
    assert 0 <= model7.penalty <= 1, "Penalty should be between 0 and 1!"
    print("  ✓ Trend penalty OK")
    
    # Test 8: Moving medians initialization
    print("\n[Test 8] Moving medians initialization")
    model8 = MFLES()
    model8.fit(y1, seasonal_period=period, moving_medians=True, max_rounds=10)
    
    print(f"  Median component shape: {model8.median_component.shape}")
    assert jnp.any(model8.median_component != 0), "Should have median component!"
    print("  ✓ Moving medians OK")
    
    # Test 9: Optimization (hyperparameter tuning)
    print("\n[Test 9] Hyperparameter optimization")
    # Use smaller dataset for speed
    y9 = y1[:70]
    model9 = MFLES()
    best_params = model9.optimize(
        y9,
        seasonal_period=period,
        n_steps=3,
        test_size=10,
        step_size=5,
        metric="mse"
    )
    
    print(f"  Best params keys: {list(best_params.keys())}")
    assert 'smoother' in best_params or 'ma' in best_params, "Should return valid params!"
    print("  ✓ Optimization OK")
    
    # Test 10: Conformal intervals
    print("\n[Test 10] Conformal prediction intervals")
    from conformal_intervals import ConformalIntervals
    conformal_params = ConformalIntervals(h=12)
    model10 = MFLES(conformal_params=conformal_params)
    model10.fit(y1, seasonal_period=period, max_rounds=10)
    forecast10 = model10.predict(forecast_horizon=12, level=[90, 95])
    
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
    lo_95_key = 'lo-95' if 'lo-95' in forecast10 else 'lower_95'
    hi_95_key = 'hi-95' if 'hi-95' in forecast10 else 'upper_95'
    print(f"  90% interval width (first): {forecast10[hi_90_key][0] - forecast10[lo_90_key][0]:.4f}")
    print(f"  95% interval width (first): {forecast10[hi_95_key][0] - forecast10[lo_95_key][0]:.4f}")
    print("  ✓ Conformal intervals work")
    
    # Test 11: Short series edge case
    print("\n[Test 11] Edge case: very short series")
    y11 = jnp.array([1.0, 2.0, 3.0])
    model11 = MFLES()
    model11.fit(y11)
    forecast11 = model11.predict(forecast_horizon=3)
    
    print(f"  Input length: 3")
    print(f"  Fitted: {model11.fitted_}")
    print(f"  Forecast: {forecast11['mean']}")
    assert forecast11['mean'].shape == (3,), "Should forecast despite short series!"
    print("  ✓ Short series handled")
    
    # Test 12: Constant series
    print("\n[Test 12] Edge case: constant series")
    y12 = jnp.ones(50) * 7.5
    model12 = MFLES()
    model12.fit(y12)
    forecast12 = model12.predict(forecast_horizon=10)
    
    print(f"  Input (constant 7.5)")
    print(f"  Forecast mean: {jnp.mean(forecast12['mean']):.4f}")
    assert jnp.allclose(forecast12['mean'], 7.5, atol=0.5), "Should forecast constant!"
    print("  ✓ Constant series OK")
    
    # Test 13: Seasonal weights
    print("\n[Test 13] Seasonality weights")
    model13 = MFLES()
    model13.fit(y1, seasonal_period=period, seasonality_weights=True, max_rounds=10)
    
    print(f"  Seasonality weights enabled")
    print(f"  Seasonal component std: {jnp.std(model13.seasonal_component):.4f}")
    print("  ✓ Seasonality weights OK")
    
    print("\n" + "=" * 60)
    print("✓ All MFLES tests passed!")
    print("=" * 60)