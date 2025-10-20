"""
1. ensure_float() has one signature, a JAX array, and returns one object, a JAX array.
   It is a helper method for to ensure that the datatypes within a given array are float32. 
"""


import jax
import jax.numpy as jnp
from functools import partial as _partial
from typing import Optional, List, Dict, Union, Tuple
import theta_jax as _theta
from jax.scipy.stats import norm
import math
from collections import namedtuple
import jax.random as jrandom

def ensure_float(y: jnp.ndarray) -> jnp.ndarray:
    if not jnp.issubdtype(y.dtype, jnp.floating):
        return y.astype(jnp.float32)
    return y

@jax.jit
def calculate_sigma(residuals: jnp.ndarray, n: int) -> jnp.ndarray:
    """Calculate sigma for residuals using JAX operations.

    Args:
        residuals: Residual values
        n: Number of degrees of freedom

    Returns:
        Sigma value as JAX array
    """
    sigma = jnp.where(
        n > 0,
        jnp.sqrt(jnp.nansum(residuals**2) / n),
        0.0
    )
    return sigma

def _jax_norm_ppf(p):
    """JAX implementation of normal percent point function (inverse CDF).

    Uses Beasley-Springer-Moro approximation for the inverse normal CDF.
    """
    # Clamp p to avoid numerical issues
    p = jnp.clip(p, 1e-10, 1 - 1e-10)

    # For p > 0.5, use symmetry
    sign = jnp.where(p > 0.5, 1.0, -1.0)
    p_adj = jnp.where(p > 0.5, p, 1.0 - p)

    # Beasley-Springer-Moro approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = jnp.sqrt(-2 * jnp.log(1 - p_adj))
    z = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    return sign * z

def _quantiles(level: list[int | float]) -> jnp.ndarray:
    """
    Convert confidence levels to z-scores using normal inverse CDF.
    JAX equivalent of statsforecast.utils._quantiles()
    
    Args:
        level: List of confidence levels (0-100), e.g., [80, 95]
        
    Returns:
        Array of z-scores corresponding to each level
    """
    level_arr = jnp.atleast_1d(jnp.asarray(level, jnp.float32))
    p = 0.5 + (level_arr / 200.0)
    return jax.vmap(_jax_norm_ppf)(p)

def _calculate_intervals(
    res: dict,
    level: list[int],
    h: int,
    sigmah: jnp.ndarray | float
) -> dict:
    """
    Calculate native (non-conformal) prediction intervals using normal quantiles.
    Compatible with SeasonalNaive.predict() calls.
    """
    # Ensure mean is a JAX array
    mean = jnp.asarray(res["mean"], dtype=jnp.float32)

    sigmah = jnp.asarray(sigmah, dtype=jnp.float32)

    if sigmah.ndim == 0:
        sigmah = jnp.ones(h, dtype=jnp.float32) * sigmah
    elif sigmah.shape[0] != h:
        raise ValueError(f"sigmah shape {sigmah.shape} does not match h={h}")

    z = jnp.asarray(_quantiles(level), dtype=jnp.float32)

    lo = mean[:, None] - sigmah[:, None] * z[None, :]
    hi = mean[:, None] + sigmah[:, None] * z[None, :]

    out = {}
    for i, lv in enumerate(level[::-1]):
        out[f"lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"hi-{int(lv)}"] = hi[:, i]

    return out

def _add_fitted_pi(
    fitted: jnp.ndarray, 
    sigmah: jnp.ndarray | float, 
    level: list[int]
) -> dict:
    """
    Calculate NATIVE (non-conformal) fitted (in-sample) prediction intervals.
    JAX equivalent of statsforecast.models._add_fitted_pi()
    
    Args:
        fitted: Fitted values of shape (t,)
        sigmah: Standard error for predictions (scalar or array)
        level: Sorted list of confidence levels
        
    Returns:
        Dictionary with 'fitted-lo-XX' and 'fitted-hi-XX' keys for each level
    """
    z = _quantiles(level)
    # Broadcast to shape (t, len(level))
    lo = fitted[:, None] - z[None, :] * sigmah
    hi = fitted[:, None] + z[None, :] * sigmah
    
    out = {}
    for i, lv in enumerate(level[::-1]):
        out[f"fitted-lo-{int(lv)}"] = lo[:, len(level) - 1 - i]
    for i, lv in enumerate(level):
        out[f"fitted-hi-{int(lv)}"] = hi[:, i]
    return out

from functools import partial as _partial

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val_seas(season_vals: jnp.ndarray, h: int) -> jnp.ndarray:
    """
    Tile seasonal values to cover forecast horizon h.
    JAX equivalent of statsforecast.utils._repeat_val_seas()
    
    Args:
        season_vals: Seasonal pattern of shape (season_length,)
        h: Forecast horizon (static - must be known at compile time)
        
    Returns:
        Tiled pattern of length h
        
    Example:
        >>> season_vals = jnp.array([10.0, 20.0, 30.0])
        >>> _repeat_val_seas(season_vals, h=7)
        array([10., 20., 30., 10., 20., 30., 10.])
    """
    import math
    repeats = math.ceil(h / season_vals.size)
    return jnp.tile(season_vals, repeats)[:h]

@_partial(jax.jit, static_argnums=(1,))
def _repeat_val(val: float, h: int) -> jnp.ndarray:
    """
    Repeat scalar value h times.
    JAX equivalent of statsforecast.utils._repeat_val()
    
    Args:
        val: Scalar value to repeat
        h: Number of repetitions (forecast horizon)
        
    Returns:
        Array of length h filled with val
    """
    return jnp.full(h, val, dtype=jnp.float32)
def _seasonal_naive(
    y,
    h: int,
    season_length: int,
    fitted: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    JAX implementation of seasonal-naive forecast.
    
    Args:
        y: 1-D array-like (length T). Will be converted to jax array (float32).
        h: forecast horizon (int >= 1)
        season_length: seasonal period m (int >= 1)
        fitted: if True, also return in-sample fitted values
        
    Returns:
        dict with keys:
          - "mean": jnp.ndarray shape (h,)
          - optionally "fitted": jnp.ndarray shape (T,)
    """
    # convert input to jax array float32
    y_j = jnp.asarray(y, dtype=jnp.float32).squeeze()
    if y_j.ndim != 1:
        raise ValueError("y must be a 1-D array")
    T = y_j.shape[0]
    m = int(season_length)
    if m <= 0:
        raise ValueError("season_length must be a positive integer")
    if T < m:
        raise ValueError(f"Series length T={T} must be at least season_length={m}")
    if not isinstance(h, int) or h < 1:
        raise ValueError("h must be a positive integer")

    # last m observations (shape (m,))
    last_m = y_j[-m:]

    # build mean forecast by cycling through last_m
    idx = jnp.arange(h) % m            # shape (h,)
    mean = last_m[idx]                 # shape (h,)

    out = {"mean": mean}

    if fitted:
        # build fitted array: NaN for first m entries, and y[0:T-m] mapped to positions m..T-1
        fitted = jnp.full((T,), jnp.nan, dtype=jnp.float32)
        # values to place: y[0 : T-m]
        vals = y_j[: T - m]
        # scatter assignment into fitted at positions m..T-1
        positions = jnp.arange(m, T)
        fitted = fitted.at[positions].set(vals)
        out["fitted"] = fitted

    return out

def _add_fitted_pi(res, se, level):
    level = sorted(level)
    level = jnp.asarray(level)
    quantiles = _quantiles(level=level)
    lo = res["fitted"].reshape(-1, 1) - quantiles * se.reshape(-1, 1)
    hi = res["fitted"].reshape(-1, 1) + quantiles * se.reshape(-1, 1)
    lo = lo[:, ::-1]
    lo = {f"fitted-lo-{l}": lo[:, i] for i, l in enumerate(reversed(level))}
    hi = {f"fitted-hi-{l}": hi[:, i] for i, l in enumerate(level)}
    res = {**res, **lo, **hi}
    return res

def _add_conformal_distribution_intervals(
    fcst: dict,
    cs: jnp.ndarray,
    level: list[float] | list[int],
) -> dict:
    """
    Adds conformal intervals to the `fcst` dict based on conformal scores `cs`.
    `level` should be already sorted. This strategy creates forecast paths
    based on errors and calculates quantiles using those paths.
    """
    level = sorted(level)
    alphas = jnp.array([100 - lv for lv in level], dtype=jnp.float32)
    cuts_lower = (alphas / 200.0)[::-1]          # lower cuts reversed
    cuts_upper = 1.0 - (alphas / 200.0)         # upper cuts
    cuts = jnp.concatenate([cuts_lower, cuts_upper])

    mean = fcst["mean"].reshape(1, -1)          # 2D: 1 x horizon
    scores = jnp.vstack([mean - cs, mean + cs]) # shape: 2 x horizon
    quantiles = jnp.quantile(scores, cuts, axis=0)

    # generate column names
    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols

    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]

    return fcst

def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        # "conformal_error": _add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]

# Optional: jitted wrapper (uncomment to use)
# _seasonal_naive_jit = jax.jit(_seasonal_naive, static_argnums=(1,2,3))

@jax.jit
def _ses_forecast(x, alpha):
    complement = 1 - alpha
    n = x.size
    fitted = jnp.full_like(x, jnp.nan)
    fitted = fitted.at[0].set(x[0])  # first value

    def body_fun(i, val):
        fitted_arr, j = val
        fitted_arr = fitted_arr.at[i].set(alpha * x[j] + complement * fitted_arr[j])
        j += 1
        return fitted_arr, j

    fitted, _ = jax.lax.fori_loop(1, n, body_fun, (fitted, 0))
    forecast = alpha * x[-1] + complement * fitted[-1]
    fitted = fitted.at[0].set(jnp.nan)  # match original behavior
    return forecast, fitted


def _seasonal_exponential_smoothing(y, h, fitted, season_length, alpha):
    n = y.size
    if n < season_length:
        return {"mean": jnp.full(h, jnp.nan, dtype=y.dtype)}

    season_vals = jnp.full((season_length,), jnp.nan, dtype=y.dtype)
    fitted_vals = jnp.full_like(y, jnp.nan)

    for i in range(season_length):
        init_idx = i + n % season_length
        x = y[init_idx::season_length]  # Python slice, works fine

        forecast, fitted_season = _ses_forecast(x, alpha)

        season_vals = season_vals.at[i].set(forecast)

        for k in range(fitted_season.size):
            fitted_vals = fitted_vals.at[init_idx + k * season_length].set(fitted_season[k])

    out = _repeat_val_seas(season_vals, h)
    fcst = {"mean": out}
    if fitted:
        fcst["fitted"] = fitted_vals
    return fcst

def _conformal_method(obj, *, fcst, cs, level):
    """
    Compute conformal prediction intervals.
    Accepts keyword-only arguments to match the call.
    """
    res = fcst.copy()
    for l in level:
        res[f"level_{l}"] = fcst["mean"] + l * 0.01  # replace with real formula
    return res

def _store_cs(self, y, X):
    if self.prediction_intervals is not None:
        self._cs = self.conformity_scores(y, X)

def _add_conformal_intervals(self, fcst, y, X, level):
    if self.prediction_intervals is not None and level is not None:
        cs = self.conformity_scores(y, X) if y is not None else self._cs
        # call utils function directly
        res = _conformal_method(self, fcst=fcst, cs=cs, level=level)
        return res
    return fcst

def _add_predict_conformal_intervals(self, fcst, level):
    from utils import _add_conformal_intervals
    return _add_conformal_intervals(self,fcst=fcst, y=None, X=None, level=level)

def switch_theta(model: str) -> _theta.ModelType:
    if model == "STM":
        return _theta.ModelType.STM
    if model == "OTM":
        return _theta.ModelType.OTM
    if model == "DSTM":
        return _theta.ModelType.DSTM
    if model == "DOTM":
        return _theta.ModelType.DOTM
    raise ValueError(f"Invalid model type: {model}.")

def compute_pi_samples(n, h, states, sigma, alpha, theta, mean_y, seed=0, n_samples=200):
    """
    Compute forecast samples for conformal intervals in JAX.
    """
    samples = jnp.full((h, n_samples), jnp.nan, dtype=jnp.float32)

    # Unpack last state: level, meany, An, Bn
    level, meany, A, B = states[-1, :4]
    smoothed = level

    # Initialize PRNG key
    key = jrandom.PRNGKey(seed)

    def body_fun(i, val):
        smoothed, mean_y, A, B, samples, key = val
        # deterministic part
        mu = smoothed + (1 - 1 / theta) * (A * ((1 - alpha) ** i) + B * (1 - (1 - alpha) ** (i + 1)) / alpha)

        # random noise
        key, subkey = jrandom.split(key)
        eps = jrandom.normal(subkey, shape=(n_samples,), dtype=jnp.float32) * sigma

        # sample for this step
        s = mu + eps

        # update smoothed, mean, A, B
        smoothed_new = alpha * jnp.mean(s) + (1 - alpha) * smoothed
        mean_y_new = (i * mean_y + jnp.mean(s)) / (i + 1)
        B_new = ((i - 1) * B + 6 * (jnp.mean(s) - mean_y_new) / (i + 1)) / (i + 2)
        A_new = mean_y_new - B_new * (i + 2) / 2

        samples = samples.at[i - n].set(s)
        return smoothed_new, mean_y_new, A_new, B_new, samples, key

    # Loop over forecast horizon
    smoothed, mean_y, A, B, samples, key = jax.lax.fori_loop(
        n, n + h, body_fun, (smoothed, mean_y, A, B, samples, key)
    )

    return samples

def initparamtheta(
    initial_smoothed: float,
    alpha: float,
    theta: float,
    y: jnp.ndarray,
    modeltype: _theta.ModelType,
):
    if modeltype in [_theta.ModelType.STM, _theta.ModelType.DSTM]:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        theta = 2.0  # no optimize
        optimize_theta = False
    else:
        if math.isnan(initial_smoothed):
            initial_smoothed = y[0] / 2
            optimize_level = True
        else:
            optimize_level = False
        if math.isnan(alpha):
            alpha = 0.5
            optimize_alpha = True
        else:
            optimize_alpha = False
        if math.isnan(theta):
            theta = 2.0
            optimize_theta = True
        else:
            optimize_theta = False
    return {
        "initial_smoothed": initial_smoothed,
        "optimize_initial_smoothed": optimize_level,
        "alpha": alpha,
        "optimize_alpha": optimize_alpha,
        "theta": theta,
        "optimize_theta": optimize_theta,
    }

def optimize_theta_target_fn(
    init_par,
    lower,
    upper,
    init_level,
    init_alpha,
    init_theta,
    opt_level,
    opt_alpha,
    opt_theta,
    y,
    modeltype,
    nmse,
):
    opt_res = _theta.minimize(
        x0=init_par,
        lower=lower,
        upper=upper,
        init_level=init_level,
        init_alpha=init_alpha,
        init_theta=init_theta,
        opt_level=opt_level,
        opt_alpha=opt_alpha,
        opt_theta=opt_theta,
        y=y,
        model_type=modeltype,
        nmse=nmse,
    )

    # ✅ Make sure these are NumPy scalars, not JAX DeviceArrays
    x = jax.device_get(opt_res["x"])
    fn = float(jax.device_get(opt_res["fun"]))
    nit = int(opt_res.get("nit", 0))

    results = namedtuple("results", "x fn nit simplex")
    return results(x, fn, nit, None)

# Initializations
init_level = 0.5
init_alpha = 0.3
init_theta = 2.0

opt_level = True
opt_alpha = True
opt_theta = True

lower = jnp.array([0.0, 0.0, 1.0])
upper = jnp.array([1.0, 1.0, 10.0])

# Initial guess vector (x0)
par = jnp.array([init_level, init_alpha, init_theta])

def thetamodel(
    y: jnp.ndarray,
    m: int,
    modeltype: str,
    initial_smoothed: float,
    alpha: float,
    theta: float,
    nmse: int,
):
    y = y.astype(jnp.float64, copy=False)
    model_type = switch_theta(modeltype)
    # initial parameters
    par = initparamtheta(
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
        y=y,
        modeltype=model_type,
    )
    optimize_params = {
        key.replace("optimize_", ""): val for key, val in par.items() if "optim" in key
    }
    x0 = jnp.array([par["initial_smoothed"], par["alpha"], par["theta"]], dtype=jnp.float32)
    # parameter optimization
    fred = optimize_theta_target_fn(
    init_par=x0,
    lower=lower,
    upper=upper,
    init_level=init_level,
    init_alpha=init_alpha,
    init_theta=init_theta,
    opt_level=opt_level,
    opt_alpha=opt_alpha,
    opt_theta=opt_theta,
    y=y,
    modeltype=model_type,
    nmse=nmse,
)

    if fred is not None:
        fit_par = fred.x

    j = 0
    if optimize_params.get("initial_smoothed", False):
        par["initial_smoothed"] = float(fit_par[j])
        j += 1
    if optimize_params.get("alpha", False):
        par["alpha"] = float(fit_par[j])
        j += 1
    if optimize_params.get("theta", False):
        par["theta"] = float(fit_par[j])
        j += 1

    amse, e, states, mse = _theta.pegels_resid(
        y,
        model_type,
        par["initial_smoothed"],
        par["alpha"],
        par["theta"],
        nmse,
    )

    return dict(
        mse=mse,
        amse=amse,
        fit=fred,
        residuals=e,
        m=m,
        states=states,
        par=par,
        n=len(y),
        modeltype=modeltype,
        mean_y=jnp.mean(y),
    )

def forecast_theta(obj, h, level=None):
    # Extract parameters
    n_obs = obj["n"]  # number of observed steps
    # Initialize states if not already correct shape
    states = obj["states"]
    if states.ndim == 1 or states.shape[1] != 4:
        # assume 1D array, reshape/pad to (n_obs, 4)
        states = jnp.zeros((n_obs, 5), dtype=jnp.float32)
        # optionally fill initial level/mean/An/Bn
        states = states.at[:, 0].set(obj.get("mean_y", 0.0))   # level
        states = states.at[:, 1].set(obj.get("mean_y", 0.0))   # mean
        states = states.at[:, 2].set(0.0)                      # An
        states = states.at[:, 3].set(0.0)                      # Bn
        states = states.at[:, 4].set(0.0)     

    alpha = obj["par"]["alpha"]
    theta = obj["par"]["theta"]
    model_type = switch_theta(obj["modeltype"])

    # Call the JAX forecast function
    states, forecast = _theta.forecast(states, h, model_type, alpha, theta)

    # Build result dictionary
    res = {"mean": forecast}

    # Compute prediction intervals if requested
    if level is not None:
        sigma = jnp.std(obj["residuals"][3:], ddof=1)
        mean_y = obj["mean_y"]
        samples = compute_pi_samples(
            n=obj["n"],
            h=h,
            states=states,
            sigma=sigma,
            alpha=alpha,
            theta=theta,
            mean_y=mean_y,
        )

        for lv in level:
            min_q = (100 - lv) / 200
            max_q = min_q + lv / 100
            res[f"lo-{lv}"] = jnp.quantile(samples, min_q, axis=1)
            res[f"hi-{lv}"] = jnp.quantile(samples, max_q, axis=1)

    # Recompose if seasonal decomposition was used
    if obj.get("decompose", False):
        seas_forecast = _repeat_val_seas(obj["seas_forecast"]["mean"], h=h)
        for key in res:
            if obj["decomposition_type"] == "multiplicative":
                res[key] = res[key] * seas_forecast
            else:
                res[key] = res[key] + seas_forecast

    return res

def is_constant(x):
    return jnp.all(x[0] == x)

def seasonal_decompose(y: jnp.ndarray, model: str = "additive", period: int = 1):
    """
    Simple seasonal decomposition using moving average.
    Returns a dict with 'trend', 'seasonal', and 'resid' like statsmodels.
    """
    n = len(y)
    # Moving average for trend
    kernel = jnp.ones(period) / period
    trend = jnp.convolve(y, kernel, mode="same")

    if model == "additive":
        detrended = y - trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended - seasonal
    else:  # multiplicative
        detrended = y / trend
        seasonal = jnp.tile(
            jnp.mean(detrended.reshape(-1, period), axis=0), n // period + 1
        )[:n]
        resid = detrended / seasonal

    return {
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
    }

def acf(x: jnp.ndarray, nlags: int) -> jnp.ndarray:
    """
    Compute autocorrelation function up to `nlags` for 1D array x using JAX.
    Equivalent to statsmodels.tsa.stattools.acf(x, nlags=nlags, fft=False).
    """
    x = x - jnp.mean(x)
    n = x.shape[0]
    denom = jnp.dot(x, x)
    acf_vals = jnp.array([jnp.dot(x[: n - lag], x[lag:]) / denom for lag in range(nlags + 1)])
    return acf_vals

def auto_theta(
    y,
    m,
    model=None,
    initial_smoothed=None,
    alpha=None,
    theta=None,
    nmse=3,
    decomposition_type="multiplicative",
):
    # converting params to floats
    # to improve numba compilation
    if initial_smoothed is None:
        initial_smoothed = jnp.nan
    if alpha is None:
        alpha = jnp.nan
    if theta is None:
        theta = jnp.nan
    if nmse < 1 or nmse > 30:
        raise ValueError("nmse out of range")
    # constan values
    if is_constant(y):
        thetamodel(
            y=y,
            m=m,
            modeltype="STM",
            nmse=nmse,
            initial_smoothed=jnp.mean(y) / 2,
            alpha=0.5,
            theta=2.0,
        )
    # seasonal decomposition if needed
    decompose = False
    # seasonal test
    if m >= 4 and len(y) >= 2 * m:
        r = acf(y, nlags=m, fft=False)[1:]
        stat = jnp.sqrt((1 + 2 * jnp.sum(r[:-1] ** 2)) / len(y))
        decompose = jnp.abs(r[-1]) / stat > norm.ppf(0.95)

    data_positive = min(y) > 0
    if decompose:
        # change decomposition type if data is not positive
        if decomposition_type == "multiplicative" and not data_positive:
            decomposition_type = "additive"
        y_decompose = seasonal_decompose(y, model=decomposition_type, period=m).seasonal
        if decomposition_type == "multiplicative" and any(y_decompose < 0.01):
            decomposition_type = "additive"
            y_decompose = seasonal_decompose(y, model="additive", period=m).seasonal
        if decomposition_type == "additive":
            y = y - y_decompose
        else:
            y = y / y_decompose
        seas_forecast = _seasonal_naive(
            y=y_decompose, h=m, season_length=m, fitted=False
        )

    # validate model
    if model not in [None, "STM", "OTM", "DSTM", "DOTM"]:
        raise ValueError(f"Invalid model type: {model}.")

    n = len(y)
    npars = 3
    # non-optimized tiny datasets
    if n <= npars:
        raise NotImplementedError("tiny datasets")
    if model is None:
        modeltype = ["STM", "OTM", "DSTM", "DOTM"]
    else:
        modeltype = [model]

    best_ic = jnp.inf
    for mtype in modeltype:
        fit = thetamodel(
            y=y,
            m=m,
            modeltype=mtype,
            nmse=nmse,
            initial_smoothed=initial_smoothed,
            alpha=alpha,
            theta=theta,
        )
        fit_ic = fit["mse"]
        if not jnp.isnan(fit_ic):
            if fit_ic < best_ic:
                model = fit
                best_ic = fit_ic
    if jnp.isinf(best_ic):
        raise Exception("no model able to be fitted")

    if decompose:
        if decomposition_type == "multiplicative":
            model["residuals"] = model["residuals"] * y_decompose
        else:
            model["residuals"] = model["residuals"] + y_decompose
        model["decompose"] = decompose
        model["decomposition_type"] = decomposition_type
        model["seas_forecast"] = dict(seas_forecast)
    return model


def forward_theta(fitted_model, y):
    m = fitted_model["m"]
    model = fitted_model["modeltype"]
    initial_smoothed = fitted_model["par"]["initial_smoothed"]
    alpha = fitted_model["par"]["alpha"]
    theta = fitted_model["par"]["theta"]
    return auto_theta(
        y=y,
        m=m,
        model=model,
        initial_smoothed=initial_smoothed,
        alpha=alpha,
        theta=theta,
    )