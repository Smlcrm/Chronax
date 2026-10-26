"""
AutoMFLES (automated Multi-Feature Locally Exponential Smoothing).

A threaded grid-search wrapper around the MFLES forecaster that selects
hyperparameters by rolling time-series cross-validation (CV).
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Dict, Any, Optional, List, Union, Tuple, Iterable
import numpy as np
import itertools
import concurrent.futures
import threading  # thread-safe progress tracking

from .mfles import MFLES, _ROBUST_AUTODETECT_MAX_N
from chronax.models.base_forecaster import BaseForecaster

# =============================================================================
# 1. JAX just-in-time (JIT) compiled kernels
# =============================================================================

@jax.jit
def _standardize_data(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Standardize data using pre-computed column statistics.
    
    Args:
        x (jnp.ndarray): The input data array to standardize.
        mean (jnp.ndarray): The pre-computed mean values.
        std (jnp.ndarray): The pre-computed standard deviation values.
        
    Returns:
        jnp.ndarray: The standardized data array.
    """
    safe_std = jnp.where(std < 1e-6, 1.0, std)
    return (x - mean) / safe_std

@jax.jit
def _get_stats(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute column-wise mean and standard deviation on device.
    
    Args:
        x (jnp.ndarray): The input array (usually exogenous regressors).
        
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the column-wise mean and standard deviation.
    """
    return jnp.mean(x, axis=0), jnp.std(x, axis=0)

@jax.jit
def _calculate_sigma(residuals: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Calculates residual standard deviation on device.
    
    Args:
        residuals (jnp.ndarray): The array of model error residuals.
        eps (float, optional): A minimum epsilon floor to prevent exactly zero std. Defaults to 1e-8.
        
    Returns:
        jnp.ndarray: The bounded standard deviation of the residuals.
    """
    return jnp.maximum(jnp.std(residuals), eps)

@jax.jit
def _gaussian_bounds(mean: jnp.ndarray, sigma: jnp.ndarray, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized prediction interval bounds.
    
    Args:
        mean (jnp.ndarray): The array of predicted mean values.
        sigma (jnp.ndarray): The calculated residual standard deviation.
        z (jnp.ndarray): An array of Z-scores corresponding to desired confidence levels.
        
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Lower bounds and upper bounds arrays.
    """
    margin = z[:, None] * sigma
    return mean[None, :] - margin, mean[None, :] + margin

@jax.jit
def _z_from_levels(levels_float: jnp.ndarray) -> jnp.ndarray:
    """Computes Z-scores from confidence levels.
    
    Args:
        levels_float (jnp.ndarray): An array of confidence levels in percentage (e.g., 90.0, 95.0).
        
    Returns:
        jnp.ndarray: The corresponding normal distribution quantiles (Z-scores).
    """
    alpha = (100.0 - levels_float) / 100.0
    return norm.ppf(1.0 - alpha / 2.0)

# --- Metrics ---
@jax.jit
def _mse(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculate the mean squared error (MSE).
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar MSE value.
    """
    return jnp.mean((y - yhat) ** 2)

@jax.jit
def _mae(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculate the mean absolute error (MAE).
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar MAE value.
    """
    return jnp.mean(jnp.abs(y - yhat))

@jax.jit
def _mape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculate the mean absolute percentage error (MAPE).
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar MAPE value.
    """
    mask = jnp.abs(y) > 1e-6
    return jnp.mean(jnp.where(mask, jnp.abs((y - yhat) / y), 0.0))

@jax.jit
def _smape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculate the symmetric mean absolute percentage error (sMAPE).
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar sMAPE value.
    """
    denominator = jnp.abs(y) + jnp.abs(yhat)
    return jnp.mean(2.0 * jnp.abs(y - yhat) / (denominator + 1e-6))

_METRIC_MAP: Dict[str, Any] = {
    "mse": _mse,
    "mae": _mae,
    "mape": _mape,
    "smape": _smape
}

# =============================================================================
# 2. PURE LOGIC HANDLERS (Validation & Grid Search)
# =============================================================================

def _ensure_float(x: Any) -> jnp.ndarray:
    """Flatten and ensure a floating dtype, preserving the input's precision.

    Preserves float64 (matching plain MFLES and statsforecast); only non-float
    inputs are cast to float32. Fit and the fast forecast path both route
    through here, so the cached conformity scores and the per-window fits
    stay at one consistent dtype.
    """
    a = jnp.asarray(x).ravel()
    return a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(jnp.float32)


def _ensure_float_2d(x: Any) -> jnp.ndarray:
    """As _ensure_float but keeps the (n, n_features) exogenous shape."""
    a = jnp.asarray(x)
    a = a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(jnp.float32)
    return a.reshape(-1, 1) if a.ndim == 1 else a

def _validate_levels(level: Optional[Union[List[int], Tuple[int, ...]]]) -> Optional[List[int]]:
    """Sanitizes prediction interval levels.
    
    Args:
        level (Optional[Union[List[int], Tuple[int, ...]]]): The list of confidence levels to sanitize.
        
    Returns:
        Optional[List[int]]: A sorted list of unique, valid integer levels.
        
    Raises:
        ValueError: If levels are empty or fall outside the valid (0, 100) range.
    """
    if level is None: return None
    if not level: raise ValueError("Level must be non-empty.")
    clean = sorted(list(set(int(l) for l in level)))
    if any(l <= 0 or l >= 100 for l in clean):
        raise ValueError("Levels must be between 0 and 100.")
    return clean

def _logic_check(keys_to_check: Iterable[str], keys: Iterable[str]) -> bool:
    """Checks if a subset of keys exists within a target set of keys.
    
    Args:
        keys_to_check (Iterable[str]): The keys required to be present.
        keys (Iterable[str]): The available keys to check against.
        
    Returns:
        bool: True if all target keys exist, False otherwise.
    """
    return set(keys_to_check).issubset(keys)

def _is_valid_config(param_dict: Dict[str, Any]) -> bool:
    """Filters invalid hyperparameter combinations (StatsForecast parity).
    
    Args:
        param_dict (Dict[str, Any]): A dictionary representing a single hyperparameter configuration.
        
    Returns:
        bool: True if the configuration is logically valid, False if it conflicts.
    """
    keys = param_dict.keys()
    
    if _logic_check(["seasonal_period", "max_rounds"], keys):
        if param_dict["seasonal_period"] is None and param_dict["max_rounds"] < 4:
            return False
    if _logic_check(["smoother", "ma"], keys):
        if param_dict["smoother"] and param_dict["ma"] is not None:
            return False
    if _logic_check(["seasonal_period", "seasonality_weights"], keys):
        if param_dict["seasonality_weights"] and param_dict["seasonal_period"] is None:
            return False
    return True

def generate_search_grid(
    season_length: Optional[Union[int, List[int]]], 
    user_config: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Generates the hyperparameter grid.
    
    Args:
        season_length (Optional[Union[int, List[int]]]): The primary periodicity of the data.
        user_config (Optional[List[Dict[str, Any]]], optional): Custom user-defined configurations.
        
    Returns:
        List[Dict[str, Any]]: A list of all valid parameter configurations to test.
    """
    if user_config is not None:
        if isinstance(user_config, list): return user_config
        keys, values = zip(*user_config.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    sl = season_length
    if sl is not None:
        if not isinstance(sl, list): sl = [sl]
        configs = {
            "seasonality_weights": [True, False],
            "smoother": [True, False],
            "ma": [int(min(sl)), int(min(sl) // 2), None],
            "seasonal_period": [None, sl],
        }
    else:
        configs = {
            "smoother": [True, False],
            "cov_threshold": [0.5, -1],
            "max_rounds": [5, 20],
            "seasonal_period": [None],
        }
    
    keys, values = zip(*configs.items())
    grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return [g for g in grid if _is_valid_config(g)]

def add_gaussian_intervals(res: Dict[str, Any], level: List[int], sigma: float) -> Dict[str, Any]:
    """Applies vectorized Gaussian intervals.
    
    Args:
        res (Dict[str, Any]): The dictionary containing the forecasted means.
        level (List[int]): The list of confidence levels to evaluate.
        sigma (float): The residual standard error calculated during model fit.
        
    Returns:
        Dict[str, Any]: The input dictionary updated with interval keys.
    """
    mean = res["mean"]
    lv_arr = jnp.array(level, dtype=jnp.float32)
    z = _z_from_levels(lv_arr)
    lo, hi = _gaussian_bounds(mean, sigma, z)
    
    for i, lv in enumerate(level):
        res[f"lo-{lv}"] = lo[i]
        res[f"hi-{lv}"] = hi[i]
    return res

# =============================================================================
# 3. THREADED OPTIMIZATION WITH TRACKER
# =============================================================================

# --- TRACKER CLASS ---
class GridTracker:
    """A thread-safe tracking object for monitoring grid search progress."""
    
    def __init__(self) -> None:
        """Initializes the counter and locking mechanism."""
        self.total: int = 0
        self.success: int = 0
        self.lock: threading.Lock = threading.Lock()

    def set_total(self, n: int) -> None:
        """Sets the upper limit of expected iterations.
        
        Args:
            n (int): The total number of configurations to evaluate.
        """
        self.total = n

    def increment_success(self) -> None:
        """Increments the success counter using a thread-safe lock."""
        with self.lock:
            self.success += 1

def cross_validation(
    folds: List[tuple],
    test_size: int,
    config: Dict[str, Any],
    has_exogenous: bool,
    cv_max_rounds: int = 10,
) -> float:
    """Run rolling CV for one config using pre-sliced folds.

    Uses a reduced ``max_rounds`` during CV: enough to rank configs, not full
    convergence. The final model fit uses the full ``max_rounds``. The score
    is always the symmetric mean absolute percentage error (sMAPE).
    
    Args:
        folds (List[tuple]): A list containing pre-sliced validation chunks.
        test_size (int): The number of future steps to predict per fold.
        config (Dict[str, Any]): The specific hyperparameter set to evaluate.
        has_exogenous (bool): Flag indicating if regressors exist in the folds.
        cv_max_rounds (int, optional): Evaluation max boosting rounds constraint. Defaults to 10.
        
    Returns:
        float: The average validation score across all tested windows.
    """
    cfg = config.copy()
    sp = cfg.pop("seasonal_period", None)
    # Use reduced iterations for CV speed (config ranking doesn't need full convergence)
    if "max_rounds" not in cfg:
        cfg["max_rounds"] = cv_max_rounds
    total_score = jnp.float32(0.0)
    count = 0

    for fold in folds:
        if has_exogenous:
            y_train, y_test, X_train, X_test = fold
        else:
            y_train, y_test = fold
            X_train = X_test = None

        model = MFLES(verbose=0, alias="CV_Model")
        model.fit(y=y_train, X=X_train, seasonal_period=sp, **cfg)
        preds = model.predict(h=test_size, X=X_test)
        total_score = total_score + _smape(y_test, preds["mean"])
        count += 1

    if count == 0:
        return float("inf")
    return float((total_score / count).item())


def optimize_grid_threaded(
    y: jnp.ndarray,
    X: Optional[jnp.ndarray],
    test_size: int,
    n_windows: int,
    step_size: int,
    grid: List[Dict[str, Any]],
    n_jobs: int = 4,
    verbose: bool = False
) -> Dict[str, Any]:
    """Parallel optimization using ThreadPoolExecutor.
    Pre-slices CV folds once for all configs to reduce overhead.
    
    Args:
        y (jnp.ndarray): Target time series.
        X (Optional[jnp.ndarray]): Matrix of exogenous variables.
        test_size (int): Size of validation sets.
        n_windows (int): Number of rolling cross-validation backtests.
        step_size (int): Temporal spacing between validation chunks.
        grid (List[Dict[str, Any]]): Master list of hyperparameter combinations.
        n_jobs (int, optional): Thread execution limit. Defaults to 4.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        
    Returns:
        Dict[str, Any]: The configuration dict that reported the lowest average error.
    """
    n_samples = y.shape[0]
    max_possible_windows = (n_samples - test_size - 4) // step_size + 1
    if max_possible_windows < 1:
        return grid[0]
    actual_windows = min(n_windows, max_possible_windows)
    has_exogenous = X is not None

    # Pre-slice all CV folds once (shared across all configs)
    folds = []
    for split_idx in range(actual_windows):
        cutoff = n_samples - test_size - (split_idx * step_size)
        if cutoff < 4:
            break
        y_train = y[:cutoff]
        y_test = y[cutoff: cutoff + test_size]
        if has_exogenous:
            folds.append((y_train, y_test, X[:cutoff], X[cutoff: cutoff + test_size]))
        else:
            folds.append((y_train, y_test))

    if not folds:
        return grid[0]

    # Adaptive CV max_rounds: short series converge fast, long series need more
    max_train_len = max(f[0].shape[0] for f in folds)
    if max_train_len < 500:
        cv_rounds = 10
    elif max_train_len < 2000:
        cv_rounds = 20
    else:
        cv_rounds = 30

    tracker = GridTracker()
    tracker.set_total(len(grid))

    def _worker(cfg: Dict[str, Any]) -> float:
        """Internal worker function for mapping parallel threads."""
        try:
            score = cross_validation(folds, test_size, cfg, has_exogenous, cv_max_rounds=cv_rounds)
            tracker.increment_success()
            return score
        except Exception as e:
            if verbose:
                print(f"Config failed: {e}")
            return float("inf")

    if n_jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            scores = list(executor.map(_worker, grid))
    else:
        scores = [_worker(g) for g in grid]

    if verbose:
        print(f"  [AutoMFLES] Grid Search: Planned {tracker.total} runs | "
              f"Completed {tracker.success} successfully.")

    best_idx = int(np.argmin(scores))
    return grid[best_idx]


# =============================================================================
# 4. CLASS WRAPPER
# =============================================================================

class AutoMFLES(BaseForecaster):
    """MFLES wrapper that selects hyperparameters by threaded grid search.

    ``fit`` scores every configuration in the grid with rolling CV (sMAPE),
    then fits one :class:`MFLES` with the best configuration. Exogenous
    regressors are standardised with statistics from the training ``X``.
    Prediction intervals are conformal when ``prediction_intervals`` is set,
    otherwise Gaussian bands from the in-sample residual standard deviation.

    Args:
        test_size (int): Forecast horizon of each CV window.
        season_length (int | list[int] | None): Seasonal period(s). Drives the
            default grid and the ``seasonal_period`` passed to the final fit
            (the first period when a list is given). Default is None.
        n_windows (int): Number of rolling CV windows. Default is 2.
        config (list[dict] | dict[str, list] | None): Grid override. A list of
            :meth:`MFLES.fit` keyword dicts is used as is; a dict of lists is
            expanded to its Cartesian product. None builds the default grid
            from ``season_length``. Default is None.
        step_size (int | None): Observations between CV windows. None uses
            ``test_size``. Default is None.
        metric (str): Stored on the instance; the grid search always scores
            sMAPE. Default is "smape".
        verbose (bool): Print grid-search progress. Default is False.
        prediction_intervals (ConformalIntervals | None): Conformal interval
            configuration, stored as ``conformal_params`` and passed to the
            inner MFLES. None selects Gaussian intervals. Default is None.
        alias (str): Model name identifier. Default is "AutoMFLES".
        n_jobs (int): Number of threads for the grid search. Default is 4.

    Attributes:
        model_ (dict | None): ``{"model": MFLES, "fitted": jnp.ndarray}`` after fit.
        best_params_ (dict | None): Selected :meth:`MFLES.fit` keyword arguments.
        sigma_ (jnp.ndarray): In-sample residual standard deviation after fit.

    Raises:
        ValueError: If ``test_size`` or ``n_windows`` is not positive.
    """

    def __init__(
        self,
        test_size: int,
        season_length: Optional[Union[int, List[int]]] = None,
        n_windows: int = 2,
        config: Optional[List[Dict[str, Any]]] = None,
        step_size: Optional[int] = None,
        metric: str = "smape",
        verbose: bool = False,
        prediction_intervals: Optional[Any] = None,
        alias: str = "AutoMFLES",
        n_jobs: int = 4
    ) -> None:
        """Store the configuration; see the class docstring for the arguments."""
        if test_size <= 0: raise ValueError("test_size must be > 0")
        if n_windows <= 0: raise ValueError("n_windows must be > 0")

        self.test_size: int = test_size
        self.season_length: Optional[Union[int, List[int]]] = season_length
        self.n_windows: int = n_windows
        self.config: Optional[List[Dict[str, Any]]] = config
        self.step_size: int = step_size if step_size is not None else test_size
        self.metric: str = metric
        self.verbose: bool = verbose
        self.prediction_intervals: Optional[Any] = prediction_intervals
        # Base-class conformal contract reads `conformal_params` (the constructor
        # name `prediction_intervals` is kept for interface compatibility).
        self.conformal_params = prediction_intervals
        self.alias: str = alias
        self.n_jobs: int = n_jobs
        
        self.model_: Optional[Dict[str, Any]] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.sigma_: float = 0.0
        self.scaling_stats_: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        self._cached_y_hash: Optional[int] = None  # Cache key for y (to detect changes)

    def fit(self, y: Union[np.ndarray, jnp.ndarray], X: Optional[jnp.ndarray] = None) -> "AutoMFLES":
        """Select hyperparameters by grid search and fit the inner MFLES.

        The grid search runs only when ``y`` changed since the last fit (or on
        the first fit); otherwise the cached ``best_params_`` are reused.

        Args:
            y (np.ndarray | jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``;
                standardised column-wise before fitting. Default is None.

        Returns:
            AutoMFLES: ``self``, for method chaining.
        """
        y = _ensure_float(y)
        y_hash = hash(y.tobytes())  # Simple hash to detect y changes
        
        if X is not None:
            X = _ensure_float_2d(X)
            self.scaling_stats_ = _get_stats(X)
            X = _standardize_data(X, *self.scaling_stats_)
        
        # Skip optimization if already done and y/X unchanged
        if self.best_params_ is None or self._cached_y_hash != y_hash:
            search_grid = generate_search_grid(self.season_length, self.config)

            # Threaded grid search
            self.best_params_ = optimize_grid_threaded(
                y=y,
                X=X,
                test_size=self.test_size,
                n_windows=self.n_windows,
                step_size=self.step_size,
                grid=search_grid,
                n_jobs=self.n_jobs, 
                verbose=self.verbose
            )
            self._cached_y_hash = y_hash  # Cache the hash
        
        # Always fit the model with best params (fast after caching)
        model = MFLES(
            verbose=int(self.verbose),
            conformal_params=self.prediction_intervals,
            alias=self.alias,
        )
        
        # season_length may be a list; its first period is the fallback
        default_sl = None if self.season_length is None else (self.season_length[0] if isinstance(self.season_length, list) else self.season_length)
        final_sl = self.best_params_.get("seasonal_period", default_sl)
        fit_params = {k: v for k, v in self.best_params_.items() if k != "seasonal_period"}
        
        model.fit(y=y, seasonal_period=final_sl, X=X, **fit_params)
        
        self.model_ = {"model": model, "fitted": model.fitted_}
        self.sigma_ = _calculate_sigma(y - model.fitted_)
        return self

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None) -> Dict[str, Any]:
        """Forecast ``h`` steps ahead from the fitted inner MFLES.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray | None): Future exogenous regressors, shape ``(h, F)``;
                standardised with the training statistics. Default is None.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Default is None.

        Returns:
            dict: ``"mean"`` of shape ``(h,)``, plus ``"lo-{lv}"`` and
            ``"hi-{lv}"`` of shape ``(h,)`` per level when ``level`` is given
            (conformal when ``prediction_intervals`` is set, otherwise Gaussian).

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If ``X`` is given but the model was fitted without
                ``X``; if ``level`` is empty or has a value outside (0, 100);
                or, with ``prediction_intervals`` set, if ``h`` differs from
                its ``h``.
        """
        if self.model_ is None: raise RuntimeError("Model not fitted.")
        level = _validate_levels(level)

        if X is not None:
            if self.scaling_stats_ is None:
                raise ValueError("Model trained without X, but X provided for prediction.")
            X = jnp.asarray(X, dtype=jnp.float32)
            if X.ndim == 1: X = X.reshape(-1, 1)
            X = _standardize_data(X, *self.scaling_stats_)

        inner_level = level if self.prediction_intervals is not None else None
        res = self.model_["model"].predict(h=h, X=X, level=inner_level)

        if level is not None and self.prediction_intervals is None:
            res = add_gaussian_intervals(res, level, self.sigma_)

        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """Fit-and-forecast in one call.

        On an unfitted instance this calls :meth:`fit` (grid search included)
        and then :meth:`predict`, so it stores state on ``self``. On a fitted
        instance it is stateless: the inner MFLES re-fits on ``y`` with the
        selected configuration and nothing on ``self`` changes. The base-class
        ``conformity_scores`` maps the fitted path over CV windows.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape ``(T, F)``. Default is None.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape ``(h, F)``. Default is None.
            level (list[int | float] | None): Confidence levels in (0, 100), e.g. ``[80, 95]``. Default is None.
            fitted (bool): If True, also return the in-sample fitted values of the full-series fit. Default is False.

        Returns:
            dict: ``"mean"`` of shape ``(h,)``; ``"lo-{lv}"`` and ``"hi-{lv}"`` per level when ``level`` is given; ``"fitted"`` of shape ``(T,)`` when ``fitted`` is True.

        Raises:
            ValueError: If ``X`` or ``X_future`` is given but the model was fitted without ``X``; if ``level`` is empty or has a value outside (0, 100) and ``prediction_intervals`` is None; or, with ``prediction_intervals`` set, if ``h`` differs from its ``h``.
        """
        if self.best_params_ is None or self.model_ is None:
            # First call: the grid search is a host-side threaded loop and
            # needs concrete data.
            self.fit(y, X=X)
            res = self.predict(h=h, X=X_future, level=level)
            if fitted:
                res['fitted'] = self.model_['fitted']
            return res

        # Fitted fast path: fully traceable and writes NOTHING to self (the
        # base conformity_scores vmaps this method). Delegates to the inner
        # MFLES's stateless forecast, which replays the resolved fit config,
        # so every CV window re-fits the selected configuration on its own y.
        # Fit at the selected model's precision regardless of caller dtype so
        # scores match the _cs cached at fit (f32-vs-f64 window fits can flip
        # lax.cond accept/reject decisions in the MFLES loop).
        y = _ensure_float(y)
        if X is not None:
            if self.scaling_stats_ is None:
                raise ValueError("Model trained without X, but X provided.")
            X = _ensure_float_2d(X)
            X = _standardize_data(X, *self.scaling_stats_)
        if X_future is not None:
            if self.scaling_stats_ is None:
                raise ValueError("Model trained without X, but X_future provided.")
            X_future = _ensure_float_2d(X_future)
            X_future = _standardize_data(X_future, *self.scaling_stats_)

        # Mirror predict(): delegate level to the inner conformal machinery
        # only when intervals were configured; otherwise fall back to the
        # Gaussian bands so the fitted path matches the unfitted path.
        inner_level = level if self.prediction_intervals is not None else None
        res = self.model_["model"].forecast(
            y=y, h=h, X=X, X_future=X_future, level=inner_level
        )
        if level is not None and self.prediction_intervals is None:
            res = add_gaussian_intervals(res, _validate_levels(level), self.sigma_)
        if fitted:
            # In-sample fitted values from the full-series fit (eager cache);
            # no current traced path requests fitted=True.
            res['fitted'] = self.model_['fitted']
        return res

    def conformity_scores(
        self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Conformity scores with eager selection and a per-window re-fit.

        The grid search (a threaded Python loop with host-side scoring) cannot
        trace, so it runs once, eagerly, here. The per-window ``forecast`` then
        re-fits the selected configuration on each window through the inner
        MFLES config-replay path. The replayed config does not pin ``robust``,
        so windows shorter than ``_ROBUST_AUTODETECT_MAX_N`` auto-detect it as
        a traced flag; those run sequentially to avoid the batched-``cond``
        Siegel cost, longer windows run under the base-class vmap.

        Calibration caveat: the configuration is selected with sight of the
        full series; per-window parameters are still re-fit, so scores vary
        across windows.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``. Default is None.

        Returns:
            jnp.ndarray: Conformity scores of shape ``(n_windows, h)``.
        """
        if self.best_params_ is None or self.model_ is None:
            self.fit(y, X)
            # fit() already ran the walk-forward on exactly this y via the
            # inner model's _cs cache; reuse instead of paying CV twice.
            inner_cs = getattr(self.model_["model"], "_cs", None)
            if inner_cs is not None:
                return inner_cs
        if self.conformal_params is not None:
            window_n = jnp.asarray(y).size - self.conformal_params.h
            if window_n < _ROBUST_AUTODETECT_MAX_N:
                return self._conformity_scores_sequential(y=y, X=X)
        return super().conformity_scores(y, X)
