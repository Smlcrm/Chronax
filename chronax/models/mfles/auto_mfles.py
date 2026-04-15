"""
AutoMFLES (Automated Multi-Feature Locally Exponential Smoothing)

This module provides an automated, parallelized grid-search wrapper for the MFLES 
forecasting engine. It automatically identifies the optimal hyperparameters 
using time-series cross-validation.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Dict, Any, Optional, List, Union, Tuple, Iterable
import numpy as np
import itertools
import concurrent.futures
import threading  # Added for thread-safe tracking

# Assumes mfles.py is in the same directory
from .mfles import MFLES
from chronax.models.base_forecaster import BaseForecaster

# =============================================================================
# 1. JAX JIT KERNELS (Compute Heavy / GPU)
# =============================================================================

@jax.jit
def _standardize_data(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Standardizes data using pre-computed stats (Fused GPU Kernel).
    
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
    """Computes mean/std on GPU to avoid CPU-sync.
    
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
    """Calculates Mean Squared Error.
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar MSE value.
    """
    return jnp.mean((y - yhat) ** 2)

@jax.jit
def _mae(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculates Mean Absolute Error.
    
    Args:
        y (jnp.ndarray): Ground truth values.
        yhat (jnp.ndarray): Predicted values.
        
    Returns:
        jnp.ndarray: The scalar MAE value.
    """
    return jnp.mean(jnp.abs(y - yhat))

@jax.jit
def _mape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """Calculates Mean Absolute Percentage Error.
    
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
    """Calculates Symmetric Mean Absolute Percentage Error.
    
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
    """Flattens and casts to float32 for JAX.
    
    Args:
        x (Any): The input data structure containing time series values.
        
    Returns:
        jnp.ndarray: A flattened, 1-dimensional array of 32-bit floats.
    """
    return jnp.asarray(x, dtype=jnp.float32).ravel()

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
    """Runs rolling CV for one config using pre-sliced folds.
    
    Uses reduced max_rounds during CV for speed — enough to rank configs
    but not full convergence. The final model fit uses full max_rounds.
    
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

    print(f"  [AutoMFLES] Grid Search: Planned {tracker.total} runs | Completed {tracker.success} successfully.")

    best_idx = int(np.argmin(scores))
    return grid[best_idx]


# =============================================================================
# 4. CLASS WRAPPER
# =============================================================================

class AutoMFLES(BaseForecaster):
    """Automated MFLES wrapper with parallelized grid-search hyperparameter optimization.

    Inherits from BaseForecaster, providing the standard ``fit()`` / ``predict()`` /
    ``forecast()`` interface. Internally wraps an MFLES base-estimator, automatically
    selecting optimal hyperparameters via time-series cross-validation.
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
        conformal_params: Optional[Any] = None,
        alias: str = "AutoMFLES",
        n_jobs: int = 4,
    ) -> None:
        """Initializes the AutoMFLES wrapper class.
        
        Args:
            test_size (int): Primary step horizon to evaluate internal cross validation.
            season_length (Optional[Union[int, List[int]]], optional): Structural repetition frequency.
            n_windows (int, optional): Allowed number of cross validation iterations. Defaults to 2.
            config (Optional[List[Dict[str, Any]]], optional): Hardcoded overrides. Defaults to None.
            step_size (Optional[int], optional): Steps separating CV windows. Defaults to test_size.
            metric (str, optional): Assessed target loss metric. Defaults to 'smape'.
            verbose (bool, optional): Reporting status flag. Defaults to False.
            conformal_params (Optional[Any], optional): Conformal prediction configuration.
            alias (str, optional): Custom system tracking ID. Defaults to "AutoMFLES".
            n_jobs (int, optional): Authorized CPU Thread limits. Defaults to 4.
            
        Raises:
            ValueError: If test_size or n_windows are <= 0.
        """
        if test_size <= 0: raise ValueError("test_size must be > 0")
        if n_windows <= 0: raise ValueError("n_windows must be > 0")

        self.test_size: int = test_size
        self.season_length: Optional[Union[int, List[int]]] = season_length
        self.n_windows: int = n_windows
        self.config: Optional[List[Dict[str, Any]]] = config
        self.step_size: int = step_size if step_size is not None else test_size
        self.metric: str = metric
        self.verbose: bool = verbose

        self.conformal_params: Optional[Any] = conformal_params
        self.alias: str = alias
        self.n_jobs: int = n_jobs
        
        self.model_: Optional[Dict[str, Any]] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.sigma_: float = 0.0
        self.scaling_stats_: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
        self._cached_y_hash: Optional[int] = None  # Cache key for y (to detect changes)

    def fit(self, y: Union[np.ndarray, jnp.ndarray], X: Optional[jnp.ndarray] = None) -> "AutoMFLES":
        """Fits the AutoMFLES engine to the given time series and regressors.
        
        Accepts training time series data, conducts parallelized grid optimization,
        applies required structural scaling, and fits the base system.
        
        Args:
            y (Union[np.ndarray, jnp.ndarray]): Vector array of historical values.
            X (Optional[jnp.ndarray], optional): Structural feature regressor inputs. Defaults to None.
            
        Returns:
            AutoMFLES: A reference mapping back to itself to permit operation chaining.
        """
        y = _ensure_float(y)
        y_hash = hash(y.tobytes())  # Simple hash to detect y changes
        
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float32)
            if X.ndim == 1: X = X.reshape(-1, 1)
            self.scaling_stats_ = _get_stats(X)
            X = _standardize_data(X, *self.scaling_stats_)
        
        # Skip optimization if already done and y/X unchanged
        if self.best_params_ is None or self._cached_y_hash != y_hash:
            search_grid = generate_search_grid(self.season_length, self.config)

            # Use THREADED optimization (Safe & Fast)
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
            conformal_params=self.conformal_params,
            alias=self.alias,
        )
        
        # Fix: Handle season_length as list safely
        default_sl = None if self.season_length is None else (self.season_length[0] if isinstance(self.season_length, list) else self.season_length)
        final_sl = self.best_params_.get("seasonal_period", default_sl)
        fit_params = {k: v for k, v in self.best_params_.items() if k != "seasonal_period"}
        
        model.fit(y=y, seasonal_period=final_sl, X=X, **fit_params)
        
        self.model_ = {"model": model, "fitted": model.fitted_}
        self.sigma_ = _calculate_sigma(y - model.fitted_)
        return self

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None) -> Dict[str, Any]:
        """Calculates out-of-sample forward observations utilizing parameterized state mapping.
        
        Args:
            h (int): Out-of-sample target evaluation step count.
            X (Optional[jnp.ndarray], optional): Expected out-of-sample features array. Defaults to None.
            level (Optional[List[int]], optional): Percentage integer bounds (e.g. 90, 95). Defaults to None.
            
        Returns:
            Dict[str, Any]: Dictionary keys mapping "mean", and conditionally bound arrays.
            
        Raises:
            RuntimeError: Tripped if action executed without preceding fit procedure.
            ValueError: Tripped if inference attempts feature mapping absent historical features.
        """
        if self.model_ is None: raise RuntimeError("Model not fitted.")
        level = _validate_levels(level)

        if X is not None:
            if self.scaling_stats_ is None:
                raise ValueError("Model trained without X, but X provided for prediction.")
            X = jnp.asarray(X, dtype=jnp.float32)
            if X.ndim == 1: X = X.reshape(-1, 1)
            X = _standardize_data(X, *self.scaling_stats_)

        inner_level = level if self.conformal_params is not None else None
        res = self.model_["model"].predict(h=h, X=X, level=inner_level)

        if level is not None and self.conformal_params is None:
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
        """Stateless fit+predict in one call.

        Parameters
        ----------
        y : jnp.ndarray
            Input time series.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            In-sample exogenous variables.
        X_future : jnp.ndarray or None, default None
            Future exogenous variables.
        level : list[int | float] or None, default None
            Confidence levels for prediction intervals.
        fitted : bool, default False
            Whether to return in-sample fitted values.

        Returns
        -------
        dict
            Keys: 'mean', and optionally 'lo-{lv}', 'hi-{lv}', 'fitted'.
        """
        self.fit(y, X=X)
        res = self.predict(h=h, X=X_future, level=level)
        if fitted:
            res['fitted'] = self.model_['fitted']
        return res