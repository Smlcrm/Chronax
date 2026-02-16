import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import itertools
import concurrent.futures
import threading  # Added for thread-safe tracking

# Assumes mfles.py is in the same directory
from mfles import MFLES

# =============================================================================
# 1. JAX JIT KERNELS (Compute Heavy / GPU)
# =============================================================================

@jax.jit
def _standardize_data(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    """Standardizes data using pre-computed stats (Fused GPU Kernel)."""
    safe_std = jnp.where(std < 1e-6, 1.0, std)
    return (x - mean) / safe_std

@jax.jit
def _get_stats(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes mean/std on GPU to avoid CPU-sync."""
    return jnp.mean(x, axis=0), jnp.std(x, axis=0)

@jax.jit
def _calculate_sigma(residuals: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Calculates residual standard deviation on device."""
    return jnp.maximum(jnp.std(residuals), eps)

@jax.jit
def _gaussian_bounds(mean: jnp.ndarray, sigma: jnp.ndarray, z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized prediction interval bounds."""
    margin = z[:, None] * sigma
    return mean[None, :] - margin, mean[None, :] + margin

@jax.jit
def _z_from_levels(levels_float: jnp.ndarray) -> jnp.ndarray:
    """Computes Z-scores from confidence levels."""
    alpha = (100.0 - levels_float) / 100.0
    return norm.ppf(1.0 - alpha / 2.0)

# --- Metrics ---
@jax.jit
def _mse(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((y - yhat) ** 2)

@jax.jit
def _mae(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y - yhat))

@jax.jit
def _mape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.abs(y) > 1e-6
    return jnp.mean(jnp.where(mask, jnp.abs((y - yhat) / y), 0.0))

@jax.jit
def _smape(y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    denominator = jnp.abs(y) + jnp.abs(yhat)
    return jnp.mean(2.0 * jnp.abs(y - yhat) / (denominator + 1e-6))

_METRIC_MAP = {
    "mse": _mse,
    "mae": _mae,
    "mape": _mape,
    "smape": _smape
}

# =============================================================================
# 2. PURE LOGIC HANDLERS (Validation & Grid Search)
# =============================================================================

def _ensure_float(x):
    """Flattens and casts to float32 for JAX."""
    return jnp.asarray(x, dtype=jnp.float32).ravel()

def _validate_levels(level: Optional[Union[List[int], Tuple[int, ...]]]) -> Optional[List[int]]:
    """Sanitizes prediction interval levels."""
    if level is None: return None
    if not level: raise ValueError("Level must be non-empty.")
    clean = sorted(list(set(int(l) for l in level)))
    if any(l <= 0 or l >= 100 for l in clean):
        raise ValueError("Levels must be between 0 and 100.")
    return clean

def _logic_check(keys_to_check, keys):
    return set(keys_to_check).issubset(keys)

def _is_valid_config(param_dict: Dict[str, Any]) -> bool:
    """Filters invalid hyperparameter combinations (StatsForecast parity)."""
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
    """Generates the hyperparameter grid."""
    
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
    """Applies vectorized Gaussian intervals."""
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
    def __init__(self):
        self.total = 0
        self.success = 0
        self.lock = threading.Lock()

    def set_total(self, n):
        self.total = n

    def increment_success(self):
        with self.lock:
            self.success += 1

def cross_validation(
    y: jnp.ndarray,
    X: Optional[jnp.ndarray],
    test_size: int,
    n_windows: int,
    step_size: int,
    config: Dict[str, Any]
) -> float:
    """Runs rolling CV for one config."""
    scores = []
    n_samples = y.shape[0]

    for split_idx in range(n_windows):
        cutoff = n_samples - test_size - (split_idx * step_size)
        if cutoff < 4: break

        y_train = y[:cutoff]
        y_test = y[cutoff : cutoff + test_size]
        X_train = X[:cutoff] if X is not None else None
        X_test = X[cutoff : cutoff + test_size] if X is not None else None

        # Instantiate Model (Verbose 0)
        model = MFLES(verbose=0, alias="CV_Model")
        
        cfg = config.copy()
        sp = cfg.pop("seasonal_period", None)
        
        # This call is safe in threads (no vmap issues)
        model.fit(y=y_train, X=X_train, seasonal_period=sp, **cfg)
        
        preds = model.predict(h=test_size, X=X_test)
        
        # Use JIT metric
        score = _smape(y_test, preds["mean"])
        scores.append(score)

    if not scores: return float('inf')
    # Use item() to convert JAX array to Python float immediately
    return float(jnp.mean(jnp.array(scores)).item())

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
    """
    Parallel optimization using ThreadPoolExecutor.
    Includes GridTracker to report progress.
    """
    
    # Validation
    max_possible_windows = (y.shape[0] - test_size - 4) // step_size + 1
    if max_possible_windows < 1: return grid[0]
    actual_windows = min(n_windows, max_possible_windows)

    # --- TRACKER SETUP ---
    tracker = GridTracker()
    tracker.set_total(len(grid))

    # Worker Wrapper
    def _worker(cfg):
        try:
            score = cross_validation(y, X, test_size, actual_windows, step_size, cfg)
            # If successful (even if score is poor), increment counter
            tracker.increment_success() 
            return score
        except Exception as e:
            if verbose: print(f"Config failed: {e}")
            return float('inf')

    # Parallel Execution
    scores = []
    if n_jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            scores = list(executor.map(_worker, grid))
    else:
        scores = [_worker(g) for g in grid]

    # --- REPORT PROGRESS ---
    print(f"  [AutoMFLES] Grid Search: Planned {tracker.total} runs | Completed {tracker.success} successfully.")

    # Selection
    best_idx = np.argmin(scores)
    return grid[best_idx]


# =============================================================================
# 4. CLASS WRAPPER
# =============================================================================

class AutoMFLES:
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
        n_jobs: int = 4 # Defaults to 4 threads
    ):
        if test_size <= 0: raise ValueError("test_size must be > 0")
        if n_windows <= 0: raise ValueError("n_windows must be > 0")

        self.test_size = test_size
        self.season_length = season_length
        self.n_windows = n_windows
        self.config = config
        self.step_size = step_size if step_size is not None else test_size
        self.metric = metric
        self.verbose = verbose
        self.prediction_intervals = prediction_intervals
        self.alias = alias
        self.n_jobs = n_jobs
        
        self.model_ = None
        self.best_params_ = None
        self.sigma_ = 0.0
        self.scaling_stats_ = None

    def fit(self, y: Union[np.ndarray, jnp.ndarray], X: Optional[jnp.ndarray] = None) -> "AutoMFLES":
        y = _ensure_float(y)
        
        if X is not None:
            X = jnp.asarray(X, dtype=jnp.float32)
            if X.ndim == 1: X = X.reshape(-1, 1)
            self.scaling_stats_ = _get_stats(X)
            X = _standardize_data(X, *self.scaling_stats_)

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

        model = MFLES(
            verbose=int(self.verbose),
            conformal_params=self.prediction_intervals,
            alias=self.alias,
        )
        
        final_sl = self.best_params_.get("seasonal_period", self.season_length)
        fit_params = {k: v for k, v in self.best_params_.items() if k != "seasonal_period"}

        model.fit(y=y, seasonal_period=final_sl, X=X, **fit_params)

        self.model_ = {"model": model, "fitted": model.fitted_}
        self.sigma_ = _calculate_sigma(y - model.fitted_)
        return self

    def predict(self, h: int, X: Optional[jnp.ndarray] = None, level: Optional[List[int]] = None) -> Dict[str, Any]:
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
