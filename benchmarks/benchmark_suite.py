"""
File: benchmark_suite.py

High-level Purpose:
    Provides the per-model benchmark execution engine used by the evaluation
    harness to measure forecasting runtime and accuracy consistently across
    Chronax and StatsForecast implementations.

Problem Solved:
    Centralizes benchmark data preparation, lazy model loading, metric
    computation, and result serialization so orchestration code can run isolated
    model jobs in subprocesses with uniform outputs.

Architectural Role:
    Operates as the worker module in the benchmark subsystem. It is invoked by
    `run_benchmark.py` (or directly via CLI) and returns machine-readable JSON
    records consumed by the orchestrator.

Major Classes/Functions:
    - `SafeEncoder`: JSON encoder for NumPy/JAX scalar/array types.
    - `BenchmarkLogger`: Record collector and CSV writer.
    - `SFWrapper` / `ChronaxWrapper`: Library-specific adapter wrappers.
    - `ModelRegistry`: Lazy model resolver.
    - `run_single_model`: Core single-task benchmark execution entry point.

External Dependencies:
    - `pandas`, `numpy`
    - Optional runtime imports: `jax`, `statsforecast`, `yaml`
    - Standard library: `json`, `time`, `os`, `sys`, `warnings`

Expected Inputs and Outputs:
    - Input: dataset selection, model metadata, benchmark configuration, and
      optional target column.
    - Output: JSON payload containing timing/accuracy metrics or forecast-mode
      data suitable for downstream plotting and archival.

Example:
    >>> # CLI usage
    >>> # python benchmark_suite.py --model ARIMA --library chronax \\
    >>> #     --dataset Trend --config benchmarks/config.yaml

Assumptions:
    - Model names are registered in `ModelRegistry`.
    - Dataset files, when provided, are readable CSV inputs.
    - Required optional libraries are installed in the active environment.

Side Effects:
    - Prints status messages to stdout.
    - Creates output files when logger save paths are used.
    - Executes lazy imports for model backends at runtime.

Author:
    Auto-documented
Date:
    2026-02-21
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type
import warnings

# Suppress specific numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

# Add parent directory of chronax package for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ==========================================
# 1. CONFIGURATION
# ==========================================

SCALES = [100, 500, 1_000, 5_000, 10_000, 50_000]
HORIZON = 24
SEASONALITY = 24
CONFIDENCE_LEVEL = 95
RANDOM_SEED = 42
N_ITERATIONS = 5


# ==========================================
# 2. LOGGER & UTILS
# ==========================================

import json
class SafeEncoder(json.JSONEncoder):
    """JSON encoder that serializes NumPy/JAX scalar and array types."""

    def default(self, obj: Any) -> Any:
        """
        Convert unsupported numeric objects into JSON-safe values.

        Detailed Description:
            Intercepts NumPy and JAX scalar/array objects and converts them to
            native Python numbers/lists before JSON serialization.

        Args:
            obj (Any): Arbitrary object encountered by JSON serialization.

        Returns:
            Any: JSON-serializable replacement value.

        Raises:
            TypeError: When object cannot be serialized by base encoder.

        Side Effects:
            None.

        Example:
            >>> SafeEncoder().default(np.array([1, 2]))

        Notes:
            JAX import is lazy to avoid hard dependency in non-JAX runs.
        """
        import numpy as np
        try:
            import jax.numpy as jnp
        except ImportError:
            jnp = None
            
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if jnp is not None:
             if isinstance(obj, jnp.integer):
                return int(obj)
             if isinstance(obj, jnp.floating):
                return float(obj)
             if isinstance(obj, jnp.ndarray):
                return obj.tolist()
        return super().default(obj)

class BenchmarkLogger:
    """In-memory benchmark record collector and CSV writer."""

    def __init__(self, output_dir: str = "results") -> None:
        """Initialize logger output directory and in-memory record buffer."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, output_dir)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.records: List[Dict[str, Any]] = []

    def log(self, 
            dataset: str, 
            length: int, 
            model: str, 
            time_cold: float, 
            time_warm: float, 
            time_interval: Optional[float] = None,
            overhead_pct: Optional[float] = None,
            mape: Optional[float] = None,
            mae: Optional[float] = None,
            rmse: Optional[float] = None,
            mase: Optional[float] = None) -> None:
        """
        Append one benchmark record to the in-memory log and print a summary line.

        Detailed Description:
            Builds a single record from the provided timing and accuracy metrics,
            appends it to the instance's records list, and prints a one-line
            summary to stdout (model name, series length, warm/cold times, and
            optional MAPE/MAE/RMSE/MASE). Used by the benchmark harness to
            accumulate results before saving to CSV.

        Args:
            dataset (str): Name or identifier of the dataset (e.g. "Trend", "AirlinePassengers").
            length (int): Series length (number of observations) for this run.
            model (str): Model identifier (e.g. "Chronax_ARIMA", "StatsForecast_Naive").
            time_cold (float): Cold-start runtime in seconds (first run including compile/load).
            time_warm (float): Warm runtime in seconds (average of repeated runs).
            time_interval (float, optional): Time in seconds for run with prediction intervals.
            overhead_pct (float, optional): Percentage overhead of interval run vs point forecast.
            mape (float, optional): Mean absolute percentage error.
            mae (float, optional): Mean absolute error.
            rmse (float, optional): Root mean squared error.
            mase (float, optional): Mean absolute scaled error.

        Returns:
            None. Mutates self.records and prints to stdout.

        Raises:
            None. Missing optional metrics are stored as None in the record.

        Side Effects:
            Appends to self.records; prints one line to stdout.

        Example:
            >>> logger.log("Trend", 1000, "Chronax_ARIMA", 2.5, 0.1, mape=12.3, mae=1.5)

        Notes:
            Role: Central recording point for benchmark results before batch
            write to CSV via save().
        """
        record = {
            'Dataset': dataset,
            'Length': length,
            'Model': model,
            'Time_Cold_Sec': round(time_cold, 6),
            'Time_Warm_Sec': round(time_warm, 6),
            'Time_Interval_Sec': round(time_interval, 6) if time_interval else None,
            'Interval_Overhead_Pct': round(overhead_pct, 2) if overhead_pct else None,
            'MAPE': round(mape, 4) if mape is not None else None,
            'MAE': round(mae, 6) if mae is not None else None,
            'RMSE': round(rmse, 6) if rmse is not None else None,
            'MASE': round(mase, 6) if mase is not None else None
        }
        self.records.append(record)
        
        # Real-time console feedback
        metrics_str = ""
        if record['MAPE'] is not None:
            metrics_str += f" | MAPE={record['MAPE']}%"
        if record['MAE'] is not None:
            metrics_str += f" | MAE={record['MAE']}"
        if record['RMSE'] is not None:
            metrics_str += f" | RMSE={record['RMSE']}"
        if record['MASE'] is not None:
            metrics_str += f" | MASE={record['MASE']}"
        print(f"  > [{model}] L={length}: Warm={record['Time_Warm_Sec']}s | Cold={record['Time_Cold_Sec']}s{metrics_str}")

    def save(self, filename: str = "benchmark_results.csv") -> pd.DataFrame:
        """
        Write all accumulated benchmark records to a CSV file and return a DataFrame.

        Detailed Description:
            Converts the in-memory records list into a pandas DataFrame, writes
            it to a CSV file under the logger's output directory, and prints
            the output path. The DataFrame is returned for further analysis or
            plotting. Does not clear the records list, so additional log()
            calls can be made and save() called again.

        Args:
            filename (str, optional): Name of the CSV file under output_dir;
                default "benchmark_results.csv".

        Returns:
            pd.DataFrame: DataFrame built from self.records, with columns
                Dataset, Length, Model, Time_Cold_Sec, Time_Warm_Sec, etc.

        Raises:
            None. IO errors from writing the file may propagate.

        Side Effects:
            Creates or overwrites a file under self.output_dir; prints path to stdout.

        Example:
            >>> df = logger.save("benchmark_20260221.csv")

        Notes:
            Role: Persists benchmark session results for reporting and
            post-processing; used at the end of a benchmark run.
        """
        df = pd.DataFrame(self.records)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"\n✅ Results saved to {path}")
        return df


# ==========================================
# 3. SYNTHETIC DATA
# ==========================================

def generate_series(series_type: str, length: int, seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic univariate time series for benchmarking.

    Detailed Description:
        Produces a deterministic-plus-noise series of a given length and type.
        "Trend" is linear trend plus Gaussian noise; "Seasonality" is a
        sinusoidal seasonal pattern (period 24) plus noise; "Stochastic" is
        a random walk (cumulative sum of standard normal increments). Used to
        stress-test models across different data characteristics without
        external data files.

    Args:
        series_type (str): One of "Trend", "Seasonality", "Stochastic".
            Any other value raises ValueError.
        length (int): Number of observations to generate.
        seed (int, optional): Random seed for reproducibility; default 42.

    Returns:
        np.ndarray: 1D float32 array of shape (length,) containing the
            synthetic series.

    Raises:
        ValueError: If series_type is not "Trend", "Seasonality", or "Stochastic".

    Side Effects:
        None. Uses fixed RNG state via seed.

    Example:
        >>> y = generate_series("Seasonality", 500, seed=0)

    Notes:
        Role: Supplies standardized synthetic inputs for the benchmark suite
        when no external dataset is specified; enables comparable runs across
        environments.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    noise = rng.standard_normal(length).astype(np.float32)
    
    if series_type == 'Trend':
        return (0.05 * t + noise).astype(np.float32)
    elif series_type == 'Seasonality':
        seasonal = 2.0 * np.sin(2 * np.pi * t / 24)
        return (seasonal + noise).astype(np.float32)
    elif series_type == 'Stochastic':
        return np.cumsum(noise).astype(np.float32)
    else:
        raise ValueError(f"Unknown series_type: {series_type}")


def prepare_inputs(y_array: np.ndarray, return_jax: bool = True) -> Tuple[Optional[Any], pd.DataFrame]:
    """
    Convert a raw numpy time series into Chronax- and StatsForecast-ready inputs.

    Detailed Description:
        Takes a 1D numpy array of training observations and produces two
        representations: (1) an optional JAX array (float32) for Chronax
        models, created only if return_jax is True and JAX is importable;
        (2) a pandas DataFrame with columns unique_id, ds (hourly datetime
        index from 2020-01-01), and y, as required by StatsForecast. This
        allows the same series to be passed to either library in the
        benchmark pipeline without duplicating data-loading logic.

    Args:
        y_array (np.ndarray): 1D array of training values (any length).
        return_jax (bool, optional): If True, attempt to build a JAX array
            for Chronax; if False or JAX unavailable, first element of the
            return tuple is None. Default True.

    Returns:
        Tuple[Optional[Any], pd.DataFrame]: (jax_input, sf_input). jax_input
            is a jnp.ndarray or None; sf_input is a DataFrame with
            unique_id, ds, y.

    Raises:
        None. JAX import failure results in None for the first element.

    Side Effects:
        None. Lazy JAX import only when return_jax is True.

    Example:
        >>> jax_arr, sf_df = prepare_inputs(np.array([1.0, 2.0, 3.0]))

    Notes:
        Role: Single point of conversion from raw series to library-specific
        formats in the benchmark worker; keeps run_single_model independent
        of library input conventions.
    """
    jax_input = None
    if return_jax:
        try:
            # LAZY IMPORT: Only import JAX when actually needed
            import jax.numpy as jnp
            jax_input = jnp.array(y_array, dtype=jnp.float32)
        except ImportError:
            pass
    
    # StatsForecast Input: DataFrame Construction
    n = len(y_array)
    sf_input = pd.DataFrame({
        'unique_id': ['series_0'] * n,
        'ds': pd.date_range(start='2020-01-01', periods=n, freq='h'),
        'y': y_array
    })
    
    return jax_input, sf_input


# ==========================================
# 4. WRAPPERS (LAZY IMPORTS)
# ==========================================

class SFWrapper:
    """StatsForecast wrapper with lazy imports."""
    
    def __init__(
        self,
        model_cls: Type[Any],
        model_params: Dict[str, Any],
        horizon: int,
        seasonality: int,
    ) -> None:
        """
        Construct a StatsForecast model wrapper with lazy backend import.

        Detailed Description:
            Imports StatsForecast only at construction time to avoid loading
            the library when running Chronax-only benchmarks. Applies
            model-specific parameter name mappings (e.g. period -> season_length
            for MSTL, removal of season_type for HoltWinters), instantiates
            the provided model class with the (possibly adjusted) params, and
            wraps it in a StatsForecast instance with hourly frequency and
            n_jobs=1. The wrapper is then used by the benchmark runner to
            call fit_predict on a prepared DataFrame.

        Args:
            model_cls (Type[Any]): The StatsForecast model class (e.g. Naive, ARIMA).
            model_params (Dict[str, Any]): Keyword arguments for the model
                constructor; may be modified for name compatibility.
            horizon (int): Forecast horizon (number of steps) for subsequent
                fit_predict calls.
            seasonality (int): Seasonal period used for parameter mapping and
                configuration.

        Returns:
            None. Initializes self.model_obj, self.sf, self.horizon, self.seasonality.

        Raises:
            ImportError: If statsforecast is not installed.

        Side Effects:
            Performs lazy import of statsforecast; may mutate model_params copy.

        Example:
            >>> wrapper = SFWrapper(ARIMA, {"order": (1,1,1)}, horizon=24, seasonality=24)

        Notes:
            Role: Abstracts StatsForecast setup and parameter naming so the
            benchmark loop can treat all models uniformly via fit_predict(df).
        """
        # LAZY IMPORT: Only import when wrapper is instantiated
        from statsforecast import StatsForecast
        
        self.horizon = horizon
        self.seasonality = seasonality
        
        final_params = model_params.copy()
        
        # Handle param name mappings
        if model_cls.__name__ in ['MSTL', 'SFMSTL']:
            if 'period' in final_params:
                final_params['season_length'] = final_params.pop('period')
        
        if model_cls.__name__ in ['HoltWinters', 'SFHoltWinters']:
            if 'season_type' in final_params:
                final_params.pop('season_type')

        self.model_obj = model_cls(**final_params)
        
        self.sf = StatsForecast(
            models=[self.model_obj],
            freq='h',
            n_jobs=1 
        )
        
    def fit_predict(self, df: pd.DataFrame, intervals: bool = False) -> pd.DataFrame:
        """
        Run StatsForecast fit and forecast and return the forecast DataFrame.

        Detailed Description:
            Calls the wrapped StatsForecast instance's forecast method with the
            provided DataFrame (which must have unique_id, ds, y), the
            configured horizon, and optionally level=[95] for prediction
            intervals. If interval generation fails (e.g. model does not
            support it), falls back to point forecast and prints a warning when
            intervals=True. Used by the benchmark worker to obtain predictions
            and measure timing for StatsForecast models.

        Args:
            df (pd.DataFrame): Training data with columns unique_id, ds, y.
            intervals (bool, optional): If True, request 95% prediction
                intervals; on failure, fall back to point forecast. Default False.

        Returns:
            pd.DataFrame: Forecast output from StatsForecast (point forecasts
                and optionally interval columns).

        Raises:
            Exception: Re-raised from StatsForecast.forecast when intervals
                is False and the call fails.

        Side Effects:
            May print a warning to stdout if intervals=True and interval
            generation fails.

        Example:
            >>> preds = sf_wrapper.fit_predict(train_df, intervals=True)

        Notes:
            Role: Single entry point for running a StatsForecast model in the
            benchmark; return value is used to extract point forecasts and
            compute accuracy metrics.
        """
        level = [95] if intervals else []
        try:
            forecasts = self.sf.forecast(df=df, h=self.horizon, level=level)
        except Exception as e:
            if intervals:
                print(f"    [SFWrapper] Warning: Interval generation failed ({e}). Falling back to point forecast.")
                forecasts = self.sf.forecast(df=df, h=self.horizon)
            else:
                raise e
        return forecasts


class ChronaxWrapper:
    """Chronax wrapper with lazy imports."""
    
    def __init__(
        self,
        model_cls: Type[Any],
        model_params: Dict[str, Any],
        horizon: int,
        seasonality: int,
    ) -> None:
        """
        Construct a Chronax model wrapper with lazy JAX import.

        Detailed Description:
            Imports JAX only at construction time so that StatsForecast-only
            runs do not load JAX. Copies model params and applies
            Chronax-specific mappings (e.g. MFLES season_length/seasonal_period
            -> extra_fit_params; AutoETS default season_length from
            seasonality). Instantiates the Chronax model class and stores it
            for use in fit_predict. The wrapper provides a uniform fit_predict(y_array)
            interface for the benchmark runner regardless of whether the model
            uses forecast() or fit()+predict().

        Args:
            model_cls (Type[Any]): The Chronax forecaster class (e.g. ARIMA, Naive).
            model_params (Dict[str, Any]): Keyword arguments for the model
                constructor; may be modified and/or moved to extra_fit_params.
            horizon (int): Forecast horizon for predict/forecast calls.
            seasonality (int): Seasonal period used for default params (e.g. AutoETS).

        Returns:
            None. Initializes self.model, self.horizon, self.seasonality,
            self.extra_fit_params, self.key.

        Raises:
            ImportError: If jax is not available.

        Side Effects:
            Lazy import of jax.random; may mutate a copy of model_params.

        Example:
            >>> wrapper = ChronaxWrapper(ARIMA, {"order": (1,1,1)}, 24, 24)

        Notes:
            Role: Abstracts Chronax model construction and parameter naming
            so the benchmark loop can call fit_predict(y_array) for all
            Chronax models.
        """
        # LAZY IMPORT: Only import JAX when wrapper is instantiated
        from jax import random
        
        self.horizon = horizon
        self.seasonality = seasonality
        self.key = random.PRNGKey(0)
        
        final_params = model_params.copy()
        
        self.extra_fit_params = {}
        if model_cls.__name__ == 'MFLES':
            for key in ['season_length', 'seasonal_period']:
                if key in final_params:
                    self.extra_fit_params['seasonal_period'] = final_params.pop(key)

        if 'season_length' not in final_params and model_cls.__name__ in ['AutoETS']:
            final_params['season_length'] = seasonality
            
        self.model = model_cls(**final_params)

    def _dummy_predict_impl(self, y_array: Any) -> Dict[str, Any]:
        """
        Run the appropriate fit-and-predict path for the wrapped Chronax model.

        Detailed Description:
            If the model has a forecast() method and no extra_fit_params (e.g.
            no MFLES-style seasonal_period), calls forecast(h=horizon, y=y_array)
            for a one-shot fit-and-forecast. Otherwise fits the model with
            fit(y_array, **extra_fit_params) then calls predict(h=horizon).
            Returns the prediction dictionary (e.g. {"mean": array}); the
            benchmark runner typically extracts the "mean" array and optionally
            synchronizes with block_until_ready().

        Args:
            y_array (Any): Training series; JAX array or array-like accepted
                by the underlying model's fit or forecast.

        Returns:
            Dict[str, Any]: Prediction dict with at least "mean" key; may
                include interval keys depending on the model.

        Raises:
            Any exception raised by the underlying model's fit, forecast, or predict.

        Side Effects:
            May mutate the wrapped model's internal state (fit).

        Example:
            >>> out = wrapper._dummy_predict_impl(jnp.array([1.0, 2.0, 3.0]))

        Notes:
            Role: Internal adapter so fit_predict() can support both
            forecast-only and fit-then-predict Chronax models uniformly.
        """
        if hasattr(self.model, 'forecast') and not self.extra_fit_params:
            return self.model.forecast(h=self.horizon, y=y_array)
        else:
            self.model.fit(y_array, **self.extra_fit_params)
            return self.model.predict(h=self.horizon)

    def fit_predict(self, y_array: Any, intervals: bool = False) -> Any:
        """
        Produce point forecasts from the wrapped Chronax model and sync JAX.

        Detailed Description:
            Calls _dummy_predict_impl(y_array) to obtain the prediction
            dictionary, then forces completion of the "mean" array computation
            via block_until_ready() so that benchmark timings include full JAX
            execution. Returns the "mean" array for the benchmark worker to
            compare with actuals and compute MAPE/MAE/RMSE/MASE. The intervals
            argument is accepted for API compatibility with SFWrapper but
            is not used; Chronax interval handling would be model-specific.

        Args:
            y_array (Any): Training series (JAX array or array-like).
            intervals (bool, optional): Ignored; kept for signature compatibility
                with SFWrapper.fit_predict. Default False.

        Returns:
            Any: The "mean" forecast array from the model (typically
                jnp.ndarray of shape (horizon,)).

        Raises:
            Any exception from _dummy_predict_impl or JAX runtime.

        Side Effects:
            Triggers JAX computation (block_until_ready); may mutate model state.

        Example:
            >>> mean_fc = chronax_wrapper.fit_predict(jnp.array(train_data))

        Notes:
            Role: Single entry point for running a Chronax model in the
            benchmark and obtaining point forecasts for accuracy evaluation.
        """
        preds = self._dummy_predict_impl(y_array)
        preds['mean'].block_until_ready()        
        return preds['mean']


# ==========================================
# 5. MODEL REGISTRY (LAZY IMPORTS)
# ==========================================

class ModelRegistry:
    """Central registry with lazy model loading."""
    
    _MODELS = {
        # Simple Baselines
        "Naive": {"params": {}},
        "SeasonalNaive": {"params": {}},
        "HistoricAverage": {"params": {}},
        "RandomWalkWithDrift": {"params": {}},
        "WindowAverage": {"params": {"window_size": 24}},
        "SeasonalWindowAverage": {"params": {"window_size": 24, "season_length": 24}},
        # Smoothing Models
        "SimpleExponentialSmoothing": {"params": {}},
        "SeasonalExponentialSmoothing": {"params": {}},
        "Holt": {"params": {}},
        "HoltWinters": {"params": {}},
        # Intermittent Demand Models
        "ADIDA": {"params": {}},
        "CrostonClassic": {"params": {}},
        "TSB": {"params": {}},
        "IMAPA": {"params": {}},
        # Decomposition & Volatility
        "MSTL": {"params": {}},
        "GARCH": {"params": {}},
        # Auto Models (base + Auto pairs)
        "AutoETS": {"params": {}},
        "Theta": {"params": {}},
        "AutoTheta": {"params": {}},
        "AutoCES": {"params": {}},
        "ARIMA": {"params": {"order": (1, 1, 1)}},
        "AutoARIMA": {"params": {}},
        "MFLES": {"params": {}},
        "AutoMFLES": {"params": {}},
        "TBATS": {"params": {}},
        "AutoTBATS": {"params": {}},
    }

    @staticmethod
    def get_chronax_model(name: str) -> Optional[Type[Any]]:
        """
        Resolve a registered model name to its Chronax class via lazy import.

        Detailed Description:
            Maps the given string name (e.g. "ARIMA", "Naive") to the
            corresponding Chronax forecaster class by performing a conditional
            import of the appropriate module (e.g. arima, naive) and
            returning the class. This defers importing heavy or optional
            dependencies until the model is actually requested. Returns None
            if the name is unknown or the import fails (ImportError,
            AttributeError, NameError).

        Args:
            name (str): Registered model name; must be a key in _MODELS.

        Returns:
            Optional[Type[Any]]: The Chronax model class, or None if
                unavailable or not found.

        Raises:
            None. Import and attribute errors are caught and result in None.

        Side Effects:
            May import chronax submodules (e.g. arima, auto_arima) on first
            use for a given name.

        Example:
            >>> cls = ModelRegistry.get_chronax_model("ARIMA")

        Notes:
            Role: Central resolver for Chronax models in the benchmark
            worker; used by get_model_entry() and run_single_model().
        """
        try:
            # Import the centralized Chronax models package and resolve by attribute name.
            from chronax import models
            return getattr(models, name, None)
        except Exception as exc:
            # Surface the underlying import error instead of failing silently.
            import sys
            import traceback

            msg = (
                f"[Chronax][ModelRegistry] Failed to import Chronax model '{name}': "
                f"{exc!r}"
            )
            print(msg, file=sys.stderr)
            traceback.print_exc()
            return None

    @staticmethod
    def get_sf_model(name: str) -> Optional[Type[Any]]:
        """
        Resolve a registered model name to its StatsForecast model class.

        Detailed Description:
            Looks up the model in _MODELS and uses the optional "sf_name" key
            for a different StatsForecast class name if needed; otherwise
            uses the same name. Imports statsforecast.models lazily and
            returns getattr(sf_models, sf_name, None). Returns None if
            statsforecast is not installed or the attribute is missing. Used
            so that the benchmark can support both Chronax and StatsForecast
            for the same logical model (e.g. "ARIMA") without loading both
            libraries up front.

        Args:
            name (str): Registered model name (key in _MODELS).

        Returns:
            Optional[Type[Any]]: The StatsForecast model class, or None.

        Raises:
            None. ImportError is caught and returns None.

        Side Effects:
            May import statsforecast.models on first call.

        Example:
            >>> sf_cls = ModelRegistry.get_sf_model("ARIMA")

        Notes:
            Role: Central resolver for StatsForecast models; used by
            get_model_entry() and run_single_model() when library is
            "statsforecast".
        """
        config = ModelRegistry._MODELS.get(name, {})
        sf_name = config.get("sf_name", name)
        
        try:
            # LAZY IMPORT
            from statsforecast import models as sf_models
            return getattr(sf_models, sf_name, None)
        except ImportError:
            return None

    @staticmethod
    def get_model_entry(model_name: str) -> Dict[str, Any]:
        """
        Return the full model entry (Chronax class, StatsForecast class, params) for a name.

        Detailed Description:
            Validates that model_name exists in _MODELS, then builds and
            returns a dictionary with "chronax_cls" (from get_chronax_model),
            "sf_cls" (from get_sf_model), and "params" (the default constructor
            params for the model). This is the single entry point used by
            run_single_model to obtain the correct class and params for either
            library, and to merge in any config overrides.

        Args:
            model_name (str): Registered model name; must be a key in _MODELS.

        Returns:
            Dict[str, Any]: Keys "chronax_cls", "sf_cls", "params". Either
                class may be None if that backend is unavailable.

        Raises:
            ValueError: If model_name is not in _MODELS.

        Side Effects:
            May trigger lazy imports via get_chronax_model and get_sf_model.

        Example:
            >>> entry = ModelRegistry.get_model_entry("ARIMA")
            >>> entry["params"]["order"]

        Notes:
            Role: Unified lookup for the benchmark worker; ensures consistent
            model and parameter resolution across Chronax and StatsForecast runs.
        """
        if model_name not in ModelRegistry._MODELS:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        
        config = ModelRegistry._MODELS[model_name]
        
        return {
            "chronax_cls": ModelRegistry.get_chronax_model(model_name),
            "sf_cls": ModelRegistry.get_sf_model(model_name),
            "params": config["params"]
        }


# ==========================================
# 6. METRICS (from TempusBench)
# ==========================================

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error in percentage units (0--100).

    Detailed Description:
        Computes the mean of |y_true - y_pred| / (|y_true| + epsilon) and
        multiplies by 100 so the result is in percentage points. Used in the
        benchmark suite to report scale-independent accuracy for each
        model/dataset run. Epsilon avoids division by zero when actuals are zero.

    Args:
        y_true (np.ndarray): Actual (ground truth) values, typically test set.
        y_pred (np.ndarray): Predicted values; same shape as y_true.

    Returns:
        float: MAPE in percentage units (e.g. 12.5 for 12.5%).

    Raises:
        None.

    Side Effects:
        None.

    Example:
        >>> calculate_mape(y_test, y_hat)

    Notes:
        Role: Standard accuracy metric reported in benchmark CSV and logs;
        aligns with TempusBench-style evaluation.
    """
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error as a Python float.

    Detailed Description:
        Returns the mean of the absolute element-wise errors between y_true
        and y_pred, as a scalar float (via .item()) for JSON serialization and
        logging. Scale-dependent; used alongside MAPE/RMSE/MASE in benchmark
        results to give interpretable magnitude of error.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values; same shape as y_true.

    Returns:
        float: Mean absolute error in same units as the data.

    Raises:
        None.

    Side Effects:
        None.

    Example:
        >>> calculate_mae(y_test, y_hat)

    Notes:
        Role: Benchmark metric for point-forecast accuracy; from TempusBench.
    """
    return np.mean(np.abs(y_true - y_pred)).item()

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error as a Python float.

    Detailed Description:
        Computes the square root of the mean squared error and returns it as
        a scalar float for benchmark logging and JSON output. Scale-dependent;
        penalizes large errors more than MAE. Used with MAPE/MAE/MASE for
        comprehensive accuracy reporting.

    Args:
        y_true (np.ndarray): Actual values.
        y_pred (np.ndarray): Predicted values; same shape as y_true.

    Returns:
        float: RMSE in same units as the data.

    Raises:
        None.

    Side Effects:
        None.

    Example:
        >>> calculate_rmse(y_test, y_hat)

    Notes:
        Role: Standard L2 benchmark metric; from TempusBench.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2)).item()

def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Scaled Error using in-sample naive forecast.

    Detailed Description:
        Scales the mean absolute error of the forecasts by the mean absolute
        error of a one-step naive forecast (|y_true[1:] - y_true[:-1]|).
        Returns a scalar float for benchmark output. MASE < 1 indicates
        the model beats the naive baseline; scale-independent. Denominator
        is clamped to at least 1e-10 to avoid division by zero.

    Args:
        y_true (np.ndarray): Actual values (test). Length at least 2.
        y_pred (np.ndarray): Predicted values; same length as y_true.

    Returns:
        float: MASE value; dimensionless.

    Raises:
        None. Very short series may give extreme values.

    Side Effects:
        None.

    Example:
        >>> calculate_mase(y_test, y_hat)

    Notes:
        Role: Scale-independent benchmark metric; from TempusBench. Uses
        in-sample naive scaling rather than seasonal naive.
    """
    denom = np.maximum(1e-10, np.mean(np.abs(y_true[1:] - y_true[:-1])))
    mase = np.mean(np.abs(y_true - y_pred)) / denom
    return mase.item()


# ==========================================
# 7. SINGLE MODEL RUNNER (for subprocess calls)
# ==========================================

def run_single_model(
    model_name: str,
    library: str,
    dataset_path: str,
    config: Dict[str, Any],
    target_column: Optional[str] = None,
    forecast_mode: bool = False,
) -> str:
    """
    Execute a single model benchmark for one library and dataset, returning JSON.

    Detailed Description:
        This is the main worker entry point invoked by run_benchmark.py in a
        subprocess. It loads the benchmark config, resolves the model from
        ModelRegistry (Chronax or StatsForecast), loads or generates the
        dataset (synthetic via generate_series or external CSV), and prepares
        library-specific inputs via prepare_inputs. It then runs the model
        once for cold timing, computes accuracy metrics (MAPE, MAE, RMSE,
        MASE) using the test slice, runs warm iterations for average timing,
        and optionally measures interval overhead. The result is serialized
        as JSON using SafeEncoder (for JAX/NumPy types) and returned as a
        string. In forecast_mode, returns train/test/predictions and MAPE for
        plotting and CSV export instead of full benchmark metrics. Exceptions
        are caught and returned as JSON with an "error" key so the orchestrator
        can log and continue.

    Args:
        model_name (str): Registered model name (e.g. "ARIMA", "Naive").
        library (str): Either "chronax" or "statsforecast"; selects wrapper
            and input format.
        dataset_path (str): For synthetic: "Trend", "Seasonality", or
            "Stochastic". For external: path to CSV file.
        config (Dict[str, Any]): Full benchmark config with "experiment"
            (horizon, seasonality, n_iterations, scales) and optional "models"
            list for param overrides.
        target_column (Optional[str]): If dataset_path is a CSV, optional
            column name for the target series; otherwise "y" or last column.
        forecast_mode (bool, optional): If True, return forecast artifact
            payload (y_train, y_test, predictions, MAPE) for plotting; if
            False, return standard benchmark metrics (timing, MAPE, MAE, etc.).
            Default False.

    Returns:
        str: JSON-serialized result. In benchmark mode: Model, Dataset,
            Length, Time_Cold_Sec, Time_Warm_Sec, Time_Interval_Sec,
            Interval_Overhead_Pct, MAPE, MAE, RMSE, MASE. In forecast mode:
            Model, Dataset, Length, y_train, y_test, predictions, MAPE,
            forecast_mode=True. On error: {"error": "<message and traceback>"}.

    Raises:
        None. Exceptions are caught and returned in the JSON "error" field.

    Side Effects:
        Prints progress to stdout; may create/write files only if caller
        redirects or uses logger (this function itself does not write CSV).
        Triggers lazy imports (JAX, statsforecast) when loading the model.

    Example:
        >>> result_json = run_single_model("ARIMA", "chronax", "Trend", config)
        >>> data = json.loads(result_json)

    Notes:
        Role: Isolated benchmark worker so each model run has a clean process
        and consistent JSON contract for the orchestrator in run_benchmark.py.
    """
    import json
    
    experiment_cfg = config['experiment']
    horizon = experiment_cfg['horizon']
    seasonality = experiment_cfg['seasonality']
    n_iterations = experiment_cfg.get('n_iterations', 5)
    
    # Get model from registry
    entry = ModelRegistry.get_model_entry(model_name)
    model_params = entry.get('params', {}).copy()
    
    # Override with config params
    for mod in config.get('models', []):
        if mod['name'] == model_name and mod['library'] == library:
            model_params.update(mod.get('params', {}))
            break
    
    # Load data
    if os.path.isfile(dataset_path):
        # External dataset
        df = pd.read_csv(dataset_path)
        if target_column and target_column in df.columns:
            raw_data = df[target_column].values.astype(np.float32)
        elif 'y' in df.columns:
            raw_data = df['y'].values.astype(np.float32)
        else:
            raw_data = df.iloc[:, -1].values.astype(np.float32)
        total_length = len(raw_data)
        y_train = raw_data[:total_length - horizon]
        y_test = raw_data[total_length - horizon:]
        scale = len(y_train)
    else:
        # Synthetic dataset
        scale = config['experiment']['scales'][0] if config['experiment']['scales'] else 100
        total_length = scale + horizon
        raw_data = generate_series(dataset_path, total_length)
        y_train = raw_data[:scale]
        y_test = raw_data[scale:]
    
    # Prepare inputs
    jax_input, sf_input = prepare_inputs(y_train, return_jax=(library == 'chronax'))
    
    try:
        if library == 'chronax':
            chronax_cls = entry['chronax_cls']
            if chronax_cls is None:
                return json.dumps({"error": f"Model {model_name} not available in Chronax"})
            
            # Cold start
            t0 = time.perf_counter()
            wrapper = ChronaxWrapper(
                model_cls=chronax_cls,
                model_params=model_params,
                horizon=horizon,
                seasonality=seasonality
            )
            preds = wrapper.fit_predict(jax_input)
            t_cold = time.perf_counter() - t0
            
            pred_values = np.array(preds)
            
        else:  # statsforecast
            sf_cls = entry['sf_cls']
            if sf_cls is None:
                return json.dumps({"error": f"Model {model_name} not available in StatsForecast"})
            
            # Cold start
            t0 = time.perf_counter()
            wrapper = SFWrapper(
                model_cls=sf_cls,
                model_params=model_params,
                horizon=horizon,
                seasonality=seasonality
            )
            preds = wrapper.fit_predict(sf_input)
            t_cold = time.perf_counter() - t0
            
            pred_values = preds.iloc[:, -1].values

        # --- FORECAST MODE RETURN ---
        if forecast_mode:
            result = {
                'Model': f"{library.capitalize()}_{model_name}",
                'Dataset': dataset_path,
                'Length': scale,
                'y_train': y_train.tolist(),
                'y_test': y_test.tolist(),
                'predictions': pred_values.tolist(),
                'forecast_mode': True
            }
            # Add metrics just in case
            result['MAPE'] = calculate_mape(y_test, pred_values)
            return json.dumps(result, cls=SafeEncoder)
        
        # Calculate metrics for Benchmark Mode
        mape = calculate_mape(y_test, pred_values)
        mae = calculate_mae(y_test, pred_values)
        rmse = calculate_rmse(y_test, pred_values)
        mase = calculate_mase(y_test, pred_values)
        
        # Warm runs
        warm_times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            if library == 'chronax':
                wrapper.fit_predict(jax_input, intervals=False)
            else:
                wrapper.fit_predict(sf_input, intervals=False)
            warm_times.append(time.perf_counter() - t0)
        t_warm = sum(warm_times) / len(warm_times)
        
        # Interval overhead
        t0 = time.perf_counter()
        if library == 'chronax':
            wrapper.fit_predict(jax_input, intervals=True)
        else:
            wrapper.fit_predict(sf_input, intervals=True)
        t_interval = time.perf_counter() - t0
        overhead = ((t_interval - t_warm) / t_warm) * 100 if t_warm > 0 else 0
        
        result = {
            'Model': f"{library.capitalize()}_{model_name}",
            'Dataset': dataset_path,
            'Length': scale,
            'Time_Cold_Sec': float(t_cold),
            'Time_Warm_Sec': float(t_warm),
            'Time_Interval_Sec': float(t_interval),
            'Interval_Overhead_Pct': float(overhead),
            'MAPE': float(mape),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MASE': float(mase)
        }
        
        return json.dumps(result, cls=SafeEncoder)
        
    except Exception as e:
        import traceback
        return json.dumps({"error": f"{str(e)}\n{traceback.format_exc()}"})


# ==========================================
# 8. CLI INTERFACE
# ==========================================

if __name__ == "__main__":
    import argparse
    import yaml
    import json
    
    parser = argparse.ArgumentParser(description="Benchmark Suite with Lazy Imports")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--library", type=str, required=True, choices=["chronax", "statsforecast"])
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--target-column", type=str, default=None, help="Target column for external CSV")
    parser.add_argument("--forecast", action="store_true", help="Run in forecast mode (return raw data)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run single model and output JSON result
    result_json = run_single_model(
        model_name=args.model,
        library=args.library,
        dataset_path=args.dataset,
        config=config,
        target_column=args.target_column,
        forecast_mode=args.forecast
    )
    
    # Output with delimiter for parsing
    print(f"RESULT_JSON:::{result_json}", flush=True)
