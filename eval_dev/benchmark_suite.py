"""
Benchmark Suite with Lazy Imports
==================================
Refactored to avoid importing JAX or StatsForecast at module level.
Each wrapper class imports its dependencies only when instantiated.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings

# Suppress specific numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

# Add parent directory for chronax model imports
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
    """Handles JAX/Numpy types for JSON serialization."""
    def default(self, obj):
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
    def __init__(self, output_dir: str = "results"):
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
            time_interval: float = None,
            overhead_pct: float = None,
            mape: float = None,
            mae: float = None,
            rmse: float = None,
            mase: float = None):
        """Appends a record to the session log."""
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

    def save(self, filename: str = "benchmark_results.csv"):
        """Flushes records to CSV."""
        df = pd.DataFrame(self.records)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"\n✅ Results saved to {path}")
        return df


# ==========================================
# 3. SYNTHETIC DATA
# ==========================================

def generate_series(series_type: str, length: int, seed: int = 42) -> np.ndarray:
    """Generates synthetic time series data for benchmarking."""
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


def prepare_inputs(y_array: np.ndarray, return_jax: bool = True):
    """
    Converts raw numpy array into library-specific inputs.
    Uses lazy import for JAX.
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
    
    def __init__(self, model_cls, model_params: dict, horizon: int, seasonality: int):
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
        
    def fit_predict(self, df: pd.DataFrame, intervals: bool = False):
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
    
    def __init__(self, model_cls, model_params: dict, horizon: int, seasonality: int):
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

    def _dummy_predict_impl(self, y_array):
        if hasattr(self.model, 'forecast') and not self.extra_fit_params:
            return self.model.forecast(h=self.horizon, y=y_array)
        else:
            self.model.fit(y_array, **self.extra_fit_params)
            return self.model.predict(h=self.horizon)

    def fit_predict(self, y_array, intervals: bool = False):
        preds = self._dummy_predict_impl(y_array)
        preds['mean'].block_until_ready()        
        return preds['mean']


# ==========================================
# 5. MODEL REGISTRY (LAZY IMPORTS)
# ==========================================

class ModelRegistry:
    """Central registry with lazy model loading."""
    
    _MODELS = {
        "WindowAverage": {"params": {"window_size": 24}},
        "ADIDA": {"params": {}},
        "CrostonClassic": {"params": {}},
        "AutoETS": {"params": {}},
        "GARCH": {"params": {}},
        "AutoCES": {"params": {}},
        "Theta": {"params": {}},
        "AutoTheta": {"params": {}},
        "Holt": {"params": {}},
        "HoltWinters": {"params": {}},
        "MSTL": {"params": {}},
        "MFLES": {"params": {}},
        "TBATS": {"params": {}},
        "AutoTBATS": {"params": {}},
        "HistoricAverage": {"params": {}},
        "IMAPA": {"params": {}},
        "Naive": {"params": {}},
        "RandomWalkWithDrift": {"params": {}},
        "ARIMA": {"params": {"order": (1, 1, 1)}},
        "SeasonalExponentialSmoothing": {"params": {}},
        "SeasonalNaive": {"params": {}},
        "SeasonalWindowAverage": {"params": {"window_size": 24, "season_length": 24}},
        "SimpleExponentialSmoothing": {"params": {}},
        "TSB": {"params": {}},
    }

    @staticmethod
    def get_chronax_model(name: str):
        """Lazy loader for Chronax models."""
        try:
            if name == "WindowAverage":
                import window_average; return window_average.WindowAverage
            elif name == "ADIDA":
                import adida; return adida.ADIDA
            elif name == "CrostonClassic":
                import croston_classic; return croston_classic.CrostonClassic
            elif name == "AutoETS":
                import auto_ets; return auto_ets.AutoETS
            elif name == "GARCH":
                try: import garch; return garch.GARCH
                except ImportError: return None
            elif name == "AutoCES":
                try: import ces; return ces.AutoCES
                except ImportError: return None
            elif name == "Theta":
                try: import theta_model; return theta_model.Theta
                except ImportError: return None
            elif name == "AutoTheta":
                try: import theta_model; return theta_model.AutoTheta
                except ImportError: return None
            elif name == "Holt":
                try: import holt; return holt.Holt
                except ImportError: return None
            elif name == "HoltWinters":
                try: import holt_winters; return holt_winters.HoltWinters
                except ImportError: return None
            elif name == "MSTL":
                try: import mstl; return mstl.MSTL
                except ImportError: return None
            elif name == "MFLES":
                try: import mfles; return mfles.MFLES
                except ImportError: return None
            elif name == "TBATS":
                try: import tbats_model; return tbats_model.TBATS
                except ImportError: return None
            elif name == "AutoTBATS":
                try: import tbats_model; return tbats_model.AutoTBATS
                except ImportError: return None
            elif name == "HistoricAverage":
                import historic_average; return historic_average.HistoricAverage
            elif name == "IMAPA":
                import imapa; return imapa.IMAPA
            elif name == "Naive":
                import naive; return naive.Naive
            elif name == "RandomWalkWithDrift":
                import randomWalkWithDrift; return randomWalkWithDrift.RandomWalkWithDrift
            elif name == "ARIMA":
                import arima; return arima.ARIMA
            elif name == "SeasonalExponentialSmoothing":
                import seasonal_exponential_smoothing; return seasonal_exponential_smoothing.SeasonalExponentialSmoothing
            elif name == "SeasonalNaive":
                import seasonal_naive; return seasonal_naive.SeasonalNaive
            elif name == "SeasonalWindowAverage":
                import seasonal_window_average; return seasonal_window_average.SeasonalWindowAverage
            elif name == "SimpleExponentialSmoothing":
                import simple_exponential_smoothing; return simple_exponential_smoothing.SimpleExponentialSmoothing
            elif name == "TSB":
                import tsb; return tsb.TSB
        except (ImportError, AttributeError, NameError):
            pass 
        return None

    @staticmethod
    def get_sf_model(name: str):
        """Lazy loader for StatsForecast models."""
        config = ModelRegistry._MODELS.get(name, {})
        sf_name = config.get("sf_name", name)
        
        try:
            # LAZY IMPORT
            from statsforecast import models as sf_models
            return getattr(sf_models, sf_name, None)
        except ImportError:
            return None

    @staticmethod
    def get_model_entry(model_name: str):
        """Unified entry point."""
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

def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error (from TempusBench)."""
    return np.mean(np.abs(y_true - y_pred)).item()

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error (from TempusBench)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2)).item()

def calculate_mase(y_true, y_pred):
    """Mean Absolute Scaled Error (from TempusBench)."""
    denom = np.maximum(1e-10, np.mean(np.abs(y_true[1:] - y_true[:-1])))
    mase = np.mean(np.abs(y_true - y_pred)) / denom
    return mase.item()


# ==========================================
# 7. SINGLE MODEL RUNNER (for subprocess calls)
# ==========================================

def run_single_model(model_name: str, library: str, dataset_path: str, 
                     config: dict, target_column: str = None, forecast_mode: bool = False):
    """
    Run a single model benchmark. Called by run_benchmark.py via subprocess.
    Returns results as JSON string.
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
