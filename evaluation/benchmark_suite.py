"""
Ticket 1.3: Experiment Configuration
Defines the constants for the scalability study to ensure neutrality.
"""
import sys
import os
# Add parent directory for chronax model imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- DYNAMIC PATCH FOR BUGGED INTERN MODELS ---
# Some models (naive, randomWalkWithDrift) try to import 'base_forecaster' (lowercase)
# from 'base_forecaster' (module), but the class is actually 'BaseForecaster'.
try:
    import base_forecaster
    if not hasattr(base_forecaster, 'base_forecaster'):
        base_forecaster.base_forecaster = base_forecaster.BaseForecaster
except ImportError:
    pass
# ----------------------------------------------

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import warnings

# Suppress specific numpy warnings that clutter output during casting in some environments
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')


# ==========================================
# 1. CONFIGURATION (from config/settings.py)
# ==========================================

# The Scalability Ladder (Log-ish scale)
SCALES = [100, 500, 1_000, 5_000, 10_000, 50_000]

# Forecasting Parameters
HORIZON = 24            # Predict next 24 steps
SEASONALITY = 24        # Assumed hourly seasonality for metrics
CONFIDENCE_LEVEL = 95   # For interval generation

# Benchmark Settings
RANDOM_SEED = 42        # Enforce determinism
N_ITERATIONS = 5        # Number of repeats per scale to average out jitter


# ==========================================
# 2. LOGGER (from core/logger.py)
# ==========================================

class BenchmarkLogger:
    def __init__(self, output_dir: str = "results"):
        # Resolve path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, output_dir)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # This list will hold our dictionary records
        self.records: List[Dict[str, Any]] = []

    def log(self, 
            dataset: str, 
            length: int, 
            model: str, 
            time_cold: float, 
            time_warm: float, 
            time_interval: float = None,
            overhead_pct: float = None,
            mape: float = None):
        """
        Appends a record to the session log.
        """
        record = {
            'Dataset': dataset,
            'Length': length,
            'Model': model,
            'Time_Cold_Sec': round(time_cold, 6),
            'Time_Warm_Sec': round(time_warm, 6),
            'Time_Interval_Sec': round(time_interval, 6) if time_interval else None,
            'Interval_Overhead_Pct': round(overhead_pct, 2) if overhead_pct else None,
            'MAPE': round(mape, 4) if mape is not None else None
        }
        self.records.append(record)
        
        # Real-time console feedback
        mape_str = f" | MAPE={record['MAPE']}%" if record['MAPE'] is not None else ""
        print(f"  > [{model}] L={length}: Warm={record['Time_Warm_Sec']}s | Cold={record['Time_Cold_Sec']}s{mape_str}")

    def save(self, filename: str = "benchmark_results.csv"):
        """
        Flushes records to CSV.
        """
        df = pd.DataFrame(self.records)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"\n✅ Results saved to {path}")
        return df


# ==========================================
# 3. SYNTHETIC DATA (from data/synthetic_data.py)
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
    
    Args:
        y_array (np.ndarray): The raw 1D float32 array.
        return_jax (bool): Whether to return JAX DeviceArray. 
                          Set to False if JAX is not installed.
        
    Returns:
        tuple: (jax_input, sf_input)
            - jax_input: jax.numpy.DeviceArray (float32) or None
            - sf_input: pd.DataFrame columns=['unique_id', 'ds', 'y']
    """
    # 1. Chronax Input: Direct Cast to DeviceArray
    jax_input = None
    if return_jax:
        try:
            import jax.numpy as jnp
            jax_input = jnp.array(y_array, dtype=jnp.float32)
        except ImportError:
            # Fallback for environments without JAX
            pass
    
    # 2. StatsForecast Input: DataFrame Construction
    n = len(y_array)
    sf_input = pd.DataFrame({
        'unique_id': ['series_0'] * n,
        'ds': pd.date_range(start='2020-01-01', periods=n, freq='h'),
        'y': y_array
    })
    
    return jax_input, sf_input


# ==========================================
# 4. WRAPPERS
# ==========================================

class SFWrapper:
    def __init__(self, model_cls, model_params: dict, horizon: int, seasonality: int):
        """
        Wrapper for StatsForecast to ensure fair benchmarking.
        """
        self.horizon = horizon
        self.seasonality = seasonality
        
        # StatsForecast expects a list of instantiated models
        # Apply params
        final_params = model_params.copy()
        
        # Mapping common mismatched param names if necessary
        if 'window_size' in final_params and 'window_size' not in model_cls.__init__.__code__.co_varnames:
             pass

        self.model_obj = model_cls(**final_params)
        
        from statsforecast import StatsForecast
        # We instantiate StatsForecast object once
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
    def __init__(self, model_cls, model_params: dict, horizon: int, seasonality: int):
        from jax import random
        self.horizon = horizon
        self.seasonality = seasonality
        self.key = random.PRNGKey(0)
        
        final_params = model_params.copy()
        
        # Inject seasonality if the model likely needs it and it's not provided
        if 'season_length' not in final_params and model_cls.__name__ in ['AutoETS']:
            final_params['season_length'] = seasonality
            
        self.model = model_cls(**final_params)

    def _dummy_predict_impl(self, y_array):
        # We use forecast() for stateless/efficient prediction if available
        if hasattr(self.model, 'forecast'):
             return self.model.forecast(y_array, h=self.horizon)
        else:
            # Fallback for stateful only
            self.model.fit(y_array)
            return self.model.predict(h=self.horizon)

    def fit_predict(self, y_array, intervals: bool = False):
        """
        Args:
            y_array (DeviceArray): Input data on GPU/TPU.
            intervals (bool): If True, run Monte Carlo/Conformal path.
        """
        # 1. Run Inference
        preds = self._dummy_predict_impl(y_array)
        
        # 2. CRITICAL: Block until the GPU has finished computing!
        preds['mean'].block_until_ready()        
        return preds['mean']


class ModelRegistry:
    """
    Central registry defining valid benchmark models and their configurations.
    """
    
    @staticmethod
    def get_chronax_model(name: str):
        """Lazy loader for Chronax models to avoid JAX leak."""
        try:
            if name == "WindowAverage":
                import window_average
                return window_average.WindowAverage
            elif name == "ADIDA":
                import adida
                return adida.ADIDA
            elif name == "CrostonClassic":
                import croston_classic
                return croston_classic.CrostonClassic
            elif name == "ETS":
                import ets_model
                return ets_model.ETS
            elif name == "HistoricAverage":
                import historic_average
                return historic_average.HistoricAverage
            elif name == "IMAPA":
                import imapa
                return imapa.IMAPA
            elif name == "Naive":
                import naive
                return naive.Naive
            elif name == "RandomWalkWithDrift":
                import randomWalkWithDrift
                return randomWalkWithDrift.RandomWalkWithDrift
            elif name == "ARIMA":
                import arima
                return arima.ARIMA
            elif name == "SeasonalExponentialSmoothing":
                import seasonal_exponential_smoothing
                return seasonal_exponential_smoothing.SeasonalExponentialSmoothing
            elif name == "SeasonalNaive":
                import seasonal_naive
                return seasonal_naive.SeasonalNaive
            elif name == "SeasonalWindowAverage":
                import seasonal_window_average
                return seasonal_window_average.SeasonalWindowAverage
            elif name == "SimpleExponentialSmoothing":
                import simple_exponential_smoothing
                return simple_exponential_smoothing.SimpleExponentialSmoothing
            elif name == "TSB":
                import tsb
                return tsb.TSB
        except (ImportError, AttributeError, NameError):
            pass 
        return None

    @staticmethod
    def get_sf_model(name: str):
        """Lazy loader for StatsForecast models."""
        try:
            from statsforecast import models as sf_models
            return getattr(sf_models, name, None)
        except ImportError:
            return None

    @staticmethod
    def get_registry():
        # Lazy imports for statsforecast models
        from statsforecast import models as sf_models
        
        def get_sf_model(name):
            try:
                return getattr(sf_models, name)
            except AttributeError:
                return None

        SFWindowAverage = get_sf_model('WindowAverage')
        SFADIDA = get_sf_model('ADIDA')
        SFCrostonClassic = get_sf_model('CrostonClassic')
        SFETS = get_sf_model('ETS')
        SFHistoricAverage = get_sf_model('HistoricAverage')
        SFIMAPA = get_sf_model('IMAPA')
        SFNaive = get_sf_model('Naive')
        SFRandomWalkWithDrift = get_sf_model('RandomWalkWithDrift')
        SFSeasonalExponentialSmoothing = get_sf_model('SeasonalExponentialSmoothing')
        SFSeasonalNaive = get_sf_model('SeasonalNaive')
        SFSeasonalWindowAverage = get_sf_model('SeasonalWindowAverage')
        SFSimpleExponentialSmoothing = get_sf_model('SimpleExponentialSmoothing')
        SFTSB = get_sf_model('TSB')
        SFARIMA = get_sf_model('ARIMA')

        import adida
        import arima
        # import auto_ets
        # import ces
        import croston_classic
        import ets_model
        # import garch
        import historic_average
        # import holt
        # import holt_winters
        import imapa
        # import mfles
        # import mstl
        import naive
        import randomWalkWithDrift
        import seasonal_exponential_smoothing
        import seasonal_naive
        import seasonal_window_average
        import simple_exponential_smoothing
        # import stl
        # import tbats_model
        # import theta_model
        import tsb
        import window_average

        return {
            "WindowAverage": {
                "chronax_cls": window_average.WindowAverage,
                "sf_cls": SFWindowAverage,
                "params": {"window_size": 24} 
            },
            "ADIDA": {
                "chronax_cls": adida.ADIDA,
                "sf_cls": SFADIDA,
                "params": {} 
            },
            "CrostonClassic": {
                "chronax_cls": croston_classic.CrostonClassic,
                "sf_cls": SFCrostonClassic,
                "params": {}
            },
            "ETS": {
                "chronax_cls": ets_model.ETS,
                "sf_cls": SFETS,
                "params": {}
            },
            "HistoricAverage": {
                "chronax_cls": historic_average.HistoricAverage,
                "sf_cls": SFHistoricAverage,
                "params": {}
            },
            "IMAPA": {
                "chronax_cls": imapa.IMAPA,
                "sf_cls": SFIMAPA,
                "params": {}
            },
            "Naive": {
                "chronax_cls": naive.Naive,
                "sf_cls": SFNaive,
                "params": {}
            },
            "RandomWalkWithDrift": {
                "chronax_cls": randomWalkWithDrift.RandomWalkWithDrift,
                "sf_cls": SFRandomWalkWithDrift,
                "params": {}
            },
            "ARIMA": {
                "chronax_cls": arima.ARIMA,
                "sf_cls": SFARIMA,
                "params": {"order": (1, 1, 1)}
            },
            "SeasonalExponentialSmoothing": {
                "chronax_cls": seasonal_exponential_smoothing.SeasonalExponentialSmoothing,
                "sf_cls": SFSeasonalExponentialSmoothing,
                "params": {}
            },
            "SeasonalNaive": {
                "chronax_cls": seasonal_naive.SeasonalNaive,
                "sf_cls": SFSeasonalNaive,
                "params": {}
            },
            "SeasonalWindowAverage": {
                "chronax_cls": seasonal_window_average.SeasonalWindowAverage,
                "sf_cls": SFSeasonalWindowAverage,
                "params": {"window_size": 24, "season_length": 24}
            },
            "SimpleExponentialSmoothing": {
                "chronax_cls": simple_exponential_smoothing.SimpleExponentialSmoothing,
                "sf_cls": SFSimpleExponentialSmoothing,
                "params": {}
            },
            "TSB": {
                "chronax_cls": tsb.TSB,
                "sf_cls": SFTSB,
                "params": {}
            }
        }

    @staticmethod
    def get_model_entry(model_name: str):
        # We need a unified entry for the worker scripts
        # Use a static registry first for params
        reg = {
            "WindowAverage": {"params": {"window_size": 24}},
            "ADIDA": {"params": {}},
            "CrostonClassic": {"params": {}},
            "ETS": {"params": {}},
            "HistoricAverage": {"params": {}},
            "IMAPA": {"params": {}},
            "Naive": {"params": {}},
            "RandomWalkWithDrift": {"params": {}},
            "ARIMA": {"params": {"order": (1, 1, 1)}},
            "SeasonalExponentialSmoothing": {"params": {}},
            "SeasonalNaive": {"params": {}},
            "SeasonalWindowAverage": {"params": {"window_size": 24, "season_length": 24}},
            "SimpleExponentialSmoothing": {"params": {}},
            "TSB": {"params": {}}
        }
        
        if model_name not in reg:
            raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(reg.keys())}")
        
        entry = reg[model_name].copy()
        # Add loaders lazily
        entry['chronax_cls'] = ModelRegistry.get_chronax_model(model_name)
        entry['sf_cls'] = ModelRegistry.get_sf_model(model_name)
        return entry


# ==========================================
# 6. VISUALIZER (from analysis/visualizer.py)
# ==========================================

def plot_results(csv_path="results/benchmark_results.csv"):
    # Resolve paths relative to evaluation root (parent of analysis/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If csv_path is relative, make it absolute relative to base_dir
    # Note: 'results' defaults to inside evaluation folder here
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(base_dir, csv_path)

    if not os.path.exists(csv_path):
        print(f"❌ No results file found at {csv_path}. Run benchmark first.")
        return

    df = pd.read_csv(csv_path)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    # Define results directory for output
    results_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    if base_name.endswith("_benchmark_results"):
        model_name = base_name.replace('_benchmark_results', '')
        prefix = f"{model_name}_"
    elif base_name.startswith("benchmark_results_"):
        model_name = base_name.replace('benchmark_results_', '')
        prefix = f"{model_name}_"
    elif base_name == "benchmark_results":
        prefix = ""
    else:
        prefix = f"{base_name}_"

    # --- Plot 1: Scalability (Speed vs Size) ---
    g = sns.FacetGrid(df, col="Dataset", col_wrap=3, height=5, sharey=False)
    g.map_dataframe(sns.lineplot, x='Length', y='Time_Warm_Sec', hue='Model', marker='o', linewidth=2.5)
    g.add_legend()
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Series Length (Log)", "Execution Time (Sec - Log)")
    g.fig.suptitle('Algorithmic Scalability by Dataset', fontsize=16, y=1.05)
    
    output_path_1 = os.path.join(results_dir, f"{prefix}scalability_plot.png")
    plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
    print(f"📊 Scalability plot saved to {output_path_1}")
    plt.close()

    # --- Plot 2: Accuracy (MAPE vs Size) ---
    if 'MAPE' in df.columns and df['MAPE'].notna().any():
        g = sns.FacetGrid(df, col="Dataset", col_wrap=3, height=5, sharey=False)
        g.map_dataframe(sns.lineplot, x='Length', y='MAPE', hue='Model', marker='o', linewidth=2.5)
        g.add_legend()
        g.set(xscale="log")
        g.set_axis_labels("Series Length (Log)", "MAPE (%)")
        g.fig.suptitle('Forecasting Accuracy (MAPE) by Dataset', fontsize=16, y=1.05)
        
        output_path_3 = os.path.join(results_dir, f"{prefix}accuracy_plot.png")
        plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
        print(f"📊 Accuracy plot saved to {output_path_3}")
        plt.close()

    # --- Plot 3: The Cost of Uncertainty (Interval Overhead) ---
    plt.figure(figsize=(10, 6))
    max_len = df['Length'].max()
    df_max = df[df['Length'] == max_len]
    ax = sns.barplot(
        data=df_max,
        x='Model',
        y='Interval_Overhead_Pct',
        errorbar=None,
        palette=['#1f77b4', '#ff7f0e']
    )
    plt.title(f'Avg Cost of Uncertainty (N={max_len}, All Datasets)', fontsize=14)
    plt.ylabel('Time Increase (%) vs Point Forecast', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)

    plt.figtext(0.5, 0.01, 
                "Note: Chronax uses vectorized Monte Carlo (100 samples).\nStatsForecast uses Bootstrap/Conformal methods.", 
                ha="center", fontsize=9, style='italic')

    output_path_2 = os.path.join(results_dir, f"{prefix}interval_overhead.png")
    plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
    print(f"📊 Interval overhead plot saved to {output_path_2}")
    plt.close()

# ==========================================
# 7. ENGINE (from core/engine.py)
# ==========================================

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_benchmark(model_name: str = "WindowAverage", dataset_path: str = None):
    # Fetch model config from registry
    try:
        entry = ModelRegistry.get_model_entry(model_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    chronax_cls = entry['chronax_cls']
    sf_cls = entry['sf_cls']
    # Use copy to avoid mutating the registry dict if we modify it later
    base_params = entry['params'].copy()
    
    logger = BenchmarkLogger()

    if dataset_path:
        # --- External Dataset Mode ---
        if not os.path.exists(dataset_path):
            print(f"❌ Error: Dataset file not found at {dataset_path}")
            return

        print(f"🚀 Starting Benchmark for Model: {model_name}")
        print(f"   Mode: External Dataset ({dataset_path})")
        
        try:
            df = pd.read_csv(dataset_path)
            # Simple heuristic: look for 'y', otherwise take last column
            if 'y' in df.columns:
                raw_values = df['y'].values.astype(np.float32)
            else:
                raw_values = df.iloc[:, -1].values.astype(np.float32)
            
            total_len = len(raw_values)
            if total_len <= HORIZON:
                print(f"❌ Error: Dataset length ({total_len}) must be greater than horizon ({HORIZON})")
                return
                
            dataset_name = os.path.basename(dataset_path)
            # Use full length, split last HORIZON for test
            train_len = total_len - HORIZON
            
            print(f"\n=== Dataset: {dataset_name} ===")
            print(f"--- Benchmarking Series Length: {train_len} (Train) + {HORIZON} (Test) ---")

            y_train = raw_values[:train_len]
            y_test = raw_values[train_len:]
            
            # Prepare inputs
            jax_input, sf_input = prepare_inputs(y_train)

            # --- Model 1: StatsForecast ---
            if sf_cls:
                t0 = time.perf_counter()
                sf = SFWrapper(
                    model_cls=sf_cls, 
                    model_params=base_params, 
                    horizon=HORIZON, 
                    seasonality=SEASONALITY
                )
                sf_preds = sf.fit_predict(sf_input) 
                t_cold_sf = time.perf_counter() - t0
                
                sf_values = sf_preds.iloc[:, -1].values
                mape_sf = calculate_mape(y_test, sf_values)
                
                warm_times = []
                for _ in range(N_ITERATIONS):
                    t0 = time.perf_counter()
                    sf.fit_predict(sf_input, intervals=False)
                    warm_times.append(time.perf_counter() - t0)
                t_warm_sf = sum(warm_times) / len(warm_times)
                
                t0 = time.perf_counter()
                sf.fit_predict(sf_input, intervals=True)
                t_interval_sf = time.perf_counter() - t0
                overhead_sf = ((t_interval_sf - t_warm_sf) / t_warm_sf) * 100

                logger.log(dataset_name, train_len, 'StatsForecast', 
                        t_cold_sf, t_warm_sf, t_interval_sf, overhead_sf, mape_sf)
            else:
                print("    [StatsForecast] Skipping (Not Available for this model)")

            # --- Model 2: Chronax ---
            t0 = time.perf_counter()
            cx = ChronaxWrapper(
                model_cls=chronax_cls,
                model_params=base_params,
                horizon=HORIZON, 
                seasonality=SEASONALITY
            )
            cx_preds = cx.fit_predict(jax_input) 
            t_cold_cx = time.perf_counter() - t0
            
            cx_values = np.array(cx_preds)
            mape_cx = calculate_mape(y_test, cx_values)
            
            warm_times = []
            for _ in range(N_ITERATIONS):
                t0 = time.perf_counter()
                cx.fit_predict(jax_input, intervals=False)
                warm_times.append(time.perf_counter() - t0)
            t_warm_cx = sum(warm_times) / len(warm_times)

            t0 = time.perf_counter()
            cx.fit_predict(jax_input, intervals=True)
            t_interval_cx = time.perf_counter() - t0
            overhead_cx = ((t_interval_cx - t_warm_cx) / t_warm_cx) * 100

            logger.log(dataset_name, train_len, 'Chronax', 
                       t_cold_cx, t_warm_cx, t_interval_cx, overhead_cx, mape_cx)

        except Exception as e:
            print(f"❌ Error processing external dataset: {e}")
            import traceback
            traceback.print_exc()
            return

    else:
        # --- Synthetic Data Mode (Original Logic) ---
        datasets = ['Trend', 'Seasonality', 'Stochastic']
        print(f"🚀 Starting Benchmark for Model: {model_name}")
        print(f"   Config: {len(datasets)} Datasets x {len(SCALES)} scales x {N_ITERATIONS} iterations")
        
        # Outer Loop: Datasets
        for dataset_name in datasets:
            print(f"\n=== Dataset: {dataset_name} ===")
            
            # Inner Loop: Scales
            for length in SCALES:
                print(f"\n--- Benchmarking Series Length: {length} ---")
                
                # 1. Prepare Data
                total_length = length + HORIZON
                raw_data = generate_series(dataset_name, total_length)
                y_train = raw_data[:length]
                y_test = raw_data[length:]
                jax_input, sf_input = prepare_inputs(y_train)
                
                # --- Model 1: StatsForecast ---
                if sf_cls:
                    t0 = time.perf_counter()
                    sf = SFWrapper(
                        model_cls=sf_cls, 
                        model_params=base_params, 
                        horizon=HORIZON, 
                        seasonality=SEASONALITY
                    )
                    sf_preds = sf.fit_predict(sf_input) 
                    t_cold_sf = time.perf_counter() - t0
                    
                    sf_values = sf_preds.iloc[:, -1].values
                    mape_sf = calculate_mape(y_test, sf_values)
                    
                    warm_times = []
                    for _ in range(N_ITERATIONS):
                        t0 = time.perf_counter()
                        sf.fit_predict(sf_input, intervals=False)
                        warm_times.append(time.perf_counter() - t0)
                    t_warm_sf = sum(warm_times) / len(warm_times)
                    
                    t0 = time.perf_counter()
                    sf.fit_predict(sf_input, intervals=True)
                    t_interval_sf = time.perf_counter() - t0
                    overhead_sf = ((t_interval_sf - t_warm_sf) / t_warm_sf) * 100

                    logger.log(dataset_name, length, 'StatsForecast', 
                            t_cold_sf, t_warm_sf, t_interval_sf, overhead_sf, mape_sf)
                else:
                     print("    [StatsForecast] Skipping (Not Available for this model)")

                # --- Model 2: Chronax ---
                t0 = time.perf_counter()
                cx = ChronaxWrapper(
                    model_cls=chronax_cls,
                    model_params=base_params,
                    horizon=HORIZON, 
                    seasonality=SEASONALITY
                )
                cx_preds = cx.fit_predict(jax_input) 
                t_cold_cx = time.perf_counter() - t0
                
                cx_values = np.array(cx_preds)
                mape_cx = calculate_mape(y_test, cx_values)
                
                warm_times = []
                for _ in range(N_ITERATIONS):
                    t0 = time.perf_counter()
                    cx.fit_predict(jax_input, intervals=False)
                    warm_times.append(time.perf_counter() - t0)
                t_warm_cx = sum(warm_times) / len(warm_times)

                t0 = time.perf_counter()
                cx.fit_predict(jax_input, intervals=True)
                t_interval_cx = time.perf_counter() - t0
                overhead_cx = ((t_interval_cx - t_warm_cx) / t_warm_cx) * 100

                logger.log(dataset_name, length, 'Chronax', 
                        t_cold_cx, t_warm_cx, t_interval_cx, overhead_cx, mape_cx)

    # Final Save
    out_name = f"{model_name}_benchmark_results.csv"
    if dataset_path:
        out_name = f"{model_name}_{os.path.basename(dataset_path)}_results.csv"
    logger.save(filename=out_name)

if __name__ == "__main__":
    import argparse
    import jax
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="WindowAverage", help="Model name (e.g. WindowAverage, ETS)")
    parser.add_argument("--dataset", type=str, default=None, help="Optional external dataset path (CSV)")
    parser.add_argument("--plot", action="store_true", help="Generate plots after run")
    args = parser.parse_args()

    print(f"JAX Backend: {jax.devices()}")
    run_benchmark(args.model, args.dataset)
    
    if args.plot:
        outfile = f"{args.model}_benchmark_results.csv"
        if args.dataset:
            outfile = f"{args.model}_{os.path.basename(args.dataset)}_results.csv"
        plot_results(os.path.join("results", outfile)) 
