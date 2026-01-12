import sys
import os
import traceback

# 1. PRINT IMMEDIATELY (Before loading heavy libraries)
print("DEBUG: Worker process started.", flush=True)

# 2. Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(current_dir)
sys.path.append(parent_dir)
print(f"DEBUG: Paths added. Parent: {parent_dir}", flush=True)

try:
    # 3. Import Standard Libs
    import argparse
    import json
    import time
    import yaml
    
    # 4. Import Heavy Libs (inside try/except to catch DLL errors)
    print("DEBUG: Importing Pandas/Numpy...", flush=True)
    import pandas as pd
    import numpy as np

    print("DEBUG: Importing JAX...", flush=True)
    import jax
    print(f"DEBUG: JAX Imported. Device: {jax.devices()[0]}", flush=True)
    
    print("DEBUG: Importing Benchmark Suite...", flush=True)
    from benchmark_suite import ChronaxWrapper, ModelRegistry, generate_series, prepare_inputs, calculate_mape
    print("DEBUG: Benchmark Suite Imported.", flush=True)

except Exception as e:
    print(f"DEBUG: CRITICAL IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

def run_worker(model_name, dataset_name, scale, config_path):
    print(f"DEBUG: Running worker logic for {model_name}...", flush=True)
    
    # Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_cfg = config['experiment']
    horizon = experiment_cfg['horizon']
    seasonality = experiment_cfg['seasonality']
    n_iterations = experiment_cfg.get('n_iterations', 5)

    # Find model params
    # entry = ModelRegistry.get_model_entry(model_name)
    # model_params = entry.get('params', {}).copy()

    # To:
    try:
        entry = ModelRegistry.get_model_entry(model_name)
    except Exception as e:
        msg = json.dumps({"error": f"Registry error for {model_name}: {str(e)}"})
        print(f"RESULT_JSON:::{msg}", flush=True)
        return

    model_params = entry.get('params', {}).copy()
        
    config_params = {}
    for mod in config['models']:
        if mod['name'] == model_name and mod['library'] == 'chronax':
            config_params = mod.get('params', {})
            break
    model_params.update(config_params)
    
    chronax_cls = entry['chronax_cls']
    if chronax_cls is None:
        # Show more detail about why it's None
        msg = json.dumps({
            "error": f"Model {model_name} not available in Chronax. Import may have failed. Check that imapa.py exists and has no syntax errors."
        })
        print(f"RESULT_JSON:::{msg}", flush=True)
        return

    # Generate Data
    total_length = scale + horizon
    raw_data = generate_series(dataset_name, total_length)
    y_train = raw_data[:scale]
    y_test = raw_data[scale:]
    
    jax_input, _ = prepare_inputs(y_train)

    # Instantiate Wrapper
    print("DEBUG: Instantiating model...", flush=True)
    cw = ChronaxWrapper(
        model_cls=chronax_cls,
        model_params=model_params,
        horizon=horizon,
        seasonality=seasonality
    )

    # Cold Start
    print("DEBUG: Running Cold Start...", flush=True)
    t0 = time.perf_counter()
    cw_preds = cw.fit_predict(jax_input)
    t_cold = time.perf_counter() - t0
    
    cx_values = np.array(cw_preds)
    mape = calculate_mape(y_test, cx_values)

    # Warm Start
    print("DEBUG: Running Warm Start...", flush=True)
    warm_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        cw.fit_predict(jax_input, intervals=False)
        warm_times.append(time.perf_counter() - t0)
    t_warm = sum(warm_times) / len(warm_times)

    # Interval Overhead
    t0 = time.perf_counter()
    cw.fit_predict(jax_input, intervals=True)
    t_interval = time.perf_counter() - t0
    overhead = ((t_interval - t_warm) / t_warm) * 100

    results = {
        'Model': f"Chronax_{model_name}",
        'Dataset': dataset_name,
        'Length': scale,
        'Time_Cold_Sec': float(t_cold),
        'Time_Warm_Sec': float(t_warm),
        'Time_Interval_Sec': float(t_interval),
        'Interval_Overhead_Pct': float(overhead),
        'MAPE': float(mape)
    }
    
    print(f"RESULT_JSON:::{json.dumps(results)}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_worker(args.model, args.dataset, args.scale, args.config)