import sys
import os
import argparse
import json
import time
import pandas as pd
import numpy as np
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_suite import SFWrapper, ModelRegistry, generate_series, prepare_inputs, calculate_mape

def run_worker(model_name, dataset_name, scale, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_cfg = config['experiment']
    horizon = experiment_cfg['horizon']
    seasonality = experiment_cfg['seasonality']
    n_iterations = experiment_cfg.get('n_iterations', 5)

    # Find model params
    entry = ModelRegistry.get_model_entry(model_name)
    model_params = entry.get('params', {}).copy()
    
    config_params = {}
    for mod in config['models']:
        if mod['name'] == model_name and mod['library'] == 'statsforecast':
            config_params = mod.get('params', {})
            break
            
    model_params.update(config_params)
    
    sf_cls = entry['sf_cls']

    if sf_cls is None:
        print(json.dumps({"error": "Model not available in StatsForecast"}))
        return

    # Generate/Load Data
    total_length = scale + horizon
    raw_data = generate_series(dataset_name, total_length)
    y_train = raw_data[:scale]
    y_test = raw_data[scale:]
    
    _, sf_input = prepare_inputs(y_train, return_jax=False)

    # Setup Wrapper
    sfw = SFWrapper(
        model_cls=sf_cls,
        model_params=model_params,
        horizon=horizon,
        seasonality=seasonality
    )

    # Cold Start
    t0 = time.perf_counter()
    sf_preds = sfw.fit_predict(sf_input)
    t_cold = time.perf_counter() - t0
    
    sf_values = sf_preds.iloc[:, -1].values
    mape = calculate_mape(y_test, sf_values)

    # Warm Start
    warm_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        sfw.fit_predict(sf_input, intervals=False)
        warm_times.append(time.perf_counter() - t0)
    t_warm = sum(warm_times) / len(warm_times)

    # Interval Overhead
    t0 = time.perf_counter()
    sfw.fit_predict(sf_input, intervals=True)
    t_interval = time.perf_counter() - t0
    overhead = ((t_interval - t_warm) / t_warm) * 100

    results = {
        'Model': f"StatsForecast_{model_name}",
        'Dataset': dataset_name,
        'Length': scale,
        'Time_Cold_Sec': float(t_cold),
        'Time_Warm_Sec': float(t_warm),
        'Time_Interval_Sec': float(t_interval),
        'Interval_Overhead_Pct': float(overhead),
        'MAPE': float(mape)
    }
    
    print(f"RESULT_JSON:::{json.dumps(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    run_worker(args.model, args.dataset, args.scale, args.config)
