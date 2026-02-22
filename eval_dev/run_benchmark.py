"""
File: run_benchmark.py

High-level Purpose:
    Coordinates end-to-end benchmark execution by iterating configured datasets
    and model/library pairs, spawning isolated worker subprocesses, and
    aggregating outputs into benchmark artifacts.

Problem Solved:
    Provides a single orchestration entry point for repeatable runtime and
    accuracy evaluations without requiring manual execution of each model.

Architectural Role:
    Acts as the benchmark controller layer above `benchmark_suite.py`, handling
    config parsing, filtering, subprocess lifecycle management, and final result
    reporting/visualization.

Major Classes/Functions:
    - `run_benchmark`: Main orchestration routine.
    - `process_forecast_result`: Persists forecast-mode data and plots.
    - `plot_results`: Generates benchmark aggregate plots.

External Dependencies:
    - `yaml`, `pandas`, `numpy`
    - Standard library: `argparse`, `subprocess`, `json`, `datetime`, `os`, `sys`
    - Optional plotting dependency: `matplotlib`

Expected Inputs and Outputs:
    - Input: path to benchmark config and optional model/dataset filters.
    - Output: CSV benchmark summaries, optional forecast plot/data artifacts, and
      console logs for execution progress.

Example:
    >>> # python eval_dev/run_benchmark.py --config eval_dev/config.yaml
    >>> # python eval_dev/run_benchmark.py --model ARIMA --dataset AirlinePassengers

Assumptions:
    - Config file schema matches expected keys (`experiment`, `datasets`,
      `models`).
    - Worker module `benchmark_suite.py` is available in the same directory.

Side Effects:
    - Creates result directories and files.
    - Executes subprocesses for each benchmark task.
    - Emits status and error messages to stdout.

Author:
    Auto-documented
Date:
    2026-02-21
"""

import sys
import os
import argparse
import yaml
import subprocess
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional


def run_benchmark(
    config_path: str,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    forecast_mode: bool = False,
) -> None:
    """
    Execute benchmark orchestration over configured datasets and models.

    Detailed Description:
        Loads benchmark configuration, applies optional dataset/model filters,
        dispatches per-model jobs to `benchmark_suite.py` subprocesses, and
        aggregates returned JSON metrics into benchmark result artifacts.

    Args:
        config_path (str): Absolute or relative path to configuration YAML.
        model_filter (str | None, optional): Restrict run to a single model.
        dataset_filter (str | None, optional): Restrict run to one dataset name
            or external dataset path.
        forecast_mode (bool, optional): Enable forecast artifact mode instead of
            benchmark aggregation mode.

    Returns:
        None: Writes artifacts and prints progress to stdout.

    Raises:
        subprocess.CalledProcessError: Captured and logged per worker run.

    Side Effects:
        Creates result directories/files and launches subprocesses.

    Example:
        >>> run_benchmark("config.yaml", model_filter="ARIMA")

    Notes:
        Subprocess isolation keeps model execution environments independent.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_cfg = config['experiment']
    models_cfg = config['models']
    datasets_cfg = config['datasets']
    
    scales = experiment_cfg['scales']
    results_records = []
    
    # Single Python interpreter for everything
    python_exe = sys.executable
    
    # Define output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if forecast_mode:
        results_dir = os.path.join(script_dir, "forecast_results")
        print(f"🚀 Starting Forecast Generation (Single Env Mode)")
    else:
        results_dir = os.path.join(script_dir, "benchmark_results")
        print(f"🚀 Starting Benchmark (Single Env Mode)")

    # Ensure output directories exist
    os.makedirs(results_dir, exist_ok=True)
    if forecast_mode:
         os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
         os.makedirs(os.path.join(results_dir, "data"), exist_ok=True)

    print(f"   Python: {python_exe}")
    
    if model_filter:
        print(f"   🎯 Model Filter: {model_filter}")
    if dataset_filter:
        print(f"   🎯 Dataset Filter: {dataset_filter}")
    
    # Handle dataset filter override
    if dataset_filter:
        if os.path.isfile(dataset_filter):
            datasets_cfg = [{'name': dataset_filter, 'type': 'external', 'path': dataset_filter}]
        else:
            datasets_cfg = [d for d in datasets_cfg if d['name'] == dataset_filter]
            if not datasets_cfg:
                print(f"⚠️ Warning: Dataset '{dataset_filter}' not found in config.")
    
    # Iterate through datasets
    for dataset in datasets_cfg:
        dataset_name = dataset['name']
        dataset_type = dataset.get('type', 'synthetic')
        dataset_path = dataset.get('path', dataset_name)
        target_column = dataset.get('target_column', None)
        
        print(f"\n=== Dataset: {dataset_name} ===")
        
        # For external datasets, run once (workers determine length)
        if dataset_type == 'external' or os.path.isfile(dataset_path):
            scales_to_run = [0]
            print(f"   (External dataset - using full file length)")
            if target_column:
                print(f"   Target column: {target_column}")
        else:
            scales_to_run = scales
        
        for scale in scales_to_run:
            if scale > 0:
                print(f"\n--- Scale: {scale} ---")
            
            for model in models_cfg:
                model_name = model['name']
                
                # Apply model filter
                if model_filter and model_name != model_filter:
                    continue
                
                library = model['library']
                
                print(f"   > Running {library.capitalize()} {model_name}...")
                
                # Build command - uses CURRENT Python for both libraries
                cmd = [
                    python_exe,
                    os.path.join(script_dir, "benchmark_suite.py"),
                    "--model", model_name,
                    "--library", library,
                    "--dataset", dataset_path,
                    "--config", config_path
                ]
                
                if forecast_mode:
                    cmd.append("--forecast")
                
                # Add target column if specified
                if target_column:
                    cmd.extend(["--target-column", target_column])
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    output = result.stdout
                    if "RESULT_JSON:::" in output:
                        json_str = output.split("RESULT_JSON:::")[-1].strip()
                        record = json.loads(json_str)
                        
                        # Update Dataset field to use friendly name
                        if dataset_type == 'external':
                            record['Dataset'] = dataset_name
                        
                        if 'error' in record:
                            print(f"     ⚠️ {library}: {record['error']}")
                        else:
                            # --- FORECAST MODE HANDLING ---
                            if forecast_mode and 'forecast_mode' in record:
                                process_forecast_result(record, results_dir, dataset_name, library, model_name)
                            else:
                                # Standard Benchmark Handling
                                results_records.append(record)
                                
                                # Build metrics string
                                metrics_str = ""
                                if record.get('MAPE') is not None:
                                    metrics_str += f" | MAPE={record['MAPE']:.2f}%"
                                if record.get('MAE') is not None:
                                    metrics_str += f" | MAE={record['MAE']:.4f}"
                                if record.get('RMSE') is not None:
                                    metrics_str += f" | RMSE={record['RMSE']:.4f}"
                                if record.get('MASE') is not None:
                                    metrics_str += f" | MASE={record['MASE']:.4f}"
                                
                                print(f"     ✅ Done: Warm={round(record.get('Time_Warm_Sec', 0), 4)}s{metrics_str}")
                    else:
                        print(f"     ❌ Error: Output format mismatch.")
                        print(f"        STDOUT: {output[:300]}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"     ❌ Error running {model_name} in {library}.")
                    print(f"        EXIT CODE: {e.returncode}")
                    if e.stdout:
                        print(f"        STDOUT: {e.stdout[:300]}")
                    if e.stderr:
                        print(f"        STDERR: {e.stderr[:300]}")
                except Exception as e:
                    print(f"     ❌ Unexpected error: {e}")
    
    # Aggregate and save results (ONLY FOR BENCHMARK)
    if not forecast_mode and results_records:
        df = pd.DataFrame(results_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_csv = os.path.join(results_dir, f"benchmark_{timestamp}.csv")
        df.to_csv(final_csv, index=False)
        print(f"\n✅ Results saved to {final_csv}")
        
        # Also save as latest
        latest_csv = os.path.join(results_dir, "benchmark_results.csv")
        df.to_csv(latest_csv, index=False)
    elif not forecast_mode:
        print("\n⚠️ No results collected.")


def process_forecast_result(
    record: Dict[str, Any],
    output_dir: str,
    dataset_name: str,
    library: str,
    model_name: str,
) -> None:
    """
    Persist a single forecast result to CSV and PNG for inspection and reporting.

    Detailed Description:
        Consumes a forecast-mode result record from the benchmark worker
        (containing y_train, y_test, predictions, and optionally MAPE). Writes
        a CSV file under output_dir/data with columns y_true, split (train/test),
        y_pred (forecast aligned to test period), library, and model. Then
        generates a matplotlib figure plotting training series, test actuals,
        and forecast series with a title including dataset, library, model, and
        MAPE; saves the figure as a PNG under output_dir/plots. Used by
        run_benchmark when forecast_mode is True to produce per-model artifacts
        without changing the worker's JSON contract.

    Args:
        record (Dict[str, Any]): Must contain "y_train", "y_test", "predictions"
            (lists or array-like), and optionally "MAPE". Keys are as returned
            by run_single_model in forecast mode.
        output_dir (str): Base directory for results; data and plots are written
            to output_dir/data and output_dir/plots.
        dataset_name (str): Human-readable dataset name for filenames and plot title.
        library (str): "chronax" or "statsforecast" for filenames and title.
        model_name (str): Model name for filenames and title.

    Returns:
        None. Writes one CSV and one PNG; prints the saved plot basename to stdout.

    Raises:
        KeyError: If record is missing "y_train", "y_test", or "predictions".
        IOError: If writing the CSV or PNG fails.

    Side Effects:
        Creates output_dir/data and output_dir/plots if needed; writes
        {dataset_name}_{library}_{model_name}.csv and .png; imports
        matplotlib; prints one line to stdout.

    Example:
        >>> process_forecast_result(record, "forecast_results", "Airline", "chronax", "ARIMA")

    Notes:
        Role: Bridges forecast-mode JSON output from the worker to
        human-inspectable data and plots for analysis and reporting.
    """
    import matplotlib.pyplot as plt
    
    y_train = np.array(record['y_train'])
    y_test = np.array(record['y_test'])
    preds = np.array(record['predictions'])
    
    # 1. Save Data CSV
    data_csv_path = os.path.join(output_dir, "data", f"{dataset_name}_{library}_{model_name}.csv")
    
    max_len = len(y_train) + len(y_test)
    df_data = pd.DataFrame(index=range(max_len))
    df_data['y_true'] = np.concatenate([y_train, y_test])
    df_data['split'] = ['train'] * len(y_train) + ['test'] * len(y_test)
    
    pred_col = np.full(max_len, np.nan)
    pred_col[len(y_train):] = preds
    df_data['y_pred'] = pred_col
    
    df_data['library'] = library
    df_data['model'] = model_name
    
    df_data.to_csv(data_csv_path, index=False)
    
    # 2. Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_train)), y_train, label='Train', color='gray', alpha=0.6)
    plt.plot(range(len(y_train), max_len), y_test, label='Test (True)', color='green')
    plt.plot(range(len(y_train), max_len), preds, label='Forecast', color='red', linestyle='--')
    
    mape = record.get('MAPE', 0)
    plt.title(f"Forecast: {library} {model_name} on {dataset_name}\nMAPE: {mape:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, "plots", f"{dataset_name}_{library}_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"     ✅ Forecast saved: {os.path.basename(plot_path)}")


def plot_results(csv_path: str) -> None:
    """
    Generate benchmark summary visualizations from an aggregated results CSV.

    Detailed Description:
        Loads the benchmark CSV (with columns Dataset, Model, Length,
        Time_Warm_Sec, MAPE, etc.) and produces two types of plots in the
        same directory as the CSV. (1) Scalability plot: for each dataset, a
        subplot of Time_Warm_Sec vs Length (log-log) per model, to compare
        runtime scaling. (2) Accuracy plot: for each dataset, a subplot of
        MAPE vs Length (log x) per model, to compare forecast accuracy. Uses
        a fixed color map across models for consistency. Saves scalability_plot.png
        and accuracy_plot.png and prints their paths. If the CSV is missing,
        prints an error and returns without writing. Designed for post-run
        analysis and reporting; uses only matplotlib for portability.

    Args:
        csv_path (str): Absolute or relative path to the benchmark results
            CSV (e.g. benchmark_results.csv or benchmark_YYYYMMDD_HHMMSS.csv).

    Returns:
        None. Writes up to two PNG files and prints their paths to stdout.

    Raises:
        None. Missing file is handled with a message and early return.

    Side Effects:
        Reads csv_path; creates scalability_plot.png and accuracy_plot.png
        in the same directory; imports matplotlib; prints save locations.

    Example:
        >>> plot_results(os.path.join("eval_dev", "benchmark_results", "benchmark_results.csv"))

    Notes:
        Role: Post-processing visualization for benchmark runs; enables
        quick comparison of models across datasets and scales without
        external tooling.
    """
    import matplotlib.pyplot as plt
    
    if not os.path.exists(csv_path):
        print(f"❌ No results file found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    results_dir = os.path.dirname(csv_path)
    datasets = df['Dataset'].unique()
    models = df['Model'].unique()
    
    # Color cycle for models
    cmap = plt.cm.get_cmap('tab20', len(models))
    color_map = {m: cmap(i) for i, m in enumerate(models)}
    
    # Scalability Plot
    if 'Length' in df.columns and 'Time_Warm_Sec' in df.columns:
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
        
        for idx, ds in enumerate(datasets):
            ax = axes[0, idx]
            ds_df = df[df['Dataset'] == ds]
            for model in models:
                m_df = ds_df[ds_df['Model'] == model].sort_values('Length')
                if not m_df.empty:
                    ax.plot(m_df['Length'], m_df['Time_Warm_Sec'], marker='o',
                            linewidth=2.5, label=model, color=color_map[model])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Series Length (Log)')
            ax.set_ylabel('Execution Time (Sec - Log)')
            ax.set_title(ds)
            ax.grid(True, alpha=0.3)
        
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(models), 5), fontsize=8)
        fig.suptitle('Algorithmic Scalability by Dataset', fontsize=16)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = os.path.join(results_dir, "scalability_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Scalability plot saved to {output_path}")
        plt.close()
    
    # Accuracy Plot
    if 'MAPE' in df.columns and df['MAPE'].notna().any():
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
        
        for idx, ds in enumerate(datasets):
            ax = axes[0, idx]
            ds_df = df[df['Dataset'] == ds]
            for model in models:
                m_df = ds_df[ds_df['Model'] == model].sort_values('Length')
                if not m_df.empty and m_df['MAPE'].notna().any():
                    ax.plot(m_df['Length'], m_df['MAPE'], marker='o',
                            linewidth=2.5, label=model, color=color_map[model])
            ax.set_xscale('log')
            ax.set_xlabel('Series Length (Log)')
            ax.set_ylabel('MAPE (%)')
            ax.set_title(ds)
            ax.grid(True, alpha=0.3)
        
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                   ncol=min(len(models), 5), fontsize=8)
        fig.suptitle('Forecasting Accuracy (MAPE) by Dataset', fontsize=16)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = os.path.join(results_dir, "accuracy_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Accuracy plot saved to {output_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Orchestrator (Single Environment)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name or path")
    parser.add_argument("--forecast", action="store_true", help="Run forecast mode (plots + data CSVs)")
    args = parser.parse_args()
    
    # Resolve config path
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    config_abs_path = os.path.abspath(config_path)
    if not os.path.exists(config_abs_path):
        print(f"❌ Error: Config file not found at {config_abs_path}")
        sys.exit(1)
    
    run_benchmark(config_abs_path, model_filter=args.model, dataset_filter=args.dataset, forecast_mode=args.forecast)
