"""
Run Benchmark Orchestrator (Single Environment)
=================================================
Same as run_benchmark.py but runs ALL models (chronax + statsforecast)
using the current Python environment instead of separate interpreters.

Fairness is preserved via subprocess isolation: each model still runs
in its own fresh process so there's no shared state, warm caches, or
memory pressure from previously loaded libraries.
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


def run_benchmark(config_path, model_filter=None, dataset_filter=None, forecast_mode=False):
    """
    Main benchmark orchestrator.
    Spawns benchmark_suite.py for each model/library combination
    using the CURRENT Python interpreter for both libraries.
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


def process_forecast_result(record, output_dir, dataset_name, library, model_name):
    """
    Handles a single forecast result:
    1. Saves data to CSV
    2. Generates plot
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


def plot_results(csv_path):
    """Generate benchmark visualization plots using matplotlib only."""
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
