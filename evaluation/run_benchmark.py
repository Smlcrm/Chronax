import sys
import os
import argparse
import yaml
import subprocess
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark_suite import BenchmarkLogger

def run_benchmark(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_cfg = config['experiment']
    envs_cfg = config['environments']
    models_cfg = config['models']
    datasets_cfg = config['datasets']
    
    scales = experiment_cfg['scales']
    results_records = []
    
    # DEFINE BASE DIRS (Add this at the start of the function)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"🚀 Starting Multi-Environment Benchmark")
    print(f"   Chronax Env: {envs_cfg['chronax_python']}")
    print(f"   StatsForecast Env: {envs_cfg['sf_python']}")
    
    for dataset in datasets_cfg:
        dataset_name = dataset['name']
        print(f"\n=== Dataset: {dataset_name} ===")
        
        for scale in scales:
            print(f"\n--- Scale: {scale} ---")
            
            for model in models_cfg:
                model_name = model['name']
                library = model['library']
                
                python_exe = envs_cfg['chronax_python'] if library == 'chronax' else envs_cfg['sf_python']
                worker_script = "chronax_worker.py" if library == 'chronax' else "sf_worker.py"
                
                print(f"   > Running {library.capitalize()} {model_name}...")
                
                cmd = [
                    python_exe,
                    os.path.join(os.path.dirname(__file__), worker_script),
                    "--model", model_name,
                    "--dataset", dataset_name,
                    "--scale", str(scale),
                    "--config", config_path
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print("result", result)
                    # Robust Parsing: Search for the special delimiter
                    output = result.stdout
                    if "RESULT_JSON:::" in output:
                        json_str = output.split("RESULT_JSON:::")[-1].strip()
                        record = json.loads(json_str)
                        results_records.append(record)
                        
                        mape_str = f" | MAPE={record['MAPE']}%" if record['MAPE'] is not None else ""
                        print(f"     ✅ Done: Warm={round(record['Time_Warm_Sec'], 4)}s{mape_str}")
                    else:
                        print(f"     ❌ Error: Worker output format mismatch.")
                        print(f"        STDOUT: {output[:200]}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"     ❌ Error running {model_name} in {library}.")
                    print(f"        EXIT CODE: {e.returncode}")
                    # --- ADD THESE LINES ---
                    print(f"        STDOUT: {e.stdout}") 
                    print(f"        STDERR: {e.stderr}")
                    # -----------------------
                except Exception as e:
                    print(f"     ❌ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()  # Add this line to see full error

    # Aggregation
    if results_records:
        df = pd.DataFrame(results_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# UPDATE PATHS HERE:
        final_csv = os.path.join(results_dir, f"multi_env_benchmark_{timestamp}.csv")
        df.to_csv(final_csv, index=False)
        print(f"\n✅ All results aggregated to {final_csv}")
        
        # Symbolic link or copy to latest
        latest_csv = os.path.join(results_dir, "benchmark_results.csv")        
        df.to_csv(latest_csv, index=False)
        
        # Plotting
        print("\n📊 Generating Plots...")
        try:
            from benchmark_suite import plot_results
            plot_results(latest_csv)
        except ImportError:
            print("⚠️ matplotlib or seaborn not found in current environment. Skipping plots.")
        except Exception as e:
            print(f"⚠️ Error generating plots: {e}")
    else:
        print("\n⚠️ No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    # Resolve config path: try relative to CWD first, then relative to script dir
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), args.config)
    
    config_abs_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_abs_path):
        print(f"❌ Error: Config file not found at {config_abs_path}")
        sys.exit(1)
        
    run_benchmark(config_abs_path)
