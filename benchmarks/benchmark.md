# Benchmarking Suite Documentation

This document outlines how to run and configure the benchmarking suite for Chronax and StatsForecast models.

## Overview

The benchmark suite is designed to compare the performance and accuracy of time series forecasting models from **Chronax** (JAX-based) and **StatsForecast**. It supports:

*   **Lazy Loading**: Libraries are imported only when needed.
*   **Performance Metrics**: Cold start time, warm execution time, interval overhead.
*   **Accuracy Metrics**: MAPE, MAE, RMSE, MASE.
*   **Datasets**: Synthetic (Trend, Seasonality, Stochastic) and External (CSV).

## Running the Benchmark

The main entry point is `benchmarks/run_benchmark.py`. It orchestrates the execution of individual model runs.
It now runs both libraries using a **single Python environment** (the interpreter used to launch the script).

### Basic Usage

Run the full benchmark defined in your configuration file:

```bash
python benchmarks/run_benchmark.py --config benchmarks/config.yaml
```

### Command Line Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--config` | Path to the YAML configuration file. | `--config benchmarks/config.yaml` |
| `--model` | specific model to run (filters the config). | `--model AutoETS` |
| `--dataset` | Run a specific dataset (name or path). | `--dataset "airline-passengers"` |
| `--forecast` | Run in forecast mode (save data/plots, no aggregate stats). | `--forecast` |

### Examples

**Run only the AutoETS model:**
```bash
python benchmarks/run_benchmark.py --config benchmarks/config.yaml --model AutoETS
```

**Run on a specific external dataset:**
```bash
python benchmarks/run_benchmark.py --config benchmarks/config.yaml --dataset "C:/path/to/data.csv"
```

**Generate Forecast Data (CSV):**
*Note: Plotting is currently disabled in the code.*
```bash
python benchmarks/run_benchmark.py --config benchmarks/config.yaml --forecast
```

## Configuration (`config.yaml`)

The `config.yaml` file controls the experiment setup.

```yaml
experiment:
  scales: [100, 1000]       # Lengths for synthetic data
  horizon: 24               # Forecast horizon
  seasonality: 24           # Seasonality period
  n_iterations: 5           # Number of warm runs for averaging

datasets:
  - name: "Trend"           # Synthetic dataset type
  - name: "my-data"
    type: "external"
    path: "data.csv"
    target_column: "value"  # Optional target column

models:
  - name: "AutoETS"
    library: "chronax"      # or "statsforecast"
    params: { season_length: 24 }
```

> Note: `run_benchmark.py` ignores the `environments` block if present in `config.yaml`.

## Benchmarking Methodology

For each model and dataset, the suite performs the following steps:

1.  **Cold Start Run**: The model is instantiated and compiled (if JAX) and run once. The time is recorded as `Time_Cold_Sec`. This includes JIT compilation overhead for Chronax.
2.  **Warm Runs**: The model is run `n_iterations` times (default: 5) on the same data. The average execution time is recorded as `Time_Warm_Sec`. This represents the pure inference speed after initialization.
3.  **Interval Overhead**: The model is run once with prediction intervals enabled (e.g., 95% confidence). The time difference compared to a warm run is calculated as `Interval_Overhead_Pct`.
4.  **Accuracy Evaluation**: The last `horizon` points of the dataset are held out as a test set. The model forecasts this horizon, and metrics are calculated against the actual values.

## Metrics

| Metric | Full Name | Description |
| :--- | :--- | :--- |
| **MAPE** | Mean Absolute Percentage Error | Average percentage difference between forecast and actual. |
| **MAE** | Mean Absolute Error | Average absolute difference. |
| **RMSE** | Root Mean Squared Error | Square root of the average squared errors. penalizes large errors. |
| **MASE** | Mean Absolute Scaled Error | Error scaled by the in-sample mean absolute error of a naive benchmark. < 1 means better than naive. |
| **Time_Cold_Sec** | Cold Start Time | Time for first run (setup + compile + fit/predict). |
| **Time_Warm_Sec** | Warm Execution Time | Average time for subsequent runs (pure execution). |
