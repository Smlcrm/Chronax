# Chronax Benchmark Suite Documentation

This guide provides a comprehensive explanation of the `benchmark_suite.py` evaluation framework and instructions for running the `Benchmark.ipynb` notebook.

## Overview

The `benchmark_suite.py` is a consolidated evaluation framework designed to benchmark **Chronax** (JAX-based forecasting) against **StatsForecast** (Numba-compiled forecasting) in terms of **scalable performance** and **accuracy**. 

It ensures a "neutral ground" comparison by:
1. Using identical synthetic datasets.
2. Enforcing consistent hardware state (warm-up runs).
3. Measuring both "Cold Start" (compilation/overhead) and "Warm Start" (throughput) speeds.
4. Comparing standard Point Forecasts vs. Prediction Intervals (Probabilistic Forecasting).

---

## Key Functionalities

*   **Algorithmic Scalability Testing**: Measures execution time across a logarithmic scale of series lengths (100 to 50,000 time steps).
*   **Dual-Model Comparison**: Automatically pairs a requested Chronax model with its StatsForecast equivalent (e.g., `WindowAverage` vs `StatsForecast.WindowAverage`) defined in a central `ModelRegistry`.
*   **Synthetic Data Generation**: Generates three distinct data patterns (Trend, Seasonality, Stochastic) to test model robustness.
*   **Metric Calculation**:
    *   **Cold Start Time**: Initial run time including JIT compilation or initialization overhead.
    *   **Warm Start Time**: Average execution time over `N_ITERATIONS` after the system is warm.
    *   **Interval Overhead**: The percentage cost of generating prediction intervals vs. simple point forecasts.
    *   **Accuracy (MAPE)**: Mean Absolute Percentage Error on a holdout test set.
*   **Automated Visualization**: Generates plots for Scalability, Accuracy, and Interval Overhead immediately after execution.

---

## High-Level Workflow

1.  **Configuration**: The script reads constants (Scales, Horizon, Seasonality) to set up the experiment.
2.  **Registry Lookup**: It looks up the requested model (e.g., `ETS`) in the `ModelRegistry` to find the corresponding Chronax class and StatsForecast class.
3.  **Data Generation Loop**:
    *   Iterates through Datasets: `Trend`, `Seasonality`, `Stochastic`.
    *   Iterates through Lengths: `100`, `500`, `1,000`, ... `50,000`.
4.  **Execution & Timing**:
    *   **StatsForecast**: Runs `fit_predict`. Measures Cold time. Runs N loops for Warm time. Runs with `intervals=True` for Overhead. Calculates MAPE.
    *   **Chronax**: Similar process. **Crucially**, it ensures JAX measures are blocked (`block_until_ready()`) to capture actual GPU execution time, not just dispatch time.
5.  **Logging**: All metrics are saved to a list of records.
6.  **Output**: Data is saved to `results/<model>_benchmark_results.csv` and plots are generated in the `results/` folder.

---

## Detailed Evaluation Mechanics

### 1. The Wrapper System `SFWrapper` & `ChronaxWrapper`
To ensure fair play, both libraries are wrapped to expose an identical `fit_predict(y, intervals=False)` interface.
*   **StatsForecast Wrapper**: Handles the creation of the required pandas DataFrame with `unique_id` and `ds` columns, which StatsForecast requires even for single series.
*   **Chronax Wrapper**: Handles moving numpy data to JAX `DeviceArray` (GPU memory) *before* the timing loop starts, so we measure pure inference speed, not data transfer speed (which is a separate bottleneck).

### 2. Time Measurement
*   **Cold Start**: The timer starts before the model is instantiated/called for the first time and stops after the first result is returned. This captures Python overhead, JIT compilation (for JAX), and Numba compilation (for StatsForecast).
*   **Warm Start**: The model is called `N_ITERATIONS` (default 5) times. The average duration is taken. This represents the "throughput" speed in a production environment where the model is already loaded.
*   **JAX Async Dispatch**: JAX operations are asynchronous. If we just stopped the timer after the function returns, we would only measure how long it took to *dispatch* the kernel to the GPU. The benchmark calls `.block_until_ready()` on the result buffer to force the CPU to wait until the GPU has actually finished the computation.

### 3. Accuracy (MAPE)
For every dataset and length, the last `HORIZON` (24 steps) points are held out as the test set. The model generates 24 predictions.
$$ \text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{y_{true} - y_{pred}}{y_{true}} \right| $$
This check ensures that while optimizing for speed, the models remain accurate using standard configurations.

### 4. Interval Overhead
We measure how much slower the model becomes when asked to produce probabilistic intervals (95% confidence) compared to a simple mean forecast.
*   **StatsForecast**: Uses Conformal Prediction or Bootstrap (often slower).
*   **Chronax**: Uses Vectorized Monte Carlo sampling (often very fast on GPU).
This metric highlights a key architectural advantage of JAX-based randomization.

---

## Limitations

1.  **Single Series Focus**: The scalable benchmark (Length 100-50k) tests *single time series* performance. It does not currently test "Global Model" capabilities (training one model on 10k different series simultaneously).
2.  **Synthetic Data**: The built-in datasets are synthetic. While they capture mathematical properties (trend, seasonality), they lack the messiness (outliers, missing values) of real-world data unless an external dataset is provided via `--dataset`.
3.  **Hardware Dependency**: Results are highly dependent on the CPU (for StatsForecast) vs GPU (for Chronax) balance of the host machine.
4.  **Model Availability**: Not all Chronax models currently have a direct StatsForecast equivalent enabled in the registry (some are commented out).

---

## Tutorial: Running `Benchmark.ipynb`

The provided Jupyter Notebook `Benchmark.ipynb` is the easiest way to run and analyze experiments.

### Prerequisites
*   Ensure you are in the `chronax` environment.
*   Ensure `benchmark_suite.py` is in the same folder (or `evaluation` folder).
*   Correct DLL/Library paths set if on Windows (handled by the notebook's initial cells).

### Step-by-Step

**1. Setup**
Run the first cell to import dependencies and setup the path.
```python
import sys, os
sys.path.append('..') 
# ... imports ...
```

**2. Configure the Run**
Decide which model you want to test. The default is `WindowAverage` (fastest).
You can choose from: `ETS`, `SeasonalNaive`, `AutoETS`, `TSB`, etc. (Check `ModelRegistry.get_registry().keys()` for list).

**3. Run the Benchmark Command**
Use the `!python` magic command to run the script from the notebook.
```python
# Run benchmark for ETS model and plot results
!python benchmark_suite.py --model ETS --plot
```
*   `--model`: Name of the model to test.
*   `--plot`: Automatically generates PNG plots in `results/`.
*   `--dataset`: (Optional) Path to a custom CSV if not using synthetic data.

**4. View Results**
Use the provided code block to load and view the latest results CSV directly in the notebook:
```python
import pandas as pd
import os
from IPython.display import display

results_dir = "results"
# ... (See notebook for full code) ...
display(df)
```

**5. View Plots**
The script saves plots to `results/`. You can view them using `IPython.display.Image`:
```python
from IPython.display import Image
Image(filename='results/ETS_scalability_plot.png') 
```
(Changing `ETS` to match your model name).

### Interpreting the Output
*   **Scalability Plot**: X-axis is series length (log), Y-axis is time (log). Lower is better. Look for where the lines diverge. Chronax usually shines at L > 10,000.
*   **Overhead Plot**: Bar chart showing % slowdown for intervals. Lower is better.
*   **Console Output**:
    *   `Warm=...`: Throughput speed.
    *   `Cold=...`: Compilation cost.
    *   `MAPE=...`: Accuracy check.
