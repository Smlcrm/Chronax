![Simulacrum Logo](https://github.com/Smlcrm/assets/blob/main/Asset%201@4x-8.png?raw=true "Simulacrum — Chronax")

# chronax

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/chronax.svg)](https://pypi.org/project/chronax/)

A high-performance, JAX-accelerated time-series forecasting library. `chronax` provides a comprehensive suite of classical and modern forecasting models — including AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, and more — with a unified `fit` / `predict` interface and hardware-accelerated execution via JAX.

---

## Features

- ⚡ **JAX-accelerated** — JIT-compiled model fitting and forecasting on CPU, GPU, or TPU
- 📈 **20+ forecasting models** including AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, STL, and more
- 🔁 **Unified API** — every model follows the same `fit()` → `predict()` pattern
- 📊 **Prediction intervals** — built-in conformal and native interval support
- ✅ **NumPy compatible** — accepts and returns standard array types
- 🧪 **Benchmarked** against established libraries for correctness and speed

---

## Installation

### From PyPI (recommended)

> Requires Python ≥ 3.11

```bash
pip install chronax
```

### From TestPyPI (pre-release testing)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chronax
```

### From GitHub source

Install directly from the latest commit:

```bash
pip install git+https://github.com/Smlcrm/ml-library-chronax.git
```

For local development:

```bash
git clone https://github.com/Smlcrm/ml-library-chronax.git
cd ml-library-chronax
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -e .
```

---

## Usage Overview

### Fitting a model and forecasting

All `chronax` models follow the same interface: instantiate, `fit()`, then `predict()`.

```python
import jax.numpy as jnp
from chronax.models import AutoARIMA

# Sample time series
y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

# Fit
model = AutoARIMA(season_length=12)
model = model.fit(y)

# Forecast 6 steps ahead
forecast = model.predict(h=6)
print("Forecast:", forecast['mean'])
```

### Prediction intervals

Request probabilistic forecasts by passing `level`:

```python
forecast = model.predict(h=6, level=[80, 95])

print("Point forecast:", forecast['mean'])
print("95% lower:", forecast['lo-95'])
print("95% upper:", forecast['hi-95'])
```

### In-sample fitted values

Retrieve the model's in-sample predictions after fitting:

```python
insample = model.predict_in_sample()
print("Fitted values:", insample['fitted'])
```

### Memory-efficient forecasting

Use `forecast()` to fit and predict in a single call without storing model state:

```python
from chronax.models import AutoETS

model = AutoETS(season_length=12)
result = model.forecast(y, h=6, level=[90])
print("Forecast:", result['mean'])
```

### Comparing multiple models

```python
from chronax.models import AutoARIMA, AutoETS, AutoTheta

models = {
    "AutoARIMA": AutoARIMA(season_length=12),
    "AutoETS": AutoETS(season_length=12),
    "AutoTheta": AutoTheta(season_length=12),
}

for name, m in models.items():
    m = m.fit(y)
    pred = m.predict(h=6)
    print(f"{name}: {pred['mean']}")
```

---

## Available Models

| Model | Class | Description |
|-------|-------|-------------|
| **AutoARIMA** | `AutoARIMA` | Automatic ARIMA with seasonal support |
| **ARIMA** | `ARIMA` | Manual ARIMA specification |
| **AutoETS** | `AutoETS` | Automatic Exponential Smoothing (Error, Trend, Seasonality) |
| **ETS** | `ETS` | Manual ETS specification |
| **AutoTheta** | `AutoTheta` | Automatic Theta method (STM, OTM, DSTM, DOTM) |
| **Theta** | `Theta` | Standard Theta Method |
| **TBATS** | `TBATS`, `AutoTBATS` | Trigonometric seasonality, Box-Cox, ARMA, Trend, Seasonality |
| **MFLES** | `MFLES`, `AutoMFLES` | Multiple Frequency Locally Estimated Scatterplot Smoothing |
| **AutoCES** | `AutoCES` | Complex Exponential Smoothing |
| **GARCH** | `GARCH` | Generalized Autoregressive Conditional Heteroskedasticity |
| **STL / MSTL** | `STL`, `MSTL` | Seasonal-Trend decomposition using LOESS |
| **Holt** | `Holt` | Holt's linear trend method |
| **Holt-Winters** | `HoltWinters` | Holt-Winters seasonal method |
| **SES** | `SimpleExponentialSmoothing` | Simple Exponential Smoothing |
| **Seasonal ES** | `SeasonalExponentialSmoothing` | Seasonal Exponential Smoothing |
| **Naive** | `Naive`, `SeasonalNaive` | Naive and Seasonal Naive baselines |
| **Window Avg** | `WindowAverage`, `SeasonalWindowAverage` | Moving average methods |
| **Croston** | `CrostonClassic` | Intermittent demand forecasting |
| **TSB** | `TSB` | Teunter-Syntetos-Babai method |
| **ADIDA** | `ADIDA` | Aggregate-Disaggregate Intermittent Demand Approach |
| **IMAPA** | `IMAPA` | Intermittent Multiple Aggregation Prediction Algorithm |
| **RWD** | `RandomWalkWithDrift` | Random Walk with Drift |
| **Historic Avg** | `HistoricAverage` | Simple historical average |

All models are importable from `chronax.models`.

---

## Tutorial: Forecast a Time Series in Five Steps

### 1. Install the package

```bash
pip install chronax
```

### 2. Create a project

```bash
mkdir my-forecast && cd my-forecast
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install chronax
```

### 3. Write a forecast script (`forecast.py`)

```python
import jax.numpy as jnp
from chronax.models import AutoETS

def main() -> None:
    # Monthly airline passengers (subset)
    y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

    model = AutoETS(season_length=12)
    model = model.fit(y)

    forecast = model.predict(h=6, level=[80, 95])
    print("Point forecast:", forecast['mean'].tolist())
    print("80% interval:", list(zip(forecast['lo-80'].tolist(), forecast['hi-80'].tolist())))
    print("95% interval:", list(zip(forecast['lo-95'].tolist(), forecast['hi-95'].tolist())))

if __name__ == "__main__":
    main()
```

### 4. Run the script

```bash
python forecast.py
```

### 5. Explore further

Try swapping `AutoETS` for `AutoARIMA` or `AutoTheta` and compare results — the API is identical across all models.

---

## Evaluation Benchmarks

We provide a benchmarking suite to evaluate `chronax` against other time-series libraries.

> **Note:** Benchmark dependencies (such as `statsforecast`, `pandas`, etc.) are **not** included in the core package to keep the installation lightweight.

```bash
# Install benchmark dependencies
pip install statsforecast pandas matplotlib

# Run the benchmark suite
python benchmarks/benchmark_suite.py
```

---

## Documentation

| Component | Description |
|-----------|-------------|
| `chronax.models.*` | All forecasting model classes (see table above) |
| `chronax.utils.*` | Utilities — loss functions, plotting, conformal intervals |

Explore inline docstrings for detailed parameter and return-type information.

---

## License

MIT © Simulacrum, Inc.
