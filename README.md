![Simulacrum Logo](https://github.com/Smlcrm/smlcrm-brand-assets/blob/main/Asset%201@4x-8.png?raw=true "Simulacrum — Chronax")

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

### Automatic Forecasting

Automatic model-selection wrappers that search over candidate configurations.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `AutoARIMA` | ✓ | ✓ | ✓ | Native |
| `AutoETS` | ✓ | ✓ | — | Native + conformal |
| `AutoTheta` | ✓ | ✓ | — | Monte Carlo + conformal |
| `AutoMFLES` | ✓ | ✓ | ✓ | Gaussian approx. + conformal |
| `AutoTBATS` | ✓ | ✓ | — | Conformal |
| `AutoCES` | ✓ | ✓ | — | Conformal |

### ARIMA Family

Autoregressive integrated moving-average models for autocorrelated series.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `ARIMA` | ✓ | ✓ | ✓ | Native |

### Theta Family

Theta-method forecasters for trend and seasonality decomposition.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `Theta` | ✓ | ✓ | — | Monte Carlo + conformal |

### Multiple Seasonalities & Decomposition

Models designed for multiple seasonal patterns or explicit trend-seasonal decomposition.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `MFLES` | ✓ | ✓ | ✓ | Conformal |
| `TBATS` | ✓ | ✓ | — | Conformal |
| `MSTL` | ✓ | ✓ | — | Conformal |
| `STL` | ✓ | ✓ | — | Conformal |

### Volatility Models

Models specialized for time-varying variance and heteroskedastic dynamics.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `GARCH` | ✓ | ✓ | — | Native + conformal |

### Baseline Models

Simple reference forecasters used as strong, interpretable baselines.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `HistoricAverage` | ✓ | ✓ | — | Native + conformal |
| `Naive` | ✓ | ✓ | — | Native + conformal |
| `SeasonalNaive` | ✓ | ✓ | — | Native + conformal |
| `WindowAverage` | ✓ | ✓ | — | Conformal |
| `SeasonalWindowAverage` | ✓ | ✓ | — | Conformal |
| `RandomWalkWithDrift` | ✓ | ✓ | — | Native + conformal |

### Exponential Smoothing

Level, trend, and seasonal smoothing models with recursive state updates.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `ETS` | ✓ | ✓ | — | Native + conformal |
| `Holt` | ✓ | ✓ | — | Native + conformal |
| `HoltWinters` | ✓ | ✓ | — | Native + conformal |
| `SimpleExponentialSmoothing` | ✓ | ✓ | — | Conformal |
| `SeasonalExponentialSmoothing` | ✓ | ✓ | — | Conformal |

### Sparse / Intermittent Demand

Forecasters tailored to sparse series with many zeros or irregular demand arrivals.

| Model | Point Forecast | Probabilistic Forecast | Exogenous Regressors | Interval Type |
|-------|----------------|------------------------|----------------------|---------------|
| `ADIDA` | ✓ | ✓ | — | Conformal |
| `CrostonClassic` | ✓ | ✓ | — | Conformal |
| `IMAPA` | ✓ | ✓ | — | Conformal |
| `TSB` | ✓ | ✓ | — | Native + conformal |

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

## Citation

If you use `chronax` in your research, please cite the library using the following BibTeX entry:

```bibtex
@software{chronax,
  title = {Chronax: High-performance, JAX-accelerated time-series forecasting},
  author = {Simulacrum},
  url = {https://github.com/Smlcrm/Chronax},
  year = {2026}
}
```

---

## License

MIT © Simulacrum, Inc.
