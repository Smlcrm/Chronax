<!-- smlcrm:begin header -->
<!-- Logo: Smlcrm/design-system assets/logo/logo-digital-1.svg @ adc00ff. Light fill #2121a5 = token product.logo-ink; dark fill #ffffff = token brand-book.brand-white. -->
<p align="center">
  <a href="https://smlcrm.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/smlcrm-logo-dark.svg">
      <img alt="Simulacrum" src=".github/assets/smlcrm-logo-light.svg" width="300">
    </picture>
  </a>
</p>

<h1 align="center">Chronax</h1>

<p align="center">A JAX time-series forecasting library: statistical forecasting models behind one fit / predict interface.</p>

<p align="center"><a href="https://smlcrm.com">smlcrm.com</a></p>
<!-- smlcrm:end header -->

<!-- smlcrm:begin badges -->
<!-- Badge colours: 2121a5 = token brand-book.brand-dark-blue (license, language); 3483fa = token brand-book.brand-bright-blue (release). -->
<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/Smlcrm/Chronax?color=2121a5"></a>
  <a href="https://github.com/Smlcrm/Chronax/releases"><img alt="Release: latest GitHub release" src="https://img.shields.io/github/v/release/Smlcrm/Chronax?color=3483fa"></a>
  <!-- smlcrm:no-ci: the repository has no build or test workflow (web-docs-prod.yml only publishes documentation) -->
  <a href="pyproject.toml"><img alt="Language: Python 3.11 or later" src="https://img.shields.io/badge/python-%E2%89%A53.11-2121a5"></a>
</p>
<!-- smlcrm:end badges -->

<!-- smlcrm:begin overview -->
## Overview

Chronax implements classical forecasting models in JAX: ARIMA, ETS, Theta, TBATS, MFLES, MSTL and STL, GARCH, exponential smoothing, intermittent-demand methods (Croston, ADIDA, IMAPA, TSB) and simple baselines, with automatic model selection for ARIMA, ETS, Theta, TBATS, MFLES and CES. Every model is fitted with `fit(y)` and forecasts with `predict(h)`, which returns a dictionary of JAX arrays, and adds native or conformal prediction intervals when you pass `level`. Fitting and forecasting run under JAX, so the same code runs on CPU, GPU or TPU. The package also contains early neural forecasters (Autoformer, iTransformer, KAN) that the model tables below do not cover yet.

**Who it is for.** Python developers and researchers who forecast univariate time series and want statistical models that run inside a JAX workflow.

**What it does not do.** It forecasts one univariate series per model; it does not load data frames or manage panels of series for you. It ships no pretrained foundation models and is not a hosted forecasting service. The benchmark suite's comparison libraries (statsforecast, pandas) are not installed with the package.
<!-- smlcrm:end overview -->

<!-- smlcrm:begin quickstart -->
## Quickstart

<!-- smlcrm:tested 2026-10-05 macOS 26.6 (arm64), Python 3.11.14, fresh venv, chronax 0.1.1 and jax 0.10.2 from PyPI -->
Requires Python 3.11 or later.

```bash
pip install chronax
```

Save this as `forecast.py` and run `python forecast.py`. It fits an automatically selected ETS model to two years of monthly data and forecasts six months with a 95% interval. The first run takes longer while JAX compiles.

```python
import jax.numpy as jnp
from chronax.models import AutoETS

# Monthly airline passengers, first two years
y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140],
              dtype=jnp.float32)

model = AutoETS(season_length=12).fit(y)
forecast = model.predict(h=6, level=[95])

print("mean: ", [round(v, 1) for v in forecast["mean"].tolist()])
print("lo-95:", [round(v, 1) for v in forecast["lo-95"].tolist()])
print("hi-95:", [round(v, 1) for v in forecast["hi-95"].tolist()])
```

Expected output:

```text
mean:  [139.5, 139.5, 139.5, 139.5, 139.5, 139.5]
lo-95: [108.7, 96.3, 86.8, 78.8, 71.7, 65.2]
hi-95: [170.3, 182.6, 192.2, 200.2, 207.3, 213.7]
```
<!-- smlcrm:end quickstart -->

## Features

- ⚡ **JAX-accelerated** — JIT-compiled model fitting and forecasting on CPU, GPU, or TPU
- 📈 **20+ forecasting models** including AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, STL, and more
- 🔁 **Unified API** — every model follows the same `fit()` → `predict()` pattern
- 📊 **Prediction intervals** — built-in conformal and native interval support
- ✅ **NumPy compatible** — accepts and returns standard array types
- 🧪 **Benchmarked** against established libraries for correctness and speed

---

## Installation

The release from PyPI is in the [Quickstart](#quickstart). Other ways to install:

### From TestPyPI (pre-release testing)

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chronax
```

### From GitHub source

Install directly from the latest commit:

```bash
pip install git+https://github.com/Smlcrm/Chronax.git
```

For local development:

```bash
git clone https://github.com/Smlcrm/Chronax.git
cd Chronax
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

<!-- smlcrm:begin links -->
## Links

- Documentation: [smlcrm.com/docs/chronax](https://www.smlcrm.com/docs/chronax/)
- Package: [chronax on PyPI](https://pypi.org/project/chronax/)
- Issues: [Smlcrm/Chronax/issues](https://github.com/Smlcrm/Chronax/issues)
- Releases: [Smlcrm/Chronax/releases](https://github.com/Smlcrm/Chronax/releases)
- Website: [smlcrm.com](https://smlcrm.com)
- Related repositories: [Smlcrm/TempusBench](https://github.com/Smlcrm/TempusBench), the time-series forecasting benchmark whose metrics Chronax's benchmark suite uses
<!-- smlcrm:end links -->

<!-- smlcrm:begin citation -->
## Citation

If you use Chronax in your work, cite it as below. GitHub's "Cite this repository" button reads the same data from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{smlcrm_chronax,
  title   = {Chronax},
  author  = {{Simulacrum, Inc.}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Smlcrm/Chronax}
}
```
<!-- smlcrm:end citation -->

<!-- smlcrm:begin license -->
## License and contact

Released under the MIT license. See [LICENSE](LICENSE).

Contact: [support@smlcrm.com](mailto:support@smlcrm.com) · [smlcrm.com](https://smlcrm.com) · [github.com/Smlcrm](https://github.com/Smlcrm)
<!-- smlcrm:end license -->
