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
<!-- Badge colours: 2121a5 = token brand-book.brand-dark-blue (arXiv, license, language); 3483fa = token brand-book.brand-bright-blue (release). -->
<p align="center">
  <a href="https://arxiv.org/abs/2604.16719"><img alt="arXiv: 2604.16719" src="https://img.shields.io/badge/arXiv-2604.16719-2121a5"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/github/license/Smlcrm/Chronax?color=2121a5"></a>
  <a href="https://github.com/Smlcrm/Chronax/releases"><img alt="Release: latest GitHub release" src="https://img.shields.io/github/v/release/Smlcrm/Chronax?color=3483fa"></a>
  <!-- smlcrm:no-ci: the repository has no build or test workflow (web-docs-prod.yml only publishes documentation) -->
  <a href="pyproject.toml"><img alt="Language: Python 3.11 or later" src="https://img.shields.io/badge/python-%E2%89%A53.11-2121a5"></a>
</p>
<!-- smlcrm:end badges -->

<!-- smlcrm:begin overview -->
## Overview

Chronax implements 59 time-series forecasting models in JAX behind one `fit(y)` / `predict(h)` interface. The 28 classical models are JAX ports of the statsforecast family: ARIMA, ETS, Theta, TBATS, MFLES, MSTL and STL, CES, GARCH, Holt and Holt-Winters exponential smoothing, intermittent-demand methods (Croston, ADIDA, IMAPA, TSB) and simple baselines, with automatic model selection for ARIMA, ETS, Theta, TBATS, MFLES and CES. The 31 neural models are Flax NNX ports of the neuralforecast family: MLP, NHITS, NBEATS, DLinear, NLinear, TiDE, TSMixer, TimeMixer, TimeXer, TFT, Informer, PatchTST, iTransformer, TCN, BiTCN, GRU, RNN, DeepAR, xLSTM and others, plus HINT for hierarchical reconciliation. `predict(h)` returns a dictionary of JAX arrays and adds native or conformal prediction intervals when you pass `level`. Fitting and forecasting run under JAX, so the same code runs on CPU, GPU or TPU.

**Who it is for.** Python developers and researchers who forecast univariate time series and want classical and neural models that run inside a JAX workflow.

**What it does not do.** It forecasts one univariate series per model (HINT reconciles a hierarchy, and MLP accepts a panel); it does not load data frames or manage panels of series for you. It ships no pretrained foundation models and is not a hosted forecasting service. The benchmark suite's comparison libraries (statsforecast, neuralforecast, pandas) are not installed with the package. Five neural wrappers (DeepAR, NBEATS, RNN, TiDE, TSMixer) keep their own fit and forecast signatures and return no prediction intervals; Autoformer and FEDformer forecast points only.
<!-- smlcrm:end overview -->

<!-- smlcrm:begin quickstart -->
## Quickstart

<!-- smlcrm:tested 2026-10-26 macOS 26.3.1 (arm64), Python 3.13.14, fresh venv, chronax 0.1.0 installed from the repository source with jax 0.7.2 and flax 0.10.7 from PyPI; the same output on two runs -->
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
mean:  [128.9, 135.0, 145.2, 141.8, 135.5, 149.1]
lo-95: [116.9, 118.2, 124.7, 118.2, 109.1, 120.2]
hi-95: [140.9, 151.7, 165.7, 165.4, 161.8, 178.0]
```
<!-- smlcrm:end quickstart -->

## Features

- **JAX-accelerated**: just-in-time (JIT) compiled model fitting and forecasting on CPU, GPU or TPU
- **59 forecasting models**: 28 classical (AutoARIMA, AutoETS, AutoTheta, TBATS, MFLES, GARCH, STL and more) and 31 neural (TFT, Informer, PatchTST, NHITS, TiDE, TCN, XLSTM and more)
- **Unified API**: every model follows the same `fit()` then `predict()` pattern
- **Prediction intervals**: conformal and native interval support
- **NumPy compatible**: accepts and returns standard array types
- **Benchmarked** against statsforecast and neuralforecast for accuracy and speed

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

## Usage overview

### Fitting a model and forecasting

All `chronax` models follow the same interface: instantiate, `fit()`, then `predict()`.

```python
import jax.numpy as jnp
from chronax.models import AutoARIMA

# Sample time series
y = jnp.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])

# Fit
model = AutoARIMA(period=12)
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
from chronax.models import AutoETS

model = AutoETS(season_length=12).fit(y)
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
    "AutoARIMA": AutoARIMA(period=12),
    "AutoETS": AutoETS(season_length=12),
    "AutoTheta": AutoTheta(season_length=12),
}

for name, m in models.items():
    m = m.fit(y)
    pred = m.predict(h=6)
    print(f"{name}: {pred['mean']}")
```

---

## Available models

### Automatic forecasting

Automatic model-selection wrappers that search over candidate configurations.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `AutoARIMA` | yes | yes | yes | Native |
| `AutoETS` | yes | yes | no | Native + conformal |
| `AutoTheta` | yes | yes | no | Monte Carlo + conformal |
| `AutoMFLES` | yes | yes | yes | Gaussian approx. + conformal |
| `AutoTBATS` | yes | yes | no | Conformal |
| `AutoCES` | yes | yes | no | Conformal |

### ARIMA family

Autoregressive integrated moving-average models for autocorrelated series.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `ARIMA` | yes | yes | yes | Native |

### Theta family

Theta-method forecasters for trend and seasonality decomposition.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `Theta` | yes | yes | no | Monte Carlo + conformal |

### Multiple seasonalities and decomposition

Models designed for multiple seasonal patterns or explicit trend-seasonal decomposition.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `MFLES` | yes | yes | yes | Conformal |
| `TBATS` | yes | yes | no | Conformal |
| `MSTL` | yes | yes | no | Conformal |
| `STL` | yes | yes | no | Conformal |

### Volatility models

Models specialized for time-varying variance and heteroskedastic dynamics.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `GARCH` | yes | yes | no | Native + conformal |

### Baseline models

Simple reference forecasters used as strong, interpretable baselines.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `HistoricAverage` | yes | yes | no | Native + conformal |
| `Naive` | yes | yes | no | Native + conformal |
| `SeasonalNaive` | yes | yes | no | Native + conformal |
| `WindowAverage` | yes | yes | no | Conformal |
| `SeasonalWindowAverage` | yes | yes | no | Conformal |
| `RandomWalkWithDrift` | yes | yes | no | Native + conformal |

### Exponential smoothing

Level, trend, and seasonal smoothing models with recursive state updates.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `ETS` | yes | yes | no | Native + conformal |
| `Holt` | yes | yes | no | Native + conformal |
| `HoltWinters` | yes | yes | no | Native + conformal |
| `SimpleExponentialSmoothing` | yes | yes | no | Conformal |
| `SeasonalExponentialSmoothing` | yes | yes | no | Conformal |

### Sparse and intermittent demand

Forecasters tailored to sparse series with many zeros or irregular demand arrivals.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `ADIDA` | yes | yes | no | Conformal |
| `CrostonClassic` | yes | yes | no | Conformal |
| `IMAPA` | yes | yes | no | Conformal |
| `TSB` | yes | yes | no | Native + conformal |

### Deep learning

Neural forecasters implemented in JAX/Flax, ported from the Nixtla `neuralforecast` family. "Conformal" means `predict(h, level=...)` uses the base-class conformal machinery once `conformal_params` is set; "native" means the model's own interval head is used when it is configured.

| Model | Point forecast | Probabilistic forecast | Exogenous regressors | Interval type |
|-------|----------------|------------------------|----------------------|---------------|
| `MLP` | yes | yes | yes | Conformal; native with a quantile or GMM loss |
| `NHITS` | yes | yes | yes | Conformal; native with a quantile or GMM loss |
| `HINT` | yes | yes | yes | Native coherent samples + conformal |
| `NBEATS` | yes | no | no | none |
| `DLinear` | yes | yes | no | Conformal |
| `NLinear` | yes | yes | no | Conformal |
| `XLinear` | yes | yes | no | Conformal |
| `RMoK` | yes | yes | no | Conformal |
| `KAN` | yes | yes | no | Conformal |
| `TiDE` | yes | no | no | none |
| `TSMixer` | yes | no | no | none |
| `TimeMixer` | yes | yes | no | Conformal |
| `TimeXer` | yes | yes | yes | Conformal |
| `SOFTS` | yes | yes | no | Conformal |
| `SOFTSSharp` | yes | yes | no | Conformal |
| `TFT` | yes | yes | yes | Conformal; native with a quantile loss |
| `Informer` | yes | yes | yes | Conformal; native with a quantile loss |
| `iTransformer` | yes | yes | no | Conformal |
| `VanillaTransformer` | yes | yes | no | Conformal |
| `PatchTST` | yes | yes | no | Conformal |
| `Autoformer` | yes | no | no | none |
| `FEDformer` | yes | no | no | none |
| `TCN` | yes | yes | yes | Conformal; native with a quantile loss |
| `BiTCN` | yes | yes | no | Conformal |
| `StemGNN` | yes | yes | no | Conformal; native with a quantile loss |
| `DeepNPTS` | yes | yes | no | Conformal |
| `RNN` | yes | no | yes | none |
| `GRU` | yes | yes | no | Conformal |
| `DilatedRNN` | yes | yes | no | Conformal |
| `XLSTM` | yes | yes | no | Conformal |
| `DeepAR` | yes | yes | yes | Monte Carlo sample paths |

Exogenous regressors enter through `fit(y, X=...)` for MLP, NHITS, TCN, TimeXer and HINT (historical covariates), through `futr_exog=` for Informer, and through either for TFT; RNN takes `hist_exog`, `futr_exog` and `stat_exog` sequences; DeepAR takes `x_futr_list` and `x_stat_list`. NBEATS, TiDE, TSMixer and RNN subclass `BaseForecaster` but their `forecast(y, h)` returns an array, and `predict(h, level=...)` raises `NotImplementedError`. DeepAR has its own `fit(y_series, ...)` / `predict(y_series, ..., num_samples)` API and does not subclass `BaseForecaster`. Autoformer and FEDformer are point forecasters: `predict(h, level=...)` is not supported.

All models are importable from `chronax.models`.

---

## Tutorial: forecast a time series in five steps

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

Swap `AutoETS` for `AutoARIMA` or `AutoTheta` and compare the results. The API is identical across all models.

---

## Evaluation benchmarks

We provide a benchmarking suite to evaluate `chronax` against other time-series libraries.

> **Note:** Benchmark dependencies such as `statsforecast` and `pandas` are not included in the core package.

```bash
# Install benchmark dependencies
pip install statsforecast pandas matplotlib

# Run the benchmark suite
python benchmarks/benchmark_suite.py
```

A separate neural benchmark harness lives under `benchmarks/neural/` and compares Chronax neural forecasters against their Nixtla `neuralforecast` counterparts under a symmetric training budget. Set up the reference environment with `bash benchmarks/setup_nf_venv.sh`, then run it via `python benchmarks/neural/run.py` (models and settings are configured in `benchmarks/neural/config.yaml`).

---

## Documentation

| Component | Description |
|-----------|-------------|
| `chronax.models.*` | All forecasting model classes (see table above) |
| `chronax.utils.*` | Utilities: loss functions, plotting, conformal intervals |

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

If you use Chronax in your work, cite the paper, [arXiv:2604.16719](https://arxiv.org/abs/2604.16719), and the software you used. GitHub's "Cite this repository" button reads the same data from [`CITATION.cff`](CITATION.cff) and offers the paper first.

```bibtex
@misc{smlcrm_chronax_paper,
  title         = {Chronax: A Jax Library for Univariate Statistical Forecasting and Conformal Inference},
  author        = {Carey, Xan and Deshmukh, Yash and Huang, Aileen and Jadhav, Sunit and Tekawade, Omkar and Yang, Lorraine and Tiwary, Anvesha and Riano, Gerardo and Greenwald, Amy and Goktas, Denizalp},
  year          = {2026},
  eprint        = {2604.16719},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.16719}
}
```

The software:

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
