# Changelog

All notable changes to chronax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-03-09

### Added

- **BatchedForecaster** — Multi-series wrapper for fitting and forecasting multiple time series with one or more models. Supports dict, 2D arrays, pandas, and polars input formats.
- **20+ forecasting models** with a unified `fit()` / `predict()` interface:
  - **Automatic**: AutoARIMA, AutoETS, AutoTheta, AutoMFLES, AutoTBATS, AutoCES
  - **ARIMA**: ARIMA
  - **Theta**: Theta
  - **Multiple seasonalities**: MFLES, TBATS, MSTL, STL
  - **Volatility**: GARCH
  - **Exponential smoothing**: ETS, Holt, HoltWinters, SimpleExponentialSmoothing, SeasonalExponentialSmoothing
  - **Baselines**: HistoricAverage, Naive, SeasonalNaive, WindowAverage, SeasonalWindowAverage, RandomWalkWithDrift
  - **Sparse demand**: ADIDA, CrostonClassic, IMAPA, TSB
- **Prediction intervals** — Native and conformal interval support across models.
- **Memory-efficient forecasting** — `forecast()` method for stateless fit+predict without storing model state.
- **Read the Docs** — Sphinx documentation with model capability tables and API reference.
- **PyPI packaging** — Full packaging configuration for distribution.
- **Tutorials** — Model tutorials and notebooks for getting started.
- **Benchmark suite** — Evaluation against StatsForecast and other libraries.

### Changed

- **API consistency** — All models align with `BaseForecaster`; `forecast()` signatures standardized.
- **Project layout** — Restructured into proper Python package layout (`chronax/` namespace).
- **Documentation** — Complete type annotations and docstrings across all model files.
- **Utils refactor** — Reorganized `utils.py` into labeled sections, fixed dtype bugs, removed redundant helpers.

### Fixed

- ARIMA and AutoARIMA speed improvements.
- Seasonal Exponential Smoothing performance and tests.
- AutoMFLES MLE accuracy and performance.
- MFLES: adaptive changepoint policy, seasonal-tail fix, large-n optimizations.
- Theta and AutoTheta fixes.
- ADIDA warm start improvements.
- IMAPA efficiency and fixes.
- GARCH performance (on par with StatsForecast).
- Various dtype and JIT compilation issues.

### Dependencies

- Python ≥ 3.11
- JAX ≥ 0.4.0, jaxlib ≥ 0.4.0
- jaxopt, optax, numpy, matplotlib, seaborn

---

## [0.1.1] - Unreleased

### Fixed

- PyPI project description: logo now loads correctly (README image URL uses `raw/prod` branch).
- Repository URLs updated to https://github.com/Smlcrm/Chronax throughout (pyproject.toml, docs, README).

---

[0.1.0]: https://github.com/Smlcrm/Chronax/releases/tag/v0.1.0
[0.1.1]: https://github.com/Smlcrm/Chronax/releases/tag/v0.1.1
