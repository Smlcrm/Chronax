"""
BatchedForecaster — multi-series wrapper for Chronax models.

Fits and forecasts multiple time series with one or more models by looping
over series sequentially. Each model's internals are JIT-compiled, so the
sequential loop still outperforms pure-Python frameworks like StatsForecast.

Supported input formats (no hard dependency on any DataFrame library):
- dict[str, jnp.ndarray]: {series_id: array} — primary/internal format
- 2D jnp.ndarray or np.ndarray: (n_series, n_timesteps), equal-length only
- pandas DataFrame: grouped by id_col, sorted by time_col, target in target_col
- polars DataFrame: same semantics as pandas

All formats are converted internally to dict[str, jnp.ndarray].

Instance Attributes:
1. models: list[BaseForecaster] - Model instances to fit/forecast with
2. id_col: str - Column name for series identifier (DataFrame input)
3. time_col: str - Column name for timestamp (DataFrame input)
4. target_col: str - Column name for target values (DataFrame input)
5. _aliases: list[str] - Resolved display names (disambiguated if duplicates exist)
6. _fitted_models: dict[str, list[BaseForecaster]] | None - Per-series fitted
   model copies, populated by fit(), keyed by series ID

Methods:
1. fit(data, X) - Fit each model to each series, storing independent state
2. predict(h, X, level) - Generate h-step forecasts from fitted models
3. forecast(data, h, X, X_future, level) - Stateless fit+predict (no stored state)

Output Format:
All prediction methods return dict[str, dict[str, dict]] structured as
{series_id: {model_alias: prediction_dict}} where prediction_dict contains
'mean' and optionally 'lo-{level}', 'hi-{level}' keys.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from collections import Counter

from chronax.models.base_forecaster import BaseForecaster

__all__ = ['BatchedForecaster']


def _resolve_aliases(models: list[BaseForecaster]) -> list[str]:
    """Return unique display names, appending _1, _2, ... for duplicates."""
    counts = Counter(m.alias for m in models)
    seen: dict[str, int] = {}
    aliases = []
    for m in models:
        name = m.alias
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            aliases.append(f"{name}_{seen[name]}")
        else:
            aliases.append(name)
    return aliases


# =============================================================================
# Input coercion
# =============================================================================

def _coerce_to_dict(
    data: dict[str, jnp.ndarray] | jnp.ndarray | np.ndarray | object,
    id_col: str,
    time_col: str,
    target_col: str,
) -> dict[str, jnp.ndarray]:
    """Convert supported input formats to dict[str, jnp.ndarray].

    Tries each format in order: dict → 2D array → pandas → polars.
    Raises TypeError if none match.
    """
    if isinstance(data, dict):
        return data

    # 2D array — rows are series, columns are timesteps
    if isinstance(data, (jnp.ndarray, np.ndarray)):
        if data.ndim != 2:
            raise ValueError(
                f"Array input must be 2D (n_series, n_timesteps), got {data.ndim}D"
            )
        return {str(i): jnp.asarray(data[i]) for i in range(data.shape[0])}

    # Lazy pandas check — avoids hard dependency
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            result = {}
            for sid, group in data.groupby(id_col):
                result[str(sid)] = jnp.asarray(
                    group.sort_values(time_col)[target_col].values
                )
            return result
    except ImportError:
        pass

    # Lazy polars check — avoids hard dependency
    try:
        import polars as pl  # type: ignore[import-unresolved]

        if isinstance(data, pl.DataFrame):
            result = {}
            for sid, group in data.group_by(id_col):
                # polars group_by returns (key_tuple, df)
                key = str(sid[0]) if isinstance(sid, tuple) else str(sid)
                result[key] = jnp.asarray(
                    group.sort(time_col)[target_col].to_numpy()
                )
            return result
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected dict, 2D array, pandas DataFrame, or polars DataFrame."
    )


# =============================================================================
# BatchedForecaster
# =============================================================================

class BatchedForecaster:
    """Multi-series wrapper that fits and forecasts with multiple models.

    Loops over series sequentially, leveraging JIT-compiled model internals
    for speed. Accepts dict, 2D array, pandas, or polars DataFrame inputs.
    """

    def __init__(
        self,
        models: list[BaseForecaster],
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> None:
        if not models:
            raise ValueError("models list must not be empty.")
        self.models = models
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._aliases = _resolve_aliases(models)
        self._fitted_models: dict[str, list[BaseForecaster]] | None = None

    def fit(
        self,
        data: dict[str, jnp.ndarray] | jnp.ndarray | np.ndarray | object,
        X: dict[str, jnp.ndarray] | None = None,
    ) -> BatchedForecaster:
        """Fit each model to each series, storing independent fitted copies.

        Creates a shallow copy (BaseForecaster.new()) of each model per series
        so that fitted state is independent across series.

        Parameters
        ----------
        data : dict, 2D array, pandas DataFrame, or polars DataFrame
            Input time series data in any supported format.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series in-sample exogenous variables, keyed by series ID.
            Each value should be shape (n_timesteps, n_features). Series
            without exogenous data can be omitted from the dict.

        Returns
        -------
        self
        """
        series_dict = _coerce_to_dict(
            data, self.id_col, self.time_col, self.target_col
        )
        self._fitted_models = {}
        for sid, y in series_dict.items():
            x_sid = X.get(sid) if X is not None else None
            fitted_for_series = []
            for model in self.models:
                # shallow copy is safe here — all pre-fit attrs are immutable;
                # fit() assigns a new model_ dict rather than mutating in place
                model_copy = model.new()
                model_copy.fit(y, X=x_sid)
                fitted_for_series.append(model_copy)
            self._fitted_models[sid] = fitted_for_series
        return self

    def predict(
        self,
        h: int,
        X: dict[str, jnp.ndarray] | None = None,
        level: list[int | float] | None = None,
    ) -> dict[str, dict[str, dict]]:
        """Generate h-step forecasts from fitted models.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series future exogenous variables, keyed by series ID.
            Each value should be shape (h, n_features).
        level : list[int | float] | None
            Confidence levels for prediction intervals (e.g. [80, 95]).

        Returns
        -------
        dict[str, dict[str, dict]]
            {series_id: {model_alias: prediction_dict}}
        """
        if self._fitted_models is None:
            raise ValueError("Must call fit() before predict().")
        results: dict[str, dict[str, dict]] = {}
        for sid, models in self._fitted_models.items():
            x_sid = X.get(sid) if X is not None else None
            results[sid] = {}
            for alias, model in zip(self._aliases, models):
                results[sid][alias] = model.predict(h=h, X=x_sid, level=level)
        return results

    def forecast(
        self,
        data: dict[str, jnp.ndarray] | jnp.ndarray | np.ndarray | object,
        h: int,
        X: dict[str, jnp.ndarray] | None = None,
        X_future: dict[str, jnp.ndarray] | None = None,
        level: list[int | float] | None = None,
    ) -> dict[str, dict[str, dict]]:
        """Stateless fit+predict for each model on each series.

        Creates a fresh copy of each model per series via new() to prevent
        cross-series state leakage. Does not store fitted state on self.

        Parameters
        ----------
        data : dict, 2D array, pandas DataFrame, or polars DataFrame
            Input time series data in any supported format.
        h : int
            Forecast horizon.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series in-sample exogenous variables, keyed by series ID.
        X_future : dict[str, jnp.ndarray] or None, default None
            Per-series future exogenous variables, keyed by series ID.
        level : list[int | float] | None
            Confidence levels for prediction intervals (e.g. [80, 95]).

        Returns
        -------
        dict[str, dict[str, dict]]
            {series_id: {model_alias: prediction_dict}}
        """
        series_dict = _coerce_to_dict(
            data, self.id_col, self.time_col, self.target_col
        )
        results: dict[str, dict[str, dict]] = {}
        for sid, y in series_dict.items():
            x_sid = X.get(sid) if X is not None else None
            x_future_sid = X_future.get(sid) if X_future is not None else None
            results[sid] = {}
            for alias, model in zip(self._aliases, self.models):
                results[sid][alias] = model.new().forecast(
                    y=y, h=h, X=x_sid, X_future=x_future_sid, level=level
                )
        return results
