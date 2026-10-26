"""
BatchedForecaster: multi-series wrapper for Chronax models.

Fits and forecasts several time series with one or more models by looping
over the series sequentially; each model's internals are JIT-compiled.

Supported input formats (no hard dependency on any DataFrame library):

- ``dict[str, jnp.ndarray]``: ``{series_id: array}``, the internal format
- 2-D ``jnp.ndarray`` or ``np.ndarray``: ``(n_series, n_timesteps)``,
  equal-length series only
- pandas DataFrame: grouped by ``id_col``, sorted by ``time_col``, target in
  ``target_col``
- polars DataFrame: same semantics as pandas

All formats are converted internally to ``dict[str, jnp.ndarray]``.

Every prediction method returns ``{series_id: {model_alias: prediction_dict}}``
where ``prediction_dict`` holds ``"mean"`` and optionally
``"lo-{lv}"``/``"hi-{lv}"``.
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

    Tries each format in order: dict, 2-D array, pandas, polars.

    Raises
    ------
    ValueError
        If an array input is not 2-D.
    TypeError
        If no supported format matches.
    """
    if isinstance(data, dict):
        return data

    # 2-D array: rows are series, columns are timesteps
    if isinstance(data, (jnp.ndarray, np.ndarray)):
        if data.ndim != 2:
            raise ValueError(
                f"Array input must be 2D (n_series, n_timesteps), got {data.ndim}D"
            )
        return {str(i): jnp.asarray(data[i]) for i in range(data.shape[0])}

    # Lazy pandas import: no hard dependency
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

    # Lazy polars import: no hard dependency
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
    """Multi-series wrapper that fits and forecasts with several models.

    Loops over the series sequentially; each model's internals are
    JIT-compiled. Accepts dict, 2-D array, pandas or polars DataFrame input.

    Parameters
    ----------
    models : list of BaseForecaster
        Model instances to fit and forecast with. Duplicate aliases are
        suffixed ``_1``, ``_2``, ... in the output.
    id_col : str, default "unique_id"
        Series identifier column (DataFrame input).
    time_col : str, default "ds"
        Timestamp column used for sorting (DataFrame input).
    target_col : str, default "y"
        Target column (DataFrame input).

    Raises
    ------
    ValueError
        If ``models`` is empty.
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

        Creates a shallow copy (``BaseForecaster.new()``) of each model per
        series so that fitted state is independent across series.

        Parameters
        ----------
        data : dict, 2-D array, pandas DataFrame, or polars DataFrame
            Input time series in any supported format.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series in-sample exogenous regressors keyed by series ID,
            each of shape ``(T, F)``. Series without exogenous data can be
            omitted.

        Returns
        -------
        BatchedForecaster
            ``self``.

        Raises
        ------
        ValueError
            If an array input is not 2-D.
        TypeError
            If ``data`` is not a supported format.
        """
        series_dict = _coerce_to_dict(
            data, self.id_col, self.time_col, self.target_col
        )
        self._fitted_models = {}
        for sid, y in series_dict.items():
            x_sid = X.get(sid) if X is not None else None
            fitted_for_series = []
            for model in self.models:
                # A shallow copy is safe: pre-fit attributes are immutable and
                # fit() assigns a new model_ dict rather than mutating in place.
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
        """Generate h-step forecasts from the fitted models.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series future exogenous regressors keyed by series ID, each
            of shape ``(h, F)``.
        level : list of int or float, or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.

        Returns
        -------
        dict[str, dict[str, dict]]
            ``{series_id: {model_alias: prediction_dict}}``.

        Raises
        ------
        ValueError
            If called before :meth:`fit`.
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

        Creates a fresh copy of each model per series via ``new()`` so no
        state leaks across series or onto ``self``.

        Parameters
        ----------
        data : dict, 2-D array, pandas DataFrame, or polars DataFrame
            Input time series in any supported format.
        h : int
            Forecast horizon.
        X : dict[str, jnp.ndarray] or None, default None
            Per-series in-sample exogenous regressors keyed by series ID,
            each of shape ``(T, F)``.
        X_future : dict[str, jnp.ndarray] or None, default None
            Per-series future exogenous regressors keyed by series ID, each
            of shape ``(h, F)``.
        level : list of int or float, or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.

        Returns
        -------
        dict[str, dict[str, dict]]
            ``{series_id: {model_alias: prediction_dict}}``.

        Raises
        ------
        ValueError
            If an array input is not 2-D.
        TypeError
            If ``data`` is not a supported format.
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
