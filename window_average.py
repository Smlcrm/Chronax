from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp

# from utils.py import (
#     ConformalIntervals,
#     _ensure_float,
# )


#Import the Helper functions here

def _add_conformal_distribution_intervals(
    fcst: Dict,
    cs: jnp.ndarray,
    level: List[Union[int, float]],
) -> Dict:
    r"""
    Adds conformal intervals to the `fcst` dict based on conformal scores `cs`.
    `level` should be already sorted. This strategy creates forecasts paths
    based on errors and calculate quantiles using those paths.
    """
    alphas = [100 - lv for lv in level]
    cuts = [alpha / 200 for alpha in reversed(alphas)]
    cuts.extend(1 - alpha / 200 for alpha in alphas)
    mean = fcst["mean"].reshape(1, -1)
    scores = jnp.vstack([mean - cs, mean + cs])
    quantiles = jnp.quantile(
        scores,
        cuts,
        axis=0,
    )
    quantiles = quantiles.reshape(len(cuts), -1)
    lo_cols = [f"lo-{lv}" for lv in reversed(level)]
    hi_cols = [f"hi-{lv}" for lv in level]
    out_cols = lo_cols + hi_cols
    for i, col in enumerate(out_cols):
        fcst[col] = quantiles[i]
    return fcst

def _get_conformal_method(method: str):
    available_methods = {
        "conformal_distribution": _add_conformal_distribution_intervals,
        # "conformal_error": _add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f"please choose one of {', '.join(available_methods.keys())}"
        )
    return available_methods[method]

def _repeat_val(val: float, h: int) -> jnp.ndarray:
    return jnp.full((h,), jnp.asarray(val))

def _window_average(
    y: jnp.ndarray,  # time series
    h: int,  # forecasting horizon
    fitted: bool,  # fitted values
    window_size: int,  # window size
) -> Dict[str, jnp.ndarray]:
    if fitted:
        raise NotImplementedError("return fitted")
    if y.size < window_size:
        return {"mean": jnp.full((h,), jnp.nan, dtype=y.dtype)}
    wavg = jnp.mean(y[-window_size:])
    mean = _repeat_val(val=wavg, h=h)
    return {"mean": mean}

def _ensure_float(x: jnp.ndarray) -> jnp.ndarray:
    if x.dtype not in (jnp.float32, jnp.float64):
        x = x.astype(jnp.float32)
    return x

# Classes
class ConformalIntervals:
    """Class for storing conformal intervals metadata information.

    Args:
        n_windows (int, optional): Number of windows for conformal intervals. Defaults to 2.
        h (int, optional): Forecasting horizon. Defaults to 1.
        method (str, optional): Method for conformal intervals. Defaults to "conformal_distribution".
    """

    def __init__(
        self,
        n_windows: int = 2,
        h: int = 1,
        method: str = "conformal_distribution",
    ):
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.h = h
        self.method = method

class _TS:
    uses_exog = False

    def new(self):
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b

    def __repr__(self):
        return self.alias

    
    def _conformity_scores(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        y = _ensure_float(y)  # assume your JAX version from earlier
        n_windows = self.prediction_intervals.n_windows  # type: ignore[attr-defined]
        h = self.prediction_intervals.h                 # type: ignore[attr-defined]
        n_samples = y.size

        # use as many windows as possible for short series
        # subtract 1 for the training set
        n_windows = int(min(n_windows, (n_samples - 1) // h))
        if n_windows < 2:
            raise ValueError(
                f"Prediction intervals settings require at least {2 * h + 1:,} samples, "
                f"serie has {n_samples:,}."
            )

        test_size = n_windows * h
        cs = jnp.empty((n_windows, h), dtype=y.dtype)

        for i_window in range(n_windows):
            train_end = n_samples - test_size + i_window * h
            y_train = y[:train_end]
            y_test  = y[train_end : train_end + h]

            if X is not None:
                X_train = X[:train_end]
                X_test  = X[train_end : train_end + h]
            else:
                X_train = None
                X_test  = None

            fcst_window = self.forecast(h=h, y=y_train, X=X_train, X_future=X_test)  # type: ignore[attr-defined]
            row = jnp.abs(fcst_window["mean"] - y_test)
            cs = cs.at[i_window].set(row)

        return cs
    
    @property
    def _conformal_method(self):
        return _get_conformal_method(self.prediction_intervals.method)

    def _store_cs(self, y, X):
        if self.prediction_intervals is not None:
            self._cs = self._conformity_scores(y, X)

    def _add_conformal_intervals(self, fcst, y, X, level):
        if self.prediction_intervals is not None and level is not None:
            cs = self._conformity_scores(y, X) if y is not None else self._cs
            res = self._conformal_method(fcst=fcst, cs=cs, level=level)
            return res
        return fcst

    def _add_predict_conformal_intervals(self, fcst, level):
        return self._add_conformal_intervals(fcst=fcst, y=None, X=None, level=level)
    
   


class WindowAverage(_TS):
    def __init__(self, window_size: str, alias: str = "WindowAverage",
        prediction_intervals: Optional[ConformalIntervals] = None):
        
        r"""WindowAverage model.

        Uses the average of the last $k$ observations, with $k$ the length of the window.
        Wider windows will capture global trends, while narrow windows will reveal local trends.
        The length of the window selected should take into account the importance of past
        observations and how fast the series changes.

        References:
            - [Rob J. Hyndman and George Athanasopoulos (2018). "forecasting principles and practice, Simple Methods"](https://otexts.com/fpp3/simple-methods.html).

        Args:
            window_size (int): Size of truncated series on which average is estimated.
            alias (str): Custom name of the model.
            prediction_intervals (Optional[ConformalIntervals]): Information to compute conformal prediction intervals.
                This is required for generating future prediction intervals.
        r"""

        self.window_size = window_size
        self.alias = alias
        self.prediction_intervals = prediction_intervals
        self.only_conformal_intervals = True

    def fit(self, y: jnp.ndarray, X: Optional[jnp.ndarray] = None):
        r"""Fit the WindowAverage model.

        Fit an WindowAverage to a time series (numpy array) `y`
        and optionally exogenous variables (numpy array) `X`.

        Args:
            y (jnp.ndarray): Clean time series of shape (t, ).
            X (array-like): Optional exogenous of shape (t, n_x).

        Returns:
            self: WindowAverage fitted model.
        """

        y = _ensure_float(y) 
        mod = _window_average(y=y, h=1, window_size=self.window_size, fitted=False) 
        self.model_ = dict(mod) 
        self._store_cs(y=y, X=X) #store past conformity scores
        return self
    

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ):
        r"""Predict with fitted WindowAverage.

        Args:
            h (int): Forecast horizon.
            X (jnp.ndarray): Optional exogenous of shape (h, n_x).
            level (List[float]): Confidence levels (0-100) for prediction intervals.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        mean = _repeat_val(self.model_["mean"][0], h=h) 
        res = {"mean": mean}
        if level is None:
            return res
        level = sorted(level)
        if self.prediction_intervals is not None:
            res = self._add_predict_conformal_intervals(res, level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res

    
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ):
        r"""Memory Efficient WindowAverage predictions.

        This method avoids memory burden due from object storage.
        It is analogous to `fit_predict` without storing information.
        It assumes you know the forecast horizon in advance.

        Args:
            y (jnp.ndarray): Clean time series of shape (n, ).
            h (int): Forecast horizon.
            X (Optional[jnp.ndarray]): Optional insample exogenous of shape (t, n_x).
            X_future (Optional[jnp.ndarray]): Optional exogenous of shape (h, n_x).
            level (Optional[List[int]]): Confidence levels (0-100) for prediction intervals.
            fitted (bool): Whether or not to return insample predictions.

        Returns:
            dict: Dictionary with entries `mean` for point predictions and `level_*` for probabilistic predictions.
        """
        y = _ensure_float(y) 
        res = _window_average(y=y, h=h, fitted=fitted, window_size=self.window_size) #compute window avg
        res = dict(res) #make it a dict
        if level is None:
            return res
        level = sorted(level) 
        if self.prediction_intervals is not None: #compute conformal intervals
            res = self._add_conformal_intervals(fcst=res, y=y, X=X, level=level)
        else:
            raise Exception("You must pass `prediction_intervals` to compute them.")
        return res