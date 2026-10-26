"""
IMAPA (Intermittent Multiple Aggregation Prediction Algorithm) in JAX.

Port of statsforecast's ``IMAPA`` on top of ``BaseForecaster``,
``ConformalIntervals`` and ``utils._imapa``. The series is aggregated at
several temporal levels, simple exponential smoothing (SES) is fitted at each
level, and the back-mapped forecasts are averaged.

Differences from statsforecast:

- Array semantics, dtypes and broadcasting follow JAX rules, so values can
  differ at rounding level.
- Forecast intervals are conformal only (``only_conformal_intervals = True``).
- In-sample fitted intervals are symmetric ``z * sigma`` bands around the
  fitted values. Back-mapping can leave NaNs at early indices where the
  aggregated history is too short.
- ``predict(level=...)`` uses conformity scores cached at ``fit``; without them
  it raises ``ValueError`` instead of recomputing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

from chronax.utils import ConformalIntervals

from chronax.models.base_forecaster import BaseForecaster

from chronax.utils import _repeat_val, _window_average, ensure_float, calculate_sigma as _calculate_sigma, _add_fitted_pi, _imapa

jax.config.update("jax_enable_x64", True)

class IMAPA(BaseForecaster):
    """Intermittent Multiple Aggregation Prediction Algorithm (IMAPA).

    IMAPA targets intermittent demand series, where many observations are
    zero and demand occurs sporadically. It extends simple exponential
    smoothing (SES) in three steps:

    1. Aggregate the series at temporal levels 1, 2, ..., K to reduce
       sparsity.
    2. Fit SES independently at each aggregation level.
    3. Back-map each level's forecast to the original scale (divide by the
       aggregation factor) and average across levels.

    Point forecasts follow statsforecast's ``IMAPA``. Prediction intervals
    are conformal (distribution-free) via :class:`ConformalIntervals`.

    Parameters
    ----------
    alias : str, default ``"IMAPA"``
        Display name for the model.
    conformal_params : ConformalIntervals or None, default ``None``
        Configuration for conformal prediction intervals. When provided,
        conformity scores are cached at :meth:`fit` time so that
        :meth:`predict` can emit intervals without recomputation.

    Attributes
    ----------
    model_ : dict or None
        Dictionary produced by ``utils._imapa`` after :meth:`fit`. Contains
        the key ``"mean"`` (the flat forecast, length 1).
    only_conformal_intervals : bool
        Always ``True``: this model only supports conformal intervals.

    References
    ----------
    Syntetos, A. A. & Boylan, J. E. (2021). *Intermittent Demand Forecasting:
    Context, Methods and Applications.* John Wiley & Sons.

    Examples
    --------
    >>> model = IMAPA()
    >>> model.fit(y_train)
    >>> model.predict(h=6)
    {'mean': Array([...], dtype=float64)}
    """

    def __init__(
        self,
        alias: str = "IMAPA",
        conformal_params: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize the IMAPA estimator configuration."""
        self.alias: str = alias
        self.conformal_params: Optional[ConformalIntervals] = conformal_params
        self.only_conformal_intervals: bool = True
        self._cs: Optional[jnp.ndarray] = None
        self.model_: Optional[Dict[str, jnp.ndarray]] = None

    def conformity_scores(self, y, X=None):
        """Sequential-window cross-validation (CV), overriding the base vmapped path.

        IMAPA's fit is golden-section and while-loop heavy. Under the base CV
        vmap the traced-bound aggregation ``fori_loop`` batches its whole carry
        and the ``lax.cond`` calls lower to ``select``, so both complete
        searches execute per level per window. Windows, masking and values are
        identical; only the execution regime changes (base
        ``_conformity_scores_sequential``).

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``. Ignored by the model.

        Returns
        -------
        jnp.ndarray
            Signed scores, shape ``(n_windows, h)``.
        """
        return self._conformity_scores_sequential(y=y, X=X)

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "IMAPA":
        """Fit IMAPA to a univariate time series.

        Aggregates ``y`` at multiple temporal levels, fits SES at each level,
        and stores the resulting model state in :attr:`model_`.  If
        ``conformal_params`` was provided at construction, conformity scores
        are also computed and cached so that :meth:`predict` can emit
        intervals without re-fitting.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``. Ignored; only forwarded
            to :meth:`conformity_scores`.

        Returns
        -------
        IMAPA
            ``self``.

        Raises
        ------
        ValueError
            If ``conformal_params`` is set and the series is too short for 2
            CV windows.
        """
        y = ensure_float(y)
        self.model_ = _imapa(y=y, h=1, fitted=False)
        self._y = y

        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(
        self,
        h: int,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forecast ``h`` steps ahead from the fitted IMAPA state.

        The forecast is flat: every step equals the scalar stored in
        ``model_["mean"]``.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``. Ignored.
        level : list of int or None, default None
            Confidence levels in (0, 100) for conformal prediction intervals.
            Requires ``conformal_params`` and a prior :meth:`fit`.

        Returns
        -------
        dict
            ``"mean"`` of shape ``(h,)``, plus ``"lo-{lv}"`` and ``"hi-{lv}"``
            per level when ``level`` is given.

        Raises
        ------
        ValueError
            If called before :meth:`fit`, if ``level`` is given without
            ``conformal_params``, or if no conformity scores are cached.
        """
        if self.model_ is None:
            raise ValueError("Call fit(...) before predict().")

        mean = _repeat_val(val=self.model_["mean"][0], h=h)
        res = {"mean": mean}

        if level is None:
            return res
        level = sorted(level)

        if self.conformal_params is None:
            raise ValueError("You must pass `conformal_params` to compute intervals.")

        # Missing or empty cache: raise instead of recomputing implicitly.
        if self._cs is None or (hasattr(self._cs, "size") and self._cs.size == 0):
            raise ValueError(
                "Conformity scores are not available. Fit the model first (fit(...)) "
                "with `conformal_params` set so predict() can use cached scores."
            )

        return self.add_confidence_intervals(
            fcst=res,
            cs=self._cs,
            level=list(level),
            method=self.conformal_params.method,
        )


    def predict_in_sample(
        self,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Return in-sample fitted values and optional fitted intervals.

        Re-runs the IMAPA aggregation and back-mapping with ``fitted=True`` to
        produce one-step-ahead fitted values for the training series. Early
        indices can be NaN where the aggregated history is too short.

        Parameters
        ----------
        level : list of int or None, default None
            Confidence levels in (0, 100). When provided, symmetric
            ``z * sigma`` bands are added around the fitted values.

        Returns
        -------
        dict
            ``"fitted"`` of shape ``(T,)``, plus ``"fitted-lo-{lv}"`` and
            ``"fitted-hi-{lv}"`` per level when ``level`` is given.
        """
        fitted = _imapa(y=self._y, h=1, fitted=True)["fitted"]
        res = {"fitted": fitted}
        if level is not None:
            sigma = _calculate_sigma(self._y - fitted, self._y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Stateless fit-and-predict in a single call.

        Equivalent to :meth:`fit` followed by :meth:`predict`, with no state
        stored on the instance.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            In-sample exogenous regressors, shape ``(T, F)``. Ignored.
        X_future : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``. Ignored.
        level : list of int or None, default None
            Confidence levels in (0, 100) for conformal prediction intervals.
            Requires ``conformal_params``.
        fitted : bool, default False
            If ``True``, also return ``"fitted"`` (one-step-ahead in-sample
            predictions, shape ``(T,)``).

        Returns
        -------
        dict
            ``"mean"`` of shape ``(h,)``; ``"fitted"`` when ``fitted``;
            ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given,
            plus ``"fitted-lo-{lv}"``/``"fitted-hi-{lv}"`` when both are set.

        Raises
        ------
        Exception
            If ``level`` is given but ``conformal_params`` is ``None``.
        ValueError
            If ``level`` is given and the series is too short for 2 CV
            windows.
        """
        y = ensure_float(y)
        res = _imapa(y=y, h=h, fitted=fitted)
        if level is None:
            return res
        level = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=None)
            res = self.add_confidence_intervals(res, cs, level, self.conformal_params.method)
        else:
            raise Exception(
                "You have to instantiate the class with `conformal_params`"
                "to calculate them"
            )
        if fitted:
            sigma = _calculate_sigma(y - res["fitted"], y.size)
            res = _add_fitted_pi(res=res, se=sigma, level=level)
        return res
