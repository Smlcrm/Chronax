# ets_model.py
"""Fixed-specification ETS (Error, Trend, Seasonality) forecaster.

This module provides the :class:`ETS` class — a user-facing wrapper around
the core ``ets_functions`` engine.  Unlike :class:`AutoETS` (which evaluates
a grid of model structures), :class:`ETS` fits **exactly one** user-specified
model string (e.g. ``"ANN"``, ``"AAN"``, ``"AAA"``).

The API surface is identical to :class:`AutoETS`: ``fit``, ``predict``,
``predict_in_sample``, ``forecast``, and ``forward``.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import os

import jax.numpy as jnp

from chronax.utils import ConformalIntervals, ensure_float, _add_fitted_pi, calculate_sigma
from chronax.models.base_forecaster import BaseForecaster
from .ets_functions import ets_f, forecast_ets, forward_ets

_PHI_LOWER: float = 0.8
_PHI_UPPER: float = 0.98


def _init_jax_compilation_cache() -> None:
    """Set up a local JAX compilation cache if one is not already configured."""
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(__file__), ".jax_cache")
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir


_init_jax_compilation_cache()


class ETS(BaseForecaster):
    """Fixed-specification Exponential Smoothing (ETS) forecaster.

    ETS decomposes a time series into **level**, **trend**, and **seasonal**
    components whose states are updated at each time step via smoothing
    parameters (α, β, γ, ϕ).  The three-character model string specifies
    **(Error, Trend, Season)**:

    +-----------+----------------+----------------+------------------+
    | Character | Error          | Trend          | Season           |
    +===========+================+================+==================+
    | ``A``     | Additive       | Additive       | Additive         |
    +-----------+----------------+----------------+------------------+
    | ``M``     | Multiplicative | Multiplicative | Multiplicative   |
    +-----------+----------------+----------------+------------------+
    | ``N``     | —              | No trend       | No seasonality   |
    +-----------+----------------+----------------+------------------+

    For example, ``"AAN"`` = Additive error + Additive trend + No seasonality.

    Parameters
    ----------
    season_length : int, default ``1``
        Seasonal period (e.g. ``12`` for monthly, ``4`` for quarterly).
        Use ``1`` for non-seasonal models.
    model : str, default ``"ANN"``
        Fixed ETS specification string.  Common choices:

        * ``"ANN"`` — simple exponential smoothing
        * ``"AAN"`` — Holt's linear trend
        * ``"AAA"`` — additive trend + additive seasonality
    damped : bool or None, default ``None``
        Whether to apply trend damping.  ``None`` is treated as ``False``.
    phi : float or None, default ``None``
        Damping coefficient.  Must be in ``[0.8, 0.98]`` when provided.
    max_iter : int or None, default ``None``
        Number of ``optax`` gradient-descent iterations.  ``None`` lets the
        engine choose a sensible default based on data length and model
        complexity.
    optax_lr : float, default ``1e-2``
        Learning rate for the ``optax`` Adam optimiser.
    optax_clip : float, default ``1.0``
        Gradient clipping threshold.
    alias : str, default ``"ETS"``
        Display name for the model.
    prediction_intervals : ConformalIntervals or None, default ``None``
        Configuration for conformal prediction intervals.  When provided,
        conformity scores are cached at :meth:`fit` time.

    Attributes
    ----------
    model_ : dict
        Internal state dictionary produced by ``ets_f(…)`` after
        :meth:`fit`.  Contains fitted parameters, AICc, fitted values,
        residuals, etc.

    Examples
    --------
    >>> ets = ETS(season_length=1, model="AAN", max_iter=200)
    >>> ets.fit(y_train)
    >>> ets.predict(h=6, level=[80, 95])
    {'mean': Array([...]), 'lo-80': ..., 'hi-80': ..., ...}
    """

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ANN",
        damped: Optional[bool] = None,
        phi: Optional[float] = None,
        max_iter: Optional[int] = None,
        optax_lr: float = 1e-2,
        optax_clip: float = 1.0,
        alias: str = "ETS",
        prediction_intervals: Optional[ConformalIntervals] = None,
    ) -> None:
        """Initialize a fixed-spec ETS estimator."""
        self.season_length: int = season_length
        self.model: str = model
        if damped is None:
            damped = False
        self.damped: bool = damped
        if phi is not None:
            if not isinstance(phi, float):
                raise ValueError("phi must be `None` or float.")
            if not (_PHI_LOWER <= phi <= _PHI_UPPER):
                raise ValueError(f"Valid range for phi is [{_PHI_LOWER}, {_PHI_UPPER}]")
        self.phi: Optional[float] = phi
        self.max_iter: Optional[int] = max_iter
        self.optax_lr: float = optax_lr
        self.optax_clip: float = optax_clip
        self.alias: str = alias
        self.conformal_params: Optional[ConformalIntervals] = prediction_intervals
        self.optax_steps: Optional[int] = max_iter

    def fit(
        self,
        y: jnp.ndarray,
        X: Optional[jnp.ndarray] = None,
    ) -> "ETS":
        """Fit the ETS model to a univariate time series.

        Optimises smoothing parameters and initial states via ``optax``
        gradient descent on the likelihood, then stores the full model state
        in :attr:`model_`.

        Parameters
        ----------
        y : jnp.ndarray
            One-dimensional time series of shape ``(n,)``.
        X : jnp.ndarray or None
            Ignored — present for API compatibility with
            :class:`BaseForecaster`.

        Returns
        -------
        ETS
            ``self``, for method chaining.
        """
        y = ensure_float(y)
        self.model_ = ets_f(
            y,
            m=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            optax_steps=self.max_iter,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
        )
        self.model_["actual_residuals"] = y - self.model_["fitted"]

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
        """Generate *h*-step-ahead forecasts from the fitted model.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.
        level : list of int or None
            Confidence levels in ``[0, 100]``.  When provided and
            ``conformal_params`` is set, conformal intervals are returned;
            otherwise native Gaussian ETS intervals are used.

        Returns
        -------
        dict
            Always contains ``"mean"`` of shape ``(h,)``.  When *level* is
            given, also contains ``"lo-{level}"`` and ``"hi-{level}"`` for
            each requested level.

        Raises
        ------
        Exception
            If called before :meth:`fit`.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        fcst = forecast_ets(self.model_, h=h, level=level)
        out = {"mean": fcst["mean"]}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            if self._cs is None:
                raise ValueError(
                    "Conformity scores not cached. Fit with conformal_params set, "
                    "or use forecast(y, ...) which recomputes them."
                )
            return self.add_confidence_intervals(
                fcst=out, cs=self._cs, level=level_sorted, method=self.conformal_params.method
            )

        out.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)})
        out.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted})
        return out

    def predict_in_sample(
        self,
        level: Optional[List[int]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Return in-sample fitted values (and optional prediction intervals).

        Fitted values are one-step-ahead predictions for the training data,
        useful for computing residuals and evaluating goodness of fit.

        Parameters
        ----------
        level : list of int or None
            Confidence levels in ``[0, 100]``.  When provided, symmetric
            ±z·σ intervals are appended using the residual standard error.

        Returns
        -------
        dict
            ``{"fitted": jnp.ndarray}`` of shape ``(n,)``.  When *level* is
            given, also contains ``"fitted-lo-{level}"`` and
            ``"fitted-hi-{level}"`` keys.

        Raises
        ------
        Exception
            If called before :meth:`fit`.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        res = {"fitted": self.model_["fitted"]}
        if level is not None:
            residuals = self.model_["actual_residuals"]
            # se = _calculate_sigma(residuals, len(residuals) - self.model_["n_params"])
            se = calculate_sigma(residuals, len(residuals) - self.model_["n_params"])

            res = _add_fitted_pi(res=res, se=se, level=level)
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

        Fits the ETS model to *y* and immediately produces *h*-step-ahead
        forecasts **without** persisting any model state on the instance.
        Ideal for cross-validation loops and batch evaluation.

        Parameters
        ----------
        y : jnp.ndarray
            One-dimensional time series of shape ``(n,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.
        X_future : jnp.ndarray or None
            Ignored — present for API compatibility.
        level : list of int or None
            Confidence levels in ``[0, 100]`` for prediction intervals.
        fitted : bool, default ``False``
            If ``True``, the returned dict also includes ``"fitted"``
            (in-sample predictions of shape ``(n,)``).

        Returns
        -------
        dict
            Always contains ``"mean"`` of shape ``(h,)``.  Optionally
            includes ``"fitted"``, ``"lo-{level}"``, ``"hi-{level}"``,
            ``"fitted-lo-{level}"``, and ``"fitted-hi-{level}"``.
        """
        y = ensure_float(y)
        mod = ets_f(
            y,
            m=self.season_length,
            model=self.model,
            damped=self.damped,
            phi=self.phi,
            optax_steps=self.optax_steps,
            optax_lr=self.optax_lr,
            optax_clip=self.optax_clip,
        )
        fcst = forecast_ets(mod, h=h, level=level)

        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        out = {k: fcst[k] for k in keys}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            out = self.add_confidence_intervals(
                fcst=out, cs=cs, level=level_sorted, method=self.conformal_params.method
            )
        else:
            out = {
                **out,
                **{f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)},
                **{f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted},
            }

        if fitted:
            # se = _calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            se = calculate_sigma(y - mod["fitted"], len(y) - mod["n_params"])
            out = _add_fitted_pi(res=out, se=se, level=level_sorted)
        return out

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: Optional[jnp.ndarray] = None,
        X_future: Optional[jnp.ndarray] = None,
        level: Optional[List[int]] = None,
        fitted: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Apply the previously fitted model structure to a **new** series.

        Reuses the model specification (error/trend/season type, damping,
        etc.) learned by :meth:`fit` and re-estimates parameters on *y* via
        ``forward_ets``.  This is useful for walk-forward evaluation where
        the model structure is fixed but re-fitted on expanding windows.

        Parameters
        ----------
        y : jnp.ndarray
            New time series of shape ``(n,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None
            Ignored — present for API compatibility.
        X_future : jnp.ndarray or None
            Ignored — present for API compatibility.
        level : list of int or None
            Confidence levels in ``[0, 100]`` for prediction intervals.
        fitted : bool, default ``False``
            If ``True``, include in-sample fitted values in the output.

        Returns
        -------
        dict
            Same structure as :meth:`forecast`.

        Raises
        ------
        Exception
            If called before :meth:`fit`.
        """
        if not hasattr(self, "model_"):
            raise Exception("You have to use the `fit` method first")
        y = ensure_float(y)
        mod = forward_ets(self.model_, y=y)
        fcst = forecast_ets(mod, h=h, level=level)

        keys = ["mean"]
        if fitted:
            keys.append("fitted")
        out = {k: fcst[k] for k in keys}

        if level is None:
            return out

        level_sorted = sorted(level)
        if self.conformal_params is not None:
            cs = self.conformity_scores(y=y, X=X)
            out = self.add_confidence_intervals(
                fcst=out, cs=cs, level=level_sorted, method=self.conformal_params.method
            )
        else:
            out.update({f"lo-{l}": fcst[f"lo-{l}"] for l in reversed(level_sorted)})
            out.update({f"hi-{l}": fcst[f"hi-{l}"] for l in level_sorted})

            if fitted:
                # se = _calculate_sigma(y - mod["fitted"], len(y) - int(mod["n_params"]))
                se = calculate_sigma(y - mod["fitted"], len(y) - int(mod["n_params"]))
                out = _add_fitted_pi(res=out, se=se, level=level_sorted)

        return out
