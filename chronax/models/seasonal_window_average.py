"""
SeasonalWindowAverage: average of the last ``window_size`` seasonal cycles.

The JIT-compiled ``_seasonal_window_average`` takes the last
``window_size * season_length`` observations, reshapes them to
``(window_size, season_length)`` and averages across cycles; the forecast
tiles that pattern. Intervals are conformal only. Fitted values are not
supported (``predict_in_sample`` and ``forecast(fitted=True)`` raise
``NotImplementedError``). A series shorter than
``season_length * window_size`` yields a NaN forecast.
"""

import jax
import jax.numpy as jnp
from jax import lax

from chronax import utils
from chronax.models.base_forecaster import BaseForecaster
from chronax.utils import ConformalIntervals


# ---------- Core: Single jitted function ----------

from functools import partial

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _seasonal_window_average(
    y: jnp.ndarray, 
    h: int, 
    fitted: bool, 
    season_length: int, 
    window_size: int
) -> dict:
    """Core SeasonalWindowAverage computation (pure, jitted).

    Averages the last ``window_size`` complete seasonal cycles position by
    position and tiles the result over the horizon.

    Args:
        y (jnp.ndarray): Univariate series, shape ``(T,)``.
        h (int): Forecast horizon (static).
        fitted (bool): Must be ``False``; fitted values are not supported
            (static).
        season_length (int): Observations per seasonal cycle (static).
        window_size (int): Number of most recent cycles to average (static).

    Returns:
        dict: ``"mean"`` of shape ``(h,)``; NaN when
        ``T < season_length * window_size``.

    Raises:
        NotImplementedError: If ``fitted`` is ``True``.
    """
    if fitted:
        raise NotImplementedError("fitted values not supported for SeasonalWindowAverage")

    min_samples = season_length * window_size  # static

    # Pad with min_samples NaNs so the trailing slice always has the static
    # length min_samples that the reshape below needs.
    y_padded = jnp.concatenate([jnp.full(min_samples, jnp.nan, dtype=y.dtype), y])

    y_window = y_padded[-min_samples:]

    def sufficient_data(_):
        """Average the (window_size, season_length) matrix across cycles and tile."""
        season_avgs = y_window.reshape(window_size, season_length).mean(axis=0)

        return utils._repeat_val_seas(season_avgs, h)

    def insufficient_data(_):
        """Return NaN when the series is too short."""
        # lax.cond needs identical shape AND dtype from both branches; the
        # sufficient branch inherits y's dtype.
        return jnp.full(h, jnp.nan, dtype=y.dtype)

    # y.size is static, but lax.cond still traces both branches.
    mean = lax.cond(
        y.size >= min_samples,
        sufficient_data,
        insufficient_data,
        None
    )
    
    return {"mean": mean}


# ---------- Model class ----------

class SeasonalWindowAverage(BaseForecaster):
    """SeasonalWindowAverage model.

    Forecasts each seasonal position as the average of its last
    ``window_size`` values. Intervals are conformal only. Fitted values are
    not supported: ``predict_in_sample`` and ``forecast(fitted=True)`` raise
    ``NotImplementedError``. A series shorter than
    ``season_length * window_size`` yields a NaN forecast.

    Args:
        season_length (int): Observations per seasonal cycle, e.g. 24 for
            hourly data with a daily cycle.
        window_size (int): Number of most recent cycles to average, e.g. 7
            to average the same hour over the last 7 days.
        alias (str): Custom name of the model. Defaults to ``"SeasWA"``.
        conformal_params (ConformalIntervals | None): Configuration for
            conformal prediction intervals; required for ``level``. Defaults
            to ``None``.

    Example:
        >>> from chronax.utils import ConformalIntervals
        >>> ci = ConformalIntervals(h=24, n_windows=10)
        >>> model = SeasonalWindowAverage(season_length=24, window_size=7, conformal_params=ci)
        >>> model.fit(y_train)
        >>> forecasts = model.predict(h=24, level=[80, 95])
    """

    uses_exog = False

    def __init__(
        self,
        season_length: int,
        window_size: int,
        alias: str = "SeasWA",
        conformal_params: ConformalIntervals | None = None,
    ) -> None:
        self.season_length = season_length
        self.window_size = window_size
        self.alias = alias
        self.conformal_params = conformal_params
        self.only_conformal_intervals = True
        self.model_ = {}

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "SeasonalWindowAverage":
        """Fit the SeasonalWindowAverage model.

        Stores one averaged seasonal cycle in ``model_``. When
        ``conformal_params`` is set, conformity scores are computed on the
        training series and cached for :meth:`predict`.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            X (jnp.ndarray | None): Exogenous regressors, shape ``(T, F)``.
                Only forwarded to ``conformity_scores``. Defaults to ``None``.

        Returns:
            SeasonalWindowAverage: ``self``.

        Raises:
            ValueError: If ``conformal_params`` is set and the series is too
                short for 2 cross-validation (CV) windows.
        """
        y = utils.ensure_float(y)

        mod = _seasonal_window_average(
            y,
            h=self.season_length,  # one full seasonal cycle
            fitted=False,
            season_length=self.season_length,
            window_size=self.window_size
        )
        self.model_ = mod
        # Score the training series, not the stored one-cycle pattern.
        if self.conformal_params is not None:
            self._cs = self.conformity_scores(y=y, X=X)
        else:
            self._cs = None
        return self

    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int] | None = None) -> dict:
        """Predict with the fitted SeasonalWindowAverage model.

        Args:
            h (int): Forecast horizon. With ``level``, must equal
                ``conformal_params.h``.
            X (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100), e.g.
                ``[80, 95]``. Requires ``conformal_params``. Defaults to
                ``None``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``) plus ``"lo-{lv}"``/``"hi-{lv}"``
            per level when ``level`` is given.

        Raises:
            ValueError: If called before ``fit``; if no conformity scores are
                cached; or if ``h != conformal_params.h``.
            Exception: If ``level`` is given without ``conformal_params``.
        """
        self._require_fitted()
        mean = utils._repeat_val_seas(self.model_["mean"], h)
        res = {"mean": mean}

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if getattr(self, "_cs", None) is None:
            raise ValueError(
                "Conformity scores are not available. Fit the model first (fit(...)) "
                "with `conformal_params` set so predict() can use cached scores."
            )
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )

        res = self.add_confidence_intervals(
            res,
            self._cs,
            level,
            self.conformal_params.method,
        )

        return res

    def predict_in_sample(self, level: list[int] | None = None) -> dict:
        """Not supported: SeasonalWindowAverage has no fitted values.

        Args:
            level (list[int] | None): Unused. Defaults to ``None``.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("predict_in_sample not supported for SeasonalWindowAverage")

    def forecast(
        self, 
        y: jnp.ndarray, 
        h: int, 
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None, 
        level: list[int] | None = None, 
        fitted: bool = False
    ) -> dict:
        """Stateless fit+predict in one call.

        Args:
            y (jnp.ndarray): Univariate series, shape ``(T,)``.
            h (int): Forecast horizon. With ``level``, must equal
                ``conformal_params.h``.
            X (jnp.ndarray | None): In-sample exogenous regressors, shape
                ``(T, F)``. Ignored. Defaults to ``None``.
            X_future (jnp.ndarray | None): Future exogenous regressors, shape
                ``(h, F)``. Ignored. Defaults to ``None``.
            level (list[int] | None): Confidence levels in (0, 100). Requires
                ``conformal_params``. Defaults to ``None``.
            fitted (bool): Must be ``False``; fitted values are not
                supported. Defaults to ``False``.

        Returns:
            dict: ``"mean"`` (shape ``(h,)``, NaN when
            ``T < season_length * window_size``) plus
            ``"lo-{lv}"``/``"hi-{lv}"`` per level when ``level`` is given.

        Raises:
            NotImplementedError: If ``fitted`` is ``True``.
            Exception: If ``level`` is given without ``conformal_params``.
            ValueError: If ``h != conformal_params.h``, or if the series is
                too short for 2 CV windows.
        """
        y = utils.ensure_float(y)

        res = dict(_seasonal_window_average(y, h, fitted, self.season_length, self.window_size))

        if level is None:
            return res

        level = sorted(level)
        if self.conformal_params is None:
            raise Exception("You must pass `conformal_params` to compute them.")
        if h != self.conformal_params.h:
            raise ValueError(
                f"h={h} does not match conformal_params.h={self.conformal_params.h}; "
                "conformity scores cover exactly conformal_params.h steps."
            )

        res = self.add_confidence_intervals(
            res,
            self.conformity_scores(y, X=None),
            level,
            self.conformal_params.method,
        )

        return res
