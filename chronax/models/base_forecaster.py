"""
BaseForecaster defines the shared interface and common infrastructure for all models in Chronax.

Instance Attributes
-------------------
1. ``alias`` -- model name, declared in model's ``__init__``
2. ``conformal_params`` -- a conformal_intervals object, in statsforecast
   previously named prediction_intervals
3. ``model_`` -- stores fitted model post-training

Class Attributes
----------------
1. ``uses_exog`` -- boolean representing model's exogenous variable handling

Methods
-------
1. ``new()`` returns a shallow copy of the object, used internally to clone
   a model without mutating state.

2. ``__repr__`` returns the model's alias for easy identification.

3. ``fit(y, X=None)`` [abstractmethod]
   Must be implemented by every subclass. Fits the model to univariate time
   series y, sets ``self.model_``, and returns self. X is optional exogenous input.

4. ``predict(h, X=None, level=None)`` [abstractmethod]
   Must be implemented by every subclass. Generates h-step-ahead forecasts,
   returning a dict with at least {"mean": jnp.ndarray}. Optionally adds
   confidence intervals when level is provided.

5. ``forecast(y, h, X=None, X_future=None, level=None, fitted=False)`` [abstractmethod]
   Must be implemented by every subclass. Stateless fit+predict on y,
   forecasting h steps ahead. Implementations differ significantly across
   models (fitted values, model-specific kwargs, etc.). Subclasses may extend
   the signature with additional optional parameters.

6. ``forward(y, h, X=None, X_future=None, level=None, fitted=False)`` [concrete, overridable]
   Updates the model on new data y and forecasts h steps ahead. Default
   delegates to forecast(). Subclasses with warm-start re-estimation
   (e.g. Holt, HoltWinters, exponential smoothing (ETS)) override this.

7. ``conformity_scores(y, X=None)`` computes signed conformity scores on y
   as a 2D JAX array of shape (n_windows, h). Each score is (actual - forecast)
   for a walk-forward cross-validation window. Positive = underprediction.
   Uses vmap for parallelization over windows.

8. ``add_confidence_intervals(fcst, cs, level, method)`` [staticmethod]
   Adds conformal prediction intervals to a forecast dict using pre-computed
   signed conformity scores. Two methods are available:
   - ``conformal_distribution``: symmetric intervals via mean +/- abs(scores) (2W paths)
   - ``conformal_signed``: asymmetric intervals via mean + scores (W paths)

Notes
-----
- Exogenous variable support is model-specific, not framework-level.
  The boolean uses_exog must be overridden in the model's implementation.

- Signature quirks in subclasses: Naive and RandomWalkWithDrift take
  ``(h, y, ...)`` in that order in ``forecast``/``forward``, and
  ``RandomWalkWithDrift.predict`` has no ``X`` parameter. Both survive because
  ``conformity_scores`` calls ``forecast`` with keyword arguments.
"""
import jax
import jax.numpy as jnp
from chronax import utils
from abc import ABC, abstractmethod
from jax import lax, vmap

from chronax.utils import _get_conformal_method

class BaseForecaster(ABC):
    """Abstract base class defining the shared interface for all Chronax forecasting models.

    All models must implement ``fit``, ``predict``, and ``forecast``.
    See module docstring for full attribute and method documentation.
    """

    uses_exog = False

    def new(self) -> "BaseForecaster":
        """Return a shallow copy of this model instance.

        Used internally to clone a model without mutating shared state.

        Returns
        -------
        BaseForecaster
            A new instance of the same type with a shallow-copied ``__dict__``.
        """
        b = type(self).__new__(type(self))
        b.__dict__.update(self.__dict__)
        return b

    def __repr__(self) -> str:
        """Return the model's alias as its string representation."""
        return self.alias

    @staticmethod
    def _resolve_season_length(
        season_length: "int | float | str",
        y: jnp.ndarray,
        *,
        strict: bool = True,
    ) -> int:
        """Resolve a model's ``season_length`` to a concrete Python int.

        The single canonical entry point any model references to support
        ``season_length="auto"``. An explicit int is used verbatim
        (statsforecast parity: a deliberate period is domain knowledge and is
        never overridden). ``"auto"`` delegates to
        :func:`chronax.utils.detect_period` (``strict=True`` means
        periodogram + at least 5 cycles + folded-strength + divisor check).

        Eager-only contract: this materialises a Python int (the period sizes
        static seasonal-state shapes, so it must be static for jit). Call it
        once at ``fit()`` on concrete ``y`` and cache the result
        (``self._m_eff``); never invoke it inside a jit/vmap trace. The
        cached int is what flows to jitted kernels and the vmap forecast
        path.

        Parameters
        ----------
        season_length : int | float | "auto"
            Configured period, or the literal ``"auto"`` to infer.
        y : jnp.ndarray
            Training series (concrete; used only when ``"auto"``).
        strict : bool, default True
            Passed to :func:`detect_period`. Canonical "auto" uses strict.

        Returns
        -------
        int
            Period ≥ 1 (1 = non-seasonal).
        """
        if isinstance(season_length, str):
            if season_length != "auto":
                raise ValueError(
                    f'season_length string must be "auto", got {season_length!r}'
                )
            from chronax.utils import detect_period

            # Cap the search range by the series length so long-period series
            # (e.g. 5-min data with a daily m=288) are reachable on the auto
            # path, while short series keep the default 60 ceiling. Strict
            # detection needs >=5 cycles, so n//5 is the natural upper bound;
            # 400 bounds the ACF grid on very long series.
            n = int(jnp.asarray(y).shape[0])
            max_period = min(max(60, n // 5), 400)
            m = int(detect_period(
                jnp.asarray(y, dtype=jnp.float64), strict=strict, max_period=max_period))
            return m if m >= 1 else 1
        if not isinstance(season_length, (int, float)) or int(season_length) < 1:
            raise ValueError(
                f'season_length must be an int >= 1 or "auto", got {season_length!r}'
            )
        return int(season_length)

    @abstractmethod
    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "BaseForecaster":
        """Fit the model to a univariate time series.

        Implementations must set ``self.model_`` and return ``self``.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``. Ignored by models with
            ``uses_exog=False``.

        Returns
        -------
        BaseForecaster
            ``self``.
        """

    def _require_fitted(self) -> None:
        """Raise an actionable ValueError if ``predict`` is called before ``fit``.

        The unfitted sentinel is ``model_`` being ``None`` or an empty container;
        without this, unfitted ``predict`` calls surface as opaque
        ``TypeError: 'NoneType' object is not subscriptable`` or ``KeyError``.
        """
        m = getattr(self, "model_", None)
        if m is None or (isinstance(m, (dict, list, tuple)) and len(m) == 0):
            raise ValueError(
                f"{type(self).__name__} is not fitted; call fit(y) before predict()."
            )

    @abstractmethod
    def predict(self, h: int, X: jnp.ndarray | None = None, level: list[int | float] | None = None) -> dict:
        """Generate h-step-ahead forecasts from the fitted state.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``.
        level : list of int or float, or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.

        Returns
        -------
        dict
            At least ``"mean"`` (shape ``(h,)``); ``"lo-{lv}"``/``"hi-{lv}"``
            per level when ``level`` is given.
        """

    @abstractmethod
    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """Stateless fit+predict on ``y``, forecasting ``h`` steps ahead.

        ``conformity_scores`` vmaps this method over cross-validation (CV) windows with
        ``level=None``, so everything reachable from it at ``level=None`` must
        trace under ``jax.vmap``. Subclasses may extend the signature with
        model-specific optional parameters.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            In-sample exogenous regressors, shape ``(T, F)``.
        X_future : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``.
        level : list of int or float, or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.
        fitted : bool, default False
            Whether to also return in-sample fitted values under ``"fitted"``.

        Returns
        -------
        dict
            At least ``"mean"`` (shape ``(h,)``).
        """

    def forward(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """Apply the model to a new series ``y`` and forecast ``h`` steps ahead.

        The default delegates to :meth:`forecast`. Subclasses with warm-start
        behaviour (Holt, HoltWinters, ETS) override this.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        h : int
            Forecast horizon.
        X : jnp.ndarray or None, default None
            In-sample exogenous regressors, shape ``(T, F)``.
        X_future : jnp.ndarray or None, default None
            Future exogenous regressors, shape ``(h, F)``.
        level : list of int or float, or None, default None
            Confidence levels in (0, 100), e.g. ``[80, 95]``.
        fitted : bool, default False
            Whether to also return in-sample fitted values under ``"fitted"``.

        Returns
        -------
        dict
            Whatever :meth:`forecast` returns.
        """
        return self.forecast(y=y, h=h, X=X, X_future=X_future, level=level, fitted=fitted)

    def conformity_scores(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute signed conformity scores via walk-forward cross-validation.

        Scores are signed residuals (actual - forecast) across ``n_windows``
        expanding windows of horizon ``h``. Uses ``vmap`` for parallelization.
        The interval construction method determines how scores are used:
        ``conformal_distribution`` takes their absolute value for symmetric
        intervals; ``conformal_signed`` uses them directly for asymmetric intervals.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate time series of observed values.
        X : jnp.ndarray or None, optional
            Exogenous features array, shape ``(n_samples, n_features)``. Only
            used by models where ``uses_exog=True``.

        Returns
        -------
        jnp.ndarray
            2-D array of shape ``(n_windows, h)`` containing signed forecast
            errors for each window and horizon step.

        Raises
        ------
        ValueError
            If ``conformal_params`` is ``None``.
        ValueError
            If the series is too short to form at least 2 windows.
        """
        if self.conformal_params is None:
            raise ValueError(
                "The instance attribute conformal_params must be initialized as a conformal_intervals object."
            )
        n_windows = self.conformal_params.n_windows
        h = self.conformal_params.h
        y = utils.ensure_float(y)
        n_samples = y.size
        # use as many windows as possible for short series
        # subtract 1 for the training set
        n_windows = min(n_windows, (n_samples - 1) // h)
        if n_windows < 2:
            raise ValueError(
                f"Conformal prediction requires at least {2 * h + 1:,} samples per window; series has {n_samples:,}."
            )
        test_size = n_windows * h
        base_train_end = n_samples - test_size
        max_train_size = base_train_end + (n_windows - 1) * h

        # Pad arrays to maximum training size for fixed-size operations
        y_padded = jnp.pad(y, (0, max(0, max_train_size - n_samples)), mode='edge')
        if X is not None:
            X_padded = jnp.pad(X, ((0, max(0, max_train_size - n_samples)), (0, 0)), mode='edge')
        else:
            X_padded = None

        def compute_window_scores(i_window):
            train_end = base_train_end + i_window * h

            # Slice with STATIC maximum size (JAX-compatible), then mask
            # data beyond train_end with edge value to prevent data leakage
            y_train = lax.dynamic_slice(y_padded, (0,), (max_train_size,))
            mask = jnp.arange(max_train_size) < train_end
            y_train = jnp.where(mask, y_train, y_train[train_end - 1])

            y_test = lax.dynamic_slice(y_padded, (train_end,), (h,))

            if X_padded is not None:
                X_train = lax.dynamic_slice(X_padded, (0, 0), (max_train_size, X_padded.shape[1]))
                X_train = jnp.where(mask[:, None], X_train, X_train[train_end - 1])
                X_test = lax.dynamic_slice(X_padded, (train_end, 0), (h, X_padded.shape[1]))
            else:
                X_train = None
                X_test = None

            fcst_window = self.forecast(h=h, y=y_train, X=X_train, X_future=X_test)  # type: ignore[attr-defined]
            window_scores = y_test - fcst_window['mean'].astype(y_test.dtype)
            return window_scores

        # Use vmap for parallel processing across windows
        cs = vmap(compute_window_scores)(jnp.arange(n_windows))
        return cs

    def _conformity_scores_sequential(self, y, X=None):
        """Sequential-window variant of :meth:`conformity_scores`.

        Identical windows, masking, and per-window computation; only the
        execution regime differs (a Python loop with concrete indices instead
        of the CV ``vmap``).

        For models whose ``fit`` contains while-loops or data-dependent
        ``lax.cond`` calls with scalar-sized bodies (golden-section searches,
        jaxopt/optax line searches), the CV vmap is an anti-optimization:
        batched predicates batch the whole carry, conds lower to ``select``
        and execute both branches per window, and tight loops degrade to
        gather/select code. Models opt in by overriding ``conformity_scores``
        to delegate here; pure-scan fits (the ETS family) amortize the vmap
        well and should not.

        The body deliberately mirrors :meth:`conformity_scores` line for line
        rather than sharing a refactored core; keep the two in sync when
        editing either.

        Parameters
        ----------
        y : jnp.ndarray
            Univariate series, shape ``(T,)``.
        X : jnp.ndarray or None, default None
            Exogenous regressors, shape ``(T, F)``.

        Returns
        -------
        jnp.ndarray
            Signed scores, shape ``(n_windows, h)``.

        Raises
        ------
        ValueError
            If ``conformal_params`` is ``None`` or the series is too short
            for at least 2 windows.
        """
        if self.conformal_params is None:
            raise ValueError(
                "The instance attribute conformal_params must be initialized as a conformal_intervals object."
            )
        n_windows = self.conformal_params.n_windows
        h = self.conformal_params.h
        y = utils.ensure_float(y)
        n_samples = y.size
        n_windows = min(n_windows, (n_samples - 1) // h)
        if n_windows < 2:
            raise ValueError(
                f"Conformal prediction requires at least {2 * h + 1:,} samples per window; series has {n_samples:,}."
            )
        test_size = n_windows * h
        base_train_end = n_samples - test_size
        max_train_size = base_train_end + (n_windows - 1) * h

        y_padded = jnp.pad(y, (0, max(0, max_train_size - n_samples)), mode='edge')
        if X is not None:
            X_padded = jnp.pad(X, ((0, max(0, max_train_size - n_samples)), (0, 0)), mode='edge')
        else:
            X_padded = None

        scores = []
        for i_window in range(n_windows):
            train_end = base_train_end + i_window * h
            y_train = lax.dynamic_slice(y_padded, (0,), (max_train_size,))
            mask = jnp.arange(max_train_size) < train_end
            y_train = jnp.where(mask, y_train, y_train[train_end - 1])
            y_test = lax.dynamic_slice(y_padded, (train_end,), (h,))
            if X_padded is not None:
                X_train = lax.dynamic_slice(X_padded, (0, 0), (max_train_size, X_padded.shape[1]))
                X_train = jnp.where(mask[:, None], X_train, X_train[train_end - 1])
                X_test = lax.dynamic_slice(X_padded, (train_end, 0), (h, X_padded.shape[1]))
            else:
                X_train = None
                X_test = None
            fcst_window = self.forecast(h=h, y=y_train, X=X_train, X_future=X_test)  # type: ignore[attr-defined]
            scores.append(y_test - fcst_window['mean'].astype(y_test.dtype))
        return jnp.stack(scores)


    @staticmethod
    def add_confidence_intervals(
        fcst: dict,
        cs: jnp.ndarray,
        level: list[int | float],
        method: str,
    ) -> dict:
        """Add conformal prediction intervals to a forecast dict.

        Mutates and returns ``fcst`` with interval columns added in-place,
        keyed as ``"lo-{level}"`` and ``"hi-{level}"`` for each requested level.

        Parameters
        ----------
        fcst : dict
            Forecast dictionary containing at least ``{"mean": jnp.ndarray}``.
        cs : jnp.ndarray
            Signed conformity scores, shape ``(n_windows, h)``.
        level : list of int or float
            Confidence levels, e.g. ``[80, 95]``.
        method : str
            ``"conformal_distribution"`` (symmetric, uses ``|scores|``) or
            ``"conformal_signed"`` (asymmetric, uses raw signed scores).

        Returns
        -------
        dict
            The input ``fcst`` dict with interval arrays added.

        Raises
        ------
        ValueError
            If ``method`` is not a recognised conformal method.
        """
        def conformal_distribution_intervals(fcst, cs, level):
            level = [int(lv) if float(lv).is_integer() else lv
                     for lv in sorted(level)]
            alphas = jnp.array([100 - lv for lv in level])
            cuts_lower = alphas / 200.0
            cuts_upper = 1 - alphas / 200.0
            cuts_lower = cuts_lower[::-1]
            cuts = jnp.concatenate([cuts_lower, cuts_upper])
            mean = fcst["mean"]
            cs_abs = jnp.abs(cs)
            scores = jnp.concatenate([
                mean[None, :] - cs_abs,
                mean[None, :] + cs_abs,
            ], axis=0)
            quantiles = jnp.quantile(scores, cuts, axis=0)
            lo_cols = [f"lo-{lv}" for lv in reversed(level)]
            hi_cols = [f"hi-{lv}" for lv in level]
            out_cols = lo_cols + hi_cols
            for i, col in enumerate(out_cols):
                fcst[col] = quantiles[i]
            return fcst

        def conformal_signed_intervals(fcst, cs, level):
            level = [int(lv) if float(lv).is_integer() else lv
                     for lv in sorted(level)]
            alphas = jnp.array([100 - lv for lv in level])
            cuts_lower = (alphas / 200.0)[::-1]
            cuts_upper = 1 - alphas / 200.0
            cuts = jnp.concatenate([cuts_lower, cuts_upper])
            mean = fcst["mean"]
            scores = mean[None, :] + cs
            quantiles = jnp.quantile(scores, cuts, axis=0)
            lo_cols = [f"lo-{lv}" for lv in reversed(level)]
            hi_cols = [f"hi-{lv}" for lv in level]
            out_cols = lo_cols + hi_cols
            for i, col in enumerate(out_cols):
                fcst[col] = quantiles[i]
            return fcst

        allowed_methods = {
            "conformal_distribution": conformal_distribution_intervals,
            "conformal_signed": conformal_signed_intervals,
        }
        if method not in allowed_methods:
            raise ValueError(f"{method} is not valid. Choose from {str(allowed_methods)[1:-1]}")
        
        return allowed_methods[method](fcst, cs, level)
