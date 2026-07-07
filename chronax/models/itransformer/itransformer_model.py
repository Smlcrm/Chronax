"""iTransformer forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.itransformer.itransformer_losses import LossFn, resolve as _resolve_loss
from chronax.models.itransformer.itransformer_module import ITransformerNet
from chronax.models.itransformer.itransformer_training import predict_step, train
from chronax.utils import ConformalIntervals

# --- Box-Cox variance-stabilizing transform (self-contained) ---------------
# Mirrors the ``use_boxcox`` convention of chronax.models.TBATS: stabilize the
# variance of multiplicative / strongly-trending series before modelling, then
# invert on the output. ``lambda = 0`` is the log transform. The lambda is
# selected once at fit time by maximizing the Box-Cox profile log-likelihood
# (the standard MLE), kept self-contained so this module does not depend on
# another model's internals.

_BOXCOX_LAMBDA_GRID = jnp.linspace(-1.0, 2.0, 31)  # step 0.1, includes 0.0 (log)


def _boxcox(y: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Box-Cox transform of strictly-positive ``y``. ``lam=0`` -> ``log(y)``."""
    y = jnp.asarray(y, dtype=jnp.float32)
    if abs(lam) < 1e-8:
        return jnp.log(y)
    return (jnp.power(y, lam) - 1.0) / lam


def _inv_boxcox(z: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Inverse Box-Cox. Clamps the domain so ``lam != 0`` never yields NaN."""
    z = jnp.asarray(z, dtype=jnp.float32)
    if abs(lam) < 1e-8:
        return jnp.exp(z)
    base = jnp.maximum(lam * z + 1.0, 1e-8)
    return jnp.power(base, 1.0 / lam)


def _select_boxcox_lambda(y: jnp.ndarray) -> float:
    """Pick lambda by maximizing the Box-Cox profile log-likelihood (MLE).

    ``LL(lam) = -n/2 * log(var(boxcox(y, lam))) + (lam - 1) * sum(log(y))``,
    evaluated over a fixed grid. Runs eagerly (concrete ``y``), returning a
    plain ``float`` so it pickles trivially with the fitted estimator.
    """
    y = jnp.asarray(y, dtype=jnp.float32)
    n = y.size
    log_sum = jnp.sum(jnp.log(y))

    def ll(lam: float) -> jnp.ndarray:
        z = _boxcox(y, lam)
        var = jnp.maximum(jnp.var(z), 1e-12)
        return -0.5 * n * jnp.log(var) + (lam - 1.0) * log_sum

    lls = jnp.array([ll(float(lam)) for lam in _BOXCOX_LAMBDA_GRID])
    return float(_BOXCOX_LAMBDA_GRID[int(jnp.argmax(lls))])


class iTransformer(BaseForecaster):
    """Univariate iTransformer forecaster (JAX/Flax-NNX port of neuralforecast.iTransformer).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via ``predict(level=...)``,
        pickle round-trip, and ``forecast(fitted=True)``.

    The "inverted" encoder embeds the lookback window of each variate into a token
    (``Linear(input_size -> hidden_size)``) and runs a stack of transformer layers
    (full softmax attention across variate tokens + a pointwise GELU feed-forward,
    LayerNorm post-norm) before projecting to the horizon. RevIN-style per-window
    mean/std normalization (``use_norm``) is applied inside the network and inverted
    on the output. Trained in original scale with Optax ``adam`` and a pluggable point
    loss. ``float32`` throughout. The series is univariate (``n_series = 1``), so the
    encoder operates on a single token; the backbone is written N-generically to allow
    a future multivariate path.
    """

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_norm: bool = True,
        use_boxcox: bool = False,
        activation: str = "gelu",
        max_steps: int = 1000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        windows_batch_size: int = 32,
        random_seed: int = 1,
        alias: str = "iTransformer",
        loss: Union[str, LossFn] = "mae",
    ):
        """Initialize an iTransformer forecaster.

        Stores hyperparameters; the network is built lazily at ``fit`` time so
        construction is cheap and side-effect free. Defaults match
        neuralforecast.iTransformer (``hidden_size=512``, ``n_heads=8``,
        ``e_layers=2``, ``d_ff=2048``, ``dropout=0.1``, ``use_norm=True``,
        ``max_steps=1000``, ``learning_rate=1e-3``, ``windows_batch_size=32``).
        ``input_size=-1`` resolves to ``3 * h``. ``loss`` is a registry name
        (``"mae"``/``"mse"``/``"huber"``) or a callable; ``learning_rate`` is a scalar
        or an ``optax.ScalarOrSchedule``; ``activation`` is ``"gelu"`` or ``"relu"``.
        If the fitted estimator will be pickled, any callable passed for
        ``loss``/``learning_rate`` must itself be picklable (a class-based callable or
        module-level function).

        ``use_boxcox`` (default ``False``) applies a variance-stabilizing Box-Cox
        transform to the series before modelling and inverts it on the forecast,
        mirroring the ``use_boxcox`` option of :class:`chronax.models.TBATS`. The
        lambda is selected once at fit time by maximizing the Box-Cox profile
        log-likelihood. It helps multiplicative / strongly-trending series (e.g.
        airline passengers) and **requires strictly positive values**. Left off,
        the model is a faithful port of neuralforecast.iTransformer.
        """
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_boxcox = use_boxcox
        self.activation = activation
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params: ConformalIntervals | None = None
        self.model_: ITransformerNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None
        self._bc_lambda: float | None = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    def _build_net(self) -> ITransformerNet:
        return ITransformerNet(
            h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
            n_heads=self.n_heads, e_layers=self.e_layers, d_ff=self.d_ff,
            dropout=self.dropout, use_norm=self.use_norm, activation=self.activation,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "iTransformer":
        """Fit the network on a 1-D series.

        Builds the network and runs ``max_steps`` Adam steps over rolling windows of
        length ``input_size + h``, sampled per step the way neuralforecast does.

        Args:
            y: 1-D series of length ``>= input_size + h``.
            X: Reserved for exogenous regressors; must be None.

        Returns:
            iTransformer: ``self``, with ``model_`` populated.

        Raises:
            NotImplementedError: If ``X`` is provided.
            ValueError: If ``y`` is not 1-D or is shorter than ``input_size + h``.
            RuntimeError: If a non-finite training loss is observed (divergence).
        """
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(
                f"Series length {y.shape[0]} too short for "
                f"input_size={self.input_size} + h={self.h}."
            )
        # Box-Cox stabilizes the variance before modelling; the network and its
        # RevIN then operate entirely in the transformed space, and predict()
        # inverts on the way out. ``_train_y`` is kept RAW so conformity_scores
        # re-forecasts through the public API (which re-applies the transform).
        y_raw = y
        if self.use_boxcox:
            if bool(jnp.any(y_raw <= 0)):
                raise ValueError(
                    "use_boxcox=True requires strictly positive series values."
                )
            self._bc_lambda = _select_boxcox_lambda(y_raw)
            y = _boxcox(y_raw, self._bc_lambda)
        else:
            self._bc_lambda = None
        net = self._build_net()
        train(
            net, y, h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            windows_batch_size=self.windows_batch_size, lr=self.learning_rate,
            seed=self.random_seed, loss_fn=self._loss_fn,
        )
        self.model_ = net
        self._context = y[-self.input_size:]
        self._train_y = y_raw
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
    ) -> dict:
        """Forecast ``h`` steps from the fitted context.

        Args:
            h: Forecast horizon; must satisfy ``1 <= h <= self.h`` (the model is
                direct-decoded for ``self.h`` steps and sliced).
            X: Reserved for exogenous regressors; ignored.
            level: Optional confidence levels (e.g. ``[80, 95]``). When set, returns
                conformal ``lo-XX``/``hi-XX`` keys via the inherited ``BaseForecaster``
                path and requires ``self.conformal_params``. Each call re-fits the model
                per CV window under ``vmap`` — expect minutes.

        Returns:
            dict: ``{"mean": jnp.ndarray of shape (h,)}`` plus interval keys when
            ``level`` is provided.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If ``h < 1`` or ``h > self.h``, or if ``level`` is given without
                ``self.conformal_params`` set.
        """
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got h={h}.")
        if h > self.h:
            raise ValueError(
                f"iTransformer was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        full = predict_step(self.model_, self._context, h=self.h, input_size=self.input_size)
        if self.use_boxcox and self._bc_lambda is not None:
            full = _inv_boxcox(full, self._bc_lambda)
        fcst = {"mean": full[:h]}
        if level is not None:
            if self.conformal_params is None:
                raise ValueError(
                    "predict(h, level=...) requires `model.conformal_params` to be set. "
                    "Note: conformity_scores re-fits the model per CV window; on "
                    "iTransformer this is expensive — expect MINUTES per call, scaling "
                    "with n_windows and max_steps."
                )
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            # Run the walk-forward on a clone: conformity_scores re-fits under vmap,
            # and those fit() writes would leave leaked tracers on this fitted estimator.
            cs = self.new().conformity_scores(self._train_y)
            method = self.conformal_params.method
            fcst = BaseForecaster.add_confidence_intervals(fcst, cs, level, method)
        return fcst

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        """Stateless fit-then-predict on ``y``.

        Equivalent to ``self.fit(y).predict(h=h, level=level)``, optionally adding a
        ``"fitted"`` key with one-step-ahead in-sample predictions.

        Args:
            y: 1-D training series.
            h: Forecast horizon (``<= self.h``).
            X / X_future: Reserved for exogenous regressors; must be None.
            level: Optional confidence levels; see ``predict``.
            fitted: If True, include ``"fitted"`` — one-step-ahead values over the
                training series, NaN for the first ``input_size`` entries.

        Returns:
            dict: ``{"mean": ..., optional "fitted": ...}``.

        Raises:
            NotImplementedError: If ``X`` or ``X_future`` is provided.
            ValueError / RuntimeError: Forwarded from ``fit`` / ``predict``.
        """
        if X is not None or X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN
        (all-NaN if the series is too short to form any rolling window)."""
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
        # The network lives in the (Box-Cox) transformed space; window the
        # transformed series, then invert the one-step-ahead outputs back to
        # the original scale.
        if self.use_boxcox and self._bc_lambda is not None:
            y_model = _boxcox(y, self._bc_lambda)
        else:
            y_model = y
        n_windows = y.shape[0] - self.input_size
        if n_windows <= 0:
            return jnp.full((y.shape[0],), jnp.nan, dtype=jnp.float32)
        idx = jnp.arange(self.input_size)[None, :] + jnp.arange(n_windows)[:, None]
        in_windows = y_model[idx][..., None]                # [n_windows, L, 1]
        pred = self.model_(in_windows, deterministic=True)
        finite_part = pred[:, 0, 0]                         # one-step-ahead
        if self.use_boxcox and self._bc_lambda is not None:
            finite_part = _inv_boxcox(finite_part, self._bc_lambda)
        nan_head = jnp.full((self.input_size,), jnp.nan, dtype=jnp.float32)
        return jnp.concatenate([nan_head, finite_part])

    def __getstate__(self) -> dict:
        """Serialize only NNX state (params). The GraphDef holds references to JAX
        ufuncs that are not pickle-stable, so we drop it and rebuild via
        ``_build_net()`` on unpickle, then reload the saved state with ``nnx.update``.
        (iTransformer uses LayerNorm, not BatchNorm, so there are no running stats —
        only ``nnx.Param`` variables are captured.)
        """
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            _, full_state = nnx.split(state["model_"])
            state["model_"] = ("__params_only__", full_state)
        return state

    def __setstate__(self, state: dict) -> None:
        m = state.get("model_")
        if isinstance(m, tuple) and m and m[0] == "__params_only__":
            saved_state = m[1]
            state["model_"] = None
            self.__dict__.update(state)
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)
