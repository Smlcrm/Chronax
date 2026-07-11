"""DeepNPTS forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.deepnpts.deepnpts_losses import LossFn, resolve as _resolve_loss
from chronax.models.deepnpts.deepnpts_module import DeepNPTSNet
from chronax.models.deepnpts.deepnpts_training import predict_step, train
from chronax.utils import ConformalIntervals

# --- Box-Cox variance-stabilizing transform (self-contained) ---------------
# Mirrors the ``use_boxcox`` convention of chronax.models.TBATS / BiTCN.

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
    """Pick lambda by maximizing the Box-Cox profile log-likelihood (MLE)."""
    y = jnp.asarray(y, dtype=jnp.float32)
    n = y.size
    log_sum = jnp.sum(jnp.log(y))

    def ll(lam: float) -> jnp.ndarray:
        z = _boxcox(y, lam)
        var = jnp.maximum(jnp.var(z), 1e-12)
        return -0.5 * n * jnp.log(var) + (lam - 1.0) * log_sum

    lls = jnp.array([ll(float(lam)) for lam in _BOXCOX_LAMBDA_GRID])
    return float(_BOXCOX_LAMBDA_GRID[int(jnp.argmax(lls))])


class DeepNPTS(BaseForecaster):
    """Univariate DeepNPTS forecaster (JAX/Flax-NNX port of neuralforecast.DeepNPTS).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via ``predict(level=...)``,
        pickle round-trip, and ``forecast(fitted=True)``.

    Deep Non-Parametric Time Series forecaster (Rangapuram, Gasthaus, Stella,
    Flunkert, Salinas, Wang & Januschowski, 2023). A small MLP reads the lookback
    window and emits, per horizon step, a softmax weight over each window
    position; the forecast is the weighted sum of the raw in-sample values, i.e.
    a learned non-parametric resample of the context. For the univariate,
    no-exogenous case the MLP input dimension is ``input_size``. NF default
    ``scaler_type="identity"`` is mirrored (trained in raw scale) with Optax
    ``adam`` and a pluggable point loss. ``float32`` throughout.
    """

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = False,
        use_boxcox: bool = False,
        max_steps: int = 1000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-3,
        windows_batch_size: int = 1024,
        random_seed: int = 1,
        alias: str = "DeepNPTS",
        loss: Union[str, LossFn] = "mae",
    ):
        """Initialize a DeepNPTS forecaster.

        Stores hyperparameters; the network is built lazily at ``fit`` time.
        Defaults match neuralforecast.DeepNPTS for the univariate case
        (``hidden_size=32``, ``n_layers=2``, ``dropout=0.1``, ``max_steps=1000``,
        ``learning_rate=1e-3``, ``windows_batch_size=1024``,
        ``scaler_type="identity"``), with one documented override: ``batch_norm``
        defaults to ``False`` (NF defaults ``True``). ``input_size=-1`` resolves to
        ``3 * h``. ``loss`` is a registry name (``"mae"``/``"mse"``/``"huber"``) or
        a callable; ``learning_rate`` is a scalar or an ``optax.ScalarOrSchedule``.
        If the fitted estimator will be pickled, any callable passed for
        ``loss``/``learning_rate`` must itself be picklable.

        ``batch_norm`` (default ``False``) applies Batch Normalization after each
        dense layer, matching NF's default architecture when enabled. It is left
        off by default because BatchNorm carries non-parameter running statistics
        that add state complexity to training, inference, and pickling; when
        enabled it uses ``momentum=0.9`` (equivalent to torch's ``0.1``) and
        ``eps=1e-5``.

        ``use_boxcox`` (default ``False``) applies a variance-stabilizing Box-Cox
        transform before modelling and inverts it on the forecast, mirroring
        :class:`chronax.models.TBATS`. It **requires strictly positive values**.
        Left off, the model is a faithful port of neuralforecast.DeepNPTS.
        """
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_boxcox = use_boxcox
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params: ConformalIntervals | None = None
        self.model_: DeepNPTSNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None
        self._bc_lambda: float | None = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    def _build_net(self) -> DeepNPTSNet:
        return DeepNPTSNet(
            h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
            n_layers=self.n_layers, dropout=self.dropout, batch_norm=self.batch_norm,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "DeepNPTS":
        """Fit the network on a 1-D series of length ``>= input_size + h``."""
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
        """Forecast ``h`` steps (``1 <= h <= self.h``) from the fitted context."""
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        if h < 1:
            raise ValueError(f"h must be a positive integer; got h={h}.")
        if h > self.h:
            raise ValueError(
                f"DeepNPTS was trained for h={self.h}; predict(h={h}) is not supported. "
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
                    "DeepNPTS this is expensive — expect MINUTES per call."
                )
            if self._train_y is None:
                raise RuntimeError("Call fit(y) before predict(h, level=...).")
            cs = self.conformity_scores(self._train_y)
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
        """Stateless fit-then-predict on ``y``. Optionally add ``"fitted"``."""
        if X is not None or X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported.")
        self.fit(y)
        result = self.predict(h=h, level=level)
        if fitted:
            result["fitted"] = self._compute_fitted_values()
        return result

    def _compute_fitted_values(self) -> jnp.ndarray:
        """One-step-ahead fitted values; first ``input_size`` entries are NaN."""
        if self._train_y is None or self.model_ is None:
            raise RuntimeError("Call fit(y) before computing fitted values.")
        y = self._train_y
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
        """Serialize NNX state (params + any BatchStat); rebuild GraphDef on unpickle.

        A full ``nnx.split`` captures every variable — ``nnx.Param`` and, when
        ``batch_norm=True``, ``nnx.BatchStat`` running statistics — so the round
        trip is exact regardless of the BatchNorm setting.
        """
        state = self.__dict__.copy()
        if state.get("model_") is not None:
            _, full_state = nnx.split(state["model_"])
            state["model_"] = ("__nnx_state__", full_state)
        return state

    def __setstate__(self, state: dict) -> None:
        m = state.get("model_")
        if isinstance(m, tuple) and m and m[0] == "__nnx_state__":
            saved_state = m[1]
            state["model_"] = None
            self.__dict__.update(state)
            net = self._build_net()
            nnx.update(net, saved_state)
            self.model_ = net
            return
        self.__dict__.update(state)
