"""VanillaTransformer forecaster: BaseForecaster wrapper around the JAX/Flax-NNX backbone."""
from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.vanillatransformer.vanillatransformer_losses import LossFn, resolve as _resolve_loss
from chronax.models.vanillatransformer.vanillatransformer_module import VanillaTransformerNet
from chronax.models.vanillatransformer.vanillatransformer_training import predict_step, train
from chronax.utils import ConformalIntervals

# --- Box-Cox variance-stabilizing transform (self-contained) ---------------
# Mirrors the ``use_boxcox`` convention of chronax.models.TBATS / iTransformer.

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


class VanillaTransformer(BaseForecaster):
    """Univariate VanillaTransformer forecaster (JAX/Flax-NNX port of neuralforecast.VanillaTransformer).

    Maintenance Status:
        Active univariate forecaster. Integrates with the ``BaseForecaster``
        interface, including conformal prediction intervals via ``predict(level=...)``,
        pickle round-trip, and ``forecast(fitted=True)``.

    A classic encoder-decoder Transformer (Informer baseline). The encoder embeds
    the lookback window (circular token conv + fixed sinusoidal positional
    embedding) and runs full softmax self-attention; the decoder embeds
    ``concat(last label_len of input, zeros for h)`` and runs full self-attention
    + cross-attention against the encoder output, then projects to one channel and
    returns the last ``h`` steps. NF default ``scaler_type="identity"`` is mirrored
    (trained in raw scale) with Optax ``adam`` and a pluggable point loss.
    ``float32`` throughout.
    """

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 128,
        n_heads: int = 4,
        conv_hidden_size: int = 32,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        decoder_input_size_multiplier: float = 0.5,
        dropout: float = 0.05,
        use_boxcox: bool = False,
        activation: str = "gelu",
        max_steps: int = 5000,
        learning_rate: Union[float, Callable[[int], float]] = 1e-4,
        windows_batch_size: int = 1024,
        random_seed: int = 1,
        alias: str = "VanillaTransformer",
        loss: Union[str, LossFn] = "mae",
    ):
        """Initialize a VanillaTransformer forecaster.

        Stores hyperparameters; the network is built lazily at ``fit`` time.
        Defaults match neuralforecast.VanillaTransformer (``hidden_size=128``,
        ``n_head=4``, ``conv_hidden_size=32``, ``encoder_layers=2``,
        ``decoder_layers=1``, ``decoder_input_size_multiplier=0.5``,
        ``dropout=0.05``, ``activation="gelu"``, ``max_steps=5000``,
        ``learning_rate=1e-4``, ``windows_batch_size=1024``,
        ``scaler_type="identity"``). ``input_size=-1`` resolves to ``3 * h``.
        ``loss`` is a registry name (``"mae"``/``"mse"``/``"huber"``) or a callable;
        ``learning_rate`` is a scalar or an ``optax.ScalarOrSchedule``; ``activation``
        is ``"gelu"`` or ``"relu"``. If the fitted estimator will be pickled, any
        callable passed for ``loss``/``learning_rate`` must itself be picklable.

        ``use_boxcox`` (default ``False``) applies a variance-stabilizing Box-Cox
        transform before modelling and inverts it on the forecast, mirroring
        :class:`chronax.models.TBATS`. It **requires strictly positive values**.
        Left off, the model is a faithful port of neuralforecast.VanillaTransformer.
        """
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.conv_hidden_size = conv_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.decoder_input_size_multiplier = decoder_input_size_multiplier
        self.dropout = dropout
        self.use_boxcox = use_boxcox
        self.activation = activation
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.windows_batch_size = windows_batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.loss = loss
        self.conformal_params: ConformalIntervals | None = None
        self.model_: VanillaTransformerNet | None = None
        self._context: jnp.ndarray | None = None
        self._train_y: jnp.ndarray | None = None
        self._bc_lambda: float | None = None

    @property
    def _loss_fn(self) -> LossFn:
        """Resolve ``self.loss`` to a callable. Lazy so string forms pickle."""
        return _resolve_loss(self.loss)

    def _build_net(self) -> VanillaTransformerNet:
        return VanillaTransformerNet(
            h=self.h, input_size=self.input_size, hidden_size=self.hidden_size,
            n_heads=self.n_heads, conv_hidden_size=self.conv_hidden_size,
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers,
            dropout=self.dropout, activation=self.activation,
            decoder_input_size_multiplier=self.decoder_input_size_multiplier,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "VanillaTransformer":
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
                f"VanillaTransformer was trained for h={self.h}; predict(h={h}) is not supported. "
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
                    "VanillaTransformer this is expensive — expect MINUTES per call."
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
        """Serialize only NNX state (params); rebuild GraphDef on unpickle.

        VanillaTransformer uses LayerNorm (no BatchNorm), so only ``nnx.Param``
        variables are captured.
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
