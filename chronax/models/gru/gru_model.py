"""GRU forecaster (BaseForecaster wrapper).

Univariate, direct-decoding GRU with per-window robust scaling. Defaults mirror
Nixtla `neuralforecast.GRU`: hidden=200, n_layers=2, decoder_hidden=128,
decoder_layers=2, max_steps=1000, lr=1e-3.

Known v1 limitations
--------------------
- Univariate only. Multi-series / cross-learning is a v2 concern. The fit
  signature will change in v2 to accept ``[N, T]`` instead of ``[T]``; users
  relying on v1 shape should pin the package version.
- ``BaseForecaster.conformity_scores`` works through inheritance but is
  impractical: it re-trains a 1000-step optimizer per cross-validation window.
  Use a smaller ``max_steps`` if you need this on GRU, or wait for the
  conformal-aware v1.1 path.
- No GPU tuning. The model runs on GPU but is benchmarked on CPU only.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from chronax.models.base_forecaster import BaseForecaster
from chronax.models.gru.gru_module import GRUNet
from chronax.models.gru.gru_scaler import RobustScaler
from chronax.models.gru.gru_training import predict_step, train


class GRU(BaseForecaster):
    """JAX/Flax/Optax GRU forecaster. Univariate, direct decoding."""

    uses_exog = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 200,
        n_layers: int = 2,
        decoder_hidden_size: int = 128,
        dropout: float = 0.0,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        random_seed: int = 1,
        alias: str = "GRU",
    ):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.alias = alias
        self.conformal_params = None
        self.model_: GRUNet | None = None
        self._context: jnp.ndarray | None = None  # last `input_size` of fit-time y
        self._scaler = RobustScaler()

    def _build_net(self) -> GRUNet:
        return GRUNet(
            in_features=1,
            encoder_hidden=self.hidden_size,
            encoder_layers=self.n_layers,
            decoder_hidden=self.decoder_hidden_size,
            decoder_layers=2,  # v1 only; GRUNet enforces this invariant.
            dropout=self.dropout,
            h=self.h,
            input_size=self.input_size,
            rngs=nnx.Rngs(self.random_seed),
        )

    def fit(self, y: jnp.ndarray, X: jnp.ndarray | None = None) -> "GRU":
        if X is not None:
            raise NotImplementedError("Exogenous variables are not supported in v1.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D; got shape {y.shape}.")
        if y.shape[0] < self.input_size + self.h:
            raise ValueError(
                f"Series length {y.shape[0]} too short for "
                f"input_size={self.input_size} + h={self.h}."
            )
        net = self._build_net()
        train(
            net, y,
            h=self.h, input_size=self.input_size,
            max_steps=self.max_steps, batch_size=self.batch_size,
            lr=self.learning_rate, seed=self.random_seed,
            scaler=self._scaler,
        )
        self.model_ = net
        # Only the final `input_size` window is needed for direct-decoding
        # forecast; storing the full series would be wasteful for long inputs.
        self._context = y[-self.input_size :]
        return self

    def predict(
        self,
        h: int,
        X: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
    ) -> dict:
        if h > self.h:
            raise ValueError(
                f"GRU was trained for h={self.h}; predict(h={h}) is not supported. "
                f"Pass h <= {self.h} or re-fit with a larger h."
            )
        if level is not None:
            raise NotImplementedError("Probabilistic intervals are not supported in v1.")
        if self.model_ is None or self._context is None:
            raise RuntimeError("Call fit(y) before predict(h).")
        full = predict_step(
            self.model_, self._context,
            h=self.h, input_size=self.input_size, scaler=self._scaler,
        )
        return {"mean": full[:h]}

    def forecast(
        self,
        y: jnp.ndarray,
        h: int,
        X: jnp.ndarray | None = None,
        X_future: jnp.ndarray | None = None,
        level: list[int | float] | None = None,
        fitted: bool = False,
    ) -> dict:
        if X is not None or X_future is not None:
            raise NotImplementedError("Exogenous variables are not supported in v1.")
        if fitted:
            raise NotImplementedError("In-sample fitted values are not supported in v1.")
        return self.fit(y).predict(h=h, level=level)
