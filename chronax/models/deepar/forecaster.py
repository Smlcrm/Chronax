"""High-level forecaster API for DeepAR (JAX/FLAX/OPTAX, NumPy-free)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax

from .model import DeepAR_EncDec, forecast_mc
from .data import create_batch
from .loss import nll_gaussian_masked
from .train import TrainState, make_loss_fn, make_train_step


class DeepARForecaster:
    """High-level fit/predict API wrapping DeepAR_EncDec.

    Args:
        h: Forecast horizon.
        hidden_size: LSTM hidden dimension.
        dropout_rate: Dropout probability.
        min_sigma: Minimum σ floor.
        seed: Master random seed.
    """

    def __init__(
        self,
        h: int,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
        min_sigma: float = 0.02,
        seed: int = 0,
    ):
        self.h = h
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.min_sigma = min_sigma
        self.seed = seed

        self.model: Optional[nn.Module] = None
        self.params: Optional[dict] = None
        self.input_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y_series: List[jnp.ndarray],
        x_futr_list: Optional[List[Optional[jnp.ndarray]]] = None,
        x_stat_list: Optional[List[Optional[jnp.ndarray]]] = None,
        input_size: int = 168,
        num_steps: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "DeepARForecaster":
        """Fit DeepAR to a collection of time series.

        Args:
            y_series: List of 1-D JAX arrays (one per series).
            x_futr_list: Future exogenous per series [T+h, F] or None.
            x_stat_list: Static features per series [S] or None.
            input_size: History window length.
            num_steps: Training steps.
            batch_size: Ignored (single-series batches currently).
            learning_rate: Learning rate.
            weight_decay: AdamW weight-decay.
            verbose: Print progress.

        Returns:
            self
        """
        self.input_size = input_size
        self.model = DeepAR_EncDec(
            hidden=self.hidden_size,
            dropout_rate=self.dropout_rate,
            min_sigma=self.min_sigma,
        )

        key = random.PRNGKey(self.seed)
        key, init_key = random.split(key)

        # Determine covariate dimensions from first series
        d_f = 0 if x_futr_list is None or x_futr_list[0] is None else int(x_futr_list[0].shape[1])
        d_s = 0 if x_stat_list is None or x_stat_list[0] is None else int(x_stat_list[0].shape[0])

        dummy_y = jnp.ones(input_size, dtype=jnp.float32)
        dummy_xf = jnp.zeros((input_size, d_f), dtype=jnp.float32) if d_f > 0 else None
        dummy_xs = jnp.zeros(d_s, dtype=jnp.float32) if d_s > 0 else None

        self.params = self.model.init(
            {"params": init_key, "dropout": init_key},
            dummy_y,
            dummy_xf,   # futr_exog — history window length
            dummy_xs,   # x_static
            True,
        )

        if verbose:
            print(f"Initialized. Training for {num_steps} steps...")

        self.params = self._train_loop(
            y_series=y_series,
            x_futr_list=x_futr_list,
            x_stat_list=x_stat_list,
            input_size=input_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose,
        )
        return self

    def _train_loop(
        self,
        y_series: List[jnp.ndarray],
        x_futr_list: Optional[List[Optional[jnp.ndarray]]],
        x_stat_list: Optional[List[Optional[jnp.ndarray]]],
        input_size: int,
        num_steps: int,
        learning_rate: float,
        weight_decay: float,
        verbose: bool,
    ) -> dict:
        tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        opt_state = tx.init(self.params)
        loss_fn = make_loss_fn(self.model)

        key = random.PRNGKey(self.seed)
        params = self.params

        for step in range(num_steps):
            key, bkey, skey = random.split(key, 3)
            idx = step % len(y_series)

            futr = x_futr_list[idx] if x_futr_list else None
            stat = x_stat_list[idx] if x_stat_list else None

            # Build batch using only the history window of futr_exog
            # (create_batch stores futr_exog as [L+h, F], sliced in loss_fn)
            batch = create_batch(
                y_series=[y_series[idx]],
                input_size=input_size,
                h=self.h,
                futr_exog_list=[futr] if futr is not None else None,
                stat_exog_list=[stat] if stat is not None else None,
            )

            loss, grads = jax.value_and_grad(
                lambda p: loss_fn(p, batch, skey)
            )(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step+1:5d} | Loss: {float(loss):.4f}")

        return params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        y_series: jnp.ndarray,
        x_futr_hist: Optional[jnp.ndarray] = None,
        x_futr: Optional[jnp.ndarray] = None,
        x_stat: Optional[jnp.ndarray] = None,
        num_samples: int = 100,
        seed: int = 0,
    ) -> jnp.ndarray:
        """Monte Carlo sample paths.

        Args:
            y_series: Historical series [T].
            x_futr: Future exogenous for the forecast horizon [h, F] or None.
            x_stat: Static features [S] or None.
            num_samples: Number of MC paths.
            seed: Random seed.

        Returns:
            Sample paths [num_samples, h].
        """
        if self.model is None or self.params is None:
            raise ValueError("Call fit() before predict().")

        return forecast_mc(
            self.params,
            self.model,
            y_series,
            x_f_hist=x_futr_hist,
            x_f_future=x_futr,
            x_static=x_stat,
            H=self.h,
            N=num_samples,
            seed=seed,
        )

    def quantile(
        self,
        y_series: jnp.ndarray,
        x_futr_hist: Optional[jnp.ndarray] = None,
        x_futr: Optional[jnp.ndarray] = None,
        x_stat: Optional[jnp.ndarray] = None,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        num_samples: int = 1000,
        seed: int = 0,
    ) -> Dict[float, jnp.ndarray]:
        """Compute quantile forecasts.

        Returns:
            Dict mapping quantile level → [h] array.
        """
        paths = self.predict(y_series, x_futr_hist=x_futr_hist, x_futr=x_futr, x_stat=x_stat,
                             num_samples=num_samples, seed=seed)
        return {q: jnp.quantile(paths, q, axis=0) for q in quantiles}

    def forecast(
        self,
        y_series: jnp.ndarray,
        x_futr_hist: Optional[jnp.ndarray] = None,
        x_futr: Optional[jnp.ndarray] = None,
        x_stat: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Point forecast (median) with 10/90 uncertainty intervals.

        Returns:
            Dict with keys ``median``, ``lower``, ``upper``, each [h].
        """
        q = self.quantile(y_series, x_futr_hist=x_futr_hist, x_futr=x_futr, x_stat=x_stat,
                          quantiles=(0.1, 0.5, 0.9), num_samples=500)
        return {"median": q[0.5], "lower": q[0.1], "upper": q[0.9]}
