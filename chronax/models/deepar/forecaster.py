"""High-level forecaster API for DeepAR (JAX/FLAX/OPTAX, NumPy-free)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax

from chronax.models.base_forecaster import BaseForecaster
from .model import DeepAR_EncDec, forecast_mc, forecast_point
from .data import create_batch
from .loss import nll_gaussian_masked
from .train import TrainState, make_loss_fn, make_train_step


class DeepARForecaster(BaseForecaster):
    """High-level fit/predict API wrapping DeepAR_EncDec.

    Args:
        h: Forecast horizon.
        input_size: History window length; -1 (default) uses ``3 * h``.
        hidden_size: LSTM hidden dimension.
        dropout_rate: Dropout probability.
        min_sigma: Minimum σ floor.
        random_seed: Master random seed.
        max_steps: Optimiser steps used by :meth:`fit` (harness override name;
            same role as ``num_steps``).
        learning_rate: AdamW peak learning rate.
        num_lr_decays: Number of stepwise LR halvings across ``max_steps``
            (parity with neuralforecast DeepAR defaults).
        loss: Accepted for API parity with the other Chronax neural
            forecasters; unused — DeepAR is trained by Gaussian NLL
            (:func:`chronax.models.deepar.loss.nll_gaussian_masked`), which is
            not swappable for a point loss like MAE/MSE.
    """

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        hidden_size: int = 64,
        dropout_rate: float = 0.1,
        min_sigma: float = 0.02,
        random_seed: int = 0,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        num_lr_decays: int = 3,
        loss: str = "mae",
    ):
        if input_size < 1:
            input_size = 3 * h
        self.h = h
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.min_sigma = min_sigma
        self.random_seed = random_seed
        self.seed = random_seed
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_lr_decays = num_lr_decays
        self.loss = loss

        self.model: Optional[nn.Module] = None
        self.params: Optional[dict] = None
        self._fit_y: Optional[jnp.ndarray] = None
        self._scaler: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        y_series: Union[jnp.ndarray, List[jnp.ndarray]],
        x_futr_list: Optional[List[Optional[jnp.ndarray]]] = None,
        x_stat_list: Optional[List[Optional[jnp.ndarray]]] = None,
        input_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> "DeepARForecaster":
        """Fit DeepAR to a single series or a collection of time series.

        Args:
            y_series: A single 1-D JAX array, or a list of 1-D JAX arrays for
                panel training (one per series).
            x_futr_list: Future exogenous per series [T+h, F] or None.
            x_stat_list: Static features per series [S] or None.
            input_size: History window length. Defaults to the value derived
                from ``h`` at construction (``self.input_size``) when None.
            num_steps: Training steps. Defaults to ``self.max_steps``.
            batch_size: Ignored (single-series batches currently).
            learning_rate: Learning rate. Defaults to ``self.learning_rate``.
            weight_decay: AdamW weight-decay.
            verbose: Print progress.

        Returns:
            self
        """
        if isinstance(y_series, jnp.ndarray) and y_series.ndim == 1:
            self._fit_y = y_series
            y_series = [y_series]
        else:
            self._fit_y = y_series[0] if y_series else None

        if input_size is None:
            input_size = self.input_size
        if num_steps is None:
            num_steps = self.max_steps
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.input_size = input_size

        # Per-fit z-score (same as train_model): NLL without scaling collapses μ≈0
        # on large-level series (AirlinePassengers etc.).
        y0 = y_series[0]
        y_mean = float(jnp.mean(y0))
        y_std = float(jnp.std(y0)) + 1e-8
        self._scaler = (y_mean, y_std)
        y_series = [(y - y_mean) / y_std for y in y_series]

        self.model = DeepAR_EncDec(
            hidden=self.hidden_size,
            dropout_rate=self.dropout_rate,
            min_sigma=self.min_sigma,
        )

        key = random.PRNGKey(self.seed)
        key, init_key = random.split(key)

        # Determine covariate dimensions from first series.
        # +4 for lag-1/7/h/2h features always concatenated in make_loss_fn /
        # forecast_mc (must match /proj input width).
        d_f_user = 0 if x_futr_list is None or x_futr_list[0] is None else int(x_futr_list[0].shape[1])
        d_s = 0 if x_stat_list is None or x_stat_list[0] is None else int(x_stat_list[0].shape[0])
        d_f = d_f_user + 4

        dummy_y = jnp.ones(input_size, dtype=jnp.float32)
        dummy_xf = jnp.zeros((input_size, d_f), dtype=jnp.float32)
        dummy_xs = jnp.zeros(d_s, dtype=jnp.float32) if d_s > 0 else None

        self.params = self.model.init(
            {"params": init_key, "dropout": init_key},
            dummy_y,
            dummy_xf,   # futr_exog (+ lag dims) — history window length
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
        # StepLR-style decays matching neuralforecast DeepAR (num_lr_decays=3).
        n_decays = max(int(self.num_lr_decays), 0)
        if n_decays > 0:
            decay_every = max(num_steps // (n_decays + 1), 1)
            schedule = optax.piecewise_constant_schedule(
                init_value=learning_rate,
                boundaries_and_scales={
                    decay_every * (i + 1): 0.5 for i in range(n_decays)
                },
            )
            tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        else:
            tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        opt_state = tx.init(self.params)
        loss_fn = make_loss_fn(self.model, h=self.h)

        # JIT-compiled gradient step. Compiles once (batch shape/structure is
        # fixed for the whole loop — win, futr/stat presence never change
        # across steps) and is then reused for every one of ``num_steps``
        # iterations, instead of retracing+running eagerly every step.
        @jax.jit
        def _step(params, opt_state, batch, rng):
            loss, grads = jax.value_and_grad(
                lambda p: loss_fn(p, batch, rng)
            )(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        key = random.PRNGKey(self.seed)
        params = self.params
        win = input_size + self.h

        for step in range(num_steps):
            key, bkey, skey, wkey = random.split(key, 4)
            idx = int(step % len(y_series))
            y = y_series[idx]
            T = int(y.shape[0])
            n_wins = max(1, T - win + 1)
            start = int(random.randint(wkey, (), 0, n_wins))
            y_win = y[start: start + win]

            futr = x_futr_list[idx] if x_futr_list else None
            stat = x_stat_list[idx] if x_stat_list else None
            if futr is not None:
                # Align futr_exog to the sampled window ([L+h, F]).
                futr = futr[start: start + win]

            batch = create_batch(
                y_series=[y_win],
                input_size=input_size,
                h=self.h,
                futr_exog_list=[futr] if futr is not None else None,
                stat_exog_list=[stat] if stat is not None else None,
            )

            params, opt_state, loss = _step(params, opt_state, batch, skey)

            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step+1:5d} | Loss: {float(loss):.4f}")

        return params

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_mc(
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
            raise ValueError("Call fit() before predict_mc().")

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
            scaler=self._scaler,
            input_size=self.input_size,
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
        paths = self.predict_mc(y_series, x_futr_hist=x_futr_hist, x_futr=x_futr, x_stat=x_stat,
                                num_samples=num_samples, seed=seed)
        return {q: jnp.quantile(paths, q, axis=0) for q in quantiles}

    def forecast(
        self,
        y_series: jnp.ndarray,
        x_futr_hist: Optional[jnp.ndarray] = None,
        x_futr: Optional[jnp.ndarray] = None,
        x_stat: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Point forecast (greedy μ) with 10/90 MC uncertainty intervals.

        Returns:
            Dict with keys ``median``, ``lower``, ``upper``, each [h].
        """
        point = forecast_point(
            self.params,
            self.model,
            y_series,
            x_f_hist=x_futr_hist,
            x_f_future=x_futr,
            x_static=x_stat,
            H=self.h,
            scaler=self._scaler,
            input_size=self.input_size,
        )
        q = self.quantile(y_series, x_futr_hist=x_futr_hist, x_futr=x_futr, x_stat=x_stat,
                          quantiles=(0.1, 0.9), num_samples=100)
        return {"median": point, "lower": q[0.1], "upper": q[0.9]}

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[jnp.ndarray] = None,
        level: Optional[List[Union[int, float]]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forecast from the series passed to :meth:`fit`.

        Satisfies the ``BaseForecaster`` contract: returns a greedy-μ point
        forecast as ``"mean"`` (MAE-aligned; no MC noise).

        Args:
            h: Forecast horizon. Must equal ``self.h`` if provided.
            X: Reserved for future exogenous regressors; unused.
            level: Not yet supported for this model.

        Returns:
            dict: ``{"mean": jnp.ndarray}``.
        """
        if self.model is None or self.params is None:
            raise RuntimeError("Call .fit() before .predict().")
        if h is not None and h != self.h:
            raise ValueError(
                f"DeepARForecaster was configured with h={self.h}; got h={h}. "
                f"Construct a new forecaster for a different horizon."
            )
        if level is not None:
            raise NotImplementedError(
                "DeepARForecaster.predict(level=...) is not yet supported."
            )
        mean = forecast_point(
            self.params,
            self.model,
            self._fit_y,
            H=self.h,
            scaler=self._scaler,
            input_size=self.input_size,
        )
        return {"mean": mean}
