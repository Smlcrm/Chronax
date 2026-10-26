"""High-level forecaster interface for DeepAR (JAX/FLAX/OPTAX, NumPy-free)."""

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
    """High-level fit/predict interface wrapping :class:`DeepAR_EncDec`.

    Probabilistic forecaster: a long short-term memory (LSTM) encoder-decoder
    emits a Gaussian mean and scale per step, and :meth:`predict` draws Monte
    Carlo (MC) sample paths from it. This class does not subclass
    :class:`~chronax.models.base_forecaster.BaseForecaster`: its ``fit``,
    ``predict`` and ``forecast`` signatures and return types differ from that
    contract and no conformal intervals are available.

    Args:
        h (int): Forecast horizon.
        hidden_size (int): LSTM hidden dimension. Default ``64``.
        dropout_rate (float): Dropout probability. Default ``0.1``.
        min_sigma (float): Lower floor added to the predicted Gaussian scale.
            Default ``0.02``.
        seed (int): Master random seed for parameter init and training.
            Default ``0``.
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

        Each training step teacher-forces one series (cycling through
        ``y_series`` in order) on the last ``input_size + h`` points of that
        series and takes one AdamW step on the masked Gaussian negative
        log-likelihood.

        Args:
            y_series (list[jnp.ndarray]): One 1-D array per series, shape
                ``(T_i,)``.
            x_futr_list (list[jnp.ndarray | None] | None): Future exogenous
                regressors per series, shape ``[T_i + h, F]``, or ``None``.
            x_stat_list (list[jnp.ndarray | None] | None): Static features
                per series, shape ``[S]``, or ``None``.
            input_size (int): History window length. Default ``168``.
            num_steps (int): Training steps. Default ``1000``.
            batch_size (int): Accepted for interface parity; unused, every
                step trains on a single series. Default ``32``.
            learning_rate (float): AdamW learning rate. Default ``1e-3``.
            weight_decay (float): AdamW weight decay. Default ``1e-5``.
            verbose (bool): Print the loss every 100 steps. Default ``True``.

        Returns:
            DeepARForecaster: ``self``.
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
            dummy_xf,   # futr_exog over the history window
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
        """Draw Monte Carlo sample paths over the horizon.

        The whole of ``y_series`` is encoded; the decoder then samples ``h``
        steps autoregressively, ``num_samples`` times. Inputs are cast to
        float32.

        Args:
            y_series (jnp.ndarray): Historical series, shape ``(T,)``.
            x_futr_hist (jnp.ndarray | None): Future-known exogenous
                regressors aligned with the history, shape ``[T, F]``, or
                ``None``.
            x_futr (jnp.ndarray | None): Future exogenous regressors for the
                forecast horizon, shape ``[h, F]``, or ``None``.
            x_stat (jnp.ndarray | None): Static features, shape ``[S]``, or
                ``None``.
            num_samples (int): Number of MC paths. Default ``100``.
            seed (int): Random seed for the sampling. Default ``0``.

        Returns:
            jnp.ndarray: Sample paths, shape ``[num_samples, h]``.

        Raises:
            ValueError: If called before :meth:`fit`.
        """
        if self.model is None or self.params is None:
            raise ValueError("Call fit() before predict().")

        # The model is float32 throughout (init + training batches are cast in
        # create_batch); coerce inference inputs likewise so a float64 series
        # (easy to produce when the global x64 flag is on) cannot poison the
        # LSTM scan carry dtype.
        y_series = jnp.asarray(y_series, jnp.float32)
        x_futr_hist = None if x_futr_hist is None else jnp.asarray(x_futr_hist, jnp.float32)
        x_futr = None if x_futr is None else jnp.asarray(x_futr, jnp.float32)
        x_stat = None if x_stat is None else jnp.asarray(x_stat, jnp.float32)

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
            # The forecaster's training pipeline (train.py loss path) does not
            # use the lag-feature augmentation, so inference must not either;
            # otherwise the encoder input width no longer matches the params.
            use_lags=False,
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
        """Compute quantile forecasts from MC sample paths.

        Args:
            y_series (jnp.ndarray): Historical series, shape ``(T,)``.
            x_futr_hist (jnp.ndarray | None): Future-known exogenous
                regressors aligned with the history, shape ``[T, F]``, or
                ``None``.
            x_futr (jnp.ndarray | None): Future exogenous regressors for the
                horizon, shape ``[h, F]``, or ``None``.
            x_stat (jnp.ndarray | None): Static features, shape ``[S]``, or
                ``None``.
            quantiles (tuple[float, ...]): Quantile levels in (0, 1).
                Default ``(0.1, 0.5, 0.9)``.
            num_samples (int): Number of MC paths. Default ``1000``.
            seed (int): Random seed for the sampling. Default ``0``.

        Returns:
            dict[float, jnp.ndarray]: Quantile level to array of shape
            ``(h,)``.

        Raises:
            ValueError: If called before :meth:`fit`.
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
        """Median point forecast with 10th and 90th percentile bounds.

        Draws 500 MC sample paths with ``seed=0``. Not the
        :class:`~chronax.models.base_forecaster.BaseForecaster` ``forecast``
        signature: it takes no ``h`` or ``level`` and returns different keys.

        Args:
            y_series (jnp.ndarray): Historical series, shape ``(T,)``.
            x_futr_hist (jnp.ndarray | None): Future-known exogenous
                regressors aligned with the history, shape ``[T, F]``, or
                ``None``.
            x_futr (jnp.ndarray | None): Future exogenous regressors for the
                horizon, shape ``[h, F]``, or ``None``.
            x_stat (jnp.ndarray | None): Static features, shape ``[S]``, or
                ``None``.

        Returns:
            dict[str, jnp.ndarray]: Keys ``"median"``, ``"lower"`` (10th
            percentile) and ``"upper"`` (90th percentile), each of shape
            ``(h,)``.

        Raises:
            ValueError: If called before :meth:`fit`.
        """
        q = self.quantile(y_series, x_futr_hist=x_futr_hist, x_futr=x_futr, x_stat=x_stat,
                          quantiles=(0.1, 0.5, 0.9), num_samples=500)
        return {"median": q[0.5], "lower": q[0.1], "upper": q[0.9]}
