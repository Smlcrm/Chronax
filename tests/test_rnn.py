"""End-to-end tests for the Chronax JAX/Flax RNN.

A small PyTorch ground-truth (``_PyTorchRNNReference``) is defined inline at
the top of this file. It re-implements only the architecture and forward pass
of Nixtla NeuralForecast's ``RNN`` (the heavy ``BaseModel`` + Lightning +
NeuralForecast machinery is intentionally absent) so we can compare forward
passes byte-for-byte after transferring weights between the two state dicts.

Test plan:

* ``test_loss_*``               — numerical correctness of MAE/MSE under
                                  masking and horizon-weighting.
* ``test_data_pipeline``        — batch shapes, padding, available/sample masks.
* ``test_data_alignment``       — covariate alignment to ``input_size + h``.
* ``test_forward_parity_*``     — JAX vs PT forward outputs match (recurrent
                                  on/off, with and without exogs, ReLU, no-bias,
                                  upsample, masked targets).
* ``test_autoregressive_*``     — multi-step rollout matches a PT autoregressive
                                  loop driven by the same weights.
* ``test_train_step_runs``      — full train + eval step on a JIT path produces
                                  finite losses and parameter changes.
* ``test_forecaster_*``         — high-level RNNForecaster fit/forecast smoke.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as torch_nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chronax.models.rnn.data import (  # noqa: E402
    align_covariates,
    create_batch,
    pad_sequence,
)
from chronax.models.rnn.loss import masked_mae, masked_mse  # noqa: E402
from chronax.models.rnn.model import (  # noqa: E402
    RNN,
    RNNConfig,
    autoregressive_predict,
)
from chronax.models.rnn.train import (  # noqa: E402
    create_train_state,
    eval_step,
    train_step,
)


# ---------------------------------------------------------------------------
# Inline PyTorch reference (architecture-only copy of Nixtla NeuralForecast's
# ``RNN``). Kept inside the test module so the parity tests remain
# self-contained and independent of any external reference package.
# ---------------------------------------------------------------------------


class _RefMLP(torch_nn.Module):
    """MLP head matching ``neuralforecast.common._modules.MLP``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers == 1:
            self.layers = torch_nn.Sequential(
                torch_nn.Linear(in_features=in_features, out_features=out_features)
            )
        else:
            self.activation = getattr(torch_nn, activation)()
            layers = [
                torch_nn.Linear(in_features=in_features, out_features=hidden_size),
                self.activation,
                torch_nn.Dropout(dropout),
            ]
            for _ in range(num_layers - 2):
                layers += [
                    torch_nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    self.activation,
                    torch_nn.Dropout(dropout),
                ]
            layers += [
                torch_nn.Linear(in_features=hidden_size, out_features=out_features)
            ]
            self.layers = torch_nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _PyTorchRNNReference(torch_nn.Module):
    """Architecture-only mirror of Nixtla NeuralForecast's ``RNN.forward``.

    Intentionally minimal: only the parts of ``forward`` that produce the
    output tensor are reproduced. Parameter naming matches the upstream model
    so a state_dict from this module can be transferred 1:1 into the Flax port.
    """

    def __init__(
        self,
        h: int,
        input_size: int,
        encoder_hidden_size: int = 128,
        encoder_n_layers: int = 2,
        encoder_activation: str = "tanh",
        encoder_bias: bool = True,
        encoder_dropout: float = 0.0,
        decoder_hidden_size: int = 128,
        decoder_layers: int = 2,
        futr_exog_size: int = 0,
        hist_exog_size: int = 0,
        stat_exog_size: int = 0,
        output_size: int = 1,
        recurrent: bool = False,
    ):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.futr_exog_size = futr_exog_size
        self.hist_exog_size = hist_exog_size
        self.stat_exog_size = stat_exog_size
        self.RECURRENT = recurrent

        input_encoder = 1 + hist_exog_size + stat_exog_size + futr_exog_size

        self.rnn_state: Optional[torch.Tensor] = None
        self.maintain_state = False

        self.hist_encoder = torch_nn.RNN(
            input_size=input_encoder,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_n_layers,
            bias=encoder_bias,
            dropout=encoder_dropout,
            batch_first=True,
            nonlinearity=encoder_activation,
        )

        if recurrent:
            self.proj = torch_nn.Linear(encoder_hidden_size, output_size)
        else:
            self.mlp_decoder = _RefMLP(
                in_features=encoder_hidden_size + futr_exog_size,
                out_features=output_size,
                hidden_size=decoder_hidden_size,
                num_layers=decoder_layers,
                activation="ReLU",
                dropout=0.0,
            )
            if h > input_size:
                self.upsample_sequence = torch_nn.Linear(input_size, h)

    @staticmethod
    def _to_tensor(x):
        if x is None:
            return None
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    def forward(self, batch: dict) -> torch.Tensor:
        encoder_input = self._to_tensor(batch["insample_y"])
        futr_exog = self._to_tensor(batch.get("futr_exog"))
        hist_exog = self._to_tensor(batch.get("hist_exog"))
        stat_exog = self._to_tensor(batch.get("stat_exog"))

        _, seq_len = encoder_input.shape[:2]

        if self.hist_exog_size > 0:
            encoder_input = torch.cat((encoder_input, hist_exog), dim=2)
        if self.stat_exog_size > 0:
            stat_expanded = stat_exog.unsqueeze(1).repeat(1, seq_len, 1)
            encoder_input = torch.cat((encoder_input, stat_expanded), dim=2)
        if self.futr_exog_size > 0:
            encoder_input = torch.cat(
                (encoder_input, futr_exog[:, :seq_len]), dim=2
            )

        if self.RECURRENT:
            rnn_state = self.rnn_state if self.maintain_state else None
            output, rnn_state = self.hist_encoder(encoder_input, rnn_state)
            output = self.proj(output)
            if self.maintain_state:
                self.rnn_state = rnn_state
        else:
            hidden_state, _ = self.hist_encoder(encoder_input, None)
            if self.h > self.input_size:
                hidden_state = hidden_state.permute(0, 2, 1)
                hidden_state = self.upsample_sequence(hidden_state)
                hidden_state = hidden_state.permute(0, 2, 1)
            else:
                hidden_state = hidden_state[:, -self.h:]

            if self.futr_exog_size > 0:
                futr_exog_futr = futr_exog[:, -self.h:]
                hidden_state = torch.cat((hidden_state, futr_exog_futr), dim=-1)

            output = self.mlp_decoder(hidden_state)

        return output[:, -self.h:]


# Alias used throughout the parity tests below.
PyTorchRNNReference = _PyTorchRNNReference


# ---------------------------------------------------------------------------
# Weight transfer PyTorch -> JAX
# ---------------------------------------------------------------------------


def _transfer_weights(pt_model, jax_params, config: RNNConfig):
    """Map a ``PyTorchRNNReference`` state_dict into Flax param dict shape.

    Tensors are transposed where Flax stores transposed weight matrices
    (Dense kernels: ``[in, out]`` vs PyTorch ``[out, in]``).
    """
    from flax.core import freeze, unfreeze

    p = unfreeze(jax_params)
    sd = pt_model.state_dict()

    for layer in range(config.encoder_n_layers):
        cell = p["params"]["hist_encoder"][f"cell_{layer}"]
        cell["ih"]["kernel"] = sd[f"hist_encoder.weight_ih_l{layer}"].numpy().T
        cell["hh"]["kernel"] = sd[f"hist_encoder.weight_hh_l{layer}"].numpy().T
        if config.encoder_bias:
            cell["ih"]["bias"] = sd[f"hist_encoder.bias_ih_l{layer}"].numpy()
            cell["hh"]["bias"] = sd[f"hist_encoder.bias_hh_l{layer}"].numpy()

    if config.recurrent:
        p["params"]["proj"]["kernel"] = sd["proj.weight"].numpy().T
        p["params"]["proj"]["bias"] = sd["proj.bias"].numpy()
    else:
        if config.h > config.input_size:
            p["params"]["upsample_sequence"]["kernel"] = sd[
                "upsample_sequence.weight"
            ].numpy().T
            p["params"]["upsample_sequence"]["bias"] = sd[
                "upsample_sequence.bias"
            ].numpy()

        pt_idx, jax_idx = 0, 0
        while f"mlp_decoder.layers.{pt_idx}.weight" in sd:
            block = p["params"]["mlp_decoder"][f"Dense_{jax_idx}"]
            block["kernel"] = sd[f"mlp_decoder.layers.{pt_idx}.weight"].numpy().T
            block["bias"] = sd[f"mlp_decoder.layers.{pt_idx}.bias"].numpy()
            pt_idx += 3  # Linear, Activation, Dropout
            jax_idx += 1

    return freeze(p)


def _build_pair(
    config: RNNConfig,
    seed: int = 42,
) -> Tuple["PyTorchRNNReference", RNN]:
    torch.manual_seed(seed)
    pt = PyTorchRNNReference(
        h=config.h,
        input_size=config.input_size,
        encoder_hidden_size=config.encoder_hidden_size,
        encoder_n_layers=config.encoder_n_layers,
        encoder_activation=config.encoder_activation,
        encoder_bias=config.encoder_bias,
        encoder_dropout=config.encoder_dropout,
        decoder_hidden_size=config.decoder_hidden_size,
        decoder_layers=config.decoder_layers,
        futr_exog_size=config.futr_exog_size,
        hist_exog_size=config.hist_exog_size,
        stat_exog_size=config.stat_exog_size,
        output_size=config.output_size,
        recurrent=config.recurrent,
    )
    pt.eval()
    return pt, RNN(config)


def _make_inputs(config: RNNConfig, batch_size: int = 2, seed: int = 0):
    rng = np.random.RandomState(seed)
    y = rng.randn(batch_size, config.input_size, 1).astype(np.float32)
    hist = (
        rng.randn(batch_size, config.input_size, config.hist_exog_size).astype(np.float32)
        if config.hist_exog_size > 0 else None
    )
    futr = (
        rng.randn(batch_size, config.input_size + config.h, config.futr_exog_size).astype(np.float32)
        if config.futr_exog_size > 0 else None
    )
    stat = (
        rng.randn(batch_size, config.stat_exog_size).astype(np.float32)
        if config.stat_exog_size > 0 else None
    )
    return y, hist, futr, stat


def _init_params(jax_model: RNN, y, hist, futr, stat, seed: int = 0):
    return jax_model.init(
        jax.random.PRNGKey(seed),
        jnp.array(y),
        hist_exog=jnp.array(hist) if hist is not None else None,
        futr_exog=jnp.array(futr) if futr is not None else None,
        stat_exog=jnp.array(stat) if stat is not None else None,
    )


# ===========================================================================
# Loss tests
# ===========================================================================


def test_loss_basic_mae_mse():
    y = jnp.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    y_hat = jnp.array([[[1.1], [2.0], [2.8]], [[3.5], [5.0], [6.2]]])
    mask = jnp.array([[[1.0], [1.0], [0.0]], [[1.0], [0.0], [1.0]]])

    mae = float(masked_mae(y, y_hat, mask))
    mse = float(masked_mse(y, y_hat, mask))

    assert mae == pytest.approx(0.2, abs=1e-6)
    assert mse == pytest.approx(0.075, abs=1e-6)


def test_loss_no_mask_matches_unmasked():
    y = jnp.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    y_hat = jnp.array([[[1.5], [1.5]], [[3.5], [3.5]]])
    expected_mae = float(jnp.mean(jnp.abs(y - y_hat)))
    expected_mse = float(jnp.mean((y - y_hat) ** 2))
    assert float(masked_mae(y, y_hat)) == pytest.approx(expected_mae, abs=1e-6)
    assert float(masked_mse(y, y_hat)) == pytest.approx(expected_mse, abs=1e-6)


def test_loss_horizon_weight():
    y = jnp.array([[[1.0], [2.0], [3.0]]])
    y_hat = jnp.array([[[2.0], [4.0], [6.0]]])
    horizon_weight = jnp.array([0.0, 0.0, 1.0])
    # Only the third position counts: |3-6| = 3.
    assert float(masked_mae(y, y_hat, horizon_weight=horizon_weight)) == pytest.approx(3.0)


def test_loss_all_masked_returns_zero():
    y = jnp.ones((2, 4, 1))
    y_hat = jnp.zeros((2, 4, 1))
    mask = jnp.zeros_like(y)
    assert float(masked_mae(y, y_hat, mask)) == 0.0
    assert float(masked_mse(y, y_hat, mask)) == 0.0


def test_loss_is_jit_compatible():
    y = jnp.ones((2, 4, 1))
    y_hat = jnp.zeros((2, 4, 1))
    mask = jnp.ones_like(y)
    jit_loss = jax.jit(masked_mae)
    out = float(jit_loss(y, y_hat, mask))
    assert out == pytest.approx(1.0, abs=1e-6)


def test_loss_grad_finite():
    """Gradient should flow and never produce NaN even with empty masks."""
    y = jnp.array([[[1.0], [2.0]]])
    mask = jnp.array([[[0.0], [0.0]]])

    def loss(y_hat):
        return masked_mse(y, y_hat, mask)

    g = jax.grad(loss)(jnp.zeros_like(y))
    assert jnp.all(jnp.isfinite(g))


# ===========================================================================
# Data pipeline tests
# ===========================================================================


def test_data_pipeline_basic():
    input_size, h, B = 10, 5, 4
    rng = np.random.RandomState(0)
    y_series = [rng.randn(15 + i) for i in range(B)]
    hist = [rng.randn(15 + i, 2) for i in range(B)]
    futr = [rng.randn(15 + i + h, 3) for i in range(B)]
    stat = [rng.randn(4) for _ in range(B)]
    sample_masks = [np.ones(h) for _ in range(B)]

    batch = create_batch(
        y_series, input_size, h,
        hist_exog_list=hist,
        futr_exog_list=futr,
        stat_exog_list=stat,
        sample_mask_list=sample_masks,
    )

    assert batch["insample_y"].shape == (B, input_size, 1)
    assert batch["outsample_y"].shape == (B, h, 1)
    assert batch["available_mask"].shape == (B, input_size, 1)
    assert batch["sample_mask"].shape == (B, h, 1)
    assert batch["hist_exog"].shape == (B, input_size, 2)
    assert batch["futr_exog"].shape == (B, input_size + h, 3)
    assert batch["stat_exog"].shape == (B, 4)
    assert (np.asarray(batch["sample_mask"]) == 1.0).all()


def test_data_pipeline_no_exog():
    input_size, h, B = 8, 3, 2
    y_series = [np.arange(20.0) for _ in range(B)]
    batch = create_batch(y_series, input_size, h)
    assert batch["hist_exog"] is None
    assert batch["futr_exog"] is None
    assert batch["stat_exog"] is None
    assert batch["insample_y"].shape == (B, input_size, 1)


def test_data_pipeline_padding():
    """A short series should be left-padded; available_mask reflects that."""
    input_size, h = 10, 3
    short_series = np.arange(5.0)  # 5 history points (after horizon split = 2)
    batch = create_batch([short_series], input_size, h)
    avail = np.asarray(batch["available_mask"])[0, :, 0]
    insample = np.asarray(batch["insample_y"])[0, :, 0]

    n_pad = input_size - 2  # we keep 2 history points
    assert avail[:n_pad].sum() == 0.0
    assert avail[n_pad:].sum() == 2.0
    assert (insample[:n_pad] == 0).all()


def test_align_covariates_pad_and_truncate():
    aligned_pad = align_covariates(
        hist_exog=np.ones((3, 2)),
        futr_exog=np.ones((4, 1)),
        stat_exog=np.ones(5),
        input_size=10,
        h=4,
    )
    assert aligned_pad["hist_exog"].shape == (10, 2)
    assert aligned_pad["futr_exog"].shape == (14, 1)
    assert aligned_pad["stat_exog"].shape == (5,)
    assert (aligned_pad["hist_exog"][:7] == 0).all()
    assert (aligned_pad["hist_exog"][7:] == 1).all()

    aligned_trunc = align_covariates(
        hist_exog=np.arange(20).reshape(20, 1).astype(np.float32),
        futr_exog=None,
        stat_exog=None,
        input_size=5,
        h=2,
    )
    assert aligned_trunc["hist_exog"].shape == (5, 1)
    assert aligned_trunc["hist_exog"].ravel().tolist() == [15, 16, 17, 18, 19]


def test_pad_sequence_left_pads():
    padded, mask = pad_sequence(np.array([1.0, 2.0, 3.0]), 5)
    assert padded.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


# ===========================================================================
# Forward-pass parity tests
# ===========================================================================


_PARITY_CONFIGS = [
    # description,                config kwargs
    ("direct_no_exog", dict(h=3, input_size=10, encoder_hidden_size=8,
                            encoder_n_layers=2, decoder_hidden_size=8,
                            decoder_layers=2, recurrent=False)),
    ("recurrent_no_exog", dict(h=3, input_size=10, encoder_hidden_size=8,
                               encoder_n_layers=2, recurrent=True)),
    ("direct_all_exog", dict(h=3, input_size=10, encoder_hidden_size=8,
                             encoder_n_layers=2, decoder_hidden_size=8,
                             decoder_layers=2, hist_exog_size=2,
                             futr_exog_size=3, stat_exog_size=4,
                             recurrent=False)),
    ("recurrent_all_exog", dict(h=3, input_size=10, encoder_hidden_size=8,
                                encoder_n_layers=2, hist_exog_size=2,
                                futr_exog_size=3, stat_exog_size=4,
                                recurrent=True)),
    ("direct_upsample_h_gt_L", dict(h=20, input_size=10, encoder_hidden_size=8,
                                    encoder_n_layers=2, decoder_hidden_size=8,
                                    decoder_layers=2, recurrent=False)),
    ("relu_activation", dict(h=3, input_size=10, encoder_hidden_size=8,
                             encoder_n_layers=2, encoder_activation="relu",
                             decoder_hidden_size=8, decoder_layers=2,
                             recurrent=False)),
    ("no_bias_single_layer_decoder", dict(h=3, input_size=10, encoder_hidden_size=8,
                                          encoder_n_layers=1, encoder_bias=False,
                                          decoder_layers=1, recurrent=False)),
]


@pytest.mark.parametrize("name,kwargs", _PARITY_CONFIGS, ids=[c[0] for c in _PARITY_CONFIGS])
def test_forward_pass_parity(name, kwargs):
    cfg = RNNConfig(**kwargs)
    pt, jax_m = _build_pair(cfg)
    y, hist, futr, stat = _make_inputs(cfg)

    params = _init_params(jax_m, y, hist, futr, stat)
    params = _transfer_weights(pt, params, cfg)

    with torch.no_grad():
        pt_out = pt({
            "insample_y": torch.tensor(y),
            "futr_exog": torch.tensor(futr) if futr is not None else None,
            "hist_exog": torch.tensor(hist) if hist is not None else None,
            "stat_exog": torch.tensor(stat) if stat is not None else None,
        }).numpy()

    jax_out, _ = jax_m.apply(
        params,
        jnp.array(y),
        hist_exog=jnp.array(hist) if hist is not None else None,
        futr_exog=jnp.array(futr) if futr is not None else None,
        stat_exog=jnp.array(stat) if stat is not None else None,
        deterministic=True,
    )
    np.testing.assert_allclose(pt_out, np.array(jax_out), atol=1e-4, rtol=1e-4)


# ===========================================================================
# Multi-step / autoregressive parity
# ===========================================================================


def test_autoregressive_parity():
    """Our autoregressive_predict matches a PT autoregressive loop step-by-step."""
    cfg = RNNConfig(
        h=4, input_size=10, encoder_hidden_size=8, encoder_n_layers=1, recurrent=True
    )
    pt, jax_m = _build_pair(cfg)
    y, _, _, _ = _make_inputs(cfg)

    params = _init_params(jax_m, y, None, None, None)
    params = _transfer_weights(pt, params, cfg)

    # PT autoregressive loop
    pt.h = 1
    pt.maintain_state = True
    pt.rnn_state = None
    out_seq = []
    with torch.no_grad():
        cur = pt({
            "insample_y": torch.tensor(y),
            "futr_exog": None, "hist_exog": None, "stat_exog": None,
        })
        out_seq.append(cur.clone())
        for _t in range(1, cfg.h):
            cur = pt({
                "insample_y": cur,
                "futr_exog": None, "hist_exog": None, "stat_exog": None,
            })
            out_seq.append(cur.clone())
    pt.maintain_state = False
    pt.h = cfg.h
    pt_out = torch.cat(out_seq, dim=1).numpy()

    jax_out = np.array(autoregressive_predict(jax_m, params, jnp.array(y), h=cfg.h))
    np.testing.assert_allclose(pt_out, jax_out, atol=1e-4, rtol=1e-4)


def test_autoregressive_with_futr_exog():
    cfg = RNNConfig(
        h=3, input_size=10, encoder_hidden_size=8, encoder_n_layers=1,
        futr_exog_size=2, recurrent=True,
    )
    pt, jax_m = _build_pair(cfg)
    y, _, futr, _ = _make_inputs(cfg)

    params = _init_params(jax_m, y, None, futr, None)
    params = _transfer_weights(pt, params, cfg)

    pt.h = 1
    pt.maintain_state = True
    pt.rnn_state = None
    L = cfg.input_size
    out_seq = []
    with torch.no_grad():
        cur = pt({
            "insample_y": torch.tensor(y),
            "futr_exog": torch.tensor(futr[:, :L]),
            "hist_exog": None, "stat_exog": None,
        })
        out_seq.append(cur.clone())
        for t in range(1, cfg.h):
            cur = pt({
                "insample_y": cur,
                "futr_exog": torch.tensor(futr[:, L + t - 1: L + t]),
                "hist_exog": None, "stat_exog": None,
            })
            out_seq.append(cur.clone())
    pt.maintain_state = False
    pt.h = cfg.h
    pt_out = torch.cat(out_seq, dim=1).numpy()

    jax_out = np.array(autoregressive_predict(
        jax_m, params, jnp.array(y), h=cfg.h, futr_exog=jnp.array(futr)
    ))
    np.testing.assert_allclose(pt_out, jax_out, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Masking interaction
# ===========================================================================


def test_loss_under_random_mask_matches_manual_average():
    """Compute MSE on random predictions with mask, sanity-check vs numpy."""
    rng = np.random.RandomState(7)
    y = rng.randn(3, 6, 1).astype(np.float32)
    y_hat = rng.randn(3, 6, 1).astype(np.float32)
    mask = rng.randint(0, 2, size=(3, 6, 1)).astype(np.float32)

    j_loss = float(masked_mse(jnp.array(y), jnp.array(y_hat), jnp.array(mask)))
    diff_sq = (y - y_hat) ** 2
    expected = (diff_sq * mask).sum() / max(mask.sum(), 1.0)
    assert j_loss == pytest.approx(float(expected), abs=1e-5)


# ===========================================================================
# Train/eval steps
# ===========================================================================


def test_train_and_eval_step_run():
    cfg = RNNConfig(h=3, input_size=8, encoder_hidden_size=8, encoder_n_layers=1)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, cfg, learning_rate=1e-2)

    rng_step = jax.random.PRNGKey(1)
    rng_data = np.random.RandomState(0)
    y_series = [rng_data.randn(20) for _ in range(4)]
    batch = create_batch(y_series, cfg.input_size, cfg.h)

    new_state, loss, preds = train_step(state, batch, rng_step)
    assert preds.shape == (4, cfg.h, cfg.output_size)
    assert jnp.isfinite(loss)
    # Params should change after a step.
    leaves_before = jax.tree_util.tree_leaves(state.params)
    leaves_after = jax.tree_util.tree_leaves(new_state.params)
    diffs = [
        float(jnp.abs(a - b).max()) for a, b in zip(leaves_after, leaves_before)
    ]
    assert max(diffs) > 0.0

    eval_loss, eval_preds = eval_step(new_state, batch)
    assert eval_preds.shape == (4, cfg.h, cfg.output_size)
    assert jnp.isfinite(eval_loss)


def test_training_decreases_loss_on_constant_signal():
    """Optimiser actually reduces loss on an easy synthetic task."""
    cfg = RNNConfig(h=3, input_size=6, encoder_hidden_size=8, encoder_n_layers=1)
    state = create_train_state(jax.random.PRNGKey(0), cfg, learning_rate=1e-1)

    # Constant series — model should learn the mean quickly.
    y_series = [np.ones(20, dtype=np.float32) * 3.0 for _ in range(8)]
    batch = create_batch(y_series, cfg.input_size, cfg.h)

    rng = jax.random.PRNGKey(123)
    losses = []
    for _ in range(50):
        rng, sub = jax.random.split(rng)
        state, loss, _ = train_step(state, batch, sub)
        losses.append(float(loss))

    assert losses[-1] < losses[0] * 0.5, f"expected loss to drop, got {losses[0]} -> {losses[-1]}"


# ===========================================================================
# RNNForecaster (high-level fit/forecast adapter)
# ===========================================================================


def test_forecaster_fit_predict_runs_univariate():
    """RNNForecaster.fit_predict on a single seasonal series returns finite [h]."""
    from chronax.models.rnn.forecaster import RNNForecaster

    fc = RNNForecaster(
        h=12, input_size=24, encoder_hidden_size=8, encoder_n_layers=1,
        recurrent=True,
        max_steps=30, learning_rate=1e-2, batch_size=4, random_seed=0,
    )
    t = np.arange(120, dtype=np.float32)
    y = np.sin(2 * np.pi * t / 12) + 0.1 * np.random.RandomState(0).randn(120)

    preds = fc.fit_predict(y.astype(np.float32))
    assert preds.shape == (fc.config.h,)
    assert np.all(np.isfinite(preds))


def test_forecaster_fit_predict_runs_panel():
    """RNNForecaster on a panel of 3 series returns [n_series, h] preds."""
    from chronax.models.rnn.forecaster import RNNForecaster

    rng = np.random.RandomState(1)
    series = [
        np.sin(2 * np.pi * np.arange(80) / 12).astype(np.float32) + 0.05 * rng.randn(80)
        for _ in range(3)
    ]
    fc = RNNForecaster(
        h=6, input_size=18, encoder_hidden_size=8, encoder_n_layers=1,
        recurrent=True,
        max_steps=30, learning_rate=1e-2, batch_size=3, random_seed=0,
    )
    preds = fc.fit_predict(series)
    assert preds.shape == (3, fc.config.h)
    assert np.all(np.isfinite(preds))


def test_forecaster_direct_decoder_runs():
    """Direct (non-recurrent) MLP-decoder path also produces finite predictions."""
    from chronax.models.rnn.forecaster import RNNForecaster

    y = np.sin(np.linspace(0, 8 * np.pi, 60)).astype(np.float32)
    fc = RNNForecaster(
        h=4, input_size=12, encoder_hidden_size=8, encoder_n_layers=1,
        decoder_hidden_size=8, decoder_layers=2, recurrent=False,
        max_steps=20, learning_rate=1e-2, batch_size=2, random_seed=0,
    )
    preds = fc.fit_predict(y)
    assert preds.shape == (fc.config.h,)
    assert np.all(np.isfinite(preds))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
