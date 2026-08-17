"""Forward-parity gate: NF DilatedRNN (torch) vs Chronax DilatedRNNNet (JAX) at identical weights.

Usage:
    .venv/bin/python benchmarks/dilated_rnn_weight_parity.py

Builds ``neuralforecast.models.DilatedRNN`` in ``benchmarks/.venv-nf``, runs one
fixed input window through its ``forward`` in eval mode, and dumps
(input, output, named weights) to an ``.npz``. Then builds Chronax
``DilatedRNNNet`` at the same config, copies the torch weights into the Flax
params (transposing Linear kernels: torch ``[out, in]`` -> Flax ``[in, out]``),
runs the same window, and asserts ``max|delta| < 1e-4``.

This isolates architecture-port bugs from optimization noise — in particular it
pins the two places this port departs from the reference in FORM:

  * the dilation pack/unpack, rewritten from NF's per-rate Python list
    comprehensions into pure reshapes, and
  * the recurrent cells, written out by hand (torch's GRU applies its reset gate
    after the hidden bias; torch's LSTM uses (i, f, g, o) gate order) rather than
    delegating to ``flax.nnx``'s fused-bias cells.

Every cell type is checked, since each has its own gate algebra. The scaler is
NOT exercised here: NF applies it in ``BaseModel``, outside ``forward``, so this
gate covers the backbone only.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.dilated_rnn.dilated_rnn_module import DilatedRNNNet

REPO = Path(__file__).resolve().parents[1]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"
DUMP = REPO / "benchmarks" / "benchmark_results" / "_dilated_rnn_parity_dump.npz"

CFG = dict(h=8, input_size=24, encoder_hidden_size=12, decoder_hidden_size=10,
           decoder_layers=2, dilations=[[1, 2], [3]])

TOL = 1e-4

_TORCH_SIDE = '''
import numpy as np, torch
from neuralforecast.models import DilatedRNN
cfg = {cfg!r}
cell_type = {cell!r}
torch.manual_seed(0)
m = DilatedRNN(h=cfg["h"], input_size=cfg["input_size"], cell_type=cell_type,
               dilations=cfg["dilations"],
               encoder_hidden_size=cfg["encoder_hidden_size"],
               decoder_hidden_size=cfg["decoder_hidden_size"],
               decoder_layers=cfg["decoder_layers"],
               max_steps=1)
m.eval()
rng = np.random.RandomState(0)
x = rng.randn(4, cfg["input_size"], 1).astype("float32")   # [B, L, 1]
batch = {{"insample_y": torch.from_numpy(x), "futr_exog": None,
          "hist_exog": None, "stat_exog": None}}
with torch.no_grad():
    y = m(batch).numpy()                                    # [B, h, 1]
sd = {{k: v.detach().numpy() for k, v in m.state_dict().items()}}
np.savez({dump!r}, x=x, y=y, **{{f"w::{{k}}": v for k, v in sd.items()}})
print("DUMP_OK", len(sd), "tensors")
'''


class _NoReference(Exception):
    """Raised when neuralforecast cannot run a cell type, so parity is undefined."""


def _t(w: np.ndarray) -> jnp.ndarray:
    """torch Linear weight [out, in] -> Flax kernel [in, out]."""
    return jnp.asarray(w.T)


def _load_cell(cell, prefix: str, W: dict, cell_type: str) -> None:
    """Copy one torch recurrent cell's parameters into a Chronax cell."""
    if cell_type in ("GRU", "RNN", "LSTM"):
        # torch's built-in nn.GRU / nn.RNN / nn.LSTM, one layer.
        cell.ih.kernel.value = _t(W[f"{prefix}.weight_ih_l0"])
        cell.ih.bias.value = jnp.asarray(W[f"{prefix}.bias_ih_l0"])
        cell.hh.kernel.value = _t(W[f"{prefix}.weight_hh_l0"])
        cell.hh.bias.value = jnp.asarray(W[f"{prefix}.bias_hh_l0"])
        return
    if cell_type == "ResLSTM":
        c = f"{prefix}.cell"
        cell.ii.kernel.value = _t(W[f"{c}.weight_ii"])
        cell.ii.bias.value = jnp.asarray(W[f"{c}.bias_ii"])
        cell.ih.kernel.value = _t(W[f"{c}.weight_ih"])
        cell.ih.bias.value = jnp.asarray(W[f"{c}.bias_ih"])
        cell.ic.kernel.value = _t(W[f"{c}.weight_ic"])
        cell.ic.bias.value = jnp.asarray(W[f"{c}.bias_ic"])
        cell.hh.kernel.value = _t(W[f"{c}.weight_hh"])
        cell.hh.bias.value = jnp.asarray(W[f"{c}.bias_hh"])
        cell.ir.kernel.value = _t(W[f"{c}.weight_ir"])
        return
    if cell_type == "AttentiveLSTM":
        c = f"{prefix}.cell"
        cell.cell.ih.kernel.value = _t(W[f"{c}.weight_ih"])
        cell.cell.ih.bias.value = jnp.asarray(W[f"{c}.bias_ih"])
        cell.cell.hh.kernel.value = _t(W[f"{c}.weight_hh"])
        cell.cell.hh.bias.value = jnp.asarray(W[f"{c}.bias_hh"])
        cell.attn_in.kernel.value = _t(W[f"{prefix}.attn_layer.0.weight"])
        cell.attn_in.bias.value = jnp.asarray(W[f"{prefix}.attn_layer.0.bias"])
        cell.attn_out.kernel.value = _t(W[f"{prefix}.attn_layer.2.weight"])
        cell.attn_out.bias.value = jnp.asarray(W[f"{prefix}.attn_layer.2.bias"])
        return
    raise ValueError(f"unhandled cell_type {cell_type!r}")


def _load_into(net: DilatedRNNNet, W: dict, cell_type: str) -> None:
    for g, group in enumerate(net.rnn_stack):
        for i, cell in enumerate(group.cells):
            _load_cell(cell, f"rnn_stack.{g}.cells.{i}", W, cell_type)
    net.context_adapter.kernel.value = _t(W["context_adapter.weight"])
    net.context_adapter.bias.value = jnp.asarray(W["context_adapter.bias"])
    # NF's MLP is an nn.Sequential: Linear, ReLU, Dropout, Linear, ... so the
    # Linear layers sit at indices 0, 3, 6, ... and the final one right after.
    torch_idx = [int(k.split(".")[2]) for k in W
                 if k.startswith("mlp_decoder.layers.") and k.endswith(".weight")]
    for chx_layer, ti in zip(net.mlp_decoder.layers, sorted(torch_idx)):
        chx_layer.kernel.value = _t(W[f"mlp_decoder.layers.{ti}.weight"])
        chx_layer.bias.value = jnp.asarray(W[f"mlp_decoder.layers.{ti}.bias"])


def check(cell_type: str) -> tuple[bool, float]:
    code = _TORCH_SIDE.format(cfg=CFG, cell=cell_type, dump=str(DUMP))
    out = subprocess.run([str(NF_VENV_PY), "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        if cell_type == "AttentiveLSTM" and "got tuple" in out.stderr:
            # Upstream defect, neuralforecast 3.2.0: AttentiveLSTMLayer.forward
            # does `inputs = inputs.unbind(0)` (a tuple) and then uses `inputs` as
            # a tensor in `torch.cat(...)` and `inputs.permute(...)`. The cell type
            # cannot run in NF at all, so there is no reference to compare against.
            # The Chronax port implements the evident intent (attend over the whole
            # window) and is covered by tests/test_dilated_rnn.py instead.
            raise _NoReference(cell_type)
        raise SystemExit(f"torch side failed for {cell_type}:\n{out.stderr}")
    d = np.load(DUMP)
    W = {k[3:]: d[k] for k in d.files if k.startswith("w::")}
    net = DilatedRNNNet(
        h=CFG["h"], input_size=CFG["input_size"], in_features=1, cell_type=cell_type,
        dilations=CFG["dilations"], encoder_hidden_size=CFG["encoder_hidden_size"],
        decoder_hidden_size=CFG["decoder_hidden_size"],
        decoder_layers=CFG["decoder_layers"], rngs=nnx.Rngs(0),
    )
    _load_into(net, W, cell_type)
    y_chx = np.asarray(net(jnp.asarray(d["x"]), deterministic=True))
    delta = float(np.max(np.abs(y_chx - d["y"])))
    return delta < TOL, delta


def main() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    failures, skipped = [], []
    for cell_type in ("LSTM", "GRU", "RNN", "ResLSTM", "AttentiveLSTM"):
        try:
            ok, delta = check(cell_type)
        except _NoReference:
            skipped.append(cell_type)
            print(f"[SKIP] {cell_type:14s} neuralforecast 3.2.0 cannot run this cell "
                  f"type (upstream bug in AttentiveLSTMLayer.forward: `inputs` is "
                  f"unbound to a tuple, then used as a tensor) — no reference to "
                  f"compare against")
            continue
        print(f"[{'PASS' if ok else 'FAIL'}] {cell_type:14s} max|delta| = {delta:.3e}")
        if not ok:
            failures.append(cell_type)
    if failures:
        raise SystemExit(f"forward parity FAILED for: {', '.join(failures)}")
    covered = ", ".join(c for c in ("LSTM", "GRU", "RNN", "ResLSTM", "AttentiveLSTM")
                        if c not in skipped)
    print(f"\nWithin {TOL:g} for: {covered}. Backbone reproduces neuralforecast exactly.")
    if skipped:
        print(f"No reference available for: {', '.join(skipped)}.")


if __name__ == "__main__":
    main()
