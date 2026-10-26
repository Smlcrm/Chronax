"""Forward-parity gate: NF SOFTS (torch) vs Chronax SOFTSNet (JAX) at identical weights.

Usage:
    .venv/bin/python benchmarks/softs_weight_parity.py

Builds ``neuralforecast.models.SOFTS`` in ``benchmarks/.venv-nf``, runs one
fixed input window through its ``forward`` in EVAL mode, and dumps
(input, output, named weights) to an ``.npz``. Then builds Chronax
``SOFTSNet`` at the same config, copies the torch weights into the Flax
params (transposing Linear kernels: torch ``[out, in]`` -> Flax ``[in, out]``),
runs the same window, and asserts ``max|delta| < 1e-4``.

Eval mode is what makes this comparison well defined: it is the branch where
STAD is deterministic — the core is the softmax-weighted mean rather than a
multinomial sample, and dropout is off. The train-mode branch is stochastic by
construction and cannot be compared pointwise across RNG implementations;
tests/test_softs.py covers it instead.

This gate is what caught the extra final encoder LayerNorm the port carried
since PR #95: NF's TransEncoder applies norm_layer only when one is passed, and
SOFTS passes none, so NF's state_dict has no ``encoder.norm.*`` to copy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.softs.softs_module import SOFTSNet

REPO = Path(__file__).resolve().parents[1]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"
DUMP = REPO / "benchmarks" / "benchmark_results" / "_softs_parity_dump.npz"

CFG = dict(h=8, input_size=24, hidden_size=16, d_core=8, e_layers=2, d_ff=32)

TOL = 1e-4

_TORCH_SIDE = f'''
import numpy as np, torch
from neuralforecast.models import SOFTS
cfg = {CFG!r}
torch.manual_seed(0)
m = SOFTS(h=cfg["h"], input_size=cfg["input_size"], n_series=1,
               hidden_size=cfg["hidden_size"], d_core=cfg["d_core"],
               e_layers=cfg["e_layers"], d_ff=cfg["d_ff"], dropout=0.0,
               use_norm=True, max_steps=1)
m.eval()
rng = np.random.RandomState(0)
x = rng.randn(4, cfg["input_size"], 1).astype("float32")   # [B, L, n_series]
with torch.no_grad():
    y = m({{"insample_y": torch.from_numpy(x)}}).numpy()     # [B, h, n_series]
sd = {{k: v.detach().numpy() for k, v in m.state_dict().items()}}
np.savez({str(DUMP)!r}, x=x, y=y, **{{f"w::{{k}}": v for k, v in sd.items()}})
print("DUMP_OK", len(sd), "tensors")
'''


def _t(w: np.ndarray) -> jnp.ndarray:
    """torch Linear weight [out, in] -> Flax kernel [in, out]."""
    return jnp.asarray(w.T)


def _linear(dst, W: dict, prefix: str) -> None:
    dst.kernel.value = _t(W[f"{prefix}.weight"])
    dst.bias.value = jnp.asarray(W[f"{prefix}.bias"])


def _assert_no_encoder_norm(net, W: dict) -> None:
    """Guard the final-LayerNorm divergence structurally, not numerically.

    A forward-delta check CANNOT catch this one: a re-added LayerNorm is
    initialized scale=1/bias=0 and every encoder layer already ends in `norm2`,
    so re-normalizing an already-normalized vector is near-identity and the
    output delta stays inside the tolerance (measured: 1.07e-06, a clean PASS).
    What actually distinguishes the two architectures is the PARAMETER SET -- the
    extra norm's learnable scale/bias -- so assert on that instead: NF's
    state_dict must carry no `encoder.norm.*`, and neither may the port.
    """
    nf_keys = [k for k in W if k.startswith("encoder.norm.")]
    assert not nf_keys, f"reference unexpectedly HAS a final encoder norm: {nf_keys}"
    assert not hasattr(net.encoder, "norm"), (
        "port has a final encoder LayerNorm the reference does not: NF builds "
        "TransEncoder with no norm_layer, so the extra learnable scale/bias are "
        "parameters the reference lacks."
    )


def _load_into(net: SOFTSNet, W: dict) -> None:
    _linear(net.enc_embedding.value_embedding, W, "enc_embedding.value_embedding")
    for i, layer in enumerate(net.encoder.layers):
        p = f"encoder.attn_layers.{i}"
        for gen in ("gen1", "gen2", "gen3", "gen4"):
            _linear(getattr(layer.stad, gen), W, f"{p}.attention.{gen}")
        # NF's conv1/conv2 are Conv1d(kernel_size=1) — weights are [out, in, 1],
        # mathematically a pointwise Linear, so squeeze the trailing kernel axis.
        layer.conv1.kernel.value = jnp.asarray(W[f"{p}.conv1.weight"][:, :, 0].T)
        layer.conv1.bias.value = jnp.asarray(W[f"{p}.conv1.bias"])
        layer.conv2.kernel.value = jnp.asarray(W[f"{p}.conv2.weight"][:, :, 0].T)
        layer.conv2.bias.value = jnp.asarray(W[f"{p}.conv2.bias"])
        layer.norm1.scale.value = jnp.asarray(W[f"{p}.norm1.weight"])
        layer.norm1.bias.value = jnp.asarray(W[f"{p}.norm1.bias"])
        layer.norm2.scale.value = jnp.asarray(W[f"{p}.norm2.weight"])
        layer.norm2.bias.value = jnp.asarray(W[f"{p}.norm2.bias"])
    # NF builds TransEncoder with no norm_layer, so there is deliberately no
    # final encoder LayerNorm to copy here (see TransEncoder's docstring).
    _assert_no_encoder_norm(net, W)
    net.projector.kernel.value = _t(W["projection.weight"])
    net.projector.bias.value = jnp.asarray(W["projection.bias"])


def main() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run([str(NF_VENV_PY), "-c", _TORCH_SIDE],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"torch side failed:\n{out.stderr}")
    d = np.load(DUMP)
    W = {k[3:]: d[k] for k in d.files if k.startswith("w::")}
    net = SOFTSNet(
        h=CFG["h"], input_size=CFG["input_size"], hidden_size=CFG["hidden_size"],
        d_core=CFG["d_core"], e_layers=CFG["e_layers"], d_ff=CFG["d_ff"],
        dropout=0.0, use_norm=True, rngs=nnx.Rngs(0),
    )
    _load_into(net, W)
    y_chx = np.asarray(net(jnp.asarray(d["x"]), deterministic=True))
    delta = float(np.max(np.abs(y_chx - d["y"])))
    print(f"[{'PASS' if delta < TOL else 'FAIL'}] SOFTS eval forward  "
          f"max|delta| = {delta:.3e}")
    if delta >= TOL:
        raise SystemExit("forward parity FAILED")
    print(f"\nWithin {TOL:g}. Backbone reproduces neuralforecast exactly "
          f"(eval branch: softmax-weighted core).")


if __name__ == "__main__":
    main()
