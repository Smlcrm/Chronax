"""Forward-parity gate: NF PatchTST (torch) vs Chronax PatchTSTNet (JAX) at identical weights.

Usage:
    .venv/bin/python benchmarks/patchtst_weight_parity.py

Builds NF's ``PatchTST_backbone`` in ``benchmarks/.venv-nf``, runs one fixed input
window through it in eval mode, and dumps (input, output, named weights) to an
``.npz``. Then builds Chronax ``PatchTSTNet`` at the same config, copies the torch
weights into the Flax params (transposing Linear kernels: torch ``[out, in]`` ->
Flax ``[in, out]``), runs the same window, and asserts ``max|Δ| < 1e-4``.

This is acceptance gate #5 — it proves the JAX port reproduces neuralforecast's
PatchTST exactly, isolating architecture-port bugs from optimization noise.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.patchtst.patchtst_module import PatchTSTNet

REPO = Path(__file__).resolve().parents[1]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"
DUMP = REPO / "benchmarks" / "benchmark_results" / "_patchtst_parity_dump.npz"

CFG = dict(h=8, input_size=32, patch_len=16, stride=8, hidden_size=16, n_heads=2,
           encoder_layers=2, linear_hidden_size=32)

_TORCH_SIDE = f'''
import numpy as np, torch
from neuralforecast.models.patchtst import PatchTST_backbone
cfg = {CFG!r}
torch.manual_seed(0)
m = PatchTST_backbone(
    c_in=1, c_out=1, input_size=cfg["input_size"], h=cfg["h"],
    patch_len=cfg["patch_len"], stride=cfg["stride"], n_layers=cfg["encoder_layers"],
    hidden_size=cfg["hidden_size"], n_heads=cfg["n_heads"],
    linear_hidden_size=cfg["linear_hidden_size"], norm="BatchNorm",
    attn_dropout=0.0, dropout=0.0, act="gelu", res_attention=True, pre_norm=False,
    pe="zeros", learn_pe=True, fc_dropout=0.0, head_dropout=0.0, padding_patch="end",
    pretrain_head=False, head_type="flatten", individual=False, revin=True,
    affine=False, subtract_last=True,
)
m.eval()
rng = np.random.RandomState(0)
x = rng.randn(4, 1, cfg["input_size"]).astype("float32")   # [B, c_in, L]
with torch.no_grad():
    y = m(torch.from_numpy(x)).numpy()                     # [B, c_in, h]
sd = {{k: v.detach().numpy() for k, v in m.state_dict().items()}}
np.savez({str(DUMP)!r}, x=x, y=y, **{{f"w::{{k}}": v for k, v in sd.items()}})
print("DUMP_OK", len(sd), "tensors")
'''


def _load_into(net, W):
    """Copy torch weights ``W`` (flat dict) into the Flax ``net``; return consumed keys."""
    used = set()

    def take(k):
        used.add(k)
        return jnp.asarray(W[k])

    def lin(dst, prefix):  # torch Linear [out,in] -> Flax kernel [in,out]
        dst.kernel.value = take(prefix + ".weight").T
        dst.bias.value = take(prefix + ".bias")

    def bn(dst, prefix):
        dst.scale.value = take(prefix + ".weight")
        dst.bias.value = take(prefix + ".bias")
        dst.mean.value = take(prefix + ".running_mean")
        dst.var.value = take(prefix + ".running_var")

    lin(net.embedding.proj, "backbone.W_P")
    net.embedding.pos.value = take("backbone.W_pos")       # not a Linear; no transpose
    for i, layer in enumerate(net.encoder.layers):
        p = f"backbone.encoder.layers.{i}"
        lin(layer.attn.w_q, p + ".self_attn.W_Q")
        lin(layer.attn.w_k, p + ".self_attn.W_K")
        lin(layer.attn.w_v, p + ".self_attn.W_V")
        lin(layer.attn.w_o, p + ".self_attn.to_out.0")
        lin(layer.ff1, p + ".ff.0")
        lin(layer.ff2, p + ".ff.3")
        bn(layer.norm_attn, p + ".norm_attn.1")
        bn(layer.norm_ffn, p + ".norm_ffn.1")
    lin(net.head.linear, "head.linear")                    # pure transpose (FlattenHead is hidden-major)
    return used


def main() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.unlink(missing_ok=True)   # never read a stale dump if the torch side crashes
    out = subprocess.run([str(NF_VENV_PY), "-c", _TORCH_SIDE], capture_output=True, text=True)
    if out.returncode != 0 or "DUMP_OK" not in out.stdout:
        sys.exit(f"torch side failed (rc={out.returncode}):\nSTDOUT {out.stdout}\nSTDERR {out.stderr}")
    print(out.stdout.strip())

    data = np.load(DUMP)
    x = data["x"]            # [B, 1, L]
    y_ref = data["y"]        # [B, 1, h]
    W = {k[len("w::"):]: data[k] for k in data.files if k.startswith("w::")}

    net = PatchTSTNet(
        h=CFG["h"], input_size=CFG["input_size"], patch_len=CFG["patch_len"],
        stride=CFG["stride"], hidden_size=CFG["hidden_size"], n_heads=CFG["n_heads"],
        encoder_layers=CFG["encoder_layers"], linear_hidden_size=CFG["linear_hidden_size"],
        dropout=0.0, fc_dropout=0.0, head_dropout=0.0, attn_dropout=0.0,
        revin=True, revin_affine=False, revin_subtract_last=True, rngs=nnx.Rngs(0),
    )
    used = _load_into(net, W)

    # every torch key must be consumed or explicitly skippable
    skip = ("num_batches_tracked", "sdp_attn.scale")
    leftover = [k for k in W if k not in used and not k.endswith(skip)]
    assert not leftover, f"unmapped torch keys: {leftover}"

    x_jax = jnp.asarray(x).transpose(0, 2, 1)            # [B, L, 1]
    y_jax = np.asarray(net(x_jax, deterministic=True, use_running_average=True))[:, :, 0]
    max_diff = float(np.max(np.abs(y_jax - y_ref[:, 0, :])))
    print(f"max|Δ| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"forward parity failed: max|Δ|={max_diff:.2e}"
    print("PARITY_OK")


if __name__ == "__main__":
    main()
