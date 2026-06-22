"""Forward-parity gate: NF KAN (torch) vs Chronax KANNet (JAX) at identical weights.

Usage:
    .venv/bin/python benchmarks/kan_weight_parity.py

Dumps NF KAN's KANLinear weights + grid + a fixed input/output, transplants them into
KANNet (pure copy — both torch-layout), and asserts forward max|delta| < 1e-4. This
pins b_splines, the base+spline product, and the flatten order.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.kan.kan_module import KANNet

REPO = Path(__file__).resolve().parents[1]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"
DUMP = REPO / "benchmarks" / "benchmark_results" / "_kan_parity_dump.npz"
CFG = dict(h=8, input_size=16, grid_size=5, spline_order=3, n_hidden_layers=1, hidden_size=12)

_TORCH = f'''
import numpy as np, torch
from neuralforecast.models.kan import KAN
cfg = {CFG!r}
torch.manual_seed(0)
m = KAN(h=cfg["h"], input_size=cfg["input_size"], grid_size=cfg["grid_size"],
        spline_order=cfg["spline_order"], n_hidden_layers=cfg["n_hidden_layers"],
        hidden_size=cfg["hidden_size"], scaler_type="identity", max_steps=1)
m.eval()
rng = np.random.RandomState(0)
x = rng.randn(4, cfg["input_size"]).astype("float32")          # [B, L]
wb = {{"insample_y": torch.from_numpy(x)[:, :, None], "futr_exog": None, "hist_exog": None, "stat_exog": None}}
with torch.no_grad():
    y = m(wb).numpy()                                          # [B, h, 1]
sd = {{k: v.detach().numpy() for k, v in m.state_dict().items()}}
np.savez({str(DUMP)!r}, x=x, y=y, **{{f"w::{{k}}": v for k, v in sd.items()}})
print("DUMP_OK", len(sd))
'''


def main():
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.unlink(missing_ok=True)
    out = subprocess.run([str(NF_VENV_PY), "-c", _TORCH], capture_output=True, text=True)
    if out.returncode != 0 or "DUMP_OK" not in out.stdout:
        sys.exit(f"torch side failed (rc={out.returncode}):\n{out.stdout}\n{out.stderr}")
    data = np.load(DUMP)
    x, y_ref = data["x"], data["y"]
    W = {k[len("w::"):]: data[k] for k in data.files if k.startswith("w::")}

    net = KANNet(h=CFG["h"], input_size=CFG["input_size"], n_hidden_layers=CFG["n_hidden_layers"],
                 hidden_size=CFG["hidden_size"], grid_size=CFG["grid_size"], spline_order=CFG["spline_order"],
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True,
                 grid_range=(-1.0, 1.0), rngs=nnx.Rngs(0))
    used = set()

    def take(k):
        used.add(k)
        return jnp.asarray(W[k])

    for i, layer in enumerate(net.layers):
        p = f"layers.{i}"
        layer.base_weight.value = take(f"{p}.base_weight")        # [out, in] — no transpose
        layer.spline_weight.value = take(f"{p}.spline_weight")
        layer.spline_scaler.value = take(f"{p}.spline_scaler")
        layer.grid.value = take(f"{p}.grid")

    # Only the KANLinear weights (layers.*) must be transplanted; ignore any
    # BaseModel buffers (loss/scaler/padder) that carry no KANNet counterpart.
    leftover = [k for k in W if k.startswith("layers.") and k not in used]
    assert not leftover, f"unmapped KANLinear keys: {leftover}"

    y_jax = np.asarray(net(jnp.asarray(x)[:, :, None]))          # [B, h, 1]
    max_diff = float(np.max(np.abs(y_jax - y_ref)))
    print(f"max|Δ| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"forward parity failed: {max_diff:.2e}"
    print("PARITY_OK")


if __name__ == "__main__":
    main()
