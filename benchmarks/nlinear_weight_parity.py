"""Forward-parity gate: NF NLinear (torch) vs Chronax NLinearNet (JAX) at identical weights.

Usage:
    .venv/bin/python benchmarks/nlinear_weight_parity.py

Dumps NF NLinear's linear weight/bias + a fixed input/output, transplants them into
NLinearNet (pure copy — both torch-layout), and asserts forward max|delta| < 1e-4.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from chronax.models.nlinear.nlinear_module import NLinearNet

REPO = Path(__file__).resolve().parents[1]
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"
DUMP = REPO / "benchmarks" / "benchmark_results" / "_nlinear_parity_dump.npz"
CFG = dict(h=8, input_size=16)

_TORCH = f'''
import numpy as np, torch
from neuralforecast.models import NLinear
cfg = {CFG!r}
torch.manual_seed(0)
m = NLinear(h=cfg["h"], input_size=cfg["input_size"], scaler_type="identity", max_steps=1)
m.eval()
rng = np.random.RandomState(0)
x = rng.randn(4, cfg["input_size"]).astype("float32")          # [B, L]
wb = {{"insample_y": torch.from_numpy(x)[:, :, None], "futr_exog": None, "hist_exog": None, "stat_exog": None}}
with torch.no_grad():
    y = m(wb).numpy()                                          # [B, h, 1]
sd = {{k: v.detach().numpy() for k, v in m.state_dict().items() if k.startswith("linear.")}}
np.savez({str(DUMP)!r}, x=x, y=y, **{{f"w::{{k}}": v for k, v in sd.items()}})
print("DUMP_OK", len(sd))
'''


def main() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.unlink(missing_ok=True)
    out = subprocess.run([str(NF_VENV_PY), "-c", _TORCH], capture_output=True, text=True)
    if out.returncode != 0 or "DUMP_OK" not in out.stdout:
        sys.exit(f"torch side failed (rc={out.returncode}):\n{out.stdout}\n{out.stderr}")
    data = np.load(DUMP)
    x, y_ref = data["x"], data["y"]

    net = NLinearNet(h=CFG["h"], input_size=CFG["input_size"], rngs=nnx.Rngs(0))
    net.weight.value = jnp.asarray(data["w::linear.weight"])   # [h, in] — no transpose
    net.bias.value = jnp.asarray(data["w::linear.bias"])       # [h]

    y_jax = np.asarray(net(jnp.asarray(x)[:, :, None]))        # [B, h, 1]
    max_diff = float(np.max(np.abs(y_jax - y_ref)))
    print(f"max|Δ| = {max_diff:.2e}")
    assert max_diff < 1e-4, f"forward parity failed: {max_diff:.2e}"
    print("PARITY_OK")


if __name__ == "__main__":
    main()
