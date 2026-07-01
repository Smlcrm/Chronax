"""Chronax-only smoke test: run the worker subprocess with max_steps=2, 1 seed.

Deterministic, offline (no NF). Asserts a well-formed RESULT_JSON row with a
finite mae. airline train length = 120 >= input_size+h = 96.
"""
import json
import subprocess
import sys
import textwrap
from math import isfinite
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "benchmarks" / "neural" / "worker.py"
NF_VENV_PY = REPO / "benchmarks" / ".venv-nf" / "bin" / "python"


def _tiny_config(tmp_path):
    p = tmp_path / "smoke.yaml"
    p.write_text(textwrap.dedent("""
    experiment: { h: 24, input_size: 72, seeds: [42], warmup_seeds: 0, threads: 1 }
    datasets:
      - { name: AirlinePassengers, path: benchmarks/benchmark_datasets/airline-passengers.csv,
          ds_col: Month, y_col: Passengers, freq: MS }
    models:
      - name: GRU
        chronax_params: { max_steps: 2, learning_rate: 0.001, loss: mae }
        nf_params: { max_steps: 2, learning_rate: 0.001, scaler_type: robust, loss: MAE }
    """))
    return str(p)


def test_worker_chronax_smoke(tmp_path):
    cfg = _tiny_config(tmp_path)
    out = subprocess.run(
        [sys.executable, str(WORKER), "--model", "GRU", "--dataset", "AirlinePassengers",
         "--library", "chronax", "--config", cfg],
        cwd=str(REPO), capture_output=True, text=True, check=True,
    )
    rows = [json.loads(ln.split("RESULT_JSON:::")[-1])
            for ln in out.stdout.splitlines() if "RESULT_JSON:::" in ln]
    assert len(rows) == 1
    r = rows[0]
    assert r["library"] == "chronax" and r["model"] == "GRU"
    assert r["dataset"] == "AirlinePassengers" and int(r["seed"]) == 42
    assert r["error"] == ""
    assert isfinite(r["mae"]) and r["mae"] >= 0


def test_all_models_construct_from_config():
    """§6: assert each model's config chronax_params against its Chronax
    constructor (all three: GRU/PatchTST/KAN). Construction only — no fit — so
    this is cheap; a bad param raises TypeError here."""
    from benchmarks.neural import registry
    from benchmarks.neural.run import load_config
    cfg = load_config(str(REPO / "benchmarks" / "neural" / "config.yaml"))
    exp = cfg["experiment"]
    for m in cfg["models"]:
        cls = registry.resolve_chronax(m["name"])
        cls(h=exp["h"], input_size=exp["input_size"], random_seed=1, **m["chronax_params"])


@pytest.mark.skipif(not NF_VENV_PY.exists(),
                    reason="benchmarks/.venv-nf absent; run benchmarks/setup_nf_venv.sh")
def test_worker_nixtla_smoke(tmp_path):
    cfg = _tiny_config(tmp_path)
    out = subprocess.run(
        [sys.executable, str(WORKER), "--model", "GRU", "--dataset", "AirlinePassengers",
         "--library", "nixtla", "--config", cfg],
        cwd=str(REPO), capture_output=True, text=True, check=True,
    )
    rows = [json.loads(ln.split("RESULT_JSON:::")[-1])
            for ln in out.stdout.splitlines() if "RESULT_JSON:::" in ln]
    assert len(rows) == 1 and rows[0]["library"] == "nixtla"
    assert rows[0]["error"] == "" and isfinite(rows[0]["mae"])
