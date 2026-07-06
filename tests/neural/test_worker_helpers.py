"""Offline unit tests for pure helpers in benchmarks/neural/worker.py.

No JAX and no NF execution: these only build strings/dicts.
"""
import pytest

from benchmarks.neural import worker


def test_build_thread_env_single_thread():
    env = worker.build_thread_env(1)
    assert env["OMP_NUM_THREADS"] == "1"
    assert "--xla_cpu_multi_thread_eigen=false" in env["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=1" in env["XLA_FLAGS"]


def test_build_thread_env_multi_thread():
    env = worker.build_thread_env(4)
    assert env["OMP_NUM_THREADS"] == "4"
    assert "--xla_cpu_multi_thread_eigen=true" in env["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=4" in env["XLA_FLAGS"]


def test_build_thread_env_rejects_zero():
    with pytest.raises(ValueError, match="threads must be >= 1"):
        worker.build_thread_env(0)


def test_nf_kwargs_str_gru_includes_scaler_and_loss():
    params = {"max_steps": 1000, "learning_rate": 0.001, "scaler_type": "robust", "loss": "MAE"}
    s = worker._nf_kwargs_str(params)
    assert "max_steps=1000" in s
    assert "learning_rate=0.001" in s
    assert "scaler_type='robust'" in s
    assert "loss=MAE()" in s


def test_nf_kwargs_str_patchtst_scaler_identity():
    params = {"max_steps": 5000, "learning_rate": 0.0001, "scaler_type": "identity", "loss": "MAE"}
    s = worker._nf_kwargs_str(params)
    assert "scaler_type='identity'" in s
    assert "max_steps=5000" in s


def test_nf_subprocess_code_pins_threads_and_imports_model():
    spec = {"name": "AirlinePassengers", "path": "/abs/airline.csv",
            "ds_col": "Month", "y_col": "Passengers", "freq": "MS"}
    params = {"max_steps": 1000, "learning_rate": 0.001, "scaler_type": "robust", "loss": "MAE"}
    code = worker.nf_subprocess_code("GRU", spec, 24, 72, params, 42, threads=1)
    assert "torch.set_num_threads(1)" in code
    assert "from neuralforecast.models import GRU" in code
    assert "h=24" in code and "input_size=72" in code
    assert "random_seed=42" in code
    assert "/abs/airline.csv" in code
    # import is before the timer (fair timing, spec §1): t0 set on the fit line
    assert code.index("import torch") < code.index("t0 = time.perf_counter()")
