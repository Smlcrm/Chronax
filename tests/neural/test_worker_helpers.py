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


def test_nf_extra_dict_excludes_loss():
    # loss is fixed to MAE in the template, so it is not rendered into the dict
    assert worker._nf_extra_dict({"loss": "MAE"}) == "{}"


def test_nf_extra_dict_renders_overrides():
    assert worker._nf_extra_dict({"loss": "MAE", "max_steps": 3}) == "{'max_steps': 3}"


def test_nf_extra_dict_rejects_non_mae_loss():
    with pytest.raises(ValueError, match="only loss=MAE"):
        worker._nf_extra_dict({"loss": "MSE"})


def test_nf_subprocess_code_pins_threads_and_imports_model():
    spec = {"name": "AirlinePassengers", "path": "/abs/airline.csv",
            "ds_col": "Month", "y_col": "Passengers", "freq": "MS"}
    params = {"loss": "MAE"}
    code = worker.nf_subprocess_code("GRU", spec, 24, 72, params, 42, threads=1)
    assert "torch.set_num_threads(1)" in code
    assert "from neuralforecast.models import GRU" in code
    assert "h=24" in code and "input_size=72" in code
    assert "random_seed=42" in code and "loss=MAE()" in code
    assert "/abs/airline.csv" in code
    # multivariate models are auto-given n_series=1 (univariate benchmark)
    assert "n_series" in code
    # import is before the timer (fair timing): t0 set on the fit line
    assert code.index("import torch") < code.index("t0 = time.perf_counter()")
