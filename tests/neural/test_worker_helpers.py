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
            "ds_col": "Month", "y_col": "Passengers", "freq": "MS", "kind": "univariate"}
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


def test_nf_subprocess_code_multivariate_fits_per_series():
    """NF must run PER-SERIES on multivariate datasets, mirroring the Chronax side.

    Chronax is univariate-only by design (one model per series), so a joint
    n_series=N NF fit would compare N independent models against one
    cross-learning model — NF would win on shared parameters, not on the engine.
    Both libraries therefore fit N independent univariate models and average.
    """
    spec = {
        "name": "WeeklyCushingUS", "path": "/abs/cushing.csv", "ds_col": "Datetime",
        "kind": "multivariate",
        "y_cols": ["a", "b", "c"], "freq": "W-WED",
    }
    code = worker.nf_subprocess_code("TSMixer", spec, 24, 72, {"loss": "MAE"}, 42, 1)
    assert "kw['n_series'] = 3" not in code       # no joint fit
    assert "n_series', 1" in code or "n_series'] = 1" in code
    assert "for col in ['a', 'b', 'c']" in code   # one fit per series
    assert "np.mean(maes)" in code                # averaged like the chronax side
    # one timer around ALL N fits, matching run_chronax_seed's single elapsed
    assert code.count("t0 = time.perf_counter()") == 1


def test_nf_subprocess_code_covariate_passes_exog_lists():
    spec = {
        "name": "CapitalBikeshare", "path": "/abs/bike.csv", "ds_col": "Datetime",
        "kind": "covariate", "y_col": "bike_trips",
        "hist_exog_cols": ["temp", "hum", "windspeed"], "freq": "D",
    }
    code = worker.nf_subprocess_code("TFT", spec, 24, 72, {"loss": "MAE"}, 42, 1)
    assert "hist_exog_list" in code
    assert "futr_exog_list" in code
    assert "EXOGENOUS_HIST" in code
    assert "temp" in code and "hum" in code


def test_nf_subprocess_code_covariate_skips_hist_when_futr_only():
    """Autoformer/FEDformer/Informer accept hist_exog_list kw but reject hist at runtime."""
    spec = {
        "name": "CapitalBikeshare", "path": "/abs/bike.csv", "ds_col": "Datetime",
        "kind": "covariate", "y_col": "bike_trips",
        "hist_exog_cols": ["temp", "hum"], "freq": "D",
    }
    code = worker.nf_subprocess_code("Autoformer", spec, 24, 72, {"loss": "MAE"}, 42, 1)
    assert "futr_exog_list" in code
    assert "supports_hist" in code
    assert "EXOGENOUS_HIST" in code
    # hist list assignment is gated; unconditional assignment must not appear
    assert "kw['hist_exog_list']" in code  # still present behind supports_hist guard
    assert "and supports_hist" in code

def test_resolve_horizon_uses_dataset_override():
    spec = {"h": 7, "input_size": 35}
    exp = {"h": 24, "input_size": 72}
    assert worker.resolve_horizon(spec, exp) == (7, 35)
    assert worker.resolve_horizon({}, exp) == (24, 72)


def test_load_dataset_shapes(tmp_path):
    import pandas as pd

    uni = tmp_path / "uni.csv"
    pd.DataFrame({"Datetime": ["2020-01-01", "2020-01-02"], "y": [1.0, 2.0]}).to_csv(uni, index=False)
    data = worker.load_dataset({
        "kind": "univariate", "path": str(uni), "ds_col": "Datetime", "y_col": "y", "freq": "D",
        "name": "U",
    })
    assert data["y"].shape == (2,) and data["X"] is None

    multi = tmp_path / "multi.csv"
    pd.DataFrame({
        "Datetime": ["2020-01-01", "2020-01-02"],
        "a": [1.0, 2.0], "b": [3.0, 4.0],
    }).to_csv(multi, index=False)
    data = worker.load_dataset({
        "kind": "multivariate", "path": str(multi), "ds_col": "Datetime",
        "y_cols": ["a", "b"], "freq": "D", "name": "M",
    })
    assert data["y"].shape == (2, 2)

    cov = tmp_path / "cov.csv"
    pd.DataFrame({
        "Datetime": ["2020-01-01", "2020-01-02"],
        "y": [1.0, 2.0], "x1": [0.1, 0.2], "x2": [0.3, 0.4],
    }).to_csv(cov, index=False)
    data = worker.load_dataset({
        "kind": "covariate", "path": str(cov), "ds_col": "Datetime",
        "y_col": "y", "hist_exog_cols": ["x1", "x2"], "freq": "D", "name": "C",
    })
    assert data["y"].shape == (2,) and data["X"].shape == (2, 2)
