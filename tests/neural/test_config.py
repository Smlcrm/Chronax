"""Offline unit tests for config load/validation in benchmarks/neural/run.py."""
import textwrap

import pytest

from benchmarks.neural.run import ConfigError, load_config, validate_config


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


_VALID = """
experiment: { h: 24, input_size: 72, seeds: [42, 43], warmup_seeds: 1, threads: 1 }
datasets:
  - { name: A, path: p.csv, ds_col: d, y_col: y, freq: D }
models:
  - name: GRU
    chronax_params: { max_steps: 2, learning_rate: 0.001, loss: mae }
    nf_params: { max_steps: 2, learning_rate: 0.001, scaler_type: robust, loss: MAE }
"""


def test_load_valid_config(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID))
    assert cfg["experiment"]["h"] == 24
    assert cfg["models"][0]["name"] == "GRU"


def test_missing_experiment_key_raises(tmp_path):
    bad = _VALID.replace("threads: 1 }", "}")
    with pytest.raises(ConfigError, match="experiment missing keys"):
        load_config(_write(tmp_path, bad))


def test_warmup_out_of_range_raises(tmp_path):
    bad = _VALID.replace("warmup_seeds: 1", "warmup_seeds: 2")  # 2 not < len(seeds)=2
    with pytest.raises(ConfigError, match="warmup_seeds"):
        load_config(_write(tmp_path, bad))


def test_model_entry_incomplete_raises(tmp_path):
    bad = _VALID.replace("nf_params: { max_steps: 2, learning_rate: 0.001, scaler_type: robust, loss: MAE }", "")
    with pytest.raises(ConfigError, match="model entry incomplete"):
        load_config(_write(tmp_path, bad))


def test_validate_config_is_callable_directly():
    with pytest.raises(ConfigError, match="missing 'experiment'"):
        validate_config({"datasets": [], "models": []})


def test_malformed_yaml_raises_config_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("key: [unclosed bracket\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(str(p))
