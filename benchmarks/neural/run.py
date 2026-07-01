"""Neural benchmark orchestrator (Chronax vs Nixtla neuralforecast).

Built up across tasks: config load/validate + error taxonomy (Task 4), resume
load (Task 8), accept-gate (Task 9), orchestration/modes + baseline metadata
(Task 10).
"""
from __future__ import annotations

import yaml


class NeuralBenchError(Exception):
    """Base error for the neural benchmark harness."""


class ConfigError(NeuralBenchError):
    """Raised when config.yaml is missing required keys or has bad values."""


_REQUIRED_EXP = {"h", "input_size", "seeds", "warmup_seeds", "threads"}


def validate_config(cfg: dict) -> None:
    """Raise ConfigError if `cfg` is missing required structure."""
    if not isinstance(cfg, dict):
        raise ConfigError("config is empty or not a YAML mapping")
    if "experiment" not in cfg:
        raise ConfigError("missing 'experiment' section")
    exp = cfg["experiment"]
    missing = _REQUIRED_EXP - set(exp)
    if missing:
        raise ConfigError(f"experiment missing keys: {sorted(missing)}")
    if not cfg.get("datasets"):
        raise ConfigError("no datasets defined")
    if not cfg.get("models"):
        raise ConfigError("no models defined")
    for m in cfg["models"]:
        if not {"name", "chronax_params", "nf_params"} <= set(m):
            raise ConfigError(f"model entry incomplete: {m}")
    warm, n = exp["warmup_seeds"], len(exp["seeds"])
    if not (0 <= warm < n):
        raise ConfigError(f"warmup_seeds {warm} out of range for {n} seeds")


def load_config(path: str) -> dict:
    """Load and validate config.yaml."""
    with open(path) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML: {e}") from e
    validate_config(cfg)
    return cfg
