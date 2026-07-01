"""Neural benchmark orchestrator (Chronax vs Nixtla neuralforecast).

Built up across tasks: config load/validate + error taxonomy (Task 4), resume
load (Task 8), accept-gate (Task 9), orchestration/modes + baseline metadata
(Task 10).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
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
    """Load and validate config.yaml.

    A missing or unreadable file surfaces as ConfigError (not a bare OSError) so
    every config failure stays inside the NeuralBenchError taxonomy.
    """
    try:
        f = open(path)
    except OSError as e:  # FileNotFoundError / PermissionError / ...
        raise ConfigError(f"cannot read config {path!r}: {e}") from e
    with f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML: {e}") from e
    validate_config(cfg)
    return cfg


FIELDS = ["library", "dataset", "model", "seed", "iter_idx", "is_warmup",
          "mae", "smape", "wallclock_s", "error"]

_NUM_COLS = ["mae", "smape", "wallclock_s"]


def load_results_tolerant(path) -> pd.DataFrame:
    """Read the results CSV, dropping any torn line left by a crash.

    The worker streams one full row per seed via csv.DictWriter, so a clean line
    has exactly len(FIELDS) fields (error messages with commas/newlines are
    quoted). We parse with the stdlib csv reader — quote-aware, so a multi-line
    quoted error field is reassembled into one record — and keep only rows whose
    field count is exactly len(FIELDS). A crash-truncated trailing line has too
    few fields and is dropped. This does NOT rely on NaN-key/ParserError: pandas
    NaN-pads a short final line instead of raising, so a torn line truncated
    after the key columns must be caught by field count, not by dropna-on-keys.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=FIELDS)
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] != FIELDS:
        return pd.DataFrame(columns=FIELDS)
    good = [r for r in rows[1:] if len(r) == len(FIELDS)]
    df = pd.DataFrame(good, columns=FIELDS)
    if len(df):
        df["seed"] = df["seed"].astype(int)
        df["iter_idx"] = df["iter_idx"].astype(int)
        for c in _NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["error"] = df["error"].fillna("").astype(str)
    return df.reset_index(drop=True)


def done_keys(df: pd.DataFrame) -> set:
    """Set of (model, dataset, library, seed) rows already present."""
    return {(r.model, r.dataset, r.library, int(r.seed)) for r in df.itertuples()}


def remaining_seeds(model, dataset, library, seeds, done) -> list:
    """Seeds not yet present for this {model,dataset,library}."""
    return [s for s in seeds if (model, dataset, library, int(s)) not in done]
