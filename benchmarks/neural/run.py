"""Neural benchmark orchestrator (Chronax vs Nixtla neuralforecast).

Built up across tasks: config load/validate + error taxonomy (Task 4), resume
load (Task 8), accept-gate (Task 9), orchestration/modes + baseline metadata
(Task 10).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
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
    overrides = cfg.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ConfigError("'overrides' must be a mapping of model -> params")
        for name, ov in overrides.items():
            if not isinstance(ov, dict) or not set(ov) <= {"chronax_params", "nf_params"}:
                raise ConfigError(
                    f"override for {name!r} may only contain 'chronax_params' / 'nf_params'")
    warm, n = exp["warmup_seeds"], len(exp["seeds"])
    if not (0 <= warm < n):
        raise ConfigError(f"warmup_seeds {warm} out of range for {n} seeds")


def model_params(cfg: dict, name: str) -> tuple[dict, dict]:
    """Resolve (chronax_params, nf_params) for `name`.

    Models are auto-discovered, so config carries no per-model block by default;
    `cfg['overrides'][name]` is an optional override. MAE loss is forced on both
    sides. A standard model needs no override — both libraries fall back to their
    (matching) defaults.
    """
    ov = (cfg.get("overrides") or {}).get(name, {})
    chronax_params = {"loss": "mae", **(ov.get("chronax_params") or {})}
    nf_params = {"loss": "MAE", **(ov.get("nf_params") or {})}
    return chronax_params, nf_params


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


def load_results_tolerant(path: str | Path) -> pd.DataFrame:
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


def remaining_seeds(model: str, dataset: str, library: str, seeds: Sequence[int], done: set[tuple]) -> list[int]:
    """Seeds not yet present for this {model,dataset,library}."""
    return [s for s in seeds if (model, dataset, library, int(s)) not in done]


# The accept-gate certifies a verdict only for a full paired run over BOTH
# libraries — independent of which --libs the current invocation actually ran.
REQUIRED_LIBS = ["chronax", "nixtla"]


def summarize(df: pd.DataFrame, warmup_seeds: int, group_cols: list) -> pd.DataFrame:
    """Mean/std over all seeds for MAE/sMAPE/wall-clock, plus an after-warmup
    wall-clock mean/std over the non-warmup seeds (spec §8)."""
    total = df.groupby(group_cols).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        smape_mean=("smape", "mean"), smape_std=("smape", "std"),
        wallclock_mean=("wallclock_s", "mean"), wallclock_std=("wallclock_s", "std"),
    ).round(4)
    after = df[df.iter_idx >= warmup_seeds].groupby(group_cols).agg(
        wallclock_mean_after=("wallclock_s", "mean"),
        wallclock_std_after=("wallclock_s", "std"),
    ).round(4)
    return total.join(after)


def _paired(df, model, dataset, value_col, sub_filter=None):
    d = df if sub_filter is None else df[sub_filter(df)]
    def series(lib):
        s = d[(d.model == model) & (d.dataset == dataset) & (d.library == lib)]
        return s.sort_values("seed").set_index("seed")[value_col]
    c, n = series("chronax"), series("nixtla")
    seeds = c.index.intersection(n.index)
    delta = c.loc[seeds] - n.loc[seeds]
    return c, n, seeds, delta


def paired_accuracy(df: pd.DataFrame, model: str, dataset: str) -> dict:
    c, n, seeds, delta = _paired(df, model, dataset, "mae")
    return {
        "chronax_mean": float(c.mean()), "chronax_std": float(c.std(ddof=1)),
        "nixtla_mean": float(n.mean()), "nixtla_std": float(n.std(ddof=1)),
        "paired_mean_delta": float(delta.mean()), "n": int(len(seeds)),
        "pass": bool(delta.mean() <= 0),  # chronax accuracy >= NF (spec §8)
    }


def paired_speed(df: pd.DataFrame, model: str, dataset: str, warmup_seeds: int) -> dict:
    c, n, seeds, delta = _paired(
        df, model, dataset, "wallclock_s",
        sub_filter=lambda d: d.iter_idx >= warmup_seeds)
    return {
        "chronax_mean": float(c.mean()), "chronax_std": float(c.std(ddof=1)),
        "nixtla_mean": float(n.mean()), "nixtla_std": float(n.std(ddof=1)),
        "paired_mean_delta": float(delta.mean()), "n": int(len(seeds)),
        "pass": bool(c.mean() < n.mean()),  # chronax after-warmup speed > NF
    }


def is_canonical(df: pd.DataFrame, models: Sequence[str], datasets: Sequence[str], libs: Sequence[str], seeds: Sequence[int]) -> bool:
    """True iff df is exactly one clean, full, single-run set — the only file the
    gate will certify (spec §9). Requires:
      1. every (model,dataset,library,seed) present exactly once (no missing/dup),
      2. no error rows, and
      3. per (model,dataset,library) the iter_idx values are exactly
         range(len(seeds)).

    Check 3 is what makes a RESUMED file non-canonical: a resume runs only the
    remaining seeds, so the worker restarts iter_idx at 0 and the group's
    iter_idx multiset (e.g. [0,0,1,1,2]) no longer equals range(len(seeds)) even
    though every seed is present once. Resumed seeds re-pay JIT warmup, so their
    wall-clock is contaminated and the gate must refuse.
    """
    want = list(seeds)
    expected = Counter((m, d, l, int(s))
                       for m in models for d in datasets for l in libs for s in want)
    actual = Counter((r.model, r.dataset, r.library, int(r.seed))
                     for r in df.itertuples())
    if actual != expected:
        return False
    if df["error"].fillna("").astype(str).str.len().gt(0).any():
        return False
    want_iter = list(range(len(want)))
    for m in models:
        for d in datasets:
            for l in libs:
                grp = df[(df.model == m) & (df.dataset == d) & (df.library == l)]
                if sorted(int(x) for x in grp["iter_idx"]) != want_iter:
                    return False
    return True


def accept_gate_report(df: pd.DataFrame, models: Sequence[str], datasets: Sequence[str], libs: Sequence[str], seeds: Sequence[int], warmup_seeds: int) -> str:
    """Per-cell paired accept-gate table + overall verdict, or a non-canonical
    notice (spec §8/§9). `libs` is the REQUIRED library set (both chronax and
    nixtla) — a run missing a library, resumed, incomplete, or carrying error
    rows is non-canonical. Returns text; never raises, never exits non-zero."""
    if not is_canonical(df, models, datasets, libs, seeds):
        return ("=== Accept-gate ===\n"
                "non-canonical timing (resumed / incomplete / missing a library / "
                "error rows present) — rerun a clean full run for a verdict.")
    lines = ["=== Accept-gate (paired per-seed; chronax vs nixtla) ==="]
    overall_pass, fails = True, []
    for model in models:
        for dataset in datasets:
            acc = paired_accuracy(df, model, dataset)
            spd = paired_speed(df, model, dataset, warmup_seeds)
            cell_pass = acc["pass"] and spd["pass"]
            overall_pass = overall_pass and cell_pass
            cell = f"{model}/{dataset}"
            if not cell_pass:
                fails.append(cell)
            lines.append(
                f"[{'PASS' if cell_pass else 'FAIL'}] {cell}  "
                f"MAE chx={acc['chronax_mean']:.4f}±{acc['chronax_std']:.4f} "
                f"nf={acc['nixtla_mean']:.4f}±{acc['nixtla_std']:.4f} "
                f"Δ={acc['paired_mean_delta']:+.4f}(n={acc['n']})  |  "
                f"time(after) chx={spd['chronax_mean']:.2f}±{spd['chronax_std']:.2f} "
                f"nf={spd['nixtla_mean']:.2f}±{spd['nixtla_std']:.2f} "
                f"Δ={spd['paired_mean_delta']:+.2f}(n={spd['n']})")
    verdict = "PASS" if overall_pass else f"FAIL ({', '.join(fails)})"
    lines.append(f"OVERALL VERDICT: {verdict}")
    return "\n".join(lines)


WORKER_PY = REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PY = REPO_ROOT / "benchmarks" / "neural" / "worker.py"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "benchmark_results" / "neural"
BASELINES_DIR = REPO_ROOT / "benchmarks" / "baselines"

# benchmarks/ is not an installed package; ensure the repo root is on sys.path
# so `from benchmarks.neural import registry` resolves when run as a script.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.neural import registry  # noqa: E402  (after sys.path fixup)


def _resolve_nf_venv_py(repo_root: Path) -> Path:
    """Windows: Scripts/python.exe; Unix: bin/python."""
    base = Path(repo_root) / "benchmarks" / ".venv-nf"
    candidates = (
        base / "Scripts" / "python.exe",
        base / "bin" / "python",
        base / "bin" / "python3",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if sys.platform.startswith("win") else candidates[1]


NF_VENV_PY = _resolve_nf_venv_py(REPO_ROOT)


def check_nf_venv(libs, venv_py=NF_VENV_PY) -> None:
    """Fail fast up front if NF is requested but .venv-nf is absent (spec §10)."""
    if "nixtla" in libs and not Path(venv_py).exists():
        raise NeuralBenchError(
            f"{venv_py} not found — create the isolated neuralforecast venv first:\n"
            "  Unix:  bash benchmarks/setup_nf_venv.sh\n"
            "  Windows:\n"
            "    python -m venv benchmarks\\.venv-nf\n"
            "    benchmarks\\.venv-nf\\Scripts\\pip.exe install -r benchmarks\\requirements-nf.txt")


def stream_worker_to_csv(model, dataset, library, config_path, csv_path, done, seeds) -> None:
    """Spawn one worker for {model,dataset,library}; append each streamed
    RESULT_JSON row to csv_path (skipping already-done keys). Resumable by
    construction.

    On resume the worker is handed only the remaining seeds, so it restarts
    iter_idx at 0 — which is exactly what is_canonical (Task 9) uses to flag a
    resumed file as non-canonical and refuse a verdict."""
    remaining = remaining_seeds(model, dataset, library, seeds, done)
    if not remaining:
        print(f"  skip {model}/{dataset}/{library}: all seeds done", flush=True)
        return
    cmd = [sys.executable, str(WORKER_PY), "--model", model, "--dataset", dataset,
           "--library", library, "--config", config_path,
           "--seeds", *[str(s) for s in remaining]]
    write_header = not Path(csv_path).exists()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader(); f.flush()
        for line in proc.stdout:
            if "RESULT_JSON:::" not in line:
                continue
            row = json.loads(line.split("RESULT_JSON:::")[-1].strip())
            key = (row["model"], row["dataset"], row["library"], int(row["seed"]))
            if key in done:
                continue
            writer.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush(); done.add(key)
            tag = row.get("error") or f"mae={row['mae']:.4f} t={row['wallclock_s']:.1f}s"
            print(f"  {library:8s} {model}/{dataset} seed={row['seed']} {tag}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"[warn] worker {model}/{dataset}/{library} exit {proc.returncode}:\n"
              f"{proc.stderr.read()}", flush=True)


_NF_PROBE = '''
import json, sys, platform
from importlib.metadata import version
import torch
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import {nf_name}
params = json.loads({params_json!r})
params.pop("loss", None)
m = {nf_name}(h={h}, input_size={input_size}, loss=MAE(), **params)
def _safe(v):
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return type(v).__name__
hp = {{k: _safe(v) for k, v in dict(m.hparams).items()}}
print(json.dumps({{
    "versions": {{"python": sys.version.split()[0], "torch": torch.__version__,
                  "neuralforecast": version("neuralforecast"),
                  "platform": platform.platform()}},
    "model_args": hp}}))
'''


def nf_baseline_metadata(nf_name, h, input_size, nf_params, venv_py=NF_VENV_PY) -> dict:
    """Query .venv-nf for env versions + the fully-resolved model hyperparameters
    so protocol.json stays reconstructible even if NF's architecture defaults
    drift later (spec §13). One short subprocess in the isolated venv."""
    code = _NF_PROBE.format(nf_name=nf_name, h=h, input_size=input_size,
                            params_json=json.dumps(nf_params))
    out = subprocess.run([str(venv_py), "-c", code],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def write_baseline(model, df, cfg) -> None:
    """Write baselines/<model>/{protocol.json, baseline_nixtla_raw.csv,
    baseline_nixtla_summary.csv} from an NF-only results frame (spec §8, §13).
    protocol.json records env versions + the resolved NF architecture (encoder
    sizes, layers, etc.) so numbers stay reproducible. Called by
    --refresh-baseline. Does NOT commit — commit is gated (Task 14)."""
    exp = cfg["experiment"]
    out = BASELINES_DIR / model
    out.mkdir(parents=True, exist_ok=True)
    _, nf_params = model_params(cfg, model)
    meta = nf_baseline_metadata(registry.nf_model_name(model), exp["h"],
                                exp["input_size"], nf_params)
    protocol = {
        "h": exp["h"], "input_size": exp["input_size"], "seeds": exp["seeds"],
        "warmup_seeds": exp["warmup_seeds"], "threads": exp["threads"],
        "config_nf_params": nf_params,
        "resolved_model_args": meta["model_args"],
        "versions": meta["versions"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    raw = df[["dataset", "seed", "mae", "smape", "wallclock_s"]].sort_values(["dataset", "seed"])
    raw.to_csv(out / "baseline_nixtla_raw.csv", index=False)
    summarize(df, exp["warmup_seeds"], group_cols=["dataset"]).to_csv(
        out / "baseline_nixtla_summary.csv")


def _run_grid(cfg, models, datasets, libs, csv_path, resume) -> pd.DataFrame:
    seeds = cfg["experiment"]["seeds"]
    csv_path = Path(csv_path)
    if not resume and csv_path.exists():
        csv_path.unlink()  # fresh run: start clean so a re-run never appends dup keys
    done = done_keys(load_results_tolerant(csv_path)) if resume else set()
    for m in models:
        for d in datasets:
            for lib in libs:
                stream_worker_to_csv(m, d, lib, cfg["_config_path"], csv_path, done, seeds)
    return load_results_tolerant(csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Neural benchmark orchestrator.")
    ap.add_argument("--config", default=str(REPO_ROOT / "benchmarks" / "neural" / "config.yaml"))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--libs", nargs="+", default=["chronax", "nixtla"],
                    choices=["chronax", "nixtla"])
    ap.add_argument("--refresh-baseline", metavar="MODEL", default=None,
                    help="Run NF only and (over)write baselines/<MODEL>/. Commit is gated (Task 14).")
    ap.add_argument("--check-committed", action="store_true",
                    help="Run chronax only; compare to committed baselines/<MODEL>/summary "
                         "(requires a post-recapture summary — see --refresh-baseline).")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_config_path"] = args.config
    all_models = registry.list_models()
    all_datasets = [d["name"] for d in cfg["datasets"]]
    models = args.models or all_models
    datasets = args.datasets or all_datasets

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh_baseline:
        model = args.refresh_baseline
        check_nf_venv(["nixtla"])
        csv_path = RESULTS_DIR / f"baseline_{model}.csv"
        df = _run_grid(cfg, [model], datasets, ["nixtla"], csv_path, args.resume)
        write_baseline(model, df[df.library == "nixtla"], cfg)
        print(f"Wrote baselines/{model}/ — REVIEW then commit with sign-off (Task 14).")
        return

    if args.check_committed:
        csv_path = RESULTS_DIR / "neural_check.csv"
        df = _run_grid(cfg, models, datasets, ["chronax"], csv_path, args.resume)
        for model in models:
            summary_path = BASELINES_DIR / model / "baseline_nixtla_summary.csv"
            if not summary_path.exists():
                raise NeuralBenchError(
                    f"no committed baseline for {model} at {summary_path}; capture it with "
                    f"`benchmarks/neural/run.py --refresh-baseline {model}` first (Task 14).")
            committed = pd.read_csv(summary_path)
            sub = df[(df.library == "chronax") & (df.model == model)]
            print(f"\n--- {model} ---")
            print(check_committed_report(sub, committed, cfg["experiment"]["warmup_seeds"]))
        return

    # full mode
    check_nf_venv(args.libs)
    csv_path = RESULTS_DIR / "neural_results.csv"
    df = _run_grid(cfg, models, datasets, args.libs, csv_path, args.resume)
    summarize(df, cfg["experiment"]["warmup_seeds"],
              group_cols=["model", "library", "dataset"]).to_csv(
        RESULTS_DIR / "neural_results_summary.csv")
    print()
    # The gate requires BOTH libraries (REQUIRED_LIBS), independent of --libs, so a
    # chronax-only or nixtla-only run yields a non-canonical notice, not a bogus verdict.
    print(accept_gate_report(df, models, datasets, REQUIRED_LIBS,
                             cfg["experiment"]["seeds"], cfg["experiment"]["warmup_seeds"]))
    print(f"\nWrote {csv_path.name} + neural_results_summary.csv under "
          f"benchmarks/benchmark_results/neural/")


def check_committed_report(chronax_df: pd.DataFrame, committed_summary: pd.DataFrame, warmup_seeds: int) -> str:
    """Coarse (unpaired) chronax-vs-committed-NF-summary check (spec §8
    --check-committed). Not the paired canonical verdict; cheap iteration.

    Requires a post-recapture summary (with the after-warmup wall-clock column);
    the migrated legacy GRU summary predates it, so guard explicitly rather than
    KeyError mid-report."""
    if "wallclock_mean_after" not in committed_summary.columns:
        raise NeuralBenchError(
            "committed baseline summary lacks the 'wallclock_mean_after' column "
            "(pre-pinned-thread schema) — recapture it with --refresh-baseline.")
    chx = summarize(chronax_df, warmup_seeds, group_cols=["dataset"])
    lines = ["=== Check against committed NF baseline (unpaired) ==="]
    committed = committed_summary.set_index("dataset")
    for dataset, row in chx.iterrows():
        nf = committed.loc[dataset]
        acc_pass = row["mae_mean"] <= nf["mae_mean"]
        spd_pass = row["wallclock_mean_after"] < nf["wallclock_mean_after"]
        verdict = "PASS" if (acc_pass and spd_pass) else "FAIL"
        lines.append(
            f"[{verdict}] {dataset}  MAE chx={row['mae_mean']:.4f} "
            f"nf={nf['mae_mean']:.4f}  time(after) chx={row['wallclock_mean_after']:.2f} "
            f"nf={nf['wallclock_mean_after']:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
