#!/usr/bin/env bash
# Create the isolated venv used by benchmarks/neural/run.py to run Nixtla
# neuralforecast in a subprocess. Keeps torch + Lightning out of the main
# Chronax dev environment.
#
# Run from the repo root:
#   bash benchmarks/setup_nf_venv.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d benchmarks/.venv-nf ]]; then
  echo "benchmarks/.venv-nf already exists; reinstalling deps in place."
else
  python3 -m venv benchmarks/.venv-nf
fi

benchmarks/.venv-nf/bin/pip install --upgrade pip
benchmarks/.venv-nf/bin/pip install -r benchmarks/requirements-nf.txt

echo
echo "Done. Run the benchmark with:"
echo "  python benchmarks/neural/run.py"
