import sys
from pathlib import Path

# benchmarks/ is excluded from packaging, so put the repo root on sys.path
# to allow `import benchmarks.neural.*` in these tests.
# tests/neural/conftest.py -> parents[2] == repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
