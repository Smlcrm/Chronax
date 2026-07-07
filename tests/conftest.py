import sys
import os
from pathlib import Path

# Add the repo root (the parent directory of 'chronax') to sys.path.
# This allows 'import chronax.models...' to resolve correctly during tests
# even without an editable install.
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
