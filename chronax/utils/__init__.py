"""
Chronax utilities package.

This package centralizes shared numeric helpers and conformal-interval
functionality for models and benchmarks.
"""

from .conformal_intervals import ConformalIntervals
from . import utils as _core_utils
from . import plotting as _plotting
from . import loss_functions as loss_functions  # keep metrics under chronax.utils.loss_functions
from .loss_functions import *  # optionally re-export metric functions at top-level

# Re-export names from utils/utils.py and plotting.py. Underscore-prefixed
# helpers stay importable (several models do `from chronax.utils import
# _repeat_val, ...`) but are NOT advertised in __all__ — they are internal
# API and must not leak through `import *` or documentation.
for _name in dir(_core_utils):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_core_utils, _name)

for _name in dir(_plotting):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_plotting, _name)

__all__ = ["ConformalIntervals", "loss_functions"]
__all__ += sorted(
    name
    for name in globals()
    if name not in __all__ and not name.startswith("_")
)
