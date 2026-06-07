"""PatchTST forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.PatchTST)."""
import flax

# Pinned to flax 0.10.x — NNX is pre-1.0 and the scan / Optimizer / BatchNorm
# APIs have churned across minor versions. Loud failure here beats silent
# miscompilation on a newer flax.
if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.PatchTST is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop, nnx.BatchNorm wiring, and residual-attention "
        f"path will likely need updates on a newer flax."
    )

from chronax.models.patchtst.patchtst_model import PatchTST

__all__ = ["PatchTST"]
