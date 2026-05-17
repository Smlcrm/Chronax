"""GRU forecasting model."""
import flax

# Sanity check: this model is pinned to flax 0.10.x. NNX is pre-1.0 and the
# scan / Optimizer / GRUCell APIs have churned across minor versions.
# Per the Maintenance Status banner in the GRU class docstring, this model is
# shipped as-is; loud failure here beats silent miscompilation on a newer flax.
if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.GRU is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"If you need to support a newer flax, the GRU's nnx.scan-based training "
        f"loop and nnx.GRUCell wiring will likely need updates. See the "
        f"`Maintenance Status` banner in the GRU class docstring."
    )

from chronax.models.gru.gru_model import GRU

__all__ = ["GRU"]
