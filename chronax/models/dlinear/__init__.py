"""DLinear forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.DLinear)."""
import flax

if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.DLinear is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop may need updates on a newer flax."
    )

from chronax.models.dlinear.dlinear_model import DLinear

__all__ = ["DLinear"]
