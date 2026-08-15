"""RMoK forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.RMoK)."""
import flax

if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.RMoK is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop may need updates on a newer flax."
    )

from chronax.models.rmok.rmok_model import RMoK

__all__ = ["RMoK"]
