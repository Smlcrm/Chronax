"""KAN forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.KAN)."""
import flax

if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.KAN is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop and nnx.Variable grid wiring may need updates on a newer flax."
    )

from chronax.models.kan.kan_model import KAN

__all__ = ["KAN"]
