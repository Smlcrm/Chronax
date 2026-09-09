"""TimesNet forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.TimesNet)."""
import flax

if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.TimesNet is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop may need updates on a newer flax."
    )

from chronax.models.timesnet.timesnet_model import TimesNet

__all__ = ["TimesNet"]
