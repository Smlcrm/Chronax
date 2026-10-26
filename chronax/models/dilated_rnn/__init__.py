"""DilatedRNN forecasting model (univariate, JAX/Flax-NNX port of neuralforecast.DilatedRNN)."""
import flax

# Pinned to flax 0.10.x — NNX is pre-1.0 and the scan / Optimizer / Linear APIs
# have churned across minor versions. Loud failure here beats silent
# miscompilation on a newer flax. (Same pin as the sibling GRU / SOFTS ports.)
if not flax.__version__.startswith("0.10"):
    raise ImportError(
        f"chronax.models.DilatedRNN is pinned to flax==0.10.x; got flax {flax.__version__}. "
        f"The nnx.scan training loop and nnx.Optimizer wiring will likely need updates "
        f"on a newer flax."
    )

from chronax.models.dilated_rnn.dilated_rnn_model import DilatedRNN

__all__ = ["DilatedRNN"]
