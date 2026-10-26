"""StemGNN: Spectral Temporal Graph Neural Network forecaster.

flax.nnx port of ``neuralforecast.models.StemGNN`` (Cao et al., 2020,
https://arxiv.org/abs/2103.07719) behind the chronax ``BaseForecaster`` API.
"""
from chronax.models.stemgnn.stemgnn_model import StemGNN

__all__ = ["StemGNN"]
