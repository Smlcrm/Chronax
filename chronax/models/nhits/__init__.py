"""NHITS forecaster package: flax.nnx port of neuralforecast.NHITS with point,
multi-quantile, and GMM distribution losses."""
from chronax.models.nhits.nhits_losses import GMM
from chronax.models.nhits.nhits_model import NHITS

__all__ = ["NHITS", "GMM"]
