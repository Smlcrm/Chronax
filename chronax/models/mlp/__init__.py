"""Multilayer perceptron (MLP) forecaster package: flax.nnx port of neuralforecast.MLP with point,
multi-quantile, and Gaussian mixture model (GMM) distribution losses."""
from chronax.models.mlp.mlp_losses import GMM
from chronax.models.mlp.mlp_model import MLP

__all__ = ["MLP", "GMM"]
