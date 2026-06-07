"""Flax NNX modules for the PatchTST forecaster.

Univariate port of neuralforecast.PatchTST. The series carries a channel
dimension of size 1 (``c_in = 1``); RevIN operates on ``[B, L, 1]`` and the
transformer encoder operates on ``[B, patch_num, hidden_size]``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class RevIN(nnx.Module):
    """Reversible instance normalization (Kim et al. 2022), per-window.

    Faithful to neuralforecast's RevIN at PatchTST's defaults: centers on the
    last timestep (``subtract_last=True``), divides by ``sqrt(var + eps)`` with
    population variance (``ddof=0``), and applies no learnable affine
    (``affine=False``). Statistics are returned explicitly rather than cached,
    so the module is pure and vmap/scan-safe.
    """

    def __init__(
        self,
        num_features: int,
        *,
        subtract_last: bool = True,
        affine: bool = False,
        eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.num_features = num_features
        self.subtract_last = subtract_last
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
            self.beta = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))

    def norm(self, x: jnp.ndarray):
        """x: [B, L, C] -> (z: [B, L, C], loc: [B, 1, C], scale: [B, 1, C])."""
        x = x.astype(jnp.float32)
        if self.subtract_last:
            loc = x[:, -1:, :]
        else:
            loc = jnp.mean(x, axis=1, keepdims=True)
        var = jnp.var(x, axis=1, keepdims=True)  # ddof=0 (population)
        scale = jnp.sqrt(var + self.eps)
        z = (x - loc) / scale
        if self.affine:
            z = z * self.gamma.value[None, None, :] + self.beta.value[None, None, :]
        return z, loc, scale

    def denorm(self, z: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Invert norm. z: [B, T, C] with broadcastable loc/scale [B, 1, C]."""
        if self.affine:
            z = (z - self.beta.value[None, None, :]) / (self.gamma.value[None, None, :] + self.eps * self.eps)
        return z * scale + loc


def compute_patch_num(input_size: int, patch_len: int, stride: int) -> int:
    """Number of patches after end-padding by ``stride`` (NF padding_patch='end').

    Mirrors NF's ``int((input_size - patch_len) / stride + 1) + 1`` exactly
    (truncation toward zero, not Python floor) so it stays correct even if called
    with an unclamped ``patch_len``; callers normally pass the clamped
    ``patch_len = min(input_size + stride, patch_len)``.
    """
    return int((input_size - patch_len) / stride + 1) + 1


def patchify(x: jnp.ndarray, *, patch_len: int, stride: int) -> jnp.ndarray:
    """x: [B, L] -> patches: [B, patch_num, patch_len].

    Replicates torch ``ReplicationPad1d((0, stride))`` then
    ``unfold(dim=-1, size=patch_len, step=stride)``.
    """
    x = x.astype(jnp.float32)
    orig_L = x.shape[1]                                 # input_size, before padding
    x = jnp.pad(x, ((0, 0), (0, stride)), mode="edge")  # end padding
    n = compute_patch_num(orig_L, patch_len, stride)
    starts = jnp.arange(n) * stride
    offs = jnp.arange(patch_len)
    idx = starts[:, None] + offs[None, :]               # [patch_num, patch_len]
    return x[:, idx]                                     # [B, patch_num, patch_len]
