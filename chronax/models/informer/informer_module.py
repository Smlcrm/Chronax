"""Distilling encoder, generative decoder, and the top-level network (flax.nnx).

Assembles the encoder/decoder layers from ``informer_layers.py`` into the full
Informer backbone (NF ``Encoder`` + ``Decoder`` + top-level ``Informer.forward``).
The encoder interleaves ``encoder_layers`` ProbSparse attention layers with
``encoder_layers - 1`` distilling ``ConvLayer``s (the last attention layer has no
conv after it, so the sequence is halved ``encoder_layers - 1`` times, not
``encoder_layers`` times); the decoder runs masked self-attention then
cross-attention against the encoder output through ``decoder_layers`` layers and
carries its own output projection (NF convention: the head lives inside the
decoder, not the top-level net, unlike ``TFTNet``/``ITransformerNet``). The
public wrapper is registered as ``chronax.models.Informer``, matching the
style of ``chronax/models/tft/tft_module.py``. ``float32`` throughout; every path
is vmap-pure (all shapes are static Python ints).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from chronax.models.informer.informer_layers import (
    ConvLayer,
    DataEmbedding,
    TransDecoderLayer,
    TransEncoderLayer,
    _torch_linear,
)


def distilled_length(input_size: int, n_conv: int) -> int:
    """Sequence length remaining after ``n_conv`` distilling ``ConvLayer``s.

    Each conv applies ``L -> (L+1)//2 + 1`` (NF-parity circular padding=2 conv
    expands to L+2, then the stride-2 maxpool arithmetic from ``ConvLayer``).
    Pure Python int arithmetic on static shapes, so callers (e.g. the
    training/model wrappers) can size buffers without invoking the net.
    """
    L = input_size
    for _ in range(n_conv):
        L = (L + 1) // 2 + 1
    return L


class TransEncoder(nnx.Module):
    """Stack of ``encoder_layers`` ProbSparse attention layers, optionally distilled.

    NF interleave (``distil=True``): ``attn -> conv -> attn -> conv -> ... -> attn``
    -- ``encoder_layers`` attention layers alternating with ``encoder_layers - 1``
    ``ConvLayer``s, where the LAST attention layer has no conv after it. Without
    distilling, it is a plain attention stack (no length reduction). Either way a
    final ``LayerNorm`` closes the encoder (NF ``norm_layer``). Each attention
    layer consumes its own entry of ``sample_keys`` (ProbSparse's data-independent
    key subsample, one per attention site); the ``ConvLayer``s consume the shared
    ``use_running_average`` flag (BatchNorm running-stat toggle).
    """

    def __init__(
        self, *, encoder_layers: int, hidden_size: int, n_head: int, conv_hidden_size: int,
        factor: int, dropout: float, activation: str = "gelu", distil: bool, rngs: nnx.Rngs,
    ) -> None:
        self.distil = distil
        self.attn_layers = [
            TransEncoderLayer(
                hidden_size=hidden_size, n_head=n_head, conv_hidden_size=conv_hidden_size,
                factor=factor, dropout=dropout, activation=activation, rngs=rngs,
            )
            for _ in range(encoder_layers)
        ]
        self.conv_layers = (
            [ConvLayer(hidden_size, rngs=rngs) for _ in range(encoder_layers - 1)]
            if distil else []
        )
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, *, sample_keys, deterministic: bool, use_running_average: bool,
    ) -> jnp.ndarray:
        """x: ``[B, L, hidden]`` -> ``[B, L', hidden]`` (``L' == L`` iff not distilled)."""
        # Static Python-int length check (trace-time, jit/vmap-safe, same class as the
        # `L_Q == L_K` assert in `_prob_attention`): the distil branch below zips
        # `sample_keys[:-1]` against `attn_layers[:-1]`/`conv_layers`, which would
        # silently truncate to the shortest iterable on a miscount instead of erroring.
        assert len(sample_keys) == len(self.attn_layers), "one sample_key per attention layer"
        if self.distil:
            for attn, conv, key in zip(self.attn_layers[:-1], self.conv_layers, sample_keys[:-1]):
                x = attn(x, sample_key=key, deterministic=deterministic)
                x = conv(x, use_running_average=use_running_average)
            x = self.attn_layers[-1](x, sample_key=sample_keys[-1], deterministic=deterministic)
        else:
            for attn, key in zip(self.attn_layers, sample_keys):
                x = attn(x, sample_key=key, deterministic=deterministic)
        return self.norm(x)


class TransDecoder(nnx.Module):
    """Stack of ``decoder_layers`` masked-self + cross ProbSparse attention layers, plus head.

    Each layer consumes a ``(self_key, cross_key)`` pair taken as consecutive
    entries of ``sample_keys``, in order. After the stack, a final ``LayerNorm``
    (NF ``norm_layer``) precedes the output ``projection`` -- unlike
    ``TFTNet``/``ITransformerNet``, the output head lives INSIDE the decoder (NF
    ``Decoder.projection`` convention), not as a separate top-level adapter.
    """

    def __init__(
        self, *, decoder_layers: int, hidden_size: int, n_head: int, conv_hidden_size: int,
        factor: int, dropout: float, activation: str = "gelu", c_out: int, rngs: nnx.Rngs,
    ) -> None:
        self.layers = [
            TransDecoderLayer(
                hidden_size=hidden_size, n_head=n_head, conv_hidden_size=conv_hidden_size,
                factor=factor, dropout=dropout, activation=activation, rngs=rngs,
            )
            for _ in range(decoder_layers)
        ]
        self.norm = nnx.LayerNorm(hidden_size, epsilon=1e-5, rngs=rngs)
        self.projection = _torch_linear(hidden_size, c_out, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, cross: jnp.ndarray, *, sample_keys, deterministic: bool,
    ) -> jnp.ndarray:
        """x: ``[B, L_dec, hidden]``, cross: ``[B, L_enc, hidden]`` -> ``[B, L_dec, c_out]``."""
        # Static Python-int length check (trace-time, jit/vmap-safe, same class as the
        # `L_Q == L_K` assert in `_prob_attention`): a miscount here would silently index
        # `sample_keys[2 * i]` / `[2 * i + 1]` out of sync with the intended layer.
        assert len(sample_keys) == 2 * len(self.layers), "(self, cross) key pair per decoder layer"
        for i, layer in enumerate(self.layers):
            x = layer(
                x, cross, self_key=sample_keys[2 * i], cross_key=sample_keys[2 * i + 1],
                deterministic=deterministic,
            )
        return self.projection(self.norm(x))


class InformerNet(nnx.Module):
    """Full Informer backbone: embed -> distilling encoder -> generative decoder -> head.

    I/O mirrors the other Chronax neural nets: ``__call__(insample_y: [B, L, 1],
    futr_exog: [B, L+h, F] | None) -> [B, h, outputsize_multiplier]``. Encoder and
    decoder get SEPARATE ``DataEmbedding``s (NF does not share embedding weights
    between them), both fed the same future-known time-feature marks sliced to
    their own window. The decoder input is NF's generative "start token" trick:
    the last ``label_len`` steps of the raw history concatenated with ``h`` zero
    placeholders, so a single forward pass produces all ``h`` horizon steps (no
    autoregressive loop). One ``sample_key`` is split into ``n_attn_sites =
    encoder_layers + 2 * decoder_layers`` independent keys per call -- one per
    ProbSparse attention site (encoder self-attention layers, in order, then
    decoder ``(self, cross)`` pairs, in order) -- so no key is ever reused across
    sites.
    """

    def __init__(
        self, *, h, input_size, label_len, hidden_size=128, n_head=4, factor=3,
        conv_hidden_size=32, encoder_layers=2, decoder_layers=1, distil=True,
        dropout=0.05, activation="gelu", futr_exog_size=0, outputsize_multiplier=1,
        rngs: nnx.Rngs,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.label_len = label_len
        self.encoder_layers = encoder_layers
        self.futr_exog_size = futr_exog_size
        self.n_attn_sites = encoder_layers + 2 * decoder_layers

        self.enc_embedding = DataEmbedding(
            c_in=1, exog_input_size=futr_exog_size, hidden_size=hidden_size,
            dropout=dropout, rngs=rngs,
        )
        self.dec_embedding = DataEmbedding(
            c_in=1, exog_input_size=futr_exog_size, hidden_size=hidden_size,
            dropout=dropout, rngs=rngs,
        )
        self.encoder = TransEncoder(
            encoder_layers=encoder_layers, hidden_size=hidden_size, n_head=n_head,
            conv_hidden_size=conv_hidden_size, factor=factor, dropout=dropout,
            activation=activation, distil=distil, rngs=rngs,
        )
        self.decoder = TransDecoder(
            decoder_layers=decoder_layers, hidden_size=hidden_size, n_head=n_head,
            conv_hidden_size=conv_hidden_size, factor=factor, dropout=dropout,
            activation=activation, c_out=outputsize_multiplier, rngs=rngs,
        )

    def __call__(self, insample_y, futr_exog=None, *, sample_key,
                 deterministic: bool, use_running_average: bool) -> jnp.ndarray:
        insample_y = jnp.asarray(insample_y, jnp.float32)          # [B, L, 1]
        B, L = insample_y.shape[0], insample_y.shape[1]
        x_mark_enc = x_mark_dec = None
        if self.futr_exog_size > 0:
            futr_exog = jnp.asarray(futr_exog, jnp.float32)        # [B, L+h, F]
            x_mark_enc = futr_exog[:, :L]
            x_mark_dec = futr_exog[:, -(self.label_len + self.h):]
        x_dec = jnp.concatenate(
            [insample_y[:, -self.label_len:, :], jnp.zeros((B, self.h, 1), jnp.float32)], axis=1)
        keys = jax.random.split(sample_key, self.n_attn_sites)
        enc = self.enc_embedding(insample_y, x_mark_enc, deterministic)
        enc = self.encoder(enc, sample_keys=keys[: self.encoder_layers],
                           deterministic=deterministic, use_running_average=use_running_average)
        dec = self.dec_embedding(x_dec, x_mark_dec, deterministic)
        dec = self.decoder(dec, enc, sample_keys=keys[self.encoder_layers:],  # (self, cross) pairs in order
                           deterministic=deterministic)
        return dec[:, -self.h:]                                    # [B, h, c_out]
