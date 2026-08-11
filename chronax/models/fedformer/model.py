"""JAX/Flax FEDformer -- univariate, direct-decoding long-horizon forecaster.

This module implements **FEDformer** (Frequency Enhanced Decomposed Transformer)
from scratch in pure JAX / Flax (linen). It is a faithful, self-contained port
of the architecture used by ``neuralforecast`` for the univariate, no-exogenous
case -- it does NOT import from any other chronax model.

WHAT is FEDformer?
    A Transformer-style encoder/decoder for long-term time-series forecasting
    whose attention sub-layers operate in the **frequency domain** (via the FFT)
    instead of the time domain. By keeping only a small, fixed number of Fourier
    modes it attains *linear* complexity in the sequence length, while a built-in
    seasonal-trend decomposition gives it a strong global view of the series.

WHY frequency domain?
    Most real-world series are *sparse in a Fourier basis*: a handful of
    frequencies carry most of the signal's energy. Standard self-attention costs
    O(L^2); transforming to frequency space and learning filters on a few modes
    costs O(L log L) with a tiny constant -- effectively linear in practice.

HOW is it organised (build order, bottom-up)?
    Block 1  series decomposition primitives (moving_avg, series_decomp)
    Block 2  frequency-mode selection         (get_frequency_modes)
    Block 3  FourierBlock           (FEB-f, replaces self-attention)
    Block 4  FourierCrossAttention  (FEA-f, replaces cross-attention)
    Block 5  MultiHeadProjection    (Q/K/V projection wrapper around FEB / FEA)
    Block 6  SeasonalLayerNorm
    Block 7  TokenEmbedding
    Block 8  EncoderLayer
    Block 9  Encoder
    Block 10 DecoderLayer
    Block 11 Decoder
    Block 12 FEDformerConfig
    Block 13 FEDformerModel (wires everything together)

Reference:
    Zhou, T. et al. "FEDformer: Frequency Enhanced Decomposed Transformer for
    Long-term Series Forecasting." ICML/AAAI 2022/2023. arXiv:2201.12740.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import flax.linen as fnn
import jax
import jax.numpy as jnp
import numpy as np  # host-side only: deterministic Fourier-mode index selection


# ===========================================================================
# Small shared helpers
# ===========================================================================

# Torch-matching initializers (parity with neuralforecast / PyTorch defaults).
# TokenEmbedding uses kaiming-normal (NF override); Linear/other Conv1d use
# torch's default uniform U(-1/sqrt(fan_in), 1/sqrt(fan_in)).


class _TorchUniformInit:
    """``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` — torch ``nn.Linear`` / ``nn.Conv1d`` default."""

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _KaimingNormalConvInit:
    """TokenEmbedding: torch ``kaiming_normal_(fan_in, leaky_relu)``."""

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        gain = math.sqrt(2.0 / (1.0 + 0.01 ** 2))
        return (gain / math.sqrt(self.fan_in)) * jax.random.normal(key, shape, dtype)


def _torch_dense(features: int, fan_in: int, *, name: str):
    init = _TorchUniformInit(fan_in)
    return fnn.Dense(features, kernel_init=init, bias_init=init, name=name)


def _torch_conv(
    features: int,
    fan_in: int,
    *,
    kernel_size: tuple,
    name: str,
    padding: str = "SAME",
    token_embed: bool = False,
):
    init = _KaimingNormalConvInit(fan_in) if token_embed else _TorchUniformInit(fan_in)
    return fnn.Conv(
        features=features,
        kernel_size=kernel_size,
        padding=padding,
        use_bias=False,
        kernel_init=init,
        name=name,
    )


# FEDformer hard-codes the multi-head count to 8 inside its learnable spectral
# weight tensor (its leading dimension is literally 8). We expose the same
# constant so every block agrees on the head count.
_N_HEADS_FIXED = 8


def _activation(name: str, x: jnp.ndarray) -> jnp.ndarray:
    """Pointwise nonlinearity used by the position-wise feed-forward network.

    Only ``relu`` and ``gelu`` are supported, matching neuralforecast's guard.
    """
    if name == "relu":
        return jax.nn.relu(x)
    # ``approximate=False`` -> exact erf-based GELU (PyTorch's default GELU).
    return jax.nn.gelu(x, approximate=False)


# ===========================================================================
# BLOCK 1 -- Series Decomposition (pure-JAX primitives)
# ===========================================================================
#
# WHAT: Split a signal x into a slow-moving *trend* and a residual *seasonal*
#       part:   x = seasonal + trend,   trend = moving_average(x).
# WHY:  FEDformer is a "deep decomposition" architecture. After every attention
#       and feed-forward sub-layer it re-runs this decomposition so the network
#       can model the smooth trend and the oscillatory seasonal part separately
#       -- this is what gives it a robust *global* view of long series.
# HOW:  The trend is a centred moving average. We pad both ends by repeating the
#       boundary values (so the output keeps the same length T) and compute the
#       windowed mean efficiently with a cumulative sum (O(T) instead of O(T*k)).


def moving_avg(x: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
    """Centred moving-average smoother on ``[B, T, C]`` -> ``[B, T, C]``.

    Args:
        x: Input series, shape ``[batch, time, channels]``.
        kernel_size: Averaging window length (use an odd number, e.g. 25, so
            the window is symmetric and the output length is exactly ``T``).

    Returns:
        The trend component: a same-length, smoothed version of ``x``.

    Implementation detail:
        We replicate ``neuralforecast``'s ``MovingAvg`` boundary handling --
        pad ``(kernel_size - 1) // 2`` steps on each side using the first/last
        observed value (edge padding) -- then take the running mean. The
        cumulative-sum trick computes every window mean in a single vectorised
        pass, which is both fast and cleanly differentiable under XLA.
    """
    pad = (kernel_size - 1) // 2
    # Repeat the boundary samples so the smoothed trend does not "fall off" at
    # the edges (a plain zero-pad would bias the ends toward zero).
    front = jnp.repeat(x[:, :1, :], pad, axis=1)   # [B, pad, C]
    end = jnp.repeat(x[:, -1:, :], pad, axis=1)     # [B, pad, C]
    xp = jnp.concatenate([front, x, end], axis=1)   # [B, T + 2*pad, C]

    # Windowed sum via prefix sums: sum(i:i+k) = cs[i+k] - cs[i].
    cs = jnp.cumsum(xp, axis=1)
    cs = jnp.concatenate([jnp.zeros_like(cs[:, :1, :]), cs], axis=1)  # prepend 0
    windowed_sum = cs[:, kernel_size:, :] - cs[:, :-kernel_size, :]
    return windowed_sum / kernel_size


def series_decomp(
    x: jnp.ndarray, kernel_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose ``x`` into ``(seasonal, trend)``.

    ``trend`` is the moving average; ``seasonal`` is whatever remains after
    removing the trend. Returned in the same ``(seasonal, trend)`` order that
    the encoder/decoder layers expect.
    """
    trend = moving_avg(x, kernel_size)
    seasonal = x - trend
    return seasonal, trend


# ===========================================================================
# BLOCK 2 -- Frequency-Mode Selection
# ===========================================================================
#
# WHAT: Decide *which* Fourier frequencies a spectral block will keep.
# WHY:  This is the single design choice that makes FEDformer linear-complexity.
#       A length-L signal has L//2 + 1 real-FFT frequencies; instead of using
#       all of them we keep only `modes` of them (e.g. 64). Everything else --
#       the learnable weights, the einsums -- then scales with `modes`, not L.
# HOW:  Two strategies (matching the paper / neuralforecast):
#         * "random": sample `modes` distinct frequency indices. The paper shows
#           (Thm. 1) random selection still bounds the approximation error and
#           empirically beats always taking the lowest frequencies, because real
#           signals also carry information in mid/high bands.
#         * else ("lowest"): take the first `modes` (lowest) frequencies.
#       Selection happens once, at model-construction time, on the host with a
#       *seeded* RNG so the chosen indices are reproducible and identical across
#       parameter init and every forward pass.


def get_frequency_modes(
    seq_len: int,
    modes: int = 64,
    mode_select_method: str = "random",
    seed: int = 0,
) -> Tuple[int, ...]:
    """Return a sorted tuple of frequency indices to retain.

    Args:
        seq_len: Length of the (time-domain) sequence this block will see.
        modes: Desired number of frequencies to keep. Automatically capped at
            ``seq_len // 2`` (the number of non-trivial rFFT bins).
        mode_select_method: ``"random"`` for seeded random sampling, anything
            else for the lowest-``modes`` frequencies.
        seed: Seed for the local RNG so random selection is deterministic.

    Returns:
        Ascending tuple of integer frequency indices, each in ``[0, seq_len//2)``.
    """
    # Never request more modes than there are usable rFFT bins.
    modes = min(modes, max(1, seq_len // 2))
    if mode_select_method == "random":
        # Local RNG (does NOT touch global numpy state) -> reproducible indices.
        rng = np.random.RandomState(seed)
        index = list(range(0, seq_len // 2))
        rng.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return tuple(index)


def _complex_uniform_init(scale: float):
    """Flax param initialiser: ``scale * Uniform[0, 1)`` (one real array).

    FEDformer's spectral weights are complex. Flax/JAX cannot store complex
    parameters directly, so each complex weight is represented by *two* real
    parameters (real and imaginary parts), each initialised with this function.
    The ``scale = 1 / (in_channels * out_channels)`` factor mirrors the upstream
    PyTorch initialisation, keeping the spectral filters small at the start.
    """

    def init(key: jax.Array, shape: Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
        return scale * jax.random.uniform(key, shape, dtype)

    return init


# ===========================================================================
# BLOCK 3 -- FourierBlock (FEB-f): the self-attention replacement
# ===========================================================================
#
# WHAT: A "Frequency Enhanced Block". Given a multi-head sequence it transforms
#       to the frequency domain, keeps only the selected modes, multiplies them
#       by a learnable complex tensor (a per-mode, per-head linear map across the
#       feature dimension), then transforms back to the time domain.
# WHY:  This is FEDformer's drop-in replacement for self-attention. A standard
#       attention layer mixes information across time at O(L^2) cost. Here the
#       FFT moves *global* temporal structure into frequencies, and a cheap
#       learnable filter on a few modes captures the dominant patterns at O(L log L).
# HOW (shapes, with H=8 heads and E = hidden_size // 8 features-per-head):
#       q : [B, L, H, E]
#         -> transpose to [B, H, E, L]
#         -> rfft over time  -> [B, H, E, L//2+1] (complex)
#         -> gather selected modes -> [B, H, E, n_modes]
#         -> einsum with weights [H, E, E_out, n_modes] -> [B, H, E_out, n_modes]
#         -> scatter back into a zero spectrum, irfft -> [B, H, E_out, L]
#         -> transpose back to [B, L, H, E_out]
#
# NOTE on faithfulness: upstream FEDformer writes the per-mode outputs into the
# *lowest* output-spectrum bins (positions 0..n_modes-1), regardless of which
# input frequencies were read. We reproduce that exact behaviour so results line
# up with the neuralforecast reference we benchmark against.


class FourierBlock(fnn.Module):
    """Frequency Enhanced Block (FEB-f) -- self-attention in Fourier space.

    Attributes:
        in_channels: Total input feature width (== ``hidden_size``).
        out_channels: Total output feature width (== ``hidden_size``).
        seq_len: Sequence length used to pick the frequency modes.
        modes: Number of Fourier modes to keep.
        mode_select: ``"random"`` or ``"lowest"``.
        mode_seed: Seed for deterministic random mode selection.
    """

    in_channels: int
    out_channels: int
    seq_len: int
    modes: int = 64
    mode_select: str = "random"
    mode_seed: int = 0

    def setup(self) -> None:
        # Choose which frequencies to keep (done once, deterministically).
        self.index = get_frequency_modes(
            self.seq_len, self.modes, self.mode_select, self.mode_seed
        )
        n_modes = len(self.index)

        head = _N_HEADS_FIXED
        e_in = self.in_channels // head    # features per head (in)
        e_out = self.out_channels // head  # features per head (out)
        scale = 1.0 / (self.in_channels * self.out_channels)

        # Learnable complex spectral filter, stored as two real tensors.
        # Shape [head, e_in, e_out, n_modes]: an independent (e_in -> e_out)
        # linear map for every head and every retained frequency.
        self.weights_real = self.param(
            "weights_real", _complex_uniform_init(scale), (head, e_in, e_out, n_modes)
        )
        self.weights_imag = self.param(
            "weights_imag", _complex_uniform_init(scale), (head, e_in, e_out, n_modes)
        )

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,  # unused (self-attention reads only the query stream)
        v: jnp.ndarray,  # unused
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, L, H, E = q.shape

        # [B, L, H, E] -> [B, H, E, L] so the FFT runs along the time axis.
        x = jnp.transpose(q, (0, 2, 3, 1))

        # Real FFT over time: real input -> complex spectrum [B, H, E, L//2+1].
        x_ft = jnp.fft.rfft(x, axis=-1)

        # Keep only the selected frequencies: [B, H, E, n_modes].
        idx = jnp.asarray(self.index)
        x_sel = x_ft[..., idx]

        # Reassemble the complex filter and apply it.
        # "bhix,hiox->bhox": for each batch b, head h and mode x, map the
        # input feature axis i to the output feature axis o.
        weights = self.weights_real + 1j * self.weights_imag
        out_sel = jnp.einsum("bhix,hiox->bhox", x_sel, weights)  # [B, H, E_out, n_modes]

        # Place the processed modes into the lowest bins of a zero spectrum
        # (faithful to upstream), then invert back to the time domain.
        freq_len = L // 2 + 1
        pad = freq_len - out_sel.shape[-1]
        zeros = jnp.zeros(out_sel.shape[:-1] + (pad,), dtype=out_sel.dtype)
        out_ft = jnp.concatenate([out_sel, zeros], axis=-1)  # [B, H, E_out, L//2+1]
        out = jnp.fft.irfft(out_ft, n=L, axis=-1)            # [B, H, E_out, L]

        # Back to [B, L, H, E_out] for the multi-head wrapper.
        return jnp.transpose(out, (0, 3, 1, 2))


# ===========================================================================
# BLOCK 4 -- FourierCrossAttention (FEA-f): the cross-attention replacement
# ===========================================================================
#
# WHAT: A "Frequency Enhanced Attention". It performs attention between two
#       sequences (decoder queries vs encoder keys/values) entirely in the
#       frequency domain.
# WHY:  This is FEDformer's replacement for encoder-decoder cross-attention. It
#       lets the decoder's frequencies attend to the encoder's frequencies, so
#       the forecast horizon is conditioned on the historical context -- again at
#       linear cost because only the selected modes participate.
# HOW (H=8 heads, E features-per-head; mq / mkv selected modes for query / key):
#       1. rfft query & key, gather their selected modes -> Q~ [B,H,E,mq], K~ [B,H,E,mkv]
#       2. frequency "attention scores": einsum over the feature axis
#             "bhex,bhey->bhxy"  -> A [B, H, mq, mkv]
#          pass A through tanh (default) or softmax(|A|)
#       3. apply scores to keys:  "bhxy,bhey->bhex" -> [B, H, E, mq]
#       4. learnable complex map:  "bhex,heox->bhox" -> [B, H, E_out, mq]
#       5. scatter results back to the *actual* query frequencies, irfft,
#          and normalise by 1/(in_channels*out_channels).
#
# Unlike FEB-f, here the outputs ARE written back at the true selected query
# frequencies (this matches the upstream FEA-f implementation).


class FourierCrossAttention(fnn.Module):
    """Frequency Enhanced Attention (FEA-f) -- cross-attention in Fourier space.

    Attributes:
        in_channels: Input feature width (== ``hidden_size``).
        out_channels: Output feature width (== ``hidden_size``).
        seq_len_q: Decoder (query) sequence length, for query-mode selection.
        seq_len_kv: Encoder (key/value) sequence length, for key-mode selection.
        modes: Number of Fourier modes to keep on each side.
        mode_select: ``"random"`` or ``"lowest"``.
        activation: Frequency-domain score nonlinearity: ``"tanh"`` or ``"softmax"``.
        mode_seed: Seed for deterministic random mode selection.
    """

    in_channels: int
    out_channels: int
    seq_len_q: int
    seq_len_kv: int
    modes: int = 64
    mode_select: str = "random"
    activation: str = "tanh"
    mode_seed: int = 0

    def setup(self) -> None:
        # Query and key/value generally have *different* lengths, so each gets
        # its own set of frequency modes.
        self.index_q = get_frequency_modes(
            self.seq_len_q, self.modes, self.mode_select, self.mode_seed
        )
        self.index_kv = get_frequency_modes(
            self.seq_len_kv, self.modes, self.mode_select, self.mode_seed + 1
        )

        head = _N_HEADS_FIXED
        e_in = self.in_channels // head
        e_out = self.out_channels // head
        scale = 1.0 / (self.in_channels * self.out_channels)

        # Learnable complex map applied after the frequency-domain attention.
        # One (e_in -> e_out) map per head and per *query* mode.
        n_modes_q = len(self.index_q)
        self.weights_real = self.param(
            "weights_real", _complex_uniform_init(scale), (head, e_in, e_out, n_modes_q)
        )
        self.weights_imag = self.param(
            "weights_imag", _complex_uniform_init(scale), (head, e_in, e_out, n_modes_q)
        )

    def __call__(
        self,
        q: jnp.ndarray,     # decoder stream  [B, Lq, H, E]
        k: jnp.ndarray,     # encoder stream  [B, Lk, H, E]
        v: jnp.ndarray,     # encoder stream (unused; FEA reuses k as values)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, Lq, H, E = q.shape

        # Move time to the last axis for the FFT.
        xq = jnp.transpose(q, (0, 2, 3, 1))  # [B, H, E, Lq]
        xk = jnp.transpose(k, (0, 2, 3, 1))  # [B, H, E, Lk]

        # rfft + gather selected modes for each side.
        iq = jnp.asarray(self.index_q)
        ikv = jnp.asarray(self.index_kv)
        xq_ft = jnp.fft.rfft(xq, axis=-1)[..., iq]   # [B, H, E, mq]
        xk_ft = jnp.fft.rfft(xk, axis=-1)[..., ikv]  # [B, H, E, mkv]

        # (1) Frequency-domain attention scores by contracting the feature axis.
        scores = jnp.einsum("bhex,bhey->bhxy", xq_ft, xk_ft)  # [B, H, mq, mkv] complex
        if self.activation == "tanh":
            scores = jnp.tanh(scores)  # complex tanh, keeps phase information
        elif self.activation == "softmax":
            # Softmax over key modes using magnitudes, then re-cast to complex.
            scores = jax.nn.softmax(jnp.abs(scores), axis=-1).astype(xq_ft.dtype)
        else:
            raise ValueError(
                f"activation must be 'tanh' or 'softmax', got {self.activation!r}."
            )

        # (2) Apply scores to the key spectrum -> attended values per query mode.
        attended = jnp.einsum("bhxy,bhey->bhex", scores, xk_ft)  # [B, H, E, mq]

        # (3) Learnable complex feature map.
        weights = self.weights_real + 1j * self.weights_imag
        out_modes = jnp.einsum("bhex,heox->bhox", attended, weights)  # [B, H, E_out, mq]

        # (4) Scatter to the TRUE query frequencies in a zero spectrum, invert.
        freq_len = Lq // 2 + 1
        out_ft = jnp.zeros(out_modes.shape[:-1] + (freq_len,), dtype=out_modes.dtype)
        out_ft = out_ft.at[..., iq].set(out_modes)  # [B, H, E_out, Lq//2+1]
        out = jnp.fft.irfft(out_ft, n=Lq, axis=-1)
        out = out / self.in_channels / self.out_channels  # upstream normalisation

        return jnp.transpose(out, (0, 3, 1, 2))  # [B, Lq, H, E_out]


# ===========================================================================
# BLOCK 5 -- MultiHeadProjection (Q/K/V projection wrapper)
# ===========================================================================
#
# WHAT: The standard "multi-head" plumbing around a spectral block. It linearly
#       projects queries/keys/values, splits them into heads, hands them to an
#       inner block (a FourierBlock or FourierCrossAttention), then merges the
#       heads and applies an output projection.
# WHY:  Multiple heads let the model learn several independent spectral filters
#       in parallel and recombine them, exactly as multi-head attention does.
#       (neuralforecast calls this ``AutoCorrelationLayer``; we use the clearer
#       name ``MultiHeadProjection``.)
# HOW:  Dense projections map hidden_size -> hidden_size, reshape to
#       [B, L, H, E], call the inner block, reshape back, then a final Dense.


class MultiHeadProjection(fnn.Module):
    """Wrap a spectral block with multi-head Q/K/V/output projections.

    Attributes:
        inner: The spectral mixing module (FourierBlock or FourierCrossAttention).
            Its ``__call__`` must accept ``(q, k, v, deterministic)`` with the
            per-head layout ``[B, L, H, E]``.
        hidden_size: Total feature width.
        n_heads: Number of heads (must be 8 for FEDformer).
    """

    inner: fnn.Module
    hidden_size: int
    n_heads: int

    @fnn.compact
    def __call__(
        self,
        queries: jnp.ndarray,
        keys: jnp.ndarray,
        values: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads

        # Project then split into heads: [B, L, hidden] -> [B, L, H, hidden//H].
        fan = self.hidden_size
        q = _torch_dense(self.hidden_size, fan, name="q_proj")(queries).reshape(B, L, H, -1)
        k = _torch_dense(self.hidden_size, fan, name="k_proj")(keys).reshape(B, S, H, -1)
        v = _torch_dense(self.hidden_size, fan, name="v_proj")(values).reshape(B, S, H, -1)

        # Spectral mixing (self- or cross-).
        out = self.inner(q, k, v, deterministic)  # [B, L, H, E_out]

        # Merge heads and project out.
        out = out.reshape(B, L, -1)
        return _torch_dense(self.hidden_size, fan, name="out_proj")(out)


# ===========================================================================
# BLOCK 6 -- SeasonalLayerNorm
# ===========================================================================
#
# WHAT: LayerNorm specialised for the seasonal stream: a standard LayerNorm
#       followed by subtracting the per-feature mean over the time axis.
# WHY:  Seasonal components are, by definition, zero-mean fluctuations around the
#       trend. A vanilla LayerNorm can introduce a small constant offset; removing
#       the time-mean afterwards keeps the seasonal signal centred at zero so it
#       does not leak into (and double-count with) the trend pathway.
# HOW:  out = LN(x) - mean_over_time(LN(x)).


class SeasonalLayerNorm(fnn.Module):
    """LayerNorm then remove the time-mean (keeps the seasonal part zero-mean)."""

    hidden_size: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_hat = fnn.LayerNorm(epsilon=1e-5, name="ln")(x)
        return x_hat - jnp.mean(x_hat, axis=1, keepdims=True)


# ===========================================================================
# BLOCK 7 -- TokenEmbedding
# ===========================================================================
#
# WHAT: Lift the raw 1-channel series into the model's hidden dimension with a
#       width-3 caus*ish* 1-D convolution (circular padding), followed by dropout.
# WHY:  A 1-D conv with kernel 3 sees a small temporal neighbourhood, so each
#       embedded token already encodes local shape (slope/curvature), which is a
#       better starting representation than a pointwise linear map. Circular
#       padding avoids edge artifacts on periodic-ish series. This matches the
#       embedding used throughout the Autoformer/Informer/FEDformer family.
# HOW:  Conv1D(in=1 -> out=hidden, k=3, circular, no bias, kaiming-normal) + Dropout.


class TokenEmbedding(fnn.Module):
    """Conv1D(k=3, circular) value embedding + dropout."""

    hidden_size: int
    dropout_rate: float

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # fan_in = kernel_size * in_channels; univariate c_in=1 -> 3.
        c_in = x.shape[-1]
        y = _torch_conv(
            self.hidden_size, 3 * c_in,
            kernel_size=(3,), padding="CIRCULAR", name="token_conv",
            token_embed=True,
        )(x)
        return fnn.Dropout(rate=self.dropout_rate, name="emb_drop")(
            y, deterministic=deterministic
        )


# ===========================================================================
# BLOCK 8 -- EncoderLayer
# ===========================================================================
#
# WHAT: One encoder block with progressive decomposition:
#         self-FEB (+residual) -> decomp -> conv FFN (+residual) -> decomp.
# WHY:  The FEB mixes information across time in frequency space; the two
#       decompositions strip the trend after each sub-layer so only the seasonal
#       residual is propagated. The encoder therefore refines a clean seasonal
#       representation of the history (its trend is intentionally discarded --
#       the decoder reconstructs the forecast trend separately).
# HOW:  See the call body; the Conv FFN is the usual position-wise MLP expressed
#       as two width-1 convolutions (hidden -> conv_hidden -> hidden).


class EncoderLayer(fnn.Module):
    """FEDformer encoder layer (FEB self-attention + conv FFN, both decomposed)."""

    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    moving_avg_window: int
    dropout_rate: float
    activation: str
    # Spectral-block configuration:
    seq_len: int
    modes: int
    mode_select: str
    mode_seed: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # --- Frequency-enhanced self-attention (FEB-f) with residual ---
        feb = FourierBlock(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            seq_len=self.seq_len,
            modes=self.modes,
            mode_select=self.mode_select,
            mode_seed=self.mode_seed,
            name="feb",
        )
        attn = MultiHeadProjection(
            inner=feb, hidden_size=self.hidden_size, n_heads=self.n_heads, name="attn"
        )
        new_x = attn(x, x, x, deterministic)
        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop1")(
            new_x, deterministic=deterministic
        )
        # Strip the trend introduced so far; keep the seasonal residual.
        x, _ = series_decomp(x, self.moving_avg_window)

        # --- Position-wise feed-forward (two width-1 convolutions) ---
        y = _torch_conv(
            self.conv_hidden_size, self.hidden_size,
            kernel_size=(1,), name="conv1",
        )(x)
        y = _activation(self.activation, y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop2")(y, deterministic=deterministic)
        y = _torch_conv(
            self.hidden_size, self.conv_hidden_size,
            kernel_size=(1,), name="conv2",
        )(y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop3")(y, deterministic=deterministic)

        # Second decomposition on the (residual + FFN) sum.
        res, _ = series_decomp(x + y, self.moving_avg_window)
        return res


# ===========================================================================
# BLOCK 9 -- Encoder (stack of EncoderLayers + final SeasonalLayerNorm)
# ===========================================================================
#
# WHAT: Stack ``n_layers`` encoder layers, then apply SeasonalLayerNorm.
# WHY:  Depth lets the model compose increasingly abstract spectral features of
#       the history; the final seasonal norm stabilises the representation that
#       the decoder will cross-attend to.


class Encoder(fnn.Module):
    """Stack of EncoderLayers followed by a seasonal LayerNorm."""

    n_layers: int
    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    moving_avg_window: int
    dropout_rate: float
    activation: str
    seq_len: int
    modes: int
    mode_select: str
    mode_seed: int

    @fnn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for i in range(self.n_layers):
            x = EncoderLayer(
                hidden_size=self.hidden_size,
                conv_hidden_size=self.conv_hidden_size,
                n_heads=self.n_heads,
                moving_avg_window=self.moving_avg_window,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select=self.mode_select,
                # Distinct seed per layer -> different random mode sets per layer.
                mode_seed=self.mode_seed + i,
                name=f"layer_{i}",
            )(x, deterministic)
        return SeasonalLayerNorm(self.hidden_size, name="norm")(x)


# ===========================================================================
# BLOCK 10 -- DecoderLayer
# ===========================================================================
#
# WHAT: One decoder block with THREE decompositions:
#         self-FEB  -> decomp(trend1)
#         cross-FEA -> decomp(trend2)
#         conv FFN  -> decomp(trend3)
#         trend_proj( trend1 + trend2 + trend3 )
# WHY:  The decoder reconstructs BOTH parts of the forecast. The seasonal stream
#       flows through the spectral/FFN sub-layers; each sub-layer can introduce a
#       little trend, which the three decompositions peel off. Those trend pieces
#       are summed and projected to the output width, then accumulated into the
#       running forecast trend. Cross-attention (FEA) is where the decoder looks
#       back at the encoded history.
# HOW:  Mirrors the encoder layer but adds the cross-attention sub-layer and the
#       trend-projection convolution (kernel 3, circular) that maps hidden->c_out.


class DecoderLayer(fnn.Module):
    """FEDformer decoder layer (self-FEB, cross-FEA, conv FFN; trend extracted)."""

    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    c_out: int
    moving_avg_window: int
    dropout_rate: float
    activation: str
    # Spectral-block configuration:
    self_seq_len: int       # length seen by the decoder self-FEB
    cross_seq_len_q: int    # query length for cross-FEA (== decoder length)
    cross_seq_len_kv: int   # key length for cross-FEA   (== encoder length)
    modes: int
    mode_select: str
    fea_activation: str
    mode_seed: int

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cross: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # --- (a) Self-attention in frequency space (+residual) -> decomp ---
        self_feb = FourierBlock(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            seq_len=self.self_seq_len,
            modes=self.modes,
            mode_select=self.mode_select,
            mode_seed=self.mode_seed,
            name="self_feb",
        )
        self_attn = MultiHeadProjection(
            inner=self_feb, hidden_size=self.hidden_size, n_heads=self.n_heads, name="self_attn"
        )
        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop1")(
            self_attn(x, x, x, deterministic), deterministic=deterministic
        )
        x, trend1 = series_decomp(x, self.moving_avg_window)

        # --- (b) Cross-attention to the encoder memory (+residual) -> decomp ---
        cross_fea = FourierCrossAttention(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            seq_len_q=self.cross_seq_len_q,
            seq_len_kv=self.cross_seq_len_kv,
            modes=self.modes,
            mode_select=self.mode_select,
            activation=self.fea_activation,
            # Offset the seed so cross-attn modes differ from self-attn modes.
            mode_seed=self.mode_seed + 100,
            name="cross_fea",
        )
        cross_attn = MultiHeadProjection(
            inner=cross_fea, hidden_size=self.hidden_size, n_heads=self.n_heads, name="cross_attn"
        )
        x = x + fnn.Dropout(rate=self.dropout_rate, name="drop2")(
            cross_attn(x, cross, cross, deterministic), deterministic=deterministic
        )
        x, trend2 = series_decomp(x, self.moving_avg_window)

        # --- (c) Position-wise feed-forward (+residual) -> decomp ---
        y = _torch_conv(
            self.conv_hidden_size, self.hidden_size,
            kernel_size=(1,), name="conv1",
        )(x)
        y = _activation(self.activation, y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop3")(y, deterministic=deterministic)
        y = _torch_conv(
            self.hidden_size, self.conv_hidden_size,
            kernel_size=(1,), name="conv2",
        )(y)
        y = fnn.Dropout(rate=self.dropout_rate, name="drop4")(y, deterministic=deterministic)
        x, trend3 = series_decomp(x + y, self.moving_avg_window)

        # --- Trend pathway: sum the three trend pieces and project to c_out ---
        residual_trend = trend1 + trend2 + trend3
        residual_trend = _torch_conv(
            self.c_out, 3 * self.hidden_size,
            kernel_size=(3,), padding="CIRCULAR", name="trend_proj",
        )(residual_trend)
        return x, residual_trend


# ===========================================================================
# BLOCK 11 -- Decoder (stack + seasonal norm + final projection)
# ===========================================================================
#
# WHAT: Stack ``n_layers`` decoder layers, accumulate their trend outputs into a
#       running trend, seasonal-norm the seasonal stream, and project it to c_out.
# WHY:  The decoder returns two signals -- the refined seasonal part and the
#       accumulated trend -- which the top-level model adds to form the forecast.
# HOW:  trend starts from the decoder's trend initialisation and grows by each
#       layer's projected residual trend.


class Decoder(fnn.Module):
    """Stack of DecoderLayers + seasonal LayerNorm + final linear projection."""

    n_layers: int
    hidden_size: int
    conv_hidden_size: int
    n_heads: int
    c_out: int
    moving_avg_window: int
    dropout_rate: float
    activation: str
    self_seq_len: int
    cross_seq_len_q: int
    cross_seq_len_kv: int
    modes: int
    mode_select: str
    fea_activation: str
    mode_seed: int

    @fnn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        cross: jnp.ndarray,
        trend: jnp.ndarray,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for i in range(self.n_layers):
            x, residual_trend = DecoderLayer(
                hidden_size=self.hidden_size,
                conv_hidden_size=self.conv_hidden_size,
                n_heads=self.n_heads,
                c_out=self.c_out,
                moving_avg_window=self.moving_avg_window,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                self_seq_len=self.self_seq_len,
                cross_seq_len_q=self.cross_seq_len_q,
                cross_seq_len_kv=self.cross_seq_len_kv,
                modes=self.modes,
                mode_select=self.mode_select,
                fea_activation=self.fea_activation,
                mode_seed=self.mode_seed + i,
                name=f"layer_{i}",
            )(x, cross, deterministic)
            # Accumulate the forecast trend layer by layer.
            trend = trend + residual_trend

        x = SeasonalLayerNorm(self.hidden_size, name="norm")(x)
        x = _torch_dense(self.c_out, self.hidden_size, name="projection")(x)
        return x, trend


# ===========================================================================
# BLOCK 12 -- FEDformerConfig
# ===========================================================================


@dataclass(frozen=True)
class FEDformerConfig:
    """Hyperparameters for :class:`FEDformerModel`.

    Args:
        h: Forecast horizon (number of future steps to predict).
        input_size: Length of the historical context fed to the encoder.
        hidden_size: Embedding / attention hidden dimension.
        n_heads: Number of heads. FEDformer requires exactly 8 (its spectral
            weight tensor is built with a leading dimension of 8).
        modes: Number of Fourier modes kept by each spectral block.
        mode_select: Mode-selection strategy: ``"random"`` or ``"lowest"``.
        moving_avg_window: Kernel size for the trend moving-average filter.
        encoder_layers: Number of stacked encoder layers.
        decoder_layers: Number of stacked decoder layers.
        conv_hidden_size: Hidden channels of the position-wise conv FFN.
        decoder_input_size_multiplier: Fraction of ``input_size`` used as the
            decoder start-token ("label") length; must be in ``(0, 1)``.
        dropout: Dropout rate applied throughout (training only).
        activation: Conv-FFN nonlinearity -- ``"relu"`` or ``"gelu"``.
        fea_activation: Frequency cross-attention score nonlinearity --
            ``"tanh"`` or ``"softmax"``.
        random_seed: Base seed for deterministic Fourier-mode selection.
        futr_exog_size: Number of future-known exogenous channels (0 = none).
    """

    h: int = 24
    input_size: int = 96
    hidden_size: int = 128
    n_heads: int = 8
    modes: int = 64
    mode_select: str = "random"
    moving_avg_window: int = 25
    encoder_layers: int = 2
    decoder_layers: int = 1
    conv_hidden_size: int = 32
    decoder_input_size_multiplier: float = 0.5
    dropout: float = 0.05
    activation: str = "gelu"
    fea_activation: str = "tanh"
    random_seed: int = 1
    futr_exog_size: int = 0

    def __post_init__(self) -> None:
        for name in ("h", "input_size", "hidden_size", "n_heads", "modes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")
        if self.futr_exog_size < 0:
            raise ValueError(f"futr_exog_size must be >= 0, got {self.futr_exog_size}.")
        # FEDformer's learnable spectral weights hard-code 8 heads.
        if self.n_heads != _N_HEADS_FIXED:
            raise ValueError(
                f"FEDformer requires n_heads == {_N_HEADS_FIXED}, got {self.n_heads}."
            )
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.activation not in ("relu", "gelu"):
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {self.activation!r}."
            )
        if self.fea_activation not in ("tanh", "softmax"):
            raise ValueError(
                f"fea_activation must be 'tanh' or 'softmax', got {self.fea_activation!r}."
            )
        if self.mode_select not in ("random", "lowest"):
            raise ValueError(
                f"mode_select must be 'random' or 'lowest', got {self.mode_select!r}."
            )
        ll = int(math.ceil(self.input_size * self.decoder_input_size_multiplier))
        if ll <= 0 or ll >= self.input_size:
            raise ValueError(
                f"decoder_input_size_multiplier={self.decoder_input_size_multiplier} "
                f"yields label_len={ll}; must be in (0, input_size)."
            )

    @property
    def label_len(self) -> int:
        """Decoder start-token length derived from ``decoder_input_size_multiplier``."""
        return int(math.ceil(self.input_size * self.decoder_input_size_multiplier))


# ===========================================================================
# BLOCK 13 -- FEDformerModel (top-level: wires every block together)
# ===========================================================================
#
# WHAT: The full univariate FEDformer. Maps a history window [B, input_size, 1]
#       to a forecast [B, h, 1].
# WHY / HOW (the end-to-end flow):
#   1. Decompose the history into seasonal & trend. Build the decoder inputs:
#        - trend init  = [ last label_len trend steps | repeated series mean (h) ]
#        - season init = [ last label_len season steps | zeros (h) ]
#      The decoder thus starts from the recent past and "extends" it by h steps,
#      with the mean providing a sensible trend prior over the horizon.
#   2. Encoder: embed the raw history and run the encoder stack -> memory.
#   3. Decoder: embed the seasonal init, run the decoder stack while cross-
#      attending to the encoder memory; it returns (seasonal_part, trend_part).
#   4. Forecast = (trend_part + seasonal_part), take the last h steps.


class FEDformerModel(fnn.Module):
    """Univariate FEDformer forecaster built with ``flax.linen``.

    Forward pass: ``[B, input_size, 1] -> [B, h, 1]``.

    When ``config.futr_exog_size > 0``, pass ``futr_exog`` of shape
    ``[B, input_size + h, F]``; a bias-free temporal embedding is added to the
    token embeddings (NF FEDformer). When ``futr_exog_size == 0``, no temporal
    modules are created and ``futr_exog`` is ignored — identical to the
    no-exogenous path.

    During training pass ``deterministic=False`` and supply a ``"dropout"`` RNG
    via ``rngs={"dropout": key}`` in ``model.apply(...)``.
    """

    config: FEDformerConfig

    @fnn.compact
    def __call__(
        self,
        insample_y: jnp.ndarray,
        futr_exog: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        b, _, c = insample_y.shape
        h, label_len = cfg.h, cfg.label_len
        c_out = 1  # univariate point forecast

        # --- (1) Decomposition-based decoder initialisation (y-only) ---
        # Horizon trend prior = the historical mean, broadcast over h steps.
        mean = jnp.broadcast_to(
            jnp.mean(insample_y, axis=1, keepdims=True), (b, h, c)
        )
        zeros = jnp.zeros((b, h, c), dtype=insample_y.dtype)
        seasonal_init, trend_init = series_decomp(insample_y, cfg.moving_avg_window)
        # Seed the decoder with the last `label_len` steps, then extend by h.
        trend_init = jnp.concatenate([trend_init[:, -label_len:, :], mean], axis=1)
        seasonal_init = jnp.concatenate(
            [seasonal_init[:, -label_len:, :], zeros], axis=1
        )

        # --- (2) Encoder ---
        # F=0 keeps TokenEmbedding (param-identical to prior). F>0 inlines
        # value embed + temporal add + dropout (NF DataEmbedding order).
        if cfg.futr_exog_size > 0:
            if futr_exog is None:
                raise ValueError(
                    f"futr_exog of shape [B, L+h, {cfg.futr_exog_size}] is required "
                    f"when futr_exog_size={cfg.futr_exog_size}."
                )
            mark_enc = futr_exog[:, : cfg.input_size, :]
            enc_out = _torch_conv(
                cfg.hidden_size, 3 * c,
                kernel_size=(3,), padding="CIRCULAR", name="enc_embedding",
                token_embed=True,
            )(insample_y)
            enc_out = enc_out + fnn.Dense(
                cfg.hidden_size, use_bias=False, name="enc_temporal"
            )(mark_enc.astype(jnp.float32))
            enc_out = fnn.Dropout(rate=cfg.dropout, name="enc_drop")(
                enc_out, deterministic=deterministic
            )
        else:
            enc_out = TokenEmbedding(
                hidden_size=cfg.hidden_size, dropout_rate=cfg.dropout, name="enc_embedding"
            )(insample_y, deterministic)
        enc_out = Encoder(
            n_layers=cfg.encoder_layers,
            hidden_size=cfg.hidden_size,
            conv_hidden_size=cfg.conv_hidden_size,
            n_heads=cfg.n_heads,
            moving_avg_window=cfg.moving_avg_window,
            dropout_rate=cfg.dropout,
            activation=cfg.activation,
            seq_len=cfg.input_size,
            modes=cfg.modes,
            mode_select=cfg.mode_select,
            mode_seed=cfg.random_seed,
            name="encoder",
        )(enc_out, deterministic)

        # --- (3) Decoder ---
        # The decoder operates on length (label_len + h). neuralforecast picks
        # its self/cross query modes using (input_size // 2 + h); we match that.
        dec_seq_len = cfg.input_size // 2 + h
        if cfg.futr_exog_size > 0:
            mark_dec = futr_exog[:, -(label_len + h) :, :]
            dec_out = _torch_conv(
                cfg.hidden_size, 3 * c,
                kernel_size=(3,), padding="CIRCULAR", name="dec_embedding",
                token_embed=True,
            )(seasonal_init)
            dec_out = dec_out + fnn.Dense(
                cfg.hidden_size, use_bias=False, name="dec_temporal"
            )(mark_dec.astype(jnp.float32))
            dec_out = fnn.Dropout(rate=cfg.dropout, name="dec_drop")(
                dec_out, deterministic=deterministic
            )
        else:
            dec_out = TokenEmbedding(
                hidden_size=cfg.hidden_size, dropout_rate=cfg.dropout, name="dec_embedding"
            )(seasonal_init, deterministic)
        seasonal_part, trend_part = Decoder(
            n_layers=cfg.decoder_layers,
            hidden_size=cfg.hidden_size,
            conv_hidden_size=cfg.conv_hidden_size,
            n_heads=cfg.n_heads,
            c_out=c_out,
            moving_avg_window=cfg.moving_avg_window,
            dropout_rate=cfg.dropout,
            activation=cfg.activation,
            self_seq_len=dec_seq_len,
            cross_seq_len_q=dec_seq_len,
            cross_seq_len_kv=cfg.input_size,
            modes=cfg.modes,
            mode_select=cfg.mode_select,
            fea_activation=cfg.fea_activation,
            mode_seed=cfg.random_seed + 1000,
            name="decoder",
        )(dec_out, enc_out, trend_init, deterministic)

        # --- (4) Combine seasonal + trend, return the horizon tail ---
        out = trend_part + seasonal_part
        return out[:, -h:, :]
