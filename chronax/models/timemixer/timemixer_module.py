"""Flax NNX modules for the TimeMixer forecaster (port of neuralforecast.TimeMixer).

TimeMixer forecasts by mixing information across time scales: the input window
is repeatedly downsampled, each scale is instance-normalized (RevIN) and
season/trend-decomposed, seasonal parts mix bottom-up (fine -> coarse) and
trend parts top-down (coarse -> fine) through small Linear+GELU chains, and a
per-scale predictor head maps each mixed scale onto the horizon before the
scale outputs are summed and denormalized.

Convolutions are expressed as shifted GEMMs rather than ``lax.conv`` primitives
(the XLA CPU backend pessimizes conv primitives inside the training scan while
matmuls keep their fast dot path): the k=3 circular token embedding is three
rolled matmuls, and the optional stride-2 conv downsampler is three strided
gathers + matmuls. ``input_size`` must be divisible by
``down_sampling_window ** down_sampling_layers`` — the reference's mixed
floor/ceil width expressions coincide exactly under that constraint (and its
torch layers error out when they don't).

Unreachable reference paths are not ported: the future-temporal-mark machinery
(the reference never receives future exogenous features), the never-applied
positional embedding, and the frozen sinusoidal temporal embedding. The
reference's per-block LayerNorm is constructed but never applied — kept here,
unused, so parameter inventories line up. The reference builds its ``conv``
downsampling filter inside the forward pass, leaving it untrained and
resampled per call; here it is a proper trained parameter. ``float32``
throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws both weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``;
    ``fan_in`` is the layer's ``in_features`` (the bias shares the weight's fan_in).
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def __eq__(self, other):
        # Value equality: initializer instances are static graphdef leaves, and an
        # identity-based __eq__ would make same-config graphdefs unequal, so the
        # module-level @nnx.jit inference cache could never hit across refits.
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):
        return hash((type(self), self.fan_in))


class _KaimingNormalConvInit:
    """Picklable initializer matching ``nn.init.kaiming_normal_(mode='fan_in',
    nonlinearity='leaky_relu')`` with a=0 — draws from ``N(0, 2/fan_in)``."""

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * math.sqrt(2.0 / self.fan_in)

    def __eq__(self, other):
        return type(other) is type(self) and other.fan_in == self.fan_in

    def __hash__(self):
        return hash((type(self), self.fan_in))


class _TorchConvInit:
    """Picklable initializer matching torch ``nn.Conv1d`` default:
    ``U(-k, k)`` with ``k = sqrt(1 / (in_channels * kernel_size))``."""

    __slots__ = ("bound",)

    def __init__(self, in_channels: int, kernel_size: int) -> None:
        self.bound = math.sqrt(1.0 / (in_channels * kernel_size))

    def __call__(self, key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-self.bound, maxval=self.bound)

    def __eq__(self, other):
        return type(other) is type(self) and other.bound == self.bound

    def __hash__(self):
        return hash((type(self), self.bound))


def _torch_linear(n_in: int, n_out: int, *, rngs: nnx.Rngs) -> nnx.Linear:
    init = _TorchLinearInit(n_in)
    return nnx.Linear(n_in, n_out, kernel_init=init, bias_init=init, rngs=rngs)


# ---------------------------------------------------------------------------
# Decomposition primitives
# ---------------------------------------------------------------------------

def _moving_avg(x: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
    """Trend extraction: replicate-pad ``(k-1)//2`` on both ends of the time
    axis, then a stride-1 windowed mean via cumulative-sum differences —
    two slices per call instead of a k-term add chain (the chain dominated
    the training step at the default k=25, which runs at ~10 decomposition
    sites per forward). No conv primitive. x: [B, T, N]. Even kernels would
    shorten the output by one in the reference (its pooling errors
    downstream), so they are rejected."""
    if kernel_size % 2 == 0:
        raise ValueError(f"moving_avg kernel must be odd; got {kernel_size}.")
    pad = (kernel_size - 1) // 2
    T = x.shape[1]
    front = jnp.repeat(x[:, :1, :], pad, axis=1)
    end = jnp.repeat(x[:, -1:, :], pad, axis=1)
    xp = jnp.concatenate([front, x, end], axis=1)          # [B, T + 2*pad, N]
    cs = jnp.cumsum(xp, axis=1)
    cs = jnp.concatenate([jnp.zeros_like(cs[:, :1]), cs], axis=1)
    return (cs[:, kernel_size:kernel_size + T] - cs[:, 0:T]) / kernel_size


def _series_decomp(x: jnp.ndarray, kernel_size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Moving-average decomposition: returns ``(season, trend)`` with
    ``season = x - trend``."""
    trend = _moving_avg(x, kernel_size)
    return x - trend, trend


def _dft_decomp(x: jnp.ndarray, top_k: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """DFT decomposition, expression-verbatim from the reference: rfft over the
    LAST axis, magnitudes with the first leading-axis entry zeroed, a single
    GLOBAL threshold at the minimum of the per-position top-k magnitudes, and
    bins at or below the threshold suppressed before the inverse transform.
    Returns ``(season, trend)`` with ``trend = x - season``."""
    xf = jnp.fft.rfft(x)
    freq = jnp.abs(xf)
    freq = freq.at[0].set(0.0)
    topk_vals = jax.lax.top_k(freq, top_k)[0]
    thresh = jnp.min(topk_vals)
    xf = jnp.where(freq <= thresh, 0.0 + 0.0j, xf)
    season = jnp.fft.irfft(xf, n=x.shape[-1]).astype(x.dtype)
    return season, x - season


# ---------------------------------------------------------------------------
# Embedding / normalization
# ---------------------------------------------------------------------------

class TokenEmbedding(nnx.Module):
    """k=3 circular-padded 1-D conv token embedding, computed as three rolled
    matmuls. Weight shape ``[hidden, c_in, 3]`` (torch conv layout), no bias,
    Kaiming-normal fan_in init. x: [B, T, c_in] -> [B, T, hidden]."""

    def __init__(self, c_in: int, hidden_size: int, *, rngs: nnx.Rngs):
        init = _KaimingNormalConvInit(c_in * 3)
        self.weight = nnx.Param(init(rngs.params(), (hidden_size, c_in, 3)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = self.weight.value                                # [d, c, 3]
        prev = jnp.roll(x, 1, axis=1)                        # circular wrap
        nxt = jnp.roll(x, -1, axis=1)
        return prev @ w[:, :, 0].T + x @ w[:, :, 1].T + nxt @ w[:, :, 2].T


class RevIN(nnx.Module):
    """Reversible instance normalization with optional learnable affine.

    Per-window mean over time (``subtract_last`` unsupported here — the
    reference instantiates TimeMixer's copies with the default mean mode),
    population variance, ``sqrt(var + eps)`` scale. The affine inverse divides
    by ``gamma + eps**2`` — eps squared, an inherited reference quirk. With
    ``non_norm=True`` both directions pass through unchanged while the affine
    parameters still exist (reference parity: it constructs them regardless).
    Statistics are returned explicitly so the module stays pure and
    vmap/scan-safe; they carry no parameter dependence, so no stop-gradient is
    needed for gradient equivalence with the reference's detach.
    """

    def __init__(self, num_features: int, *, affine: bool = True,
                 non_norm: bool = False, eps: float = 1e-5, rngs: nnx.Rngs):
        self.num_features = num_features
        self.affine = affine
        self.non_norm = non_norm
        self.eps = eps
        if affine:
            self.gamma = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
            self.beta = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))

    def norm(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """x: [B, T, C] -> (z, loc [B,1,C], scale [B,1,C])."""
        x = x.astype(jnp.float32)
        loc = jnp.mean(x, axis=1, keepdims=True)
        scale = jnp.sqrt(jnp.var(x, axis=1, keepdims=True) + self.eps)
        if self.non_norm:
            return x, loc, scale
        z = (x - loc) / scale
        if self.affine:
            z = z * self.gamma.value[None, None, :] + self.beta.value[None, None, :]
        return z, loc, scale

    def denorm(self, z: jnp.ndarray, loc: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        if self.non_norm:
            return z
        if self.affine:
            z = (z - self.beta.value[None, None, :]) / (self.gamma.value[None, None, :] + self.eps * self.eps)
        return z * scale + loc


# ---------------------------------------------------------------------------
# Multi-scale mixing
# ---------------------------------------------------------------------------

class _MixChain(nnx.Module):
    """One Linear -> GELU (exact) -> Linear unit operating on the TIME axis of
    ``[B, d, T]`` tensors."""

    def __init__(self, t_in: int, t_out: int, *, rngs: nnx.Rngs):
        self.lin1 = _torch_linear(t_in, t_out, rngs=rngs)
        self.lin2 = _torch_linear(t_out, t_out, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.lin2(jax.nn.gelu(self.lin1(x), approximate=False))


class MultiScaleSeasonMixing(nnx.Module):
    """Bottom-up (fine -> coarse) season mixing across scales.

    Inputs/outputs are lists over scales; tensors are ``[B, d, T_i]`` inside
    the chain and ``[B, T_i, d]`` in the returned list (reference layout)."""

    def __init__(self, widths: list[int], *, rngs: nnx.Rngs):
        self.chains = [
            _MixChain(widths[i], widths[i + 1], rngs=rngs)
            for i in range(len(widths) - 1)
        ]

    def __call__(self, season_list: list[jnp.ndarray]) -> list[jnp.ndarray]:
        out_high = season_list[0]
        out_low = season_list[1]
        outs = [jnp.transpose(out_high, (0, 2, 1))]
        for i in range(len(season_list) - 1):
            out_low_res = self.chains[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            outs.append(jnp.transpose(out_high, (0, 2, 1)))
        return outs


class MultiScaleTrendMixing(nnx.Module):
    """Top-down (coarse -> fine) trend mixing across scales (reversed lists)."""

    def __init__(self, widths: list[int], *, rngs: nnx.Rngs):
        # Reference builds chains for i in reversed(range(layers)): coarse->fine.
        self.chains = [
            _MixChain(widths[i + 1], widths[i], rngs=rngs)
            for i in reversed(range(len(widths) - 1))
        ]

    def __call__(self, trend_list: list[jnp.ndarray]) -> list[jnp.ndarray]:
        rev = list(reversed(trend_list))
        out_low = rev[0]
        out_high = rev[1]
        outs = [jnp.transpose(out_low, (0, 2, 1))]
        for i in range(len(rev) - 1):
            out_high_res = self.chains[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            outs.append(jnp.transpose(out_low, (0, 2, 1)))
        outs.reverse()
        return outs


class PastDecomposableMixing(nnx.Module):
    """One PDM block: per-scale decomposition, optional channel-crossing MLP,
    season/trend mixing, and (channel-independent mode only) a residual through
    the output-crossing MLP. The reference also constructs a LayerNorm it never
    applies — kept here, unused, for parameter-inventory parity."""

    def __init__(self, *, widths: list[int], d_model: int, d_ff: int, dropout: float,
                 channel_independence: int, decomp_method: str, moving_avg: int,
                 top_k: int, rngs: nnx.Rngs):
        self.channel_independence = channel_independence
        self.decomp_method = decomp_method
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.layer_norm = nnx.LayerNorm(d_model, rngs=rngs)   # constructed, never applied
        if channel_independence == 0:
            self.cross_w1 = _torch_linear(d_model, d_ff, rngs=rngs)
            self.cross_w2 = _torch_linear(d_ff, d_model, rngs=rngs)
        self.season_mixing = MultiScaleSeasonMixing(widths, rngs=rngs)
        self.trend_mixing = MultiScaleTrendMixing(widths, rngs=rngs)
        self.out_cross_w1 = _torch_linear(d_model, d_ff, rngs=rngs)
        self.out_cross_w2 = _torch_linear(d_ff, d_model, rngs=rngs)

    def _decomp(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.decomp_method == "moving_avg":
            return _series_decomp(x, self.moving_avg)
        return _dft_decomp(x, self.top_k)

    def __call__(self, x_list: list[jnp.ndarray]) -> list[jnp.ndarray]:
        lengths = [x.shape[1] for x in x_list]
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self._decomp(x)
            if self.channel_independence == 0:
                season = self.cross_w2(jax.nn.gelu(self.cross_w1(season), approximate=False))
                trend = self.cross_w2(jax.nn.gelu(self.cross_w1(trend), approximate=False))
            season_list.append(jnp.transpose(season, (0, 2, 1)))
            trend_list.append(jnp.transpose(trend, (0, 2, 1)))

        out_season = self.season_mixing(season_list)
        out_trend = self.trend_mixing(trend_list)

        outs = []
        for ori, season, trend, length in zip(x_list, out_season, out_trend, lengths):
            out = season + trend
            if self.channel_independence:
                out = ori + self.out_cross_w2(jax.nn.gelu(self.out_cross_w1(out), approximate=False))
            outs.append(out[:, :length, :])
        return outs


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class TimeMixerNet(nnx.Module):
    """Full TimeMixer: multi-scale downsampling -> per-scale RevIN -> (channel-
    dependent mode) season/trend pre-decomposition -> token embedding ->
    ``e_layers`` PDM blocks -> per-scale horizon predictors -> scale sum ->
    denormalization with the finest scale's statistics.

    ``__call__(insample_y [B, L, N], deterministic) -> [B, h, N * mult]``
    (the ``mult > 1`` head is the reference's ``distr_output`` Linear over the
    series axis).
    """

    def __init__(self, *, h: int, input_size: int, n_series: int, d_model: int = 32,
                 d_ff: int = 32, dropout: float = 0.1, e_layers: int = 4,
                 top_k: int = 5, decomp_method: str = "moving_avg",
                 moving_avg: int = 25, channel_independence: int = 0,
                 down_sampling_layers: int = 1, down_sampling_window: int = 2,
                 down_sampling_method: str = "avg", use_norm: bool = True,
                 outputsize_multiplier: int = 1, rngs: nnx.Rngs):
        if decomp_method not in ("moving_avg", "dft_decomp"):
            raise ValueError(f"decomp_method must be 'moving_avg' or 'dft_decomp'; got {decomp_method!r}.")
        if down_sampling_method not in ("avg", "max", "conv"):
            raise ValueError(f"down_sampling_method must be 'avg', 'max' or 'conv'; got {down_sampling_method!r}.")
        if moving_avg % 2 == 0:
            raise ValueError(f"moving_avg kernel must be odd; got {moving_avg}.")
        if down_sampling_layers < 1:
            raise ValueError(
                f"down_sampling_layers must be >= 1; got {down_sampling_layers}. "
                "The season/trend mixing chains need at least two scales (the "
                "reference errors out at zero layers too)."
            )
        divisor = down_sampling_window ** down_sampling_layers
        if input_size % divisor != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by down_sampling_window ** "
                f"down_sampling_layers ({divisor}); the reference's layer widths error out otherwise."
            )
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.channel_independence = channel_independence
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.decomp_method = decomp_method
        self.moving_avg = moving_avg
        self.outputsize_multiplier = outputsize_multiplier

        widths = [input_size // down_sampling_window ** i
                  for i in range(down_sampling_layers + 1)]
        self._widths = widths

        c_in = 1 if channel_independence == 1 else n_series
        self.enc_embedding = TokenEmbedding(c_in, d_model, rngs=rngs)
        self.emb_dropout = nnx.Dropout(dropout, rngs=rngs)

        self.pdm_blocks = [
            PastDecomposableMixing(
                widths=widths, d_model=d_model, d_ff=d_ff, dropout=dropout,
                channel_independence=channel_independence, decomp_method=decomp_method,
                moving_avg=moving_avg, top_k=top_k, rngs=rngs,
            )
            for _ in range(e_layers)
        ]

        self.normalize_layers = [
            RevIN(n_series, affine=True, non_norm=not use_norm, rngs=rngs)
            for _ in range(down_sampling_layers + 1)
        ]
        self.predict_layers = [
            _torch_linear(widths[i], h, rngs=rngs)
            for i in range(down_sampling_layers + 1)
        ]

        if channel_independence == 1:
            self.projection_layer = _torch_linear(d_model, 1, rngs=rngs)
        else:
            self.projection_layer = _torch_linear(d_model, n_series, rngs=rngs)
            self.out_res_layers = [
                _torch_linear(widths[i], widths[i], rngs=rngs)
                for i in range(down_sampling_layers + 1)
            ]
            self.regression_layers = [
                _torch_linear(widths[i], h, rngs=rngs)
                for i in range(down_sampling_layers + 1)
            ]

        if down_sampling_method == "conv":
            init = _TorchConvInit(n_series, 3)
            self.down_conv_weight = nnx.Param(
                init(rngs.params(), (n_series, n_series, 3)))

        if outputsize_multiplier > 1:
            self.distr_output = _torch_linear(n_series, n_series * outputsize_multiplier, rngs=rngs)

    # ---- multi-scale input processing ---------------------------------------
    def _downsample_step(self, x: jnp.ndarray) -> jnp.ndarray:
        """One downsampling step [B, T, N] -> [B, T // w, N]."""
        B, T, N = x.shape
        w = self.down_sampling_window
        if self.down_sampling_method == "avg":
            return x[:, : T // w * w].reshape(B, T // w, w, N).mean(axis=2)
        if self.down_sampling_method == "max":
            return x[:, : T // w * w].reshape(B, T // w, w, N).max(axis=2)
        # conv: k=3, stride w, circular pad 1 — three strided gathers + GEMMs.
        wgt = self.down_conv_weight.value                     # [N, N, 3]
        xc = jnp.concatenate([x[:, -1:], x, x[:, :1]], axis=1)  # [B, T+2, N]
        t_out = T // w
        idx = w * jnp.arange(t_out)
        out = None
        for k in range(3):
            term = xc[:, idx + k, :] @ wgt[:, :, k].T
            out = term if out is None else out + term
        return out

    def _multi_scale_inputs(self, x: jnp.ndarray) -> list[jnp.ndarray]:
        xs = [x]
        cur = x
        for _ in range(self.down_sampling_layers):
            cur = self._downsample_step(cur)
            xs.append(cur)
        return xs

    # ---- forward -------------------------------------------------------------
    def __call__(self, insample_y: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = insample_y.astype(jnp.float32)                    # [B, L, N]
        B = x.shape[0]
        N = self.n_series
        ci = self.channel_independence == 1

        x_scales = self._multi_scale_inputs(x)
        x_list, loc0, scale0 = [], None, None
        for i, xi in enumerate(x_scales):
            z, loc, scale = self.normalize_layers[i].norm(xi)
            if i == 0:
                loc0, scale0 = loc, scale
            if ci:
                Ti = z.shape[1]
                z = jnp.transpose(z, (0, 2, 1)).reshape(B * N, Ti, 1)
            x_list.append(z)

        # pre_enc: channel-dependent mode splits season/trend before embedding.
        # The pre-decomposition is ALWAYS the moving-average form; decomp_method
        # only selects the decomposition inside the PDM blocks.
        if ci:
            enc_in_list, res_list = x_list, None
        else:
            enc_in_list, res_list = [], []
            for z in x_list:
                season, trend = _series_decomp(z, self.moving_avg)
                enc_in_list.append(season)
                res_list.append(trend)

        enc_out_list = [
            self.emb_dropout(self.enc_embedding(z), deterministic=deterministic)
            for z in enc_in_list
        ]

        for blk in self.pdm_blocks:
            enc_out_list = blk(enc_out_list)

        # Future multipredictor mixing.
        dec_outs = []
        for i, enc_out in enumerate(enc_out_list):
            dec = self.predict_layers[i](jnp.transpose(enc_out, (0, 2, 1)))
            dec = jnp.transpose(dec, (0, 2, 1))               # [B(,N), h, d]
            if ci:
                dec = self.projection_layer(dec)              # [B*N, h, 1]
                dec = jnp.transpose(dec.reshape(B, N, self.h, 1)[..., 0], (0, 2, 1))
            else:
                dec = self.projection_layer(dec)              # [B, h, N]
                res = jnp.transpose(res_list[i], (0, 2, 1))   # [B, N, T_i]
                res = self.out_res_layers[i](res)
                res = jnp.transpose(self.regression_layers[i](res), (0, 2, 1))
                dec = dec + res
            dec_outs.append(dec)

        dec_out = sum(dec_outs)                               # [B, h, N]
        dec_out = self.normalize_layers[0].denorm(dec_out, loc0, scale0)
        if self.outputsize_multiplier > 1:
            dec_out = self.distr_output(dec_out)              # [B, h, N * mult]
        return dec_out
