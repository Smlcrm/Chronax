"""Flax NNX modules for the StemGNN forecaster (port of neuralforecast's StemGNN).

A latent correlation layer (GRU over the NODE axis + additive attention) learns
a graph over series; a 4-term Chebyshev expansion of its normalized Laplacian
forms the graph Fourier transform ("GFT"); and each ``StockBlockLayer`` runs the
Spe-Seq cell — a 4-point DFT over the CHEBYSHEV-ORDER axis (not time), GLU stacks
on the real/imag parts, an ``irfft`` back — followed by a per-order graph-conv
kernel and sigmoid-gated forecast/backcast heads over two residual stacks.

The 4-point DFT/IDFT are materialized as CONSTANT-MATRIX GEMMs rather than
``jnp.fft`` primitives. For length 4 the forward twiddles are exactly
``{0, ±1}`` and the ``irfft(n=4)`` linear map is exact quarters, so the GEMMs
equal the FFT to float rounding, and keeping the transform as a matmul stays on
the XLA-CPU fast path inside the ``lax.scan`` training body (FFT primitives
there are heavily pessimized). ``irfft(n=4)`` consumes only the first
``n//2+1 = 3`` complex bins and ignores the imaginary parts of bins 0 and 2
(DC/Nyquist); ``_IRFFT4`` reproduces that map exactly, top-frequency bin
discarded, as in torch.

At ``n_series=1`` the network is degenerate: softmax over a single node gives
``A=[[a]]``, so ``diag(degree)-A == 0`` regardless of ``a`` (dropout included),
and the Chebyshev expansion uses ``T0 = zeros`` (not identity), hence
``mul_L == 0``. Every block's forecast head then sees only layer biases and the
output is an input-independent constant per horizon step (in scaled space); the
input reaches only block 0's backcast shortcut, which dies in block 1's
``mul_L @ X``. ``chebyshev_first_term="identity"`` instead uses the paper's
standard basis ``T0 = I`` (at N=1 ``mul_L = [I, 0, -I, 0]``), restoring a real
data path so the univariate model genuinely forecasts. No epsilon is added
inside the Laplacian ``sqrt``: if a training step's attention dropout zeroes the
whole batch (probability ``~2^-wbs`` per step at N=1) then ``degree == 0`` and
the sqrt's infinite gradient produces a non-finite step, which the training-time
finite check surfaces.

All shapes/branches are static ctor config, so everything traces under
``jax.vmap`` (``BaseForecaster.conformity_scores``). ``float32`` throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# ---------------------------------------------------------------------------
# Exact 4-point DFT / IRFFT as constant matrices (module-level numpy globals —
# NEVER module attributes: plain arrays on an nnx.Module land in the static
# GraphDef (unhashable), and nnx.Param would train them).
# ---------------------------------------------------------------------------
_DFT4 = np.fft.fft(np.eye(4), axis=0)                    # M[k, t] = e^{-2*pi*i*k*t/4}
_DFT4_RE = np.ascontiguousarray(_DFT4.real, dtype=np.float32)   # entries exactly {0, ±1}
_DFT4_IM = np.ascontiguousarray(_DFT4.imag, dtype=np.float32)   # entries exactly {0, ±1}


def _build_irfft4() -> np.ndarray:
    """Build the constant ``irfft(n=4)`` matrix by evaluating ``np.fft.irfft(., n=4)``
    on the 6 real/imag basis vectors of the 3-bin half-spectrum. Columns are
    ordered [Re0, Re1, Re2, Im0, Im1, Im2]; the Im0/Im2 columns come out exactly
    zero (the imaginary parts of the DC and Nyquist bins are ignored) and every
    entry is an exact quarter."""
    cols = []
    for j in range(3):
        e = np.zeros(3, dtype=np.complex128)
        e[j] = 1.0
        cols.append(np.fft.irfft(e, n=4))
    for j in range(3):
        e = np.zeros(3, dtype=np.complex128)
        e[j] = 1.0j
        cols.append(np.fft.irfft(e, n=4))
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)   # [4, 6]


_IRFFT4 = _build_irfft4()


def _dft4(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full complex 4-point DFT over axis 1 of ``[B, 4, N, W]`` (== ``torch.fft.fft(x, dim=1)``
    on real input). Returns (real, imag), each ``[B, 4, N, W]``."""
    re = jnp.einsum("kj,bjnw->bknw", _DFT4_RE, x)
    im = jnp.einsum("kj,bjnw->bknw", _DFT4_IM, x)
    return re, im


def _irfft4(re: jnp.ndarray, im: jnp.ndarray) -> jnp.ndarray:
    """``torch.fft.irfft(-, n=4, dim=1)`` on a 4-bin complex spectrum ``[B, 4, N, W]``:
    crops to the first 3 Hermitian bins (top-frequency bin discarded) and maps
    back to 4 real samples via an exact constant linear map."""
    coeffs = jnp.concatenate([re[:, :3], im[:, :3]], axis=1)     # [B, 6, N, W]
    return jnp.einsum("tj,bjnw->btnw", _IRFFT4, coeffs)          # [B, 4, N, W]


# ---------------------------------------------------------------------------
# Picklable initializers (torch-parity distributions)
# ---------------------------------------------------------------------------
class _TorchLinearInit:
    """Picklable initializer matching torch ``nn.Linear`` default.

    torch draws weight and bias from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))``,
    where ``fan_in`` is the layer's ``in_features``. torch ``nn.GRU`` uses the
    same law with ``fan_in = hidden_size`` for ALL its weights and biases.
    """

    __slots__ = ("fan_in",)

    def __init__(self, fan_in: int) -> None:
        self.fan_in = fan_in

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = 1.0 / math.sqrt(self.fan_in)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _XavierUniformInit:
    """torch ``nn.init.xavier_uniform_`` with explicit fans (our stored shapes
    differ from torch's, so fans cannot be inferred): ``U(-b, b)`` with
    ``b = gain * sqrt(6 / (fan_in + fan_out))``."""

    __slots__ = ("fan_in", "fan_out", "gain")

    def __init__(self, fan_in: int, fan_out: int, gain: float = 1.0) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.gain = gain

    def __call__(self, key, shape, dtype=jnp.float32):
        bound = self.gain * math.sqrt(6.0 / (self.fan_in + self.fan_out))
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class _XavierNormalInit:
    """torch ``nn.init.xavier_normal_`` with explicit fans:
    ``N(0, std^2)`` with ``std = gain * sqrt(2 / (fan_in + fan_out))``."""

    __slots__ = ("fan_in", "fan_out", "gain")

    def __init__(self, fan_in: int, fan_out: int, gain: float = 1.0) -> None:
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.gain = gain

    def __call__(self, key, shape, dtype=jnp.float32):
        std = self.gain * math.sqrt(2.0 / (self.fan_in + self.fan_out))
        return std * jax.random.normal(key, shape, dtype)


def _dropout(x: jnp.ndarray, rate: float, key, deterministic: bool) -> jnp.ndarray:
    """torch-semantics dropout: zero with prob ``rate``, scale kept by ``1/(1-rate)``.

    ``rate`` and ``deterministic`` are static config, so the Python branch is
    trace-safe; the mask shape follows ``x`` (vmap-batched shapes included).
    """
    if deterministic or rate == 0.0:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class GLU(nnx.Module):
    """Gated linear unit: ``linear_left(x) * sigmoid(linear_right(x))``.

    The two linear maps are fused into one ``Linear(in, 2*out)`` whose output is
    split — identical math (concatenated output columns), same init law (both
    halves drawn from ``U(±1/sqrt(in))``), and half the GEMM dispatches on the
    hot path.
    """

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        init = _TorchLinearInit(in_features)
        self.out_features = out_features
        self.linear = nnx.Linear(
            in_features, 2 * out_features, kernel_init=init, bias_init=init, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = self.linear(x)
        left, right = jnp.split(z, 2, axis=-1)
        return left * jax.nn.sigmoid(right)


class TorchGRU(nnx.Module):
    """Single-layer GRU with torch ``nn.GRU`` weight layout and gate math.

    Parameters mirror torch exactly for pure-copy weight transplant:
    ``w_ih [3H, I]``, ``w_hh [3H, H]``, separate ``b_ih [3H]``/``b_hh [3H]``,
    gates packed ``r|z|n``. The candidate gate keeps ``b_hn`` INSIDE the reset
    product — ``n = tanh(i_n + b_in + r * (h_n + b_hn))`` — which is why flax's
    ``nnx.GRUCell`` (fused single bias) is NOT transplant-exact and is not used.
    All params init ``U(±1/sqrt(H))`` (torch GRU default).
    """

    def __init__(self, input_size: int, hidden_size: int, *, rngs: nnx.Rngs):
        init = _TorchLinearInit(hidden_size)
        self.hidden_size = hidden_size
        self.w_ih = nnx.Param(init(rngs.params(), (3 * hidden_size, input_size)))
        self.w_hh = nnx.Param(init(rngs.params(), (3 * hidden_size, hidden_size)))
        self.b_ih = nnx.Param(init(rngs.params(), (3 * hidden_size,)))
        self.b_hh = nnx.Param(init(rngs.params(), (3 * hidden_size,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: ``[seq, B, I]`` -> full output sequence ``[seq, B, H]``; ``h0 = 0``.

        StemGNN runs the sequence over the NODE axis (seq = n_series), so this
        scan has length N — a single step in the univariate wrapper.
        """
        w_ih = self.w_ih.value
        w_hh = self.w_hh.value
        b_ih = self.b_ih.value
        b_hh = self.b_hh.value

        def cell(h, x_t):
            gi = x_t @ w_ih.T + b_ih                     # [B, 3H]
            gh = h @ w_hh.T + b_hh                       # [B, 3H]
            i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
            h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
            r = jax.nn.sigmoid(i_r + h_r)
            z = jax.nn.sigmoid(i_z + h_z)
            n = jnp.tanh(i_n + r * h_n)
            h_new = (1.0 - z) * n + z * h
            return h_new, h_new

        h0 = jnp.zeros((x.shape[1], self.hidden_size), x.dtype)
        _, outs = jax.lax.scan(cell, h0, x)
        return outs


class StockBlockLayer(nnx.Module):
    """One StemGNN block: GFT -> Spe-Seq cell -> per-order graph-conv kernel ->
    sigmoid-gated forecast head (+ backcast head on block 0 only).

    ``stack_cnt`` is the block index (0 or 1). Block 1 has no ``backcast``
    Linear but still carries a ``backcast_short_cut`` that never influences its
    forward; it is built unconditionally to keep the parameter layout aligned
    with the reference for weight transplant.
    """

    def __init__(self, time_step: int, unit: int, multi_layer: int, stack_cnt: int,
                 *, rngs: nnx.Rngs):
        L = time_step
        S = multi_layer * L                          # per-order feature width
        O = 4 * multi_layer * L                      # GLU width
        self.time_step = L
        self.multi = multi_layer
        self.stack_cnt = stack_cnt

        def lin(n_in: int, n_out: int) -> nnx.Linear:
            init = _TorchLinearInit(n_in)
            return nnx.Linear(n_in, n_out, kernel_init=init, bias_init=init, rngs=rngs)

        # Weight is stored squeezed as [4, S, S]; the xavier fans follow the
        # reference's 5-D [1, 4, 1, S, S] layout: fan_in = 4*S*S, fan_out = S*S.
        w_init = _XavierNormalInit(4 * S * S, S * S)
        self.weight = nnx.Param(w_init(rngs.params(), (4, S, S)))
        self.forecast = lin(S, S)
        self.forecast_result = lin(S, L)
        if stack_cnt == 0:
            self.backcast = lin(S, L)
        self.backcast_short_cut = lin(L, L)          # unused on block 1
        # 6 GLUs ordered [r0, i0, r1, i1, r2, i2]; stage 0 maps 4L -> O, rest O -> O.
        self.glus = [GLU(4 * L, O, rngs=rngs), GLU(4 * L, O, rngs=rngs)] + [
            GLU(O, O, rngs=rngs) for _ in range(4)
        ]

    def spe_seq_cell(self, x: jnp.ndarray) -> jnp.ndarray:
        """``[B, 4, N, L]`` -> ``[B, 4, N, S]``: DFT over the ORDER axis, GLU
        stacks on real/imag, irfft back (top bin cropped)."""
        B, _, N, L = x.shape
        re, im = _dft4(x)
        re = re.transpose(0, 2, 1, 3).reshape(B, N, 4 * L)
        im = im.transpose(0, 2, 1, 3).reshape(B, N, 4 * L)
        for i in range(3):
            re = self.glus[2 * i](re)
            im = self.glus[2 * i + 1](im)
        re = re.reshape(B, N, 4, -1).transpose(0, 2, 1, 3)       # [B, 4, N, S]
        im = im.reshape(B, N, 4, -1).transpose(0, 2, 1, 3)
        return _irfft4(re, im)

    def __call__(self, x: jnp.ndarray, mul_L: jnp.ndarray):
        """x: ``[B, N, L]``; mul_L: ``[4, N, N]`` -> (forecast ``[B, N, L]``,
        backcast ``[B, N, L]`` | None). The single input channel is dropped
        (in_channel == 1 always), so the einsums are the squeezed equivalents of
        the reference's broadcast matmuls."""
        gfted = jnp.einsum("knm,bml->bknl", mul_L, x)            # [B, 4, N, L]
        g = self.spe_seq_cell(gfted)                             # [B, 4, N, S]
        igfted = jnp.einsum("bkns,kst->bnt", g, self.weight.value)   # sum over orders
        forecast = self.forecast_result(jax.nn.sigmoid(self.forecast(igfted)))
        if self.stack_cnt == 0:
            backcast = jax.nn.sigmoid(self.backcast(igfted) - self.backcast_short_cut(x))
        else:
            backcast = None
        return forecast, backcast


class StemGNNNet(nnx.Module):
    """Full StemGNN network forward.

    ``insample_z [B, L, N]`` (scaled) -> ``[B, h, outputsize_multiplier * N]``.
    At N>1 the final reshape interleaves quantile heads and series; the
    univariate wrapper always runs N=1, where it collapses to ``[B, h, mult]``.
    """

    def __init__(self, *, h: int, input_size: int, n_series: int = 1,
                 n_stacks: int = 2, multi_layer: int = 5, dropout_rate: float = 0.5,
                 leaky_rate: float = 0.2, outputsize_multiplier: int = 1,
                 chebyshev_first_term: str = "nf_zero", rngs: nnx.Rngs):
        if n_stacks != 2:
            raise ValueError("StemGNN currently only supports n_stacks=2.")
        if chebyshev_first_term not in ("nf_zero", "identity"):
            raise ValueError(
                "chebyshev_first_term must be 'nf_zero' (NF parity) or 'identity' "
                f"(paper T0=I); got {chebyshev_first_term!r}."
            )
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.multi_layer = multi_layer
        self.dropout_rate = float(dropout_rate)
        self.leaky_rate = float(leaky_rate)
        self.outputsize_multiplier = outputsize_multiplier
        self.chebyshev_first_term = chebyshev_first_term

        att_init = _XavierUniformInit(fan_in=1, fan_out=n_series, gain=1.414)
        self.weight_key = nnx.Param(att_init(rngs.params(), (n_series, 1)))
        self.weight_query = nnx.Param(att_init(rngs.params(), (n_series, 1)))
        self.gru = TorchGRU(input_size, n_series, rngs=rngs)
        self.stock_blocks = [
            StockBlockLayer(input_size, n_series, multi_layer, i, rngs=rngs)
            for i in range(n_stacks)
        ]

        def lin(n_in: int, n_out: int) -> nnx.Linear:
            init = _TorchLinearInit(n_in)
            return nnx.Linear(n_in, n_out, kernel_init=init, bias_init=init, rngs=rngs)

        self.fc1 = lin(input_size, input_size)
        self.fc2 = lin(input_size, h * outputsize_multiplier)

    def _latent_correlation(self, x: jnp.ndarray, dropout_key, deterministic: bool) -> jnp.ndarray:
        """``[B, L, N]`` -> ``mul_L [4, N, N]``. The GRU rolls over NODES with
        hidden size = n_series; the additive attention operates on the
        (hidden, node) axes of its full output sequence; one graph is formed by
        averaging attention over the batch. ``degree`` uses PRE-symmetrization
        row sums."""
        gru_out = self.gru(jnp.transpose(x, (2, 0, 1)))          # [N, B, N]
        inp = jnp.transpose(gru_out, (1, 0, 2))                  # [B, N(seq), N(hid)]
        inp = jnp.transpose(inp, (0, 2, 1))                      # [B, N(hid), N(seq)]
        key = inp @ self.weight_key.value                        # [B, N, 1]
        query = inp @ self.weight_query.value                    # [B, N, 1]
        data = key[:, :, 0][:, :, None] + query[:, :, 0][:, None, :]   # key_i + query_j
        data = jax.nn.leaky_relu(data, negative_slope=self.leaky_rate)
        att = jax.nn.softmax(data, axis=2)
        att = _dropout(att, self.dropout_rate, dropout_key, deterministic)
        A = jnp.mean(att, axis=0)                                # [N, N] batch-mean graph
        degree = jnp.sum(A, axis=1)
        A = 0.5 * (A + A.T)
        dinv = 1.0 / (jnp.sqrt(degree) + 1e-7)
        lap = dinv[:, None] * (jnp.diag(degree) - A) * dinv[None, :]
        n = A.shape[0]
        if self.chebyshev_first_term == "nf_zero":
            t0 = jnp.zeros((n, n), lap.dtype)                    # default: T0 = 0 (not the standard-basis I)
        else:
            t0 = jnp.eye(n, dtype=lap.dtype)                     # paper mode: standard Chebyshev basis T0 = I
        t1 = lap
        t2 = 2.0 * (lap @ t1) - t0
        t3 = 2.0 * (lap @ t2) - t1
        return jnp.stack([t0, t1, t2, t3])                       # [4, N, N]

    def __call__(self, insample_z: jnp.ndarray, dropout_key=None,
                 deterministic: bool = True) -> jnp.ndarray:
        """insample_z: ``[B, L, N]`` scaled -> ``[B, h, mult * N]`` scaled.

        ``deterministic=True`` (inference/eval) needs no key — dropout is off.
        Training passes a per-step dropout key.
        """
        x = insample_z.astype(jnp.float32)
        if not deterministic and dropout_key is None:
            raise ValueError("Training forward (deterministic=False) requires dropout_key.")
        mul_L = self._latent_correlation(x, dropout_key, deterministic)
        X = jnp.transpose(x, (0, 2, 1))                          # [B, N, L]
        f0, X = self.stock_blocks[0](X, mul_L)
        f1, _ = self.stock_blocks[1](X, mul_L)
        out = self.fc2(jax.nn.leaky_relu(self.fc1(f0 + f1), negative_slope=0.01))
        out = jnp.transpose(out, (0, 2, 1))                      # [B, h*mult, N]
        return out.reshape(x.shape[0], self.h, self.outputsize_multiplier * self.n_series)
