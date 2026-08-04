"""RMoK network: RevIN + mixture of KAN experts (NF-faithful forward).

Univariate port of neuralforecast.RMoK's forward. RevIN is inline and NF-exact
(stop-grad mean AND std, std=sqrt(var+eps), forward-local stats). Four experts
map the lookback L -> h: Taylor-KAN (poly expansion), Jacobi-KAN (Jacobi
recurrence), Wave-KAN (mexican-hat wavelet + BatchNorm), and a plain Linear; a
softmax gate mixes them. BatchNorm running stats + dropout are driven by a
``deterministic`` flag (DeepNPTS pattern): False in training (batch stats,
running-stat update, dropout on), True at inference. Expert/gate weights are raw
torch-layout params ([out,in], forward x@w.T+b) so an NF state_dict copies over
without transposing or reshaping any weight (the BatchNorm state still needs a
torch->flax key mapping). The
WaveKAN ``weight1`` param exists (faithful pytree) but is unused (frozen). Init
distributions replicate torch (draws differ). Only ``mexican_hat`` is parity-
checked; the other four wavelets run finite but have no NF-parity arbiter.
float32 throughout.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

_WAVELETS = ("mexican_hat", "morlet", "dog", "meyer", "shannon")


def _revin_norm(x: jnp.ndarray, affine_weight: "jnp.ndarray | None",
                affine_bias: "jnp.ndarray | None", *, affine: bool, eps: float = 1e-5
                ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NF RevINMultivariate norm. x [B,L,N] -> (xn, mean, std); stats stop-grad."""
    mean = jax.lax.stop_gradient(jnp.mean(x, axis=1, keepdims=True))
    std = jax.lax.stop_gradient(jnp.sqrt(jnp.var(x, axis=1, keepdims=True) + eps))
    xn = (x - mean) / std
    if affine:
        xn = xn * affine_weight + affine_bias
    return xn, mean, std


def _revin_denorm(y: jnp.ndarray, affine_weight: "jnp.ndarray | None",
                  affine_bias: "jnp.ndarray | None", *, affine: bool,
                  mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    if affine:
        y = (y - affine_bias) / affine_weight
    return y * std + mean


def _taylor(x: jnp.ndarray, coeffs: jnp.ndarray, bias: jnp.ndarray, order: int) -> jnp.ndarray:
    """coeffs [h,L,order], bias [1,h]; x [B,L] -> [B,h]. y = Σ_i (x^i · c[:,:,i]).sum_L + b."""
    xe = x[:, None, :]                                   # [B,1,L]
    y = jnp.zeros((x.shape[0], coeffs.shape[0]), jnp.float32)
    for i in range(order):
        y = y + ((xe ** i) * coeffs[None, :, :, i]).sum(axis=-1)
    return y + bias


def _jacobi(x: jnp.ndarray, coeffs: jnp.ndarray, degree: int, a: float = 1.0, b: float = 1.0) -> jnp.ndarray:
    """coeffs [L,h,degree+1]; x [B,L] -> [B,h] via tanh + Jacobi recurrence + einsum."""
    x = jnp.tanh(x)
    cols = [jnp.ones_like(x)]
    if degree > 0:
        cols.append(((a - b) + (a + b + 2) * x) / 2)
    for i in range(2, degree + 1):
        tk = (2 * i + a + b) * (2 * i + a + b - 1) / (2 * i * (i + a + b))
        tk1 = (2 * i + a + b - 1) * (a * a - b * b) / (2 * i * (i + a + b) * (2 * i + a + b - 2))
        tk2 = (i + a - 1) * (i + b - 1) * (2 * i + a + b) / (i * (i + a + b) * (2 * i + a + b - 2))
        cols.append((tk * x + tk1) * cols[i - 1] - tk2 * cols[i - 2])
    jac = jnp.stack(cols, axis=-1)                       # [B,L,degree+1]
    return jnp.einsum("bid,iod->bo", jac, coeffs)


def _wavelet(x: jnp.ndarray, scale: jnp.ndarray, translation: jnp.ndarray,
             weights: jnp.ndarray, kind: str) -> jnp.ndarray:
    """Wavelet transform (pre-BN). x [B,L] -> [B,h]. Only mexican_hat is parity-checked."""
    xs = (x[:, None, :] - translation[None]) / scale[None]        # [B,h,L]
    if kind == "mexican_hat":
        psi = (2.0 / (math.sqrt(3) * math.pi ** 0.25)) * (xs ** 2 - 1.0) * jnp.exp(-0.5 * xs ** 2)
    elif kind == "morlet":
        psi = jnp.exp(-0.5 * xs ** 2) * jnp.cos(5.0 * xs)
    elif kind == "dog":
        psi = -xs * jnp.exp(-0.5 * xs ** 2)
    elif kind == "meyer":
        v = jnp.abs(xs)
        t = 2.0 * v - 1.0                                # NF evaluates nu at (2v-1)
        nu = t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
        aux = jnp.where(v <= 0.5, 1.0, jnp.where(v >= 1.0, 0.0, jnp.cos(math.pi / 2 * nu)))
        psi = jnp.sin(math.pi * v) * aux
    else:  # shannon
        L = xs.shape[-1]
        window = jnp.asarray(0.54 - 0.46 * jnp.cos(2 * math.pi * jnp.arange(L) / (L - 1)), jnp.float32)
        psi = jnp.sinc(xs / math.pi) * window[None, None, :]
    return (psi * weights[None]).sum(axis=2)                       # [B,h]


def _torch_uniform(key: jax.Array, shape: tuple[int, ...], fan_in: int) -> jnp.ndarray:
    bound = 1.0 / math.sqrt(fan_in)
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


class RMoKNet(nnx.Module):
    """RMoK forward. Generic over ``n_series`` channels; the forecaster always
    builds it with ``n_series=1``. I/O [B,L,N]->[B,h,N]."""

    def __init__(self, h: int, input_size: int, n_series: int, taylor_order: int,
                 jacobi_degree: int, wavelet_function: str, dropout: float,
                 revin_affine: bool, *, rngs: nnx.Rngs):
        if wavelet_function not in _WAVELETS:
            raise ValueError(f"wavelet_function must be one of {_WAVELETS}; got {wavelet_function!r}.")
        self.h = h
        self.input_size = input_size
        self.n_series = n_series
        self.taylor_order = taylor_order
        self.jacobi_degree = jacobi_degree
        self.wavelet_function = wavelet_function
        self.revin_affine = revin_affine
        L = input_size
        k = jax.random.split(rngs.params(), 8)
        if revin_affine:
            self.affine_weight = nnx.Param(jnp.ones((1, 1, n_series), jnp.float32))
            self.affine_bias = nnx.Param(jnp.zeros((1, 1, n_series), jnp.float32))
        # Taylor: coeffs randn*0.01, bias zeros
        self.taylor_coeffs = nnx.Param(jax.random.normal(k[0], (h, L, taylor_order), jnp.float32) * 0.01)
        self.taylor_bias = nnx.Param(jnp.zeros((1, h), jnp.float32))
        # Jacobi: normal(0, std=1/(L*(degree+1)))
        jstd = 1.0 / (L * (jacobi_degree + 1))
        self.jacobi_coeffs = nnx.Param(jax.random.normal(k[1], (L, h, jacobi_degree + 1), jnp.float32) * jstd)
        # Wave: torch kaiming_uniform(a=sqrt(5)) reduces to U(±1/sqrt(fan_in)); fan_in=L.
        kb = 1.0 / math.sqrt(L)
        self.wave_scale = nnx.Param(jnp.ones((h, L), jnp.float32))
        self.wave_translation = nnx.Param(jnp.zeros((h, L), jnp.float32))
        self.wave_weights = nnx.Param(jax.random.uniform(k[2], (h, L), jnp.float32, -kb, kb))
        self.wave_weight1 = nnx.Param(jax.random.uniform(k[3], (h, L), jnp.float32, -kb, kb))  # UNUSED
        self.wave_bn = nnx.BatchNorm(h, momentum=0.9, epsilon=1e-5, rngs=rngs)
        # Linear expert + gate: raw torch-layout params [out,in], U(±1/sqrt(L))
        self.linear_weight = nnx.Param(_torch_uniform(k[4], (h, L), L))
        self.linear_bias = nnx.Param(_torch_uniform(k[5], (h,), L))
        self.gate_weight = nnx.Param(_torch_uniform(k[6], (4, L), L))
        self.gate_bias = nnx.Param(_torch_uniform(k[7], (4,), L))
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        B, L, N = x.shape
        aw = self.affine_weight.value if self.revin_affine else None
        ab = self.affine_bias.value if self.revin_affine else None
        xn, mean, std = _revin_norm(x, aw, ab, affine=self.revin_affine)
        xn = self.dropout(xn, deterministic=deterministic)
        xd = jnp.transpose(xn, (0, 2, 1)).reshape(B * N, L)
        score = jax.nn.softmax(xd @ self.gate_weight.value.T + self.gate_bias.value, axis=-1)  # [B*N,4]
        e0 = _taylor(xd, self.taylor_coeffs.value, self.taylor_bias.value, self.taylor_order)
        e1 = _jacobi(xd, self.jacobi_coeffs.value, self.jacobi_degree)
        e2 = self.wave_bn(_wavelet(xd, self.wave_scale.value, self.wave_translation.value,
                                   self.wave_weights.value, self.wavelet_function),
                          use_running_average=deterministic)
        e3 = xd @ self.linear_weight.value.T + self.linear_bias.value
        eo = jnp.stack([e0, e1, e2, e3], axis=-1)                     # [B*N,h,4]
        y = jnp.einsum("BLE,BE->BL", eo, score)                       # [B*N,h]
        y = jnp.transpose(y.reshape(B, N, self.h), (0, 2, 1))         # [B,h,N]
        y = _revin_denorm(y, aw, ab, affine=self.revin_affine, mean=mean, std=std)
        return y.reshape(B, self.h, N)
