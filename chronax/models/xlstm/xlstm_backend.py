"""xLSTM / xLSTMTime numerics — pure-functional JAX.

Implements the matrix-LSTM cell (mLSTM) and scalar-LSTM cell (sLSTM)
from Beck et al. 2024 with the log-space stabilizer trick for numerical
safety on exponential gates. Adds the xLSTMTime extensions of
Alharthi & Mahmood 2024: reversible instance normalization (RevIN),
learnable moving-average series decomposition, and a direct linear
forecast head as an alternative to autoregressive decoding.

Master params are held in float32; forward/backward compute runs in
bfloat16. The stabilizer state m_t and the rescaled gate scalars
i_stab, f_stab are kept in float32 to bound numerical drift on long
sequences.

All public functions are pure (no class state). They are intended to be
wrapped by lru_cache'd jit builders in :mod:`xlstm_functions`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax, random


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class XLSTMConfig:
    """Static configuration for the xLSTM / xLSTMTime stack.

    Frozen so instances are hashable and usable as ``lru_cache`` keys and
    as ``static_argnames`` to ``jax.jit``. Values flow into shape decisions
    and never get traced.

    New (xLSTMTime) fields default to the original mLSTM/AR recipe so the
    existing call sites and tests continue to produce bit-identical
    behavior.
    """
    embed_dim: int = 64       # D
    num_heads: int = 4        # H
    head_dim: int = 16        # Dh ; expected D == H * Dh
    num_layers: int = 2       # L
    ctx_len: int = 64         # C
    horizon_train: int = 8    # HT (AR mode only)
    gate_clip: float = 8.0
    eps: float = 1e-6

    # xLSTMTime extensions (defaults preserve current behavior)
    block_types: Tuple[str, ...] = ("mlstm", "mlstm")  # length must equal num_layers
    use_revin: bool = False
    revin_affine: bool = True
    use_decomposition: bool = False
    decomp_kernel: int = 25
    decode_mode: str = "ar"   # "ar" | "direct"
    horizon: int = 8          # static; required when decode_mode == "direct"
    use_conv1d_in_slstm: bool = False
    conv1d_kernel: int = 4
    # sLSTM cell semantics. Defaults reproduce the original behavior; the
    # paper-faithful FlashRNN/xLSTM sLSTM is opt-in (Eq. 13-15, arXiv 2412.07752).
    slstm_forget_gate: str = "exp"      # "exp" (current) | "sigmoid" (log-sigmoid forget)
    slstm_stabilizer: str = "per_head"  # "per_head" (current) | "per_cell" (per Eq.15)

    def __post_init__(self):
        if len(self.block_types) != self.num_layers:
            raise ValueError(
                f"block_types length ({len(self.block_types)}) must equal num_layers ({self.num_layers})"
            )
        for bt in self.block_types:
            if bt not in ("mlstm", "slstm"):
                raise ValueError(f"block_types entries must be 'mlstm' or 'slstm', got {bt!r}")
        if self.decode_mode not in ("ar", "direct"):
            raise ValueError(f"decode_mode must be 'ar' or 'direct', got {self.decode_mode!r}")
        if self.decode_mode == "direct" and self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 in direct mode, got {self.horizon}")
        if self.slstm_forget_gate not in ("exp", "sigmoid"):
            raise ValueError(
                f"slstm_forget_gate must be 'exp' or 'sigmoid', got {self.slstm_forget_gate!r}"
            )
        if self.slstm_stabilizer not in ("per_head", "per_cell"):
            raise ValueError(
                f"slstm_stabilizer must be 'per_head' or 'per_cell', got {self.slstm_stabilizer!r}"
            )


# =============================================================================
# Block states
# =============================================================================

class BlockState(NamedTuple):
    """Per-block mLSTM state.

    - ``C``: covariance matrix per head, shape (H, Dh, Dh). bf16 in compute.
    - ``n``: normalizer vector per head, shape (H, Dh). bf16.
    - ``m``: log-space stabilizer per head, shape (H,). float32.
    """
    C: jnp.ndarray
    n: jnp.ndarray
    m: jnp.ndarray


class SLSTMBlockState(NamedTuple):
    """Per-block sLSTM state.

    - ``h``: hidden state per head (needed for memory mixing via R), shape (H, Dh). bf16.
    - ``c``: scalar cell per unit, shape (H, Dh). bf16.
    - ``n``: normalizer, shape (H, Dh). bf16.
    - ``m``: log-space stabilizer, float32. Shape (H,) for ``slstm_stabilizer="per_head"``
      (default) or (H, Dh) for ``"per_cell"`` (paper Eq.15).
    """
    h: jnp.ndarray
    c: jnp.ndarray
    n: jnp.ndarray
    m: jnp.ndarray


def init_block_state(cfg: XLSTMConfig, dtype=jnp.bfloat16) -> BlockState:
    H, Dh = cfg.num_heads, cfg.head_dim
    return BlockState(
        C=jnp.zeros((H, Dh, Dh), dtype=dtype),
        n=jnp.zeros((H, Dh), dtype=dtype),
        m=jnp.full((H,), -1e9, dtype=jnp.float32),
    )


def init_slstm_block_state(cfg: XLSTMConfig, dtype=jnp.bfloat16) -> SLSTMBlockState:
    H, Dh = cfg.num_heads, cfg.head_dim
    # Stabilizer is per-cell (H, Dh) for the paper-faithful path, else per-head (H,).
    m_shape = (H, Dh) if cfg.slstm_stabilizer == "per_cell" else (H,)
    return SLSTMBlockState(
        h=jnp.zeros((H, Dh), dtype=dtype),
        c=jnp.zeros((H, Dh), dtype=dtype),
        n=jnp.zeros((H, Dh), dtype=dtype),
        m=jnp.full(m_shape, -1e9, dtype=jnp.float32),
    )


# =============================================================================
# Init helpers
# =============================================================================

def _glorot(key, shape, dtype=jnp.float32):
    if len(shape) == 1:
        scale = jnp.sqrt(1.0 / shape[0])
        return random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)
    fan_in, fan_out = shape[-2], shape[-1]
    scale = jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)


def _glorot_per_head(key, H, fan_in, fan_out):
    """Glorot uniform but draw independently per head — returns shape (H, fan_in, fan_out)."""
    bound = jnp.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, (H, fan_in, fan_out), dtype=jnp.float32, minval=-bound, maxval=bound)


def init_mlstm_block(key, cfg: XLSTMConfig) -> dict:
    """Per-block params for an mLSTM layer."""
    D, H, Dh = cfg.embed_dim, cfg.num_heads, cfg.head_dim
    keys = random.split(key, 8)
    return {
        "pre_ln": {"scale": jnp.ones((D,), jnp.float32), "bias": jnp.zeros((D,), jnp.float32)},
        "up_proj": {"W": _glorot(keys[0], (D, 2 * D)), "b": jnp.zeros((2 * D,), jnp.float32)},
        "qkv": {
            "Wq": _glorot(keys[1], (H, D, Dh)), "bq": jnp.zeros((H, Dh), jnp.float32),
            "Wk": _glorot(keys[2], (H, D, Dh)), "bk": jnp.zeros((H, Dh), jnp.float32),
            "Wv": _glorot(keys[3], (H, D, Dh)), "bv": jnp.zeros((H, Dh), jnp.float32),
        },
        "gates": {
            "Wi": _glorot(keys[4], (H, D, 1)), "bi": jnp.full((H, 1), -1.0, jnp.float32),
            "Wf": _glorot(keys[5], (H, D, 1)), "bf": jnp.full((H, 1), 1.0, jnp.float32),
            "Wo": _glorot(keys[6], (H, D, Dh)), "bo": jnp.zeros((H, Dh), jnp.float32),
        },
        "group_ln": {"scale": jnp.ones((H, Dh), jnp.float32), "bias": jnp.zeros((H, Dh), jnp.float32)},
        "down_proj": {"W": _glorot(keys[7], (D, D)), "b": jnp.zeros((D,), jnp.float32)},
    }


def init_slstm_block(key, cfg: XLSTMConfig) -> dict:
    """Per-block params for an sLSTM layer (with R matrices for memory mixing)."""
    D, H, Dh = cfg.embed_dim, cfg.num_heads, cfg.head_dim
    # Keys consumed: up_proj(1) + (W,R)*4 gates(8) + down_proj(1) [+ conv1d(1)] = 10 or 11
    n_keys = 10 + (1 if cfg.use_conv1d_in_slstm else 0)
    keys = random.split(key, n_keys)
    kp = iter(keys)
    block = {
        "pre_ln": {"scale": jnp.ones((D,), jnp.float32), "bias": jnp.zeros((D,), jnp.float32)},
        "up_proj": {"W": _glorot(next(kp), (D, 2 * D)), "b": jnp.zeros((2 * D,), jnp.float32)},
        "slstm_gates": {
            "Wi": _glorot(next(kp), (H, D, Dh)), "Ri": _glorot_per_head(next(kp), H, Dh, Dh),
            "bi": jnp.full((H, Dh), -1.0, jnp.float32),
            "Wf": _glorot(next(kp), (H, D, Dh)), "Rf": _glorot_per_head(next(kp), H, Dh, Dh),
            "bf": jnp.full((H, Dh), 1.0, jnp.float32),
            "Wo": _glorot(next(kp), (H, D, Dh)), "Ro": _glorot_per_head(next(kp), H, Dh, Dh),
            "bo": jnp.zeros((H, Dh), jnp.float32),
            "Wz": _glorot(next(kp), (H, D, Dh)), "Rz": _glorot_per_head(next(kp), H, Dh, Dh),
            "bz": jnp.zeros((H, Dh), jnp.float32),
        },
        "group_ln": {"scale": jnp.ones((H, Dh), jnp.float32), "bias": jnp.zeros((H, Dh), jnp.float32)},
        "down_proj": {"W": _glorot(next(kp), (D, D)), "b": jnp.zeros((D,), jnp.float32)},
    }
    if cfg.use_conv1d_in_slstm:
        # Depthwise causal conv: (D groups, kernel, 1 in/out per group)
        block["conv1d"] = {
            "W": _glorot(next(kp), (D, cfg.conv1d_kernel)),
            "b": jnp.zeros((D,), jnp.float32),
        }
    return block


def init_params(key: jax.Array, cfg: XLSTMConfig) -> dict:
    """Initialize the full parameter pytree in float32.

    Conditionally adds RevIN, decomposition kernel, and the direct head
    based on the XLSTMConfig flags. AR-mode out_proj (D,1) is always
    present so AR decode keeps working.
    """
    D, H, Dh, L = cfg.embed_dim, cfg.num_heads, cfg.head_dim, cfg.num_layers
    if D != H * Dh:
        raise ValueError(f"embed_dim ({D}) must equal num_heads ({H}) * head_dim ({Dh})")

    # Top-level keys: input_embed, blocks*L, head_ln (no key), out_proj, out_proj_direct,
    # revin, decomp. We allocate generously and use iter.
    total_keys = 2 + L + 3
    keys = random.split(key, total_keys)
    k_iter = iter(keys)

    params = {
        "input_embed": {"W": _glorot(next(k_iter), (1, D)), "b": jnp.zeros((D,), jnp.float32)},
        "blocks": [],
    }
    for i in range(L):
        bk = next(k_iter)
        bt = cfg.block_types[i]
        if bt == "mlstm":
            params["blocks"].append(init_mlstm_block(bk, cfg))
        else:
            params["blocks"].append(init_slstm_block(bk, cfg))

    params["head_ln"] = {"scale": jnp.ones((D,), jnp.float32), "bias": jnp.zeros((D,), jnp.float32)}
    params["out_proj"] = {"W": _glorot(next(k_iter), (D, 1)), "b": jnp.zeros((1,), jnp.float32)}

    if cfg.decode_mode == "direct":
        params["out_proj_direct"] = {
            "W": _glorot(next(k_iter), (D, cfg.horizon)),
            "b": jnp.zeros((cfg.horizon,), jnp.float32),
        }

    if cfg.use_revin and cfg.revin_affine:
        params["revin"] = {
            "gamma": jnp.ones((1,), jnp.float32),
            "beta": jnp.zeros((1,), jnp.float32),
        }

    if cfg.use_decomposition:
        params["decomp"] = {
            "k": jnp.full((cfg.decomp_kernel,), 1.0 / cfg.decomp_kernel, jnp.float32),
        }

    return params


# =============================================================================
# Norm utilities
# =============================================================================

def _layer_norm(x, scale, bias, eps):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) * lax.rsqrt(var + eps)
    return x_norm * scale + bias


def _per_head_norm(x, scale, bias, eps):
    # x: (..., H, Dh) — normalize over last axis (per-head)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) * lax.rsqrt(var + eps)
    return x_norm * scale + bias


# =============================================================================
# RevIN (Reversible Instance Normalization)
# =============================================================================

def revin_normalize(x: jnp.ndarray, p: dict, eps: float = 1e-5):
    """Per-instance normalization. x: (T,) or (T, D_in).

    Returns (x_norm, stats) where stats = {"mu": (1,) or (1, D_in), "sigma": same}.
    If ``p`` is a non-empty dict, applies learnable affine ``gamma * x_norm + beta``.
    """
    mu = jnp.mean(x, axis=0, keepdims=True)
    sigma = jnp.sqrt(jnp.var(x, axis=0, keepdims=True) + eps)
    x_n = (x - mu) / sigma
    if p:
        x_n = x_n * p["gamma"] + p["beta"]
    return x_n, {"mu": mu, "sigma": sigma}


def revin_denormalize(y: jnp.ndarray, p: dict, stats: dict):
    """Inverse of :func:`revin_normalize`. y: (H,) or (H, D_in)."""
    if p:
        y = (y - p["beta"]) / p["gamma"]
    return y * stats["sigma"] + stats["mu"]


# =============================================================================
# Series decomposition (learnable moving-average)
# =============================================================================

def series_decompose(x: jnp.ndarray, p: dict, kernel: int):
    """Split x into trend and seasonal components via a learnable moving-average kernel.

    x: (T,) float. Returns (trend: (T,), seasonal: (T,)).
    Edge-replication padding to preserve length.
    """
    pad_l = (kernel - 1) // 2
    pad_r = kernel - 1 - pad_l
    xp = jnp.concatenate([jnp.repeat(x[:1], pad_l), x, jnp.repeat(x[-1:], pad_r)])
    trend = jnp.convolve(xp, p["k"], mode="valid")  # (T,)
    seasonal = x - trend
    return trend, seasonal


# =============================================================================
# mLSTM step + block
# =============================================================================

def _block_output(h_seq, gate_in_seq, residual_seq, block_params, cfg: XLSTMConfig):
    """Shared post-recurrence block output, vectorized over the leading (time) axis.

    h_seq: (..., H, Dh) raw cell outputs; gate_in_seq / residual_seq: (..., D).
    Applies per-head group-norm, flatten, SiLU gating, down-projection, residual.
    Identical math to the per-step block tail — just batched over T.
    """
    D = cfg.embed_dim
    h_norm = _per_head_norm(
        h_seq, block_params["group_ln"]["scale"], block_params["group_ln"]["bias"], cfg.eps
    )
    h_flat = h_norm.reshape(*h_norm.shape[:-2], D)
    gated = h_flat * jax.nn.silu(gate_in_seq)
    down = gated @ block_params["down_proj"]["W"] + block_params["down_proj"]["b"]
    return residual_seq + down


def mlstm_recurrence_step(precomp_t, state: BlockState, cfg: XLSTMConfig):
    """mLSTM memory recurrence given precomputed per-step input projections.

    precomp_t = (q, k, v, i_pre, f_pre, o_pre): q/k/v/o_pre (H, Dh) compute-dtype,
    i_pre/f_pre (H,) float32 (already clipped). Only the C/n/m update and retrieval
    depend on ``state`` — all input-side projections are precomputed upstream
    (FlashRNN: apply W to all timesteps before the recurrent loop).
    """
    q, k, v, i_pre, f_pre, o_pre = precomp_t
    dtype = q.dtype

    m_new = jnp.maximum(f_pre + state.m, i_pre)
    i_stab = jnp.exp(i_pre - m_new)
    f_stab = jnp.exp(f_pre + state.m - m_new)

    i_bf = i_stab.astype(dtype)
    f_bf = f_stab.astype(dtype)

    vk = jnp.einsum("hi,hj->hij", v, k)
    C_new = f_bf[:, None, None] * state.C + i_bf[:, None, None] * vk
    n_new = f_bf[:, None] * state.n + i_bf[:, None] * k

    h_num = jnp.einsum("hij,hj->hi", C_new, q)
    nq = jnp.einsum("hi,hi->h", n_new, q)
    h_denom = jnp.maximum(jnp.abs(nq), jnp.asarray(1.0, dtype=dtype))
    h_out = jax.nn.sigmoid(o_pre) * (h_num / h_denom[:, None])

    return h_out, BlockState(C=C_new, n=n_new, m=m_new)


def _mlstm_input_proj(block_params, mlstm_in, cfg: XLSTMConfig):
    """mLSTM gate/qkv input projections. Rank-polymorphic: ``mlstm_in`` may be a
    single step (D,) or a whole sequence (T, D). Returns (q, k, v, i_pre, f_pre, o_pre)."""
    Dh = cfg.head_dim
    gc = cfg.gate_clip
    qkv = block_params["qkv"]
    gates = block_params["gates"]
    dtype = mlstm_in.dtype
    sub = "hdk,d->hk" if mlstm_in.ndim == 1 else "hdk,td->thk"

    q = jnp.einsum(sub, qkv["Wq"], mlstm_in) + qkv["bq"]
    k = (jnp.einsum(sub, qkv["Wk"], mlstm_in) + qkv["bk"]) / jnp.sqrt(
        jnp.asarray(Dh, dtype=dtype)
    )
    v = jnp.einsum(sub, qkv["Wv"], mlstm_in) + qkv["bv"]

    i_pre = jnp.einsum(sub, gates["Wi"], mlstm_in).squeeze(-1) + gates["bi"].squeeze(-1)
    f_pre = jnp.einsum(sub, gates["Wf"], mlstm_in).squeeze(-1) + gates["bf"].squeeze(-1)
    i_pre = jnp.clip(i_pre.astype(jnp.float32), -gc, gc)
    f_pre = jnp.clip(f_pre.astype(jnp.float32), -gc, gc)
    o_pre = jnp.einsum(sub, gates["Wo"], mlstm_in) + gates["bo"]
    return q, k, v, i_pre, f_pre, o_pre


def mlstm_step(block_params, mlstm_in, state: BlockState, cfg: XLSTMConfig):
    """Single mLSTM timestep.

    mlstm_in : (D,) compute-dtype (bf16). state : BlockState.
    Returns (h_out: (H, Dh), new_state: BlockState).
    """
    precomp = _mlstm_input_proj(block_params, mlstm_in, cfg)
    return mlstm_recurrence_step(precomp, state, cfg)


def mlstm_block_step(block_params, x_t, state: BlockState, cfg: XLSTMConfig):
    """One mLSTM block, one timestep. x_t: (D,) compute dtype."""
    D = cfg.embed_dim
    x_n = _layer_norm(x_t, block_params["pre_ln"]["scale"], block_params["pre_ln"]["bias"], cfg.eps)
    up = x_n @ block_params["up_proj"]["W"] + block_params["up_proj"]["b"]
    mlstm_in, gate_in = jnp.split(up, 2, axis=-1)
    h_out, new_state = mlstm_step(block_params, mlstm_in, state, cfg)
    h_norm = _per_head_norm(h_out, block_params["group_ln"]["scale"], block_params["group_ln"]["bias"], cfg.eps)
    h_flat = h_norm.reshape(D)
    gated = h_flat * jax.nn.silu(gate_in)
    down = gated @ block_params["down_proj"]["W"] + block_params["down_proj"]["b"]
    return x_t + down, new_state


def mlstm_block_forward(block_params, x_seq, init_state: BlockState, cfg: XLSTMConfig):
    """Run an mLSTM block over a sequence x_seq: (T, D).

    Input-side ops (pre-LN, up-projection, qkv + gate projections) are computed for
    the whole sequence up front; only the C/n/m memory recurrence runs in the scan,
    and the block tail is vectorized over T. Numerically equivalent to scanning
    ``mlstm_block_step`` (the AR-decode path), up to bf16 reassociation.

    Perf: hoisting the input GEMMs out of the (irreducibly sequential) time loop is
    faster on CPU/XLA at the context lengths this model uses (ctx_len <= ~128; default
    64 -> ~1.3-1.4x). Beyond ~T=256 it becomes memory-bandwidth bound and regresses
    vs. per-step recompute; on GPU it wins at all sizes (the FlashRNN regime).
    """
    x_n = _layer_norm(
        x_seq, block_params["pre_ln"]["scale"], block_params["pre_ln"]["bias"], cfg.eps
    )
    up = x_n @ block_params["up_proj"]["W"] + block_params["up_proj"]["b"]
    mlstm_in_seq, gate_in_seq = jnp.split(up, 2, axis=-1)
    precomp_seq = _mlstm_input_proj(block_params, mlstm_in_seq, cfg)  # tuple of (T, ...)

    def step(state, precomp_t):
        h_t, new_state = mlstm_recurrence_step(precomp_t, state, cfg)
        return new_state, h_t
    final_state, h_seq = lax.scan(step, init_state, precomp_seq)
    out = _block_output(h_seq, gate_in_seq, x_seq, block_params, cfg)
    return out, final_state


# Backward-compat alias (older tests referenced `block_step` / `block_forward`)
block_step = mlstm_block_step
block_forward = mlstm_block_forward


# =============================================================================
# sLSTM step + block
# =============================================================================

def _slstm_stack_weights(p):
    """Stack the four sLSTM gates (order i, f, o, z) for fused matmuls.

    Returns W (4, H, D, Dh), R (4, H, Dh, Dh), b (4, H, Dh).
    """
    W = jnp.stack([p["Wi"], p["Wf"], p["Wo"], p["Wz"]])
    R = jnp.stack([p["Ri"], p["Rf"], p["Ro"], p["Rz"]])
    b = jnp.stack([p["bi"], p["bf"], p["bo"], p["bz"]])
    return W, R, b


def slstm_recurrence_step(R_stacked, Wx_t, state: SLSTMBlockState, cfg: XLSTMConfig):
    """sLSTM recurrence given the precomputed stacked input projection.

    R_stacked: (4, H, Dh, Dh) recurrent weights (gate order i, f, o, z).
    Wx_t: (4, H, Dh) precomputed (W·x + bias) for the four gates.
    Only the recurrent term R·h and the c/n/m update depend on ``state``; the
    input projection W·x is hoisted out of the time loop (FlashRNN).
    """
    gc = cfg.gate_clip
    dtype = Wx_t.dtype
    Rh = jnp.einsum("ghkj,hj->ghk", R_stacked, state.h)  # (4, H, Dh)
    pre = Wx_t + Rh

    i_pre = jnp.clip(pre[0].astype(jnp.float32), -gc, gc)
    f_pre = jnp.clip(pre[1].astype(jnp.float32), -gc, gc)
    o_pre = pre[2].astype(dtype)
    z_t = jnp.tanh(pre[3]).astype(dtype)

    # Forget gate: the paper (Eq.13-15) uses log-sigmoid; "exp" keeps the raw
    # preactivation (original Chronax behavior). cfg is static -> jit/vmap-safe.
    logf = jax.nn.log_sigmoid(f_pre) if cfg.slstm_forget_gate == "sigmoid" else f_pre

    # Stabilizer: "per_cell" keeps m per (head, cell) as in the paper (Eq.15);
    # "per_head" broadcasts one per-head scalar and recollapses (original).
    m_prev = state.m if cfg.slstm_stabilizer == "per_cell" else state.m[:, None]
    m_new = jnp.maximum(logf + m_prev, i_pre)   # (H, Dh)
    i_stab = jnp.exp(i_pre - m_new)
    f_stab = jnp.exp(logf + m_prev - m_new)

    i_bf = i_stab.astype(dtype)
    f_bf = f_stab.astype(dtype)

    c_new = f_bf * state.c.astype(dtype) + i_bf * z_t
    n_new = f_bf * state.n.astype(dtype) + i_bf
    h_new = (jax.nn.sigmoid(o_pre)
             * (c_new / jnp.maximum(jnp.abs(n_new), jnp.asarray(1.0, dtype=dtype))))

    if cfg.slstm_stabilizer == "per_cell":
        m_next = m_new.astype(jnp.float32)                     # (H, Dh)
    else:
        m_next = jnp.max(m_new, axis=-1).astype(jnp.float32)   # collapse Dh -> (H,)
    return h_new, SLSTMBlockState(h=h_new, c=c_new, n=n_new, m=m_next)


def slstm_step(block_params, x_t, state: SLSTMBlockState, cfg: XLSTMConfig):
    """Single sLSTM timestep. Gates use BOTH input and previous hidden state.

    x_t : (D,) compute-dtype (bf16). state : SLSTMBlockState.
    Returns (h_out: (H, Dh), new_state: SLSTMBlockState).
    """
    W, R, b = _slstm_stack_weights(block_params["slstm_gates"])
    Wx_t = jnp.einsum("ghdk,d->ghk", W, x_t) + b   # (4, H, Dh)
    return slstm_recurrence_step(R, Wx_t, state, cfg)


def _causal_conv1d(x_seq, conv_W, conv_b, kernel: int):
    """Depthwise causal Conv1D over a (T, D) sequence.

    conv_W: (D, kernel). Pads `kernel-1` on the left with zeros so no future
    leakage. Returns shape (T, D), same dtype as x_seq.
    """
    T, D = x_seq.shape
    pad = kernel - 1
    xp = jnp.concatenate([jnp.zeros((pad, D), dtype=x_seq.dtype), x_seq], axis=0)  # (T+pad, D)
    # Depthwise causal cross-correlation as a single fused conv (FlashRNN-style: keep the
    # input-side op as one batched kernel rather than T dynamic-slices). conv_W is laid out
    # (D, kernel) -> OIW (D, 1, kernel) with feature_group_count=D so each channel convolves
    # only with its own filter. VALID padding over the left-zero-padded signal == causal.
    dn = lax.conv_dimension_numbers((1, T + pad, D), (D, 1, kernel), ("NWC", "OIW", "NWC"))
    out = lax.conv_general_dilated(
        xp[None],                    # (1, T+pad, D)
        conv_W[:, None, :],          # (D, 1, kernel)
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=dn,
        feature_group_count=D,
    )[0]                             # (T, D)
    return out + conv_b


def slstm_block_step(block_params, x_t, state: SLSTMBlockState, cfg: XLSTMConfig):
    """One sLSTM block, one timestep (no conv1d — conv runs at block-forward level)."""
    D = cfg.embed_dim
    x_n = _layer_norm(x_t, block_params["pre_ln"]["scale"], block_params["pre_ln"]["bias"], cfg.eps)
    up = x_n @ block_params["up_proj"]["W"] + block_params["up_proj"]["b"]
    slstm_in, gate_in = jnp.split(up, 2, axis=-1)
    h_out, new_state = slstm_step(block_params, slstm_in, state, cfg)
    h_norm = _per_head_norm(h_out, block_params["group_ln"]["scale"], block_params["group_ln"]["bias"], cfg.eps)
    h_flat = h_norm.reshape(D)
    gated = h_flat * jax.nn.silu(gate_in)
    down = gated @ block_params["down_proj"]["W"] + block_params["down_proj"]["b"]
    return x_t + down, new_state


def slstm_block_forward(block_params, x_seq, init_state: SLSTMBlockState, cfg: XLSTMConfig):
    """Run an sLSTM block over a sequence x_seq: (T, D).

    If ``cfg.use_conv1d_in_slstm`` is set and the block contains a ``conv1d`` param,
    apply causal Conv1D over the entire sequence first. Input-side projections
    (pre-LN, up-proj, the four W·x gate projections) are computed for the whole
    sequence up front; only R·h and the c/n/m recurrence run in the scan.

    See ``mlstm_block_forward`` for the CPU large-T perf tradeoff. sLSTM is
    irreducibly sequential: its R·h memory mixing has no parallel-scan form (this is
    the state-tracking property), so only the per-step constant factor is reducible.
    """
    if cfg.use_conv1d_in_slstm and "conv1d" in block_params:
        x_seq = _causal_conv1d(x_seq, block_params["conv1d"]["W"], block_params["conv1d"]["b"],
                               cfg.conv1d_kernel)

    x_n = _layer_norm(
        x_seq, block_params["pre_ln"]["scale"], block_params["pre_ln"]["bias"], cfg.eps
    )
    up = x_n @ block_params["up_proj"]["W"] + block_params["up_proj"]["b"]
    slstm_in_seq, gate_in_seq = jnp.split(up, 2, axis=-1)
    W, R, b = _slstm_stack_weights(block_params["slstm_gates"])
    Wx_seq = jnp.einsum("ghdk,td->tghk", W, slstm_in_seq) + b[None]   # (T, 4, H, Dh)

    def step(state, Wx_t):
        h_t, new_state = slstm_recurrence_step(R, Wx_t, state, cfg)
        return new_state, h_t
    final_state, h_seq = lax.scan(step, init_state, Wx_seq)
    out = _block_output(h_seq, gate_in_seq, x_seq, block_params, cfg)
    return out, final_state


# =============================================================================
# Block-type dispatch
# =============================================================================

def block_init_state(cfg: XLSTMConfig, layer_idx: int, dtype=jnp.bfloat16):
    if cfg.block_types[layer_idx] == "mlstm":
        return init_block_state(cfg, dtype)
    return init_slstm_block_state(cfg, dtype)


def block_forward_dispatch(block_params, x_seq, init_state, cfg, block_type: str):
    if block_type == "mlstm":
        return mlstm_block_forward(block_params, x_seq, init_state, cfg)
    if block_type == "slstm":
        return slstm_block_forward(block_params, x_seq, init_state, cfg)
    raise ValueError(f"unknown block_type {block_type!r}")


# =============================================================================
# Direct linear head
# =============================================================================

def direct_head_forward(params_bf, h_last_seq, cfg: XLSTMConfig):
    """h_last_seq: (T, D) from final block. Take h_T, layer-norm, linear D→H.

    Returns (H,) float32.
    """
    h_T = h_last_seq[-1]  # (D,)
    h_n = _layer_norm(h_T, params_bf["head_ln"]["scale"], params_bf["head_ln"]["bias"], cfg.eps)
    out = h_n @ params_bf["out_proj_direct"]["W"] + params_bf["out_proj_direct"]["b"]
    return out.astype(jnp.float32)


# =============================================================================
# Casting helper
# =============================================================================

def _cast_pytree(tree, dtype):
    return jax.tree_util.tree_map(lambda a: a.astype(dtype), tree)


# =============================================================================
# Stack forward (orchestrator)
# =============================================================================

def _embed_and_stack_forward(x_seq, params_bf, cfg: XLSTMConfig):
    """Embed scalar series, push through L blocks. Returns (h_seq: (T, D), final_states)."""
    W_in = params_bf["input_embed"]["W"]   # (1, D)
    b_in = params_bf["input_embed"]["b"]   # (D,)
    x_emb = x_seq[:, None] @ W_in + b_in[None, :]  # (T, D)

    h = x_emb
    final_states = []
    for i, block_p in enumerate(params_bf["blocks"]):
        bt = cfg.block_types[i]
        init_state = block_init_state(cfg, i, dtype=h.dtype)
        h, fs = block_forward_dispatch(block_p, h, init_state, cfg, bt)
        final_states.append(fs)
    return h, tuple(final_states)


def xlstm_forward(params, x_seq, cfg: XLSTMConfig):
    """Full xLSTM / xLSTMTime forward pass.

    Parameters
    ----------
    params : pytree of float32 arrays (master weights).
    x_seq : (T,) float32 or bf16. Univariate scalar series.
    cfg : XLSTMConfig.

    Returns
    -------
    (preds, final_states)
        AR mode: preds shape (T,) float32 — per-position next-step prediction.
        Direct mode: preds shape (cfg.horizon,) float32 — direct multi-step forecast.
        final_states: tuple of per-block carry states (only meaningful in AR mode).
    """
    params_bf = _cast_pytree(params, jnp.bfloat16)
    x_f32 = x_seq.astype(jnp.float32)

    # RevIN (operates on f32 series for numerical stability)
    if cfg.use_revin:
        revin_p = params.get("revin", {})
        x_norm, revin_stats = revin_normalize(x_f32, revin_p)
    else:
        x_norm, revin_stats = x_f32, None

    # Decomposition: parallel branches with SHARED weights, summed
    if cfg.use_decomposition:
        decomp_p = params["decomp"]
        trend, seasonal = series_decompose(x_norm, decomp_p, cfg.decomp_kernel)
        h_trend, st_trend = _embed_and_stack_forward(trend.astype(jnp.bfloat16), params_bf, cfg)
        h_seas, _ = _embed_and_stack_forward(seasonal.astype(jnp.bfloat16), params_bf, cfg)
        h_seq = h_trend + h_seas
        final_states = st_trend
    else:
        h_seq, final_states = _embed_and_stack_forward(x_norm.astype(jnp.bfloat16), params_bf, cfg)

    # Head
    if cfg.decode_mode == "direct":
        pred = direct_head_forward(params_bf, h_seq, cfg)  # (H,)
    else:
        h_n = _layer_norm(h_seq, params_bf["head_ln"]["scale"], params_bf["head_ln"]["bias"], cfg.eps)
        out = h_n @ params_bf["out_proj"]["W"] + params_bf["out_proj"]["b"]  # (T, 1)
        pred = out.squeeze(-1).astype(jnp.float32)  # (T,)

    # RevIN denorm (only meaningful for the prediction part)
    if cfg.use_revin and revin_stats is not None:
        revin_p = params.get("revin", {})
        # Direct mode: pred is (H,) — broadcasts against (1,) stats fine.
        # AR mode: pred is (T,) — same broadcast.
        pred = revin_denormalize(pred, revin_p, revin_stats)
        pred = pred.squeeze() if pred.ndim > 1 else pred

    return pred, final_states


# =============================================================================
# Autoregressive decode (AR mode)
# =============================================================================

def decode(params, z_tail, h_steps: int, cfg: XLSTMConfig):
    """Encode ``z_tail`` then autoregress ``h_steps`` forward (AR mode only).

    Parameters
    ----------
    params : float32 pytree.
    z_tail : (ctx_len,) float32 — caller-normalized (z-score) when RevIN is off,
        RAW original-scale when RevIN is on (RevIN normalizes internally).
    h_steps : Python int (static). Number of forecast steps.
    cfg : XLSTMConfig (must have decode_mode == "ar").

    Returns
    -------
    preds : (h_steps,) float32 — normalized scale when RevIN is off (caller
        denormalizes with its z-score stats), original scale when RevIN is on.
    """
    assert cfg.decode_mode == "ar", "decode() is AR-only; use decode_direct for direct mode."
    _, final_states = xlstm_forward(params, z_tail, cfg)
    # With RevIN the recurrent states above were built from the RevIN-normalized
    # sequence and the trained out_proj emits normalized-space values (training
    # denormalizes via revin_denormalize in xlstm_forward). Roll the AR loop in
    # that same normalized space and denormalize the emissions at the end.
    if cfg.use_revin:
        revin_p = params.get("revin", {})
        z_norm, revin_stats = revin_normalize(z_tail.astype(jnp.float32), revin_p)
        last = z_norm[-1].astype(jnp.float32)
    else:
        revin_p, revin_stats = None, None
        last = z_tail[-1].astype(jnp.float32)
    params_bf = _cast_pytree(params, jnp.bfloat16)

    W_in = params_bf["input_embed"]["W"]
    b_in = params_bf["input_embed"]["b"]
    head_ln_s = params_bf["head_ln"]["scale"]
    head_ln_b = params_bf["head_ln"]["bias"]
    out_W = params_bf["out_proj"]["W"]
    out_b = params_bf["out_proj"]["b"]

    def ar_step(carry, _):
        last_val, states = carry
        last_bf = jnp.asarray(last_val, dtype=jnp.bfloat16)
        x = last_bf * W_in.squeeze(0) + b_in  # (D,)
        new_states = []
        for i, block_p in enumerate(params_bf["blocks"]):
            bt = cfg.block_types[i]
            if bt == "mlstm":
                x, ns = mlstm_block_step(block_p, x, states[i], cfg)
            else:
                x, ns = slstm_block_step(block_p, x, states[i], cfg)
            new_states.append(ns)
        h_n = _layer_norm(x, head_ln_s, head_ln_b, cfg.eps)
        pred = (h_n @ out_W + out_b).squeeze().astype(jnp.float32)
        return (pred, tuple(new_states)), pred

    _, preds = lax.scan(ar_step, (last, final_states), xs=None, length=h_steps)
    if cfg.use_revin:
        preds = revin_denormalize(preds, revin_p, revin_stats)
        preds = jnp.reshape(preds, (h_steps,))
    return preds


def decode_direct(params, z_tail, cfg: XLSTMConfig):
    """Direct multi-step forecast. z_tail: (ctx_len,) float32, possibly NOT pre-normalized
    when RevIN is enabled (RevIN runs inside xlstm_forward).

    Returns (cfg.horizon,) float32.
    """
    assert cfg.decode_mode == "direct", "decode_direct requires decode_mode='direct'."
    preds, _ = xlstm_forward(params, z_tail, cfg)
    return preds  # (cfg.horizon,)
