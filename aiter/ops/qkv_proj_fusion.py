# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Qwen3.8-Flash-Next QSA projection epilogue — routed op ``qkv_proj_fusion`` v1.

One call replaces the four steps a QSA decode/verify layer performs before its sparse
attention: the ``qkv_proj`` GEMM (gate + q + k + v), the indexer ``index_qk_proj`` GEMM,
Gemma RMSNorm of q / k / index-q, the partial NeoX mRoPE of q / k / index-q, and the K/V,
raw-index-K and rope-position cache stores. Prefill keeps the framework's own path.

The op is routed through :mod:`aiter.plugin`: an enabled out-of-tree provider may serve a
call (``aiter.plugin.configure(source=..., enable=[...])`` is the only way to enable one);
a decline, no provider, nothing enabled, or ``AITER_PLUGINS=0`` runs the native
implementation below. Tuned implementations (e.g. a fused FlyDSL gfx1250 kernel) live in
providers, not here.

Contract v1 (see ``docs/qkv_proj_fusion.md``):

* ``hidden_states`` ``[M, 2560]`` bf16; ``W_qkv`` ``[13312, 2560]`` and ``W_idx``
  ``[640, 2560]`` bf16. Column order is the layer's de-interleaved
  ``gate | q | k | v`` and ``index_q | index_k``.
* ``q_norm_w`` / ``k_norm_w`` ``[256]``, ``iq_norm_w`` ``[128]`` bf16: Gemma RMSNorm
  weights (the norm applies ``1 + w``), eps ``eps``.
* ``cos_cache`` / ``sin_cache`` ``[max_pos, rotary_dim // 2]`` bf16: the layer's cached
  mRoPE frequencies; ``positions`` int64 ``[M]``, ``[1, M]`` or ``[3, M]`` and
  ``rotary_dim`` / ``mrope_section_t|h|w`` / ``interleaved`` describe the rotation.
* ``slot_mapping`` int64 ``[M]``: flat cache slot per token (``page * block + row``);
  a negative or out-of-range slot skips every cache plane for that token.
* ``k_cache`` / ``v_cache`` ``[pages, block, kv_heads=2, 256]`` bf16, ``raw_k_cache``
  ``[pages, block, 1, 128]`` bf16, ``pos_cache`` ``[pages, block, 1, 3]`` int64
  (optional): written in place, block in {16, 32, 64}.
* Returns ``(gate [M, 6144] bf16, query [M, 24, 256] bf16, index_q [M, 4, 128] bf16)``.

The native implementation is pure torch: correct for every valid call (including CPU
tensors) and not fast. ``mrope_section_*`` / ``interleaved`` mirror ATOM's
``qsa_apply_mrope`` fast path, including its per-product rounding to the activation dtype.
Raises ``ValueError`` only for calls that violate the contract.
"""

from __future__ import annotations

import torch

from aiter.plugin import route as _route

OP_NAME = "qkv_proj_fusion"
OP_CONTRACT_VERSION = 1

#: label of the native implementation (``native_route``)
NATIVE_TORCH = "native.torch.v1"

MODEL_DIM = 2560
GATE = 6144
Q = 6144
K = 512
V = 512
INDEX_Q = 512
INDEX_K = 128
N_QKV = GATE + Q + K + V  # 13312
N_IDX = INDEX_Q + INDEX_K  # 640
Q_HEADS = 24
KV_HEADS = 2
INDEX_HEADS = 4
HEAD_DIM = 256
INDEX_HEAD_DIM = 128
BLOCKS = (16, 32, 64)
MODES = ("decode", "verify")


def _gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Row-wise RMSNorm with the Gemma ``(1 + w)`` scaling, rounded to ``x.dtype``."""

    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * scale * (1.0 + weight.float())).to(x.dtype)


def _axis_per_frequency(
    rotary_dim: int,
    section_t: int,
    section_h: int,
    section_w: int,
    interleaved: bool,
    device: torch.device,
) -> torch.Tensor:
    """mRoPE axis (0/1/2) each rotated frequency belongs to (ATOM's selector)."""

    half = rotary_dim // 2
    freq = torch.arange(rotary_dim, device=device) % half
    if interleaved:
        axis = torch.zeros_like(freq)
        axis = torch.where((freq % 3 == 1) & (freq < 3 * section_h), 1, axis)
        axis = torch.where((freq % 3 == 2) & (freq < 3 * section_w), 2, axis)
        return axis
    return torch.where(
        freq < section_t,
        0,
        torch.where(freq < section_t + section_h, 1, 2),
    )


def _three_axis_positions(positions: torch.Tensor, tokens: int) -> torch.Tensor:
    """``[3, M]`` int64 positions (a 1-D / ``[1, M]`` position repeats on every axis)."""

    if positions.dim() == 2:
        if positions.shape[0] == 3:
            return positions[:, :tokens]
        if positions.shape[0] == 1:
            return positions[:, :tokens].expand(3, tokens)
        raise ValueError("positions must be [tokens], [1, tokens] or [3, tokens]")
    return positions.reshape(1, -1)[:, :tokens].expand(3, tokens)


def _apply_mrope(
    x: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    section_t: int,
    section_h: int,
    section_w: int,
    interleaved: bool,
) -> torch.Tensor:
    """Partial NeoX mRoPE for ``[M, heads, dim]`` (ATOM ``_qsa_mrope_kernel`` semantics)."""

    tokens, _, dim = x.shape
    half = rotary_dim // 2
    if rotary_dim % 2 or rotary_dim > dim or half < 1:
        raise ValueError("rotary_dim must be even and <= head dim")
    if cos_cache.shape[-1] < half or sin_cache.shape[-1] < half:
        raise ValueError("cos/sin cache is narrower than rotary_dim // 2")
    if tokens == 0:
        return x
    pos3 = _three_axis_positions(positions, tokens).to(torch.long)
    axis = _axis_per_frequency(
        rotary_dim, section_t, section_h, section_w, interleaved, x.device
    )
    freq = (torch.arange(rotary_dim, device=x.device) % half).reshape(1, rotary_dim)
    pos = pos3[axis, :]  # [rotary_dim, M]
    index = (pos * half + freq.reshape(-1, 1)).t().reshape(-1)  # [M * rotary_dim]
    cos = cos_cache.reshape(-1)[index].reshape(tokens, 1, rotary_dim).float()
    sin = sin_cache.reshape(-1)[index].reshape(tokens, 1, rotary_dim).float()

    dtype = x.dtype
    xr = x[..., :rotary_dim]
    paired = torch.cat([xr[..., half:rotary_dim], xr[..., :half]], dim=-1)
    sign = torch.cat(
        [
            -torch.ones(half, dtype=torch.float32, device=x.device),
            torch.ones(half, dtype=torch.float32, device=x.device),
        ]
    )
    # Keep the activation-dtype roundings of each product, as ATOM's kernel does.
    xc = (xr.float() * cos).to(dtype).float()
    ys = (paired.float() * sign * sin).to(dtype).float()
    rotated = (xc + ys).to(dtype)
    if rotary_dim == dim:
        return rotated
    return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)


def _validate(
    hidden_states,
    W_qkv,
    W_idx,
    q_norm_w,
    k_norm_w,
    iq_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    raw_k_cache,
    pos_cache,
    rotary_dim,
    mode,
):
    def tensor(name, value, shape=None, dtype=None):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(value.shape)}")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
        return value

    h = tensor("hidden_states", hidden_states, None, torch.bfloat16)
    if h.dim() != 2 or h.shape[1] != MODEL_DIM:
        raise ValueError(f"hidden_states must be [M, {MODEL_DIM}]")
    if not h.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    tokens = int(h.shape[0])
    if not 1 <= tokens <= 1792:
        raise ValueError("M outside the v1 envelope [1, 1792]")
    tensor("W_qkv", W_qkv, (N_QKV, MODEL_DIM), torch.bfloat16)
    tensor("W_idx", W_idx, (N_IDX, MODEL_DIM), torch.bfloat16)
    tensor("q_norm_w", q_norm_w, (HEAD_DIM,), torch.bfloat16)
    tensor("k_norm_w", k_norm_w, (HEAD_DIM,), torch.bfloat16)
    tensor("iq_norm_w", iq_norm_w, (INDEX_HEAD_DIM,), torch.bfloat16)
    for name, cache in (("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        tensor(name, cache, None, torch.bfloat16)
        if cache.dim() != 2:
            raise ValueError(f"{name} must be [max_pos, rotary_dim // 2]")
    tensor("positions", positions, None, torch.int64)
    tensor("slot_mapping", slot_mapping, None, torch.int64)
    if positions.shape[-1] < tokens or slot_mapping.numel() < tokens:
        raise ValueError("positions / slot_mapping are shorter than M")
    for name, cache, head_dim in (
        ("k_cache", k_cache, HEAD_DIM),
        ("v_cache", v_cache, HEAD_DIM),
    ):
        tensor(name, cache, None, torch.bfloat16)
        if cache.dim() != 4 or cache.shape[2:] != (KV_HEADS, head_dim):
            raise ValueError(f"{name} must be [pages, block, {KV_HEADS}, {head_dim}]")
        if int(cache.shape[1]) not in BLOCKS:
            raise ValueError(f"{name} block must be in {BLOCKS}")
    tensor("raw_k_cache", raw_k_cache, None, torch.bfloat16)
    if raw_k_cache.dim() != 4 or raw_k_cache.shape[2:] != (1, INDEX_HEAD_DIM):
        raise ValueError(f"raw_k_cache must be [pages, block, 1, {INDEX_HEAD_DIM}]")
    if pos_cache is not None:
        tensor("pos_cache", pos_cache, None, torch.int64)
        if pos_cache.dim() != 4 or pos_cache.shape[2:] != (1, 3):
            raise ValueError("pos_cache must be [pages, block, 1, 3]")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if int(rotary_dim) % 2 or int(rotary_dim) > HEAD_DIM:
        raise ValueError("rotary_dim must be even and <= head dim")
    return tokens


def native_route(
    hidden_states,
    W_qkv,
    W_idx,
    q_norm_w,
    k_norm_w,
    iq_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    raw_k_cache,
    pos_cache=None,
    rotary_dim: int = 64,
    mrope_section_t: int = 11,
    mrope_section_h: int = 11,
    mrope_section_w: int = 10,
    interleaved: bool = True,
    eps: float = 1e-06,
    mode: str = "decode",
) -> str:
    """Which native implementation serves a call when no provider does.

    v1 has one: ``native.torch.v1``. Host-only; reads no tensor values.
    """

    return NATIVE_TORCH


@_route(op="qkv_proj_fusion", version=1)
def qkv_proj_fusion(
    hidden_states: torch.Tensor,
    W_qkv: torch.Tensor,
    W_idx: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    iq_norm_w: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    raw_k_cache: torch.Tensor,
    pos_cache: torch.Tensor | None = None,
    rotary_dim: int = 64,
    mrope_section_t: int = 11,
    mrope_section_h: int = 11,
    mrope_section_w: int = 10,
    interleaved: bool = True,
    eps: float = 1e-06,
    mode: str = "decode",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QSA projection epilogue; op contract ``qkv_proj_fusion`` v1.

    One launch/one native pass computes the gate projection, the q/k/v projections, the
    indexer q/k projections, the Gemma RMSNorm of q/k/index-q, the partial NeoX mRoPE of
    query/key/index_q, and writes the K/V, raw-index-K and rope-position caches in place.

    Routed (``aiter.plugin``, op ``qkv_proj_fusion`` v1): an enabled provider may serve
    the call (declines are safe -- the native implementation below is general); otherwise
    the native torch implementation runs. :func:`native_route` names it.

    Returns ``(gate, query, index_q)``; see the module docstring for shapes. Raises
    ``ValueError`` only for calls that violate the contract.
    """

    tokens = _validate(
        hidden_states,
        W_qkv,
        W_idx,
        q_norm_w,
        k_norm_w,
        iq_norm_w,
        cos_cache,
        sin_cache,
        positions,
        slot_mapping,
        k_cache,
        v_cache,
        raw_k_cache,
        pos_cache,
        rotary_dim,
        mode,
    )
    h = hidden_states
    y = torch.matmul(h, W_qkv.t())
    y_idx = torch.matmul(h, W_idx.t())

    gate = y[:, :GATE]
    q = y[:, GATE : GATE + Q].view(tokens, Q_HEADS, HEAD_DIM)
    k = y[:, GATE + Q : GATE + Q + K].view(tokens, KV_HEADS, HEAD_DIM)
    v = y[:, GATE + Q + K :].view(tokens, KV_HEADS, HEAD_DIM)
    iq = y_idx[:, :INDEX_Q].view(tokens, INDEX_HEADS, INDEX_HEAD_DIM)
    ik = y_idx[:, INDEX_Q:].view(tokens, 1, INDEX_HEAD_DIM)

    query = _apply_mrope(
        _gemma_rmsnorm(q, q_norm_w, eps),
        cos_cache,
        sin_cache,
        positions,
        rotary_dim,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        interleaved,
    )
    key = _apply_mrope(
        _gemma_rmsnorm(k, k_norm_w, eps),
        cos_cache,
        sin_cache,
        positions,
        rotary_dim,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        interleaved,
    )
    index_q = _apply_mrope(
        _gemma_rmsnorm(iq, iq_norm_w, eps),
        cos_cache,
        sin_cache,
        positions,
        rotary_dim,
        mrope_section_t,
        mrope_section_h,
        mrope_section_w,
        interleaved,
    )

    block = int(k_cache.shape[1])
    capacity = int(k_cache.shape[0]) * block
    slots = slot_mapping.reshape(-1)[:tokens].to(torch.long)
    valid = (slots >= 0) & (slots < capacity)
    if bool(valid.any()):
        rows = slots[valid]
        k_cache.reshape(capacity, KV_HEADS, HEAD_DIM)[rows] = key[valid]
        v_cache.reshape(capacity, KV_HEADS, HEAD_DIM)[rows] = v[valid]
        raw_k_cache.reshape(capacity, 1, INDEX_HEAD_DIM)[rows] = ik[valid]
        if pos_cache is not None:
            pos3 = _three_axis_positions(positions, tokens)
            pos_cache.reshape(capacity, 1, 3)[rows] = pos3[:, valid].t().reshape(-1, 1, 3)

    return gate, query, index_q


__all__ = [
    "NATIVE_TORCH",
    "OP_CONTRACT_VERSION",
    "OP_NAME",
    "native_route",
    "qkv_proj_fusion",
]
