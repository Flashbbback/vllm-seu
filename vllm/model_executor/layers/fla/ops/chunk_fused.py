# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices
from .op import exp
from .solve_tril import FLA_TRIL_PRECISION
from .utils import FLA_CHUNK_SIZE, input_guard


@triton.jit
def _extract_block16(b_A, i_r: tl.constexpr, i_c: tl.constexpr):
    """Extract the (i_r, i_c)-th 16x16 sub-block of a [64, 64] register tile.

    Args:
        b_A: [64, 64] fp32 register tile.
        i_r: block row index in [0, 4).
        i_c: block column index in [0, 4).

    Returns:
        The [16, 16] fp32 sub-block at block position (i_r, i_c).
    """
    o_4 = tl.arange(0, 4)
    m_blk = (o_4[:, None, None, None] == i_r) & (o_4[None, None, :, None] == i_c)
    b_4d = tl.reshape(b_A, (4, 16, 4, 16))
    return tl.sum(tl.sum(tl.where(m_blk, b_4d, 0.0), 2), 0)


@triton.jit
def _extract_row16(b_blk, i):
    """Extract row i of a [16, 16] register tile as a [16] vector.

    Args:
        b_blk: [16, 16] fp32 register tile.
        i: row index (runtime scalar).

    Returns:
        The [16] fp32 row vector.
    """
    o_i = tl.arange(0, 16)
    return tl.sum(tl.where((o_i == i)[:, None], b_blk, 0.0), 0)


@triton.jit
def _scatter_block16(b_Ai, b_blk, i_r: tl.constexpr, i_c: tl.constexpr):
    """Write a [16, 16] tile into the (i_r, i_c)-th block of a [64, 64] tile.

    Args:
        b_Ai: [64, 64] fp32 accumulator tile.
        b_blk: [16, 16] fp32 tile to write.
        i_r: block row index in [0, 4).
        i_c: block column index in [0, 4).

    Returns:
        The updated [64, 64] fp32 tile.
    """
    o_t = tl.arange(0, 64)
    m_dst = ((o_t[:, None] // 16) == i_r) & ((o_t[None, :] // 16) == i_c)
    b_full = tl.reshape(
        tl.broadcast_to(tl.expand_dims(tl.expand_dims(b_blk, 0), 2), (4, 16, 4, 16)),
        (64, 64),
    )
    return tl.where(m_dst, b_full, b_Ai)


@triton.jit
def _solve_tril_64x64(b_A, T, i_t, DOT_PRECISION: tl.constexpr):
    """Compute (I + A)^-1 of a strictly lower triangular [64, 64] fp32 tile.

    Register-resident port of ``merge_16x16_to_64x64_inverse_kernel``
    (solve_tril.py): the tile is split into 16x16 blocks, diagonal blocks are
    inverted by sequential row forward substitution, and off-diagonal blocks
    follow by block back-substitution. All math stays in fp32 with
    DOT_PRECISION dots, bit-compatible with the unfused kernel.

    Args:
        b_A: strictly lower triangular [64, 64] fp32 tile; rows/columns beyond
            the chunk boundary must already be zeroed.
        T: sequence length (runtime scalar).
        i_t: chunk index within the sequence.
        DOT_PRECISION: dot input precision, shared with solve_tril.

    Returns:
        The [64, 64] fp32 inverse tile.
    """
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    # raw diagonal blocks of A
    b_A_11 = _extract_block16(b_A, 0, 0)
    b_A_22 = _extract_block16(b_A, 1, 1)
    b_A_33 = _extract_block16(b_A, 2, 2)
    b_A_44 = _extract_block16(b_A, 3, 3)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_A_11, 0)
    b_Ai_22 = -tl.where(m_A, b_A_22, 0)
    b_Ai_33 = -tl.where(m_A, b_A_33, 0)
    b_Ai_44 = -tl.where(m_A, b_A_44, 0)

    for i in range(2, min(16, T - i_t * 64)):
        b_a_11 = -_extract_row16(b_A_11, i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * 64)):
        b_a_22 = -_extract_row16(b_A_22, i - 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * 64)):
        b_a_33 = -_extract_row16(b_A_33, i - 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * 64)):
        b_a_44 = -_extract_row16(b_A_44, i - 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    # raw off-diagonal blocks of A
    b_A_21 = _extract_block16(b_A, 1, 0)
    b_A_31 = _extract_block16(b_A, 2, 0)
    b_A_32 = _extract_block16(b_A, 2, 1)
    b_A_41 = _extract_block16(b_A, 3, 0)
    b_A_42 = _extract_block16(b_A, 3, 1)
    b_A_43 = _extract_block16(b_A, 3, 2)

    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )

    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    b_Ai = tl.zeros([64, 64], dtype=tl.float32)
    b_Ai = _scatter_block16(b_Ai, b_Ai_11, 0, 0)
    b_Ai = _scatter_block16(b_Ai, b_Ai_21, 1, 0)
    b_Ai = _scatter_block16(b_Ai, b_Ai_22, 1, 1)
    b_Ai = _scatter_block16(b_Ai, b_Ai_31, 2, 0)
    b_Ai = _scatter_block16(b_Ai, b_Ai_32, 2, 1)
    b_Ai = _scatter_block16(b_Ai, b_Ai_33, 2, 2)
    b_Ai = _scatter_block16(b_Ai, b_Ai_41, 3, 0)
    b_Ai = _scatter_block16(b_Ai, b_Ai_42, 3, 1)
    b_Ai = _scatter_block16(b_Ai, b_Ai_43, 3, 2)
    b_Ai = _scatter_block16(b_Ai, b_Ai_44, 3, 3)
    return b_Ai


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [1, 2]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_intra_kernel(
    k,
    v,
    g,
    beta,
    w,
    u,
    g_cu,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Fused intra-chunk GDN prefill: cumsum -> KKT -> tril inverse -> WY.

    Each program handles one chunk of one head and keeps the intermediate
    matrices A and Ai in registers; only g_cu (fp32), w and u (k.dtype) are
    written back to HBM, replacing the 4 eager kernels
    ``chunk_local_cumsum`` + ``chunk_scaled_dot_kkt_fwd`` + ``solve_tril`` +
    ``recompute_w_u_fwd``.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    # 1. chunk-local cumulative sum of the gate (fp32), same as cumsum.py
    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g_cu = tl.make_block_ptr(
        g_cu + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.cumsum(tl.load(p_g, boundary_check=(0,)).to(tl.float32), axis=0)
    tl.store(p_g_cu, b_g.to(p_g_cu.dtype.element_ty), boundary_check=(0,))

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    # 2. A = strict_lower((K * beta) K^T * exp(g_i - g_j)) in fp32,
    #    same as chunk_scaled_dot_kkt.py
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))

    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A = b_A * exp(b_g_diff)
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    # 3. Ai = (I + A)^-1 in fp32 (solve_tril.py), then downcast to k.dtype
    #    with rtne, matching the HBM round-trip of the unfused chain
    b_Ai = _solve_tril_64x64(b_A, T, i_t, DOT_PRECISION)
    b_Ai = b_Ai.to(k.dtype.element_ty, fp_downcast_rounding="rtne")

    # 4. u = Ai @ (v * beta); w = Ai @ (k * beta * exp(g_cu)),
    #    same as wy_fast.py
    b_g_exp = tl.exp(b_g)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g_exp[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_Ai, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def fused_chunk_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Fused intra-chunk prefill for GDN (inference only).

    Fuses the first 4 stages of ``chunk_gated_delta_rule_fwd``
    (chunk.py) into a single Triton kernel:
    chunk_local_cumsum -> chunk_scaled_dot_kkt_fwd -> solve_tril ->
    recompute_w_u_fwd. The intermediate matrices A and Ai never touch HBM.
    The dtype flow is identical to the unfused chain: the inverse is
    computed in fp32 and downcast to k.dtype before the w/u dots.

    Args:
        k (torch.Tensor):
            Keys of shape `[B, T, Hg, K]`. K must be 128.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`. V must be 128.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]`; requires B == 1.
            Default: `None`.
        chunk_indices (torch.Tensor):
            Pre-computed chunk indices. If None and cu_seqlens is provided,
            computed internally. Default: `None`.

    Returns:
        A tuple of (w, u, g_cu):
        - w: `[B, T, H, K]` in k.dtype
        - u: `[B, T, H, V]` in v.dtype
        - g_cu: `[B, T, H]` chunk-local cumulative gate in fp32

    Raises:
        ValueError: If K != 128 or V != 128; the caller is expected to fall
            back to the unfused path.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = FLA_CHUNK_SIZE
    if K != 128 or V != 128 or BT != 64:
        raise ValueError(
            f"fused_chunk_intra requires K == V == 128 and FLA_CHUNK_SIZE == 64, "
            f"but got K={K}, V={V}, FLA_CHUNK_SIZE={BT}"
        )
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = 64
    BV = 64
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    g_cu = torch.empty(B, T, H, device=k.device, dtype=torch.float32)
    fused_chunk_intra_kernel[(NT, B * H)](
        k=k,
        v=v,
        g=g,
        beta=beta,
        w=w,
        u=u,
        g_cu=g_cu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return w, u, g_cu
