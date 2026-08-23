# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity test for the fused GDN intra-chunk prefill kernel.

``fused_chunk_intra`` (fla/ops/chunk_fused.py) fuses the first 4 stages of
``chunk_gated_delta_rule_fwd`` (cumsum -> scaled_dot_kkt -> solve_tril ->
recompute_w_u) into a single Triton kernel. This test compares the full fused
prefill path against the reference ``fla_chunk_gated_delta_rule`` chain, plus
a direct comparison of the fused kernel's intermediate outputs (w, u, g_cu)
against the eager 4-stage chain.
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.model_executor.layers.fla.ops.chunk_fused import fused_chunk_intra
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

# GDN config (Qwen3-Next: head_k_dim = head_v_dim = 128, grouped q/k heads).
K = 128
V = 128
ATOL = 2e-2
RTOL = 2e-2


def _make_inputs(
    seq_lens: list[int],
    num_k_heads: int,
    num_v_heads: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build flattened (B=1) q/k/v/g/beta and initial_state for seq_lens."""
    total = sum(seq_lens)
    q = torch.randn(1, total, num_k_heads, K, dtype=dtype, device=device)
    k = torch.randn(1, total, num_k_heads, K, dtype=dtype, device=device)
    v = torch.randn(1, total, num_v_heads, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.rand(1, total, num_v_heads, device=device))
    beta = torch.rand(1, total, num_v_heads, dtype=dtype, device=device).sigmoid()
    initial_state = torch.randn(
        len(seq_lens), num_v_heads, V, K, dtype=torch.float32, device=device
    )
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.int32, device=device
    )
    return q, k, v, g, beta, initial_state, cu_seqlens


def _fused_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ChunkGatedDeltaRule.forward_fused without the CustomOp wrapper."""
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    w, u, g_cu = fused_chunk_intra(k, v, g, beta, cu_seqlens=cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cu,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cu,
        scale=k.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
    )
    return o.to(q.dtype), final_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize(
    "seq_lens", [[63], [64], [128], [300], [63, 64], [37, 300, 128]]
)
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 4), (4, 4)])
def test_fused_chunk_intra_matches_fl(
    seq_lens: list[int], num_k_heads: int, num_v_heads: int
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        seq_lens, num_k_heads, num_v_heads, dtype, device
    )
    # Single-sequence cases also exercise the non-varlen path.
    if len(seq_lens) == 1:
        cu_seqlens = None

    o_ref, state_ref = fla_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    o_fused, state_fused = _fused_gdn_prefill(
        q, k, v, g, beta, initial_state.clone(), cu_seqlens
    )

    torch.testing.assert_close(o_fused, o_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state_fused, state_ref, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device")
@pytest.mark.parametrize("seq_lens", [[63], [300], [63, 64], [37, 300, 128]])
def test_fused_chunk_intra_intermediates_match_eager(seq_lens: list[int]):
    """w/u/g_cu of the fused kernel vs the eager 4-stage FLA chain."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_k_heads, num_v_heads = 2, 4

    _, k, v, g, beta, _, cu_seqlens = _make_inputs(
        seq_lens, num_k_heads, num_v_heads, dtype, device
    )
    if len(seq_lens) == 1:
        cu_seqlens = None
    k = l2norm_fwd(k)

    w_fused, u_fused, g_cu_fused = fused_chunk_intra(
        k, v, g, beta, cu_seqlens=cu_seqlens
    )

    g_ref = chunk_local_cumsum(
        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g_ref,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w_ref, u_ref = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_ref, cu_seqlens=cu_seqlens
    )

    torch.testing.assert_close(g_cu_fused, g_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w_fused, w_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(u_fused, u_ref, atol=ATOL, rtol=RTOL)
