# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
import vllm._C  # noqa: F401

import vllm._custom_ops  # noqa: F401
from vllm.platforms import current_platform

K = 2048
D = 6144

pytestmark = pytest.mark.skipif(not current_platform.is_ppu(), reason="PPU-only kernel")


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, weight)
    output = torch.empty((1, D), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(output, gate_up)
    return output


@pytest.mark.parametrize("seed", [20260625, 20260626, 20260627])
def test_ppu_gate_silu_matches_vllm_bf16_path(seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((1, K), generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(
            (2 * D, K),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        / K**0.5
    ).contiguous()

    reference = _reference(x, weight)
    actual = torch.ops._C.ppu_gate_silu(x, weight)
    torch.cuda.synchronize()

    difference = (reference.float() - actual.float()).abs()
    cosine = F.cosine_similarity(
        reference.float().flatten(), actual.float().flatten(), dim=0
    )
    assert torch.isfinite(actual.float()).all()
    assert difference.max().item() <= 0.02
    assert cosine.item() >= 0.9999


def test_ppu_gate_silu_zero_input_is_exact_zero() -> None:
    x = torch.zeros((1, K), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((2 * D, K), device="cuda", dtype=torch.bfloat16)
    actual = torch.ops._C.ppu_gate_silu(x, weight)
    torch.cuda.synchronize()
    assert torch.count_nonzero(actual).item() == 0


def test_ppu_gate_silu_rejects_prefill_shape() -> None:
    x = torch.empty((2, K), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((2 * D, K), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match=r"x must have shape \[1, 2048\]"):
        torch.ops._C.ppu_gate_silu(x, weight)


def test_ppu_gate_silu_cudagraph_replays_updated_input() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    x = torch.randn((1, K), generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn((2 * D, K), generator=generator, device="cuda", dtype=x.dtype)
        / K**0.5
    ).contiguous()
    inputs = [x.clone(), -x, torch.zeros_like(x)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.ops._C.ppu_gate_silu(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = torch.ops._C.ppu_gate_silu(x, weight)

    for value in inputs:
        x.copy_(value)
        graph.replay()
        expected = torch.ops._C.ppu_gate_silu(x, weight)
        reference = _reference(x, weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert torch.isfinite(output).all()
        assert (output.float() - reference.float()).abs().max().item() <= 0.02
