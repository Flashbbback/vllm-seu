# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

cuda_home = (
    Path(__file__).resolve().parent
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
    / "cu13"
)
if cuda_home.exists():
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ["PATH"] = f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}"

from vllm import LLM, SamplingParams  # noqa: E402


def run(num_seqs: int = 64) -> None:
    random.seed(0)
    max_input_len = 1024
    max_output_len = 1024

    model = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(
        model=model,
        tokenizer=model,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        enforce_eager=False,
    )

    prompt_token_ids = [
        [random.randint(0, 10000) for _ in range(random.randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=random.randint(100, max_output_len),
        )
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(max_tokens=1), use_tqdm=False)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    expected_output_tokens = sum(sp.max_tokens for sp in sampling_params)
    actual_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    input_tokens = sum(len(p) for p in prompt_token_ids)

    print(f"Requests: {num_seqs}")
    print(f"Input tokens: {input_tokens}")
    print(f"Expected output tokens: {expected_output_tokens}")
    print(f"Actual output tokens: {actual_output_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Expected throughput: {expected_output_tokens / elapsed:.2f} tok/s")
    print(f"Actual throughput: {actual_output_tokens / elapsed:.2f} tok/s")
    total_token_throughput = (input_tokens + actual_output_tokens) / elapsed
    print(f"Total token throughput: {total_token_throughput:.2f} tok/s")


if __name__ == "__main__":
    num_seqs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    run(num_seqs)
