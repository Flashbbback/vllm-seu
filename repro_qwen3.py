# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
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


def main() -> None:
    model = "/home/maru/huggingface/Qwen3-0.6B"
    llm = LLM(
        model=model,
        tokenizer=model,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
    )
    outputs = llm.generate(
        ["用一句话介绍 paged attention。"],
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    for output in outputs:
        print("PROMPT:", output.prompt)
        print("OUTPUT:", output.outputs[0].text)


if __name__ == "__main__":
    main()
