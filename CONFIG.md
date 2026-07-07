# 本地模型运行配置

## Qwen3.5-2B 离线生成测试

- 模型名称：Qwen3.5-2B
- 用户给定位置：`../huggingface`
- 实际模型目录：`/home/maru/huggingface/Qwen3.5-2B`
- 配置文件路径：`/home/maru/huggingface/Qwen3.5-2B/config.json`
- 测试脚本路径：`.temp/qwen3_5_2b_generate_bench.py`
- 结果文件路径：`.temp/qwen3_5_2b_generate_bench.json`
- 日志文件路径：`.log/qwen3_5_2b_generate_bench.log`

### 运行命令

```bash
cd /home/maru/project/vllm
.venv/bin/python .temp/qwen3_5_2b_generate_bench.py > .log/qwen3_5_2b_generate_bench.log 2>&1
```

### 关键参数

- 运行方式：离线 `AsyncLLM.generate`，`RequestOutputKind.DELTA`
- 测量方式：首个流式 token 到达时间计算 TTFT，后续 token 平均间隔计算 TPOT
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `language_model_only=True`
- `skip_mm_profiling=True`
- `dtype=float16`
- `max_model_len=2048`
- `max_num_seqs=16`
- `gpu_memory_utilization=0.80`
- `batch_size=4`
- `input_len=128`
- `output_len=32`
- `warmup_iters=1`
- `bench_iters=3`

### 测试结果

- TTFT 平均值：77.75 ms
- TTFT P50：80.43 ms
- TTFT 范围：61.36 ms 到 87.65 ms
- TPOT 平均值：17.05 ms/token
- TPOT P50：16.45 ms/token
- TPOT 范围：16.09 ms/token 到 18.70 ms/token
- 每轮 batch 端到端延迟平均值：613.12 ms

### 环境备注

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB 显存。
- 首次尝试 OpenAI endpoint 时失败点包括 FlashInfer sampler 需要 `nvcc`，以及默认 `max_num_seqs=256` 超过当前 Mamba cache blocks；离线测试通过禁用 FlashInfer sampler 和降低 `max_num_seqs` 跑通。
- Qwen3.5-2B 目录包含 vision/video 配置；本次只测试文本生成，因此启用 `language_model_only=True`。
