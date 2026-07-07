# Progress

## 07-06 13:23

跑通 `/home/maru/huggingface/Qwen3.5-2B` 的离线 `AsyncLLM.generate` 文本生成测试，测试脚本位于 `.temp/qwen3_5_2b_generate_bench.py`，结果写入 `.temp/qwen3_5_2b_generate_bench.json`。本次小样本结果：TTFT 平均 77.75 ms，TPOT 平均 17.05 ms/token。
