# vLLM Inference & KV Cache Monitoring

Experiments on LLM serving performance using vLLM, Qwen2.5-0.5B-Instruct, and an NVIDIA T4 GPU.

## Experiments
- Continuous batching: concurrent requests (1, 8, 24)
- Scheduler limits: `max_num_seqs` (8, 16, 24)
- KV cache capacity: 256, 128, and 96 MiB
- Prefix caching: planned cold/warm comparison (not yet verified)

## Observations
- Increasing `max_num_seqs` from 8 to 24 for 24 concurrent requests raised throughput from 235.58 to 644.17 output tokens/s and reduced TTFT P95 from 17,016.77 to 276.90 ms.
- KV cache peak usage increased from 35.88% (256 MiB) to 96.06% (96 MiB), without observed preemptions or failed requests in these runs.

> Measurements reflect individual Colab T4 runs, not production-scale benchmarks. Prefix caching outcomes are pending verification.
