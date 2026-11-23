# E3 — Continuous Batching & Request Scheduling

## Objective

Measure how `max_num_seqs` and incoming request rate affect running/waiting requests, throughput and TTFT/TPOT in a real vLLM server. This is a **continuous batching workload sweep**, not a static-vs-continuous batching comparison.

## Experiment configuration

| Parameter | Setting |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct, FP16 |
| GPU memory fraction | 0.80 (fixed) |
| Max model length / batch token budget | 4096 / 4096 |
| Input/output tokens per request | 1024 / 256 |
| Requests per condition | 48 |
| Seed | 1993 |
| `max_num_seqs` | **4 / 8 / 16 / 32** |
| Request arrival rate | **1 / 2 / 4 / 8 requests/s** |

Sixteen conditions; `max_num_seqs` changes restart the server. Actual measured throughput, queue length and latency are reported in timestamped CSV files. GPU metrics are sampled via `/metrics`.

## Commands

From `03-pagedattention-memory-management/`:

```bash
# Smoke test
python experiments/E3_continuous_batching/run_e3.py --max-num-seqs 8 --request-rates 2 --num-prompts 12

# Full sweep
python experiments/E3_continuous_batching/run_e3.py
```

## Recorded output

`results/E3_continuous_batching/<timestamp>/summary.csv` includes output throughput, request throughput, running/waiting peak/mean, mean/P50/P99 TTFT and TPOT, and auxiliary KV cache usage.

Per-condition folders contain `benchmark.json`, `benchmark.log` and `metrics_timeseries.csv`. No performance results are claimed until execution.

## Interpretation

- At a fixed arrival rate, compare different `max_num_seqs` for changes in queueing, TTFT and throughput.
- At a fixed `max_num_seqs`, compare arrival rates to detect saturation.
- Treat KV occupancy as a check that memory pressure is not mistaken for a scheduling limitation.
- Timing and outputs depend on the particular vLLM version, GPU, and workload.

**Status:** Implementation added; awaiting actual E3 execution.
