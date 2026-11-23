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

## Results — two complete sweeps

Two complete 16-condition runs were performed, producing **32 actual benchmark results**. Each condition used 48 synthetic requests (1,024 input and 256 output tokens). All server configuration settings except the two swept parameters were held fixed.

### Request arrival rate (averaged over two runs and four max-num-seqs settings)

| Arrival rate | Output throughput (tokens/s) | Mean of P99 TTFT (ms) | Observed waiting peak |
|---:|---:|---:|---|
| **1 req/s** | **246.30** | **94.77** | **0** |
| **2 req/s** | **467.55** | **95.35** | **0** |
| **4 req/s** | **830.88** | **130.28** | **0** |
| **8 req/s** | **1,176.36** | **1,023.75** | **4–5** |

![Measured E3 throughput and TTFT by request rate](../../figures/E3_request_rate_throughput_ttft.svg)

### At 8 requests/s: effect of max-num-seqs

| Max sequences | Mean throughput (tokens/s) | Mean of P99 TTFT (ms) | Waiting peaks (run 1 / 2) |
|---:|---:|---:|---|
| 4 | 1,176.54 | 1,004.27 | 4 / 5 |
| 8 | 1,183.65 | 956.78 | 5 / 4 |
| 16 | 1,175.37 | 1,023.27 | 5 / 5 |
| 32 | 1,169.90 | 1,110.68 | 5 / 5 |

**Key observations:** Higher arrival rate raised output throughput from **246.30** to **1,176.36 tokens/s**, but at **8 requests/s** the waiting queue appeared and the mean of per-condition P99 TTFT values rose to about **1.02 seconds**. Increasing `max_num_seqs` from 4 to 32 at 8 requests/s did **not** yield a consistent throughput or tail-latency improvement. These measurements do not establish the exact GPU saturation point, and they do not compare continuous batching against a static batcher.

**Statistics:** Throughput figures are arithmetic means of the displayed measurements. P99 TTFT figures are the **mean of individual benchmark P99 values**, *not* the P99 computed from pooled requests. The observed waiting peak is from sampled metrics, which can miss shorter spikes.

**Source:** [All 32 console-transcribed measurements](../../results/E3_continuous_batching_summary_from_console.csv), runs `20261123T072647Z` and `20261123T074313Z`. The original runtime CSV files were not supplied to this repository.

## Recorded output

`results/E3_continuous_batching/<timestamp>/summary.csv` includes output throughput, request throughput, running/waiting peak/mean, mean/P50/P99 TTFT and TPOT, and auxiliary KV cache usage.

Per-condition folders contain `benchmark.json`, `benchmark.log` and `metrics_timeseries.csv`. No performance results are claimed until execution.

## Interpretation

- At a fixed arrival rate, compare different `max_num_seqs` for changes in queueing, TTFT and throughput.
- At a fixed `max_num_seqs`, compare arrival rates to detect saturation.
- Treat KV occupancy as a check that memory pressure is not mistaken for a scheduling limitation.
- Timing and outputs depend on the particular vLLM version, GPU, and workload.

**Status:** Two complete 16-condition sweeps executed (32 measurements, 2026-11-23).
