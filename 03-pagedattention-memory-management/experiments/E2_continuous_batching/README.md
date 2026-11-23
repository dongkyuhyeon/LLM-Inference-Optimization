# E2 — Continuous Batching & Request Scheduling

## Research question

How do request arrival rate and `max_num_seqs` affect **throughput, queueing, TTFT and TPOT** when a real vLLM server schedules many simultaneous inference requests?

This is an **actual model serving benchmark**, not a theoretical allocator simulation or a static-vs-continuous baseline comparison.

## Fixed conditions

| Item | Setting |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Precision | FP16 |
| GPU memory fraction | 0.80 |
| Max model length | 4096 |
| Token budget per scheduling step | 4096 |
| Synthetic input | 1024 tokens/request |
| Generated output | 256 tokens/request (ignore EOS) |
| Requests | 48 per benchmark |
| Seed | 1993 |

## Swept variables

| Variable | Values |
|---|---|
| `max_num_seqs` | **4, 8, 16, 32** |
| Arrival rate | **1, 2, 4, 8 requests/s** |

The full design contains **16 benchmark conditions**. The script **restarts the vLLM server** when changing `max_num_seqs`; arrival-rate runs reuse that server. Metrics are sampled during the benchmark.

## Execution

First check that `vllm --version` and `vllm bench serve --help` work in the GPU environment.

From `03-pagedattention-memory-management/`:

```bash
# Smoke test (1 condition)
python experiments/E2_continuous_batching/run_e2.py \
    --max-num-seqs 8 --request-rates 2 --num-prompts 12

# Full experiment (16 conditions)
python experiments/E2_continuous_batching/run_e2.py
```

Alternatively run a smaller controlled sweep:

```bash
python experiments/E2_continuous_batching/run_e2.py \
    --max-num-seqs 4 8 16 32 --request-rates 2 8 \
    --num-prompts 48
```

## Recorded metrics

| Metric | Meaning |
|---|---|
| `output_throughput_tps` | Actual output tokens per second |
| `request_throughput_rps` | Completed requests per second |
| `running_peak`, `running_mean` | Concurrent requests actively scheduled |
| `waiting_peak`, `waiting_mean` | Requests waiting for execution |
| `mean_ttft_ms`, `p99_ttft_ms` | Time from client request to first token |
| `mean_tpot_ms`, `p99_tpot_ms` | Time per generated output token |
| `kv_usage_peak` | Auxiliary KV cache occupancy |
| `preemption_delta` | Auxiliary preemption counter, if exposed |

An absent metric is left **blank**, not interpreted as zero. The sampler may miss transient peaks. Different vLLM releases expose different metric names.

## Saved artifacts

```text
results/E2_continuous_batching/<UTC_timestamp>/
├── configuration.json
├── summary.csv
└── seqs_<N>/
    ├── server.log
    └── rate_<R>/
        ├── benchmark.log
        ├── benchmark.json
        └── metrics_timeseries.csv
```

## Interpretation plan

1. Hold the workload and GPU memory budget fixed while changing `max_num_seqs`.
2. Observe when requests start accumulating in the Waiting Queue.
3. Compare TTFT p99 and throughput at the same request rate.
4. Examine TPOT for the trade-off of admitting more concurrent sequences.
5. Separate scheduling-capacity limitations from KV-capacity limitations.

**Status:** implementation committed. GPU results have not yet been collected for this renumbered E2.

### Notes

- This benchmark tests real vLLM iteration-level scheduling; it **does not** measure the improvement factor against a static batcher.
- Random synthetic tokens are used, not production chat prompts. Generated token length is controlled by benchmark settings.
- Use the installed vLLM version and GPU information when interpreting results.
