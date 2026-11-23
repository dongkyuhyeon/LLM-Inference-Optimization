# E5 — Prefix Caching & Serving Optimization

## Goal

Measure whether vLLM **Automatic Prefix Caching (APC)** reuses previously computed KV cache when several requests have **the same token prefix**, and how that affects **time to first token (TTFT), end-to-end latency, and output throughput**.

This is a real model-serving experiment, not a synthetic allocator calculation. The benchmark sends actual streamed requests to a local vLLM server.

## Controlled experimental design

A **2 × 2 × 3** design: cache ON/OFF × shared/unique prefix × prefix lengths 512/1024/2048 tokens.

| Variable | Settings |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, FP16 |
| Automatic Prefix Caching | **OFF / ON** (explicit server setting) |
| Prefix pattern | **Shared** (same beginning + different suffix) / **Unique** (different beginning per request) |
| Target prefix lengths | **512 / 1024 / 2048** tokenizer tokens |
| Number of measured requests | **12 per cell** (plus one unmeasured warm-up request) |
| Output length | **64 tokens** (`ignore_eos=True`) |
| Request arrival | **Sequential**, one at a time |
| GPU memory fraction | **0.80** |
| Max model length | **4096** |
| Max concurrently scheduled sequences | **8** |

The server restarts for each (cache setting, prefix length) combination; within a server, the unique workload is measured **first**, then the shared workload. This reduces cross-cell cache contamination. The shared workload reuses identical prefix text but a different question suffix. Unique workloads change the **beginning** of the prompt so they cannot reuse a long shared prefix.

The script builds the prefix using the actual Hugging Face tokenizer, saves the **actual prompt token length range**, and checks the server's supported CLI form for disabling prefix caching.

## Metrics

| Output column | Interpretation |
|---|---|
| `ttft_mean_ms`, `ttft_p50_ms`, `ttft_p99_ms` | Actual client-measured first streamed text latency |
| `e2e_mean_ms` | Mean request duration until streaming completes |
| `measured_output_throughput_tps` | Output tokens per second during measured sequential requests, using API-reported completion token counts |
| `prefix_queries_delta` | Increase in prefix-cache query tokens reported by vLLM |
| `prefix_hits_delta` | Increase in prefix-cache hit tokens reported by vLLM |
| `prefix_hit_ratio` | `prefix_hits_delta / prefix_queries_delta` when both are available |
| `kv_usage_after` | KV pool occupancy at the end of that workload |

A **cache-hit counter** measures the engine's observed prefix reuse. The ratio is based on **token queries/hits**, not the percentage of requests with any cache hit. In versions that do not expose a metric, the field remains blank rather than being treated as zero.

TTFT here is measured by the client from issuing an HTTP request until the first nonempty streamed text. It includes network loopback and API overhead, not solely the GPU prefill kernel time.

## Run

From `03-pagedattention-memory-management/`:

```bash
# Smoke test: two prefix patterns, ON/OFF for 512 tokens
python experiments/E5_prefix_caching/run_e5.py \
  --prefix-lengths 512 --requests 4

# Full: 3 prefix lengths x 2 cache states x 2 workload patterns
python experiments/E5_prefix_caching/run_e5.py \
  --prefix-lengths 512 1024 2048 --requests 12 \
  --output-tokens 64 --gpu-memory-utilization 0.80
```

**Expected output** (not pre-measured values):

```text
START prefix=512, cache=off
    unique #1/12: TTFT=...
    ...
  DONE unique: TTFT mean=... hit_ratio=... throughput=...
  DONE shared: TTFT mean=... hit_ratio=... throughput=...
...
SUMMARY: results/E5_prefix_caching/<UTC timestamp>/summary.csv
```

## Analysis plan

1. Compare **cache OFF + shared prefix** against **cache ON + shared prefix** at the same length to estimate the impact of caching.
2. Compare **cache ON + shared** vs **cache ON + unique** to check whether actual reusable content, not the ON flag alone, drives the difference.
3. Check `prefix_hits_delta` and `prefix_hit_ratio` alongside latency. A speed difference without cache hits is not evidence of KV reuse.
4. Assess how the benefit changes as shared prefixes lengthen from 512 to 2048 tokens.
5. Report the actual measured prompt lengths and note that sequential requests evaluate *reuse between requests*, not throughput under high concurrency.

## Output files

```text
results/E5_prefix_caching/<UTC_timestamp>/
├── configuration.json
├── summary.csv
└── prefix_<N>_cache_on_or_off/
    ├── server.log
    ├── unique_requests.csv
    └── shared_requests.csv
```

**Status:** Code implemented; awaiting actual GPU measurement. No results are claimed in advance.

## Reference

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM serving CLI](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM metrics (prefix cache queries/hits)](https://docs.vllm.ai/en/stable/design/metrics/)
