# PagedAttention & LLM Serving Optimization

Five connected experiments on KV memory management, concurrency, scheduling, and serving performance.

## Experimental roadmap

| Experiment | Focus | Status |
|---|---|---|
| **E1 — Block Size & Memory Efficiency** | KV allocation, token-slot fragmentation, CUDA tensor allocation | **CPU + CUDA allocation runs reported (2026-11-23)** |
| E2 — KV Capacity & Preemption | GPU memory budget, preemptions, waiting requests | Planned |
| E3 — Continuous Batching | Request-rate/concurrency sweeps | Planned |
| E4 — Chunked Prefill | TTFT, ITL/TPOT, token-budget trade-offs | Planned |
| E5 — Prefix Caching | Shared-prefix hit/miss and end-to-end performance | Planned |

## E1: KV Block Size & Memory Efficiency

**Question:** How does the KV block size affect storage fragmentation for short, long, and mixed request lengths?

### Experiment design

- Fixed request set generated with seed `1993`: **24 requests per workload**
- `short`: lengths drawn uniformly from **64–512 tokens**
- `long`: lengths drawn uniformly from **2048–4096 tokens**
- `mixed`: 12 short + 12 long requests
- Block sizes: **8, 16, 32, 64, 128** token slots
- Synthetic KV dimensions: **1 layer, 8 KV heads, head dimension 64, FP16**
- Fixed-reservation baseline: **8192 token slots per request**

All block sizes are evaluated on the **same token lengths** within each workload.

### Workload input lengths and token totals

E1 uses **synthetic request lengths, not real prompts or token IDs**. Each workload consists of 24 requests generated using Python `random.Random(1993)`.

| Workload | Requests | Tokens per request | Total used tokens |
|---|---:|---|---:|
| Short | 24 | Uniform random integer from 64 to 512 | **6,247** |
| Long | 24 | Uniform random integer from 2,048 to 4,096 | **74,999** |
| Mixed | 24 | 12 Short + 12 Long requests | **42,993** |

`mixed` is formed from `short[:12] + long[12:]`, reusing the original randomly sampled lengths. The `Total used tokens` values above are the actual sums from the E1 run and do not include block padding. These lengths are used solely to estimate/allocate KV storage; no model prefill or decode is executed.

### Formulas

```text
blocks(request) = ceil(request_tokens / block_size)
allocated_slots = sum(blocks(request)) * block_size
unused_slots = allocated_slots - sum(request_tokens)
fragmentation (%) = 100 * unused_slots / allocated_slots
KV payload (bytes) = allocated_slots * 2(K,V) * layers
                     * KV heads * head dimension * dtype bytes
```

### Run

From `03-pagedattention-memory-management/`:

```bash
# Step 1: theoretical allocator model (no GPU required)
python experiments/E1_block_size_memory/run_e1.py --device cpu

# Step 2: actual synthetic K/V tensor allocation on NVIDIA GPU
python experiments/E1_block_size_memory/run_e1.py --device cuda
```

Results are saved to:

- `results/E1_block_size_summary.csv`
- `figures/E1_block_size_fragmentation.png`

CSV includes the theoretical KV tensor payload, a fixed-capacity reservation baseline, unused slot fraction, and optional observed CUDA **allocated-memory delta**. The CUDA delta is for PyTorch's synthetic tensor allocation, **not** an end-to-end vLLM experiment.

### E1 results 

Both theoretical CPU calculation and the optional CUDA tensor allocation run were completed by the experimenter. **The table below was transcribed from two console screenshots; the original generated CSV and PNG were not uploaded.** CSV: [E1 block-size results transcribed from screenshots](results/E1_block_size_summary_from_screenshots.csv).

| Workload | Block size | Physical Blocks | Unused slots | Fragmentation | KV payload (MiB) | CUDA allocated delta (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Short | 8 | 790 | 73 | 1.16% | 12.34 | 12.34375 |
| Short | 16 | 401 | 169 | 2.63% | 12.53 | 12.53125 |
| Short | 32 | 207 | 377 | 5.69% | 12.94 | 12.9375 |
| Short | 64 | 111 | 857 | 12.06% | 13.88 | 14 |
| Short | 128 | 60 | 1433 | 18.66% | 15.00 | 16 |
| Long | 8 | 9385 | 81 | 0.11% | 146.64 | 146.640625 |
| Long | 16 | 4698 | 169 | 0.22% | 146.81 | 146.8125 |
| Long | 32 | 2354 | 329 | 0.44% | 147.12 | 147.125 |
| Long | 64 | 1181 | 585 | 0.77% | 147.62 | 148 |
| Long | 128 | 597 | 1417 | 1.85% | 149.25 | 150 |
| Mixed | 8 | 5384 | 79 | 0.18% | 84.12 | 84.125 |
| Mixed | 16 | 2698 | 175 | 0.41% | 84.31 | 84.3125 |
| Mixed | 32 | 1354 | 335 | 0.77% | 84.62 | 84.625 |
| Mixed | 64 | 681 | 591 | 1.36% | 85.12 | 86 |
| Mixed | 128 | 345 | 1167 | 2.64% | 86.25 | 86.25 |

**Observed pattern:** in these fixed request sets, smaller blocks reduced the fraction of unused tail slots. The impact was much more pronounced for short requests (block 8: **1.16%**, block 128: **18.66%**) than long requests (**0.11%** vs **1.85%**). Block count moved in the opposite direction: e.g. mixed workload **5,384 blocks (size 8)** vs **345 blocks (size 128)**. A smaller block may therefore cost additional block management work; no speed conclusion follows from this experiment.

**Important measurement caveat:** for some block sizes the CUDA allocated-memory delta exceeded the exact tensor payload (e.g. short, block 128: **16 MiB observed vs 15 MiB tensor bytes**). This is consistent with backend allocator granularity but was **not independently diagnosed**. Never label the CUDA delta as inference latency, full VRAM footprint, or a paged-attention kernel measurement.


