# PagedAttention & LLM Serving Optimization

Five connected experiments on KV memory management, concurrency, scheduling, and serving performance.

## Experimental roadmap

| Experiment | Focus | Status |
|---|---|---|
| **E1 — Block Size & Memory Efficiency** | KV allocation, token-slot fragmentation, CUDA tensor allocation | **Implementation added; awaiting execution** |
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

### Interpretation rules

1. **Measure before concluding.** No E1 results are claimed in this README yet.
2. Smaller blocks typically reduce maximum last-block waste per request, but increase the number of block-table entries.
3. The fixed-reservation baseline is an intentionally naive allocator for comparison, **not a universal description of other serving engines**.
4. Observed CUDA allocation is not total GPU HBM usage, allocator reserved bytes, or model inference peak memory.
5. This experiment does **not** measure PagedAttention CUDA-kernel runtime, TTFT, TPOT or throughput. Those require follow-up serving benchmarks.
6. The block-size list is an **allocator study**, not a claim that every size is supported by all vLLM engine/hardware versions.
