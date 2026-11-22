# PagedAttention: Block Allocation & Memory Fragmentation

A workspace for controlled experiments on **KV-cache block allocation**, **memory fragmentation**, and **block-size trade-offs** in LLM inference.

## Scope

The first planned experiment compares fixed-capacity KV reservation against demand-driven block allocation for requests of different sequence lengths.

- **Block size ablation:** 8, 16, 32, and 64 tokens per block
- **Measurements:** allocated blocks, token-slot capacity, unused slots, and internal fragmentation
- **Follow-up:** validate GPU memory behavior and runtime effects separately, after the allocation model is verified

## Experiment status

**Not started.** No performance results are claimed yet.

## Planned layout

```text
03-pagedattention-memory-management/
├── README.md
├── scripts/    # Allocation simulations and reproducible measurements
├── results/    # Experiment output tables
└── figures/    # Visualizations
```

Git tracks files rather than empty directories; the remaining directories will be added as experiments are implemented.

## Notes

- Theoretical token-slot allocation is not the same as measured GPU HBM consumption.
- Block tables and physical KV layouts may vary by serving engine and version.
- Compare methods at identical request lengths and explicitly state any allocator or implementation assumptions.
