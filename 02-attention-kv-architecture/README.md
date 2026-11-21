# Attention Architecture & KV Cache Experiments

Reproducible, small-scale experiments on **MHA, GQA, MQA, context-length scaling, and MLA weight absorption** to understand how attention architecture affects KV cache memory and decode computation.

This chapter follows [01. vLLM Inference & KV Cache Monitoring](../01-vllm-inference-monitoring/) and focuses on **architectural mechanisms**, rather than serving-engine scheduling.

## Experiment setup

| Component | Configuration |
|---|---|
| Environment | Google Colab / PyTorch (planned) |
| GPU | NVIDIA Tesla T4 (target; verify at runtime) |
| Precision | FP16 for KV memory comparisons; FP32 for numerical-equivalence checks |
| Model | Synthetic attention tensors; no pretrained LLM required for experiments 1–3 |
| Baseline | Multi-Head Attention (MHA) |
| Variants | Grouped-Query Attention (GQA), Multi-Query Attention (MQA), simplified MLA |
| Metrics | KV tensor size, theoretical KV footprint, numerical error, latency |
| Outputs | CSV summaries and comparative figures (to be generated after execution) |

> **Status:** Experiment plan only. No benchmark has been executed for this chapter yet. Results, plots, and performance claims will be added only after measurements. Synthetic attention benchmarks do **not** establish pretrained-model accuracy or end-to-end vLLM serving speed.

## 1. MHA vs GQA vs MQA

**Goal:** Verify how changing the number of KV heads affects actual KV tensor memory while keeping the number of query heads fixed.

**Planned configuration:** 8 query heads, 8 / 4 / 2 / 1 KV heads, 16 layers, head dimension 64, FP16.

| Variant | Query heads | KV heads | Expected KV footprint vs MHA |
|---|---:|---:|---:|
| MHA | 8 | 8 | 100% |
| GQA | 8 | 4 | 50% |
| GQA | 8 | 2 | 25% |
| MQA | 8 | 1 | 12.5% |

**Validation:** Compare analytical memory estimates with tensor sizes and verify that attention output shapes remain consistent. **This does not test language-model quality** after changing attention architecture.

**Status:** Not run.

## 2. Context-length scaling

**Goal:** Measure how KV cache requirements grow as the stored context length increases.

**Planned configuration:** 512 / 1,024 / 2,048 / 4,096 tokens, using the same MHA/GQA/MQA configurations.

Theoretical KV bytes for one sequence:

```text
KV_bytes = 2 (K and V) * num_layers * context_tokens
           * num_kv_heads * head_dim * bytes_per_element
```

**Validation:** Compare theoretical bytes, actual allocated tensor payload bytes, and the scaling trend. Distinguish tensor payload size from CUDA allocator overhead and peak GPU usage.

**Status:** Not run.

## 3. MLA weight absorption

**Goal:** Verify that a low-rank KV latent can produce the same attention scores without explicitly reconstructing high-dimensional keys for every cached token.

Compare:

```text
Explicit reconstruction: score = q^T (W_up_K c)
Query-side absorption:   score = ((W_up_K)^T q)^T c
```

**Planned measurements:**
- Maximum absolute / relative score difference (FP32 numerical check)
- Latency versus cached sequence length, with GPU warmup and synchronization
- Latent storage versus explicitly materialized K/V tensor sizes

**Scope:** Simplified content-only MLA path. The full DeepSeek MLA implementation also includes decoupled RoPE and other architectural details. The algebraic equivalence does not prove DeepSeek-scale speedups.

**Status:** Not run.

## 4. Attention sinks (optional extension)

**Goal:** Compare a plain sliding-window KV eviction policy against a sink-plus-recent-window policy on a compatible pretrained causal language model.

**Measurements:** Perplexity and retained KV token count across long streams.

**Caution:** The presence and usefulness of attention sinks are model-dependent. Synthetic tensors alone cannot establish quality preservation, and evicted old content is not recoverable through the window.

**Status:** Optional; not run.

## Key questions

1. How does KV head count affect per-token memory while query head count remains fixed?
2. Does measured KV tensor storage scale linearly with context length?
3. Does query-side weight absorption preserve attention scores while avoiding high-dimensional key reconstruction?
4. When does retaining initial attention-sink tokens help stabilize sliding-window inference?

## Measurement notes and limitations

- Report **measured** values separately from **theoretical** estimates; do not mix runs or fabricate missing conditions.
- GPU kernel timings require warmup, explicit synchronization, repetition, and reporting of the measurement method.
- KV tensor payload size is not the same as full GPU memory usage or end-to-end serving cost.
- Changing KV head count without appropriate model training does not demonstrate retained model quality.
- Comparisons are for controlled mechanisms; full-model TTFT, ITL, and throughput require separate serving benchmarks.

## Conclusion

**Pending experiments.** This chapter will document measured KV memory scaling, the MHA/GQA/MQA footprint trade-off, and the numerical and computational implications of MLA weight absorption after the scripts have been executed.

