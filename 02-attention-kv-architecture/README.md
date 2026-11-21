# Attention Architecture & KV Cache Experiments

Reproducible experiments on **context-length scaling, MLA weight absorption, and StreamingLLM-style attention sinks**. This chapter extends [01. vLLM Inference & KV Cache Monitoring](../01-vllm-inference-monitoring/) by analyzing *why* KV memory grows, how low-rank attention computations can be rearranged, and how cache-eviction policies affect next-token prediction.

## Experiment setup

| Component | Configuration |
|---|---|
| Environment | Google Colab, PyTorch; retain the previous chapter's vLLM environment where compatible |
| GPU | NVIDIA Tesla T4 (target; verify at runtime) |
| vLLM | 0.34.0 (previous chapter's version; check installation compatibility at runtime) |
| Precision | FP16 for KV storage; FP32 for MLA equivalence checks |
| Model | Synthetic tensors for experiments 2–3; a compatible pretrained causal LM for experiment 4 |
| Metrics | Theoretical/payload KV bytes, CUDA allocation, numerical error, synchronized GPU latency, perplexity |
| Outputs | Reproducible scripts, CSV results, plots (after execution) |

> **Status:** Planned, not yet run. No results or performance improvements are claimed. Experiments 2–3 test mechanisms rather than pretrained-model quality or end-to-end vLLM speed. Experiment 4 needs careful model-specific validation.

## 1. Context-length scaling

**Goal:** Verify how KV storage increases with the number of retained tokens.

**Planned lengths:** 512 / 1,024 / 2,048 / 4,096 tokens (batch size 1). Use fixed layer count, KV head count, head dimension, and dtype across lengths. The chosen architecture and full settings will be recorded with the measured results.

For one sequence, ignoring block allocation and metadata:

```text
KV_bytes = 2 (K and V) * num_layers * context_tokens
           * num_kv_heads * head_dim * bytes_per_element
```

**Measurements:** Theoretical bytes, materialized PyTorch tensor payload bytes, and CUDA allocated-memory delta. Distinguish memory payload from allocator overhead, reserved memory, and full GPU usage.

**Expected trend (theory, not measurement):** KV payload grows linearly with retained token count if all other dimensions remain fixed.

**Status:** Not run.

## 2. MLA weight absorption

**Goal:** Confirm the algebraic equivalence of explicit key reconstruction and query-side weight absorption for a simplified MLA *content* path.

```text
Explicit reconstruction: score = q^T (W_up_K c)
Query-side absorption:   score = ((W_up_K)^T q)^T c
```

**Measurements:**
- Maximum absolute/relative score error using FP32
- Repeated, synchronized GPU timing versus cached sequence length (with warmup)
- Latent memory versus materialized high-dimensional K/V storage (theoretical and tensor payload)

**Important limitations:** This is a content-path demonstration, not a full DeepSeek MLA implementation. DeepSeek MLA also uses decoupled RoPE, a separate positional key cache, and value-side absorption. Numerical equivalence is not evidence of end-to-end throughput improvement.

**Status:** Not run.

## 3. Sliding window vs attention sinks

**Goal:** Compare plain sliding-window eviction with a fixed initial-sink-plus-recent-window policy on a compatible pretrained causal language model.

**Planned comparison:** Equal total retained KV-token budget for both policies; vary window length and document length while controlling tokenization and model configuration.

**Measurements:** Perplexity (or mean next-token negative log-likelihood), retained KV-token count, and configuration-specific runtime details.

**Validation requirements:** Confirm correct attention masks, position handling (especially RoPE), and implementation compatibility before interpreting the quality comparison. An observed attention sink is model-dependent; evicted historical facts cannot be retrieved from the KV cache.

**Status:** Not run.

## Key questions

1. Does observed KV tensor payload grow linearly with retained context length?
2. Does query-side weight absorption preserve attention scores and reduce explicit reconstruction work?
3. At a fixed KV budget, when does retaining initial tokens improve next-token prediction compared with a plain sliding window?

## Measurement notes and limitations

- Keep **theoretical predictions** separate from **actual measured results**; never fabricate missing conditions.
- Record the exact Colab GPU, Python/PyTorch/vLLM versions, model revision, and random seed as applicable.
- Use GPU warmup, synchronization, and repeated trials for latency; report distribution/variability when possible.
- KV payload bytes do not equal total GPU memory or vLLM production throughput.
- For experiment 4, compare quality under an equivalent cache budget and document any deviations from standard attention semantics.
- Results from synthetic computations do not establish language-model accuracy or generalize to arbitrary serving engines.

## Conclusion

**Pending experiments.** The chapter will report the measured effect of context length on KV memory, MLA content-path algebraic and runtime checks, and the quality/memory trade-off between sliding-window and sink-preserving cache eviction.


