# E2 — Continuous Batching & Request Scheduling

Formerly planned as E3; renumbered to E2 after dropping the KV-capacity/preemption experiment.

## Research question

How do the request arrival rate and the maximum concurrently scheduled sequences affect **output throughput, queueing, TTFT, and TPOT** in a real vLLM serving workload?

## Proposed comparison

| Variable | Values |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct (FP16) |
| `max_num_seqs` | 4, 8, 16, 32 |
| Request arrival rate | 1, 2, 4, 8 requests/s |
| Request lengths | Fixed synthetic input/output lengths, held identical across settings |
| GPU memory budget | Fixed across settings |

## Metrics

- Completed request throughput (requests/s) and output throughput (tokens/s)
- Running and waiting requests (sampled from the engine)
- P50/P99 TTFT and TPOT
- KV cache occupancy as an auxiliary diagnostic

**Status:** planned. No E2 Continuous Batching benchmark code or results have been added yet.

This is a study of dynamic scheduling under changing loads, **not** a direct comparison with a separately implemented static-batching baseline.
