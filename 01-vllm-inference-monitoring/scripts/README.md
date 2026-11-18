# vLLM inference benchmark scripts

Four experiment entrypoints use the same `common.py` server manager and metric sampler.

| File | Experiment | Settings |
|---|---|---|
| `01_continuous_batching.py` | Continuous batching | 1, 8, 24 concurrent requests; `max_num_seqs=8` |
| `02_scheduler_limits.py` | Scheduler limit | 24 concurrent requests; `max_num_seqs=8,16,24` |
| `03_kv_cache_capacity.py` | KV cache memory pool | 24 requests; 256, 128, 96 MiB |
| `04_prefix_caching.py` | Unique/Cold/Warm prefix | A/B/C repeated rounds, **results not verified yet** |

## Requirements
Run in Google Colab T4 GPU or a compatible NVIDIA GPU environment. Use Python with `vllm`, `requests`, `numpy`, and `pandas` installed. The original notebook used **vLLM 0.34.0**, **Qwen/Qwen2.5-0.5B-Instruct**, FP16 and a maximum sequence length of 1,024 tokens.

Install missing packages (prefer a fresh GPU environment):

```bash
python -m pip install vllm requests numpy pandas
```

## Usage
Execute from this folder; run scripts **one at a time**:

```bash
python 01_continuous_batching.py
python 02_scheduler_limits.py
python 03_kv_cache_capacity.py
python 04_prefix_caching.py
```

The server is started as a subprocess; it is intentionally restarted when a benchmark changes settings. The state file `/content/llm_kv_project/managed_server.json` tracks ownership. The code refuses to stop an unknown server on port 8000.

Reports are saved to `/content/llm_kv_project/results/` on Colab, otherwise `./llm_kv_project/results/`. Per-request and time-series CSVs are also generated. Request-measured TTFT includes client/API overhead, and "Mean Chunk Gap" measures streaming content chunk intervals rather than guaranteed per-token ITL.

**Important:** These scripts were extracted/refactored from the Colab experiments; package-specific runtime execution on a real T4 GPU has not yet been revalidated. Historical numerical results must not be mistaken for newly run results. Prefix caching remains a proposed verification experiment.
