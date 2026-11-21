"""MHA/GQA/MQA KV tensor memory comparison and one-token decode smoke test.

Synthetic PyTorch experiment. Not a pretrained-model quality or vLLM throughput benchmark.
"""
import csv
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

LAYERS = 16
QUERY_HEADS = 8
HEAD_DIM = 64
CONTEXT_TOKENS = 4096
DTYPE = torch.float16
OPTIONS = (8, 4, 2, 1)
OUT = Path("results/01_attention_variants_summary.csv")


def verify_decode(device, num_kv_heads):
    """Verify that attention output keeps all 8 query heads."""
    torch.manual_seed(2026)
    q = torch.randn(1, QUERY_HEADS, 1, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(1, num_kv_heads, 128, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn(1, num_kv_heads, 128, HEAD_DIM, device=device, dtype=DTYPE)
    factor = QUERY_HEADS // num_kv_heads
    # Educational broadcast for shape verification, NOT optimized GQA inference.
    k = k.repeat_interleave(factor, dim=1)
    v = v.repeat_interleave(factor, dim=1)
    scores = q.float() @ k.float().transpose(-2, -1) / (HEAD_DIM ** 0.5)
    output = torch.softmax(scores, dim=-1) @ v.float()
    assert output.shape == (1, QUERY_HEADS, 1, HEAD_DIM)
    assert torch.isfinite(output).all().item()
    return tuple(output.shape)


def measure(device, num_kv_heads):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated(device)
    else:
        before = None

    cache = []
    for _ in range(LAYERS):
        shape = (1, num_kv_heads, CONTEXT_TOKENS, HEAD_DIM)
        cache.append((torch.empty(shape, dtype=DTYPE, device=device),
                      torch.empty(shape, dtype=DTYPE, device=device)))

    payload = sum(t.numel() * t.element_size() for layer in cache for t in layer)
    theoretical = 2 * LAYERS * CONTEXT_TOKENS * num_kv_heads * HEAD_DIM * 2
    assert payload == theoretical
    if before is not None:
        torch.cuda.synchronize()
        allocated_delta = torch.cuda.memory_allocated(device) - before
    else:
        allocated_delta = None
    del cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return theoretical, payload, allocated_delta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__, "| device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Layers=16, Q heads=8, dim=64, tokens=4096, FP16, batch=1")
    rows = []
    baseline = 2 * LAYERS * CONTEXT_TOKENS * QUERY_HEADS * HEAD_DIM * 2
    for n in OPTIONS:
        variant = "MHA" if n == 8 else ("MQA" if n == 1 else "GQA")
        expected, payload, actual = measure(device, n)
        shape = verify_decode(device, n)
        row = {
            "variant": variant, "query_heads": QUERY_HEADS, "kv_heads": n,
            "context_tokens": CONTEXT_TOKENS, "layers": LAYERS,
            "head_dim": HEAD_DIM,
            "theoretical_mib": round(expected / 2**20, 4),
            "tensor_payload_mib": round(payload / 2**20, 4),
            "gpu_allocated_delta_mib": "" if actual is None else round(actual / 2**20, 4),
            "reduction_vs_mha_percent": round(100 * (1 - expected / baseline), 2),
            "decode_output_shape": str(shape)
        }
        rows.append(row)
        print(f"{variant:3} KV heads={n}: expected={row['theoretical_mib']} MiB, "
              f"payload={row['tensor_payload_mib']} MiB, "
              f"GPU allocated delta={row['gpu_allocated_delta_mib']} MiB, "
              f"reduction={row['reduction_vs_mha_percent']}%, decode={shape}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", OUT)
    print("NOTE: tensor payload is not peak GPU HBM or vLLM throughput.")


if __name__ == "__main__":
    main()
