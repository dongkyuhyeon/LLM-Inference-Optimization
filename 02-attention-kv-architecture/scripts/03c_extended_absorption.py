"""Experiment 03c: extended MLA content-path decode benchmark.

Three implementations with identical random weights and latents:
  A) reconstruct full K/V on every decode step,
  B) pre-materialize and cache full K/V,
  C) cache shared low-rank latent and use weight absorption for K and V.

No RoPE, positional key cache, model training, or production vLLM.
GPU timings measure the specified single-layer operations, not model throughput.
"""
import csv
import math
import statistics
from pathlib import Path

import torch

SEED = 2026
CONTEXT_LENGTHS = (2048, 4096, 8192, 16384)
HEADS, HEAD_DIM, LATENT_DIM = 8, 64, 128
MODEL_LAYERS_FOR_ESTIMATE = 16
WARMUP, REPEATS = 10, 30
OUT = Path("results/03c_extended_absorption_summary.csv")


def reconstruct_cache(c, wk, wv):
    # c: [T,C]; wk/wv: [H,C,D]; return full K/V: [H,T,D].
    k = torch.einsum("tc,hcd->htd", c, wk)
    v = torch.einsum("tc,hcd->htd", c, wv)
    return k.contiguous(), v.contiguous()


def full_kv_attention(q, k, v):
    scores = torch.einsum("hd,htd->ht", q, k) / math.sqrt(HEAD_DIM)
    a = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,htd->hd", a, v)


def repeated_reconstruction(q, c, wk, wv):
    k, v = reconstruct_cache(c, wk, wv)
    return full_kv_attention(q, k, v)


def absorbed_attention(q, c, wk, wv):
    # Key-side weight absorption.
    q_low = torch.einsum("hd,hcd->hc", q, wk)
    scores = torch.einsum("hc,tc->ht", q_low, c) / math.sqrt(HEAD_DIM)
    a = torch.softmax(scores, dim=-1)
    # Value-side absorption: W_V is applied after the weighted latent sum.
    weighted_c = torch.einsum("ht,tc->hc", a, c)
    return torch.einsum("hc,hcd->hd", weighted_c, wv)


def timed_median_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(REPEATS):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def mib(nbytes):
    return nbytes / (1024 ** 2)


def main():
    assert torch.cuda.is_available(), "Use a GPU runtime."
    device = "cuda"
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print(f"FP32, heads={HEADS}, head_dim={HEAD_DIM}, latent_dim={LATENT_DIM}")
    print(f"GPU events: warmup={WARMUP}, repeats={REPEATS} per method.")
    wk = torch.randn(HEADS, LATENT_DIM, HEAD_DIM, device=device) / math.sqrt(LATENT_DIM)
    wv = torch.randn(HEADS, LATENT_DIM, HEAD_DIM, device=device) / math.sqrt(LATENT_DIM)
    q = torch.randn(HEADS, HEAD_DIM, device=device)
    rows = []

    for i, t in enumerate(CONTEXT_LENGTHS):
        c = torch.randn(t, LATENT_DIM, device=device)
        k, v = reconstruct_cache(c, wk, wv)

        with torch.inference_mode():
            out_a = repeated_reconstruction(q, c, wk, wv)
            out_b = full_kv_attention(q, k, v)
            out_c = absorbed_attention(q, c, wk, wv)
            max_err = max(
                (out_a - out_c).abs().max().item(),
                (out_b - out_c).abs().max().item(),
            )
            normalized_err = max_err / max(1.0, out_a.abs().max().item())
            assert torch.allclose(out_a, out_c, atol=2e-4, rtol=2e-4), (
                f"Numerical validation failed at T={t}, max_error={max_err}"
            )

            implementations = [
                ("A_reconstruct", lambda: repeated_reconstruction(q, c, wk, wv)),
                ("B_full_cache", lambda: full_kv_attention(q, k, v)),
                ("C_absorbed", lambda: absorbed_attention(q, c, wk, wv)),
            ]
            if i % 2 == 1:
                implementations.reverse()
            ms = {name: timed_median_ms(fn) for name, fn in implementations}

        latent_bytes = c.numel() * c.element_size()
        full_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
        row = {
            "context_tokens": t,
            "max_output_abs_error": max_err,
            "normalized_max_output_error": normalized_err,
            "A_reconstruct_median_ms": ms["A_reconstruct"],
            "B_full_cache_median_ms": ms["B_full_cache"],
            "C_absorbed_median_ms": ms["C_absorbed"],
            "A_over_C_latency_ratio": ms["A_reconstruct"] / ms["C_absorbed"],
            "B_over_C_latency_ratio": ms["B_full_cache"] / ms["C_absorbed"],
            "latent_cache_mib_one_layer": mib(latent_bytes),
            "full_kv_cache_mib_one_layer": mib(full_bytes),
            "latent_cache_mib_16layers_estimate": mib(latent_bytes) * MODEL_LAYERS_FOR_ESTIMATE,
            "full_kv_cache_mib_16layers_estimate": mib(full_bytes) * MODEL_LAYERS_FOR_ESTIMATE,
            "cache_payload_reduction_pct": 100 * (1 - latent_bytes / full_bytes),
        }
        rows.append(row)
        print(
            f"T={t:5d} | output_err={max_err:.2e} | "
            f"A={ms['A_reconstruct']:.4f}ms | B={ms['B_full_cache']:.4f}ms | "
            f"C={ms['C_absorbed']:.4f}ms | "
            f"A/C={row['A_over_C_latency_ratio']:.2f}x | "
            f"B/C={row['B_over_C_latency_ratio']:.2f}x | "
            f"latent={mib(latent_bytes):.2f}MiB vs fullKV={mib(full_bytes):.2f}MiB"
        )
        del c, k, v

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved:", OUT)
    print("NOTE: 16-layer sizes are theoretical sums, not measured peak HBM.")
    print("NOTE: Latent vs full-KV is a cache payload comparison only.")
    print("NOTE: Pre-materialization cost is excluded from B's decode timing.")
    print("NOTE: No decoupled RoPE, production kernels or LLM quality measurements.")


if __name__ == "__main__":
    main()
