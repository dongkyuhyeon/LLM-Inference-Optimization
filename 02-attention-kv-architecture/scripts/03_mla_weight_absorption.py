"""Experiment 03: MLA content-key weight absorption (PyTorch CUDA).

Synthetic tensors only; does not reproduce full DeepSeek MLA or vLLM inference.
"""
import csv
import math
import statistics
from pathlib import Path
import torch

LENGTHS = [512, 1024, 2048, 4096]
D, LATENT = 512, 64
WARMUP, REPEATS = 10, 40
OUT = Path("results/03_mla_weight_absorption_summary.csv")

def explicit(c, w, q):
    k = c @ w.T             # up-project every cached key: [T, D]
    return (k @ q) / math.sqrt(D)

def absorbed(c, w, q):
    q_latent = w.T @ q      # transform current query once: [latent]
    return (c @ q_latent) / math.sqrt(D)

def gpu_benchmark(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)

def main():
    assert torch.cuda.is_available(), "Select a CUDA GPU runtime in Colab."
    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print(f"FP32; D={D}, latent={LATENT}, warmup={WARMUP}, repeats={REPEATS}")
    w = torch.randn(D, LATENT, device=device, dtype=torch.float32) / math.sqrt(LATENT)
    q = torch.randn(D, device=device, dtype=torch.float32)
    rows = []

    for i, t in enumerate(LENGTHS):
        c = torch.randn(t, LATENT, device=device, dtype=torch.float32)
        a, b = explicit(c, w, q), absorbed(c, w, q)
        diff = (a - b).abs()
        err = diff.max().item()
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), (t, err)

        funcs = [("explicit", lambda: explicit(c, w, q)),
                 ("absorbed", lambda: absorbed(c, w, q))]
        if i % 2:
            funcs.reverse()
        timing = {name: gpu_benchmark(fn) for name, fn in funcs}

        row = {
            "context_tokens": t,
            "model_dim": D,
            "latent_dim": LATENT,
            "max_abs_error": err,
            "allclose_atol1e-3_rtol1e-3": True,
            "explicit_median_ms": timing["explicit"],
            "absorbed_median_ms": timing["absorbed"],
            "ratio_explicit_over_absorbed": timing["explicit"] / timing["absorbed"],
            "latent_mib": c.numel() * c.element_size() / 2**20,
            "materialized_key_mib": t * D * c.element_size() / 2**20,
        }
        rows.append(row)
        print(f"T={t:4d} | max_abs_err={err:.3e} | "
              f"explicit={timing['explicit']:.4f}ms | "
              f"absorbed={timing['absorbed']:.4f}ms | "
              f"ratio={row['ratio_explicit_over_absorbed']:.2f}x | "
              f"latent={row['latent_mib']:.3f}MiB | "
              f"reconstructed K={row['materialized_key_mib']:.3f}MiB")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", OUT)
    print("Limitations: Content-key-only toy experiment; excludes RoPE and value path.")
    print("Measured op times are not full model throughput or production speedups.")

if __name__ == "__main__":
    main()
