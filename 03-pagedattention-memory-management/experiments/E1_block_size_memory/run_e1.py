#!/usr/bin/env python3
"""E1: Paged KV block-size ablation (allocator model + optional CUDA allocation).

This is NOT a vLLM paged-attention kernel or an inference latency benchmark.
Run from the 03-pagedattention-memory-management directory.
"""
import argparse
import csv
import math
import random
from pathlib import Path

def build_workloads(requests, seed):
    rng = random.Random(seed)
    short = [rng.randint(64, 512) for _ in range(requests)]
    long = [rng.randint(2048, 4096) for _ in range(requests)]
    mixed = short[:requests // 2] + long[requests // 2:]
    return {"short": short, "long": long, "mixed": mixed}

def calculate(lengths, block_size, layers, kv_heads, head_dim, bytes_per_elem, reservation):
    assert max(lengths) <= reservation
    blocks = sum(math.ceil(t / block_size) for t in lengths)
    used = sum(lengths)
    capacity = blocks * block_size
    byte_per_token = 2 * layers * kv_heads * head_dim * bytes_per_elem
    return {
        "requests": len(lengths),
        "used_tokens": used,
        "blocks": blocks,
        "allocated_slots": capacity,
        "wasted_slots": capacity - used,
        "fragmentation_pct": 100.0 * (capacity - used) / capacity,
        "payload_mib": capacity * byte_per_token / 2**20,
        "fixed_reserved_mib": len(lengths) * reservation * byte_per_token / 2**20,
        "allocation_saving_pct": 100.0 * (1 - capacity / (len(lengths) * reservation)),
    }

def measure_cuda(blocks, block_size, layers, kv_heads, head_dim):
    import gc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Try --device cpu for theoretical results.")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    # [physical blocks, layer, K/V, KV head, token slot, head dimension]
    kv = torch.zeros((blocks, layers, 2, kv_heads, block_size, head_dim),
                     dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated() - before
    del kv
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return delta / 2**20

def save_chart(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario in ("short", "long", "mixed"):
        group = sorted((r for r in rows if r["workload"] == scenario),
                       key=lambda r: r["block_size"])
        ax.plot([r["block_size"] for r in group],
                [r["fragmentation_pct"] for r in group],
                marker="o", label=scenario)
    ax.set(xlabel="KV block size (tokens)",
           ylabel="Unused slots / allocated slots (%)",
           title="E1 | KV Block Size vs Internal Fragmentation")
    ax.set_xticks(sorted({r["block_size"] for r in rows}))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Figure:", path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--max-reservation", type=int, default=8192)
    args = parser.parse_args()
    if args.requests < 2 or args.requests % 2:
        parser.error("--requests must be even and >= 2")
    if min(args.block_sizes) <= 0 or args.layers <= 0 or args.kv_heads <= 0 or args.head_dim <= 0:
        parser.error("block sizes and model dimensions must be positive")
    if args.max_reservation < 4096:
        parser.error("--max-reservation must be >= 4096 for the built-in long workload")

    workloads = build_workloads(args.requests, args.seed)
    rows = []
    for scenario, lengths in workloads.items():
        for block_size in sorted(set(args.block_sizes)):
            row = calculate(lengths, block_size, args.layers,
                            args.kv_heads, args.head_dim, 2, args.max_reservation)
            row.update(workload=scenario, block_size=block_size,
                       device=args.device, seed=args.seed,
                       layers=args.layers, kv_heads=args.kv_heads, head_dim=args.head_dim,
                       cuda_allocated_delta_mib="")
            if args.device == "cuda":
                row["cuda_allocated_delta_mib"] = measure_cuda(
                    row["blocks"], block_size, args.layers, args.kv_heads, args.head_dim)
            rows.append(row)
            print(f"{scenario:5s} block={block_size:3d} "
                  f"blocks={row['blocks']:5d} waste={row['wasted_slots']:4d} "
                  f"fragmentation={row['fragmentation_pct']:6.2f}% "
                  f"KV={row['payload_mib']:8.2f} MiB "
                  f"CUDA_delta={row['cuda_allocated_delta_mib']}")

    root = Path(__file__).resolve().parents[2]
    result = root / "results" / "E1_block_size_summary.csv"
    figure = root / "figures" / "E1_block_size_fragmentation.png"
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("CSV:", result)
    save_chart(rows, figure)

if __name__ == "__main__":
    main()
