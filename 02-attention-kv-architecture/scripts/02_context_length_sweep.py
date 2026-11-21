"""Experiment 02: theoretical vs allocated KV tensor payload on CUDA.

Equivalent settings to the original Colab notebook. Stores summary to CSV.
"""
import csv
import gc
from pathlib import Path

import torch

LAYERS = 16
KV_HEADS = 8
HEAD_DIM = 64
BATCH_SIZE = 1
LENGTHS = (512, 1024, 2048, 4096)
DTYPE = torch.float16
OUT = Path("results/02_context_length_summary.csv")


def main():
    assert torch.cuda.is_available(), "Select a GPU runtime."
    rows = []
    print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)
    print("Layers=16, KV heads=8, head dim=64, batch=1, FP16")
    for t in LENGTHS:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        cache = []
        for _ in range(LAYERS):
            shape = (BATCH_SIZE, KV_HEADS, t, HEAD_DIM)
            cache.append((torch.zeros(shape, dtype=DTYPE, device="cuda"),
                          torch.zeros(shape, dtype=DTYPE, device="cuda")))
        torch.cuda.synchronize()
        allocated_delta = torch.cuda.memory_allocated() - before
        theoretical = 2 * LAYERS * BATCH_SIZE * KV_HEADS * t * HEAD_DIM * 2
        payload = sum(x.numel() * x.element_size() for layer in cache for x in layer)
        assert payload == theoretical
        row = {
            "context_length": t,
            "theoretical_mib": theoretical / 2**20,
            "tensor_payload_mib": payload / 2**20,
            "gpu_allocated_delta_mib": allocated_delta / 2**20,
        }
        print(row)
        rows.append(row)
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
