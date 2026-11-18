"""Experiment 1: Continuous batching with 1, 8 and 24 concurrent requests.

Fixed: max_num_seqs=8, KV cache pool=256 MiB, max output=256 tokens.
Outputs: results/01_batching_summary.csv and per-condition trace/request CSVs.
"""
from common import Server, benchmark, save_summary, warmup

def main():
    server = Server()
    server.ensure(max_seqs=8, kv_mib=256, prefix=True)
    warmup()
    rows = []
    for concurrency in (1, 8, 24):
        row = benchmark(concurrency, f"batch_{concurrency}")
        rows.append(row)
        save_summary(rows, "01_batching_summary.csv")
        if row["Failed"]:
            print("[STOP] Request failure; stopping further load.")
            break

if __name__ == "__main__":
    main()
