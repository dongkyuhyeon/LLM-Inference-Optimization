"""Experiment 3: KV cache pool size 256, 128, 96 MiB.

Fixed: max_num_seqs=24, concurrent requests=24, max output=256 tokens.
Results show KV usage and whether waiting/preemptions occur.
Outputs: results/03_kv_capacity_summary.csv and per-condition trace/request CSVs.
"""
import numpy as np
from common import Server, benchmark, save_summary, warmup

def main():
    server = Server()
    rows = []
    for pool in (256, 128, 96):
        server.ensure(max_seqs=24, kv_mib=pool, prefix=True, fresh=True)
        warmup()
        row = benchmark(24, f"kv_{pool}")
        row["KV Pool (MiB)"] = pool
        usage = row["Peak KV (%)"]
        row["Estimated Peak KV (MiB)"] = pool * usage / 100 if np.isfinite(usage) else np.nan
        rows.append(row)
        save_summary(rows, "03_kv_capacity_summary.csv")
        preemptions = row["Preemption Delta"]
        if row["Failed"] or (np.isfinite(preemptions) and preemptions > 0):
            print("[STOP] Failure or preemption observed; stopping memory reduction.")
            break

if __name__ == "__main__":
    main()
