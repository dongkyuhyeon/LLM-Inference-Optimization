"""Experiment 2: Scheduler concurrency limits for 24 simultaneous requests.

Vary max_num_seqs=8, 16, 24; keep KV cache pool=256 MiB.
Each change intentionally restarts the managed vLLM server.
Outputs: results/02_scheduler_summary.csv and per-condition trace/request CSVs.
"""
from common import Server, benchmark, save_summary, warmup

def main():
    server = Server()
    rows = []
    for limit in (8, 16, 24):
        server.ensure(max_seqs=limit, kv_mib=256, prefix=True, fresh=True)
        warmup()
        row = benchmark(24, f"scheduler_{limit}")
        row["Max Num Seqs"] = limit
        rows.append(row)
        save_summary(rows, "02_scheduler_summary.csv")
        if row["Failed"]:
            print("[STOP] Request failure; do not continue increasing the limit.")
            break

if __name__ == "__main__":
    main()
