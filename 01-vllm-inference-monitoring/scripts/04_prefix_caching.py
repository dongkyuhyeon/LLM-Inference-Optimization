"""Experiment 4: Unique vs Cold vs Warm prefix reuse (pending GPU validation).

For each of 8 rounds:
  A: unique prompt (cache-miss control)
  B: a new shared document, first request (cold)
  C: same document with a different question (warm)
Keep one vLLM server running throughout A/B/C; never restart between trials.
Output: 04_prefix_per_request.csv, 04_prefix_summary.csv, 04_prefix_paired.csv.
"""
import hashlib
import numpy as np
import pandas as pd
from common import OUT, Server, delta, metrics, request, save_summary, warmup

PARAGRAPH = (
    "The attention mechanism computes query, key, and value vectors. "
    "Prefill processes input tokens and stores key/value tensors. "
    "Decode uses previously cached tensors to generate successive tokens. "
    "Continuous batching improves aggregate throughput by combining requests. "
    "The scheduler controls running and waiting requests. "
    "TTFT measures first output latency while throughput measures tokens per second. "
)

def document(identifier):
    return f"Distinct study record {identifier}. " + PARAGRAPH * 8

def trial(round_id, condition, prompt):
    before = metrics()
    result = request(prompt, output_tokens=64)
    after = metrics()
    hits = delta(before, after, "prefix_hits")
    queries = delta(before, after, "prefix_queries")
    rate = 100 * hits / queries if np.isfinite(hits) and np.isfinite(queries) and queries > 0 else np.nan
    return {"Round":round_id, "Condition":condition, "Success":result["success"],
            "TTFT (ms)":result["ttft_ms"], "Full Latency (s)":result["latency_s"],
            "Cache Hit Tokens":hits, "Cache Query Tokens":queries, "Hit Rate (%)":rate,
            "Error":result["error"]}

def main():
    server = Server()
    server.ensure(max_seqs=8, kv_mib=256, prefix=True, fresh=True)
    warmup()
    rows = []
    for round_id in range(1, 9):
        aid = hashlib.sha256(f"unique-A-{round_id}".encode()).hexdigest()[:20]
        bid = hashlib.sha256(f"unique-B-{round_id}".encode()).hexdigest()[:20]
        shared = document(bid)
        cases = [
            ("A Unique", document(aid) + "\nQuestion: Explain the role of GPU memory."),
            ("B Cold", shared + "\nQuestion: Explain the role of GPU memory."),
            ("C Warm", shared + "\nQuestion: Explain the role of KV memory."),
        ]
        for label, prompt in cases:
            row = trial(round_id, label, prompt)
            rows.append(row)
            if not row["Success"]:
                print("[STOP]", label, row["Error"])
                break
        if not rows[-1]["Success"]:
            break

    requests_df = save_summary(rows, "04_prefix_per_request.csv")
    if requests_df.empty:
        return
    by_condition = requests_df.groupby("Condition", as_index=False, sort=False).agg(
        Requests=("Success", "size"),
        Success=("Success", "sum"),
        Median_TTFT_ms=("TTFT (ms)", "median"),
        Mean_TTFT_ms=("TTFT (ms)", "mean"),
        Hit_Tokens=("Cache Hit Tokens", "sum"),
        Query_Tokens=("Cache Query Tokens", "sum"),
    )
    by_condition["Weighted_Hit_Rate_pct"] = np.where(
        by_condition["Query_Tokens"] > 0,
        100 * by_condition["Hit_Tokens"] / by_condition["Query_Tokens"],
        np.nan,
    )
    save_summary(by_condition.to_dict("records"), "04_prefix_summary.csv")
    pairs = requests_df.pivot_table(index="Round", columns="Condition", values="TTFT (ms)", aggfunc="first")
    if {"B Cold", "C Warm"}.issubset(pairs.columns):
        pairs["Cold minus Warm (ms)"] = pairs["B Cold"] - pairs["C Warm"]
        pairs.to_csv(OUT / "04_prefix_paired.csv")
        print("\n[PAIRED COMPARISON]\n", pairs.to_string())
    if by_condition["Weighted_Hit_Rate_pct"].isna().all():
        print("[NOTE] Prefix hit/query token counters not exposed by this build.")
    print("[NOTE] Prefix caching measurement needs verification on Colab T4.")

if __name__ == "__main__":
    main()
