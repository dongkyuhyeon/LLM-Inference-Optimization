#!/usr/bin/env python3
"""E2: real vLLM serving ablation for KV capacity, batching, and preemption.

Starts one local vLLM server per (gpu_memory_utilization, max_num_seqs),
runs vllm bench serve with synthetic token workloads, samples /metrics, and
writes actual benchmark outputs. No results are fabricated or inferred.
Requires vLLM installed with a supported GPU; use --help for parameters.
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

def fetch(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def parse_metrics(text):
    """Parse aggregate Prometheus gauge/counter samples (sum labelled series)."""
    targets = {
        "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        "waiting": ["vllm:num_requests_waiting"],
        "running": ["vllm:num_requests_running"],
        "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    }
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line)
        if m:
            samples.setdefault(m.group(1), []).append(float(m.group(2)))
    output = {}
    for target, names in targets.items():
        found = next((name for name in names if name in samples), None)
        output[target] = sum(samples[found]) if found else None
    return output

def snapshot(base):
    return parse_metrics(fetch(base + "/metrics"))

def monitor(base, path, stop, interval):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seconds", "kv_usage", "waiting", "running", "preemptions"])
        writer.writeheader()
        start = time.monotonic()
        while not stop.is_set():
            try:
                state = snapshot(base)
                writer.writerow({"seconds": round(time.monotonic()-start, 3), **state})
                f.flush()
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            stop.wait(interval)

def wait_ready(base, process, timeout):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited with code {process.returncode}. See server.log")
        try:
            fetch(base + "/health", timeout=2)
            fetch(base + "/v1/models", timeout=3)
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout}s; see server.log")

def stop_server(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def first(result, *keys):
    for key in keys:
        if isinstance(result.get(key), (int, float)):
            return result[key]
    return None

def stats(values, mode):
    good = [v for v in values if v is not None]
    if not good:
        return None
    return max(good) if mode == "max" else sum(good) / len(good)

def run_one_benchmark(args, base, case_dir, rate):
    tag = str(rate).replace(".", "p")
    run_dir = case_dir / f"rate_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_json = f"bench_rate_{tag}.json"
    command = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", base,
        "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--random-range-ratio", "0.0",
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(rate),
        "--seed", str(args.seed),
        "--ignore-eos",
        "--save-result", "--result-dir", str(run_dir),
        "--result-filename", benchmark_json,
    ]
    stop = threading.Event()
    samples_path = run_dir / "metrics_timeseries.csv"
    thread = threading.Thread(target=monitor, args=(base, samples_path, stop, args.sample_interval), daemon=True)
    before = snapshot(base)
    thread.start()
    try:
        with (run_dir / "benchmark.log").open("w", encoding="utf-8") as log:
            outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        stop.set()
        thread.join(timeout=5)
    after = snapshot(base)
    if outcome.returncode:
        raise RuntimeError(f"Benchmark failed (code {outcome.returncode}). Check {run_dir / 'benchmark.log'}")
    results_path = run_dir / benchmark_json
    if not results_path.exists():
        raise FileNotFoundError(f"Benchmark JSON missing: {results_path}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    with samples_path.open(newline="", encoding="utf-8") as f:
        sampled = list(csv.DictReader(f))
    def field(key):
        out = []
        for row in sampled:
            try:
                out.append(float(row[key]))
            except (TypeError, ValueError):
                pass
        return out
    preempt_delta = None
    if before["preemptions"] is not None and after["preemptions"] is not None:
        preempt_delta = after["preemptions"] - before["preemptions"]
    return {
        "request_rate": rate,
        "completed": first(result, "completed", "successful_requests"),
        "request_throughput_rps": first(result, "request_throughput"),
        "output_throughput_tps": first(result, "output_throughput"),
        "mean_ttft_ms": first(result, "mean_ttft_ms"),
        "p99_ttft_ms": first(result, "p99_ttft_ms"),
        "mean_tpot_ms": first(result, "mean_tpot_ms"),
        "p99_tpot_ms": first(result, "p99_tpot_ms"),
        "kv_usage_mean": stats(field("kv_usage"), "mean"),
        "kv_usage_peak": stats(field("kv_usage"), "max"),
        "waiting_peak": stats(field("waiting"), "max"),
        "running_peak": stats(field("running"), "max"),
        "preemptions_delta": preempt_delta,
        "preemption_metric_available": before["preemptions"] is not None,
        "samples": len(sampled),
        "results_dir": str(run_dir),
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--memory-utilizations", type=float, nargs="+", default=[0.75, 0.85, 0.90])
    p.add_argument("--max-num-seqs", type=int, nargs="+", default=[8, 16])
    p.add_argument("--request-rates", type=float, nargs="+", default=[2, 4])
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--num-prompts", type=int, default=48)
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    if args.input_len + args.output_len > args.max_model_len:
        p.error("input_len + output_len must not exceed max_model_len")
    if min(args.memory_utilizations) <= 0 or max(args.memory_utilizations) >= 1:
        p.error("memory utilizations must be strictly between 0 and 1")
    if min(args.max_num_seqs) < 1 or min(args.request_rates) <= 0 or args.num_prompts < 1:
        p.error("sequence limits, request rates, and number of prompts must be positive")
    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or root / "results" / "E2_kv_capacity_preemption" / timestamp
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(
        json.dumps({**vars(args), "output_dir": str(output)}, indent=2, default=str), encoding="utf-8"
    )
    base = f"http://127.0.0.1:{args.port}"
    all_rows = []
    summary = output / "summary.csv"
    for mem in args.memory_utilizations:
        for max_seqs in args.max_num_seqs:
            case_dir = output / f"mem_{mem:.2f}_seqs_{max_seqs}"
            case_dir.mkdir(parents=True, exist_ok=True)
            server_cmd = [
                "vllm", "serve", args.model,
                "--host", "127.0.0.1", "--port", str(args.port),
                "--dtype", "float16",
                "--gpu-memory-utilization", str(mem),
                "--max-num-seqs", str(max_seqs),
                "--max-model-len", str(args.max_model_len),
                "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            ]
            print(f"START server: memory={mem}, max_num_seqs={max_seqs}", flush=True)
            with (case_dir / "server.log").open("w", encoding="utf-8") as server_log:
                proc = subprocess.Popen(server_cmd, stdout=server_log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    wait_ready(base, proc, args.startup_timeout)
                    for rate in args.request_rates:
                        print(f"  BENCH request_rate={rate} req/s, prompts={args.num_prompts}", flush=True)
                        row = run_one_benchmark(args, base, case_dir, rate)
                        row.update(gpu_memory_utilization=mem, max_num_seqs=max_seqs,
                                   input_tokens=args.input_len, output_tokens=args.output_len,
                                   model=args.model)
                        all_rows.append(row)
                        with summary.open("w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
                            writer.writeheader()
                            writer.writerows(all_rows)
                        print(f"  DONE throughput={row['output_throughput_tps']} tokens/s "
                              f"KV_peak={row['kv_usage_peak']} preemptions={row['preemptions_delta']}",
                              flush=True)
                finally:
                    stop_server(proc)
    print(f"RESULTS: {summary}")
    print("Metric names depend on vLLM version; missing values are left blank, not fabricated.")

if __name__ == "__main__":
    main()
