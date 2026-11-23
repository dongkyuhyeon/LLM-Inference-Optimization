#!/usr/bin/env python3
"""E3: measure continuous batching under request-rate and max-num-seqs sweeps.

Runs a real local vLLM server (one restart per sequence limit), invokes
'vllm bench serve' with fixed-length synthetic inputs and records /metrics.
Requires a working CUDA vLLM installation. No static-batching baseline.
"""
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

def http_get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def read_prometheus(base):
    text = http_get(base + "/metrics")
    parsed = {}
    regex = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    for line in text.splitlines():
        match = regex.match(line)
        if match:
            parsed.setdefault(match.group(1), []).append(float(match.group(2)))
    metrics = {
        "running": ("vllm:num_requests_running",),
        "waiting": ("vllm:num_requests_waiting",),
        "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        "preemptions": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    }
    result = {}
    for key, options in metrics.items():
        name = next((name for name in options if name in parsed), None)
        # For a single-engine/single-GPU experiment, these gauges have one series.
        result[key] = sum(parsed[name]) if name else None
    return result

def monitor(base, filename, stop, interval):
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_s", "running", "waiting", "kv_usage", "preemptions"])
        writer.writeheader()
        start = time.monotonic()
        while not stop.is_set():
            try:
                writer.writerow({"elapsed_s": round(time.monotonic()-start, 3), **read_prometheus(base)})
                f.flush()
            except (OSError, TimeoutError, urllib.error.URLError):
                pass
            stop.wait(interval)

def wait_for_server(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}; read server.log")
        try:
            http_get(base + "/health", timeout=2)
            http_get(base + "/v1/models", timeout=3)
            return
        except (OSError, TimeoutError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError("Server startup timeout; read server.log")

def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def numeric(result, key):
    value = result.get(key)
    return value if isinstance(value, (int, float)) else None

def summarize_samples(path):
    with path.open(newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    out = {"metric_samples": len(samples)}
    for name in ("running", "waiting", "kv_usage"):
        values = []
        for sample in samples:
            try:
                values.append(float(sample[name]))
            except (ValueError, TypeError):
                pass
        out[name + "_peak"] = max(values) if values else None
        out[name + "_mean"] = sum(values) / len(values) if values else None
    return out

def bench(base, args, case, request_rate):
    folder = case / ("rate_" + str(request_rate).replace(".", "p"))
    folder.mkdir(parents=True, exist_ok=True)
    json_file = folder / "benchmark.json"
    metrics_file = folder / "metrics_timeseries.csv"
    command = [
        "vllm", "bench", "serve", "--backend", "vllm",
        "--base-url", base, "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--random-range-ratio", "0.0",
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(request_rate),
        "--seed", str(args.seed),
        "--ignore-eos",
        "--save-result", "--result-dir", str(folder),
        "--result-filename", json_file.name,
    ]
    before = read_prometheus(base)
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(base, metrics_file, stop, args.sample_interval), daemon=True)
    thread.start()
    try:
        with (folder / "benchmark.log").open("w", encoding="utf-8") as log:
            exit_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        stop.set()
        thread.join(timeout=5)
    after = read_prometheus(base)
    if exit_code != 0:
        raise RuntimeError(f"Benchmark failed (exit {exit_code}): {folder / 'benchmark.log'}")
    if not json_file.exists():
        raise RuntimeError(f"Missing benchmark output: {json_file}")
    result = json.loads(json_file.read_text(encoding="utf-8"))
    row = {
        "max_num_seqs": args.current_seqs,
        "request_rate_rps": request_rate,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "completed": numeric(result, "completed"),
        "request_throughput_rps": numeric(result, "request_throughput"),
        "output_throughput_tps": numeric(result, "output_throughput"),
        "mean_ttft_ms": numeric(result, "mean_ttft_ms"),
        "p50_ttft_ms": numeric(result, "median_ttft_ms"),
        "p99_ttft_ms": numeric(result, "p99_ttft_ms"),
        "mean_tpot_ms": numeric(result, "mean_tpot_ms"),
        "p50_tpot_ms": numeric(result, "median_tpot_ms"),
        "p99_tpot_ms": numeric(result, "p99_tpot_ms"),
        "preemption_delta": (after["preemptions"]-before["preemptions"]
                             if before["preemptions"] is not None and after["preemptions"] is not None
                             else None),
        **summarize_samples(metrics_file),
        "raw_dir": str(folder),
    }
    return row

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-num-seqs", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--request-rates", nargs="+", type=float, default=[1, 2, 4, 8])
    p.add_argument("--num-prompts", type=int, default=48)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int, default=300)
    p.add_argument("--sample-interval", type=float, default=0.25)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    if any(n <= 0 for n in args.max_num_seqs) or any(r <= 0 for r in args.request_rates):
        p.error("sequence counts and request rates must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        p.error("input_len + output_len must not exceed max_model_len")
    if args.num_prompts < 1 or not 0 < args.gpu_memory_utilization < 1:
        p.error("invalid num_prompts or memory fraction")
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or root / "results" / "E3_continuous_batching" / stamp
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    base = f"http://127.0.0.1:{args.port}"
    summary_file = output / "summary.csv"
    results = []
    for seqs in args.max_num_seqs:
        args.current_seqs = seqs
        case = output / f"seqs_{seqs}"
        case.mkdir(exist_ok=True)
        server_cmd = [
            "vllm", "serve", args.model,
            "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-seqs", str(seqs),
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        ]
        print(f"START max_num_seqs={seqs}", flush=True)
        with (case / "server.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_for_server(base, proc, args.startup_timeout)
                for rate in args.request_rates:
                    print(f"  BENCH rate={rate} requests/s", flush=True)
                    row = bench(base, args, case, rate)
                    results.append(row)
                    with summary_file.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(results[0]))
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"  DONE tokens/s={row['output_throughput_tps']} "
                          f"waiting_peak={row['waiting_peak']} TTFT_P99={row['p99_ttft_ms']}", flush=True)
            finally:
                terminate(proc)
    print(f"SUMMARY: {summary_file}")
    print("Values are actual benchmark measurements. Missing metrics remain blank.")

if __name__ == "__main__":
    main()
