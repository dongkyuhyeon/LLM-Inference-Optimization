#!/usr/bin/env python3
"""E4 — real vLLM Chunked Prefill: mixed short-decode + long-prefill traffic.

Each token-budget condition runs a fresh server, then two simultaneous
vllm bench serve workloads against it: short inputs with long outputs,
and long inputs with short outputs. Results come from actual bench JSON.
Run from the chapter directory; supports --budgets 1024 for a smoke test.

No assumption that two bench processes overlap is made; timestamps and
running/waiting metric samples are saved for verification.
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


def get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def metrics(base):
    series = {}
    pattern = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
    for line in get(base + "/metrics").splitlines():
        match = pattern.match(line)
        if match:
            series.setdefault(match.group(1), []).append(float(match.group(2)))
    names = {
        "running": ("vllm:num_requests_running",),
        "waiting": ("vllm:num_requests_waiting",),
        "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    }
    result = {}
    for key, candidates in names.items():
        name = next((n for n in candidates if n in series), None)
        result[key] = sum(series[name]) if name else None
    return result


def monitor(base, path, stop, period):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_s", "running", "waiting", "kv_usage"])
        writer.writeheader()
        t0 = time.monotonic()
        while not stop.is_set():
            try:
                writer.writerow({"elapsed_s": round(time.monotonic() - t0, 3), **metrics(base)})
                f.flush()
            except (TimeoutError, urllib.error.URLError, OSError):
                pass
            stop.wait(period)


def wait_server(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server failed (exit {process.returncode}); inspect server.log")
        try:
            get(base + "/health", timeout=2)
            get(base + "/v1/models", timeout=3)
            return
        except (TimeoutError, urllib.error.URLError, OSError):
            time.sleep(2)
    raise TimeoutError(f"vLLM server did not become healthy within {timeout}s")


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def bench_command(args, base, name, folder, input_len, output_len, count, rate):
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", base,
        "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "0.0",
        "--num-prompts", str(count),
        "--request-rate", str(rate),
        "--seed", str(args.seed),
        "--ignore-eos",
        "--save-result",
        "--result-dir", str(folder),
        "--result-filename", f"{name}.json",
    ]


def number(data, *keys):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def read_summary(file):
    data = json.loads(file.read_text(encoding="utf-8"))
    return {
        "completed": number(data, "completed", "successful_requests"),
        "output_throughput_tps": number(data, "output_throughput"),
        "mean_ttft_ms": number(data, "mean_ttft_ms"),
        "p99_ttft_ms": number(data, "p99_ttft_ms"),
        "mean_tpot_ms": number(data, "mean_tpot_ms"),
        "p99_tpot_ms": number(data, "p99_tpot_ms"),
        "mean_itl_ms": number(data, "mean_itl_ms"),
        "p99_itl_ms": number(data, "p99_itl_ms"),
    }


def sample_peak(file, key):
    values = []
    with file.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                pass
    return max(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--short-input", type=int, default=128)
    parser.add_argument("--short-output", type=int, default=256)
    parser.add_argument("--short-requests", type=int, default=48)
    parser.add_argument("--short-rate", type=float, default=4.0)
    parser.add_argument("--long-input", type=int, default=4096)
    parser.add_argument("--long-output", type=int, default=128)
    parser.add_argument("--long-requests", type=int, default=12)
    parser.add_argument("--long-rate", type=float, default=2.0)
    parser.add_argument("--long-delay", type=float, default=2.0)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if min(args.budgets) < 1 or args.short_requests < 1 or args.long_requests < 1:
        parser.error("budgets and request counts must be positive")
    if any(x <= 0 for x in (args.short_rate, args.long_rate, args.max_num_seqs, args.max_model_len)):
        parser.error("request rates and capacity settings must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.long_delay < 0:
        parser.error("invalid GPU memory fraction or delay")
    if max(args.short_input + args.short_output, args.long_input + args.long_output) > args.max_model_len:
        parser.error("input + output tokens exceed --max-model-len")
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or root / "results" / "E4_chunked_prefill" / stamp
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    base = f"http://127.0.0.1:{args.port}"
    results = []
    summary_file = output / "summary.csv"

    for budget in args.budgets:
        folder = output / f"budget_{budget}"
        folder.mkdir(parents=True, exist_ok=True)
        server_cmd = [
            "vllm", "serve", args.model,
            "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "float16",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", str(budget),
            "--enable-chunked-prefill",
        ]
        print(f"START budget={budget}", flush=True)
        with (folder / "server.log").open("w", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                server_cmd, stdout=server_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_server(base, server, args.startup_timeout)
                short_cmd = bench_command(
                    args, base, "short", folder,
                    args.short_input, args.short_output,
                    args.short_requests, args.short_rate)
                long_cmd = bench_command(
                    args, base, "long", folder,
                    args.long_input, args.long_output,
                    args.long_requests, args.long_rate)
                stop = threading.Event()
                metric_file = folder / "metrics_timeseries.csv"
                thread = threading.Thread(
                    target=monitor, args=(base, metric_file, stop, args.sample_interval),
                    daemon=True)
                thread.start()
                processes = []
                started = {}
                finished = {}
                try:
                    with (folder / "short.log").open("w", encoding="utf-8") as short_log, \
                         (folder / "long.log").open("w", encoding="utf-8") as long_log:
                        started["short"] = time.time()
                        short_proc = subprocess.Popen(
                            short_cmd, stdout=short_log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        processes.append(short_proc)
                        time.sleep(args.long_delay)
                        started["long"] = time.time()
                        long_proc = subprocess.Popen(
                            long_cmd, stdout=long_log, stderr=subprocess.STDOUT,
                            start_new_session=True)
                        processes.append(long_proc)
                        short_code = short_proc.wait()
                        finished["short"] = time.time()
                        long_code = long_proc.wait()
                        finished["long"] = time.time()
                finally:
                    for process in processes:
                        if process.poll() is None:
                            terminate(process)
                    stop.set()
                    thread.join(timeout=5)
                (folder / "timing.json").write_text(
                    json.dumps({"started_unix": started, "finished_unix": finished}, indent=2),
                    encoding="utf-8")
                if short_code != 0 or long_code != 0:
                    raise RuntimeError(
                        f"Benchmark failed (short={short_code}, long={long_code}). "
                        f"Check {folder}/short.log and long.log")
                short_result = read_summary(folder / "short.json")
                long_result = read_summary(folder / "long.json")
                # The two processes overlap in wall clock, but this does not
                # prove that long prefill ran during an active short decode.
                overlap = max(0.0, min(finished["short"], finished["long"])
                              - max(started["short"], started["long"]))
                row = {
                    "token_budget": budget,
                    "short_requests": args.short_requests,
                    "long_requests": args.long_requests,
                    "short_output_throughput_tps": short_result["output_throughput_tps"],
                    "short_p99_ttft_ms": short_result["p99_ttft_ms"],
                    "short_p99_tpot_ms": short_result["p99_tpot_ms"],
                    "short_p99_itl_ms": short_result["p99_itl_ms"],
                    "long_p99_ttft_ms": long_result["p99_ttft_ms"],
                    "long_output_throughput_tps": long_result["output_throughput_tps"],
                    "running_peak": sample_peak(metric_file, "running"),
                    "waiting_peak": sample_peak(metric_file, "waiting"),
                    "kv_usage_peak": sample_peak(metric_file, "kv_usage"),
                    "bench_process_overlap_s": round(overlap, 3),
                    "short_completed": short_result["completed"],
                    "long_completed": long_result["completed"],
                    "result_path": str(folder),
                }
                results.append(row)
                with summary_file.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0]))
                    writer.writeheader()
                    writer.writerows(results)
                print(
                    f"  DONE short P99 TTFT={row['short_p99_ttft_ms']}ms "
                    f"TPOT={row['short_p99_tpot_ms']}ms "
                    f"ITL={row['short_p99_itl_ms']}ms "
                    f"short throughput={row['short_output_throughput_tps']} tokens/s "
                    f"overlap={overlap:.2f}s", flush=True)
            finally:
                terminate(server)
    print(f"SUMMARY: {summary_file}")
    print("Note: concurrent client processes do not by themselves establish prefill/decode GPU overlap.")


if __name__ == "__main__":
    main()
