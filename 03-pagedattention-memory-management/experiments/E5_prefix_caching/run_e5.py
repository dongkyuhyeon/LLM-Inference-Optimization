#!/usr/bin/env python3
"""E5: controlled, real vLLM prefix-cache ON/OFF serving benchmark.

Sequential streaming completions: shared vs unique token prefixes, varied lengths.
Per cache state and prefix length, restart the server. Within each server,
evaluate unique first, then shared (one warm-up request per workload).
Reports actual client TTFT, end-to-end latency and vLLM prefix hit counters.
No results are assumed in advance.
"""
import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def http_get(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return res.read().decode("utf-8", errors="replace")


def counter_snapshot(base):
    try:
        raw = http_get(base + "/metrics", timeout=5)
    except (OSError, TimeoutError):
        return {"queries": None, "hits": None, "kv_usage": None}
    names = {
        "queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total"),
        "hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"),
        "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    }
    samples = {}
    regex = re.compile(r"^([^\\s{]+)(?:\\{[^}]*\\})?\\s+([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)")
    for line in raw.splitlines():
        match = regex.match(line)
        if match:
            samples.setdefault(match.group(1), []).append(float(match.group(2)))
    result = {}
    for key, choices in names.items():
        present = next((name for name in choices if name in samples), None)
        result[key] = sum(samples[present]) if present else None
    return result


def wait_ready(base, process, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited ({process.returncode}); inspect server.log")
        try:
            http_get(base + "/health", timeout=2)
            http_get(base + "/v1/models", timeout=3)
            return
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Server not healthy after {timeout}s")


def stop_process(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values)-1)*fraction
    lower = int(index)
    upper = min(lower+1, len(values)-1)
    weight = index-lower
    return values[lower]*(1-weight) + values[upper]*weight


def cache_flag():
    help_result = subprocess.run(
        ["vllm", "serve", "--help"], capture_output=True, text=True)
    if help_result.returncode != 0:
        raise RuntimeError("vllm serve --help failed. Check vLLM installation.")
    flags = help_result.stdout + help_result.stderr
    if "--no-enable-prefix-caching" in flags:
        return "--no-enable-prefix-caching"
    if "--disable-prefix-caching" in flags:
        return "--disable-prefix-caching"
    raise RuntimeError("No supported prefix-caching OFF flag found in vllm serve --help")


def create_prefix(tokenizer, target, variant):
    # Different first tokens eliminate reusable shared blocks in the UNIQUE control.
    # For SHARED, all requests use exactly the same prefix string.
    header = ("Shared reference manual. " if variant == "shared"
              else f"Unique document number {variant:05d}; marker {variant*101+17}. ")
    body = " The document describes GPU memory, attention, tokens, and scheduling."
    base_ids = tokenizer.encode(header + body * (target // 8 + 30), add_special_tokens=False)
    return tokenizer.decode(base_ids[:target], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def stream_request(base, model, prompt, output_tokens, timeout):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    request = urllib.request.Request(
        base + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    first = None
    tokens = 0
    generated = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            event = json.loads(body)
            if "error" in event:
                raise RuntimeError(f"API error: {event['error']}")
            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                tokens = int(usage["completion_tokens"])
            for choice in event.get("choices", []):
                piece = choice.get("text") or ""
                if piece:
                    generated.append(piece)
                    if first is None:
                        first = time.perf_counter()
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError("No streamed output token observed")
    return {
        "ttft_ms": (first-started)*1000,
        "e2e_ms": (ended-started)*1000,
        "completion_tokens": tokens or None,
        "output_characters": sum(map(len, generated)),
    }


def run_workload(args, tokenizer, base, state, length, kind, folder):
    # Each request has its own suffix; shared request prefix remains identical.
    prompts = []
    for i in range(args.requests + 1):
        prefix = create_prefix(tokenizer, length, "shared" if kind == "shared" else i+1)
        prompts.append(prefix + f"\nQuestion {i:03d}: summarize one useful inference system idea.")
    lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    # The warm-up uses exactly the same shared prefix as measured requests.
    stream_request(base, args.model, prompts[0], args.output_tokens, args.request_timeout)
    snapshot_before = counter_snapshot(base)
    data = []
    started = time.perf_counter()
    for i in range(1, len(prompts)):
        result = stream_request(base, args.model, prompts[i], args.output_tokens, args.request_timeout)
        result.update(request_index=i, prompt_tokens=lens[i])
        data.append(result)
        print(f"    {kind} #{i}/{args.requests}: TTFT={result['ttft_ms']:.1f} ms", flush=True)
    elapsed = time.perf_counter() - started
    snapshot_after = counter_snapshot(base)
    metrics = {}
    for key in ("queries", "hits"):
        a,b = snapshot_before.get(key),snapshot_after.get(key)
        metrics[key+"_delta"] = b-a if a is not None and b is not None else None
    hit_ratio = (metrics["hits_delta"]/metrics["queries_delta"]
                 if metrics["hits_delta"] is not None
                 and metrics["queries_delta"] is not None
                 and metrics["queries_delta"] > 0 else None)
    measured_tokens = [r["completion_tokens"] for r in data]
    return_row = {
        "cache_enabled": state,
        "workload": kind,
        "target_prefix_tokens": length,
        "actual_prompt_tokens_min": min(lens[1:]),
        "actual_prompt_tokens_max": max(lens[1:]),
        "requests": args.requests,
        "ttft_mean_ms": statistics.mean(r["ttft_ms"] for r in data),
        "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in data),
        "ttft_p99_ms": percentile([r["ttft_ms"] for r in data], .99),
        "e2e_mean_ms": statistics.mean(r["e2e_ms"] for r in data),
        "measured_output_throughput_tps": (
            sum(measured_tokens)/elapsed if all(n is not None for n in measured_tokens) else None
        ),
        "prefix_queries_delta": metrics["queries_delta"],
        "prefix_hits_delta": metrics["hits_delta"],
        "prefix_hit_ratio": hit_ratio,
        "measured_wall_s": elapsed,
        "kv_usage_after": snapshot_after["kv_usage"],
    }
    with (folder / f"{kind}_requests.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return return_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.80)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.requests < 2 or any(n < 16 for n in args.prefix_lengths):
        parser.error("requests must be >= 2 and all prefix lengths >= 16")
    if max(args.prefix_lengths)+200+args.output_tokens > args.max_model_len:
        parser.error("max-model-len is too small for the largest prompt plus output")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    off_flag = cache_flag()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or root / "results" / "E5_prefix_caching" / stamp
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    all_rows = []
    summary = output / "summary.csv"
    base = f"http://127.0.0.1:{args.port}"
    for length in args.prefix_lengths:
        for enabled in (False, True):
            state = "on" if enabled else "off"
            folder = output / f"prefix_{length}_cache_{state}"
            folder.mkdir(exist_ok=True)
            server_cmd = [
                "vllm", "serve", args.model,
                "--host", "127.0.0.1", "--port", str(args.port),
                "--dtype", "float16",
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-model-len", str(args.max_model_len),
                "--max-num-seqs", str(args.max_num_seqs),
                "--enable-prefix-caching" if enabled else off_flag,
            ]
            print(f"START prefix={length}, cache={state}", flush=True)
            with (folder / "server.log").open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    wait_ready(base, proc, args.startup_timeout)
                    for kind in ("unique", "shared"):
                        row = run_workload(args, tokenizer, base, state, length, kind, folder)
                        all_rows.append(row)
                        with summary.open("w", newline="", encoding="utf-8") as file:
                            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
                            writer.writeheader()
                            writer.writerows(all_rows)
                        print(f"  DONE {kind}: TTFT mean={row['ttft_mean_ms']:.1f}ms "
                              f"hit_ratio={row['prefix_hit_ratio']} "
                              f"throughput={row['measured_output_throughput_tps']}", flush=True)
                finally:
                    stop_process(proc)
    print(f"SUMMARY: {summary}")


if __name__ == "__main__":
    main()
