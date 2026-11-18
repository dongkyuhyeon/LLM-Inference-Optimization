"""Shared vLLM server and measurement helpers for Colab/Linux with NVIDIA GPU.

Install: pip install vllm requests numpy pandas matplotlib
Python:  python 01_continuous_batching.py

The process manager only terminates servers it started and recorded.
"""
from __future__ import annotations
import json, os, re, signal, socket, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pandas as pd
import requests

ROOT = Path(os.getenv("VLLM_EXPERIMENT_ROOT", "/content/llm_kv_project" if Path("/content").exists() else "./llm_kv_project"))
OUT = ROOT / "results"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)(?:\s|$)")
ALIASES = {
    "kv_usage": ["vllm:kv_cache_usage_perc"],
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "gen_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
}
PROMPT = ("Explain GPU memory bandwidth, KV cache, continuous batching, prefill, "
          "decode and throughput. Describe their effects on LLM inference latency. ") * 5

class Server:
    """Own only the process that this project starts; detect stale state safely."""
    def __init__(self):
        self.state_path = ROOT / "managed_server.json"
        self.proc = None

    def state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def port_open():
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                return True
        except OSError:
            return False

    @staticmethod
    def ready():
        try:
            r = requests.get(BASE_URL + "/v1/models", timeout=3)
            return r.ok and any(x.get("id") == MODEL for x in r.json().get("data", []))
        except (requests.RequestException, ValueError):
            return False

    @staticmethod
    def owned(pid):
        try:
            pid = int(pid)
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            return ("vllm.entrypoints.openai.api_server" in cmd and MODEL in cmd
                    and os.getpgid(pid) == pid)
        except (OSError, ValueError):
            return False

    def stop(self):
        old = self.state()
        if not old:
            if self.port_open():
                raise RuntimeError("Port 8000 is occupied by an unmanaged process.")
            return
        pid = int(old["pid"])
        if self.owned(pid):
            os.killpg(pid, signal.SIGTERM)
            for _ in range(60):
                if not self.port_open():
                    break
                time.sleep(.5)
            if self.owned(pid) and self.port_open():
                os.killpg(pid, signal.SIGKILL)
        elif self.port_open():
            raise RuntimeError("Port 8000 occupied; recorded PID is not owned.")
        self.state_path.unlink(missing_ok=True)
        for _ in range(30):
            if not self.port_open():
                return
            time.sleep(.5)
        raise RuntimeError("Port 8000 did not become available.")

    def ensure(self, max_seqs=24, kv_mib=256, prefix=False, fresh=False):
        cfg = {"max_seqs": max_seqs, "kv_mib": kv_mib, "prefix": prefix}
        old = self.state()
        if self.ready():
            if not fresh and old and old.get("config") == cfg and self.owned(old["pid"]):
                print("[SERVER] Reusing existing server", cfg)
                return
            if old and self.owned(old["pid"]):
                self.stop()
            else:
                raise RuntimeError("Unknown existing server; refusing to terminate it.")
        elif old and self.owned(old["pid"]):
            self.stop()
        elif self.port_open():
            raise RuntimeError("Port 8000 is occupied. Inspect it before starting.")
        log = LOGS / f"vllm_seq{max_seqs}_kv{kv_mib}_prefix{int(prefix)}_{int(time.time())}.log"
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", MODEL, "--host", "127.0.0.1", "--port", "8000",
               "--dtype", "half", "--max-model-len", "1024",
               "--max-num-seqs", str(max_seqs),
               "--kv-cache-memory-bytes", f"{kv_mib}M", "--enforce-eager"]
        if prefix:
            cmd.append("--enable-prefix-caching")
        with log.open("w") as handle:
            self.proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.state_path.write_text(json.dumps({"pid": self.proc.pid, "config": cfg, "log": str(log)}))
        for _ in range(120):
            if self.proc.poll() is not None:
                break
            if self.ready():
                print("[SERVER] READY", cfg)
                return
            time.sleep(2)
        tail = log.read_text(errors="replace")[-4000:]
        if self.owned(self.proc.pid):
            self.stop()
        raise RuntimeError("vLLM startup failed. Log tail:\n" + tail)

def metrics():
    r = requests.get(BASE_URL + "/metrics", timeout=5)
    r.raise_for_status()
    raw = {}
    for line in r.text.splitlines():
        match = SAMPLE_RE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if labels and 'model_name="' in labels and f'model_name="{MODEL}"' not in labels:
            continue
        raw.setdefault(name, []).append(float(value))
    result = {}
    for key, names in ALIASES.items():
        matches = next((raw[n] for n in names if n in raw), None)
        if matches is None:
            result[key] = np.nan
        elif len(matches) != 1:
            raise RuntimeError(f"Multiple metric series for {key}: {len(matches)}")
        else:
            result[key] = matches[0]
    return result

def delta(before, after, key):
    x, y = before.get(key, np.nan), after.get(key, np.nan)
    return max(0., y - x) if np.isfinite(x) and np.isfinite(y) else np.nan

def request(prompt, gate=None, output_tokens=256):
    if gate is not None:
        gate.wait()
    begin = time.perf_counter()
    chunks, finish, error = [], None, ""
    try:
        with requests.post(BASE_URL + "/v1/chat/completions",
                           json={"model": MODEL, "messages": [{"role":"user","content":prompt}],
                                 "temperature":0,"max_tokens":output_tokens,"stream":True},
                           stream=True, timeout=(10, 180)) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(chunk_size=1):
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                for choice in event.get("choices", []):
                    if (choice.get("delta") or {}).get("content"):
                        chunks.append(time.perf_counter())
                    finish = choice.get("finish_reason") or finish
    except Exception as exc:
        error = str(exc)[:300]
    end = time.perf_counter()
    return {"success": not bool(error) and bool(chunks),
            "ttft_ms": (chunks[0]-begin)*1000 if chunks else np.nan,
            "chunk_gap_ms": float(np.mean(np.diff(chunks))*1000) if len(chunks)>1 else np.nan,
            "latency_s": end-begin, "finish_reason": finish, "error": error}

def warmup():
    r = request("Explain a GPU in one sentence.", output_tokens=24)
    if not r["success"]:
        raise RuntimeError("Model warmup failed: " + r["error"])

def benchmark(concurrency, name, interval=.25):
    if not Server.ready():
        raise RuntimeError("Server not READY; initialize it first.")
    before = metrics()
    trace, stop, gate = [], Event(), Event()
    t0 = time.perf_counter()

    def sample():
        while not stop.is_set():
            try:
                m = metrics()
                trace.append({"time_s":time.perf_counter()-t0,
                              "kv_usage_pct":m["kv_usage"]*100,
                              "running":m["running"],"waiting":m["waiting"]})
            except requests.RequestException:
                pass
            stop.wait(interval)

    worker = Thread(target=sample, daemon=True)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request, PROMPT + f"\nRequest ID: {i}. "
                                       "Provide a detailed numbered explanation with 30 points.", gate)
                       for i in range(concurrency)]
            gate.set()
            results = [f.result() for f in futures]
    finally:
        stop.set()
        worker.join(timeout=5)
    elapsed = time.perf_counter()-t0
    after = metrics()
    details = pd.DataFrame(results)
    trace_df = pd.DataFrame(trace)
    if trace_df.empty:
        trace_df = pd.DataFrame(columns=["time_s", "kv_usage_pct", "running", "waiting"])
    successful = details.loc[details["success"]]
    count = delta(before, after, "gen_tokens")
    summary = {"Condition":name,"Concurrency":concurrency,
               "Success":len(successful),"Failed":int((~details["success"]).sum()),
               "Throughput (tok/s)":count/elapsed if np.isfinite(count) else np.nan,
               "TTFT P95 (ms)":float(successful["ttft_ms"].quantile(.95)) if len(successful) else np.nan,
               "Mean Chunk Gap (ms)":float(successful["chunk_gap_ms"].mean()) if len(successful) else np.nan,
               "Peak KV (%)":float(trace_df["kv_usage_pct"].max()),
               "Peak Running":float(trace_df["running"].max()),
               "Peak Waiting":float(trace_df["waiting"].max()),
               "Preemption Delta":delta(before, after, "preemptions")}
    details.to_csv(OUT / f"{name}_requests.csv", index=False)
    trace_df.to_csv(OUT / f"{name}_trace.csv", index=False)
    print("[RESULT]", summary)
    return summary

def save_summary(rows, filename):
    df = pd.DataFrame(rows)
    path = OUT / filename
    df.to_csv(path, index=False)
    print("\n[SUMMARY]", path, "\n", df.to_string(index=False))
    return df
