"""Experiment 04: pretrained LM sliding-window vs attention-sink KV eviction.

Requires: torch, transformers, requests; runs on Colab CUDA.
Same model, exact same tokens, exact same KV retention budget.
Evaluate next-token negative log likelihood (NLL) and perplexity.
GPT-NeoX RoPE uses ORIGINAL token position IDs for both policies
(consistent absolute position, NOT official StreamingLLM reindexing).
This is a controlled sink-retention ablation, not an exact reproduction
of the original StreamingLLM positional re-mapping implementation.
"""

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import torch

MODEL_NAME = "EleutherAI/pythia-70m-deduped"
RESULTS = Path("results/04_attention_sink_summary.csv")
TRACE = Path("results/04_attention_sink_trace.csv")


DEFAULT_TEXT = Path(__file__).resolve().parents[1] / "data" / "alice_excerpt.txt"


def load_text(local_file=None):
    path = Path(local_file) if local_file else DEFAULT_TEXT
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation text not found: {path}. Run git pull origin main.")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if len(text) < 5000:
        raise ValueError(f"Evaluation text unexpectedly short: {len(text)} characters")
    print("Input text source:", path)
    return text


def get_layer_cache(cache):
    """Return (key, value) pairs with setters for supported DynamicCache layouts."""
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        pairs = []
        for layer in cache.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                pairs.append((layer.keys, layer.values, layer, "keys", "values"))
            elif hasattr(layer, "key_states") and hasattr(layer, "value_states"):
                pairs.append((layer.key_states, layer.value_states, layer, "key_states", "value_states"))
            else:
                raise RuntimeError(
                    "Unrecognized DynamicCache layer layout. "
                    "Please share transformers version; do not interpret this as an evaluated run."
                )
        return pairs
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return [
            (k, v, cache, i, i)
            for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache))
        ]
    raise RuntimeError("Unsupported transformers DynamicCache layout.")


def set_layer_cache(cache, pair, new_k, new_v):
    _, _, obj, kslot, vslot = pair
    if isinstance(kslot, int):
        obj.key_cache[kslot] = new_k
        obj.value_cache[vslot] = new_v
    else:
        setattr(obj, kslot, new_k)
        setattr(obj, vslot, new_v)


def trim(cache, capacity, sink_count):
    """Keep exactly sink_count earliest + last (capacity - sink_count) cached tokens.

    Important: called AFTER consuming token t, BEFORE consuming t+1.
    """
    pairs = get_layer_cache(cache)
    assert len(pairs) > 0
    current = pairs[0][0].shape[-2]
    if current <= capacity:
        return current
    recent = capacity - sink_count
    assert recent >= 1 and sink_count >= 0
    for pair in pairs:
        k, v = pair[0], pair[1]
        assert k.shape[-2] == current and v.shape[-2] == current
        if sink_count:
            new_k = torch.cat([k[..., :sink_count, :], k[..., -recent:, :]], dim=-2)
            new_v = torch.cat([v[..., :sink_count, :], v[..., -recent:, :]], dim=-2)
        else:
            new_k = k[..., -capacity:, :].contiguous()
            new_v = v[..., -capacity:, :].contiguous()
        set_layer_cache(cache, pair, new_k, new_v)
    return capacity


def evaluate(model, token_ids, name, capacity, sinks, eval_limit, device):
    from transformers import DynamicCache

    cache = DynamicCache()
    losses = []
    rows = []
    max_cache = 0
    # Predict the following token after processing each current token.
    # One-token passes avoid ever needing to cache the entire document.
    with torch.inference_mode():
        for t in range(eval_limit):
            if t:
                cache_len = trim(cache, capacity, sinks)
            else:
                cache_len = 0

            # cache_position is the PHYSICAL position in the trimmed cache;
            # position_ids is the ORIGINAL absolute token position for RoPE.
            kwargs = {
                "input_ids": token_ids[t:t+1].view(1, 1).to(device),
                "past_key_values": cache,
                "attention_mask": torch.ones(
                    (1, cache_len + 1), device=device, dtype=torch.long
                ),
                "use_cache": True,
                "position_ids": torch.tensor([[t]], device=device, dtype=torch.long),
                "cache_position": torch.tensor([cache_len], device=device, dtype=torch.long),
                "return_dict": True,
            }
            out = model(**kwargs)
            cache = out.past_key_values
            assert cache is not None, "Model did not produce KV cache."
            current_cache = get_layer_cache(cache)[0][0].shape[-2]
            assert current_cache <= capacity + 1, (name, t, current_cache)
            max_cache = max(max_cache, current_cache)
            log_prob = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nll = -log_prob[token_ids[t+1].item()].item()
            if not math.isfinite(nll):
                raise RuntimeError(f"Nonfinite NLL at t={t}, method={name}")
            losses.append(nll)
            if t == 0 or (t + 1) % 64 == 0 or t == eval_limit - 1:
                rows.append({
                    "policy": name, "position": t + 1,
                    "window_budget": capacity, "sink_tokens": sinks,
                    "cache_tokens_after_step": current_cache,
                    "mean_nll_so_far": statistics.mean(losses),
                    "ppl_so_far": math.exp(min(80, statistics.mean(losses)))
                })
    return {
        "policy": name,
        "sink_tokens": sinks,
        "cache_budget": capacity,
        "evaluated_next_tokens": len(losses),
        "avg_nll": statistics.mean(losses),
        "perplexity": math.exp(min(80, statistics.mean(losses))),
        "max_cache_tokens_observed": max_cache,
    }, rows



def run_self_test():
    """Offline check: local corpus and exact cache retention policies."""
    text = load_text()
    assert len(text) >= 5000
    class Layer:
        def __init__(self):
            self.keys = torch.arange(10).reshape(1, 1, 10, 1).float()
            self.values = self.keys.clone() * 10
    class Cache:
        def __init__(self):
            self.layers = [Layer(), Layer()]
    for sinks, expected in [(0, [4, 5, 6, 7, 8, 9]),
                            (2, [0, 1, 6, 7, 8, 9])]:
        cache = Cache()
        count = trim(cache, capacity=6, sink_count=sinks)
        assert count == 6
        for layer in cache.layers:
            keys = layer.keys.flatten().long().tolist()
            values = layer.values.flatten().long().tolist()
            assert keys == expected, (sinks, keys)
            assert values == [10 * i for i in expected], (sinks, values)
    print("SELF-TEST PASSED: bundled text, sliding eviction, sink retention, both K and V.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Offline corpus/eviction verification")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tokens", type=int, default=768)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--sinks", type=int, default=4)
    parser.add_argument("--text-file", type=str, default=None, help="Local UTF-8 text file instead of downloading")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    assert 0 < args.sinks < args.capacity
    assert args.tokens > args.capacity + 64
    assert torch.cuda.is_available(), "Select a GPU runtime."
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(2026)
    random.seed(2026)
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__, "transformers:", transformers.__version__)
    print("Model:", args.model)
    print("Evaluation text: bundled local Alice excerpt")
    text = load_text(args.text_file)
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    assert ids.numel() > args.tokens, (ids.numel(), args.tokens)
    ids = ids[:args.tokens+1].contiguous()
    print(f"Document tokens: {ids.numel()}; evaluated next tokens: {args.tokens}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager"
    ).to(device).eval()
    print("Attention implementation: eager; cache budget:", args.capacity)
    print("Both methods use original absolute position IDs (controlled ablation).")

    summaries, traces = [], []
    for name, sinks in [("sliding_window", 0), ("sink_plus_window", args.sinks)]:
        print("Evaluating:", name)
        row, trace = evaluate(
            model, ids, name, args.capacity, sinks, args.tokens, device
        )
        row.update({
            "model": args.model,
            "input_source": str(Path(args.text_file).resolve() if args.text_file else DEFAULT_TEXT),
            "position_policy": "original_absolute_rope",
            "dtype": "float16",
            "transformers_version": transformers.__version__
        })
        summaries.append(row)
        traces.extend(trace)
        print(f"  PPL={row['perplexity']:.4f}, NLL={row['avg_nll']:.4f}, "
              f"max cache={row['max_cache_tokens_observed']}")
        torch.cuda.empty_cache()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(RESULTS, summaries), (TRACE, traces)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print("Saved:", path)
    delta = summaries[1]["perplexity"] - summaries[0]["perplexity"]
    print(f"Sink minus sliding PPL: {delta:+.4f} (negative = sink lower)")
    print("Interpretation: model-specific, one-text, original-absolute-RoPE ablation.")
    print("This is not official StreamingLLM's positional reindexing evaluation.")


if __name__ == "__main__":
    main()
