"""Experiment 03b: label accuracy before/after MLA query-side weight absorption.

A small learned synthetic key-value retrieval task. Both inference paths use
IDENTICAL weights and data. This tests algebraic implementation equivalence,
not the quality of a pretrained language model or low-rank vs full-rank architecture.
"""
import csv
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SEED = 2026
NUM_KEYS = 64
NUM_VALUES = 16
SLOTS = 8
D = 64
LATENT = 16
TRAIN_STEPS = 400
TRAIN_BATCH = 256
TEST_SAMPLES = 4096
OUTPUT = Path("results/03b_weight_absorption_accuracy.csv")


def make_batch(n, device, generator):
    # Each example: eight distinct lookup keys paired with random values.
    indices = torch.rand(n, NUM_KEYS, device=device, generator=generator).topk(SLOTS, dim=-1).indices
    values = torch.randint(NUM_VALUES, (n, SLOTS), device=device, generator=generator)
    chosen = torch.randint(SLOTS, (n,), device=device, generator=generator)
    queries = indices.gather(1, chosen[:, None]).squeeze(1)
    labels = values.gather(1, chosen[:, None]).squeeze(1)
    return indices, values, queries, labels


class RetrievalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_latent = nn.Embedding(NUM_KEYS, LATENT)
        self.query = nn.Embedding(NUM_KEYS, D)
        self.up = nn.Parameter(torch.randn(D, LATENT) / math.sqrt(LATENT))

    def forward(self, indices, values, queries, method):
        c = self.key_latent(indices)             # [batch, slots, latent]
        q = self.query(queries)                   # [batch, D]
        if method == "explicit":
            k = c @ self.up.T                     # [batch, slots, D]
            scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)
        elif method == "absorbed":
            q_low = q @ self.up                   # [batch, latent]
            scores = torch.bmm(c, q_low.unsqueeze(-1)).squeeze(-1) / math.sqrt(D)
        else:
            raise ValueError(method)
        attn = torch.softmax(scores, dim=-1)
        value_labels = F.one_hot(values, num_classes=NUM_VALUES).to(attn.dtype)
        probs = torch.bmm(attn.unsqueeze(1), value_labels).squeeze(1)
        return probs, scores


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    print("Device:", device, "| torch:", torch.__version__)
    print("Task: predict the value paired with a queried key among 8 key-value entries")
    print("This is a trained synthetic retrieval classifier, NOT pretrained LLM accuracy.")

    model = RetrievalAttention().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.015)
    training_rng = torch.Generator(device=device).manual_seed(SEED)
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        indices, values, queries, labels = make_batch(TRAIN_BATCH, device, training_rng)
        probs, _ = model(indices, values, queries, "explicit")
        loss = F.nll_loss(torch.log(probs.clamp_min(1e-9)), labels)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 100 == 0 or step == 1:
            acc = (probs.argmax(-1) == labels).float().mean().item()
            print(f"train step {step:3d}/{TRAIN_STEPS}, loss={loss.item():.4f}, batch_acc={acc*100:.2f}%")

    model.eval()
    test_rng = torch.Generator(device=device).manual_seed(SEED + 12345)
    batch = make_batch(TEST_SAMPLES, device, test_rng)
    labels = batch[-1]
    with torch.inference_mode():
        explicit_probs, explicit_scores = model(*batch[:-1], method="explicit")
        absorbed_probs, absorbed_scores = model(*batch[:-1], method="absorbed")
        pred_a = explicit_probs.argmax(-1)
        pred_b = absorbed_probs.argmax(-1)
        acc_a = (pred_a == labels).float().mean().item()
        acc_b = (pred_b == labels).float().mean().item()
        loss_a = F.nll_loss(torch.log(explicit_probs.clamp_min(1e-9)), labels).item()
        loss_b = F.nll_loss(torch.log(absorbed_probs.clamp_min(1e-9)), labels).item()
        disagree = (pred_a != pred_b).sum().item()
        err = (explicit_scores - absorbed_scores).abs().max().item()
        prob_err = (explicit_probs - absorbed_probs).abs().max().item()
    row = {
        "task": "synthetic_key_value_retrieval",
        "test_samples": TEST_SAMPLES, "train_steps": TRAIN_STEPS,
        "explicit_accuracy": acc_a, "absorbed_accuracy": acc_b,
        "accuracy_difference_percentage_points": (acc_b - acc_a) * 100,
        "explicit_cross_entropy": loss_a, "absorbed_cross_entropy": loss_b,
        "prediction_disagreements": disagree,
        "max_abs_score_difference": err,
        "max_abs_probability_difference": prob_err,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print("\n=== HELD-OUT TEST (same weights / same samples) ===")
    print(f"Explicit accuracy       : {acc_a*100:.4f}%")
    print(f"Absorbed accuracy       : {acc_b*100:.4f}%")
    print(f"Accuracy difference     : {(acc_b-acc_a)*100:+.6f} percentage points")
    print(f"Explicit cross-entropy  : {loss_a:.6f}")
    print(f"Absorbed cross-entropy  : {loss_b:.6f}")
    print(f"Prediction disagreements: {disagree} / {TEST_SAMPLES}")
    print(f"Max attention score error: {err:.3e}")
    print(f"Max class prob error     : {prob_err:.3e}")
    print(f"Saved: {OUTPUT}")
    print("Interpretation: absorption changes operation order, not learned information.")
    print("Limit: this task does not assess pretrained LLM accuracy or low-rank compression quality.")


if __name__ == "__main__":
    main()
