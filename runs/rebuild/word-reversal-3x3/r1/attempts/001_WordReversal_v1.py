"""
Interpretable transformer for word-reversal-3x3.

Single-layer, single-head causal transformer (d_model=50) that performs
position-based token copying to reverse three 3-letter words.

No MLP, no LayerNorm, no output projection in attention.
Parameters: 11,450 (matching the original model).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture: single-layer, single-head, no MLP/LayerNorm/W_o
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """Single-layer single-head causal transformer for word reversal."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]

        # Single-head causal self-attention (no W_o, no LayerNorm, no MLP)
        q = self.W_q(h)
        k = self.W_k(h)
        v = self.W_v(h)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)

        h = h + (attn @ v)  # residual connection, no output projection

        return self.head(h)  # no final LayerNorm


# ---------------------------------------------------------------------------
# Hand-written weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Set all parameters in closed form to implement word reversal.

    Routing map (prediction_pos -> source_pos):
      pred 11 -> src 8  (w3[0])
      pred 12 -> src 9  (w3[1])
      pred 13 -> src 10 (w3[2])
      pred 14 -> src 7  (_ between w2 and w3)
      pred 15 -> src 4  (w2[0])
      pred 16 -> src 5  (w2[1])
      pred 17 -> src 6  (w2[2])
      pred 18 -> src 3  (_ between w1 and w2)
      pred 19 -> src 0  (w1[0])
      pred 20 -> src 1  (w1[1])
      pred 21 -> src 2  (w1[2])
    """
    d = model.d_model  # 50
    V = model.vocab_size  # 28

    with torch.no_grad():
        # ---- Token embedding: one-hot scale 10 in dims 0..V-1 ----
        model.token_emb.weight.zero_()
        for i in range(V):
            model.token_emb.weight[i, i] = 10.0

        # ---- Position embedding ----
        model.pos_emb.weight.zero_()
        # Position keys: positions 0-10, one-hot scale 10 in dims 28-38
        for p in range(11):
            model.pos_emb.weight[p, 28 + p] = 10.0
        # Routing queries: positions 11-21, one-hot scale 10 in dims 39-49
        for p in range(11, 22):
            model.pos_emb.weight[p, 39 + (p - 11)] = 10.0

        # ---- Routing map ----
        # routing[r] = source position for routing index r (r = pred_pos - 11)
        routing = [8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2]

        # ---- W_q: map routing query dims (39-49) to position match channels (0-10) ----
        # For routing index r, query channel routing[r] should be active
        W_q = torch.zeros(d, d)
        for r in range(11):
            src = routing[r]
            # Input dim 39+r (routing query) -> output dim src (position match channel)
            W_q[src, 39 + r] = 5.0
        model.W_q.weight.copy_(W_q)

        # ---- W_k: map position key dims (28-38) to position match channels (0-10) ----
        # For source position s, key channel s should be active
        W_k = torch.zeros(d, d)
        for s in range(11):
            # Input dim 28+s (position key) -> output dim s (position match channel)
            W_k[s, 28 + s] = 5.0
        model.W_k.weight.copy_(W_k)

        # ---- W_v: copy token identity (dims 0-27) with scale 2 ----
        W_v = torch.zeros(d, d)
        for i in range(V):
            W_v[i, i] = 2.0
        model.W_v.weight.copy_(W_v)

        # ---- Head: read dims 0-27 with scale 5 ----
        head_w = torch.zeros(V, d)
        for i in range(V):
            head_w[i, i] = 5.0
        model.head.weight.copy_(head_w)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "WordReversal_v1"
model_description = "Single-layer single-head d=50 position-routing transformer for word reversal, hand-written weights."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])
    print(f"Overall results saved → {path}")



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
