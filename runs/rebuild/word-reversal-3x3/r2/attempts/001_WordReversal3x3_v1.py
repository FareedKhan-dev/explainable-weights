"""
Interpretable transformer for word-reversal-3x3 task.

Single-layer, single-head causal transformer (d_model=50, no MLP, LayerNorm,
or output projection) that performs position-based token copying to reverse
three 3-letter words separated by underscores.

prompt: "abc_def_ghi="   (12 tokens, positions 0-11)
answer: "ghi_def_abc"    (11 tokens, predicted from positions 11-21)
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
# Architecture: single-layer, single-head, d_model=50, no MLP/LN/W_o
# ---------------------------------------------------------------------------

class SimpleTransformer(nn.Module):
    """Single-layer, single-head causal transformer (d_model=50)."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 50,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 256,
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
        # No W_o (output projection)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]

        q = self.W_q(h)  # (B, T, D)
        k = self.W_k(h)  # (B, T, D)
        v = self.W_v(h)  # (B, T, D)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)

        h = h + attn @ v  # residual connection, no output projection
        return self.head(h)


# ---------------------------------------------------------------------------
# Hand-written weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate model parameters in closed form for word-reversal-3x3.

    Layout of d_model=50 dimensions:
      dims  0-27: token identity (one-hot, 28 vocab chars)
      dims 28-38: position key (one-hot, positions 0-10)
      dims 39-49: routing query (one-hot, prediction positions 11-21)

    Attention routing (prediction position -> source position):
      11->8, 12->9, 13->10  (word3 reversed)
      14->7                  (underscore)
      15->4, 16->5, 17->6   (word2 reversed)
      18->3                  (underscore)
      19->0, 20->1, 21->2   (word1 reversed)
    """
    with torch.no_grad():
        d = model.d_model  # 50
        V = model.vocab_size  # 28

        # ---- Token embeddings: one-hot scale 10 in dims 0..27 ----
        model.token_emb.weight.zero_()
        for v in range(V):
            model.token_emb.weight[v, v] = 10.0

        # ---- Position embeddings ----
        model.pos_emb.weight.zero_()

        # Position keys: one-hot scale 10 in dims 28+p for positions 0..10
        for p in range(11):
            model.pos_emb.weight[p, 28 + p] = 10.0

        # Routing queries: one-hot scale 10 in dims 39+s for positions 11..21
        # s(p) = source position that prediction position p should attend to
        source_map = {
            11: 8,  12: 9,  13: 10,  # word3 letters
            14: 7,                     # second underscore
            15: 4,  16: 5,  17: 6,   # word2 letters
            18: 3,                     # first underscore
            19: 0,  20: 1,  21: 2,   # word1 letters
        }
        for p, s in source_map.items():
            model.pos_emb.weight[p, 39 + s] = 10.0

        # ---- W_q: extract routing query (dims 39-49) -> output dims (0-10), scale 5 ----
        # W_q.weight shape: (d_model, d_model) = (50, 50)
        # nn.Linear: output = input @ weight.T, so weight[out, in]
        model.W_q.weight.zero_()
        for k in range(11):
            model.W_q.weight[k, 39 + k] = 5.0

        # ---- W_k: extract position key (dims 28-38) -> output dims (0-10), scale 5 ----
        model.W_k.weight.zero_()
        for k in range(11):
            model.W_k.weight[k, 28 + k] = 5.0

        # ---- W_v: copy token identity (dims 0-27) with scale 2 ----
        model.W_v.weight.zero_()
        for k in range(min(28, d)):
            model.W_v.weight[k, k] = 2.0

        # ---- Output head: read dims 0-27 with scale 5 ----
        # head.weight shape: (vocab_size, d_model) = (28, 50)
        model.head.weight.zero_()
        for v in range(V):
            model.head.weight[v, v] = 5.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "WordReversal3x3_v1"
model_description = (
    "Single-layer single-head d_model=50 transformer performing position-based "
    "token copying via hard argmax attention to reverse three 3-letter words."
)


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
