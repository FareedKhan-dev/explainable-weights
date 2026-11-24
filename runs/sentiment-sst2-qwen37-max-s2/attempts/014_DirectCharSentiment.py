"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Multi-head character pattern detection with strong weights."""
    torch.manual_seed(0)
    D = model.d_model
    
    # Token embeddings: one-hot character encoding with scaling
    model.token_emb.weight.data.zero_()
    stoi = task.stoi
    for char, idx in stoi.items():
        if idx < D:
            model.token_emb.weight.data[idx, idx] = 10.0  # Scale up
    
    model.pos_emb.weight.data.zero_()
    
    # Block 0: Multi-head character detection
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    # Attention: uniform (Q=0, K=0)
    block0.attn.W_q.weight.data.zero_()
    block0.attn.W_k.weight.data.zero_()
    
    # V: copy character embeddings
    block0.attn.W_v.weight.data.zero_()
    for i in range(min(32, D)):
        block0.attn.W_v.weight.data[i, i] = 1.0
    
    # O: copy output
    block0.attn.W_o.weight.data.zero_()
    for i in range(min(32, D)):
        block0.attn.W_o.weight.data[i, i] = 1.0
    
    # MLP: Multi-head character pattern classifier
    fc1 = block0.mlp.fc1.weight.data
    fc1_b = block0.mlp.fc1.bias.data
    fc2 = block0.mlp.fc2.weight.data
    fc2_b = block0.mlp.fc2.bias.data
    
    fc1.zero_()
    fc1_b.zero_()
    fc2.zero_()
    fc2_b.zero_()
    
    # Use first 10 hidden units for different character patterns
    # Each detects a different combination
    
    # Hidden 0: Strong positive characters (g, k, p, v)
    fc1[0, 6] = 5.0   # g
    fc1[0, 10] = 5.0  # k
    fc1[0, 15] = 5.0  # p
    fc1[0, 21] = 5.0  # v
    fc1_b[0] = -15.0
    
    # Hidden 1: Moderate positive (r, y, x)
    fc1[1, 17] = 5.0  # r
    fc1[1, 24] = 5.0  # y
    fc1[1, 23] = 5.0  # x
    fc1_b[1] = -10.0
    
    # Hidden 2: Strong negative (q, j, z)
    fc1[2, 16] = 5.0  # q
    fc1[2, 9] = 5.0   # j
    fc1[2, 25] = 5.0  # z
    fc1_b[2] = -10.0
    
    # Hidden 3: Moderate negative (u, l, c, d)
    fc1[3, 20] = 5.0  # u
    fc1[3, 11] = 5.0  # l
    fc1[3, 2] = 5.0   # c
    fc1[3, 3] = 5.0   # d
    fc1_b[3] = -15.0
    
    # Hidden 4: Very negative ('s, apostrophe)
    fc1[4, 27] = 10.0  # '
    fc1_b[4] = -5.0
    
    # fc2: Combine patterns
    fc2[0, 0] = 50.0   # Strong positive
    fc2[0, 1] = 30.0   # Moderate positive
    fc2[0, 2] = -50.0  # Strong negative
    fc2[0, 3] = -30.0  # Moderate negative
    fc2[0, 4] = -100.0 # Very negative (apostrophe penalty)
    
    # Block 1, 2: identity
    for block in model.blocks[1:]:
        block.attn.W_q.weight.data.zero_()
        block.attn.W_k.weight.data.zero_()
        block.attn.W_v.weight.data.zero_()
        block.attn.W_o.weight.data.zero_()
        block.mlp.fc1.weight.data.zero_()
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.zero_()
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: classify based on sentiment score (dim 0)
    model.head.weight.data.zero_()
    model.head.weight.data[31, 0] = 10.0   # Positive if score > 0
    model.head.weight.data[30, 0] = -10.0  # Negative if score < 0
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "DirectCharSentiment"
model_description = "Direct character sentiment scores in embeddings with uniform attention averaging."


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
