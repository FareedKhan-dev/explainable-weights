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

import numpy as np
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
        # Disable LayerNorm for hand-coded weights
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SimpleTransformer(nn.Module):
    """Configurable causal transformer."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
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
    """
    Lookup table approach: use MLP as a lookup table for interpolation.
    
    Since inputs are limited (x: 0-9, y: 0-99), we can encode common cases.
    For now, use a simple heuristic: output (y1 + y2) / 2 rounded.
    """
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H
    FF = model.d_ff
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # Set LayerNorms to identity
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    # Token embeddings: one-hot
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0
    
    # Position embeddings: mark output positions
    model.pos_emb.weight.data[12, 100] = 10.0  # output tens
    model.pos_emb.weight.data[13, 101] = 10.0  # output ones
    
    # Mark y1 and y2 positions
    model.pos_emb.weight.data[2, 150] = 10.0   # y1 tens
    model.pos_emb.weight.data[3, 151] = 10.0   # y1 ones
    model.pos_emb.weight.data[7, 152] = 10.0   # y2 tens
    model.pos_emb.weight.data[8, 153] = 10.0   # y2 ones
    
    # Block 0: Extract y1 and y2
    block0 = model.blocks[0]
    
    # Head 0: y1 tens
    block0.attn.W_q.weight.data[0, 150] = 1.0
    block0.attn.W_k.weight.data[0, 150] = 10.0
    for d in range(10):
        block0.attn.W_v.weight.data[d, d] = 1.0
        block0.attn.W_o.weight.data[d, d] = 1.0
    
    # Head 1: y1 ones
    block0.attn.W_q.weight.data[dh, 151] = 1.0
    block0.attn.W_k.weight.data[dh, 151] = 10.0
    for d in range(10):
        block0.attn.W_v.weight.data[dh + d, d] = 1.0
        block0.attn.W_o.weight.data[10 + d, dh + d] = 1.0
    
    # Head 2: y2 tens
    block0.attn.W_q.weight.data[2*dh, 152] = 1.0
    block0.attn.W_k.weight.data[2*dh, 152] = 10.0
    for d in range(10):
        block0.attn.W_v.weight.data[2*dh + d, d] = 1.0
        block0.attn.W_o.weight.data[20 + d, 2*dh + d] = 1.0
    
    # Head 3: y2 ones
    block0.attn.W_q.weight.data[3*dh, 153] = 1.0
    block0.attn.W_k.weight.data[3*dh, 153] = 10.0
    for d in range(10):
        block0.attn.W_v.weight.data[3*dh + d, d] = 1.0
        block0.attn.W_o.weight.data[30 + d, 3*dh + d] = 1.0
    
    for i in range(40):
        block0.mlp.fc1.weight.data[i, i] = 1.0
        block0.mlp.fc2.weight.data[i, i] = 1.0
    
    # Block 1: Compute average
    block1 = model.blocks[1]
    
    for h in range(H):
        for i in range(min(dh, 10)):
            block1.attn.W_q.weight.data[h*dh + i, h*dh + i] = 0.01
            block1.attn.W_k.weight.data[h*dh + i, h*dh + i] = 0.01
            block1.attn.W_v.weight.data[h*dh + i, h*dh + i] = 0.01
            block1.attn.W_o.weight.data[h*dh + i, h*dh + i] = 0.01
    
    # Compute y1 and y2 as integers
    # y1 = y1_tens*10 + y1_ones
    # y2 = y2_tens*10 + y2_ones
    # avg = (y1 + y2) / 2
    
    # First, compute y1 and y2 scalars
    for d in range(10):
        block1.mlp.fc1.weight.data[100, d] = 10.0        # y1 tens * 10
        block1.mlp.fc1.weight.data[100, 10 + d] = 1.0    # y1 ones
        block1.mlp.fc1.weight.data[101, 20 + d] = 10.0   # y2 tens * 10
        block1.mlp.fc1.weight.data[101, 30 + d] = 1.0    # y2 ones
    
    block1.mlp.fc2.weight.data[100, 100] = 1.0  # y1 scalar
    block1.mlp.fc2.weight.data[101, 101] = 1.0  # y2 scalar
    
    # Block 2: Compute average and convert to digits
    block2 = model.blocks[2]
    
    for h in range(H):
        for i in range(min(dh, 10)):
            block2.attn.W_q.weight.data[h*dh + i, h*dh + i] = 0.01
            block2.attn.W_k.weight.data[h*dh + i, h*dh + i] = 0.01
            block2.attn.W_v.weight.data[h*dh + i, h*dh + i] = 0.01
            block2.attn.W_o.weight.data[h*dh + i, h*dh + i] = 0.01
    
    # avg = (y1 + y2) / 2
    block2.mlp.fc1.weight.data[110, 100] = 0.5  # y1/2
    block2.mlp.fc1.weight.data[110, 101] = 0.5  # y2/2
    block2.mlp.fc2.weight.data[110, 110] = 1.0  # avg
    
    # Convert avg to tens and ones using threshold detectors
    for d in range(10):
        # Tens digit: floor(avg / 10)
        block2.mlp.fc1.weight.data[120 + d, 110] = 1.0
        block2.mlp.fc1.bias.data[120 + d] = -(d * 10 + 5)
        
        # Ones digit: avg % 10
        block2.mlp.fc1.weight.data[130 + d, 110] = 1.0
        block2.mlp.fc1.bias.data[130 + d] = -(d + 0.5)
    
    for i in range(120, 140):
        block2.mlp.fc2.weight.data[i, i] = 1.0
    
    # Block 3: Route to output
    block3 = model.blocks[3]
    
    # Route tens to position 12
    block3.attn.W_q.weight.data[0, 100] = 1.0
    block3.attn.W_k.weight.data[0, 120] = 10.0
    for d in range(10):
        block3.attn.W_v.weight.data[d, 120 + d] = 1.0
        block3.attn.W_o.weight.data[d, d] = 1.0
    
    # Route ones to position 13
    block3.attn.W_q.weight.data[dh, 101] = 1.0
    block3.attn.W_k.weight.data[dh, 130] = 10.0
    for d in range(10):
        block3.attn.W_v.weight.data[dh + d, 130 + d] = 1.0
        block3.attn.W_o.weight.data[10 + d, dh + d] = 1.0
    
    for i in range(20):
        block3.mlp.fc1.weight.data[i, i] = 1.0
        block3.mlp.fc2.weight.data[i, i] = 1.0
    
    # Output head
    for d in range(10):
        model.head.weight.data[d, d] = 10.0
        model.head.weight.data[d, 10 + d] = 10.0


model_shorthand_name = "v23_lookup_table"
model_description = "MLP lookup table: output (y1+y2)/2 as heuristic"


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
