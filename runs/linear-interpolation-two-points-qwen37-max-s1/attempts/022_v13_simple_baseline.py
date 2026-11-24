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
    Linear interpolation with lookup table for division.
    
    Strategy: Use attention to extract values, then use MLP as a lookup table
    to compute the interpolation result directly for common cases.
    """
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H
    FF = model.d_ff
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # Token embeddings: one-hot for digits
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0
    
    # Position embeddings: strong signals for input positions
    input_positions = [0, 2, 3, 5, 7, 8, 10]
    for p in range(14):
        model.pos_emb.weight.data[p, 20 + p] = 1.0
        if p in input_positions:
            model.pos_emb.weight.data[p, 20 + p] = 10.0
    
    # Block 0: Extract 7 input values
    block0 = model.blocks[0]
    
    for h in range(7):
        target_pos = input_positions[h]
        
        # Query from position 12 (output start)
        block0.attn.W_q.weight.data[h*dh, 20 + 12] = 1.0
        
        # Key: target position
        block0.attn.W_k.weight.data[h*dh, 20 + target_pos] = 10.0
        
        # Value: extract digit
        for d in range(10):
            block0.attn.W_v.weight.data[h*dh + d, d] = 1.0
        
        # Output to dedicated channels
        for d in range(10):
            block0.attn.W_o.weight.data[h*10 + d, h*dh + d] = 1.0
    
    # Block 1: Compute scalar values
    block1 = model.blocks[1]
    
    # Pass through attention
    for h in range(H):
        block1.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: Convert one-hot to scalars
    # x1 (dims 0-9) -> scalar at dim 70
    for d in range(10):
        block1.mlp.fc1.weight.data[70, d] = d
    block1.mlp.fc2.weight.data[70, 70] = 1.0
    
    # y1 (dims 10-19 tens, 20-29 ones) -> scalar at dim 71
    for d in range(10):
        block1.mlp.fc1.weight.data[71, 10 + d] = d * 10
        block1.mlp.fc1.weight.data[71, 20 + d] = d
    block1.mlp.fc2.weight.data[71, 71] = 1.0
    
    # x2 (dims 30-39) -> scalar at dim 72
    for d in range(10):
        block1.mlp.fc1.weight.data[72, 30 + d] = d
    block1.mlp.fc2.weight.data[72, 72] = 1.0
    
    # y2 (dims 40-49 tens, 50-59 ones) -> scalar at dim 73
    for d in range(10):
        block1.mlp.fc1.weight.data[73, 40 + d] = d * 10
        block1.mlp.fc1.weight.data[73, 50 + d] = d
    block1.mlp.fc2.weight.data[73, 73] = 1.0
    
    # x3 (dims 60-69) -> scalar at dim 74
    for d in range(10):
        block1.mlp.fc1.weight.data[74, 60 + d] = d
    block1.mlp.fc2.weight.data[74, 74] = 1.0
    
    # Block 2: Compute result using approximation
    block2 = model.blocks[2]
    
    # Pass through attention
    for h in range(H):
        block2.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: Compute result = y1 + (y2-y1)*(x3-x1)/(x2-x1)
    # Use approximation: if x3 is close to x1, result ≈ y1
    # if x3 is close to x2, result ≈ y2
    # Otherwise interpolate
    
    # Compute dy = y2 - y1
    block2.mlp.fc1.weight.data[80, 73] = 1.0
    block2.mlp.fc1.weight.data[80, 71] = -1.0
    block2.mlp.fc2.weight.data[80, 80] = 1.0
    
    # Compute dx = x2 - x1
    block2.mlp.fc1.weight.data[81, 72] = 1.0
    block2.mlp.fc1.weight.data[81, 70] = -1.0
    block2.mlp.fc2.weight.data[81, 81] = 1.0
    
    # Compute dxq = x3 - x1
    block2.mlp.fc1.weight.data[82, 74] = 1.0
    block2.mlp.fc1.weight.data[82, 70] = -1.0
    block2.mlp.fc2.weight.data[82, 82] = 1.0
    
    # Compute ratio = dxq/dx (approximation)
    # For simplicity, use: if dxq < dx/2, ratio ≈ 0, else ratio ≈ 1
    # This gives us: result ≈ y1 if x3 is in first half, y2 if in second half
    block2.mlp.fc1.weight.data[83, 82] = 2.0  # 2*dxq
    block2.mlp.fc1.weight.data[83, 81] = -1.0  # -dx
    block2.mlp.fc1.bias.data[83] = 0.0
    block2.mlp.fc2.weight.data[83, 83] = 1.0
    
    # Result = y1 if ratio < 0, else y2
    # Use ReLU to implement: result = y1 + (y2-y1) * max(0, sign(ratio))
    # For now, just output y1 as baseline
    block2.mlp.fc1.weight.data[85, 71] = 1.0
    block2.mlp.fc2.weight.data[85, 85] = 1.0
    
    # Block 3: Convert to digits
    block3 = model.blocks[3]
    
    # Pass through attention
    for h in range(H):
        block3.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: Convert scalar to tens and ones digits
    for d in range(10):
        block3.mlp.fc1.weight.data[90 + d, 85] = 1.0
        block3.mlp.fc1.bias.data[90 + d] = -d * 10 - 5
        
        block3.mlp.fc1.weight.data[100 + d, 85] = 1.0
        block3.mlp.fc1.bias.data[100 + d] = -d - 0.5
    
    block3.mlp.fc2.weight.data[90:100, 90:100] = torch.eye(10)
    block3.mlp.fc2.weight.data[100:110, 100:110] = torch.eye(10)
    
    # Output head
    for d in range(10):
        model.head.weight.data[d, 90 + d] = 1.0
        model.head.weight.data[d, 100 + d] = 1.0


model_shorthand_name = "v13_simple_baseline"
model_description = "Simple baseline: extract values, output y1 as approximation"


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
