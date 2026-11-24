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
    Linear interpolation: result = y1 + (y2-y1)*(x3-x1)/(x2-x1)
    
    Strategy:
    1. Use attention to extract 7 input values
    2. Use MLP layers to compute step by step
    3. For division, use a lookup table approximation
    """
    D = model.d_model  # 256
    V = model.vocab_size  # 13
    H = model.n_heads  # 8
    dh = D // H  # 32
    FF = model.d_ff  # 1024
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDINGS ----
    # Use position encoding in token embedding
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0  # one-hot digit
    
    # ---- POSITION EMBEDDINGS ----
    # One-hot position + strong signal for input positions
    for p in range(14):
        model.pos_emb.weight.data[p, 20 + p] = 1.0
        if p in [0, 2, 3, 5, 7, 8, 10]:  # input positions
            model.pos_emb.weight.data[p, 20 + p] = 10.0
    
    # ---- BLOCK 0: Extract 7 input values via attention ----
    block0 = model.blocks[0]
    
    # Positions to extract: 0, 2, 3, 5, 7, 8, 10
    # Map to heads 0-6
    target_positions = [0, 2, 3, 5, 7, 8, 10]
    
    for h in range(7):
        target_pos = target_positions[h]
        
        # Query: look at position 12 (output position)
        block0.attn.W_q.weight.data[h*dh, 20 + 12] = 1.0
        
        # Key: target position advertises itself strongly
        block0.attn.W_k.weight.data[h*dh, 20 + target_pos] = 10.0
        
        # Value: extract the digit one-hot encoding
        for d in range(10):
            block0.attn.W_v.weight.data[h*dh + d, d] = 1.0
        
        # Output: write to dedicated channels
        # Head h writes to dims [h*10 : (h+1)*10]
        for d in range(10):
            block0.attn.W_o.weight.data[h*10 + d, h*dh + d] = 1.0
    
    # Head 7: unused (zero)
    
    # Block 0 MLP: Pass through extracted values
    block0.mlp.fc1.weight.data[:70, :70] = torch.eye(70)
    block0.mlp.fc2.weight.data[:70, :70] = torch.eye(70)
    
    # ---- BLOCK 1: Compute y1, y2, dy, dx, dxq ----
    block1 = model.blocks[1]
    
    # Attention: identity (just pass through)
    for h in range(H):
        block1.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block1.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: Compute intermediate values
    # Input: dims 0-69 contain [x1, y1_tens, y1_ones, x2, y2_tens, y2_ones, x3]
    # Each is one-hot (10 dims per value)
    
    # Output: y1 (scalar), y2 (scalar), dy (scalar), dx (scalar), dxq (scalar)
    # Write to dims 70-74
    
    # y1 = y1_tens*10 + y1_ones
    # y1_tens is in dims 10-19, y1_ones in dims 20-29
    for d in range(10):
        block1.mlp.fc1.weight.data[70, 10 + d] = d * 10  # y1_tens * 10
        block1.mlp.fc1.weight.data[70, 20 + d] = d       # + y1_ones
    
    # y2 = y2_tens*10 + y2_ones
    # y2_tens is in dims 40-49, y2_ones in dims 50-59
    for d in range(10):
        block1.mlp.fc1.weight.data[71, 40 + d] = d * 10  # y2_tens * 10
        block1.mlp.fc1.weight.data[71, 50 + d] = d       # + y2_ones
    
    # dy = y2 - y1
    block1.mlp.fc1.weight.data[72, 71] = 1.0   # y2
    block1.mlp.fc1.weight.data[72, 70] = -1.0  # - y1
    
    # dx = x2 - x1
    # x2 is in dims 30-39, x1 in dims 0-9
    for d in range(10):
        block1.mlp.fc1.weight.data[73, 30 + d] = d   # x2
        block1.mlp.fc1.weight.data[73, 0 + d] = -d   # - x1
    
    # dxq = x3 - x1
    # x3 is in dims 60-69
    for d in range(10):
        block1.mlp.fc1.weight.data[74, 60 + d] = d   # x3
        block1.mlp.fc1.weight.data[74, 0 + d] = -d   # - x1
    
    # Pass through intermediate values
    block1.mlp.fc2.weight.data[70:75, 70:75] = torch.eye(5)
    
    # ---- BLOCK 2: Compute result = y1 + dy*dxq/dx ----
    block2 = model.blocks[2]
    
    # Attention: identity
    for h in range(H):
        block2.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block2.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: result = y1 + dy*dxq/dx
    # This is hard because we need division. Let's use a different approach:
    # result = y1 + dy * (dxq/dx)
    # We can approximate dxq/dx as a fraction
    
    # For now, let's just use y1 as a baseline (will be wrong but let's see)
    block2.mlp.fc1.weight.data[80, 70] = 1.0  # result = y1
    block2.mlp.fc2.weight.data[80, 80] = 1.0
    
    # ---- BLOCK 3: Convert result to two digits ----
    block3 = model.blocks[3]
    
    # Attention: identity
    for h in range(H):
        block3.attn.W_q.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_k.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_v.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
        block3.attn.W_o.weight.data[h*dh:h*dh+dh, h*dh:h*dh+dh] = torch.eye(dh) * 0.01
    
    # MLP: tens = result // 10, ones = result % 10
    # Use threshold functions
    for d in range(10):
        # tens digit: activate if result >= d*10
        block3.mlp.fc1.weight.data[90 + d, 80] = 1.0
        block3.mlp.fc1.bias.data[90 + d] = -d * 10
        
        # ones digit: activate if (result % 10) >= d
        block3.mlp.fc1.weight.data[100 + d, 80] = 1.0
        block3.mlp.fc1.bias.data[100 + d] = -d
    
    block3.mlp.fc2.weight.data[90:100, 90:100] = torch.eye(10)
    block3.mlp.fc2.weight.data[100:110, 100:110] = torch.eye(10)
    
    # ---- OUTPUT HEAD ----
    # Map tens (dims 90-99) and ones (dims 100-109) to vocab
    # Vocab: 0-9 are digits, 10-12 are special
    for d in range(10):
        model.head.weight.data[d, 90 + d] = 1.0    # tens digit
        model.head.weight.data[d, 100 + d] = 1.0   # ones digit


model_shorthand_name = "v12_interpolation_attempt"
model_description = "Extract 7 inputs, compute y1/y2/dy/dx/dxq, output y1 (baseline)"


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
