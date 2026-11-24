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
    v27: Nearest-neighbor with proper routing.
    
    Strategy:
    - Block 0: Extract x1, x2, x3 to pos 11,12 (dims 0-29)
    - Block 1: Extract y1, y2 to pos 11,12 (dims 30-89)
    - Block 2 MLP: Compute selector = |x3-x2| - |x3-x1|
    - Block 2 attn: Route y1 or y2 based on selector
    """
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H
    FF = model.d_ff
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # LayerNorms to identity
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Token embeddings: digit d -> one-hot in dim d
    for d in range(10):
        model.token_emb.weight.data[d, d] = 10.0
    
    # Position embeddings: flag per position in dims 100+
    for pos in range(14):
        model.pos_emb.weight.data[pos, 100 + pos] = 10.0
    
    # ==================== BLOCK 0: Extract x values ====================
    blk0 = model.blocks[0]
    
    # Head 0: pos 11 queries x1 (pos 0)
    blk0.attn.W_q.weight.data[0, 111] = 10.0
    blk0.attn.W_k.weight.data[0, 100] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[d, d] = 10.0
        blk0.attn.W_o.weight.data[d, d] = 1.0  # x1 to dims 0-9
    
    # Head 1: pos 11 queries x2 (pos 5)
    blk0.attn.W_q.weight.data[dh, 111] = 10.0
    blk0.attn.W_k.weight.data[dh, 105] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[dh + d, d] = 10.0
        blk0.attn.W_o.weight.data[10 + d, dh + d] = 1.0  # x2 to dims 10-19
    
    # Head 2: pos 11 queries x3 (pos 10)
    blk0.attn.W_q.weight.data[2*dh, 111] = 10.0
    blk0.attn.W_k.weight.data[2*dh, 110] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[2*dh + d, d] = 10.0
        blk0.attn.W_o.weight.data[20 + d, 2*dh + d] = 1.0  # x3 to dims 20-29
    
    # Also copy to pos 12
    # Head 3: pos 12 queries x1
    blk0.attn.W_q.weight.data[3*dh, 112] = 10.0
    blk0.attn.W_k.weight.data[3*dh, 100] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[3*dh + d, d] = 10.0
        blk0.attn.W_o.weight.data[d, 3*dh + d] = 1.0
    
    # Head 4: pos 12 queries x2
    blk0.attn.W_q.weight.data[4*dh, 112] = 10.0
    blk0.attn.W_k.weight.data[4*dh, 105] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[4*dh + d, d] = 10.0
        blk0.attn.W_o.weight.data[10 + d, 4*dh + d] = 1.0
    
    # Head 5: pos 12 queries x3
    blk0.attn.W_q.weight.data[5*dh, 112] = 10.0
    blk0.attn.W_k.weight.data[5*dh, 110] = 10.0
    for d in range(10):
        blk0.attn.W_v.weight.data[5*dh + d, d] = 10.0
        blk0.attn.W_o.weight.data[20 + d, 5*dh + d] = 1.0
    
    # ==================== BLOCK 1: Extract y values ====================
    blk1 = model.blocks[1]
    
    # Head 0: pos 11 queries y1_tens (pos 2)
    blk1.attn.W_q.weight.data[0, 111] = 10.0
    blk1.attn.W_k.weight.data[0, 102] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[d, d] = 10.0
        blk1.attn.W_o.weight.data[30 + d, d] = 1.0  # y1t to dims 30-39
    
    # Head 1: pos 11 queries y1_ones (pos 3)
    blk1.attn.W_q.weight.data[dh, 111] = 10.0
    blk1.attn.W_k.weight.data[dh, 103] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[40 + d, dh + d] = 1.0  # y1o to dims 40-49
    
    # Head 2: pos 11 queries y2_tens (pos 7)
    blk1.attn.W_q.weight.data[2*dh, 111] = 10.0
    blk1.attn.W_k.weight.data[2*dh, 107] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[2*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[50 + d, 2*dh + d] = 1.0  # y2t to dims 50-59
    
    # Head 3: pos 11 queries y2_ones (pos 8)
    blk1.attn.W_q.weight.data[3*dh, 111] = 10.0
    blk1.attn.W_k.weight.data[3*dh, 108] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[3*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[60 + d, 3*dh + d] = 1.0  # y2o to dims 60-69
    
    # Same for pos 12
    # Head 4: y1_tens to pos 12
    blk1.attn.W_q.weight.data[4*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[4*dh, 102] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[4*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[30 + d, 4*dh + d] = 1.0
    
    # Head 5: y1_ones to pos 12
    blk1.attn.W_q.weight.data[5*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[5*dh, 103] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[5*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[40 + d, 5*dh + d] = 1.0
    
    # Head 6: y2_tens to pos 12
    blk1.attn.W_q.weight.data[6*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[6*dh, 107] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[6*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[50 + d, 6*dh + d] = 1.0
    
    # Head 7: y2_ones to pos 12
    blk1.attn.W_q.weight.data[7*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[7*dh, 108] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[7*dh + d, d] = 10.0
        blk1.attn.W_o.weight.data[60 + d, 7*dh + d] = 1.0
    
    # ==================== BLOCK 2: Compute selector and route ====================
    blk2 = model.blocks[2]
    mlp = blk2.mlp
    
    # Compute x1, x2, x3 as scalars from one-hot encodings
    # fc1 neurons 200-202: compute scalars
    for d in range(10):
        mlp.fc1.weight.data[200, d] = d       # x1 scalar
        mlp.fc1.weight.data[201, 10 + d] = d  # x2 scalar
        mlp.fc1.weight.data[202, 20 + d] = d  # x3 scalar
    mlp.fc2.weight.data[200, 200] = 1.0
    mlp.fc2.weight.data[201, 201] = 1.0
    mlp.fc2.weight.data[202, 202] = 1.0
    
    # fc1 neurons 203-206: compute |x3-x1| and |x3-x2|
    # relu(x3-x1)
    mlp.fc1.weight.data[203, 20] = 1.0
    mlp.fc1.weight.data[203, 0] = -1.0
    # relu(x1-x3)
    mlp.fc1.weight.data[204, 0] = 1.0
    mlp.fc1.weight.data[204, 20] = -1.0
    # relu(x3-x2)
    mlp.fc1.weight.data[205, 20] = 1.0
    mlp.fc1.weight.data[205, 10] = -1.0
    # relu(x2-x3)
    mlp.fc1.weight.data[206, 10] = 1.0
    mlp.fc1.weight.data[206, 20] = -1.0
    
    # fc2: combine to get absolute distances
    mlp.fc2.weight.data[203, 203] = 1.0  # relu(x3-x1)
    mlp.fc2.weight.data[203, 204] = 1.0  # + relu(x1-x3) = |x3-x1|
    mlp.fc2.weight.data[204, 205] = 1.0  # relu(x3-x2)
    mlp.fc2.weight.data[204, 206] = 1.0  # + relu(x2-x3) = |x3-x2|
    
    # Now we have |x3-x1| in neuron 203, |x3-x2| in neuron 204
    # We need to select y1 if |x3-x1| < |x3-x2| (i.e., 204 > 203)
    # Use attention to route: query based on which distance is smaller
    
    # Head 0: pos 11, query y1_tens (dim 30), key based on |x3-x2| > |x3-x1|
    blk2.attn.W_q.weight.data[0, 30] = 10.0   # query y1_tens
    blk2.attn.W_k.weight.data[0, 204] = 10.0  # key |x3-x2|
    for d in range(10):
        blk2.attn.W_v.weight.data[d, 30 + d] = 10.0
        blk2.attn.W_o.weight.data[d, d] = 1.0
    
    # Head 1: pos 11, query y2_tens (dim 50), key based on |x3-x1| > |x3-x2|
    blk2.attn.W_q.weight.data[dh, 50] = 10.0
    blk2.attn.W_k.weight.data[dh, 203] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[dh + d, 50 + d] = 10.0
        blk2.attn.W_o.weight.data[d, dh + d] = 1.0
    
    # Head 2: pos 12, query y1_ones (dim 40)
    blk2.attn.W_q.weight.data[2*dh, 40] = 10.0
    blk2.attn.W_k.weight.data[2*dh, 204] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[2*dh + d, 40 + d] = 10.0
        blk2.attn.W_o.weight.data[10 + d, 2*dh + d] = 1.0
    
    # Head 3: pos 12, query y2_ones (dim 60)
    blk2.attn.W_q.weight.data[3*dh, 60] = 10.0
    blk2.attn.W_k.weight.data[3*dh, 203] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[3*dh + d, 60 + d] = 10.0
        blk2.attn.W_o.weight.data[10 + d, 3*dh + d] = 1.0
    
    # Output head
    for d in range(10):
        model.head.weight.data[d, d] = 10.0
        model.head.weight.data[d, 10 + d] = 10.0


model_shorthand_name = "v27_nearest_neighbor_v2"
model_description = "Nearest-neighbor with MLP-based soft routing"


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
    
    # Debug: print vocab info
    if args.verbose:
        print(f"vocab_size={model.vocab_size}, max_seq_len={model.max_seq_len}")
        print(f"d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}, d_ff={model.d_ff}")
        print(f"n_params={n_params}")

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
