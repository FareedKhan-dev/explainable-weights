"""
Interpretable transformer for smallest-prime-factor task.
Direct approach: use attention to copy last digit, then check divisibility.
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
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


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
    def __init__(self, vocab_size, max_seq_len=32, d_model=128, n_heads=8, n_layers=2, d_ff_list=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        if d_ff_list is None:
            d_ff_list = [256] * n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff_list[i]) for i in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    """
    Check divisibility by 2 and 5 using the last digit (position 4).
    
    Strategy:
    - Embed last digit token in dims 0-10 at position 4
    - Use attention to copy position 4's token to a "last digit" flag dimension
    - Use MLP to check if last digit is even (→00002) or 5 (→00005)
    - Output at positions 5-9
    """
    D = model.d_model
    V = model.vocab_size
    
    # Token embeddings: one-hot in dims 0-10
    tok_emb = torch.zeros(V, D)
    for i in range(V):
        tok_emb[i, i] = 50.0
    model.token_emb.weight.data = tok_emb
    
    # Position embeddings: one-hot in dims 20-30
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(min(11, model.max_seq_len)):
        pos_emb[p, 20 + p] = 50.0
    model.pos_emb.weight.data = pos_emb
    
    # === Block 0 Attention ===
    # Copy token from position 4 to dims 40-50 at all positions.
    # Use uniform attention (all positions attend equally).
    
    attn0 = model.blocks[0].attn
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)
    
    # Uniform attention: Q and K both read a constant (dim 31)
    # Set all tokens to have value 50 in dim 31
    tok_emb[:, 31] = 50.0
    model.token_emb.weight.data = tok_emb
    
    W_q[0, 31] = 1.0  # Q reads dim 31 → head dim 0
    W_k[0, 31] = 1.0  # K reads dim 31 → head dim 0
    # This makes all Q·K scores equal → uniform attention
    
    # V copies token identity from dims 0-10
    for d in range(V):
        W_v[d, d] = 1.0
    
    # O writes to dims 40-50 (last digit copy)
    for d in range(V):
        W_o[40 + d, d] = 1.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = W_o
    
    # After uniform attention, dims 40-50 contain the AVERAGE of all tokens.
    # We need to extract just position 4's token.
    
    # Alternative: use position 4 specifically
    # Make position 4 have a unique marker in dim 31
    pos_emb[4, 31] = 100.0  # Large value at position 4
    model.pos_emb.weight.data = pos_emb
    
    # Now Q reads dim 31 (large at position 4), K reads dim 31
    # At position 4: Q·K = 100*100 = 10000 (high)
    # At other positions: Q·K = 50*50 = 2500 (low)
    # So attention will heavily favor position 4
    
    # Zero Block 0 MLP
    mlp0 = model.blocks[0].mlp
    nn.init.zeros_(mlp0.fc1.weight)
    nn.init.zeros_(mlp0.fc1.bias)
    nn.init.zeros_(mlp0.fc2.weight)
    nn.init.zeros_(mlp0.fc2.bias)
    
    # === Block 1 MLP ===
    # Check divisibility and output the right digit at positions 5-9.
    
    mlp1 = model.blocks[1].mlp
    fc1_w = torch.zeros(512, D)
    fc1_b = torch.zeros(512)
    fc2_w = torch.zeros(D, 512)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # Divisible by 2: last digit (in dims 40-50) is even
    # Check dims 40+0, 40+2, 40+4, 40+6, 40+8
    # Output "00002" at positions 5-9
    
    for out_pos in [5, 6, 7, 8]:
        for digit in [0, 2, 4, 6, 8]:
            fc1_w[neuron, 40 + digit] = 5.0
            fc1_w[neuron, 20 + out_pos] = 5.0
            fc1_b[neuron] = -7.5
            fc2_w[0, neuron] = 50.0
            neuron += 1
    
    for digit in [0, 2, 4, 6, 8]:
        fc1_w[neuron, 40 + digit] = 5.0
        fc1_w[neuron, 20 + 9] = 5.0
        fc1_b[neuron] = -7.5
        fc2_w[2, neuron] = 50.0
        neuron += 1
    
    # Divisible by 5 (not by 2): last digit is 5
    for out_pos in [5, 6, 7, 8]:
        fc1_w[neuron, 40 + 5] = 5.0
        fc1_w[neuron, 20 + out_pos] = 5.0
        fc1_b[neuron] = -7.5
        fc2_w[0, neuron] = 50.0
        neuron += 1
    
    fc1_w[neuron, 40 + 5] = 5.0
    fc1_w[neuron, 20 + 9] = 5.0
    fc1_b[neuron] = -7.5
    fc2_w[5, neuron] = 50.0
    neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Zero Block 1 attention
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    nn.init.zeros_(attn1.W_v.weight)
    nn.init.zeros_(attn1.W_o.weight)
    
    # Final head: read from dims 0-10
    head_w = torch.zeros(V, D)
    for d in range(min(V, 11)):
        head_w[d, d] = 10.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons!")


model_shorthand_name = "spf-exp24-div2-5-fix"
model_description = "Use position 4 marker to attend to last digit, check div by 2 and 5"


def upsert_overall_results(rows, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])


def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=160,
        n_heads=5,
        n_layers=2,
        d_ff_list=[512, 512],
    )
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
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
        "task": args.task,
        "accuracy": f"{accuracy:.4f}",
        "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
