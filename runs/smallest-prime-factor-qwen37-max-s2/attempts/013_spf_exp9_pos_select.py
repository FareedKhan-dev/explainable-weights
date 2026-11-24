"""
Interpretable transformer for smallest-prime-factor task.
Improved algorithmic approach with proper position-selective attention.
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
    Use strong position embeddings to make attention position-selective.
    Each head at output position attends to its corresponding input position.
    """
    torch.manual_seed(42)
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H

    # Large position embeddings to dominate attention
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        # Each position gets a unique large-magnitude pattern
        pos_emb[p, p * 10:(p+1) * 10] = 5.0
    
    # Token embeddings: smaller magnitude
    tok_emb = torch.zeros(V, D)
    for d in range(V):
        tok_emb[d, d] = 1.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Position-selective attention
    attn0 = model.blocks[0].attn
    
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    
    # At output positions 6-10, each head h attends to input position h
    for out_idx in range(5):  # output positions 6-10
        for h in range(1):  # use only first head per output
            head_base = out_idx * dh
            
            # Query at output pos 6+out_idx should match key at input pos out_idx
            # Position encoding: pos p uses dims p*10 to (p+1)*10
            q_pos_base = (6 + out_idx) * 10  # output position encoding
            k_pos_base = out_idx * 10  # input position encoding
            
            # Make query match key at target position
            for i in range(10):
                W_q[head_base, q_pos_base + i] = 3.0
                W_k[head_base, k_pos_base + i] = 3.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    
    # V: pass token identity
    W_v = torch.zeros(D, D)
    for i in range(V):
        W_v[i, i] = 1.0
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = torch.eye(D)
    
    # Block 0 MLP: Simple pass-through, let block 1 do the work
    mlp1 = model.blocks[0].mlp
    nn.init.zeros_(mlp1.fc1.weight)
    nn.init.zeros_(mlp1.fc1.bias)
    nn.init.zeros_(mlp1.fc2.weight)
    nn.init.zeros_(mlp1.fc2.bias)
    
    # Block 1: Use attention to aggregate, then MLP for SPF logic
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    attn1.W_v.weight.data = torch.eye(D)
    attn1.W_o.weight.data = torch.eye(D)
    
    mlp2 = model.blocks[1].mlp
    d_ff2 = mlp2.fc1.weight.shape[0]
    
    fc1_2_w = torch.zeros(d_ff2, D)
    fc1_2_b = torch.zeros(d_ff2)
    fc2_2_w = torch.zeros(D, d_ff2)
    fc2_2_b = torch.zeros(D)
    
    neuron = 0
    
    # Detect divisibility by 2 (last digit even)
    for d in [0, 2, 4, 6, 8]:
        fc1_2_w[neuron, d] = 5.0  # digit from head 4 (last position)
        fc1_2_b[neuron] = -3.0
        fc2_2_w[V, neuron] = 10.0  # flag in dim V
        neuron += 1
    
    # Detect divisibility by 5 (last digit 0 or 5)
    for d in [0, 5]:
        fc1_2_w[neuron, d] = 5.0
        fc1_2_b[neuron] = -3.0
        fc2_2_w[V+1, neuron] = 10.0  # flag in dim V+1
        neuron += 1
    
    # Output "00002" if div by 2
    for out_pos in range(5):
        digit = 2 if out_pos == 4 else 0
        fc1_2_w[neuron, V] = 5.0
        fc1_2_w[neuron, (6 + out_pos) * 10] = 5.0
        fc1_2_b[neuron] = -7.0
        fc2_2_w[digit, neuron] = 20.0
        neuron += 1
    
    # Output "00005" if div by 5 (and not by 2)
    for out_pos in range(5):
        digit = 5 if out_pos == 4 else 0
        fc1_2_w[neuron, V+1] = 5.0
        fc1_2_w[neuron, V] = -5.0  # NOT div by 2
        fc1_2_w[neuron, (6 + out_pos) * 10] = 5.0
        fc1_2_b[neuron] = -7.0
        fc2_2_w[digit, neuron] = 20.0
        neuron += 1
    
    # Copy input for primes (not div by 2 or 5)
    for in_pos in range(5):
        for d in range(10):
            fc1_2_w[neuron, in_pos * 10 + d] = 5.0
            fc1_2_w[neuron, V] = -5.0
            fc1_2_w[neuron, V+1] = -5.0
            fc1_2_w[neuron, (6 + in_pos) * 10] = 5.0
            fc1_2_b[neuron] = -15.0
            fc2_2_w[d, neuron] = 20.0
            neuron += 1
    
    mlp2.fc1.weight.data = fc1_2_w
    mlp2.fc1.bias.data = fc1_2_b
    mlp2.fc2.weight.data = fc2_2_w
    mlp2.fc2.bias.data = fc2_2_b
    
    # Head
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 5.0
    model.head.weight.data = head_w
    
    print("Weights initialized (position-selective attention)!")


model_shorthand_name = "spf-exp9-pos-select"
model_description = "Position-selective attention with divisibility rules"


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
        d_model=128,
        n_heads=8,
        n_layers=2,
        d_ff_list=[256, 512],
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
