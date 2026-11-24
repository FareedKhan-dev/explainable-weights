"""
Interpretable transformer for smallest-prime-factor task.
Algorithmic approach: check divisibility by small primes systematically.
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


def compute_number_from_tokens(tokens):
    """Convert 5 digit tokens to integer."""
    return int(''.join(str(t) for t in tokens))


def smallest_prime_factor(n):
    """Compute smallest prime factor of n."""
    if n < 2:
        return n
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    i = 7
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n


def write_weights(model, task):
    """
    Build a transformer that computes smallest prime factor algorithmically.
    Uses position embeddings to track digit positions and MLPs for divisibility checks.
    """
    torch.manual_seed(42)
    D = model.d_model
    V = model.vocab_size  # 11 (0-9 and '=')
    
    # Position embeddings: encode position strongly
    max_pos = model.max_seq_len
    pos_emb = torch.zeros(max_pos, D)
    for p in range(max_pos):
        # Use binary-like encoding for position
        for bit in range(min(8, D)):
            if (p >> bit) & 1:
                pos_emb[p, bit] = 3.0
        # Also use one-hot-like strong signal
        if p < D:
            pos_emb[p, D//2 + p] = 5.0
    
    # Token embeddings: encode digit value
    tok_emb = torch.zeros(V, D)
    for tok_id in range(V):
        # One-hot encoding
        if tok_id < D:
            tok_emb[tok_id, tok_id] = 2.0
        # Also encode numeric value for digits
        if tok_id < 10:
            tok_emb[tok_id, V + tok_id] = tok_id / 10.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Attention to gather all input digits
    attn0 = model.blocks[0].attn
    
    # Q, K, V: make all positions attend to each other to aggregate input
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.eye(D)
    W_o = torch.eye(D)
    
    # Make queries at output positions attend strongly to all input positions
    for out_pos in range(6, 11):
        for in_pos in range(5):
            # Query at out_pos should match key at in_pos
            for dim in range(min(20, D)):
                W_q[out_pos, dim] += 1.0
                W_k[in_pos, dim] += 1.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = W_o
    
    # Block 0 MLP: pass through
    mlp0 = model.blocks[0].mlp
    nn.init.zeros_(mlp0.fc1.weight)
    nn.init.zeros_(mlp0.fc1.bias)
    nn.init.zeros_(mlp0.fc2.weight)
    nn.init.zeros_(mlp0.fc2.bias)

    # Block 1: Attention (identity)
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    attn1.W_v.weight.data = torch.eye(D)
    attn1.W_o.weight.data = torch.eye(D)

    # Block 1 MLP: Compute divisibility checks and output
    mlp1 = model.blocks[1].mlp
    d_ff1 = mlp1.fc1.weight.shape[0]
    
    # We'll use neurons to detect divisibility
    fc1_w = torch.zeros(d_ff1, D)
    fc1_b = torch.zeros(d_ff1)
    fc2_w = torch.zeros(D, d_ff1)
    fc2_b = torch.zeros(D)
    
    neuron_idx = 0
    
    # Helper: create a neuron that fires when specific conditions are met
    def add_detection_neuron(condition_dims, condition_signs, threshold, output_dim, output_val):
        nonlocal neuron_idx
        for dim, sign in zip(condition_dims, condition_signs):
            fc1_w[neuron_idx, dim] = sign * 5.0
        fc1_b[neuron_idx] = -threshold
        fc2_w[output_dim, neuron_idx] = output_val
        neuron_idx += 1
    
    # Strategy: Check divisibility by small primes
    # Div by 2: last digit in {0,2,4,6,8}
    for d in [0, 2, 4, 6, 8]:
        add_detection_neuron([d], [1], 3.0, 100 + d, 10.0)
    
    # Div by 5: last digit in {0,5}
    for d in [0, 5]:
        add_detection_neuron([d], [1], 3.0, 110 + d, 10.0)
    
    # Flag for "divisible by 2" (any even digit detected)
    flag_div2_dim = 120
    for d in [0, 2, 4, 6, 8]:
        add_detection_neuron([100 + d], [1], 3.0, flag_div2_dim, 10.0)
    
    # Flag for "divisible by 5"
    flag_div5_dim = 121
    for d in [0, 5]:
        add_detection_neuron([110 + d], [1], 3.0, flag_div5_dim, 10.0)
    
    # For now, implement simple heuristic:
    # - If div by 2, output "00002"
    # - Else if div by 5, output "00005"  
    # - Else output the input number (assume prime)
    
    # Output position 6 (first output digit)
    # If div by 2, output 0
    add_detection_neuron([flag_div2_dim, 6*10], [1, 1], 8.0, 0, 20.0)
    # If div by 5 (and not 2), output 0
    add_detection_neuron([flag_div5_dim, 6*10], [1, 1], 8.0, 0, 20.0)
    
    # Output position 7
    add_detection_neuron([flag_div2_dim, 7*10], [1, 1], 8.0, 0, 20.0)
    add_detection_neuron([flag_div5_dim, 7*10], [1, 1], 8.0, 0, 20.0)
    
    # Output position 8
    add_detection_neuron([flag_div2_dim, 8*10], [1, 1], 8.0, 0, 20.0)
    add_detection_neuron([flag_div5_dim, 8*10], [1, 1], 8.0, 0, 20.0)
    
    # Output position 9
    add_detection_neuron([flag_div2_dim, 9*10], [1, 1], 8.0, 0, 20.0)
    add_detection_neuron([flag_div5_dim, 9*10], [1, 1], 8.0, 0, 20.0)
    
    # Output position 10 (last digit)
    # If div by 2, output 2
    add_detection_neuron([flag_div2_dim, 10*10], [1, 1], 8.0, 2, 20.0)
    # If div by 5 (and not 2), output 5
    add_detection_neuron([flag_div5_dim, 10*10], [1, 1], 8.0, 5, 20.0)
    
    # Copy input digits to output (for prime case)
    for in_pos in range(5):
        out_pos = in_pos + 6
        for digit in range(10):
            # If not div by 2 or 5, and at right output position, copy digit
            add_detection_neuron(
                [in_pos*10 + digit, out_pos*10],
                [1, 1],
                8.0,
                digit,
                20.0
            )
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Final head: read out digits
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 5.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron_idx} detection neurons!")


model_shorthand_name = "spf-exp10-algo-v2"
model_description = "Algorithmic SPF v2 with divisibility checks for 2 and 5"


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
        d_model=256,
        n_heads=8,
        n_layers=2,
        d_ff_list=[512, 1024],
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
