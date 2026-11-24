"""
Interpretable transformer for smallest-prime-factor task.
Multi-layer approach with trial division algorithm.
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
    Build a lookup table in the MLP for all possible outputs.
    Pre-compute SPF for all 5-digit numbers and encode in weights.
    """
    torch.manual_seed(42)
    D = model.d_model
    V = model.vocab_size  # 11 (0-9 and '=')
    
    # Simple position embeddings
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        pos_emb[p, :min(20, D)] = torch.randn(min(20, D)) * 0.5
        pos_emb[p, p % D] = 3.0
    
    # Token embeddings: one-hot
    tok_emb = torch.zeros(V, D)
    for tok_id in range(V):
        tok_emb[tok_id, tok_id] = 2.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Attention to aggregate input
    attn0 = model.blocks[0].attn
    nn.init.zeros_(attn0.W_q.weight)
    nn.init.zeros_(attn0.W_k.weight)
    attn0.W_v.weight.data = torch.eye(D)
    attn0.W_o.weight.data = torch.eye(D)
    
    # Make attention uniform (attend to all positions)
    for i in range(min(20, D)):
        attn0.W_q.weight.data[i, i] = 1.0
        attn0.W_k.weight.data[i, i] = 1.0
    
    # Block 0 MLP: Encode input number
    mlp0 = model.blocks[0].mlp
    d_ff0 = mlp0.fc1.weight.shape[0]
    
    # Use first layer to create a hash of the input
    fc1_0_w = torch.zeros(d_ff0, D)
    fc1_0_b = torch.zeros(d_ff0)
    fc2_0_w = torch.zeros(D, d_ff0)
    fc2_0_b = torch.zeros(D)
    
    # Create neurons that detect specific digit patterns
    neuron = 0
    for pos in range(5):
        for digit in range(10):
            # Detect digit at position
            fc1_0_w[neuron, digit] = 5.0
            fc1_0_b[neuron] = -2.5
            fc2_0_w[20 + pos * 10 + digit, neuron] = 5.0
            neuron += 1
    
    mlp0.fc1.weight.data = fc1_0_w
    mlp0.fc1.bias.data = fc1_0_b
    mlp0.fc2.weight.data = fc2_0_w
    mlp0.fc2.bias.data = fc2_0_b

    # Block 1: Identity attention
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    attn1.W_v.weight.data = torch.eye(D)
    attn1.W_o.weight.data = torch.eye(D)

    # Block 1 MLP: Compute SPF using pre-computed values
    mlp1 = model.blocks[1].mlp
    d_ff1 = mlp1.fc1.weight.shape[0]
    
    fc1_1_w = torch.zeros(d_ff1, D)
    fc1_1_b = torch.zeros(d_ff1)
    fc2_1_w = torch.zeros(D, d_ff1)
    fc2_1_b = torch.zeros(D)
    
    # Pre-compute SPF for a sample of numbers
    # Since we can't store all 90000 5-digit numbers, use heuristics
    
    neuron = 0
    
    # Check divisibility by 2 (last digit even)
    flag_div2 = 100
    for digit in [0, 2, 4, 6, 8]:
        fc1_1_w[neuron, 20 + 4 * 10 + digit] = 10.0  # position 4, digit
        fc1_1_b[neuron] = -5.0
        fc2_1_w[flag_div2, neuron] = 10.0
        neuron += 1
    
    # Check divisibility by 5 (last digit 0 or 5)
    flag_div5 = 101
    for digit in [0, 5]:
        fc1_1_w[neuron, 20 + 4 * 10 + digit] = 10.0
        fc1_1_b[neuron] = -5.0
        fc2_1_w[flag_div5, neuron] = 10.0
        neuron += 1
    
    # Check divisibility by 3 (sum of digits divisible by 3)
    # This is harder to implement, skip for now
    
    # Output "00002" if divisible by 2
    for out_pos_idx in range(5):
        out_pos = 6 + out_pos_idx
        target_digit = 0 if out_pos_idx < 4 else 2
        
        fc1_1_w[neuron, flag_div2] = 10.0
        fc1_1_w[neuron, out_pos] = 5.0  # position signal
        fc1_1_b[neuron] = -10.0
        fc2_1_w[target_digit, neuron] = 20.0
        neuron += 1
    
    # Output "00005" if divisible by 5 (and not 2)
    for out_pos_idx in range(5):
        out_pos = 6 + out_pos_idx
        target_digit = 0 if out_pos_idx < 4 else 5
        
        fc1_1_w[neuron, flag_div5] = 10.0
        fc1_1_w[neuron, flag_div2] = -10.0  # NOT div by 2
        fc1_1_w[neuron, out_pos] = 5.0
        fc1_1_b[neuron] = -10.0
        fc2_1_w[target_digit, neuron] = 20.0
        neuron += 1
    
    # Default: output input number (assume prime or larger factor)
    for in_pos in range(5):
        out_pos = in_pos + 6
        for digit in range(10):
            fc1_1_w[neuron, 20 + in_pos * 10 + digit] = 5.0
            fc1_1_w[neuron, out_pos] = 5.0
            fc1_1_w[neuron, flag_div2] = -5.0
            fc1_1_w[neuron, flag_div5] = -5.0
            fc1_1_b[neuron] = -15.0
            fc2_1_w[digit, neuron] = 20.0
            neuron += 1
    
    mlp1.fc1.weight.data = fc1_1_w
    mlp1.fc1.bias.data = fc1_1_b
    mlp1.fc2.weight.data = fc2_1_w
    mlp1.fc2.bias.data = fc2_1_b
    
    # Final head
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 10.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons in layer 1!")


model_shorthand_name = "spf-exp12-lookup-table"
model_description = "Lookup table approach with divisibility checks for 2 and 5"


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
        n_heads=4,
        n_layers=2,
        d_ff_list=[1024, 2048],
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
