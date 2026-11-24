"""
Interpretable transformer for smallest-prime-factor task.
Multi-head approach: each head routes info from one input position.
Large position embeddings to dominate LayerNorm.
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
    Strategy: Use 5 heads (with d_model=160, n_heads=5, d_head=32).
    Head h at ALL positions attends to input position h.
    Head output contains digit info from that position.
    MLP reads from head outputs to detect divisibility.
    """
    torch.manual_seed(42)
    D = model.d_model  # 160
    V = model.vocab_size  # 11
    H = model.n_heads  # 5
    dh = D // H  # 32
    
    # Position embeddings: VERY large, orthogonal, in first 80 dims
    pos_emb = torch.zeros(model.max_seq_len, D)
    SCALE = 50.0
    for p in range(min(11, model.max_seq_len)):
        # One-hot position encoding with large scale
        pos_emb[p, p * 10:(p+1)*10] = SCALE
    
    # Token embeddings: smaller scale, in dimensions 100-110
    tok_emb = torch.zeros(V, D)
    for i in range(V):
        tok_emb[i, 100 + i] = 5.0  # One-hot token encoding
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Position-selective attention with 5 heads
    attn0 = model.blocks[0].attn
    
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)
    
    # Head h: query from position 5 looks for position h's encoding
    # After LayerNorm, the large position signal dominates direction
    # So query at position 5 (which has pos_emb[5] = large in dims 50-59)
    # needs to match key at position h (which has pos_emb[h] = large in dims h*10 to h*10+9)
    
    for h_idx in range(H):
        head_start = h_idx * dh  # Start dimension for this head
        
        # Query: at position 5, the dominant signal is in dims 50-59
        # Map dims 50-59 to head's query subspace
        for i in range(min(10, dh)):
            W_q[head_start + i, 50 + i] = 3.0  # Read position 5's encoding
        
        # Key: at position h_idx, the dominant signal is in dims h_idx*10 to h_idx*10+9
        # Map those dims to head's key subspace  
        for i in range(min(10, dh)):
            W_k[head_start + i, h_idx * 10 + i] = 3.0  # Match position h_idx's encoding
    
    # Value: extract token identity from dimensions 100-110
    for i in range(V):
        W_v[i, 100 + i] = 1.0  # Copy token one-hot to output dims 0-10
    
    # Output projection: route each head's output to specific dimension range
    for h_idx in range(H):
        head_start = h_idx * dh
        for i in range(min(11, dh)):
            W_o[h_idx * 20 + i, head_start + i] = 1.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = W_o
    
    # Block 0 MLP: Pass through
    mlp0 = model.blocks[0].mlp
    nn.init.zeros_(mlp0.fc1.weight)
    nn.init.zeros_(mlp0.fc1.bias)
    nn.init.zeros_(mlp0.fc2.weight)
    nn.init.zeros_(mlp0.fc2.bias)

    # Block 1: Identity attention (just aggregate)
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    nn.init.zeros_(attn1.W_v.weight)
    nn.init.zeros_(attn1.W_o.weight)

    # Block 1 MLP: Detect divisibility using head outputs
    mlp1 = model.blocks[1].mlp
    d_ff1 = mlp1.fc1.weight.shape[0]
    
    fc1_w = torch.zeros(d_ff1, D)
    fc1_b = torch.zeros(d_ff1)
    fc2_w = torch.zeros(D, d_ff1)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # After Block 0 attention + output projection, position 5 has:
    # - dims 0-19: digit from position 0 (head 0)
    # - dims 20-39: digit from position 1 (head 1)
    # - dims 40-59: digit from position 2 (head 2)
    # - dims 60-79: digit from position 3 (head 3)
    # - dims 80-99: digit from position 4 (head 4)
    # The digit identity is encoded in the first 11 dims of each head's output
    
    # Head 4 (position 4 = last digit) output is in dims 80-99
    # Token encoding of digit d is at dim 80+d (if d < 11)
    
    # Check divisibility by 2: last digit even
    flag_div2 = 120
    for digit in [0, 2, 4, 6, 8]:
        fc1_w[neuron, 80 + digit] = 10.0  # Head 4 output, digit d
        fc1_b[neuron] = -3.0
        fc2_w[flag_div2, neuron] = 15.0
        neuron += 1
    
    # Check divisibility by 5: last digit 0 or 5
    flag_div5 = 121
    for digit in [0, 5]:
        fc1_w[neuron, 80 + digit] = 10.0
        fc1_b[neuron] = -3.0
        fc2_w[flag_div5, neuron] = 15.0
        neuron += 1
    
    # Output "00002" if divisible by 2 (at ALL output-generating positions)
    for out_digit_idx in range(5):
        out_pos = 5 + out_digit_idx  # positions 5-9 generate predictions for 6-10
        target_digit = 0 if out_digit_idx < 4 else 2
        
        fc1_w[neuron, flag_div2] = 10.0
        fc1_w[neuron, out_pos * 10] = 5.0  # position signal (after LayerNorm, this is from pos_emb)
        fc1_b[neuron] = -12.0
        fc2_w[target_digit, neuron] = 30.0
        neuron += 1
    
    # Output "00005" if divisible by 5 (and not 2)
    for out_digit_idx in range(5):
        out_pos = 5 + out_digit_idx
        target_digit = 0 if out_digit_idx < 4 else 5
        
        fc1_w[neuron, flag_div5] = 10.0
        fc1_w[neuron, flag_div2] = -10.0
        fc1_w[neuron, out_pos * 10] = 5.0
        fc1_b[neuron] = -12.0
        fc2_w[target_digit, neuron] = 30.0
        neuron += 1
    
    # Default: output 1 (smallest odd prime) - wrong but a fallback
    for out_digit_idx in range(5):
        out_pos = 5 + out_digit_idx
        fc1_w[neuron, flag_div2] = -5.0
        fc1_w[neuron, flag_div5] = -5.0
        fc1_w[neuron, out_pos * 10] = 5.0
        fc1_b[neuron] = -8.0
        fc2_w[1, neuron] = 20.0  # output digit 1 as fallback
        neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Final head
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 15.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons!")


model_shorthand_name = "spf-exp18-multihead-route"
model_description = "Multi-head routing: each head routes from one input position"


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
