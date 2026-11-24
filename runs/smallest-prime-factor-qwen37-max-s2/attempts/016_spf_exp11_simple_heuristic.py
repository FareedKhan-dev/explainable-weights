"""
Interpretable transformer for smallest-prime-factor task.
Simplified approach: focus on last digit to determine divisibility.
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
    Simple heuristic: check last digit for divisibility by 2 or 5.
    Output positions attend to input position 4 (last digit).
    """
    torch.manual_seed(42)
    D = model.d_model
    V = model.vocab_size  # 11 (0-9 and '=')
    H = model.n_heads
    
    # Position embeddings: strong position signal
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        # Strong one-hot position encoding
        if p < D // 2:
            pos_emb[p, p] = 10.0
    
    # Token embeddings: encode digit value
    tok_emb = torch.zeros(V, D)
    for tok_id in range(V):
        # One-hot in second half
        if tok_id < D // 2:
            tok_emb[tok_id, D // 2 + tok_id] = 5.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Make output positions attend to input position 4 (last digit)
    attn0 = model.blocks[0].attn
    dh = D // H
    
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)
    
    # For each output position (6-10), make it attend to position 4
    # Use first head for this
    for out_pos in range(6, 11):
        # Query at output position
        W_q[out_pos, 0:20] = 5.0  # Strong query signal
        # Key at position 4 (last input digit)
        W_k[4, 0:20] = 5.0  # Match query
    
    # Value: pass through token embedding
    for i in range(min(V, D)):
        W_v[D // 2 + i, D // 2 + i] = 1.0
    
    # Output projection: pass through
    W_o = torch.eye(D)
    
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

    # Block 1: Identity attention
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    attn1.W_v.weight.data = torch.eye(D)
    attn1.W_o.weight.data = torch.eye(D)

    # Block 1 MLP: Detect divisibility and output
    mlp1 = model.blocks[1].mlp
    d_ff1 = mlp1.fc1.weight.shape[0]
    
    fc1_w = torch.zeros(d_ff1, D)
    fc1_b = torch.zeros(d_ff1)
    fc2_w = torch.zeros(D, d_ff1)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # After attention, output positions have information about last digit
    # The token embedding is in dimensions D//2 to D//2+V-1
    
    # Detect even last digit (div by 2): digits 0,2,4,6,8
    flag_even = 200  # dimension for "divisible by 2" flag
    for digit in [0, 2, 4, 6, 8]:
        # Neuron fires if last digit is 'digit'
        fc1_w[neuron, D // 2 + digit] = 10.0
        fc1_b[neuron] = -5.0
        fc2_w[flag_even, neuron] = 10.0
        neuron += 1
    
    # Detect last digit 0 or 5 (div by 5)
    flag_div5 = 201
    for digit in [0, 5]:
        fc1_w[neuron, D // 2 + digit] = 10.0
        fc1_b[neuron] = -5.0
        fc2_w[flag_div5, neuron] = 10.0
        neuron += 1
    
    # Output digits based on flags
    # For output positions 6-10, check if we're at that position and have a flag
    
    # If divisible by 2, output "00002"
    for out_pos_idx in range(5):  # 0-4 for positions 6-10
        out_pos = 6 + out_pos_idx
        # Check if we're at this output position
        fc1_w[neuron, flag_even] = 10.0
        fc1_w[neuron, out_pos] = 10.0  # position signal
        fc1_b[neuron] = -15.0
        
        # Output digit
        if out_pos_idx < 4:
            target_digit = 0
        else:
            target_digit = 2
        fc2_w[target_digit, neuron] = 20.0
        neuron += 1
    
    # If divisible by 5 (and not 2), output "00005"
    for out_pos_idx in range(5):
        out_pos = 6 + out_pos_idx
        fc1_w[neuron, flag_div5] = 10.0
        fc1_w[neuron, flag_even] = -10.0  # NOT divisible by 2
        fc1_w[neuron, out_pos] = 10.0
        fc1_b[neuron] = -15.0
        
        if out_pos_idx < 4:
            target_digit = 0
        else:
            target_digit = 5
        fc2_w[target_digit, neuron] = 20.0
        neuron += 1
    
    # Default: output input number (assume prime)
    # Copy input digits to output positions
    for in_pos in range(5):
        out_pos = in_pos + 6
        for digit in range(10):
            # If not div by 2 or 5, copy input digit to output
            fc1_w[neuron, D // 2 + digit] = 5.0  # input digit
            fc1_w[neuron, in_pos] = 5.0  # input position
            fc1_w[neuron, out_pos] = 5.0  # output position
            fc1_w[neuron, flag_even] = -5.0  # NOT div by 2
            fc1_w[neuron, flag_div5] = -5.0  # NOT div by 5
            fc1_b[neuron] = -20.0
            fc2_w[digit, neuron] = 20.0
            neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Final head: read out digits
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 10.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons!")


model_shorthand_name = "spf-exp11-simple-heuristic"
model_description = "Simple heuristic: check last digit for div by 2 or 5"


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
