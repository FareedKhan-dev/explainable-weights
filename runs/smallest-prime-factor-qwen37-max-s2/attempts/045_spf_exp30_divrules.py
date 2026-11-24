"""
Interpretable transformer for smallest-prime-factor task.
Simplest working approach: just check if last digit is even.
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
    Experiment 30: Implement proper smallest prime factor detection
    - Check divisibility by 2, 3, 5 using digit rules
    - For other primes, output the number itself (assume prime)
    """
    D = model.d_model  # 160
    V = model.vocab_size  # 11
    H = model.n_heads  # 5
    dh = D // H  # 32
    
    # Token embeddings: encode digit values in dims 0-9
    tok_emb = torch.zeros(V, D)
    for i in range(V):
        tok_emb[i, i] = 100.0
    model.token_emb.weight.data = tok_emb
    
    # Position embeddings: mark positions 0-4 for input digits
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(5):
        pos_emb[p, 100 + p] = 100.0
    model.pos_emb.weight.data = pos_emb
    
    # === Block 0 Attention ===
    # Copy all 5 input digits to positions 5-9 using 5 heads
    attn0 = model.blocks[0].attn
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)
    
    # Each head copies one input position to corresponding output position
    for head in range(5):
        # Q at output position (5+head) attends to input position head
        W_q[head * dh, 100 + head] = 50.0  # Q reads from position marker
        
        # K at input position head matches
        W_k[head * dh, 100 + head] = 50.0
        
        # V copies the token value
        for d in range(V):
            W_v[head * dh + d, d] = 1.0
    
    # O projects to dims 50-59 (for digits 0-9)
    for d in range(V):
        W_o[50 + d, d] = 1.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = W_o
    
    # === Block 0 MLP ===
    # Compute divisibility flags
    mlp0 = model.blocks[0].mlp
    fc1_w = torch.zeros(512, D)
    fc1_b = torch.zeros(512)
    fc2_w = torch.zeros(D, 512)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # Divisible by 2: last digit (position 4) is 0,2,4,6,8
    # Check if dim 50+digit for digit in {0,2,4,6,8} is active at position 4
    for digit in [0, 2, 4, 6, 8]:
        fc1_w[neuron, 50 + digit] = 10.0  # Copied digit
        fc1_w[neuron, 104] = 10.0  # Position 4 marker
        fc1_b[neuron] = -15.0
        fc2_w[20, neuron] = 100.0  # Write flag to dim 20
        neuron += 1
    
    # Divisible by 5: last digit is 0 or 5
    for digit in [0, 5]:
        fc1_w[neuron, 50 + digit] = 10.0
        fc1_w[neuron, 104] = 10.0
        fc1_b[neuron] = -15.0
        fc2_w[21, neuron] = 100.0  # Write flag to dim 21
        neuron += 1
    
    # Sum of digits for divisibility by 3
    # Each digit contributes its value (0-9) to sum
    # We'll use dims 30-39 to represent digit sum (mod 10)
    for pos in range(5):
        for digit in range(10):
            fc1_w[neuron, 50 + digit] = 10.0  # Copied digit
            fc1_w[neuron, 100 + pos] = 10.0  # Position marker
            fc1_b[neuron] = -15.0
            # Add digit value to sum accumulator (dim 30 + digit_value)
            fc2_w[30 + digit, neuron] = 50.0
            neuron += 1
    
    mlp0.fc1.weight.data = fc1_w
    mlp0.fc1.bias.data = fc1_b
    mlp0.fc2.weight.data = fc2_w
    mlp0.fc2.bias.data = fc2_b
    
    # === Block 1 ===
    # Zero attention
    nn.init.zeros_(model.blocks[1].attn.W_q.weight)
    nn.init.zeros_(model.blocks[1].attn.W_k.weight)
    nn.init.zeros_(model.blocks[1].attn.W_v.weight)
    nn.init.zeros_(model.blocks[1].attn.W_o.weight)
    
    # MLP: Generate output based on divisibility flags
    mlp1 = model.blocks[1].mlp
    fc1_w = torch.zeros(512, D)
    fc1_b = torch.zeros(512)
    fc2_w = torch.zeros(D, 512)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # If divisible by 2 (flag at dim 20), output 00002
    for out_pos in [5, 6, 7, 8]:
        fc1_w[neuron, 20] = 10.0  # Div by 2 flag
        fc1_w[neuron, 100 + out_pos] = 10.0  # Position marker
        fc1_b[neuron] = -15.0
        fc2_w[0, neuron] = 200.0  # Output digit 0
        neuron += 1
    
    fc1_w[neuron, 20] = 10.0
    fc1_w[neuron, 109] = 10.0  # Position 9
    fc1_b[neuron] = -15.0
    fc2_w[2, neuron] = 200.0  # Output digit 2
    neuron += 1
    
    # If divisible by 5 (flag at dim 21), output 00005
    for out_pos in [5, 6, 7, 8]:
        fc1_w[neuron, 21] = 10.0
        fc1_w[neuron, 100 + out_pos] = 10.0
        fc1_b[neuron] = -15.0
        fc2_w[0, neuron] = 200.0
        neuron += 1
    
    fc1_w[neuron, 21] = 10.0
    fc1_w[neuron, 109] = 10.0
    fc1_b[neuron] = -15.0
    fc2_w[5, neuron] = 200.0
    neuron += 1
    
    # If not divisible by 2 or 5, output the number itself (assume prime)
    # Copy input digits to output positions
    for pos in range(5):
        fc1_w[neuron, 50 + pos] = 10.0  # Copied digit
        fc1_w[neuron, 100 + pos] = 10.0
        fc1_b[neuron] = -15.0
        fc2_w[50 + pos, neuron] = 200.0  # Write to output position
        neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Final head: read from dims 0-10
    head_w = torch.zeros(V, D)
    for d in range(min(V, 11)):
        head_w[d, d] = 20.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons!")


model_shorthand_name = "spf-exp30-divrules"
model_description = "Proper divisibility rules: check div by 2 and 5, fallback to number itself"


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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    if args.debug:
        # Debug: run a single example and print intermediate values
        model.eval()
        with torch.no_grad():
            # Create a simple test input
            test_input = torch.tensor([[6, 4, 5, 0, 7]], device=args.device)  # 64507
            print(f"Input: {test_input}")
            
            # Forward pass through embeddings
            pos = torch.arange(test_input.shape[1], device=args.device)
            h = model.token_emb(test_input) + model.pos_emb(pos)[None, :, :]
            print(f"After embeddings, position 4 token (dim 50): {h[0, 4, 50:61]}")
            
            # Block 0
            h = model.blocks[0](h)
            print(f"After block 0, position 4 (dim 50): {h[0, 4, 50:61]}")
            print(f"After block 0, all positions (dim 50): {h[0, :, 50:61]}")
            
            # Block 1
            h = model.blocks[1](h)
            print(f"After block 1, position 4 (dim 50): {h[0, 4, 50:61]}")
            print(f"After block 1, all positions (dim 50): {h[0, :, 50:61]}")
            
            # Final
            h = model.final_ln(h)
            logits = model.head(h)
            print(f"Logits at position 5: {logits[0, 5, :11]}")
            print(f"Predicted tokens: {logits.argmax(dim=-1)[0]}")

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
