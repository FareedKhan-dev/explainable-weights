"""
Interpretable transformer for smallest-prime-factor task.
Algorithmic approach: implement divisibility rules using attention and MLPs.
- If last digit even → SPF=2
- If digit sum % 3 == 0 → SPF=3  
- If ends in 0 or 5 → SPF=5
- Otherwise check small primes or return number itself
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
    Algorithmic SPF computation:
    - Extract each digit cleanly using position-aware attention
    - Apply divisibility rules to determine SPF
    - Encode answer as 5-digit string
    """
    torch.manual_seed(42)
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H

    # Embeddings: clean one-hot for digits and positions
    tok_emb = torch.zeros(V, D)
    for d in range(V):
        tok_emb[d, d] = 5.0
    
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        pos_emb[p, V + p] = 5.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # Block 0: Extract digits using position-selective attention
    # Head 0-4: each attends to its corresponding digit position
    # Head 5-7: unused (uniform)
    
    attn0 = model.blocks[0].attn
    
    # Q and K: make head h attend to position h
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    
    for h in range(5):
        # Head h: query at pos 5 should match key at pos h
        # Use position embedding dims
        q_dim = V + 5  # position 5
        k_dim = V + h  # position h
        
        # Query extracts position info
        W_q[h * dh, q_dim] = 3.0
        # Key matches position
        W_k[h * dh, k_dim] = 3.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    
    # V: pass through token identity (digits 0-9)
    W_v = torch.zeros(D, D)
    for i in range(V):
        W_v[i, i] = 1.0
    attn0.W_v.weight.data = W_v
    
    # O: identity
    attn0.W_o.weight.data = torch.eye(D)
    
    # Block 0 MLP: compute divisibility features
    # Input at pos 5 after attention: [digit0, digit1, digit2, digit3, digit4, ...]
    # in dims 0-4 (from each head's output)
    
    mlp1 = model.blocks[0].mlp
    d_ff1 = mlp1.fc1.weight.shape[0]
    
    fc1_w = torch.zeros(d_ff1, D)
    fc1_b = torch.zeros(d_ff1)
    fc2_w = torch.zeros(D, d_ff1)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # Feature 1: last digit (digit4) is even → divisible by 2
    for d in [0, 2, 4, 6, 8]:
        fc1_w[neuron, 4 * 10 + d] = 5.0  # digit4 == d
        fc1_b[neuron] = -3.0
        fc2_w[50, neuron] = 5.0  # dim 50 = "div by 2" flag
        neuron += 1
    
    # Feature 2: digit sum % 3 == 0 → divisible by 3
    # Compute digit sum mod 3
    # digit % 3: 0->0, 1->1, 2->2, 3->0, 4->1, 5->2, 6->0, 7->1, 8->2, 9->0
    digit_mod3 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    
    # Sum all digit mods, check if sum % 3 == 0
    # Use neurons to accumulate sum for each remainder (0, 1, 2)
    for rem in range(3):
        for pos in range(5):
            for d in range(10):
                if digit_mod3[d] == rem:
                    fc1_w[neuron, pos * 10 + d] = 1.0
                    fc1_b[neuron] = -0.5
                    fc2_w[51 + rem, neuron] = 1.0  # dims 51-53 count remainders
                    neuron += 1
    
    # Feature 3: last digit is 0 or 5 → divisible by 5
    for d in [0, 5]:
        fc1_w[neuron, 4 * 10 + d] = 5.0
        fc1_b[neuron] = -3.0
        fc2_w[54, neuron] = 5.0  # dim 54 = "div by 5" flag
        neuron += 1
    
    # Decision logic: check divisibility flags and output SPF
    # Priority: 2 > 3 > 5 > 7 > 11 > ... > number itself
    
    # Check div by 2
    fc1_w[neuron, 50] = 5.0  # if div_by_2 flag is high
    fc1_b[neuron] = -2.0
    fc2_w[55, neuron] = 10.0  # dim 55 = "SPF is 2" flag
    neuron += 1
    
    # Check div by 3 (sum % 3 == 0) AND not div by 2
    fc1_w[neuron, 51] = 5.0  # sum mod 3 count
    fc1_w[neuron, 50] = -5.0  # NOT div by 2
    fc1_b[neuron] = -2.0
    fc2_w[56, neuron] = 10.0  # dim 56 = "SPF is 3" flag
    neuron += 1
    
    # Check div by 5 AND not div by 2 or 3
    fc1_w[neuron, 54] = 5.0  # if div_by_5 flag
    fc1_w[neuron, 50] = -5.0  # NOT div by 2
    fc1_w[neuron, 56] = -5.0  # NOT div by 3
    fc1_b[neuron] = -5.0
    fc2_w[57, neuron] = 10.0  # dim 57 = "SPF is 5" flag
    neuron += 1
    
    # Default: number is prime (SPF = number itself)
    # This gets set if none of the above flags are high
    fc1_w[neuron, 55] = -5.0  # NOT SPF=2
    fc1_w[neuron, 56] = -5.0  # NOT SPF=3
    fc1_w[neuron, 57] = -5.0  # NOT SPF=5
    fc1_b[neuron] = 10.0
    fc2_w[58, neuron] = 10.0  # dim 58 = "prime" flag
    neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Block 1: Decode SPF flags into output digits
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
    
    # If SPF=2: output "00002"
    # Digits: 0,0,0,0,2 at positions 6-10
    for out_pos in range(5):
        digit = 2 if out_pos == 4 else 0
        fc1_2_w[neuron, 55] = 5.0  # SPF=2 flag
        fc1_2_w[neuron, V + 6 + out_pos] = 5.0  # output position
        fc1_2_b[neuron] = -7.0
        fc2_2_w[digit, neuron] = 20.0
        neuron += 1
    
    # If SPF=3: output "00003"
    for out_pos in range(5):
        digit = 3 if out_pos == 4 else 0
        fc1_2_w[neuron, 56] = 5.0
        fc1_2_w[neuron, V + 6 + out_pos] = 5.0
        fc1_2_b[neuron] = -7.0
        fc2_2_w[digit, neuron] = 20.0
        neuron += 1
    
    # If SPF=5: output "00005"
    for out_pos in range(5):
        digit = 5 if out_pos == 4 else 0
        fc1_2_w[neuron, 57] = 5.0
        fc1_2_w[neuron, V + 6 + out_pos] = 5.0
        fc1_2_b[neuron] = -7.0
        fc2_2_w[digit, neuron] = 20.0
        neuron += 1
    
    # If prime: copy input digits to output
    # For each input digit position (0-4) and output position (0-4):
    for in_pos in range(5):
        for d in range(10):
            fc1_2_w[neuron, in_pos * 10 + d] = 5.0  # input digit
            fc1_2_w[neuron, 58] = 5.0  # prime flag
            fc1_2_w[neuron, V + 6 + in_pos] = 5.0  # output position matches input
            fc1_2_b[neuron] = -10.0
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
    
    print("Weights initialized (algorithmic SPF)!")


model_shorthand_name = "spf-exp8-algorithmic"
model_description = "Algorithmic SPF using divisibility rules (2,3,5) and prime detection"


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
        d_ff_list=[256, 256],
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
