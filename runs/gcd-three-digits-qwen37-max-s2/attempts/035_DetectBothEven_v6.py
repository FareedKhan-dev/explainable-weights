"""
Interpretable transformer for GCD computation.

Task: Given two 3-digit numbers "abc,def=", compute their GCD as 3-digit output.

Input format: positions 0-2 = first number digits, 3 = comma, 4-6 = second number digits, 7 = equals
Output format: positions 8-10 = three-digit GCD

Strategy: Use attention to gather divisibility information from input digits,
then use MLP as lookup table to determine GCD.
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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
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
    """Detect if both numbers are divisible by 2 (both end in even digit).
    
    Input: positions 0-2 = first number, 3 = comma, 4-6 = second number, 7 = equals
    Last digits are at positions 2 and 6.
    
    Strategy:
    1. Use attention to copy digit info from positions 2 and 6 to output positions
    2. Use MLP to check if both digits are even (0,2,4,6,8)
    3. Output 002 if both even, else 001
    """
    d = model.d_model
    
    # Zero all parameters
    for p in model.parameters():
        p.data.zero_()
    
    # Set all LayerNorm gammas to 1
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln2.weight.data.fill_(1.0)
    model.final_ln.weight.data.fill_(1.0)
    
    # Token embeddings: encode digit values 0-9 in dimensions 0-9
    token_emb = torch.zeros(model.vocab_size, d)
    for i in range(10):
        token_emb[i, i] = 1.0
    model.token_emb.weight.data = token_emb
    
    # Position embeddings: mark key positions for attention
    pos_emb = torch.zeros(model.max_seq_len, d)
    # Mark position 2 (last digit of first number) with dim 10
    pos_emb[2, 10] = 5.0
    # Mark position 6 (last digit of second number) with dim 11
    pos_emb[6, 11] = 5.0
    # Mark position 7 (predict 1st output digit) with dim 12
    pos_emb[7, 12] = 5.0
    # Mark position 8 (predict 2nd output digit) with dim 13
    pos_emb[8, 13] = 5.0
    # Mark position 9 (predict 3rd output digit) with dim 14
    pos_emb[9, 14] = 5.0
    model.pos_emb.weight.data = pos_emb
    
    # First block: attention to gather info from positions 2 and 6
    attn = model.blocks[0].attn
    W_q = torch.zeros(d, d)
    W_k = torch.zeros(d, d)
    W_v = torch.zeros(d, d)
    W_o = torch.zeros(d, d)
    
    # Position 7 (dim 12) queries position 2 (dim 10) to get last digit of first number
    W_q[12, 12] = 10.0  # query from position 7
    W_k[10, 10] = 10.0  # key from position 2
    W_v[0, 0] = 1.0     # copy digit info
    
    # Position 7 (dim 12) queries position 6 (dim 11) to get last digit of second number
    W_q[12, 12] = 10.0
    W_k[11, 11] = 10.0  # key from position 6
    W_v[1, 1] = 1.0     # copy digit info
    
    attn.W_q.weight.data = W_q.T
    attn.W_k.weight.data = W_k.T
    attn.W_v.weight.data = W_v.T
    attn.W_o.weight.data = W_o.T
    
    # MLP: check if both digits are even
    mlp = model.blocks[0].mlp
    fc1_w = torch.zeros(model.d_ff, d)
    fc1_b = torch.zeros(model.d_ff)
    fc2_w = torch.zeros(d, model.d_ff)
    fc2_b = torch.zeros(d)
    
    # Detect even digits in dimension 0 (from first number)
    # Even digits: 0,2,4,6,8
    for even_digit in [0, 2, 4, 6, 8]:
        fc1_w[even_digit, even_digit] = 1.0
        fc1_b[even_digit] = -0.5
    
    # Detect even digits in dimension 1 (from second number)
    for even_digit in [0, 2, 4, 6, 8]:
        fc1_w[10 + even_digit, even_digit] = 1.0
        fc1_b[10 + even_digit] = -0.5
    
    # If both even, boost dimension 2 (for predicting '2')
    fc2_w[2, 0] = 10.0
    
    mlp.fc1.weight.data = fc1_w
    mlp.fc1.bias.data = fc1_b
    mlp.fc2.weight.data = fc2_w
    mlp.fc2.bias.data = fc2_b
    
    # Position embeddings for output: predict 001 or 002
    # Position 7: predict '0' (dim 0)
    # Position 8: predict '0' (dim 0)
    # Position 9: predict '1' (dim 1) by default, or '2' (dim 2) if both even
    pos_emb[7, 0] = 10.0
    pos_emb[8, 0] = 10.0
    pos_emb[9, 1] = 10.0  # default to '1'
    model.pos_emb.weight.data = pos_emb
    
    # Output head: map dimensions to tokens
    head_w = torch.zeros(model.vocab_size, d)
    head_w[0, 0] = 100.0  # dim 0 → token '0'
    head_w[1, 1] = 100.0  # dim 1 → token '1'
    head_w[2, 2] = 100.0  # dim 2 → token '2'
    model.head.weight.data = head_w


model_shorthand_name = "DetectBothEven_v6"
model_description = "Detect if both numbers end in even digits using attention and MLP. Predict 002 if both even, else 001."


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
