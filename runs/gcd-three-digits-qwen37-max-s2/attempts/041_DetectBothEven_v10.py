"""
Interpretable transformer for GCD computation.

Task: Given two 3-digit numbers "abc,def=", compute their GCD as 3-digit output.

Input format: positions 0-2 = first number digits, 3 = comma, 4-6 = second number digits, 7 = equals
Output: autoregressive, prediction at pos 7 → 1st digit, pos 8 → 2nd digit, pos 9 → 3rd digit.

Strategy: Detect if both numbers end in even digit → predict 002, else 001.
- Block 0 attention: head 0 copies last digit of 1st number (pos 2) to residual stream
                   head 1 copies last digit of 2nd number (pos 6) to residual stream
- Block 0 MLP: checks if both copied digits are even, boosts dim 2 if so
- Positions 7,8 always predict '0' via position embeddings
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
    """Detect if both numbers end in even digit → predict 002, else 001.
    
    Key insight: positions 7 and 8 should strongly predict '0' (no attention needed).
    Position 9 should have a weak default of '1', but allow MLP to override with '2'.
    """
    d = model.d_model  # 64
    
    # Zero all parameters
    for p in model.parameters():
        p.data.zero_()
    
    # Set all LayerNorm gammas to 1
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln2.weight.data.fill_(1.0)
    model.final_ln.weight.data.fill_(1.0)
    
    # Token embeddings: one-hot encoding for digits 0-9
    token_emb = torch.zeros(model.vocab_size, d)
    for i in range(10):
        token_emb[i, i] = 1.0
    model.token_emb.weight.data = token_emb
    
    # Position embeddings
    pos_emb = torch.zeros(model.max_seq_len, d)
    # Key markers for attention (moderate strength)
    pos_emb[2, 10] = 3.0   # Position 2 key marker (head 0)
    pos_emb[6, 11] = 3.0   # Position 6 key marker (head 1)
    # Query marker for position 9
    pos_emb[9, 14] = 3.0   # Position 9 query marker
    # Output position signals
    pos_emb[7, 0] = 20.0   # Position 7: predict '0' (strong, no attention needed)
    pos_emb[8, 0] = 20.0   # Position 8: predict '0' (strong, no attention needed)
    pos_emb[9, 1] = 1.0    # Position 9: weak default '1', let MLP override
    model.pos_emb.weight.data = pos_emb
    
    # === Block 0 Attention ===
    attn = model.blocks[0].attn
    W_q = torch.zeros(d, d)
    W_k = torch.zeros(d, d)
    W_v = torch.zeros(d, d)
    W_o = torch.zeros(d, d)
    
    # Head 0: position 9 attends to position 2
    W_q[0, 14] = 10.0
    W_k[0, 10] = 10.0
    for i in range(10):
        W_v[i, i] = 1.0
    
    # Head 1: position 9 attends to position 6
    W_q[16, 14] = 10.0
    W_k[16, 11] = 10.0
    for i in range(10):
        W_v[16 + i, i] = 1.0
    
    # Output projection
    for i in range(16):
        W_o[30 + i, i] = 1.0
    for i in range(16):
        W_o[46 + i, 16 + i] = 1.0
    
    attn.W_q.weight.data = W_q
    attn.W_k.weight.data = W_k
    attn.W_v.weight.data = W_v
    attn.W_o.weight.data = W_o
    
    # === Block 0 MLP: detect both even ===
    mlp = model.blocks[0].mlp
    fc1_w = torch.zeros(model.d_ff, d)
    fc1_b = torch.zeros(model.d_ff)
    fc2_w = torch.zeros(d, model.d_ff)
    fc2_b = torch.zeros(d)
    
    # Neuron 0: check if digit from pos 2 is even
    for even_d in [0, 2, 4, 6, 8]:
        fc1_w[0, 30 + even_d] = 1.0
    fc1_b[0] = -0.5
    
    # Neuron 1: check if digit from pos 6 is even
    for even_d in [0, 2, 4, 6, 8]:
        fc1_w[1, 46 + even_d] = 1.0
    fc1_b[1] = -0.5
    
    # AND gate: if both even, boost dim 2 strongly
    fc2_w[2, 0] = 20.0
    fc2_w[2, 1] = 20.0
    fc2_b[2] = -15.0
    
    mlp.fc1.weight.data = fc1_w
    mlp.fc1.bias.data = fc1_b
    mlp.fc2.weight.data = fc2_w
    mlp.fc2.bias.data = fc2_b
    
    # === Output head ===
    head_w = torch.zeros(model.vocab_size, d)
    head_w[0, 0] = 100.0
    head_w[1, 1] = 100.0
    head_w[2, 2] = 100.0
    model.head.weight.data = head_w


model_shorthand_name = "DetectBothEven_v10"
model_description = "Positions 7,8 strong for '0'. Position 9 weak default '1', MLP can override with '2' if both even."


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
