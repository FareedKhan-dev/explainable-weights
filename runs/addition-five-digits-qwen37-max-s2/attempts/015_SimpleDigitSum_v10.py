"""
Interpretable transformer for 5-digit addition.
Task: "12345+67890=" -> "080235"

Positions: 0-4 (A), 5 (+), 6-10 (B), 11 (=), 12-17 (answer, 6 digits)
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
# Architecture
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """
    Implement 5-digit addition by computing digit sums at each answer position.
    
    Each answer position 12+i computes:
      - Attend to A[i-1] and B[i-1] (for i=1..5)
      - Compute (A[i-1] + B[i-1]) mod 10 using MLP lookup
      - For i=0, predict 0 or 1 based on carry
    """
    D = model.d_model  # 128
    H = model.n_heads  # 8
    dh = D // H        # 16
    
    # Zero all parameters
    for p in model.parameters():
        p.data.zero_()
    
    # === TOKEN EMBEDDINGS ===
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0  # one-hot for digits
    model.token_emb.weight.data[10, 10] = 1.0  # '+'
    model.token_emb.weight.data[11, 11] = 1.0  # '='
    
    # === POSITION EMBEDDINGS ===
    for p in range(min(18, model.max_seq_len)):
        model.pos_emb.weight.data[p, 20 + p] = 1.0  # position one-hot
    
    # Answer positions target specific input positions
    # Answer pos 12+i should look at input pos i (for A) and 6+i (for B)
    for i in range(6):
        if 12 + i < model.max_seq_len:
            model.pos_emb.weight.data[12 + i, 40 + i] = 1.0  # target A position
            model.pos_emb.weight.data[12 + i, 50 + i] = 1.0  # target B position
    
    # Input positions advertise their index
    for i in range(5):
        model.pos_emb.weight.data[i, 60 + i] = 1.0  # A[i] marker
        model.pos_emb.weight.data[6 + i, 70 + i] = 1.0  # B[i] marker
    
    # LayerNorm: identity
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln2.weight.data.fill_(1.0)
    model.final_ln.weight.data.fill_(1.0)
    
    SCALE = 15.0
    
    # === LAYER 1: Answer positions copy A[i-1] and B[i-1] ===
    block0 = model.blocks[0]
    
    # Head 0: answer pos 12+i attends to input pos i (A[i])
    for j in range(6):
        block0.attn.W_q.weight.data[0*dh + j, 40 + j] = SCALE
        block0.attn.W_k.weight.data[0*dh + j, 60 + j] = 1.0
    for j in range(10):
        block0.attn.W_v.weight.data[0*dh + j, j] = 1.0
        if 90 + j < D:
            block0.attn.W_o.weight.data[90 + j, 0*dh + j] = 1.0
    
    # Head 1: answer pos 12+i attends to input pos 6+i (B[i])
    for j in range(6):
        block0.attn.W_q.weight.data[1*dh + j, 50 + j] = SCALE
        block0.attn.W_k.weight.data[1*dh + j, 70 + j] = 1.0
    for j in range(10):
        block0.attn.W_v.weight.data[1*dh + j, j] = 1.0
        if 100 + j < D:
            block0.attn.W_o.weight.data[100 + j, 1*dh + j] = 1.0
    
    # === LAYER 1 MLP: Compute (A[i] + B[i]) mod 10 ===
    # Input: dims 90-99 (A), dims 100-109 (B)
    # Output: dims 110-119 (sum mod 10 one-hot)
    
    neuron_idx = 0
    for a in range(10):
        for b in range(10):
            if neuron_idx >= model.d_ff:
                break
            block0.mlp.fc1.weight.data[neuron_idx, 90 + a] = 1.0
            block0.mlp.fc1.weight.data[neuron_idx, 100 + b] = 1.0
            block0.mlp.fc1.bias.data[neuron_idx] = -1.5
            
            result = (a + b) % 10
            block0.mlp.fc2.weight.data[110 + result, neuron_idx] += 1.0
            neuron_idx += 1
    
    # === LAYER 2 & 3: Identity (pass through) ===
    block1 = model.blocks[1]
    for j in range(min(dh, D)):
        block1.attn.W_q.weight.data[j, j] = 1.0
        block1.attn.W_k.weight.data[j, j] = 1.0
        block1.attn.W_v.weight.data[j, j] = 1.0
        block1.attn.W_o.weight.data[j, j] = 1.0
    
    block2 = model.blocks[2]
    for j in range(min(dh, D)):
        block2.attn.W_q.weight.data[j, j] = 1.0
        block2.attn.W_k.weight.data[j, j] = 1.0
        block2.attn.W_v.weight.data[j, j] = 1.0
        block2.attn.W_o.weight.data[j, j] = 1.0
    
    # === OUTPUT HEAD ===
    # Map dims 110-119 (sum mod 10) to logits for digits 0-9
    for d in range(10):
        if 110 + d < D:
            model.head.weight.data[d, 110 + d] = 10.0
    
    # Suppress special tokens
    for d in range(10, 12):
        for j in range(D):
            model.head.weight.data[d, j] = -100.0


model_shorthand_name = "SimpleDigitSum_v10"
model_description = "Copy A[i] and B[i] to answer positions, compute (A+B) mod 10 with MLP lookup."


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
    max_seq_len = max(task.seq_len, 18)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=512,
    )
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

    accuracy, details = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )
    
    # Debug: print first example
    if hasattr(details, '__iter__') and len(details) > 0:
        print(f"First example details: {details[0] if isinstance(details, list) else 'N/A'}")

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
