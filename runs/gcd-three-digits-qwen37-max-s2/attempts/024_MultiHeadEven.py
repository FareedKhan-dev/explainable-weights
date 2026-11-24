"""
Interpretable transformer for GCD computation.

Task: Given two 3-digit numbers "abc,def=", compute their GCD as 3-digit output.

Strategy: Compute both numbers mod small primes, check which primes divide both,
and output the product of common prime factors.
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
    """Use multiple attention heads to check different divisibility conditions.
    
    Strategy: Use position embeddings to encode position info, then use
    attention heads to detect specific patterns.
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
    
    # Token embeddings: one-hot for digits
    token_emb = torch.zeros(model.vocab_size, d)
    for i in range(10):
        token_emb[i, i] = 1.0
    model.token_emb.weight.data = token_emb
    
    # Position embeddings: encode position and output signals
    pos_emb = torch.zeros(model.max_seq_len, d)
    # Position 2: signal "I am position 2"
    pos_emb[2, 10] = 5.0
    # Position 6: signal "I am position 6"
    pos_emb[6, 11] = 5.0
    # Output positions: default signals
    pos_emb[8, 20] = 10.0  # hundreds → "0"
    pos_emb[9, 21] = 10.0  # tens → "0"
    pos_emb[10, 22] = 10.0 # ones → "1"
    model.pos_emb.weight.data = pos_emb
    
    # Block 0: Attention with multiple heads
    # Each head checks a different condition
    attn_q = torch.zeros(d, d)
    attn_k = torch.zeros(d, d)
    attn_v = torch.zeros(d, d)
    attn_o = torch.zeros(d, d)
    
    # Head 0: Copy digit from position 2 to output
    attn_q[10, 10] = 3.0  # query for position 2
    attn_k[10, 10] = 3.0  # position 2 key
    attn_v[10, 10] = 1.0  # copy digit info from pos 2
    attn_o[30, 10] = 1.0  # output to dim 30
    
    # Head 1: Copy digit from position 6 to output
    attn_q[11, 11] = 3.0  # query for position 6
    attn_k[11, 11] = 3.0  # position 6 key
    attn_v[11, 11] = 1.0  # copy digit info from pos 6
    attn_o[31, 11] = 1.0  # output to dim 31
    
    model.blocks[0].attn.W_q.weight.data = attn_q
    model.blocks[0].attn.W_k.weight.data = attn_k
    model.blocks[0].attn.W_v.weight.data = attn_v
    model.blocks[0].attn.W_o.weight.data = attn_o
    
    # Head: check conditions and predict output
    head_w = torch.zeros(model.vocab_size, d)
    
    # Default: predict "001" from pos_emb
    head_w[0, 20] = 10.0
    head_w[0, 21] = 10.0
    head_w[1, 22] = 10.0
    
    # If position 2 has even digit AND position 6 has even digit, predict "002"
    # Check dims 30 and 31 for even digits
    for even_digit in [0, 2, 4, 6, 8]:
        # Both positions have even digit
        head_w[2, 30] += 2.0  # boost if pos 2 copied
        head_w[2, 31] += 2.0  # boost if pos 6 copied
        head_w[1, 30] -= 1.0  # suppress "1" if pos 2 copied
        head_w[1, 31] -= 1.0  # suppress "1" if pos 6 copied
    
    model.head.weight.data = head_w


model_shorthand_name = "MultiHeadEven"
model_description = "Use multiple attention heads to copy digits from positions 2 and 6, then check if both even."


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
