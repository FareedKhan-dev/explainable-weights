"""
Interpretable transformer for character-level sequence tasks.

Word-reversal-3x3: "XXX_YYY_ZZZ=" -> "ZZZ_YYY_XXX"
Each output position copies from a fixed source position - pure copy circuit.
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
        n_layers: int = 1,
        d_ff: int = 64,
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

# Mapping: output_pos -> source_pos in prompt
# Input:  0 1 2 _ 4 5 6 _ 8 9 10 =  (positions 0..11)
# Output: Z2 Z1 Z0 _ Y2 Y1 Y0 _ X2 X1 X0 (positions 12..22)
# Source: 10 9  8  3  6  5  4  7  2  1  0

COPY_MAP = {
    12: 10, 13: 9, 14: 8,   # word3 reversed
    15: 3,                    # underscore (copy from pos 3 which is _)
    16: 6, 17: 5, 18: 4,    # word2 reversed
    19: 7,                    # underscore (copy from pos 7 which is _)
    20: 2, 21: 1, 22: 0,    # word1 reversed
}


def write_weights(model: SimpleTransformer, task) -> None:
    """Build a copying circuit using attention."""
    torch.manual_seed(42)
    V = 28  # vocab size
    D = model.d_model
    S = model.max_seq_len

    # Zero everything first
    for p in model.parameters():
        p.data.zero_()

    # Token embedding: one-hot-like encoding for each token
    # Use first V dimensions of d_model for token identity
    tok = torch.zeros(V, D)
    for i in range(V):
        if i < D:
            tok[i, i] = 1.0
    model.token_emb.weight.data = tok

    # Position embedding: encode "I am position p" and "I want to copy from position s"
    # Split d_model into: [token_space(V)] [key_space] [query_space]
    # For key positions 0..11: set key_vec to identify that position
    # For query positions 12..22: set query_vec to match desired source
    key_start = V
    query_start = key_start + 12  # 12 possible source positions

    pos = torch.zeros(S, D)
    
    # Set key vectors for source positions (0..11)
    for p in range(12):
        idx = key_start + p
        if idx < D:
            pos[p, idx] = 5.0  # strong key signal

    # Set query vectors for output positions (12..22)
    for out_p, src_p in COPY_MAP.items():
        idx = key_start + src_p  # match the key of source position
        if idx < D:
            pos[out_p, idx] = 5.0  # strong query signal

    model.pos_emb.weight.data = pos

    # W_q: extract query part (the position-based query)
    # W_k: extract key part (the position-based key)
    # W_v: extract token identity
    # W_o: pass through to output

    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)

    # Query: read from query_start region of position embedding
    for i in range(12):
        qi = query_start + i
        if qi < D:
            W_q[qi, qi] = 1.0

    # Key: read from key_start region
    for i in range(12):
        ki = key_start + i
        if ki < D:
            W_k[ki, ki] = 1.0

    # Value: read token identity (first V dims)
    for i in range(min(V, D)):
        W_v[i, i] = 1.0

    # Output projection: pass token dims through
    for i in range(min(V, D)):
        W_o[i, i] = 1.0

    block = model.blocks[0]
    block.attn.W_q.weight.data = W_q
    block.attn.W_k.weight.data = W_k
    block.attn.W_v.weight.data = W_v
    block.attn.W_o.weight.data = W_o

    # Disable MLP (zero weights)
    block.mlp.fc1.weight.data.zero_()
    block.mlp.fc1.bias.data.zero_()
    block.mlp.fc2.weight.data.zero_()
    block.mlp.fc2.bias.data.zero_()

    # LayerNorm: identity-like (gamma=1, beta=0) - already default
    block.ln1.weight.data.fill_(1.0)
    block.ln1.bias.data.zero_()
    block.ln2.weight.data.fill_(1.0)
    block.ln2.bias.data.zero_()

    # Final LN
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()

    # Head: read token dims -> vocab logits
    head_w = torch.zeros(V, D)
    for i in range(min(V, D)):
        head_w[i, i] = 3.0
    model.head.weight.data = head_w


model_shorthand_name = "CopyCircuit_v1"
model_description = "1-layer attention copy circuit: position embeddings encode source->target mapping, token embeddings carry identity."


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
