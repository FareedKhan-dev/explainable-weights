"""
Interpretable transformer for word-reversal-3x3.

Task: "XXX_YYY_ZZZ=" -> "ZZZ_YYY_XXX"
Pure copy circuit: attention copies tokens from source positions to output positions.
NO residual connections - prevents self-attention interference.
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
    """NO residual connections - pure attention then MLP."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.ln1(x))
        x = self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 56,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 56,
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

# Autoregressive generation:
#   step 0: model sees pos 0..11 (prompt), reads logits at pos 11
#   step k: model sees pos 0..(11+k), reads logits at pos 11+k
#
# Prompt layout: 0 1 2 3(_) 4 5 6 7(_) 8 9 10 11(=)
# Word1=chars[0:3], Word2=chars[4:7], Word3=chars[8:11]
# Reversed output: Word3_rev _ Word2_rev _ Word1_rev

COPY_MAP = {
    11: 10, 12: 9, 13: 8,    # Word3 reversed (Z2, Z1, Z0)
    14: 3,                     # underscore
    15: 6, 16: 5, 17: 4,     # Word2 reversed (Y2, Y1, Y0)
    18: 7,                     # underscore
    19: 2, 20: 1, 21: 0,     # Word1 reversed (X2, X1, X0)
}


def write_weights(model: SimpleTransformer, task) -> None:
    V = 28
    D = model.d_model  # 56
    S = model.max_seq_len  # 32

    # Dimension layout:
    # [0..27]  = token identity (28 dims, one-hot)
    # [28..39] = key space (12 dims, one-hot per source position)
    # [40..51] = query space (12 dims, encodes which source to copy from)
    # [52..55] = spare

    for p in model.parameters():
        p.data.zero_()

    # Token embedding: one-hot in dims [0..27]
    tok = torch.zeros(V, D)
    for i in range(V):
        tok[i, i] = 1.0
    model.token_emb.weight.data = tok

    # Position embedding:
    #   Source positions (0..11): key dim = 28+p (value 10.0)
    #   Output positions (11..21): query dim = 40+source (value 10.0)
    pos = torch.zeros(S, D)
    for p in range(12):
        pos[p, 28 + p] = 10.0
    for out_p, src_p in COPY_MAP.items():
        pos[out_p, 40 + src_p] = 10.0
    model.pos_emb.weight.data = pos

    # Attention: W_q maps query space -> key space, W_k extracts key space
    # W_v extracts token dims, W_o passes token dims through
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)

    for i in range(12):
        W_q[28 + i, 40 + i] = 1.0   # query dim 40+i -> key dim 28+i
        W_k[28 + i, 28 + i] = 1.0   # key dim 28+i -> key dim 28+i

    for i in range(V):
        W_v[i, i] = 1.0              # token dim i -> token dim i
        W_o[i, i] = 1.0              # token dim i -> token dim i

    blk = model.blocks[0]
    blk.attn.W_q.weight.data = W_q
    blk.attn.W_k.weight.data = W_k
    blk.attn.W_v.weight.data = W_v
    blk.attn.W_o.weight.data = W_o

    # MLP disabled (all zeros)

    # LayerNorms: standard (weight=1, bias=0)
    blk.ln1.weight.data.fill_(1.0)
    blk.ln1.bias.data.zero_()
    blk.ln2.weight.data.fill_(1.0)
    blk.ln2.bias.data.zero_()
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()

    # Head: token dims [0..27] -> vocab logits
    head_w = torch.zeros(V, D)
    for i in range(V):
        head_w[i, i] = 5.0
    model.head.weight.data = head_w


model_shorthand_name = "CopyNoResidual_v1"
model_description = "No-residual 1-layer copy circuit. Attention copies tokens from source positions using position-encoded key/query matching."


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
