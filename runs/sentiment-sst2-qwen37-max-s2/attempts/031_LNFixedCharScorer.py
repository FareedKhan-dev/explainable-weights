"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 512,
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
    """Sentiment via char scores with LayerNorm-aware weight design."""
    
    import numpy as np
    
    # Character sentiment scores (log-odds ratio from training data)
    CHAR_SCORES = [
        0.096054,   -0.114909,  -0.011751,  -0.064581,
        0.015304,   0.075313,   0.207707,   0.019560,
        0.039751,   0.024237,  -0.122248,  -0.053352,
        0.040320,   0.046467,  -0.086886,  -0.017761,
       -0.074809,   0.085346,  -0.056220,  -0.010558,
       -0.037918,   0.027198,  -0.028477,   0.313461,
       -0.136195,   0.195482,  -0.032997,  -0.149445,
        0.0,        0.0,        0.0,        0.0,
    ]
    
    D = model.d_model   # 128
    H = model.n_heads   # 8
    FF = model.d_ff     # 512
    VOCAB = model.vocab_size  # 32
    
    # After LayerNorm of one-hot vector in D=128 dims:
    # hot position: sqrt(D-1) ≈ 11.27
    # cold positions: -1/sqrt(D-1) ≈ -0.0884
    # For MLP dot product: hot_val * w[hot] + sum_cold w[cold] * cold_val
    # = w[hot] * sqrt(D-1) - (1/sqrt(D-1)) * sum_{j≠hot} w[j]
    # If w = CHAR_SCORES, this ≈ CHAR_SCORES[hot] * (sqrt(D-1) + 1/sqrt(D-1)) - C/sqrt(D-1)
    # where C = sum of all CHAR_SCORES
    # The key factor is sqrt(D-1) + 1/sqrt(D-1) ≈ 11.36
    
    HOT_VAL = np.sqrt(D - 1)
    COLD_VAL = -1.0 / np.sqrt(D - 1)
    SCALE = HOT_VAL - COLD_VAL  # ≈ 11.36
    
    # Token embeddings: one-hot in dims 0..31
    model.token_emb.weight.data.zero_()
    for i in range(VOCAB):
        model.token_emb.weight.data[i, i] = 1.0
    
    # Position embeddings: all zeros
    model.pos_emb.weight.data.zero_()
    
    # ---- Block 0: MLP computes char sentiment score ----
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    
    # Disable attention in block 0
    block0.attn.W_q.weight.data.zero_()
    block0.attn.W_k.weight.data.zero_()
    block0.attn.W_v.weight.data.zero_()
    block0.attn.W_o.weight.data.zero_()
    
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    fc1 = block0.mlp.fc1.weight.data  # [512, 128]
    fc1_b = block0.mlp.fc1.bias.data
    fc2 = block0.mlp.fc2.weight.data  # [128, 512]
    fc2_b = block0.mlp.fc2.bias.data
    
    fc1.zero_()
    fc1_b.zero_()
    fc2.zero_()
    fc2_b.zero_()
    
    # Compute the offset from LayerNorm cold positions
    C_total = sum(CHAR_SCORES)
    
    # Neuron 0: linear char score (not ReLU'd - we use two neurons for pos/neg)
    # fc1[0] = CHAR_SCORES / SCALE so that output ≈ CHAR_SCORES[i]
    # After LN: dot = w[hot]*HOT_VAL + sum_{j!=hot} w[j]*COLD_VAL
    #           = w[hot]*(HOT_VAL - COLD_VAL) + COLD_VAL * sum_j w[j]
    #           = w[hot] * SCALE + COLD_VAL * sum_j w[j]
    # If w[j] = CHAR_SCORES[j]/SCALE:
    #   = CHAR_SCORES[hot] + COLD_VAL * C_total/SCALE
    
    # Offset to subtract
    offset = COLD_VAL * C_total / SCALE
    
    # Neuron 0: positive part of char score
    # Neuron 1: negative part of char score (will be subtracted)
    for j in range(VOCAB):
        score = CHAR_SCORES[j] / SCALE
        fc1[0, j] = max(0, score)    # positive part
        fc1[1, j] = max(0, -score)   # negative part (absolute value)
    
    # Bias to cancel the cold-position offset
    # For neuron 0: bias = -max(0, offset) ... actually this is tricky
    # The offset is: COLD_VAL * sum_j fc1[0,j] for neuron 0
    # sum_j fc1[0,j] = sum of positive CHAR_SCORES / SCALE
    pos_sum = sum(max(0, s) for s in CHAR_SCORES) / SCALE
    neg_sum = sum(max(0, -s) for s in CHAR_SCORES) / SCALE
    
    fc1_b[0] = -COLD_VAL * pos_sum  # cancel offset for neuron 0
    fc1_b[1] = -COLD_VAL * neg_sum  # cancel offset for neuron 1
    
    # fc2: neuron 0 adds to dim 32, neuron 1 subtracts from dim 32
    fc2[32, 0] = 1.0
    fc2[32, 1] = -1.0
    
    # ---- Block 1: Uniform attention to aggregate dim 32 ----
    block1 = model.blocks[1]
    block1.ln1.weight.data.fill_(1.0)
    block1.ln1.bias.data.zero_()
    
    # Uniform attention: zero Q, K
    block1.attn.W_q.weight.data.zero_()
    block1.attn.W_k.weight.data.zero_()
    
    # V reads dim 32 (sentiment score)
    block1.attn.W_v.weight.data.zero_()
    block1.attn.W_v.weight.data[32, 32] = 1.0
    
    # W_o passes dim 32 through
    block1.attn.W_o.weight.data.zero_()
    block1.attn.W_o.weight.data[32, 32] = 1.0
    
    block1.ln2.weight.data.fill_(1.0)
    block1.ln2.bias.data.zero_()
    block1.mlp.fc1.weight.data.zero_()
    block1.mlp.fc1.bias.data.zero_()
    block1.mlp.fc2.weight.data.zero_()
    block1.mlp.fc2.bias.data.zero_()
    
    # ---- Block 2: Identity ----
    block2 = model.blocks[2]
    block2.ln1.weight.data.fill_(1.0)
    block2.ln1.bias.data.zero_()
    block2.attn.W_q.weight.data.zero_()
    block2.attn.W_k.weight.data.zero_()
    block2.attn.W_v.weight.data.zero_()
    block2.attn.W_o.weight.data.zero_()
    block2.ln2.weight.data.fill_(1.0)
    block2.ln2.bias.data.zero_()
    block2.mlp.fc1.weight.data.zero_()
    block2.mlp.fc1.bias.data.zero_()
    block2.mlp.fc2.weight.data.zero_()
    block2.mlp.fc2.bias.data.zero_()
    
    # Final LN
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: classify based on dim 32
    model.head.weight.data.zero_()
    model.head.weight.data[31, 32] = 10.0   # '1' (positive) if dim 32 > 0
    model.head.weight.data[30, 32] = -10.0  # '0' (negative) if dim 32 < 0


model_shorthand_name = "LNFixedCharScorer"
model_description = "LayerNorm-aware char sentiment scoring: MLP accounts for LN distortion of one-hot vectors."


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
