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
    """Attention heads count specific characters for sentiment classification."""
    torch.manual_seed(0)
    D = model.d_model
    n_heads = model.n_heads
    
    # Token embeddings: one-hot character encoding
    model.token_emb.weight.data.zero_()
    stoi = task.stoi
    for char, idx in stoi.items():
        if idx < D:
            model.token_emb.weight.data[idx, idx] = 10.0
    
    model.pos_emb.weight.data.zero_()
    
    # Block 0: Multi-head character counting
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    # Attention: Each head attends to specific characters
    W_q = block0.attn.W_q.weight.data
    W_k = block0.attn.W_k.weight.data
    W_v = block0.attn.W_v.weight.data
    W_o = block0.attn.W_o.weight.data
    
    W_q.zero_()
    W_k.zero_()
    W_v.zero_()
    W_o.zero_()
    
    # Positive characters: g(6), k(10), p(15), v(21), r(17)
    # Negative characters: q(16), j(9), u(20), z(25), '(27)
    
    if n_heads >= 8:
        d_h = D // n_heads
        
        # Head 0: Count 'g' (positive)
        W_k[0, 6] = 10.0  # Key: high when character is 'g'
        W_q[:, 0] = 1.0   # Query from position 0 (or any)
        W_v[:, 6] = 1.0   # Value: 'g' character
        W_o[:, 0] = 1.0   # Output to dim 0
        
        # Head 1: Count 'k' (positive)
        W_k[0, 10] = 10.0
        W_q[:, 1] = 1.0
        W_v[:, 10] = 1.0
        W_o[:, 1] = 1.0   # Output to dim 1
        
        # Head 2: Count 'p' (positive)
        W_k[0, 15] = 10.0
        W_q[:, 2] = 1.0
        W_v[:, 15] = 1.0
        W_o[:, 2] = 1.0
        
        # Head 3: Count 'v' (positive)
        W_k[0, 21] = 10.0
        W_q[:, 3] = 1.0
        W_v[:, 21] = 1.0
        W_o[:, 3] = 1.0
        
        # Head 4: Count 'q' (negative)
        W_k[0, 16] = 10.0
        W_q[:, 4] = 1.0
        W_v[:, 16] = 1.0
        W_o[:, 4] = 1.0
        
        # Head 5: Count 'j' (negative)
        W_k[0, 9] = 10.0
        W_q[:, 5] = 1.0
        W_v[:, 9] = 1.0
        W_o[:, 5] = 1.0
        
        # Head 6: Count 'u' (negative)
        W_k[0, 20] = 10.0
        W_q[:, 6] = 1.0
        W_v[:, 20] = 1.0
        W_o[:, 6] = 1.0
        
        # Head 7: Count 'z' (negative)
        W_k[0, 25] = 10.0
        W_q[:, 7] = 1.0
        W_v[:, 25] = 1.0
        W_o[:, 7] = 1.0
    
    # MLP: Combine character counts
    fc1 = block0.mlp.fc1.weight.data
    fc1_b = block0.mlp.fc1.bias.data
    fc2 = block0.mlp.fc2.weight.data
    fc2_b = block0.mlp.fc2.bias.data
    
    fc1.zero_()
    fc1_b.zero_()
    fc2.zero_()
    fc2_b.zero_()
    
    # fc1: Detect if character counts are high enough
    # Hidden 0: Sum positive character counts
    fc1[0, 0] = 1.0  # g
    fc1[0, 1] = 1.0  # k
    fc1[0, 2] = 1.0  # p
    fc1[0, 3] = 1.0  # v
    fc1_b[0] = -5.0  # Threshold
    
    # Hidden 1: Sum negative character counts
    fc1[1, 4] = 1.0  # q
    fc1[1, 5] = 1.0  # j
    fc1[1, 6] = 1.0  # u
    fc1[1, 7] = 1.0  # z
    fc1_b[1] = -5.0
    
    # fc2: Compute sentiment = positive - negative
    fc2[0, 0] = 100.0
    fc2[0, 1] = -100.0
    
    # Block 1, 2: identity
    for block in model.blocks[1:]:
        block.attn.W_q.weight.data.zero_()
        block.attn.W_k.weight.data.zero_()
        block.attn.W_v.weight.data.zero_()
        block.attn.W_o.weight.data.zero_()
        block.mlp.fc1.weight.data.zero_()
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.zero_()
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: classify based on sentiment score (dim 0)
    model.head.weight.data.zero_()
    model.head.weight.data[31, 0] = 10.0
    model.head.weight.data[30, 0] = -10.0
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "DirectCharSentiment"
model_description = "Direct character sentiment scores in embeddings with uniform attention averaging."


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
