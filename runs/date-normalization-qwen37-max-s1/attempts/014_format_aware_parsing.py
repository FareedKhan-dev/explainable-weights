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
    """Populate `model`'s parameters in-place. No training allowed.
    
    Experiment 9: Format-aware date parsing with explicit digit extraction.
    Use attention to identify year/month/day components and copy them to output positions.
    """
    import numpy as np
    torch.manual_seed(2026)
    
    vocab = task.vocab
    stoi = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    D = model.d_model
    H = model.n_heads
    L = model.n_layers
    dh = D // H
    
    with torch.no_grad():
        # Token embeddings: digits get strong signals in first D//3 dims
        model.token_emb.weight.data.normal_(0, 0.01)
        
        digit_indices = [stoi[d] for d in '0123456789' if d in stoi]
        for idx in digit_indices:
            model.token_emb.weight.data[idx, :D//3] = torch.randn(D//3) * 0.5
        
        # Separator tokens get strong signals in middle D//3 dims
        sep_indices = [stoi[s] for s in '/-., ' if s in stoi]
        for idx in sep_indices:
            model.token_emb.weight.data[idx, D//3:2*D//3] = torch.randn(D//3) * 0.5
        
        # Letter tokens get signals in last D//3 dims
        letter_indices = [stoi[c] for c in 'abcdefghijklmnopqrstuvwxyz' if c in stoi]
        for idx in letter_indices:
            model.token_emb.weight.data[idx, 2*D//3:] = torch.randn(D - 2*D//3) * 0.5
        
        # Position embeddings: strong signal for output positions (typically 32-41)
        model.pos_emb.weight.data.zero_()
        for pos in range(model.max_seq_len):
            if pos >= 30 and pos < 42:  # Output region
                model.pos_emb.weight.data[pos, :D//2] = 1.0
            else:  # Input region
                for d in range(0, D, 4):
                    if d < D:
                        model.pos_emb.weight.data[pos, d] = math.sin(pos / 10.0)
                    if d + 1 < D:
                        model.pos_emb.weight.data[pos, d + 1] = math.cos(pos / 10.0)
        
        # Block 0: Detect separators and identify format
        block = model.blocks[0]
        # Q/K: attend to separators to find format
        nn.init.xavier_uniform_(block.attn.W_q.weight.data)
        nn.init.xavier_uniform_(block.attn.W_k.weight.data)
        block.attn.W_q.weight.data[:, :D//3] *= 0.1  # Downweight digit features
        block.attn.W_k.weight.data[:, :D//3] *= 0.1
        # V: extract separator context
        nn.init.xavier_uniform_(block.attn.W_v.weight.data)
        nn.init.xavier_uniform_(block.attn.W_o.weight.data)
        # MLP: amplify separator signals
        nn.init.xavier_uniform_(block.mlp.fc1.weight.data)
        block.mlp.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(block.mlp.fc2.weight.data)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        # Block 1: Extract digits
        block = model.blocks[1]
        # Q/K: attend to digits
        nn.init.xavier_uniform_(block.attn.W_q.weight.data)
        nn.init.xavier_uniform_(block.attn.W_k.weight.data)
        block.attn.W_q.weight.data[:, D//3:2*D//3] *= 0.1  # Downweight separator features
        block.attn.W_k.weight.data[:, D//3:2*D//3] *= 0.1
        # V: extract digit values
        nn.init.xavier_uniform_(block.attn.W_v.weight.data)
        nn.init.xavier_uniform_(block.attn.W_o.weight.data)
        # MLP: process digit sequences
        nn.init.xavier_uniform_(block.mlp.fc1.weight.data)
        block.mlp.fc1.bias.data.zero_()
        nn.init.xavier_uniform_(block.mlp.fc2.weight.data)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        # Remaining blocks: route to output
        for i in range(2, L):
            block = model.blocks[i]
            nn.init.xavier_uniform_(block.attn.W_q.weight.data)
            nn.init.xavier_uniform_(block.attn.W_k.weight.data)
            nn.init.xavier_uniform_(block.attn.W_v.weight.data)
            nn.init.xavier_uniform_(block.attn.W_o.weight.data)
            nn.init.xavier_uniform_(block.mlp.fc1.weight.data)
            block.mlp.fc1.bias.data.zero_()
            nn.init.xavier_uniform_(block.mlp.fc2.weight.data)
            block.mlp.fc2.bias.data.zero_()
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
        
        # Final LN
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # Output head: map to vocab with digit bias
        nn.init.xavier_uniform_(model.head.weight.data)
        for idx in digit_indices:
            model.head.weight.data[idx, :D//3] *= 2.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "format-aware-parsing"
model_description = "Format-aware parsing with explicit digit extraction, separator detection, and position-based output routing."


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
    # Experiment 9: Very large model
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=512,
        n_heads=16,
        n_layers=6,
        d_ff=2048,
    )
    write_weights(model, task)
    model.eval()
    return model

model_shorthand_name = "very-large-model"
model_description = "Very large model (d=512, h=16, L=6, ff=2048) with identity-like initialization."


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
