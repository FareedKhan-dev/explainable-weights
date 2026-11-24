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

import numpy as np
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
    """Configurable causal transformer for hand-written addition."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
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
    """
    Addition circuit v5: Simplified column-based addition.
    
    Key insight: Use position embeddings to encode which "column" each position belongs to,
    then use attention to gather digits from the same column.
    
    Sequence: "12345+67890="
      pos 0-4: first number (ten-thousands to ones) -> columns 4,3,2,1,0
      pos 5: '+'
      pos 6-10: second number (ten-thousands to ones) -> columns 4,3,2,1,0
      pos 11: '='
      pos 12-17: output (6 digits) -> columns 5,4,3,2,1,0
    """
    D = model.d_model
    V = model.vocab_size
    
    with torch.no_grad():
        # Token embeddings: encode digit value in dims 0-9
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, d] = 1.0
        
        # Position embeddings: encode column index in dims 10-15
        # and position index in dims 20-37
        model.pos_emb.weight.zero_()
        
        # Column mapping
        pos_to_col = {
            0: 4, 1: 3, 2: 2, 3: 1, 4: 0,      # first number
            6: 4, 7: 3, 8: 2, 9: 1, 10: 0,     # second number
            13: 4, 14: 3, 15: 2, 16: 1, 17: 0, # output (excluding pos 12)
            12: 5                                # hundred-thousands (carry)
        }
        
        for pos in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[pos, 20 + pos] = 1.0  # position one-hot
            if pos in pos_to_col:
                col = pos_to_col[pos]
                model.pos_emb.weight[pos, 10 + col] = 2.0  # column (stronger signal)
        
        # LayerNorms: identity
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()
        
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
    
    # === Layer 0: Gather digits from same column ===
    _setup_layer0_attention(model, D)
    _setup_layer0_mlp(model, D)
    
    # === Layers 1-2: Identity pass-through for now ===
    for layer_idx in range(1, model.n_layers):
        block = model.blocks[layer_idx]
        with torch.no_grad():
            # Attention: near-identity
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            for i in range(D):
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0
            
            # MLP: identity
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
    
    # === Final head ===
    with torch.no_grad():
        model.head.weight.zero_()
        for d in range(min(10, V)):
            model.head.weight[d, d] = 10.0


def _setup_layer0_attention(model, D):
    """
    Layer 0 attention: output positions attend to input digits in same column.
    
    For output position p (12-17), attend to all input positions (0-10) with same column.
    Store the two digits in dims 40-49 (first) and 50-59 (second).
    """
    block = model.blocks[0]
    H = model.n_heads
    dh = D // H
    
    with torch.no_grad():
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # Use 4 heads to gather different information:
        # Head 0: output pos attend to first number digit in same column
        # Head 1: output pos attend to second number digit in same column
        # Head 2-3: pass-through
        
        # Head 0: first number
        h0_start = 0
        # Q: read column (dims 10-15)
        for col in range(6):
            block.attn.W_q.weight[h0_start + col, 10 + col] = 10.0
        # K: write column + position in first number (0-4)
        for col in range(6):
            block.attn.W_k.weight[h0_start + col, 10 + col] = 10.0
        for p in range(5):  # positions 0-4
            block.attn.W_k.weight[h0_start + 6, 20 + p] = 10.0
        # V: read digit value (0-9) -> dims 40-49
        for d in range(10):
            block.attn.W_v.weight[40 + d, d] = 1.0
        # O: pass dims 40-49
        for d in range(10):
            block.attn.W_o.weight[40 + d, h0_start + d] = 1.0
        
        # Head 1: second number
        h1_start = dh
        # Q: read column
        for col in range(6):
            block.attn.W_q.weight[h1_start + col, 10 + col] = 10.0
        # K: write column + position in second number (6-10)
        for col in range(6):
            block.attn.W_k.weight[h1_start + col, 10 + col] = 10.0
        for p in range(6, 11):  # positions 6-10
            block.attn.W_k.weight[h1_start + 6, 20 + p] = 10.0
        # V: read digit value -> dims 50-59
        for d in range(10):
            block.attn.W_v.weight[50 + d, d] = 1.0
        # O: pass dims 50-59
        for d in range(10):
            block.attn.W_o.weight[50 + d, h1_start + d] = 1.0
        
        # Heads 2-3: pass-through
        for h in [2, 3]:
            hs = h * dh
            for i in range(min(dh, D)):
                block.attn.W_v.weight[i, i] = 0.01
                block.attn.W_o.weight[i, hs + i % dh] = 0.01


def _setup_layer0_mlp(model, D):
    """
    Layer 0 MLP: compute sum of two digits (dims 40-49 and 50-59).
    Store result mod 10 in dims 60-69 and carry in dim 70.
    """
    block = model.blocks[0]
    d_ff = model.d_ff
    
    with torch.no_grad():
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        # After LayerNorm, one-hot values get normalized. Use very strong weights.
        # For each pair (d1, d2), create a unit that fires when both are present
        unit = 0
        for d1 in range(10):
            for d2 in range(10):
                if unit >= d_ff:
                    break
                
                s = d1 + d2
                result = s % 10
                
                # Activate when dim(40+d1) and dim(50+d2) are both positive
                # Use very strong weights to overcome normalization
                block.mlp.fc1.weight[unit, 40 + d1] = 20.0
                block.mlp.fc1.weight[unit, 50 + d2] = 20.0
                block.mlp.fc1.bias[unit] = -15.0
                
                # Output result digit with strong weight
                block.mlp.fc2.weight[60 + result, unit] = 20.0
                
                # Output carry if needed
                if s >= 10:
                    block.mlp.fc2.weight[70, unit] = 20.0
                
                unit += 1


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "addition_circuit_v5"
model_description = "Simplified column-based addition with attention gathering digits from same column and MLP computing sum."


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
    print(f"DEBUG: vocab_size={task.vocab_size}, seq_len={task.seq_len}")
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
