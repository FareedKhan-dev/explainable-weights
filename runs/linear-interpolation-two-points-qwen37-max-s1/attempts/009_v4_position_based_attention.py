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
    """Configurable causal transformer."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
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
    Linear interpolation: x1,y1;x2,y2;x3= -> answer (2 digits)
    Formula: answer = y1 + (y2-y1)*(x3-x1)/(x2-x1)
    
    Prompt structure (12 chars, positions 0-11):
      Pos 0: x1 (single digit)
      Pos 1: ','
      Pos 2: y1_tens
      Pos 3: y1_ones  
      Pos 4: ';'
      Pos 5: x2
      Pos 6: ','
      Pos 7: y2_tens
      Pos 8: y2_ones
      Pos 9: ';'
      Pos 10: x3
      Pos 11: '='
    
    Output (2 chars):
      Pos 12: answer_tens
      Pos 13: answer_ones
    
    Strategy: Use attention to extract 7 input values, then compute interpolation.
    """
    D = model.d_model  # 256
    V = model.vocab_size  # 13
    H = model.n_heads  # 8
    dh = D // H  # 32
    FF = model.d_ff  # 1024
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDINGS ----
    # Encode digit value as scalar
    for i in range(10):
        model.token_emb.weight.data[i, 0] = float(i)
        model.token_emb.weight.data[i, 1] = 10.0  # is_digit flag
    
    # ---- POSITION EMBEDDINGS ----
    # Role flags for each input position (dims 50-56)
    input_positions = [0, 2, 3, 5, 7, 8, 10]  # x1, y1t, y1o, x2, y2t, y2o, x3
    for idx, pos in enumerate(input_positions):
        model.pos_emb.weight.data[pos, 50 + idx] = 100.0  # Strong role signal
    
    # Output position flags (dims 57-58)
    model.pos_emb.weight.data[12, 57] = 100.0  # output_tens position
    model.pos_emb.weight.data[13, 58] = 100.0  # output_ones position
    
    # ---- BLOCK 0: Extract 7 input values via attention ----
    block0 = model.blocks[0]
    
    # Use 7 heads to extract 7 values
    # Each head: query for role flag, key advertises role, value carries digit
    for h in range(7):
        role_dim = 50 + h  # Which role to look for
        
        # Query: look for output position (dim 57 for tens, 58 for ones)
        block0.attn.W_q.weight.data[h*dh, 57] = 1.0
        
        # Key: advertise own role
        block0.attn.W_k.weight.data[h*dh, role_dim] = 1.0
        
        # Value: carry digit value (dim 0)
        block0.attn.W_v.weight.data[h*dh, 0] = 1.0
        
        # Output: write to dedicated channel (dims 100-106)
        block0.attn.W_o.weight.data[100 + h, h*dh] = 1.0
    
    # Head 7: identity (small weights to avoid zero attention)
    block0.attn.W_q.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_k.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_v.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_o.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    
    # Block 0 MLP: identity (small)
    block0.mlp.fc1.weight.data[:FF, :D] = torch.eye(FF, D) * 0.01
    block0.mlp.fc2.weight.data[:D, :FF] = torch.eye(D, FF) * 0.01
    
    # ---- BLOCKS 1-3: Identity (pass through extracted values) ----
    for block in model.blocks[1:]:
        block.attn.W_q.weight.data[:D, :D] = torch.eye(D) * 0.01
        block.attn.W_k.weight.data[:D, :D] = torch.eye(D) * 0.01
        block.attn.W_v.weight.data[:D, :D] = torch.eye(D) * 0.01
        block.attn.W_o.weight.data[:D, :D] = torch.eye(D) * 0.01
        block.mlp.fc1.weight.data[:FF, :D] = torch.eye(FF, D) * 0.01
        block.mlp.fc2.weight.data[:D, :FF] = torch.eye(D, FF) * 0.01
    
    # ---- OUTPUT HEAD ----
    # For now, output digit 0 (baseline)
    model.head.weight.data[0, 0] = 1.0


model_shorthand_name = "v4_position_based_attention"
model_description = "Position-based attention to extract 7 input values at output positions"


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
