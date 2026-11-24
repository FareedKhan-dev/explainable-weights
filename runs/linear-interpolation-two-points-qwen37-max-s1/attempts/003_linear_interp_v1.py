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
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
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
    Linear interpolation: x1,y1;x2,y2;x3= -> answer (2 digits)
    
    Prompt positions (0-indexed):
      0: x1 digit
      1: ','
      2: y1 tens
      3: y1 ones
      4: ';'
      5: x2 digit
      6: ','
      7: y2 tens
      8: y2 ones
      9: ';'
      10: x3 digit
      11: '='
      12: answer tens (output)
      13: answer ones (output)
    
    Formula: answer = y1 + (y2-y1)*(x3-x1)/(x2-x1)
    """
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H
    FF = model.d_ff
    
    # Vocab: ['0','1','2','3','4','5','6','7','8','9',',',';','=']
    # indices: 0   1   2   3   4   5   6   7   8   9  10  11  12
    
    # Zero everything first
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDINGS ----
    # Encode digit value in dimension 0 (for digits 0-9)
    # Dimension 1: is_digit flag
    # Dimension 2: is_comma, 3: is_semicolon, 4: is_equals
    tok_emb = model.token_emb.weight.data  # (V, D)
    for i in range(10):
        tok_emb[i, 0] = float(i)  # digit value
        tok_emb[i, 1] = 1.0       # is_digit
    tok_emb[10, 2] = 1.0  # comma
    tok_emb[11, 3] = 1.0  # semicolon
    tok_emb[12, 4] = 1.0  # equals
    
    # ---- POSITION EMBEDDINGS ----
    # Encode position identity and role
    # Use dimensions 5-18 for one-hot position encoding (14 positions)
    pos_emb = model.pos_emb.weight.data  # (max_seq_len, D)
    for p in range(14):
        pos_emb[p, 5 + p] = 1.0  # one-hot position
    
    # Also encode role in dedicated dimensions:
    # dim 19: is_x1_pos (pos 0)
    # dim 20: is_y1_tens_pos (pos 2)
    # dim 21: is_y1_ones_pos (pos 3)
    # dim 22: is_x2_pos (pos 5)
    # dim 23: is_y2_tens_pos (pos 7)
    # dim 24: is_y2_ones_pos (pos 8)
    # dim 25: is_x3_pos (pos 10)
    # dim 26: is_output_tens_pos (pos 12)
    # dim 27: is_output_ones_pos (pos 13)
    pos_emb[0, 19] = 1.0   # x1
    pos_emb[2, 20] = 1.0   # y1 tens
    pos_emb[3, 21] = 1.0   # y1 ones
    pos_emb[5, 22] = 1.0   # x2
    pos_emb[7, 23] = 1.0   # y2 tens
    pos_emb[8, 24] = 1.0   # y2 ones
    pos_emb[10, 25] = 1.0  # x3
    pos_emb[12, 26] = 1.0  # output tens
    pos_emb[13, 27] = 1.0  # output ones
    
    # Now we need attention at output positions to gather the input values.
    # Strategy: Use layer 1 attention to copy x1, y1t, y1o, x2, y2t, y2o, x3 to output positions.
    # Then use MLP to compute the answer.
    
    # ---- LAYER 1 ATTENTION ----
    # At position 12 and 13, we want to attend to positions 0,2,3,5,7,8,10
    # We'll use multiple heads to gather different values.
    
    # Head 0: gather x1 (from pos 0)
    # Head 1: gather y1_tens (from pos 2)
    # Head 2: gather y1_ones (from pos 3) 
    # Head 3: gather x2 (from pos 5)
    # With 4 heads, we can only gather 4 values per layer.
    # We need 7 values. So let's use more heads or more layers.
    
    # Let me use 8 heads and gather all 7 values + a bias term.
    # But we only have n_heads=4. Let me change architecture...
    # Actually the model has n_heads=4. Let me use 2 attention layers (2 blocks).
    
    # Block 0, attention: gather x1, y1_tens, y1_ones, x2 at output positions
    # Block 1, attention: gather x2 (already have), y2_tens, y2_ones, x3 at output positions
    # Then MLP layers do computation.
    
    # Actually, let me think about this differently. 
    # With causal attention at position 12, we can attend to positions 0-12.
    # Position 12 is the first output position - we haven't generated anything yet.
    # So at position 12, we can see positions 0-11 (the full input + '=').
    # At position 13, we can see positions 0-12 (input + first output digit).
    
    # The challenge: we need to compute a complex formula.
    # Let me use a more direct approach with a huge MLP.
    
    # Approach: Use attention to aggregate all 7 input digit values to output positions,
    # then use MLP to compute the answer via a lookup table.
    
    # For Block 0 attention, let's make all heads at output positions attend uniformly
    # to all input positions. Then the value vectors carry position-specific information.
    
    # Actually, a simpler approach: at output position, attend to ALL input positions.
    # The value of each input position encodes (position_id, digit_value).
    # Then the aggregated value is a sum/average of all inputs.
    # This doesn't separate the values cleanly.
    
    # Better: use one head per input value we want to extract.
    # With 4 heads per layer and 4 layers, we have 16 head-slots.
    # We need to extract 7 values and compute with them.
    
    # Let me redesign: use 8 heads.
    pass


model_shorthand_name = "linear_interp_v1"
model_description = "First attempt at linear interpolation with hand-coded weights"


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
