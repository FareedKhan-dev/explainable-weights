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
    Addition circuit v4: Column-by-column addition with explicit carry.
    
    Sequence layout for "12345+67890=":
      pos 0-4: first number digits (ten-thousands to ones)
      pos 5: '+'
      pos 6-10: second number digits (ten-thousands to ones)
      pos 11: '='
      pos 12-17: output (6 digits, hundred-thousands to ones)
    
    Embedding layout (d_model=128):
      dims 0-9: one-hot digit value
      dims 10-27: one-hot position (0-17)
      dims 30-35: one-hot "output column" (which column of addition this position contributes to)
        col 0 = ones, col 1 = tens, ..., col 4 = ten-thousands, col 5 = hundred-thousands
      dim 40: is_digit flag (1 if position contains a digit, 0 for + and =)
      dim 41: is_input flag (1 for positions 0-10)
      dim 42: is_output flag (1 for positions 12-17)
      dim 50: numerical digit value (0-9 scaled to [0,1])
    """
    D = model.d_model
    V = model.vocab_size  # 12
    H = model.n_heads  # 4
    dh = D // H  # 32
    
    # === Token Embeddings ===
    with torch.no_grad():
        model.token_emb.weight.zero_()
        # Digits 0-9
        for d in range(10):
            model.token_emb.weight[d, d] = 1.0  # one-hot digit value
            model.token_emb.weight[d, 40] = 1.0  # is_digit
            model.token_emb.weight[d, 41] = 1.0  # is_input
            model.token_emb.weight[d, 50] = d / 9.0  # scaled digit value
        # '+' (vocab 10) and '=' (vocab 11)
        model.token_emb.weight[10, 40] = 0.0
        model.token_emb.weight[10, 41] = 1.0
        model.token_emb.weight[11, 40] = 0.0
        model.token_emb.weight[11, 41] = 1.0
    
    # === Position Embeddings ===
    with torch.no_grad():
        model.pos_emb.weight.zero_()
        # Map each position to its "addition column"
        # pos 0 → col 4 (ten-thousands), pos 1 → col 3, ..., pos 4 → col 0 (ones)
        # pos 6 → col 4, pos 7 → col 3, ..., pos 10 → col 0
        # pos 13 → col 4, pos 14 → col 3, ..., pos 17 → col 0
        # pos 12 → col 5 (hundred-thousands, just carry)
        pos_to_col = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0,
                      6: 4, 7: 3, 8: 2, 9: 1, 10: 0,
                      13: 4, 14: 3, 15: 2, 16: 1, 17: 0, 12: 5}
        
        for pos in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[pos, 10 + pos] = 1.0  # one-hot position
            if pos in pos_to_col:
                col = pos_to_col[pos]
                model.pos_emb.weight[pos, 30 + col] = 1.0  # column index
            if pos >= 12:
                model.pos_emb.weight[pos, 42] = 1.0  # is_output
    
    # === LayerNorm: set to identity-like ===
    for block in model.blocks:
        with torch.no_grad():
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()
    
    with torch.no_grad():
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
    
    # === Layer 0: Gather digit pair for each output position ===
    # Goal: each output position should attend to the two input digits in its column.
    # Use head 0 to attend to the first number's digit in the same column.
    # Use head 1 to attend to the second number's digit in the same column.
    # Heads 2-3: identity pass-through.
    
    _setup_attention_layer0(model, D, H, dh)
    
    # === Layer 0 MLP: compute sum of the two digits ===
    _setup_mlp_layer0(model, D)
    
    # === Layers 1-3: Carry propagation and final output ===
    # For now, initialize with identity-like transforms
    for layer_idx in range(1, model.n_layers):
        block = model.blocks[layer_idx]
        with torch.no_grad():
            # Attention: pass-through
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            # Make attention near-identity by having Q=K match on position dims
            # and V=O pass through
            for i in range(D):
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0
            
            # MLP: identity
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
    
    # === Final head: map digit encoding to vocab logits ===
    with torch.no_grad():
        model.head.weight.zero_()
        for d in range(10):
            model.head.weight[d, d] = 10.0  # strong signal for digit prediction


def _setup_attention_layer0(model, D, H, dh):
    """Layer 0 attention: each output pos attends to its column's input digits."""
    block = model.blocks[0]
    with torch.no_grad():
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # Head 0: output positions attend to first number's digit in same column
        # Q: read column index (dims 30-35) + is_output (dim 42)
        # K: write column index (dims 30-35) + is_input (dim 41) + position info
        # V: read digit value (dims 0-9)
        
        # For output position p, we want it to attend to the first-number position
        # that has the same column. 
        # Output pos 17 → col 0 → first number pos 4
        # Output pos 16 → col 1 → first number pos 3
        # Output pos 15 → col 2 → first number pos 2
        # Output pos 14 → col 3 → first number pos 1
        # Output pos 13 → col 4 → first number pos 0
        
        # Make Q read column from pos embedding and K write column
        # For head 0 (dims 0*dh to 1*dh):
        h0_start = 0
        h0_end = dh
        
        # Q for head 0: query based on column index
        for col in range(6):
            block.attn.W_q.weight[h0_start + col, 30 + col] = 5.0  # high weight on column
        
        # K for head 0: key based on column index + is_input flag
        for col in range(6):
            block.attn.W_k.weight[h0_start + col, 30 + col] = 5.0
        block.attn.W_k.weight[h0_start + 6, 41] = 5.0  # prefer input positions
        
        # V for head 0: extract digit value (dims 0-9) -> store in dims 60-69
        for d in range(10):
            block.attn.W_v.weight[60 + d, d] = 1.0
        
        # O for head 0: pass through dims 60-69
        for d in range(10):
            block.attn.W_o.weight[60 + d, h0_start + d] = 1.0
        
        # Head 1: output positions attend to second number's digit in same column
        h1_start = dh
        h1_end = 2 * dh
        
        # Q for head 1: query based on column index  
        for col in range(6):
            block.attn.W_q.weight[h1_start + col, 30 + col] = 5.0
        
        # K for head 1: key based on column index + position >= 6 (second number)
        for col in range(6):
            block.attn.W_k.weight[h1_start + col, 30 + col] = 5.0
        block.attn.W_k.weight[h1_start + 6, 41] = 5.0  # prefer input positions
        # We need to distinguish first vs second number... use position dims
        # Second number is at positions 6-10
        for p in range(6, 11):
            block.attn.W_k.weight[h1_start + 7, 10 + p] = 5.0
        
        # V for head 1: extract digit value -> store in dims 70-79
        for d in range(10):
            block.attn.W_v.weight[70 + d, d] = 1.0
        
        # O for head 1: pass through dims 70-79
        for d in range(10):
            block.attn.W_o.weight[70 + d, h1_start + d] = 1.0
        
        # Heads 2-3: identity pass-through
        for h in range(2, H):
            hs = h * dh
            for i in range(min(dh, D)):
                block.attn.W_v.weight[i, i] = 0.1
                block.attn.W_o.weight[i, hs + i % dh] = 0.1


def _setup_mlp_layer0(model, D):
    """Layer 0 MLP: compute sum of two digits stored in dims 60-69 and 70-79.
    Store result (sum mod 10) in dims 80-89 and carry in dim 90."""
    block = model.blocks[0]
    with torch.no_grad():
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        # The MLP sees the LayerNorm'd version of the hidden state.
        # After attention, dims 60-69 have first digit one-hot, dims 70-79 have second digit one-hot.
        # 
        # Strategy: for each pair (d1, d2), create a ReLU unit that fires when both
        # dims d1 (in 60-69) and d2 (in 70-79) are active, and output the sum.
        #
        # But this requires 100 hidden units for all pairs, and d_ff=512 is enough.
        #
        # For each (d1, d2) pair where d1+d2 = s:
        #   hidden unit fires when dim (60+d1) > 0 AND dim (70+d2) > 0
        #   outputs to dim (80 + s%10) and dim 90 if s >= 10
        
        d_ff = model.d_ff
        unit = 0
        
        for d1 in range(10):
            for d2 in range(10):
                if unit >= d_ff:
                    break
                s = d1 + d2
                # Hidden unit: activate when dim(60+d1) and dim(70+d2) are both positive
                # fc1: w[60+d1] = 1, w[70+d2] = 1, bias = -1 (fires when both > threshold)
                block.mlp.fc1.weight[unit, 60 + d1] = 2.0
                block.mlp.fc1.weight[unit, 70 + d2] = 2.0
                block.mlp.fc1.bias[unit] = -1.5  # threshold
                
                # fc2: output to sum digit and carry
                result_digit = s % 10
                block.mlp.fc2.weight[80 + result_digit, unit] = 1.0
                if s >= 10:
                    block.mlp.fc2.weight[90, unit] = 1.0  # carry
                
                unit += 1


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "addition_circuit_v4"
model_description = "Column-by-column addition with explicit digit gathering via attention and sum computation via MLP lookup."


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
    print(f"DEBUG: vocab_size={task.vocab_size}, seq_len={task.seq_len}, prompt_len={task.prompt_len}, answer_len={task.answer_len}")
    print(f"DEBUG: vocab={task.vocab}")
    print(f"DEBUG: stoi={task.stoi}")
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
