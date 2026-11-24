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
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
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
    Addition circuit v6: Direct column addition with carry chain.
    
    Strategy:
    - Use large d_model (256) for clean signal separation
    - Token emb: digit value in dims 0-9 (one-hot), special tokens get zero
    - Pos emb: encode column assignment and position identity
    - Layer 0: Output positions attend to both input digits of same column
    - Layer 1+: Carry propagation from previous column's residual
    - Final head reads result digit from dims 0-9
    
    Sequence layout for "ABCDE+FGHIJ=":
      pos 0-4: A,B,C,D,E (columns 4,3,2,1,0)
      pos 5: +
      pos 6-10: F,G,H,I,J (columns 4,3,2,1,0)
      pos 11: =
      pos 12-17: output digits (columns 0,1,2,3,4,5 - least to most significant)
    
    Wait - let me check. The target for "50494+99346=" is "149840".
    50494 + 99346 = 149840. That's 6 digits output.
    
    The output is most-significant first: "149840"
    So pos 12 = '1' (hundred-thousands), pos 13 = '4' (ten-thousands), etc.
    Output columns: pos 12=col5, 13=col4, 14=col3, 15=col2, 16=col1, 17=col0
    """
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H  # 64 per head with D=256, H=4
    
    with torch.no_grad():
        # ========== TOKEN EMBEDDINGS ==========
        # Dims 0-9: digit value (one-hot), scaled by 5.0 for strong signal
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, d] = 5.0
        # '+' (token 10) and '=' (token 11): zero digit dims
        
        # ========== POSITION EMBEDDINGS ==========
        model.pos_emb.weight.zero_()
        
        # Column mapping: which arithmetic column each position belongs to
        # Input: pos 0=col4, 1=col3, 2=col2, 3=col1, 4=col0 (first number, MSB to LSB)
        #         pos 6=col4, 7=col3, 8=col2, 9=col1, 10=col0 (second number)
        # Output: pos 12=col5, 13=col4, 14=col3, 15=col2, 16=col1, 17=col0
        
        pos_to_col = {
            0: 4, 1: 3, 2: 2, 3: 1, 4: 0,
            6: 4, 7: 3, 8: 2, 9: 1, 10: 0,
        }
        # Output positions: MSB first
        for i, p in enumerate(range(12, 18)):
            pos_to_col[p] = 5 - i  # 12->5, 13->4, ..., 17->0
        
        # Dims 10-15: column identity (one-hot), scaled by 5.0
        for pos, col in pos_to_col.items():
            if pos < model.max_seq_len:
                model.pos_emb.weight[pos, 10 + col] = 5.0
        
        # Dims 16-21: which number (0=first, 1=second, 2=output)
        for p in range(5):
            model.pos_emb.weight[p, 16] = 5.0  # first number
        for p in range(6, 11):
            model.pos_emb.weight[p, 17] = 5.0  # second number
        for p in range(12, 18):
            model.pos_emb.weight[p, 18] = 5.0  # output
        
        # Dims 22-39: position one-hot (for precise targeting)
        for pos in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[pos, 22 + pos] = 3.0
        
        # Dims 40-45: "previous output column" for carry chain
        # Output pos 13 needs carry from col 0 (pos 17's column... wait)
        # Actually carries go from LSB to MSB: col0 carry feeds col1, etc.
        # In output order (MSB first): pos 17 is col0 (computed first in autoregressive),
        # pos 16 is col1 (needs carry from col0 = pos 17), etc.
        # So pos p needs carry from pos p+1 (the previously generated digit)
        # We mark which position to look at for carry:
        # pos 16 looks at pos 17 for carry
        # pos 15 looks at pos 16 for carry
        # ...
        # pos 12 looks at pos 13 for carry
        # We'll handle this in attention.
        
        # ========== LAYER NORMS ==========
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        
        # ========== LAYER 0: Gather digits from same column ==========
        _setup_gather_digits(model, D, H, dh)
        
        # ========== LAYERS 1-4: Carry propagation ==========
        for layer_idx in range(1, model.n_layers - 1):
            _setup_carry_propagation(model, layer_idx, D, H, dh)
        
        # ========== LAYER 5 (last): Final result extraction ==========
        _setup_final_layer(model, D, H, dh)
        
        # ========== OUTPUT HEAD ==========
        model.head.weight.zero_()
        # Read dims 0-9 (digit one-hot) with strong weight
        for d in range(min(10, V)):
            model.head.weight[d, d] = 10.0


def _setup_gather_digits(model, D, H, dh):
    """
    Layer 0: For each output position, attend to both input digits in the same column.
    
    Head 0: Attend to first number's digit in same column -> store in dims 50-59
    Head 1: Attend to second number's digit in same column -> store in dims 60-69
    Head 2-3: Identity pass-through
    """
    block = model.blocks[0]
    
    with torch.no_grad():
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # --- Head 0: First number digit ---
        # Q: output positions query their column (dims 10-15)
        # K: first number positions advertise their column + number identity
        # V: read digit value (dims 0-9)
        for col in range(6):
            # Q reads column from pos emb dim 10+col
            block.attn.W_q.weight[col, 10 + col] = 10.0
            # K advertises column
            block.attn.W_k.weight[col, 10 + col] = 10.0
        # K also advertises "first number" (dim 16)
        block.attn.W_k.weight[6, 16] = 10.0
        # Q also checks for first number
        block.attn.W_q.weight[6, 16] = 10.0
        # V reads digit value -> dims 50-59
        for d in range(10):
            block.attn.W_v.weight[50 + d, d] = 5.0
        # O projects to dims 50-59
        for d in range(10):
            block.attn.W_o.weight[50 + d, d] = 1.0
        
        # --- Head 1: Second number digit ---
        h1_off = dh
        for col in range(6):
            block.attn.W_q.weight[h1_off + col, 10 + col] = 10.0
            block.attn.W_k.weight[h1_off + col, 10 + col] = 10.0
        # K advertises "second number" (dim 17)
        block.attn.W_k.weight[h1_off + 6, 17] = 10.0
        block.attn.W_q.weight[h1_off + 6, 17] = 10.0
        # V reads digit value -> dims 60-69
        for d in range(10):
            block.attn.W_v.weight[60 + d, d] = 5.0
        # O projects to dims 60-69
        for d in range(10):
            block.attn.W_o.weight[60 + d, h1_off + d] = 1.0
        
        # --- Heads 2-3: Pass-through (small weights) ---
        for h in [2, 3]:
            off = h * dh
            for i in range(dh):
                block.attn.W_v.weight[i, i] = 0.01
                block.attn.W_o.weight[i, off + i] = 0.01
        
        # --- Layer 0 MLP: Compute digit sum (no carry yet) ---
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        # For each pair (d1, d2) where d1 is in dims 50-59 and d2 in dims 60-69:
        # Compute sum = d1 + d2
        # Store (sum % 10) in dims 70-79 as one-hot
        # Store carry (sum >= 10) in dim 80
        
        unit = 0
        for d1 in range(10):
            for d2 in range(10):
                if unit >= model.d_ff:
                    break
                s = d1 + d2
                result = s % 10
                carry = 1 if s >= 10 else 0
                
                # Detect both digits present
                block.mlp.fc1.weight[unit, 50 + d1] = 20.0
                block.mlp.fc1.weight[unit, 60 + d2] = 20.0
                block.mlp.fc1.bias[unit] = -30.0
                
                # Write result digit to dims 70-79
                block.mlp.fc2.weight[70 + result, unit] = 10.0
                
                # Write carry to dim 80
                if carry:
                    block.mlp.fc2.weight[80, unit] = 10.0
                
                unit += 1


def _setup_carry_propagation(model, layer_idx, D, H, dh):
    """
    Layers 1+: Each output position looks at the previous output position's
    carry (dim 80) and adds it to the current result.
    
    The carry needs to propagate from col 0 -> col 1 -> col 2 -> ...
    In the output sequence (MSB first), col 0 is at pos 17, col 1 at pos 16, etc.
    So carry goes from pos 17 -> 16 -> 15 -> ... -> 12
    i.e., each output pos p looks at pos p+1 for carry.
    
    Head 0: Attend to position p+1 to get carry
    MLP: Add carry to current digit sum
    """
    block = model.blocks[layer_idx]
    
    with torch.no_grad():
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # --- Head 0: Get carry from previous column (position p+1) ---
        # Use position dims to match p with p+1
        # Q at position p queries for position p+1
        # K at position p advertises position p
        for p in range(12, 17):  # output positions 12-16 look at 13-17
            # Q reads position p's "I want p+1" signal
            block.attn.W_q.weight[p - 12, 22 + p + 1] = 10.0
            # K at position p+1 advertises its position
            block.attn.W_k.weight[p - 12, 22 + p + 1] = 10.0
        
        # V reads carry (dim 80) -> dim 90
        block.attn.W_v.weight[90, 80] = 5.0
        # O projects dim 90
        block.attn.W_o.weight[90, 0] = 1.0
        
        # --- Heads 1-3: Pass-through ---
        for h in [1, 2, 3]:
            off = h * dh
            for i in range(dh):
                block.attn.W_v.weight[i, i] = 0.01
                block.attn.W_o.weight[i, off + i] = 0.01
        
        # --- MLP: If carry (dim 90) is set, add 1 to result digit (dims 70-79) ---
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        # For each result digit d (0-9) in dims 70-79, with carry in dim 90:
        # New result = (d + 1) % 10, new carry = (d + 1 >= 10)
        unit = 0
        for d in range(10):
            if unit >= model.d_ff:
                break
            new_d = (d + 1) % 10
            new_carry = 1 if (d + 1) >= 10 else 0
            
            # Detect digit d and carry present
            block.mlp.fc1.weight[unit, 70 + d] = 20.0
            block.mlp.fc1.weight[unit, 90] = 20.0
            block.mlp.fc1.bias[unit] = -30.0
            
            # Subtract old digit, add new digit
            block.mlp.fc2.weight[70 + d, unit] = -10.0  # remove old
            block.mlp.fc2.weight[70 + new_d, unit] = 10.0  # add new
            
            # New carry
            if new_carry:
                block.mlp.fc2.weight[80, unit] = 10.0
            else:
                block.mlp.fc2.weight[80, unit] = -10.0  # clear carry
            
            unit += 1
        
        # Also need units that pass through when no carry
        for d in range(10):
            if unit >= model.d_ff:
                break
            # Detect digit d but NO carry
            block.mlp.fc1.weight[unit, 70 + d] = 20.0
            block.mlp.fc1.weight[unit, 90] = -20.0
            block.mlp.fc1.bias[unit] = -10.0
            
            # Keep digit d, clear carry
            block.mlp.fc2.weight[70 + d, unit] = 10.0
            block.mlp.fc2.weight[80, unit] = -10.0
            
            unit += 1


def _setup_final_layer(model, D, H, dh):
    """
    Last layer: Copy result digit from dims 70-79 to dims 0-9 for the head to read.
    Only for output positions (12-17).
    """
    block = model.blocks[model.n_layers - 1]
    
    with torch.no_grad():
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # Identity pass-through for all heads
        for h in range(H):
            off = h * dh
            for i in range(dh):
                block.attn.W_v.weight[i, i] = 0.01
                block.attn.W_o.weight[i, off + i] = 0.01
        
        # MLP: Copy dims 70-79 to dims 0-9
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        for d in range(10):
            unit = d
            block.mlp.fc1.weight[unit, 70 + d] = 20.0
            block.mlp.fc1.bias[unit] = -10.0
            block.mlp.fc2.weight[d, unit] = 10.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "addition_circuit_v6"
model_description = "Column addition with carry chain through layers. D=256, L=6, H=4, FF=1024."


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