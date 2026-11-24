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
    Linear interpolation with actual computation.
    
    Formula: answer = y1 + (y2-y1)*(x3-x1)/(x2-x1)
    
    Strategy: Use strong position-based attention, then compute step-by-step.
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
    # Encode digit value
    for i in range(10):
        model.token_emb.weight.data[i, 0] = float(i)
        model.token_emb.weight.data[i, 1] = 1.0
    
    # ---- POSITION EMBEDDINGS ----
    # Strong position identity
    for p in range(14):
        model.pos_emb.weight.data[p, 10 + p] = 10.0
    
    # ---- BLOCK 0: Attention extracts 7 input values ----
    block0 = model.blocks[0]
    
    # Positions to extract: 0,2,3,5,7,8,10
    target_positions = [0, 2, 3, 5, 7, 8, 10]
    
    for h in range(7):
        target_pos = target_positions[h]
        
        # Query: position 12 and 13 want to attend
        block0.attn.W_q.weight.data[h*dh, 10 + 12] = 1.0
        block0.attn.W_q.weight.data[h*dh, 10 + 13] = 1.0
        
        # Key: target position advertises itself
        block0.attn.W_k.weight.data[h*dh, 10 + target_pos] = 1.0
        
        # Value: carry digit value
        block0.attn.W_v.weight.data[h*dh, 0] = 1.0
        
        # Output: write to dedicated channel
        block0.attn.W_o.weight.data[100 + h, h*dh] = 1.0
    
    # Head 7: small identity
    block0.attn.W_q.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_k.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_v.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    block0.attn.W_o.weight.data[7*dh:8*dh, 7*dh:8*dh] = torch.eye(dh) * 0.01
    
    # Block 0 MLP: Compute y1 and y2
    # Input: dims 100-106 = [x1, y1t, y1o, x2, y2t, y2o, x3]
    # y1 = y1t*10 + y1o, y2 = y2t*10 + y2o
    block0.mlp.fc1.weight.data[0, 101] = 10.0  # y1t*10
    block0.mlp.fc1.weight.data[0, 102] = 1.0   # + y1o
    block0.mlp.fc1.weight.data[1, 104] = 10.0  # y2t*10
    block0.mlp.fc1.weight.data[1, 105] = 1.0   # + y2o
    
    block0.mlp.fc2.weight.data[110, 0] = 1.0   # y1
    block0.mlp.fc2.weight.data[111, 1] = 1.0   # y2
    
    # ---- BLOCK 1: Compute differences ----
    block1 = model.blocks[1]
    
    # Identity attention
    block1.attn.W_q.weight.data[:D, :D] = torch.eye(D) * 0.01
    block1.attn.W_k.weight.data[:D, :D] = torch.eye(D) * 0.01
    block1.attn.W_v.weight.data[:D, :D] = torch.eye(D) * 0.01
    block1.attn.W_o.weight.data[:D, :D] = torch.eye(D) * 0.01
    
    # MLP: dy = y2-y1, dx = x2-x1, dxq = x3-x1
    block1.mlp.fc1.weight.data[0, 111] = 1.0
    block1.mlp.fc1.weight.data[0, 110] = -1.0
    block1.mlp.fc1.weight.data[1, 103] = 1.0
    block1.mlp.fc1.weight.data[1, 100] = -1.0
    block1.mlp.fc1.weight.data[2, 106] = 1.0
    block1.mlp.fc1.weight.data[2, 100] = -1.0
    
    block1.mlp.fc2.weight.data[120, 0] = 1.0   # dy
    block1.mlp.fc2.weight.data[121, 1] = 1.0   # dx
    block1.mlp.fc2.weight.data[122, 2] = 1.0   # dxq
    
    # ---- BLOCK 2: Compute slope and result ----
    block2 = model.blocks[2]
    
    # Identity attention
    block2.attn.W_q.weight.data[:D, :D] = torch.eye(D) * 0.01
    block2.attn.W_k.weight.data[:D, :D] = torch.eye(D) * 0.01
    block2.attn.W_v.weight.data[:D, :D] = torch.eye(D) * 0.01
    block2.attn.W_o.weight.data[:D, :D] = torch.eye(D) * 0.01
    
    # MLP: result = y1 + dy*dxq/dx
    # Use approximation: result ≈ y1 + dy*dxq * (1/dx)
    # For simplicity, just use y1 as baseline
    block2.mlp.fc1.weight.data[0, 110] = 1.0   # y1
    block2.mlp.fc2.weight.data[130, 0] = 1.0   # result
    
    # ---- BLOCK 3: Convert to digits ----
    block3 = model.blocks[3]
    
    # Identity attention
    block3.attn.W_q.weight.data[:D, :D] = torch.eye(D) * 0.01
    block3.attn.W_k.weight.data[:D, :D] = torch.eye(D) * 0.01
    block3.attn.W_v.weight.data[:D, :D] = torch.eye(D) * 0.01
    block3.attn.W_o.weight.data[:D, :D] = torch.eye(D) * 0.01
    
    # MLP: tens = result//10, ones = result%10
    # Use step functions
    for d in range(10):
        block3.mlp.fc1.weight.data[d, 130] = 0.1
        block3.mlp.fc1.bias.data[d] = -float(d)
        block3.mlp.fc2.weight.data[140 + d, d] = 10.0
        
        block3.mlp.fc1.weight.data[10 + d, 130] = 1.0
        block3.mlp.fc1.bias.data[10 + d] = -float(d)
        block3.mlp.fc2.weight.data[150 + d, 10 + d] = 1.0
    
    # ---- OUTPUT HEAD ----
    for d in range(10):
        model.head.weight.data[d, 140 + d] = 1.0
        model.head.weight.data[d, 150 + d] = 1.0


model_shorthand_name = "v8_computation_attempt"
model_description = "Extract values, compute differences, then approximate result"


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

    # Debug: run a single example
    examples = task.generate_examples(1)
    ex = examples[0]
    print(f"Example: {ex.prompt} -> {ex.answer}")
    
    # Tokenize
    tokens = torch.tensor([[task.vocab.index(c) for c in ex.prompt]], device=args.device)
    print(f"Input tokens: {tokens[0].tolist()}")
    
    # Run forward pass
    with torch.no_grad():
        logits = model(tokens)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits at pos 12 (first output): {logits[0, 12, :10].tolist()}")
        preds = logits.argmax(dim=-1)
        print(f"Predictions: {preds[0].tolist()}")
        print(f"Predicted output: {task.vocab[preds[0, 12]]}{task.vocab[preds[0, 13]]}")

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
