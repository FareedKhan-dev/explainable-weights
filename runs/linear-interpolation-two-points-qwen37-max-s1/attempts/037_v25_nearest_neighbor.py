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
        # Disable LayerNorm for hand-coded weights
        x = x + self.attn(x)
        x = x + self.mlp(x)
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
    v25: Nearest-neighbor interpolation. Output y1 if x3 closer to x1, else y2.
    
    Input format: x1,y1_tens y1_ones;x2,y2_tens y2_ones;x3=
    Positions:    0  1  2    3   4 5  6  7    8   9 10 11
    Output generated at pos 11 (tens digit) and pos 12 (ones digit).
    
    Block 0: 8 heads extract 7 input values + 1 spare
    Block 1 MLP: compute selector = |x3-x2| - |x3-x1|
    Block 1 attn: route selector to pos 11,12
    Block 2 attn: route y1 or y2 digits to pos 11,12 based on selector
    Block 3 attn: final routing to output dims 0-9 (tens) and 10-19 (ones)
    """
    D = model.d_model   # 256
    V = model.vocab_size # 13
    H = model.n_heads    # 8
    dh = D // H          # 32
    FF = model.d_ff      # 1024
    
    # Zero everything first
    for p in model.parameters():
        p.data.zero_()
    
    # LayerNorms to identity
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Token embeddings: digit d -> one-hot in dim d (scale 10 for strong signal)
    for d in range(10):
        model.token_emb.weight.data[d, d] = 10.0
    
    # Position embeddings: unique flag per position in dims 100+
    for pos in range(14):
        model.pos_emb.weight.data[pos, 100 + pos] = 5.0
    
    # ==================== BLOCK 0: Extract 7 values ====================
    blk0 = model.blocks[0]
    # Positions: x1=0, y1t=2, y1o=3, x2=5, y2t=7, y2o=8, x3=10
    # Extraction targets (one-hot in dims 0-9):
    #   x1 -> dims 0-9, y1t -> dims 10-19, y1o -> dims 20-29
    #   x2 -> dims 30-39, y2t -> dims 40-49, y2o -> dims 50-59
    #   x3 -> dims 60-69
    # All queries use pos 11 flag (dim 111) so extraction happens at pos 11
    
    extractions = [
        # (head_idx, source_pos, target_dim_start)
        (0, 0, 0),    # x1
        (1, 2, 10),   # y1_tens
        (2, 3, 20),   # y1_ones
        (3, 5, 30),   # x2
        (4, 7, 40),   # y2_tens
        (5, 8, 50),   # y2_ones
        (6, 10, 60),  # x3
    ]
    
    for head_idx, src_pos, tgt_start in extractions:
        h_off = head_idx * dh
        # Query: pos 11 flag (dim 111)
        blk0.attn.W_q.weight.data[h_off, 111] = 10.0
        # Key: source position flag
        blk0.attn.W_k.weight.data[h_off, 100 + src_pos] = 10.0
        # Value: copy digit one-hot to target dims
        for d in range(10):
            blk0.attn.W_v.weight.data[h_off + d, d] = 10.0
            blk0.attn.W_o.weight.data[tgt_start + d, h_off + d] = 1.0
    
    # ==================== BLOCK 1: Compute selector ====================
    blk1 = model.blocks[1]
    # Selector = |x3-x2| - |x3-x1|
    # If selector > 0: x3 farther from x2 -> closer to x1 -> use y1
    # If selector < 0: x3 farther from x1 -> closer to x2 -> use y2
    
    mlp = blk1.mlp
    # fc1 neuron 200: x3 - x1 (from dims 60-69 and 0-9)
    for d in range(10):
        mlp.fc1.weight.data[200, 60 + d] = d   # +x3
        mlp.fc1.weight.data[200, d] = -d        # -x1
    # After ReLU: max(0, x3-x1)
    
    # fc1 neuron 201: x1 - x3
    for d in range(10):
        mlp.fc1.weight.data[201, d] = d          # +x1
        mlp.fc1.weight.data[201, 60 + d] = -d    # -x3
    # After ReLU: max(0, x1-x3)
    
    # fc1 neuron 202: x3 - x2
    for d in range(10):
        mlp.fc1.weight.data[202, 60 + d] = d
        mlp.fc1.weight.data[202, 30 + d] = -d
    
    # fc1 neuron 203: x2 - x3
    for d in range(10):
        mlp.fc1.weight.data[203, 30 + d] = d
        mlp.fc1.weight.data[203, 60 + d] = -d
    
    # fc2: combine to get |x3-x1| and |x3-x2|
    # neuron 200 -> |x3-x1| = relu(x3-x1) + relu(x1-x3)
    mlp.fc2.weight.data[200, 200] = 1.0   # relu(x3-x1)
    mlp.fc2.weight.data[200, 201] = 1.0   # relu(x1-x3)
    # neuron 201 -> |x3-x2|
    mlp.fc2.weight.data[201, 202] = 1.0
    mlp.fc2.weight.data[201, 203] = 1.0
    
    # selector = |x3-x2| - |x3-x1| -> store in neuron 202 output
    # But fc2 already used 200,201 for distances. Let's use neuron 204:
    # Actually, we need another pass. Let me restructure:
    # fc2 neuron 200: |x3-x1|
    # fc2 neuron 201: |x3-x2|
    # Then we need selector = 201 - 200, but that requires another layer.
    # Instead, compute selector directly in fc1/fc2 differently.
    
    # Simpler: just compute selector in one pass
    # Reset: use neurons 200-203 for intermediate, 204-205 for selector parts
    mlp.fc2.weight.data.zero_()  # clear fc2
    
    # fc2 neuron 200: relu(x3-x1) passed through
    mlp.fc2.weight.data[200, 200] = 1.0
    # fc2 neuron 201: relu(x1-x3) passed through
    mlp.fc2.weight.data[201, 201] = 1.0
    # fc2 neuron 202: relu(x3-x2) passed through
    mlp.fc2.weight.data[202, 202] = 1.0
    # fc2 neuron 203: relu(x2-x3) passed through
    mlp.fc2.weight.data[203, 203] = 1.0
    
    # Now we need a second MLP pass (block 2) to compute selector
    # Block 1 attention: broadcast selector info and route y1/y2 to pos 11,12
    
    # Head 0: pos 11 queries y1_tens (dims 10-19) - always copy y1 to dims 70-79
    blk1.attn.W_q.weight.data[0, 111] = 10.0    # pos 11 flag
    blk1.attn.W_k.weight.data[0, 111] = 10.0    # self-attend (pos 11 has y1t in dims 10-19)
    for d in range(10):
        blk1.attn.W_v.weight.data[d, 10 + d] = 10.0  # read y1_tens
        blk1.attn.W_o.weight.data[70 + d, d] = 1.0    # write to dims 70-79
    
    # Head 1: pos 11 copies y2_tens (dims 40-49) to dims 80-89
    blk1.attn.W_q.weight.data[dh, 111] = 10.0
    blk1.attn.W_k.weight.data[dh, 111] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[dh + d, 40 + d] = 10.0
        blk1.attn.W_o.weight.data[80 + d, dh + d] = 1.0
    
    # Head 2: pos 11 copies y1_ones (dims 20-29) to dims 90-99
    blk1.attn.W_q.weight.data[2*dh, 111] = 10.0
    blk1.attn.W_k.weight.data[2*dh, 111] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[2*dh + d, 20 + d] = 10.0
        blk1.attn.W_o.weight.data[90 + d, 2*dh + d] = 1.0
    
    # Head 3: pos 11 copies y2_ones (dims 50-59) to dims 100-109
    blk1.attn.W_q.weight.data[3*dh, 111] = 10.0
    blk1.attn.W_k.weight.data[3*dh, 111] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[3*dh + d, 50 + d] = 10.0
        blk1.attn.W_o.weight.data[100 + d, 3*dh + d] = 1.0
    
    # Heads 4-5: same for pos 12 (queries use pos 12 flag)
    # Head 4: y1_tens to dims 70-79 at pos 12
    blk1.attn.W_q.weight.data[4*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[4*dh, 112] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[4*dh + d, 10 + d] = 10.0
        blk1.attn.W_o.weight.data[70 + d, 4*dh + d] = 1.0
    
    # Head 5: y2_tens to dims 80-89 at pos 12
    blk1.attn.W_q.weight.data[5*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[5*dh, 112] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[5*dh + d, 40 + d] = 10.0
        blk1.attn.W_o.weight.data[80 + d, 5*dh + d] = 1.0
    
    # Head 6: y1_ones to dims 90-99 at pos 12
    blk1.attn.W_q.weight.data[6*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[6*dh, 112] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[6*dh + d, 20 + d] = 10.0
        blk1.attn.W_o.weight.data[90 + d, 6*dh + d] = 1.0
    
    # Head 7: y2_ones to dims 100-109 at pos 12
    blk1.attn.W_q.weight.data[7*dh, 112] = 10.0
    blk1.attn.W_k.weight.data[7*dh, 112] = 10.0
    for d in range(10):
        blk1.attn.W_v.weight.data[7*dh + d, 50 + d] = 10.0
        blk1.attn.W_o.weight.data[100 + d, 7*dh + d] = 1.0
    
    # ==================== BLOCK 2: Compute selector ====================
    blk2 = model.blocks[2]
    mlp2 = blk2.mlp
    # Read |x3-x1| = relu(x3-x1) + relu(x1-x3) from dims 200,201
    # Read |x3-x2| = relu(x3-x2) + relu(x2-x3) from dims 202,203
    # selector = |x3-x2| - |x3-x1|
    
    # Neuron 200: |x3-x2| - |x3-x1| (positive when closer to x1)
    mlp2.fc1.weight.data[200, 202] = 1.0   # relu(x3-x2)
    mlp2.fc1.weight.data[200, 203] = 1.0   # relu(x2-x3)
    mlp2.fc1.weight.data[200, 200] = -1.0  # -relu(x3-x1)
    mlp2.fc1.weight.data[200, 201] = -1.0  # -relu(x1-x3)
    # No bias -> can be negative -> after ReLU: max(0, selector)
    
    # Neuron 201: |x3-x1| - |x3-x2| (positive when closer to x2)
    mlp2.fc1.weight.data[201, 200] = 1.0
    mlp2.fc1.weight.data[201, 201] = 1.0
    mlp2.fc1.weight.data[201, 202] = -1.0
    mlp2.fc1.weight.data[201, 203] = -1.0
    
    mlp2.fc2.weight.data[200, 200] = 10.0  # use_y1 signal (positive when closer to x1)
    mlp2.fc2.weight.data[201, 201] = 10.0  # use_y2 signal
    
    # Block 2 attention: route based on selector
    # Head 0: pos 11 queries use_y1 (dim 200), copies y1_tens (dims 70-79) to dims 0-9
    blk2.attn.W_q.weight.data[0, 111] = 10.0
    blk2.attn.W_k.weight.data[0, 200] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[d, 70 + d] = 10.0
        blk2.attn.W_o.weight.data[d, d] = 1.0
    
    # Head 1: pos 11 queries use_y2 (dim 201), copies y2_tens (dims 80-89) to dims 0-9
    blk2.attn.W_q.weight.data[dh, 111] = 10.0
    blk2.attn.W_k.weight.data[dh, 201] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[dh + d, 80 + d] = 10.0
        blk2.attn.W_o.weight.data[d, dh + d] = 1.0
    
    # Head 2: pos 12 queries use_y1, copies y1_ones (dims 90-99) to dims 10-19
    blk2.attn.W_q.weight.data[2*dh, 112] = 10.0
    blk2.attn.W_k.weight.data[2*dh, 200] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[2*dh + d, 90 + d] = 10.0
        blk2.attn.W_o.weight.data[10 + d, 2*dh + d] = 1.0
    
    # Head 3: pos 12 queries use_y2, copies y2_ones (dims 100-109) to dims 10-19
    blk2.attn.W_q.weight.data[3*dh, 112] = 10.0
    blk2.attn.W_k.weight.data[3*dh, 201] = 10.0
    for d in range(10):
        blk2.attn.W_v.weight.data[3*dh + d, 100 + d] = 10.0
        blk2.attn.W_o.weight.data[10 + d, 3*dh + d] = 1.0
    
    # ==================== BLOCK 3: passthrough ====================
    # Zero (already zeroed)
    
    # ==================== OUTPUT HEAD ====================
    # Tens digit from dims 0-9, ones digit from dims 10-19
    for d in range(10):
        model.head.weight.data[d, d] = 10.0         # tens
        model.head.weight.data[d, 10 + d] = 10.0    # ones (for pos 12)


model_shorthand_name = "v25_nearest_neighbor"
model_description = "Output y1 if |x3-x1| < |x3-x2| else y2"


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
    
    # Debug: print vocab info
    if args.verbose:
        print(f"vocab_size={model.vocab_size}, max_seq_len={model.max_seq_len}")
        print(f"d_model={model.d_model}, n_heads={model.n_heads}, n_layers={model.n_layers}, d_ff={model.d_ff}")
        print(f"n_params={n_params}")

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
