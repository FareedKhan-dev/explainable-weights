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
    
    Attempt 12: Simple character bigram overlap detector.
    Strategy: Count matching character bigrams between sentences using attention.
    Threshold the count to predict paraphrase.
    """
    import numpy as np
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 64
        n_heads = model.n_heads  # 4
        dh = d // n_heads  # 16
        vocab_size = model.vocab_size
        
        # Zero everything first
        for p in model.parameters():
            p.zero_()
        
        # ============================================================
        # STEP 1: Token embeddings - simple character identity
        # ============================================================
        for tid in range(vocab_size):
            ch = task.itos.get(tid, '')
            # Binary encoding of character ID
            for bit in range(8):
                if (tid >> bit) & 1:
                    model.token_emb.weight[tid, bit] = 2.0
                else:
                    model.token_emb.weight[tid, bit] = -2.0
            
            # Dim 8: is_space (word boundary)
            if ch == ' ':
                model.token_emb.weight[tid, 8] = 5.0
            else:
                model.token_emb.weight[tid, 8] = -5.0
        
        # ============================================================
        # STEP 2: Position embeddings - sentence membership
        # ============================================================
        max_pos = model.max_seq_len
        for pos in range(max_pos):
            # Dim 10: sentence membership
            if pos < 60:
                model.pos_emb.weight[pos, 10] = 10.0   # sentence 1
            elif pos == 60:
                model.pos_emb.weight[pos, 10] = 0.0    # separator
            elif pos < 121:
                model.pos_emb.weight[pos, 10] = -10.0  # sentence 2
            else:
                model.pos_emb.weight[pos, 10] = 0.0    # query
            
            # Dim 11: position within sentence (for bigram context)
            if pos < 60:
                model.pos_emb.weight[pos, 11] = pos / 60.0
            elif pos < 121:
                model.pos_emb.weight[pos, 11] = (pos - 61) / 60.0
        
        # ============================================================
        # STEP 3: LayerNorm - identity
        # ============================================================
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.fill_(0.0)
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.fill_(0.0)
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.fill_(0.0)
        
        # ============================================================
        # STEP 4: Block 0 - build bigram representations
        # Each position attends to itself and previous position
        # ============================================================
        
        attn0 = model.blocks[0].attn
        
        # Head 0: attend to current position
        # Q, K, V all read character + position
        for i in range(dh):
            attn0.W_q.weight[i, i] = 1.0
            attn0.W_k.weight[i, i] = 1.0
            attn0.W_v.weight[i, i] = 1.0
            attn0.W_o.weight[i, i] = 1.0
        
        # Head 1: attend to previous position (for bigram)
        # Use position embedding to create offset
        for i in range(dh):
            attn0.W_q.weight[dh + i, i] = 1.0
            attn0.W_q.weight[dh + i, 11] = 2.0  # read position
            attn0.W_k.weight[dh + i, i] = 1.0
            attn0.W_k.weight[dh + i, 11] = -2.0  # negative position offset
            attn0.W_v.weight[dh + i, i] = 1.0
            attn0.W_o.weight[i, dh + i] = 1.0
        
        # ============================================================
        # STEP 5: Block 1 - count bigram matches across sentences
        # At query position, attend to both sentences
        # ============================================================
        
        attn1 = model.blocks[1].attn
        
        # Head 0: query attends to sentence 1
        attn1.W_q.weight[0, 10] = 0.0  # neutral query
        attn1.W_k.weight[0, 10] = 5.0  # positive = sentence 1
        attn1.W_v.weight[0, :8] = 1.0  # read character features
        attn1.W_o.weight[:8, 0] = 1.0
        
        # Head 1: query attends to sentence 2
        attn1.W_q.weight[dh, 10] = 0.0
        attn1.W_k.weight[dh, 10] = -5.0  # negative = sentence 2
        attn1.W_v.weight[dh, :8] = 1.0
        attn1.W_o.weight[8:16, dh] = 1.0
        
        # ============================================================
        # STEP 6: Block 1 MLP - compute similarity
        # Compare sent1 bigrams (dims 0-7) with sent2 bigrams (dims 8-15)
        # ============================================================
        
        mlp1 = model.blocks[1].mlp
        
        # fc1: compute differences
        for i in range(8):
            mlp1.fc1.weight[i, i] = 1.0      # sent1 bigram
            mlp1.fc1.weight[i, 8 + i] = -1.0  # sent2 bigram
        
        # fc2: sum differences (more similar = smaller sum)
        for i in range(8):
            mlp1.fc2.weight[0, i] = -1.0  # negative: smaller diff = higher score
        
        # ============================================================
        # STEP 7: Block 2 - pass through
        # ============================================================
        attn2 = model.blocks[2].attn
        for i in range(d):
            attn2.W_v.weight[i, i] = 1.0
            attn2.W_o.weight[i, i] = 1.0
        
        mlp2 = model.blocks[2].mlp
        for i in range(min(d, mlp2.fc1.out_features)):
            mlp2.fc1.weight[i, i] = 1.0
        for i in range(min(d, mlp2.fc2.out_features)):
            mlp2.fc2.weight[i, i] = 1.0
        
        # ============================================================
        # STEP 8: Output head
        # ============================================================
        
        tok_0 = tok_1 = None
        for tid, ch in task.itos.items():
            if ch == '0':
                tok_0 = tid
            elif ch == '1':
                tok_1 = tid
        
        if tok_0 is not None and tok_1 is not None:
            model.head.weight[tok_1, 0] = 5.0
            model.head.weight[tok_0, 0] = -5.0
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "BigramOverlap_v2"
model_description = "Detect character bigram overlap between sentences using attention and MLP."


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
