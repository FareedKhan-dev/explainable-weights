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
    
    Attempt 11: Character frequency distribution comparison.
    Strategy: Build character frequency histograms for each sentence,
    compare them using MLP, and threshold for paraphrase detection.
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
        # STEP 1: Token embeddings encode character identity
        # Use one-hot encoding in first vocab_size dimensions
        # ============================================================
        for tid in range(vocab_size):
            ch = task.itos.get(tid, '')
            # One-hot for character identity
            if tid < d:
                model.token_emb.weight[tid, tid] = 5.0
            
            # Dim 60: is_space
            if ch == ' ':
                model.token_emb.weight[tid, 60] = 3.0
            
            # Dim 61: is_separator
            if ch in '|=':
                model.token_emb.weight[tid, 61] = 10.0
        
        # ============================================================
        # STEP 2: Position embeddings encode sentence membership
        # ============================================================
        max_pos = model.max_seq_len
        for pos in range(max_pos):
            if pos < 60:
                model.pos_emb.weight[pos, 62] = 5.0   # sentence 1
            elif pos == 60:
                model.pos_emb.weight[pos, 63] = 5.0   # separator
            elif pos < 121:
                model.pos_emb.weight[pos, 62] = -5.0  # sentence 2 (negative)
            else:
                model.pos_emb.weight[pos, 63] = -5.0  # query
        
        # ============================================================
        # STEP 3: LayerNorm - set to identity
        # ============================================================
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.fill_(0.0)
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.fill_(0.0)
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.fill_(0.0)
        
        # ============================================================
        # STEP 4: Block 0 - aggregate character frequencies per sentence
        # At separator (pos 60), attend uniformly to all of sent1 (pos 0-59)
        # At query (pos 121), attend uniformly to all of sent2 (pos 61-120)
        # ============================================================
        
        attn0 = model.blocks[0].attn
        
        # Head 0: uniform attention over current sentence
        # Query: "I'm at separator or query"
        # Key: "I'm in the same sentence"
        # Value: character identity (one-hot)
        
        for h_idx in range(n_heads):
            start = h_idx * dh
            end = start + dh
            
            # Q: detect separator/query position
            attn0.W_q.weight[start, 63] = 3.0  # am I at sep or query?
            
            # K: detect sentence membership (same sign as query)
            attn0.W_k.weight[start, 62] = 3.0  # sentence flag
            
            # V: read character identity (one-hot)
            for i in range(min(dh, vocab_size)):
                if start + i < d and i < d:
                    attn0.W_v.weight[start + i, i] = 1.0
            
            # O: pass through
            for i in range(dh):
                if i < d and start + i < d:
                    attn0.W_o.weight[i, start + i] = 1.0
        
        # ============================================================
        # STEP 5: Block 0 MLP - normalize character frequencies
        # Just pass through (identity)
        # ============================================================
        mlp0 = model.blocks[0].mlp
        # fc1: identity
        for i in range(min(d, mlp0.fc1.out_features)):
            mlp0.fc1.weight[i, i] = 1.0
        # fc2: identity
        for i in range(min(d, mlp0.fc2.out_features)):
            mlp0.fc2.weight[i, i] = 1.0
        
        # ============================================================
        # STEP 6: Block 1 - cross-sentence comparison
        # At query, attend to separator to get sent1 summary
        # ============================================================
        
        attn1 = model.blocks[1].attn
        
        # Head 0: query attends to separator
        attn1.W_q.weight[0, 63] = -5.0  # I'm the query (negative sep flag)
        attn1.W_k.weight[0, 63] = 5.0   # you're the separator (positive sep flag)
        
        # Value: read character frequencies
        for i in range(min(dh, d)):
            attn1.W_v.weight[i, i] = 1.0
        
        # Output: write to dims 16-31 (sent1 frequencies)
        for i in range(dh):
            if 16 + i < d:
                attn1.W_o.weight[16 + i, i] = 1.0
        
        # Other heads: pass through
        for h_idx in range(1, n_heads):
            start = h_idx * dh
            for i in range(dh):
                attn1.W_v.weight[start + i, i] = 1.0
                attn1.W_o.weight[i, start + i] = 1.0
        
        # ============================================================
        # STEP 7: Block 1 MLP - compute similarity score
        # Input: dims 0-15 = sent2 char freqs, dims 16-31 = sent1 char freqs
        # Output: similarity score in dim 0
        # ============================================================
        
        mlp1 = model.blocks[1].mlp
        d_ff = mlp1.fc1.out_features
        
        # Compute L1 distance between sent1 and sent2 frequency distributions
        # fc1: compute differences
        for i in range(min(16, d_ff)):
            # sent2[i] - sent1[i]
            mlp1.fc1.weight[i, i] = 1.0      # sent2 char i
            mlp1.fc1.weight[i, 16 + i] = -1.0  # sent1 char i
        
        # ReLU keeps positive differences
        
        # fc2: sum absolute differences -> negative score (more different = lower)
        for i in range(min(16, d_ff)):
            mlp1.fc2.weight[0, i] = -1.0  # negative weight: more diff = lower score
        
        # ============================================================
        # STEP 8: Block 2 - threshold and decision
        # ============================================================
        
        mlp2 = model.blocks[2].mlp
        # Pass through
        for i in range(min(d, mlp2.fc1.out_features)):
            mlp2.fc1.weight[i, i] = 1.0
        for i in range(min(d, mlp2.fc2.out_features)):
            mlp2.fc2.weight[i, i] = 1.0
        
        # ============================================================
        # STEP 9: Output head
        # Predict based on similarity score in dim 0
        # ============================================================
        
        # Find '0' and '1' tokens
        tok_0 = tok_1 = None
        for tid, ch in task.itos.items():
            if ch == '0':
                tok_0 = tid
            elif ch == '1':
                tok_1 = tid
        
        if tok_0 is not None and tok_1 is not None:
            # High similarity (positive score) -> predict '1' (paraphrase)
            # Low similarity (negative score) -> predict '0' (not paraphrase)
            model.head.weight[tok_1, 0] = 10.0
            model.head.weight[tok_0, 0] = -10.0
            
            # Add bias via position embedding at query
            model.pos_emb.weight[121, 0] = 5.0  # bias towards '1' (majority class)
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "CharFreqComparator"
model_description = "Compare character frequency distributions between sentences using attention and MLP."


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
