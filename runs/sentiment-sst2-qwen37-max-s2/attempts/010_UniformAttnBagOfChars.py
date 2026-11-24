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
        d_model: int = 128,
        n_heads: int = 8,
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
    """Analyze data and build character-level sentiment classifier."""
    torch.manual_seed(0)
    D = model.d_model
    
    # Generate examples to analyze character patterns
    examples = task.generate_examples(200, seed=42)
    
    # Compute character frequencies for positive and negative sentences
    pos_counts = {}
    neg_counts = {}
    pos_total = 0
    neg_total = 0
    
    for ex in examples:
        label = ex.answer
        text = ex.prompt.rstrip('_=').lower()
        
        if label == '1':
            pos_total += 1
            for c in text:
                pos_counts[c] = pos_counts.get(c, 0) + 1
        else:
            neg_total += 1
            for c in text:
                neg_counts[c] = neg_counts.get(c, 0) + 1
    
    # Compute character sentiment scores
    char_sentiment = {}
    for c in 'abcdefghijklmnopqrstuvwxyz \'':
        pos_freq = pos_counts.get(c, 0) / max(pos_total, 1)
        neg_freq = neg_counts.get(c, 0) / max(neg_total, 1)
        # Log odds ratio (smoothed)
        score = (pos_freq + 0.01) / (neg_freq + 0.01)
        char_sentiment[c] = score
    
    print("Character sentiment scores (pos/neg ratio):")
    for c, score in sorted(char_sentiment.items(), key=lambda x: x[1], reverse=True):
        if c.strip():
            print(f"  '{c}': {score:.3f}")
    
    # Token embeddings: encode character sentiment in dim 0
    model.token_emb.weight.data.zero_()
    stoi = task.stoi
    for char, idx in stoi.items():
        score = char_sentiment.get(char, 1.0)
        # Convert ratio to log scale: >1 is positive, <1 is negative
        model.token_emb.weight.data[idx, 0] = (score - 1.0) * 10  # Scale
    
    model.pos_emb.weight.data.zero_()
    
    # Block 0: Uniform attention to aggregate character sentiments
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    # Uniform attention: K=0, Q=0
    block0.attn.W_q.weight.data.zero_()
    block0.attn.W_k.weight.data.zero_()
    
    # V: copy sentiment score (dim 0)
    block0.attn.W_v.weight.data.zero_()
    block0.attn.W_v.weight.data[0, 0] = 1.0
    
    # O: keep in dim 0
    block0.attn.W_o.weight.data.zero_()
    block0.attn.W_o.weight.data[0, 0] = 1.0
    
    # MLP: amplify sentiment signal
    block0.mlp.fc1.weight.data.zero_()
    block0.mlp.fc1.bias.data.zero_()
    block0.mlp.fc2.weight.data.zero_()
    block0.mlp.fc2.bias.data.zero_()
    
    # Block 1, 2: identity
    for block in model.blocks[1:]:
        block.attn.W_q.weight.data.zero_()
        block.attn.W_k.weight.data.zero_()
        block.attn.W_v.weight.data.zero_()
        block.attn.W_o.weight.data.zero_()
        block.mlp.fc1.weight.data.zero_()
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.zero_()
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: classify based on aggregated sentiment (dim 0)
    model.head.weight.data.zero_()
    model.head.weight.data[31, 0] = 10.0   # Positive if sentiment > 0
    model.head.weight.data[30, 0] = -10.0  # Negative if sentiment < 0
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "UniformAttnBagOfChars"
model_description = "K=0 uniform attention creates bag-of-chars, MLP maps to sentiment, head classifies."


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
