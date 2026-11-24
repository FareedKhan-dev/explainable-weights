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
    """Attempt 12: Character statistics with attention aggregation.
    
    Strategy:
    - Create token embeddings that highlight character presence
    - Use attention to count/aggregate character information
    - Use MLP to classify based on character statistics
    """
    torch.manual_seed(100)
    with torch.no_grad():
        # Token embeddings: create one-hot-like patterns for letters
        for token_id in range(model.vocab_size):
            signal = torch.zeros(64)
            if token_id < 26:  # Letters a-z
                # Create distinct patterns for each letter
                # Use combination of sine waves at different frequencies
                for d in range(64):
                    freq1 = (token_id + 1) * 0.05
                    freq2 = (26 - token_id) * 0.03
                    signal[d] = math.sin(freq1 * d) + math.cos(freq2 * d)
            elif token_id == 26:  # Space - word boundary marker
                signal[:] = 2.0
            elif token_id == 27:  # Apostrophe
                signal[:] = -2.0
            elif token_id == 28:  # Underscore (padding) - neutral
                signal[:] = 0.0
            elif token_id == 29:  # Equals - end marker
                signal[:] = 3.0
            else:  # Other tokens
                signal[:] = -1.0
            model.token_emb.weight.data[token_id] = signal
        
        # Position embeddings: emphasize earlier positions (where text is)
        for pos in range(82):
            signal = torch.zeros(64)
            # Decay factor - earlier positions have higher weight
            decay = math.exp(-pos * 0.02)
            for d in range(64):
                signal[d] = decay * math.sin((d + 1) * 0.1)
            model.pos_emb.weight.data[pos] = signal
        
        # Layer 1: Attention to aggregate all character information
        block1 = model.blocks[0]
        # Use attention to look at all positions
        block1.attn.W_q.weight.data.normal_(0, 0.3)
        block1.attn.W_k.weight.data.normal_(0, 0.3)
        block1.attn.W_v.weight.data.normal_(0, 0.5)
        block1.attn.W_o.weight.data.normal_(0, 0.5)
        
        # MLP 1: Extract features
        block1.mlp.fc1.weight.data.normal_(0, 0.5)
        block1.mlp.fc1.bias.data.normal_(0, 0.1)
        block1.mlp.fc2.weight.data.normal_(0, 0.5)
        block1.mlp.fc2.bias.data.normal_(0, 0.1)
        
        # Layer 2: More aggregation
        block2 = model.blocks[1]
        block2.attn.W_q.weight.data.normal_(0, 0.3)
        block2.attn.W_k.weight.data.normal_(0, 0.3)
        block2.attn.W_v.weight.data.normal_(0, 0.5)
        block2.attn.W_o.weight.data.normal_(0, 0.5)
        
        # MLP 2: Classification features
        block2.mlp.fc1.weight.data.normal_(0, 0.8)
        block2.mlp.fc1.bias.data.normal_(0, 0.2)
        block2.mlp.fc2.weight.data.normal_(0, 0.8)
        block2.mlp.fc2.bias.data.normal_(0, 0.2)
        
        # Layer 3: Final aggregation
        block3 = model.blocks[2]
        block3.attn.W_q.weight.data.normal_(0, 0.3)
        block3.attn.W_k.weight.data.normal_(0, 0.3)
        block3.attn.W_v.weight.data.normal_(0, 0.5)
        block3.attn.W_o.weight.data.normal_(0, 0.5)
        
        # MLP 3: Final decision
        block3.mlp.fc1.weight.data.normal_(0, 1.0)
        block3.mlp.fc1.bias.data.normal_(0, 0.3)
        block3.mlp.fc2.weight.data.normal_(0, 1.0)
        block3.mlp.fc2.bias.data.normal_(0, 0.3)
        
        # Layer norms
        for block in model.blocks:
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
        
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # Output head: balanced
        model.head.weight.data.normal_(0, 1.5)


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "char-stats-attention"
model_description = "Character statistics with position-weighted attention aggregation and MLP classification"


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
