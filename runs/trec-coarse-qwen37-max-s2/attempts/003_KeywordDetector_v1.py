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
    """Keyword-based classifier using first-word detection.
    
    Strategy: Detect question keywords (who, where, when, how, why, what) in the
    first few positions and map them to classes.
    
    Class mapping:
    - 0: Abbreviation (what does ... abbreviation)
    - 1: Entity (what is the name, what killed, what do you call)
    - 2: Description (what is, how to, why, definition)
    - 3: Human (who, whom)
    - 4: Location (where, what country, what city)
    - 5: Numeric (how many, when, what year)
    """
    torch.manual_seed(42)
    
    d_model = model.d_model
    vocab_size = model.vocab_size
    
    # Token indices
    stoi = task.stoi
    
    # Initialize with small random values
    for param in model.parameters():
        param.data.normal_(0, 0.01)
    
    # Embedding: give distinct patterns to key characters
    # w, h, o, e, r, a, t, space are important
    key_chars = ['w', 'h', 'o', 'e', 'r', 'a', 't', ' ', 'i', 'n', 'y']
    for i, char in enumerate(key_chars):
        if char in stoi:
            idx = stoi[char]
            model.token_emb.weight.data[idx, i] = 1.0
    
    # Position embedding: emphasize first 10 positions
    for pos in range(10):
        model.pos_emb.weight.data[pos, 0] = 1.0 - pos * 0.1
    
    # Attention layer 1: detect "who", "where", "when", "how", "why"
    # Use multiple heads to detect different patterns
    
    block0 = model.blocks[0]
    
    # Head 0: detect "who" -> class 3
    # Q: look for 'w' at position 0
    # K: match 'h' at position 1, 'o' at position 2
    block0.attn.W_q.weight.data[0, stoi['w']] = 2.0
    block0.attn.W_k.weight.data[0, stoi['h']] = 2.0
    block0.attn.W_k.weight.data[0, stoi['o']] = 2.0
    block0.attn.W_v.weight.data[0, 0] = 2.0
    
    # Head 1: detect "where" -> class 4
    block0.attn.W_q.weight.data[1, stoi['w']] = 2.0
    block0.attn.W_k.weight.data[1, stoi['h']] = 2.0
    block0.attn.W_k.weight.data[1, stoi['e']] = 2.0
    block0.attn.W_k.weight.data[1, stoi['r']] = 2.0
    block0.attn.W_v.weight.data[1, 1] = 2.0
    
    # Head 2: detect "when" -> class 5
    block0.attn.W_q.weight.data[2, stoi['w']] = 2.0
    block0.attn.W_k.weight.data[2, stoi['h']] = 2.0
    block0.attn.W_k.weight.data[2, stoi['e']] = 2.0
    block0.attn.W_k.weight.data[2, stoi['n']] = 2.0
    block0.attn.W_v.weight.data[2, 2] = 2.0
    
    # Head 3: detect "how" -> class 5
    block0.attn.W_q.weight.data[3, stoi['h']] = 2.0
    block0.attn.W_k.weight.data[3, stoi['o']] = 2.0
    block0.attn.W_k.weight.data[3, stoi['w']] = 2.0
    block0.attn.W_v.weight.data[3, 2] = 2.0
    
    # MLP layer 1: map detected patterns to class predictions
    # Input: aggregated keyword signals (dims 0-3)
    # Output: class logits (mapped to vocab indices 30-35 for '0'-'5')
    
    # who detected (dim 0) -> class 3
    block0.mlp.fc1.weight.data[0, 0] = 3.0
    block0.mlp.fc1.bias.data[0] = -1.0
    block0.mlp.fc2.weight.data[33, 0] = 3.0  # '3' is vocab index 33
    
    # where detected (dim 1) -> class 4
    block0.mlp.fc1.weight.data[1, 1] = 3.0
    block0.mlp.fc1.bias.data[1] = -1.0
    block0.mlp.fc2.weight.data[34, 1] = 3.0  # '4' is vocab index 34
    
    # when/how detected (dim 2) -> class 5
    block0.mlp.fc1.weight.data[2, 2] = 3.0
    block0.mlp.fc1.bias.data[2] = -1.0
    block0.mlp.fc2.weight.data[35, 2] = 3.0  # '5' is vocab index 35
    
    # Layer 2: detect "why" and "what" patterns
    block1 = model.blocks[1]
    
    # Head 0: detect "why" -> class 2
    block1.attn.W_q.weight.data[0, stoi['w']] = 2.0
    block1.attn.W_k.weight.data[0, stoi['h']] = 2.0
    block1.attn.W_k.weight.data[0, stoi['y']] = 2.0
    block1.attn.W_v.weight.data[0, 3] = 2.0
    
    # Head 1: detect "what" -> could be class 0, 1, or 2
    block1.attn.W_q.weight.data[1, stoi['w']] = 2.0
    block1.attn.W_k.weight.data[1, stoi['h']] = 2.0
    block1.attn.W_k.weight.data[1, stoi['a']] = 2.0
    block1.attn.W_k.weight.data[1, stoi['t']] = 2.0
    block1.attn.W_v.weight.data[1, 4] = 2.0
    
    # MLP layer 2: refine predictions
    # why detected (dim 3) -> class 2
    block1.mlp.fc1.weight.data[3, 3] = 3.0
    block1.mlp.fc1.bias.data[3] = -1.0
    block1.mlp.fc2.weight.data[32, 3] = 3.0  # '2' is vocab index 32
    
    # what detected (dim 4) -> class 1 (default for "what")
    block1.mlp.fc1.weight.data[4, 4] = 3.0
    block1.mlp.fc1.bias.data[4] = -1.0
    block1.mlp.fc2.weight.data[31, 4] = 3.0  # '1' is vocab index 31
    
    # Final layer: output projection
    # Map accumulated class signals to output logits
    model.head.weight.data[30, 5] = 2.0  # class 0 (rare)
    model.head.weight.data[31, 6] = 2.0  # class 1
    model.head.weight.data[32, 7] = 2.0  # class 2
    model.head.weight.data[33, 8] = 2.0  # class 3
    model.head.weight.data[34, 9] = 2.0  # class 4
    model.head.weight.data[35, 10] = 2.0  # class 5


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "KeywordDetector-v1"
model_description = "First-word keyword detector: maps 'who'->3, 'where'->4, 'when/how'->5, 'why'->2, 'what'->1"


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
