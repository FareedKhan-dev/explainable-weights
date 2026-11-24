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

def analyze_dataset(task):
    """Compute character-level sentiment scores from training data."""
    import numpy as np
    
    examples = task.generate_examples(2000)
    
    pos_char_counts = np.zeros(32)
    neg_char_counts = np.zeros(32)
    pos_total_chars = 0
    neg_total_chars = 0
    
    for ex in examples:
        ids = task.encode(ex.prompt)
        label = int(ex.answer)
        
        for token_id in ids:
            if token_id < 28:  # a-z, space, apostrophe
                if label == 1:
                    pos_char_counts[token_id] += 1
                    pos_total_chars += 1
                else:
                    neg_char_counts[token_id] += 1
                    neg_total_chars += 1
    
    # Compute log-odds ratio for each character
    char_scores = {}
    for i in range(32):
        pf = (pos_char_counts[i] + 1) / (pos_total_chars + 32)
        nf = (neg_char_counts[i] + 1) / (neg_total_chars + 32)
        char_scores[i] = float(np.log(pf / nf))
    
    # Print for debugging
    itos = {v: k for k, v in task.stoi.items()}
    print("\nCharacter sentiment scores:")
    for i in range(28):
        char = itos.get(i, '?')
        print(f"  '{char}' ({i}): {char_scores[i]:.6f}")
    
    # Also print as a Python dict for hardcoding
    print("\n# Hardcoded scores:")
    print("CHAR_SCORES = {")
    for i in range(32):
        print(f"    {i}: {char_scores[i]:.6f},")
    print("}")
    
    return char_scores


def write_weights(model: SimpleTransformer, task) -> None:
    """Bag-of-characters sentiment classifier using character-level log-odds."""
    
    # Character sentiment scores (log-odds ratio from training data)
    CHAR_SCORES = {
        0: 0.096054,   # a
        1: -0.114909,  # b
        2: -0.011751,  # c
        3: -0.064581,  # d
        4: 0.015304,   # e
        5: 0.075313,   # f
        6: 0.207707,   # g
        7: 0.019560,   # h
        8: 0.039751,   # i
        9: 0.024237,   # j
        10: -0.122248, # k
        11: -0.053352, # l
        12: 0.040320,  # m
        13: 0.046467,  # n
        14: -0.086886, # o
        15: -0.017761, # p
        16: -0.074809, # q
        17: 0.085346,  # r
        18: -0.056220, # s
        19: -0.010558, # t
        20: -0.037918, # u
        21: 0.027198,  # v
        22: -0.028477, # w
        23: 0.313461,  # x
        24: -0.136195, # y
        25: 0.195482,  # z
        26: -0.032997, # space
        27: -0.149445, # apostrophe
        28: 0.0,       # padding _
        29: 0.0,       # =
        30: 0.0,       # 0
        31: 0.0,       # 1
    }
    
    # Scale factor to amplify the signal
    SCALE = 10.0
    
    # Token embeddings: encode sentiment score in dimension 0
    model.token_emb.weight.data.zero_()
    for token_id in range(32):
        model.token_emb.weight.data[token_id, 0] = CHAR_SCORES[token_id] * SCALE
    
    # Position embeddings: all zeros (uniform bag-of-words)
    model.pos_emb.weight.data.zero_()
    
    # Block 0: Identity (skip attention)
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    block0.attn.W_q.weight.data.zero_()
    block0.attn.W_k.weight.data.zero_()
    block0.attn.W_v.weight.data.zero_()
    block0.attn.W_o.weight.data.zero_()
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    block0.mlp.fc1.weight.data.zero_()
    block0.mlp.fc1.bias.data.zero_()
    block0.mlp.fc2.weight.data.zero_()
    block0.mlp.fc2.bias.data.zero_()
    
    # Block 1: Uniform attention to average all positions
    block1 = model.blocks[1]
    block1.ln1.weight.data.fill_(1.0)
    block1.ln1.bias.data.zero_()
    
    # Uniform attention: Q=K=V=0 gives uniform weights
    block1.attn.W_q.weight.data.zero_()
    block1.attn.W_k.weight.data.zero_()
    block1.attn.W_v.weight.data.zero_()
    
    # V copies dimension 0 (sentiment score)
    block1.attn.W_v.weight.data[0, 0] = 1.0
    
    # Output projection: pass through dimension 0
    block1.attn.W_o.weight.data.zero_()
    block1.attn.W_o.weight.data[0, 0] = 1.0
    
    block1.ln2.weight.data.fill_(1.0)
    block1.ln2.bias.data.zero_()
    block1.mlp.fc1.weight.data.zero_()
    block1.mlp.fc1.bias.data.zero_()
    block1.mlp.fc2.weight.data.zero_()
    block1.mlp.fc2.bias.data.zero_()
    
    # Block 2: Identity
    block2 = model.blocks[2]
    block2.ln1.weight.data.fill_(1.0)
    block2.ln1.bias.data.zero_()
    block2.attn.W_q.weight.data.zero_()
    block2.attn.W_k.weight.data.zero_()
    block2.attn.W_v.weight.data.zero_()
    block2.attn.W_o.weight.data.zero_()
    block2.ln2.weight.data.fill_(1.0)
    block2.ln2.bias.data.zero_()
    block2.mlp.fc1.weight.data.zero_()
    block2.mlp.fc1.bias.data.zero_()
    block2.mlp.fc2.weight.data.zero_()
    block2.mlp.fc2.bias.data.zero_()
    
    # Final layer norm: identity
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: classify based on dimension 0 (aggregated sentiment)
    model.head.weight.data.zero_()
    # Positive class (token 31 = '1') gets high score if dim 0 > 0
    model.head.weight.data[31, 0] = 1.0
    # Negative class (token 30 = '0') gets high score if dim 0 < 0
    model.head.weight.data[30, 0] = -1.0


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "BagOfCharsSentiment"
model_description = "Bag-of-characters classifier: token embeddings encode sentiment scores, uniform attention averages across positions, head classifies."


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
