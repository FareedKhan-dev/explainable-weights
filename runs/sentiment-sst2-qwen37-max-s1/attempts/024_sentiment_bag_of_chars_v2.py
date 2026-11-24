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
    """Sentiment bigram/trigram detector with aggregation.
    
    Strategy:
    - Token embeddings encode character identity (one-hot in dims 0-31) 
      plus sentiment score in dim 32
    - Layer 1 attention: uniform (zero Q,K) to average character info
    - Layer 1 MLP: per-character sentiment scoring using hand-crafted weights
    - Layer 2 attention: uniform to aggregate sentiment across all positions
    - Layer 2 MLP: threshold decision
    - Output head: map to '0' or '1'
    """
    D = 64
    DH = D // 4  # 16
    stoi = task.stoi
    
    with torch.no_grad():
        # ---- TOKEN EMBEDDINGS ----
        model.token_emb.weight.data.zero_()
        
        # One-hot character identity in dims 0-31
        for i in range(32):
            model.token_emb.weight.data[i, i] = 1.0
        
        # Padding (_) should contribute nothing
        model.token_emb.weight.data[28, :] = 0.0
        
        # Sentiment scores for each character based on movie review word frequencies
        # These are NET scores (positive - negative), scaled for common movie review words
        char_sentiment = {
            'g': 0.25,    # good, great, gripping, gorgeous, genuine
            'w': 0.15,    # wonderful, witty, warm, winning (but also worst, weak, waste)
            'f': 0.20,    # fun, fine, fantastic, fresh, funny (but also fail, feeble)
            'n': 0.15,    # nice, new, notable (but also never, nothing)
            'e': 0.10,    # excellent, enjoy, engaging, entertaining, emotional
            'l': 0.10,    # love, lovely, luminous (but also lame, lackluster)
            'b': 0.05,    # beautiful, brilliant, best (but also bad, boring, bland)
            'a': 0.05,    # amazing, absorbing (but also awful, annoying)
            'm': 0.05,    # masterful, moving, magnificent (but also mediocre, muddled)
            'h': 0.05,    # hilarious, heartfelt (but also horrible, hollow)
            'p': 0.00,    # perfect, powerful (but also poor, pathetic, predictable)
            'c': 0.00,    # compelling, charming, clever (but also cliched, confused)
            's': -0.05,   # stunning, superb (but also stupid, shallow, slow, silly)
            'r': 0.05,    # remarkable, riveting (but also ridiculous, repetitive)
            'i': 0.05,    # impressive, interesting, intelligent, innovative
            't': -0.10,   # touching, terrific (but also terrible, tedious, tiresome, trite)
            'd': -0.05,   # delightful, dazzling (but also dull, disappointing, dreadful)
            'o': 0.05,    # outstanding, original (but also ordinary, overdone)
            'u': -0.15,   # unique (but also ugly, unpleasant, unconvincing, unimaginative)
            'k': -0.05,   # keen (but also klutzy)
            'v': 0.05,    # vivid, vibrant (but also vapid)
            'y': 0.05,    # yes (neutral mostly)
            'x': 0.00,    # rare
            'j': 0.05,    # joyful, joy (but also jaded)
            'q': 0.00,    # quality, quirky (rare)
            'z': 0.00,    # rare
        }
        
        for ch, score in char_sentiment.items():
            if ch in stoi:
                model.token_emb.weight.data[stoi[ch], 32] = score * 3.0
        
        # ---- POSITION EMBEDDINGS ----
        model.pos_emb.weight.data.zero_()
        # Give position 80 ('=' sign) a special marker in dim 33
        model.pos_emb.weight.data[80, 33] = 5.0
        
        # ---- LAYER 1: CHARACTER SENTIMENT SCORING ----
        block1 = model.blocks[0]
        
        # Zero Q,K for uniform attention
        block1.attn.W_q.weight.data.zero_()
        block1.attn.W_k.weight.data.zero_()
        
        # Value: pass through all dims
        block1.attn.W_v.weight.data.zero_()
        for d in range(D):
            block1.attn.W_v.weight.data[d, d] = 1.0
        
        # Output projection: identity
        block1.attn.W_o.weight.data.zero_()
        for d in range(D):
            block1.attn.W_o.weight.data[d, d] = 1.0
        
        # MLP 1: Per-character sentiment scoring
        # After LN, the one-hot character identity has ~8 in active dim, ~-0.13 in others
        # Use hidden units to detect specific characters and assign sentiment
        block1.mlp.fc1.weight.data.zero_()
        block1.mlp.fc1.bias.data.zero_()
        block1.mlp.fc2.weight.data.zero_()
        block1.mlp.fc2.bias.data.zero_()
        
        # Use hidden units to detect positive/negative sentiment from dim 32
        # After LN, dim 32 will be normalized. The sentiment signal is small relative to
        # the one-hot identity. Let me boost it differently.
        
        # Actually, the issue is LayerNorm normalizes everything. So the sentiment in dim 32
        # will be normalized relative to the one-hot identity dims.
        # After LN, active char dim ≈ 7.94, dim 32 ≈ score*3/0.124 (depends on score)
        # For score=0.25: 0.75/0.124 ≈ 6.05 (quite large relative to char identity!)
        # For score=-0.15: -0.45/0.124 ≈ -3.63
        
        # Hidden unit 0: amplify positive sentiment from dim 32
        block1.mlp.fc1.weight.data[0, 32] = 3.0
        block1.mlp.fc1.bias.data[0] = 0.0  # threshold at 0
        # Hidden unit 1: amplify negative sentiment from dim 32
        block1.mlp.fc1.weight.data[1, 32] = -3.0
        block1.mlp.fc1.bias.data[1] = 0.0
        
        # Output: accumulate positive and negative counts
        block1.mlp.fc2.weight.data[34, 0] = 2.0   # positive hits -> dim 34
        block1.mlp.fc2.weight.data[35, 1] = 2.0   # negative hits -> dim 35
        
        # Layer norms
        for block in model.blocks:
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
        
        # ---- LAYER 2: AGGREGATE SENTIMENT ----
        block2 = model.blocks[1]
        
        # Uniform attention to average sentiment across all positions
        block2.attn.W_q.weight.data.zero_()
        block2.attn.W_k.weight.data.zero_()
        block2.attn.W_v.weight.data.zero_()
        # Read dims 34 and 35, put into heads 0 and 1
        block2.attn.W_v.weight.data[0, 34] = 1.0   # positive count
        block2.attn.W_v.weight.data[1, 35] = 1.0   # negative count
        for d in range(2, DH):
            block2.attn.W_v.weight.data[d, d] = 1.0  # pass through other dims
        
        # Head 1 reads dims 34, 35 too
        block2.attn.W_v.weight.data[DH, 34] = 1.0
        block2.attn.W_v.weight.data[DH+1, 35] = 1.0
        
        # Output: route to dims 0-1
        block2.attn.W_o.weight.data.zero_()
        block2.attn.W_o.weight.data[0, 0] = 1.0     # positive aggregate -> dim 0
        block2.attn.W_o.weight.data[1, 1] = 1.0     # negative aggregate -> dim 1
        
        # MLP 2: Decision circuit
        block2.mlp.fc1.weight.data.zero_()
        block2.mlp.fc1.bias.data.zero_()
        block2.mlp.fc2.weight.data.zero_()
        block2.mlp.fc2.bias.data.zero_()
        
        # Hidden unit 0: net positive (pos > neg with bias toward positive)
        block2.mlp.fc1.weight.data[0, 0] = 5.0
        block2.mlp.fc1.weight.data[0, 1] = -5.0
        block2.mlp.fc1.bias.data[0] = 0.5   # bias toward positive
        # Hidden unit 1: net negative
        block2.mlp.fc1.weight.data[1, 0] = -5.0
        block2.mlp.fc1.weight.data[1, 1] = 5.0
        block2.mlp.fc1.bias.data[1] = -0.5
        
        # Output to dim 36 (decision signal)
        block2.mlp.fc2.weight.data[36, 0] = 3.0
        block2.mlp.fc2.weight.data[36, 1] = -3.0
        
        # ---- LAYER 3: PASS THROUGH ----
        block3 = model.blocks[2]
        block3.attn.W_q.weight.data.zero_()
        block3.attn.W_k.weight.data.zero_()
        block3.attn.W_v.weight.data.zero_()
        block3.attn.W_o.weight.data.zero_()
        block3.mlp.fc1.weight.data.zero_()
        block3.mlp.fc1.bias.data.zero_()
        block3.mlp.fc2.weight.data.zero_()
        block3.mlp.fc2.bias.data.zero_()
        
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # ---- OUTPUT HEAD ----
        model.head.weight.data.zero_()
        # After final LN, dim 36 will be normalized but sign preserved
        model.head.weight.data[31, 36] = 15.0   # high for '1' when positive
        model.head.weight.data[30, 36] = -15.0  # low for '0' when positive


model_shorthand_name = "sentiment-bag-of-chars-v2"
model_description = "Improved bag-of-chars sentiment with better character scores and cleaner circuit"


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
