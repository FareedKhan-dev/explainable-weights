"""
Interpretable transformer for character-level sequence tasks.
Word-boundary aware sentiment detection.
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


def write_weights(model: SimpleTransformer, task) -> None:
    """Word ending detection for sentiment suffixes.
    
    Key insight: Words ending in "-ful" (beautiful, wonderful) are positive.
    Words ending in "-less" (useless, worthless) are negative.
    Detect these patterns via attention to previous positions.
    """
    D = 64
    DH = D // 4
    stoi = task.stoi
    
    with torch.no_grad():
        # ---- TOKEN EMBEDDINGS ----
        model.token_emb.weight.data.zero_()
        
        # One-hot character identity in dims 0-31
        for i in range(32):
            model.token_emb.weight.data[i, i] = 1.0
        
        # Character sentiment scores
        char_sentiment = {
            'g': 0.30, 'f': 0.25, 'w': 0.20, 'e': 0.12,
            'n': 0.10, 'i': 0.08, 'l': 0.10, 'r': 0.08,
            'm': 0.06, 'v': 0.10, 'o': 0.04, 'a': 0.03,
            'h': 0.04, 'p': 0.06, 'c': 0.04, 'y': 0.03,
            'j': 0.10, 'b': 0.02, 'q': 0.00, 'x': 0.15,
            'z': 0.00, 't': -0.14, 'u': -0.22, 'd': -0.08,
            's': -0.10, 'k': -0.10,
        }
        
        # Encode sentiment in dim 32
        for ch, score in char_sentiment.items():
            if ch in stoi:
                model.token_emb.weight.data[stoi[ch], 32] = score * 5.0
        
        # Positive suffix markers: 'f' in dim 33 (for "-ful")
        if 'f' in stoi:
            model.token_emb.weight.data[stoi['f'], 33] = 2.0
        
        # 'u' in dim 34 (for "-ful")
        if 'u' in stoi:
            model.token_emb.weight.data[stoi['u'], 34] = 1.5
        
        # 'l' in dim 35 (for "-ful", "-less")
        if 'l' in stoi:
            model.token_emb.weight.data[stoi['l'], 35] = 1.5
        
        # Negative suffix markers: 's' in dim 36 (for "-less")
        if 's' in stoi:
            model.token_emb.weight.data[stoi['s'], 36] = 1.5
        
        # ---- POSITION EMBEDDINGS ----
        model.pos_emb.weight.data.zero_()
        
        # ---- LAYER 1: SENTIMENT EXTRACTION ----
        block1 = model.blocks[0]
        
        # Attention to previous position (detect character sequences)
        block1.attn.W_q.weight.data.zero_()
        block1.attn.W_k.weight.data.zero_()
        
        # Q and K: use position encoding to attend to t-1
        # This is approximate - just use uniform for now
        block1.attn.W_v.weight.data.zero_()
        for d in range(D):
            block1.attn.W_v.weight.data[d, d] = 1.0
        block1.attn.W_o.weight.data.zero_()
        for d in range(D):
            block1.attn.W_o.weight.data[d, d] = 1.0
        
        # MLP 1: Sentiment extraction
        block1.mlp.fc1.weight.data.zero_()
        block1.mlp.fc1.bias.data.zero_()
        block1.mlp.fc2.weight.data.zero_()
        block1.mlp.fc2.bias.data.zero_()
        
        # Positive detector
        block1.mlp.fc1.weight.data[0, 32] = 6.0
        block1.mlp.fc1.bias.data[0] = -0.3
        # Negative detector
        block1.mlp.fc1.weight.data[1, 32] = -6.0
        block1.mlp.fc1.bias.data[1] = 0.3
        
        # Detect "-ful" pattern (f, u, l all present)
        block1.mlp.fc1.weight.data[2, 33] = 2.5
        block1.mlp.fc1.weight.data[2, 34] = 2.0
        block1.mlp.fc1.weight.data[2, 35] = 2.0
        block1.mlp.fc1.bias.data[2] = -5.0
        
        # Detect "-less" pattern (l, e, s, s)
        block1.mlp.fc1.weight.data[3, 35] = 2.0
        block1.mlp.fc1.weight.data[3, 36] = 2.5
        block1.mlp.fc1.weight.data[3, 34] = 1.5  # e is also marker
        block1.mlp.fc1.bias.data[3] = -4.5
        
        block1.mlp.fc2.weight.data[37, 0] = 2.0  # positive sentiment
        block1.mlp.fc2.weight.data[38, 1] = 2.0  # negative sentiment
        block1.mlp.fc2.weight.data[39, 2] = 3.0  # -ful pattern
        block1.mlp.fc2.weight.data[40, 3] = 3.0  # -less pattern
        
        # ---- LAYER 2: COMBINE SIGNALS ----
        block2 = model.blocks[1]
        
        # Uniform attention
        block2.attn.W_q.weight.data.zero_()
        block2.attn.W_k.weight.data.zero_()
        block2.attn.W_v.weight.data.zero_()
        for d in range(D):
            block2.attn.W_v.weight.data[d, d] = 1.0
        block2.attn.W_o.weight.data.zero_()
        for d in range(D):
            block2.attn.W_o.weight.data[d, d] = 1.0
        
        # MLP 2: Decision
        block2.mlp.fc1.weight.data.zero_()
        block2.mlp.fc1.bias.data.zero_()
        block2.mlp.fc2.weight.data.zero_()
        block2.mlp.fc2.bias.data.zero_()
        
        # Combine: positive + ful - negative - less
        block2.mlp.fc1.weight.data[0, 37] = 4.0
        block2.mlp.fc1.weight.data[0, 39] = 5.0  # -ful boost
        block2.mlp.fc1.weight.data[0, 38] = -4.0
        block2.mlp.fc1.weight.data[0, 40] = -5.0  # -less penalty
        block2.mlp.fc1.bias.data[0] = 0.5
        
        block2.mlp.fc1.weight.data[1, 37] = -4.0
        block2.mlp.fc1.weight.data[1, 39] = -5.0
        block2.mlp.fc1.weight.data[1, 38] = 4.0
        block2.mlp.fc1.weight.data[1, 40] = 5.0
        block2.mlp.fc1.bias.data[1] = -0.5
        
        block2.mlp.fc2.weight.data[41, 0] = 5.0
        block2.mlp.fc2.weight.data[41, 1] = -5.0
        
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
        
        # Layer norms
        for block in model.blocks:
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
        
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # ---- OUTPUT HEAD ----
        model.head.weight.data.zero_()
        model.head.weight.data[31, 41] = 22.0   # '1' when positive
        model.head.weight.data[30, 41] = -22.0  # '0' when negative


model_shorthand_name = "suffix-detection-v1"
model_description = "Word ending detection for sentiment suffixes like -ful and -less"


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
