"""
Interpretable transformer for TREC coarse question classification.
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
    """
    Strategy: Encode question word signals in token embeddings.
    Use attention to aggregate to last position.
    Classify based on aggregated signals.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # Encode category signals in character embeddings
        # Question words and their typical categories:
        # who -> HUM (3), where -> LOC (4), when -> NUM (5), how -> NUM (5)
        # what/which -> depends on context
        
        # 'w' appears in who, what, where, when, which
        # 'h' appears in who, what, where, when, which, how
        # etc.
        
        # Assign category signals to characters that appear in question words
        char_cat_signals = {
            'w': {3: 0.8, 4: 0.8, 5: 0.8, 1: 0.3, 2: 0.3},  # who/where->HUM/LOC, when->NUM
            'h': {3: 0.9, 4: 0.9, 5: 0.9, 1: 0.4, 2: 0.4},  # appears in all question words
            'o': {3: 1.0, 5: 0.7},  # who, how
            'a': {1: 0.6, 2: 0.6},  # what
            't': {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5},  # what, which
            'e': {4: 0.9, 5: 0.8},  # where, when
            'r': {4: 1.0},  # where
            'n': {5: 1.0},  # when
            'i': {2: 0.7, 3: 0.5},  # which
            'c': {2: 0.7, 3: 0.5},  # which
        }
        
        # Encode signals in token embeddings
        for char, cat_scores in char_cat_signals.items():
            if char in stoi:
                token_id = stoi[char]
                for cat, score in cat_scores.items():
                    # Use dimensions 30+cat*5 to 30+cat*5+5 for category cat
                    start_dim = 30 + cat * 5
                    end_dim = min(start_dim + 5, d)
                    model.token_emb.weight[token_id, start_dim:end_dim] = score
        
        # Position embeddings: make early positions strong
        for p in range(min(10, model.max_seq_len)):
            weight = 1.0 - p * 0.1
            model.pos_emb.weight[p, 0:30] = weight * 0.5
        
        # Make last position (81) special for prediction
        model.pos_emb.weight[81, 0:6] = 10.0
        
        # Attention: identity-like to pass through signals
        for block in model.blocks:
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            
            for i in range(d):
                block.attn.W_q.weight[i, i] = 1.0
                block.attn.W_k.weight[i, i] = 1.0
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0 / model.n_heads
            
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            d_ff = block.mlp.fc1.weight.shape[0]
            for i in range(min(d, d_ff)):
                block.mlp.fc1.weight[i, i] = 1.0
                block.mlp.fc2.weight[i, i] = 1.0
        
        # Output head: map category signals to output tokens
        model.head.weight.zero_()
        
        # Output tokens 30-35 correspond to '0'-'5'
        for cat in range(6):
            start_dim = 30 + cat * 5
            end_dim = min(start_dim + 5, d)
            model.head.weight[30 + cat, start_dim:end_dim] = 1.0
    
    return


model_shorthand_name = "exp03_question_word_signals"
model_description = "Encode question word signals (who/what/where/when/how) in token embeddings, use attention to aggregate, classify via output head."


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
