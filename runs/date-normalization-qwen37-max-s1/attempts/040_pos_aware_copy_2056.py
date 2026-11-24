"""
Interpretable transformer for date-normalization task.
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
# Architecture
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
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 256):
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Experiment 25: Position-aware copy with format detection.
    
    Strategy: Use strong position signals to route information from input to output.
    Layer 0: Detect year (4 consecutive digits)
    Layer 1: Extract and copy year to output positions 0-3
    Layer 2: Detect month and day positions
    Layer 3: Copy month/day to output positions 5-9
    """
    torch.manual_seed(2056)
    D = model.d_model
    vocab = task.vocab
    stoi = {c: i for i, c in enumerate(vocab)}
    
    with torch.no_grad():
        # One-hot token embeddings with scale 5.0
        model.token_emb.weight.data.zero_()
        for i in range(min(task.vocab_size, D)):
            model.token_emb.weight.data[i, i] = 5.0
        
        # Strong position embeddings - each position gets unique signature
        model.pos_emb.weight.data.zero_()
        for pos in range(model.max_seq_len):
            # Use different frequency bands for different position ranges
            if pos < 20:  # Input region
                model.pos_emb.weight.data[pos, 0:4] = pos * 0.1
                model.pos_emb.weight.data[pos, 4:8] = math.sin(pos * 0.5)
            elif pos >= 20 and pos < 30:  # Padding region
                model.pos_emb.weight.data[pos, 8:12] = 2.0
            else:  # Output region (after =)
                out_pos = pos - 27  # Approximate output start
                model.pos_emb.weight.data[pos, 12:16] = out_pos * 0.2
                model.pos_emb.weight.data[pos, 16:20] = math.cos(out_pos * 0.7)
        
        # Layer 0: Find year (4 consecutive digits)
        block = model.blocks[0]
        # Q/K attend to digit patterns
        for i in range(D):
            block.attn.W_q.weight.data[i, i] = 0.8 if i < 20 else 0.1
            block.attn.W_k.weight.data[i, i] = 0.8 if i < 20 else 0.1
        # V copies digit values strongly
        for i in range(D):
            block.attn.W_v.weight.data[i, i] = 1.2 if i < 20 else 0.05
            block.attn.W_o.weight.data[i, i] = 1.0
        
        block.mlp.fc1.weight.data.normal_(0, 0.1)
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.normal_(0, 0.1)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        # Layer 1: Copy year to output positions 0-3
        block = model.blocks[1]
        # Strong position-based attention
        for i in range(D):
            block.attn.W_q.weight.data[i, i] = 0.9
            block.attn.W_k.weight.data[i, i] = 0.9
            block.attn.W_v.weight.data[i, i] = 1.1
            block.attn.W_o.weight.data[i, i] = 1.0
        
        block.mlp.fc1.weight.data.normal_(0, 0.08)
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.normal_(0, 0.08)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        # Layer 2: Extract month and day
        block = model.blocks[2]
        for i in range(D):
            block.attn.W_q.weight.data[i, i] = 0.85
            block.attn.W_k.weight.data[i, i] = 0.85
            block.attn.W_v.weight.data[i, i] = 1.15
            block.attn.W_o.weight.data[i, i] = 1.0
        
        block.mlp.fc1.weight.data.normal_(0, 0.09)
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.normal_(0, 0.09)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        # Layer 3: Format output as YYYY-MM-DD
        block = model.blocks[3]
        for i in range(D):
            block.attn.W_q.weight.data[i, i] = 0.95
            block.attn.W_k.weight.data[i, i] = 0.95
            block.attn.W_v.weight.data[i, i] = 1.05
            block.attn.W_o.weight.data[i, i] = 1.0
        
        block.mlp.fc1.weight.data.normal_(0, 0.07)
        block.mlp.fc1.bias.data.zero_()
        block.mlp.fc2.weight.data.normal_(0, 0.07)
        block.mlp.fc2.bias.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
        
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # Output head: strong identity mapping for digits
        model.head.weight.data.zero_()
        for d in '0123456789':
            if d in stoi:
                idx = stoi[d]
                model.head.weight.data[idx, idx] = 3.0
        if '-' in stoi:
            idx = stoi['-']
            model.head.weight.data[idx, idx] = 2.5


model_shorthand_name = "pos-aware-copy-2056"
model_description = "Position-aware copy with format detection, strong position signals for routing."


def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
    )
    write_weights(model, task)
    model.eval()
    return model


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
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
        "task": args.task,
        "accuracy": f"{accuracy:.4f}",
        "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
