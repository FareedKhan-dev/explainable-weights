"""
Interpretable transformer for character-level sequence tasks.

Task: date-normalization — convert dates to YYYY-MM-DD format.
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
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 48,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Attempt 11: Multi-layer structured copying."""
    torch.manual_seed(101)
    
    d = model.d_model
    V = model.vocab_size
    vocab = task.vocab
    c2i = {c: i for i, c in enumerate(vocab)}
    
    with torch.no_grad():
        # Token embeddings: one-hot with normalization
        model.token_emb.weight.zero_()
        for i in range(V):
            model.token_emb.weight[i, i % d] = 1.0
        
        # Position embeddings: encode position difference needed for copying
        # Output positions 32-41 need to copy from input positions 0-9
        # So position 32 should attend to position 0 (offset -32)
        # Position 33 should attend to position 1 (offset -32)
        # etc.
        model.pos_emb.weight.zero_()
        for pos in range(model.max_seq_len):
            # Encode position as a unique pattern
            for j in range(min(d, 32)):
                # Use sinusoidal encoding
                model.pos_emb.weight[pos, j] = math.sin(pos * (j + 1) / 10.0)
                if j + 1 < d:
                    model.pos_emb.weight[pos, j + 1] = math.cos(pos * (j + 1) / 10.0)
        
        # Layer 1: Build representations
        layer = model.blocks[0]
        layer.attn.W_q.weight.zero_()
        layer.attn.W_k.weight.zero_()
        layer.attn.W_v.weight.zero_()
        layer.attn.W_o.weight.zero_()
        
        # Identity attention - each position looks at itself
        for i in range(d):
            layer.attn.W_q.weight[i, i] = 2.0
            layer.attn.W_k.weight[i, i] = 2.0
            layer.attn.W_v.weight[i, i] = 1.0
            layer.attn.W_o.weight[i, i] = 0.5
        
        # MLP - pass through
        layer.mlp.fc1.weight.zero_()
        layer.mlp.fc1.bias.zero_()
        layer.mlp.fc2.weight.zero_()
        layer.mlp.fc2.bias.zero_()
        for i in range(min(d, layer.mlp.fc1.out_features)):
            layer.mlp.fc1.weight[i, i] = 1.0
        for i in range(min(layer.mlp.fc2.out_features, d)):
            layer.mlp.fc2.weight[i, i] = 1.0
        
        # Layer 2: Long-range copying
        layer = model.blocks[1]
        layer.attn.W_q.weight.zero_()
        layer.attn.W_k.weight.zero_()
        layer.attn.W_v.weight.zero_()
        layer.attn.W_o.weight.zero_()
        
        # Make attention look back ~30 positions
        n_heads = layer.attn.n_heads
        d_head = layer.attn.d_head
        
        for h in range(n_heads):
            start = h * d_head
            end = start + d_head
            
            # Each head looks at a different offset
            offset = 28 + h  # Offsets 28-35 for 8 heads
            
            for i in range(start, min(end, d)):
                # Q encodes "look back by offset"
                layer.attn.W_q.weight[i, i] = 1.5
                # K encodes position
                layer.attn.W_k.weight[i, i] = 1.5
                # V copies value
                layer.attn.W_v.weight[i, i] = 1.0
                layer.attn.W_o.weight[i, i] = 1.0 / n_heads
        
        # MLP - pass through
        layer.mlp.fc1.weight.zero_()
        layer.mlp.fc1.bias.zero_()
        layer.mlp.fc2.weight.zero_()
        layer.mlp.fc2.bias.zero_()
        for i in range(min(d, layer.mlp.fc1.out_features)):
            layer.mlp.fc1.weight[i, i] = 1.0
        for i in range(min(layer.mlp.fc2.out_features, d)):
            layer.mlp.fc2.weight[i, i] = 1.0
        
        # Layers 3-5: Refinement
        for layer in model.blocks[2:5]:
            layer.attn.W_q.weight.zero_()
            layer.attn.W_k.weight.zero_()
            layer.attn.W_v.weight.zero_()
            layer.attn.W_o.weight.zero_()
            
            for i in range(d):
                layer.attn.W_q.weight[i, i] = 1.0
                layer.attn.W_k.weight[i, i] = 1.0
                layer.attn.W_v.weight[i, i] = 1.0
                layer.attn.W_o.weight[i, i] = 0.5
            
            layer.mlp.fc1.weight.zero_()
            layer.mlp.fc1.bias.zero_()
            layer.mlp.fc2.weight.zero_()
            layer.mlp.fc2.bias.zero_()
            for i in range(min(d, layer.mlp.fc1.out_features)):
                layer.mlp.fc1.weight[i, i] = 0.5
            for i in range(min(layer.mlp.fc2.out_features, d)):
                layer.mlp.fc2.weight[i, i] = 0.5
        
        # Output head
        model.head.weight.zero_()
        for v in range(V):
            model.head.weight[v, v % d] = 2.0


model_shorthand_name = "exp11-multi-layer-copy"
model_description = "Multi-layer structured copying with long-range attention"


# ---------------------------------------------------------------------------
# Evaluation harness
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
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=512,
    )
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
