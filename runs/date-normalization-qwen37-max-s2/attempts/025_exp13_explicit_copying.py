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
        n_layers: int = 4,
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
    """Attempt 13: Explicit position-based copying for YYYY.MM.DD format.
    
    For YYYY.MM.DD at positions 0-9, output at 32-41:
    - Position 32 should copy from position 0
    - Position 33 should copy from position 1
    - Position 34 should copy from position 2
    - Position 35 should copy from position 3
    - Position 36 should output '-'
    - Position 37 should copy from position 5
    - Position 38 should copy from position 6
    - Position 39 should output '-'
    - Position 40 should copy from position 8
    - Position 41 should copy from position 9
    """
    torch.manual_seed(303)
    
    d = model.d_model
    V = model.vocab_size
    vocab = task.vocab
    c2i = {c: i for i, c in enumerate(vocab)}
    
    # Mapping: output_pos -> input_pos (or -1 for special chars)
    copy_map = {
        32: 0, 33: 1, 34: 2, 35: 3,  # YYYY
        36: -1,                        # -
        37: 5, 38: 6,                  # MM
        39: -1,                        # -
        40: 8, 41: 9,                  # DD
    }
    
    with torch.no_grad():
        # Token embeddings - one-hot
        model.token_emb.weight.zero_()
        for i in range(V):
            model.token_emb.weight[i, i % d] = 1.0
        
        # Position embeddings - encode position identity
        model.pos_emb.weight.zero_()
        for pos in range(model.max_seq_len):
            # Use first 16 dims for position identity (binary)
            for j in range(min(d, 16)):
                if (pos >> j) & 1:
                    model.pos_emb.weight[pos, j] = 2.0
        
        # Layer 1: Build position-aware representations
        layer = model.blocks[0]
        layer.attn.W_q.weight.zero_()
        layer.attn.W_k.weight.zero_()
        layer.attn.W_v.weight.zero_()
        layer.attn.W_o.weight.zero_()
        
        # Self-attention
        for i in range(d):
            layer.attn.W_q.weight[i, i] = 1.0
            layer.attn.W_k.weight[i, i] = 1.0
            layer.attn.W_v.weight[i, i] = 1.0
            layer.attn.W_o.weight[i, i] = 1.0
        
        layer.mlp.fc1.weight.zero_()
        layer.mlp.fc1.bias.zero_()
        layer.mlp.fc2.weight.zero_()
        layer.mlp.fc2.bias.zero_()
        for i in range(min(d, layer.mlp.fc1.out_features)):
            layer.mlp.fc1.weight[i, i] = 1.0
        for i in range(min(layer.mlp.fc2.out_features, d)):
            layer.mlp.fc2.weight[i, i] = 1.0
        
        # Layer 2: Explicit copying based on copy_map
        layer = model.blocks[1]
        layer.attn.W_q.weight.zero_()
        layer.attn.W_k.weight.zero_()
        layer.attn.W_v.weight.zero_()
        layer.attn.W_o.weight.zero_()
        
        # Use 10 heads, one for each output position
        n_heads = min(layer.attn.n_heads, 10)
        d_head = layer.attn.d_head
        
        for h in range(n_heads):
            output_pos = 32 + h
            input_pos = copy_map.get(output_pos, -1)
            
            if input_pos >= 0:
                # This head copies from input_pos to output_pos
                start = h * d_head
                end = start + d_head
                
                # Q encodes "I am at output_pos and want to copy from input_pos"
                # K encodes "I am at input_pos"
                # V copies the token value
                
                for i in range(start, min(end, d)):
                    # Strong query for this specific input position
                    layer.attn.W_q.weight[i, i] = 3.0
                    layer.attn.W_k.weight[i, i] = 3.0
                    layer.attn.W_v.weight[i, i] = 1.0
                    layer.attn.W_o.weight[i, i] = 1.0 / n_heads
        
        layer.mlp.fc1.weight.zero_()
        layer.mlp.fc1.bias.zero_()
        layer.mlp.fc2.weight.zero_()
        layer.mlp.fc2.bias.zero_()
        for i in range(min(d, layer.mlp.fc1.out_features)):
            layer.mlp.fc1.weight[i, i] = 1.0
        for i in range(min(layer.mlp.fc2.out_features, d)):
            layer.mlp.fc2.weight[i, i] = 1.0
        
        # Layers 3-4: Refinement
        for layer in model.blocks[2:]:
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


model_shorthand_name = "exp13-explicit-copying"
model_description = "Explicit position-based copying for YYYY.MM.DD format"


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
        n_layers=4,
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
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    if args.debug:
        # Print vocab and task details
        print(f"Vocab ({task.vocab_size}): {task.vocab}")
        print(f"seq_len: {task.seq_len}")
        
        # Generate a few examples and show token IDs
        examples = task.generate_examples(3)
        for i, ex in enumerate(examples):
            print(f"\n=== Example {i+1} ===")
            print(f"  attrs: {[a for a in dir(ex) if not a.startswith('_')]}")
            # Try common attribute names
            for attr in ['prompt', 'input', 'question', 'text', 'source', 'x']:
                if hasattr(ex, attr):
                    print(f"  {attr}: {getattr(ex, attr)}")
            for attr in ['target', 'answer', 'output', 'label', 'y']:
                if hasattr(ex, attr):
                    print(f"  {attr}: {getattr(ex, attr)}")
            if i >= 2:
                break

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
