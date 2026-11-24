"""
Interpretable transformer for TREC coarse question classification.

Approach: Uniform attention with extremely strong category signals.
Multiple layers to amplify through LayerNorm.
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


def build_model(task) -> SimpleTransformer:
    """Build a 3-layer transformer with 2 heads."""
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        n_heads=2,
        n_layers=3,
        d_ff=256
    )
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model: SimpleTransformer, task) -> None:
    """
    Uniform attention approach with extremely strong signals.
    
    Strategy:
    1. Token embeddings: extremely strong category signals (10000-20000) in dims 0-5
    2. Position embeddings: zero (uniform attention)
    3. Attention: uniform (Q/K zeros)
    4. V/O: pass category dims 0-5
    5. MLP: strong amplification (50x per layer)
    6. Multiple layers to accumulate
    
    Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === TOKEN EMBEDDINGS ===
        # Extremely strong category signals in dims 0-5
        
        # Position 0: first char
        model.token_emb.weight[stoi['w'], 0] = 12000.0   # DESC
        model.token_emb.weight[stoi['w'], 1] = 10000.0   # ENTY
        model.token_emb.weight[stoi['w'], 3] = 8000.0    # LOC
        model.token_emb.weight[stoi['w'], 4] = 6000.0    # NUM
        model.token_emb.weight[stoi['h'], 4] = 20000.0   # how → NUM
        
        # Position 1: second char
        model.token_emb.weight[stoi['h'], 0] = 5000.0
        model.token_emb.weight[stoi['o'], 4] = 12000.0
        
        # Position 2: KEY discriminator
        model.token_emb.weight[stoi['o'], 2] = 20000.0   # who → HUM
        model.token_emb.weight[stoi['o'], 4] = 18000.0   # how → NUM
        
        model.token_emb.weight[stoi['a'], 0] = 19000.0   # what → DESC
        model.token_emb.weight[stoi['a'], 1] = 15000.0   # what → ENTY
        
        model.token_emb.weight[stoi['e'], 3] = 20000.0   # where → LOC
        model.token_emb.weight[stoi['e'], 4] = 19000.0   # when → NUM
        
        model.token_emb.weight[stoi['i'], 1] = 20000.0   # which → ENTY
        model.token_emb.weight[stoi['i'], 2] = 8000.0    # which → HUM
        
        model.token_emb.weight[stoi['y'], 0] = 22000.0   # why → DESC
        
        # Position 3: refinement
        model.token_emb.weight[stoi['r'], 3] = 10000.0   # where → LOC
        model.token_emb.weight[stoi['n'], 4] = 10000.0   # when → NUM
        model.token_emb.weight[stoi['t'], 1] = 9000.0    # what → ENTY
        model.token_emb.weight[stoi['c'], 1] = 7000.0    # which → ENTY
        
        # === ALL LAYERS: Uniform attention + amplification ===
        for block_idx, block in enumerate(model.blocks):
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            
            # Uniform attention (Q/K zeros)
            # V and O pass category dims
            for i in range(6):
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0
            
            # MLP: strong amplification (increasing per layer)
            amp = 40.0 + block_idx * 20.0
            for i in range(6):
                block.mlp.fc1.weight[i, i] = amp
                block.mlp.fc2.weight[i, i] = amp
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        for cat in range(6):
            model.head.weight[30 + cat, cat] = 500.0
    
    return


model_shorthand_name = "exp40_simplified_focused"
model_description = "Simplified version of exp08: focused attention pos 0-2, moderate signals (100-400), Q/K=3.0, MLP 8x."


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
