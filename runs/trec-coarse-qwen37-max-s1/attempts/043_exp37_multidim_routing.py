"""
Interpretable transformer for TREC coarse question classification.

Key insight: LayerNorm preserves DIRECTIONS but normalizes magnitudes.
To route attention, use MANY dimensions (50+) for the position signal,
so the dot product is large enough to overcome the sqrt(dh) division.
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
    """Build a 1-layer transformer with n_heads=1 and d_model=128."""
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=1,
        n_layers=1,
        d_ff=256
    )
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model: SimpleTransformer, task) -> None:
    """
    Multi-dimensional position routing:
    - Dims 0-59 (60 dims): routing signal (+200 for pos 0-4, -200 for pos 5+)
    - Dims 60-65 (6 dims): category signals (only at positions 0-4)
    - Dims 66-127: unused
    
    After LayerNorm with d_model=128:
    - Positions 0-4: routing_norm ≈ 0.13, category_norm ≈ 0.64
    - Positions 5+: routing_norm ≈ -0.13 (opposite direction), no category
    
    Q/K use routing dims with weight=3.0:
    - Score(pos 0) ≈ +1.3 / sqrt(128) ≈ +0.115
    - Score(pos 5) ≈ -1.3 / sqrt(128) ≈ -0.115
    - Diff = 0.23 → exp(0.23) ≈ 1.26x per position
    - Total: 5 * 1.26 vs 80 * 0.8 → pos 0-4 get ~10% of attention
    
    That's still not enough. Let me use weight=10.0:
    - Score diff per dim pair = 20 * 0.26 = 5.2 / sqrt(128) = 0.46
    - exp(0.46) ≈ 1.58 → 5 * 1.58 vs 80 * 0.63 → ~15%
    
    Need even stronger. Use weight=50.0:
    - Score diff = 100 * 0.26 = 26 / sqrt(128) = 2.3
    - exp(2.3) ≈ 10 → 5 * 10 vs 80 * 0.1 → 86%!
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 128
        n_routing = 60  # dims 0-59
        cat_start = 60  # dims 60-65
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Routing dims 0-59: +200 for pos 0-4, -200 for pos 5+
        for pos_idx in range(5):
            for dim_idx in range(n_routing):
                model.pos_emb.weight[pos_idx, dim_idx] = 200.0
        
        for pos_idx in range(5, model.max_seq_len):
            for dim_idx in range(n_routing):
                model.pos_emb.weight[pos_idx, dim_idx] = -200.0
        
        # === TOKEN EMBEDDINGS ===
        # Category dims 60-65: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        # Strong signals at positions 0-4 only
        
        # Position 0: first char
        model.token_emb.weight[stoi['w'], cat_start + 1] = 2000.0   # DESC
        model.token_emb.weight[stoi['w'], cat_start + 2] = 1500.0   # ENTY
        model.token_emb.weight[stoi['w'], cat_start + 4] = 1000.0   # LOC
        model.token_emb.weight[stoi['w'], cat_start + 5] = 800.0    # NUM
        model.token_emb.weight[stoi['h'], cat_start + 5] = 5000.0   # how → NUM
        
        # Position 1: second char
        model.token_emb.weight[stoi['h'], cat_start + 1] = 1000.0   # DESC
        model.token_emb.weight[stoi['o'], cat_start + 5] = 3000.0   # how → NUM
        
        # Position 2: KEY discriminator
        model.token_emb.weight[stoi['o'], cat_start + 3] = 5000.0   # who → HUM
        model.token_emb.weight[stoi['o'], cat_start + 5] = 4500.0   # how → NUM
        
        model.token_emb.weight[stoi['a'], cat_start + 1] = 4500.0   # what → DESC
        model.token_emb.weight[stoi['a'], cat_start + 2] = 3500.0   # what → ENTY
        
        model.token_emb.weight[stoi['e'], cat_start + 4] = 5000.0   # where → LOC
        model.token_emb.weight[stoi['e'], cat_start + 5] = 4500.0   # when → NUM
        
        model.token_emb.weight[stoi['i'], cat_start + 2] = 5000.0   # which → ENTY
        model.token_emb.weight[stoi['i'], cat_start + 3] = 1500.0   # which → HUM
        
        model.token_emb.weight[stoi['y'], cat_start + 1] = 5500.0   # why → DESC
        
        # Position 3: refinement
        model.token_emb.weight[stoi['r'], cat_start + 4] = 3000.0   # where → LOC
        model.token_emb.weight[stoi['n'], cat_start + 5] = 3000.0   # when → NUM
        model.token_emb.weight[stoi['t'], cat_start + 2] = 2500.0   # what → ENTY
        model.token_emb.weight[stoi['c'], cat_start + 2] = 2000.0   # which → ENTY
        
        # Position 4: further refinement
        model.token_emb.weight[stoi['e'], cat_start + 4] = 2000.0   # where (wh-e-re)
        model.token_emb.weight[stoi['n'], cat_start + 5] = 2000.0   # when (whe-n)
        
        # === ATTENTION ===
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Q and K: use routing dims 0-59 with strong weights
        routing_weight = 50.0
        for i in range(n_routing):
            block0.attn.W_q.weight[i, i] = routing_weight
            block0.attn.W_k.weight[i, i] = routing_weight
        
        # V and O: pass category dims 60-65
        for i in range(6):
            block0.attn.W_v.weight[cat_start + i, cat_start + i] = 1.0
            block0.attn.W_o.weight[cat_start + i, cat_start + i] = 1.0
        
        # === MLP: Amplify category signals ===
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        
        for i in range(6):
            block0.mlp.fc1.weight[i, cat_start + i] = 20.0
            block0.mlp.fc2.weight[cat_start + i, i] = 20.0
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        for cat in range(6):
            model.head.weight[30 + cat, cat_start + cat] = 200.0
    
    return


model_shorthand_name = "exp37_multidim_routing"
model_description = "60-dim routing (+200/-200), 6 cat dims (2000-5500), n_heads=1, d_model=128, Q/K=50, MLP 20x."


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
