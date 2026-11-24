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


def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model: SimpleTransformer, task) -> None:
    """
    3-layer transformer with clean dimension allocation:
    - Dims 0-31: Character identity (one-hot per position)
    - Dims 32-37: Category signals (ABBR/DESC/ENTY/HUM/LOC/NUM)
    - Dim 60: Attention routing signal
    
    Position 81 attends to positions 0-3 via dim 60 routing.
    MLP detects question word patterns and writes category to dims 32-37.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Dim 60: routing signal
        # Positions 0-3: "I have info" (key signal)
        for pos_idx in range(4):
            model.pos_emb.weight[pos_idx, 60] = 50.0
        
        # Position 81: "I want info" (query signal)
        model.pos_emb.weight[81, 60] = 50.0
        
        # All other positions: dim 60 = 0 (no routing signal)
        
        # === TOKEN EMBEDDINGS ===
        # Encode character identity in position-specific dimensions
        # Position 0 (char 0): dims 0-9
        model.token_emb.weight[stoi['w'], 0] = 1.0  # w at pos 0
        model.token_emb.weight[stoi['h'], 1] = 1.0  # h at pos 0
        
        # Position 1 (char 1): dims 10-19
        model.token_emb.weight[stoi['h'], 10] = 1.0  # h at pos 1
        model.token_emb.weight[stoi['a'], 11] = 1.0  # a at pos 1
        model.token_emb.weight[stoi['o'], 12] = 1.0  # o at pos 1
        
        # Position 2 (char 2): dims 20-29
        model.token_emb.weight[stoi['o'], 20] = 1.0  # o at pos 2 (who/how)
        model.token_emb.weight[stoi['a'], 21] = 1.0  # a at pos 2 (what)
        model.token_emb.weight[stoi['e'], 22] = 1.0  # e at pos 2 (where/when)
        model.token_emb.weight[stoi['i'], 23] = 1.0  # i at pos 2 (which)
        model.token_emb.weight[stoi['y'], 24] = 1.0  # y at pos 2 (why)
        
        # Position 3 (char 3): dims 30-31
        model.token_emb.weight[stoi['r'], 30] = 1.0  # r at pos 3 (where)
        model.token_emb.weight[stoi['n'], 31] = 1.0  # n at pos 3 (when)
        
        # === ATTENTION (Block 0) ===
        # Aggregate character info from positions 0-3
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Q and K use dim 60 for routing
        for i in range(d):
            block0.attn.W_q.weight[i, 60] = 2.0
            block0.attn.W_k.weight[i, 60] = 2.0
        
        # V passes through character dims 0-31
        for i in range(32):
            block0.attn.W_v.weight[i, i] = 1.0
        
        # O passes through
        for i in range(d):
            block0.attn.W_o.weight[i, i] = 1.0
        
        # MLP (Block 0): Detect question word patterns, write to dims 32-37
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        
        # Pattern detectors using AND gates
        # Each detector: sum inputs - (n-0.5), then amplify
        
        # Pattern 0: "who" = w(0) + h(10) + o(20) → HUM (dim 35)
        block0.mlp.fc1.weight[0, 0] = 1.0
        block0.mlp.fc1.weight[0, 10] = 1.0
        block0.mlp.fc1.weight[0, 20] = 1.0
        block0.mlp.fc1.bias[0] = -2.5
        block0.mlp.fc2.weight[35, 0] = 10.0
        
        # Pattern 1: "what" = w(0) + h(10) + a(21) → DESC (dim 33)
        block0.mlp.fc1.weight[1, 0] = 1.0
        block0.mlp.fc1.weight[1, 10] = 1.0
        block0.mlp.fc1.weight[1, 21] = 1.0
        block0.mlp.fc1.bias[1] = -2.5
        block0.mlp.fc2.weight[33, 1] = 10.0
        
        # Pattern 2: "where" = w(0) + h(10) + e(22) + r(30) → LOC (dim 36)
        block0.mlp.fc1.weight[2, 0] = 1.0
        block0.mlp.fc1.weight[2, 10] = 1.0
        block0.mlp.fc1.weight[2, 22] = 1.0
        block0.mlp.fc1.weight[2, 30] = 1.0
        block0.mlp.fc1.bias[2] = -3.5
        block0.mlp.fc2.weight[36, 2] = 10.0
        
        # Pattern 3: "when" = w(0) + h(10) + e(22) + n(31) → NUM (dim 37)
        block0.mlp.fc1.weight[3, 0] = 1.0
        block0.mlp.fc1.weight[3, 10] = 1.0
        block0.mlp.fc1.weight[3, 22] = 1.0
        block0.mlp.fc1.weight[3, 31] = 1.0
        block0.mlp.fc1.bias[3] = -3.5
        block0.mlp.fc2.weight[37, 3] = 10.0
        
        # Pattern 4: "which" = w(0) + h(10) + i(23) → ENTY (dim 34)
        block0.mlp.fc1.weight[4, 0] = 1.0
        block0.mlp.fc1.weight[4, 10] = 1.0
        block0.mlp.fc1.weight[4, 23] = 1.0
        block0.mlp.fc1.bias[4] = -2.5
        block0.mlp.fc2.weight[34, 4] = 10.0
        
        # Pattern 5: "why" = w(0) + h(10) + y(24) → DESC (dim 33)
        block0.mlp.fc1.weight[5, 0] = 1.0
        block0.mlp.fc1.weight[5, 10] = 1.0
        block0.mlp.fc1.weight[5, 24] = 1.0
        block0.mlp.fc1.bias[5] = -2.5
        block0.mlp.fc2.weight[33, 5] = 10.0
        
        # Pattern 6: "how" = h(1) + o(12) + o(20) → NUM (dim 37)
        block0.mlp.fc1.weight[6, 1] = 1.0
        block0.mlp.fc1.weight[6, 12] = 1.0
        block0.mlp.fc1.weight[6, 20] = 1.0
        block0.mlp.fc1.bias[6] = -2.5
        block0.mlp.fc2.weight[37, 6] = 10.0
        
        # === BLOCKS 1-2: Pass through ===
        for block in model.blocks[1:]:
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        # Map category signals (dims 32-37) to output tokens (30-35)
        model.head.weight[30, 32] = 50.0  # ABBR
        model.head.weight[31, 33] = 50.0  # DESC
        model.head.weight[32, 34] = 50.0  # ENTY
        model.head.weight[33, 35] = 50.0  # HUM
        model.head.weight[34, 36] = 50.0  # LOC
        model.head.weight[35, 37] = 50.0  # NUM
    
    return


model_shorthand_name = "exp19_clean_routing"
model_description = "Clean dim allocation: chars in 0-31, categories in 32-37, routing in dim 60. MLP as AND-gate pattern detector."


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
