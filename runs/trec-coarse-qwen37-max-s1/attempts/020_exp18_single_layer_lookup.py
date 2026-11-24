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
    """Build a 1-layer transformer for maximum interpretability."""
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        n_heads=4,
        n_layers=1,  # Single layer
        d_ff=256
    )
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model: SimpleTransformer, task) -> None:
    """
    Single-layer transformer with aggressive MLP logic.
    Use each head to attend to a different position (0-3).
    MLP acts as a lookup table for question word patterns.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Each position gets a unique "address" in dims 60-63
        for pos_idx in range(4):
            model.pos_emb.weight[pos_idx, 60 + pos_idx] = 100.0
        
        # Position 81: query signal
        model.pos_emb.weight[81, 60:64] = 100.0
        
        # === TOKEN EMBEDDINGS ===
        # Encode characters in dims 0-59
        
        # Position 0: First character
        model.token_emb.weight[stoi['w'], 0] = 1.0  # w
        model.token_emb.weight[stoi['h'], 1] = 1.0  # h
        
        # Position 1: Second character
        model.token_emb.weight[stoi['h'], 10] = 1.0  # h
        model.token_emb.weight[stoi['a'], 11] = 1.0  # a
        model.token_emb.weight[stoi['o'], 12] = 1.0  # o
        
        # Position 2: Third character (KEY)
        model.token_emb.weight[stoi['o'], 20] = 1.0  # o (who/how)
        model.token_emb.weight[stoi['a'], 21] = 1.0  # a (what)
        model.token_emb.weight[stoi['e'], 22] = 1.0  # e (where/when)
        model.token_emb.weight[stoi['i'], 23] = 1.0  # i (which)
        model.token_emb.weight[stoi['y'], 24] = 1.0  # y (why)
        
        # Position 3: Fourth character
        model.token_emb.weight[stoi['r'], 30] = 1.0  # r (where)
        model.token_emb.weight[stoi['n'], 31] = 1.0  # n (when)
        model.token_emb.weight[stoi['t'], 32] = 1.0  # t (what)
        
        # === ATTENTION ===
        block = model.blocks[0]
        block.attn.W_q.weight.zero_()
        block.attn.W_k.weight.zero_()
        block.attn.W_v.weight.zero_()
        block.attn.W_o.weight.zero_()
        
        # Each head attends to a different position
        # Head 0 → position 0, Head 1 → position 1, etc.
        d_head = d // 4  # 16
        
        for h in range(4):
            offset = h * d_head
            # Query for head h
            block.attn.W_q.weight[offset:offset+d_head, 60 + h] = 1.0
            # Key for head h
            block.attn.W_k.weight[offset:offset+d_head, 60 + h] = 1.0
            # Value for head h: pass through character dims
            for i in range(d_head):
                block.attn.W_v.weight[offset + i, offset + i] = 1.0
        
        # Output projection: combine heads
        for i in range(d):
            block.attn.W_o.weight[i, i] = 1.0
        
        # === MLP: Lookup table for question patterns ===
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
        
        # Map character patterns to categories
        d_ff = block.mlp.fc1.weight.shape[0]
        
        # Pattern: "who" → HUM (cat 3)
        # Dims: w(0), h(10), o(20)
        block.mlp.fc1.weight[0, 0] = 1.0   # w
        block.mlp.fc1.weight[0, 10] = 1.0  # h
        block.mlp.fc1.weight[0, 20] = 1.0  # o
        block.mlp.fc1.bias[0] = -2.5       # Need all 3
        block.mlp.fc2.weight[63, 0] = 1.0  # → HUM
        
        # Pattern: "what" → DESC (cat 1)
        block.mlp.fc1.weight[1, 0] = 1.0   # w
        block.mlp.fc1.weight[1, 10] = 1.0  # h
        block.mlp.fc1.weight[1, 21] = 1.0  # a
        block.mlp.fc1.bias[1] = -2.5
        block.mlp.fc2.weight[61, 1] = 1.0  # → DESC
        
        # Pattern: "where" → LOC (cat 4)
        block.mlp.fc1.weight[2, 0] = 1.0   # w
        block.mlp.fc1.weight[2, 10] = 1.0  # h
        block.mlp.fc1.weight[2, 22] = 1.0  # e
        block.mlp.fc1.weight[2, 30] = 1.0  # r
        block.mlp.fc1.bias[2] = -3.5
        block.mlp.fc2.weight[64, 2] = 1.0  # → LOC
        
        # Pattern: "when" → NUM (cat 5)
        block.mlp.fc1.weight[3, 0] = 1.0   # w
        block.mlp.fc1.weight[3, 10] = 1.0  # h
        block.mlp.fc1.weight[3, 22] = 1.0  # e
        block.mlp.fc1.weight[3, 31] = 1.0  # n
        block.mlp.fc1.bias[3] = -3.5
        block.mlp.fc2.weight[65, 3] = 1.0  # → NUM
        
        # Pattern: "which" → ENTY (cat 2)
        block.mlp.fc1.weight[4, 0] = 1.0   # w
        block.mlp.fc1.weight[4, 10] = 1.0  # h
        block.mlp.fc1.weight[4, 23] = 1.0  # i
        block.mlp.fc1.bias[4] = -2.5
        block.mlp.fc2.weight[62, 4] = 1.0  # → ENTY
        
        # Pattern: "why" → DESC (cat 1)
        block.mlp.fc1.weight[5, 0] = 1.0   # w
        block.mlp.fc1.weight[5, 10] = 1.0  # h
        block.mlp.fc1.weight[5, 24] = 1.0  # y
        block.mlp.fc1.bias[5] = -2.5
        block.mlp.fc2.weight[61, 5] = 1.0  # → DESC
        
        # Pattern: "how" → NUM (cat 5)
        block.mlp.fc1.weight[6, 1] = 1.0   # h
        block.mlp.fc1.weight[6, 12] = 1.0  # o
        block.mlp.fc1.weight[6, 20] = 1.0  # o
        block.mlp.fc1.bias[6] = -2.5
        block.mlp.fc2.weight[65, 6] = 1.0  # → NUM
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        # Map category signals (dims 61-65) to output tokens (30-35)
        model.head.weight[30, 60] = 50.0  # ABBR
        model.head.weight[31, 61] = 50.0  # DESC
        model.head.weight[32, 62] = 50.0  # ENTY
        model.head.weight[33, 63] = 50.0  # HUM
        model.head.weight[34, 64] = 50.0  # LOC
        model.head.weight[35, 65] = 50.0  # NUM
    
    return


model_shorthand_name = "exp18_single_layer_lookup"
model_description = "1-layer transformer, each head attends to different position, MLP as lookup table for question patterns."


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
