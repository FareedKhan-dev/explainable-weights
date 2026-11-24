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
    """Build a 1-layer transformer with 4 heads."""
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        n_heads=4,
        n_layers=1,
        d_ff=256
    )
    write_weights(model, task)
    model.eval()
    return model


def write_weights(model: SimpleTransformer, task) -> None:
    """
    Multi-head position-specific attention:
    - Head 0: output position attends to position 0
    - Head 1: output position attends to position 1  
    - Head 2: output position attends to position 2
    - Head 3: output position attends to position 3
    
    Each head uses its own dimension group for routing and category signals.
    d_model=64, n_heads=4, d_head=16
    Head 0: dims 0-15, Head 1: dims 16-31, Head 2: dims 32-47, Head 3: dims 48-63
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        dh = d // model.n_heads  # 16
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Each position gets a unique signature in its head's routing dimension
        # Head h uses dims h*16 to h*16+15
        # Routing: dim h*16 for head h
        
        # Position p gets positive signal in head p's routing dim
        for p in range(4):
            model.pos_emb.weight[p, p * dh] = 100.0
        
        # Output position (assume ~80) gets positive in ALL heads' routing dims
        for h in range(4):
            model.pos_emb.weight[80, h * dh] = 100.0
        
        # Other positions get negative in all routing dims
        for pos_idx in range(4, model.max_seq_len):
            if pos_idx != 80:
                for h in range(4):
                    model.pos_emb.weight[pos_idx, h * dh] = -50.0
        
        # === TOKEN EMBEDDINGS ===
        # Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        # Encode in category dims within each head's dimension group
        # Head 0 (pos 0): cat dims 1-6 (within dims 0-15)
        # Head 1 (pos 1): cat dims 17-22 (within dims 16-31)
        # Head 2 (pos 2): cat dims 33-38 (within dims 32-47)
        # Head 3 (pos 3): cat dims 49-54 (within dims 48-63)
        
        # Position 0 chars
        model.token_emb.weight[stoi['w'], 1] = 50.0   # DESC/ENTY/LOC
        model.token_emb.weight[stoi['w'], 2] = 30.0
        model.token_emb.weight[stoi['w'], 4] = 20.0
        model.token_emb.weight[stoi['h'], 5] = 100.0  # how → NUM
        
        # Position 1 chars
        model.token_emb.weight[stoi['h'], 17] = 40.0
        model.token_emb.weight[stoi['o'], 22] = 80.0  # how → NUM
        
        # Position 2 chars (KEY discriminator)
        model.token_emb.weight[stoi['o'], 35] = 120.0  # who → HUM
        model.token_emb.weight[stoi['o'], 38] = 100.0  # how → NUM
        
        model.token_emb.weight[stoi['a'], 33] = 110.0  # what → DESC
        model.token_emb.weight[stoi['a'], 34] = 90.0   # what → ENTY
        
        model.token_emb.weight[stoi['e'], 36] = 120.0  # where → LOC
        model.token_emb.weight[stoi['e'], 38] = 100.0  # when → NUM
        
        model.token_emb.weight[stoi['i'], 34] = 110.0  # which → ENTY
        model.token_emb.weight[stoi['i'], 35] = 50.0   # which → HUM
        
        model.token_emb.weight[stoi['y'], 33] = 130.0  # why → DESC
        
        # Position 3 chars
        model.token_emb.weight[stoi['r'], 52] = 60.0   # where → LOC
        model.token_emb.weight[stoi['n'], 54] = 60.0   # when → NUM
        model.token_emb.weight[stoi['t'], 50] = 50.0   # what → ENTY
        model.token_emb.weight[stoi['c'], 50] = 40.0   # which → ENTY
        
        # === ATTENTION ===
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Each head h: Q uses routing dim h*dh, K uses routing dim h*dh
        for h in range(4):
            # Q projection: output position's routing dim
            block0.attn.W_q.weight[h * dh, h * dh] = 5.0
            # K projection: source position's routing dim
            block0.attn.W_k.weight[h * dh, h * dh] = 5.0
            
            # V projection: pass category dims through
            for cat in range(6):
                src_dim = h * dh + 1 + cat
                block0.attn.W_v.weight[src_dim, src_dim] = 1.0
                block0.attn.W_o.weight[src_dim, src_dim] = 1.0
        
        # === MLP: Combine category signals from all heads ===
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        
        # Sum category signals from all heads into shared dims 0-5
        for h in range(4):
            for cat in range(6):
                src_dim = h * dh + 1 + cat
                # fc1: project source dim to hidden unit cat
                block0.mlp.fc1.weight[cat, src_dim] = 10.0
        
        # fc2: project hidden units back to category dims
        for cat in range(6):
            block0.mlp.fc2.weight[cat, cat] = 10.0
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        for cat in range(6):
            model.head.weight[30 + cat, cat] = 50.0
    
    return


model_shorthand_name = "exp34_multihead_position"
model_description = "4 heads attend positions 0-3 respectively. Each head has own dim group for routing and categories. MLP combines."


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
