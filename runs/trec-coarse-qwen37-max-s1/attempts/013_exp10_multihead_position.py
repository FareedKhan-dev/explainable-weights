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
    Multi-head approach: each head attends to a different position.
    Head 0 -> pos 0, Head 1 -> pos 1, Head 2 -> pos 2, Head 3 -> pos 3
    MLP combines signals from all positions.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        n_heads = model.n_heads  # 4
        dh = d // n_heads  # 16
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Each position gets a unique signal in its own dimension range
        # Pos 0 uses dims 0-3, Pos 1 uses dims 4-7, Pos 2 uses dims 8-11, Pos 3 uses dims 12-15
        
        for pos_idx in range(4):
            start_dim = pos_idx * 4
            model.pos_emb.weight[pos_idx, start_dim:start_dim+4] = 100.0
        
        # Position 81: query all positions
        model.pos_emb.weight[81, 0:16] = 100.0
        
        # === TOKEN EMBEDDINGS ===
        # Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        # Use dims 16-21 for category signals
        
        # Position 0 (first char)
        model.token_emb.weight[stoi['w'], 16] = 200.0  # ABBR
        model.token_emb.weight[stoi['w'], 17] = 200.0  # DESC
        model.token_emb.weight[stoi['w'], 18] = 200.0  # ENTY
        model.token_emb.weight[stoi['w'], 19] = 200.0  # HUM
        model.token_emb.weight[stoi['w'], 20] = 200.0  # LOC
        model.token_emb.weight[stoi['w'], 21] = 200.0  # NUM
        
        model.token_emb.weight[stoi['h'], 21] = 400.0  # NUM (how)
        model.token_emb.weight[stoi['h'], 17] = 200.0  # DESC
        
        # Position 2 (third char - most discriminative)
        model.token_emb.weight[stoi['o'], 19] = 400.0  # HUM (who)
        model.token_emb.weight[stoi['o'], 21] = 200.0  # NUM (how)
        
        model.token_emb.weight[stoi['a'], 17] = 250.0  # DESC (what)
        model.token_emb.weight[stoi['a'], 18] = 200.0  # ENTY
        
        model.token_emb.weight[stoi['e'], 20] = 350.0  # LOC (where)
        model.token_emb.weight[stoi['e'], 21] = 350.0  # NUM (when)
        
        model.token_emb.weight[stoi['i'], 18] = 300.0  # ENTY (which)
        model.token_emb.weight[stoi['i'], 19] = 250.0  # HUM
        
        # === ATTENTION ===
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Head 0: attend to pos 0 (uses dims 0-3 for Q/K)
        for i in range(dh):
            block0.attn.W_q.weight[i, 0:4] = 1.0 / 4.0
            block0.attn.W_k.weight[i, 0:4] = 1.0 / 4.0
        
        # Head 1: attend to pos 1 (uses dims 4-7 for Q/K)
        for i in range(dh, 2*dh):
            block0.attn.W_q.weight[i, 4:8] = 1.0 / 4.0
            block0.attn.W_k.weight[i, 4:8] = 1.0 / 4.0
        
        # Head 2: attend to pos 2 (uses dims 8-11 for Q/K)
        for i in range(2*dh, 3*dh):
            block0.attn.W_q.weight[i, 8:12] = 1.0 / 4.0
            block0.attn.W_k.weight[i, 8:12] = 1.0 / 4.0
        
        # Head 3: attend to pos 3 (uses dims 12-15 for Q/K)
        for i in range(3*dh, 4*dh):
            block0.attn.W_q.weight[i, 12:16] = 1.0 / 4.0
            block0.attn.W_k.weight[i, 12:16] = 1.0 / 4.0
        
        # V passes through category signals (dims 16-21)
        for i in range(d):
            block0.attn.W_v.weight[i, i] = 1.0
        
        # O passes through
        for i in range(d):
            block0.attn.W_o.weight[i, i] = 1.0
        
        # MLP: combine signals
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        d_ff = block0.mlp.fc1.weight.shape[0]
        
        # Pass through category signals
        for i in range(6):
            block0.mlp.fc1.weight[i, 16 + i] = 2.0
            block0.mlp.fc2.weight[16 + i, i] = 2.0
        
        # Other blocks: identity
        for block in model.blocks[1:]:
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            for i in range(d):
                block.attn.W_q.weight[i, i] = 0.01
                block.attn.W_k.weight[i, i] = 0.01
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0
            
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            for i in range(min(d, d_ff)):
                block.mlp.fc1.weight[i, i] = 1.0
                block.mlp.fc2.weight[i, i] = 1.0
        
        # === OUTPUT HEAD ===
        model.head.weight.zero_()
        
        # Map category signals (dims 16-21) to outputs (tokens 30-35)
        for cat in range(6):
            model.head.weight[30 + cat, 16 + cat] = 30.0
    
    return


model_shorthand_name = "exp10_multihead_position"
model_description = "Multi-head attention: each head attends to different position (0-3), combine signals in MLP."


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
