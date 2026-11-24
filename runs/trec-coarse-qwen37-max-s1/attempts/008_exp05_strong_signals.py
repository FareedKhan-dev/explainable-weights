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
    Simple approach: encode strong question word signals directly in token embeddings.
    Use uniform attention to aggregate to last position.
    Output head classifies based on aggregated signals.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === TOKEN EMBEDDINGS ===
        # Encode very strong question word signals
        # Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        
        # Question words map to categories with high confidence:
        # who -> HUM (3) with 90% confidence
        # where -> LOC (4) with 95% confidence  
        # when -> NUM (5) with 95% confidence
        # how -> NUM (5) with 80% confidence
        # what -> DESC (1) with 40%, ENTY (2) with 30%, HUM (3) with 15%, LOC (4) with 10%, NUM (5) with 5%
        # which -> ENTY (2) with 60%, HUM (3) with 30%, LOC (4) with 10%
        
        # Use dimensions 0-5 for category signals (one per category)
        # Make signals VERY strong so they survive averaging over 82 positions
        
        # 'w' character - appears in who/what/where/when/which
        model.token_emb.weight[stoi['w'], 3] = 5.0  # HUM
        model.token_emb.weight[stoi['w'], 4] = 5.0  # LOC
        model.token_emb.weight[stoi['w'], 5] = 5.0  # NUM
        model.token_emb.weight[stoi['w'], 1] = 3.0  # DESC
        model.token_emb.weight[stoi['w'], 2] = 3.0  # ENTY
        
        # 'h' character - appears in who/what/where/when/how/which
        model.token_emb.weight[stoi['h'], 3] = 6.0  # HUM
        model.token_emb.weight[stoi['h'], 4] = 6.0  # LOC
        model.token_emb.weight[stoi['h'], 5] = 6.0  # NUM
        model.token_emb.weight[stoi['h'], 1] = 3.5  # DESC
        model.token_emb.weight[stoi['h'], 2] = 3.5  # ENTY
        
        # 'o' character - who, how
        model.token_emb.weight[stoi['o'], 3] = 8.0  # HUM (from who)
        model.token_emb.weight[stoi['o'], 5] = 6.0  # NUM (from how)
        
        # 'a' character - what
        model.token_emb.weight[stoi['a'], 1] = 5.0  # DESC
        model.token_emb.weight[stoi['a'], 2] = 4.0  # ENTY
        
        # 'e' character - where, when
        model.token_emb.weight[stoi['e'], 4] = 8.0  # LOC (from where)
        model.token_emb.weight[stoi['e'], 5] = 8.0  # NUM (from when)
        
        # 'r' character - where
        model.token_emb.weight[stoi['r'], 4] = 10.0  # LOC
        
        # 'n' character - when
        model.token_emb.weight[stoi['n'], 5] = 10.0  # NUM
        
        # 'i' character - which
        model.token_emb.weight[stoi['i'], 2] = 7.0  # ENTY
        model.token_emb.weight[stoi['i'], 3] = 5.0  # HUM
        
        # 'c' character - which
        model.token_emb.weight[stoi['c'], 2] = 7.0  # ENTY
        model.token_emb.weight[stoi['c'], 3] = 5.0  # HUM
        
        # === POSITION EMBEDDINGS ===
        # Make them neutral (don't interfere)
        # All zeros already
        
        # === ATTENTION LAYERS ===
        # Use simple uniform attention to average all positions
        for block in model.blocks:
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            
            # Small identity to allow information flow
            for i in range(d):
                block.attn.W_q.weight[i, i] = 0.1
                block.attn.W_k.weight[i, i] = 0.1
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0 / model.n_heads
            
            # MLP: pass through with amplification
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            
            d_ff = block.mlp.fc1.weight.shape[0]
            # Amplify category signals (dims 0-5)
            for i in range(6):
                block.mlp.fc1.weight[i, i] = 2.0
                block.mlp.fc2.weight[i, i] = 2.0
        
        # === OUTPUT HEAD ===
        # Map category signals to output tokens
        model.head.weight.zero_()
        
        # Output tokens 30-35 correspond to '0'-'5'
        for cat in range(6):
            model.head.weight[30 + cat, cat] = 10.0
    
    return


model_shorthand_name = "exp05_strong_signals"
model_description = "Very strong question word signals in token embeddings, uniform attention, amplified through MLP layers."


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
