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
    Radically different approach: Use position embeddings to create a "routing" signal.
    Position 81's token embedding encodes category predictions based on what it sees.
    Attention aggregates category signals from positions 0-2.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Positions 0-2: strong "I have information" signal
        for pos_idx in range(3):
            model.pos_emb.weight[pos_idx, 6] = 300.0
        
        # Position 81: strong "I want information" signal
        model.pos_emb.weight[81, 6] = 300.0
        
        # All other positions: no signal (dim 6 = 0)
        
        # === TOKEN EMBEDDINGS ===
        # Encode VERY strong category signals in dims 0-5
        # These need to survive LayerNorm and attention averaging
        
        # Position 0: First character gives initial bias
        model.token_emb.weight[stoi['w'], 1] = 150.0  # DESC (what/who/where/when/why)
        model.token_emb.weight[stoi['h'], 5] = 150.0  # NUM (how)
        
        # Position 1: Second character (usually 'h' or vowel)
        # Not very discriminative, skip
        
        # Position 2: Third character - THE KEY DISCRIMINATOR
        # 'o': who (HUM) or how (NUM)
        model.token_emb.weight[stoi['o'], 3] = 300.0  # HUM (very strong)
        model.token_emb.weight[stoi['o'], 5] = 200.0  # NUM
        
        # 'a': what (DESC/ENTY)
        model.token_emb.weight[stoi['a'], 1] = 280.0  # DESC
        model.token_emb.weight[stoi['a'], 2] = 220.0  # ENTY
        
        # 'e': where (LOC) or when (NUM)
        model.token_emb.weight[stoi['e'], 4] = 300.0  # LOC
        model.token_emb.weight[stoi['e'], 5] = 250.0  # NUM
        
        # 'i': which (ENTY)
        model.token_emb.weight[stoi['i'], 2] = 300.0  # ENTY
        model.token_emb.weight[stoi['i'], 3] = 150.0  # HUM
        
        # 'y': why (DESC)
        model.token_emb.weight[stoi['y'], 1] = 400.0  # DESC (very strong)
        
        # Position 3: Fourth character - disambiguator
        # 'r': where → boost LOC
        model.token_emb.weight[stoi['r'], 4] = 200.0  # LOC
        
        # 'n': when → boost NUM
        model.token_emb.weight[stoi['n'], 5] = 200.0  # NUM
        
        # 't': what → slight ENTY boost
        model.token_emb.weight[stoi['t'], 2] = 150.0  # ENTY
        
        # === ATTENTION (Block 0) ===
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Use dim 6 for attention control
        for i in range(d):
            block0.attn.W_q.weight[i, 6] = 1.5
            block0.attn.W_k.weight[i, 6] = 1.5
        
        # V passes through dims 0-5 (category signals)
        for i in range(6):
            block0.attn.W_v.weight[i, i] = 1.0
        
        # O passes through
        for i in range(d):
            block0.attn.W_o.weight[i, i] = 1.0
        
        # MLP: amplify category signals strongly
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        d_ff = block0.mlp.fc1.weight.shape[0]
        
        # Amplify dims 0-5 (category signals)
        for i in range(6):
            block0.mlp.fc1.weight[i, i] = 10.0  # Very strong amplification
            block0.mlp.fc2.weight[i, i] = 10.0
        
        # Other blocks: pass through (zero weights = identity through residual)
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
        
        # Map category signals (dims 0-5) to output tokens (30-35)
        for cat in range(6):
            model.head.weight[30 + cat, cat] = 100.0  # Very strong output weights
    
    return


model_shorthand_name = "exp16_routing_signals"
model_description = "Use routing signals: position 81 queries 0-2, strong category signals (150-400) in dims 0-5."


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
