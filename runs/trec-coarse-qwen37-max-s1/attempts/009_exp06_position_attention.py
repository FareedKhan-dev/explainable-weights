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
    Make attention strongly attend to position 0 from position 81.
    Encode question word category in token at position 0.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === POSITION EMBEDDINGS ===
        # Create strong attention pattern: position 81 attends to position 0
        # Use dim 0 for query signal, dim 1 for key signal
        
        # Position 0: strong key signal (I have important info)
        model.pos_emb.weight[0, 1] = 20.0
        
        # Position 81: strong query signal (I want important info)
        model.pos_emb.weight[81, 0] = 20.0
        
        # Other positions: weak signals
        for p in range(1, 81):
            model.pos_emb.weight[p, 0] = 1.0
            model.pos_emb.weight[p, 1] = 1.0
        
        # === TOKEN EMBEDDINGS ===
        # At position 0, encode the first character of the question
        # Question words start with: w (who/what/where/when/which), h (how)
        # Second character: h (for wh-words), o (for how)
        
        # Use dims 2-7 for category signals
        # Dim 2: ABBR, 3: DESC, 4: ENTY, 5: HUM, 6: LOC, 7: NUM
        
        # First character signals
        # 'w' -> could be who/what/where/when/which
        model.token_emb.weight[stoi['w'], 5] = 10.0  # HUM (who)
        model.token_emb.weight[stoi['w'], 6] = 10.0  # LOC (where)
        model.token_emb.weight[stoi['w'], 7] = 10.0  # NUM (when)
        model.token_emb.weight[stoi['w'], 3] = 6.0   # DESC (what)
        model.token_emb.weight[stoi['w'], 4] = 6.0   # ENTY (which)
        
        # 'h' -> how
        model.token_emb.weight[stoi['h'], 7] = 15.0  # NUM (how)
        model.token_emb.weight[stoi['h'], 3] = 8.0   # DESC (how)
        
        # === ATTENTION ===
        # Make attention use position signals
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Head 0: Q projects dim 0 (query), K projects dim 1 (key)
        # This creates high attention score for (pos 81, pos 0)
        block0.attn.W_q.weight[0, 0] = 1.0  # Q uses dim 0
        block0.attn.W_k.weight[0, 1] = 1.0  # K uses dim 1
        
        # V passes through all dims
        for i in range(d):
            block0.attn.W_v.weight[i, i] = 1.0
        
        # O passes through
        for i in range(d):
            block0.attn.W_o.weight[i, i] = 1.0
        
        # MLP: identity
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        d_ff = block0.mlp.fc1.weight.shape[0]
        for i in range(min(d, d_ff)):
            block0.mlp.fc1.weight[i, i] = 1.0
            block0.mlp.fc2.weight[i, i] = 1.0
        
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
        
        # Map category signals (dims 2-7) to outputs (tokens 30-35)
        for cat in range(6):
            model.head.weight[30 + cat, 2 + cat] = 10.0
    
    return


model_shorthand_name = "exp06_position_attention"
model_description = "Strong position-based attention from pos 81 to pos 0, question word signals in first token."


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
