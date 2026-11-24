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
    Strategy: Use position embeddings to create attention "address" system.
    Position 81 queries for positions 0-3 using dedicated dimensions.
    Category signals live in SEPARATE dimensions from position signals.
    This avoids the LayerNorm washing-out problem.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        vocab = model.vocab_size
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        # Output tokens: 30=ABBR, 31=DESC, 32=ENTY, 33=HUM, 34=LOC, 35=NUM
        
        # === DIMENSION ALLOCATION ===
        # Dims 0-7: Category signals from token embeddings
        # Dims 8-15: Position-based attention control
        # Dims 16-63: Unused (zero)
        
        # === POSITION EMBEDDINGS ===
        # Use dims 8-15 for attention control
        # Position 81 (output): strong query signal in dim 8
        model.pos_emb.weight[81, 8] = 100.0
        
        # Positions 0-3: strong "key" signal in dim 8 (these are the informative positions)
        for pos_idx in range(4):
            model.pos_emb.weight[pos_idx, 8] = 100.0
        
        # All other positions: no signal (dim 8 stays at 0)
        # This means attention from pos 81 will strongly prefer positions 0-3
        
        # === TOKEN EMBEDDINGS ===
        # Encode question word patterns in dims 0-5 (one per category)
        # The key insight: 3rd character (pos 2) is most discriminative
        
        # Character 'w' at position 0: common to what/who/where/when/why
        # Give it a slight bias toward DESC (most common)
        model.token_emb.weight[stoi['w'], 1] = 20.0  # DESC bias
        
        # Character 'h' at position 0: "how" questions
        model.token_emb.weight[stoi['h'], 5] = 30.0  # NUM (how many/much)
        model.token_emb.weight[stoi['h'], 1] = 10.0  # DESC (how did)
        
        # Position 1 characters:
        model.token_emb.weight[stoi['h'], 0] = 0.0  # 'h' after 'w': wh-questions
        
        # Position 2: the critical discriminator
        # 'o' at position 2: who (HUM) or how (NUM)
        model.token_emb.weight[stoi['o'], 3] = 50.0  # HUM (who)
        model.token_emb.weight[stoi['o'], 5] = 30.0  # NUM (how)
        
        # 'a' at position 2: what (DESC/ENTY)
        model.token_emb.weight[stoi['a'], 1] = 40.0  # DESC
        model.token_emb.weight[stoi['a'], 2] = 30.0  # ENTY
        
        # 'e' at position 2: where (LOC) or when (NUM)
        model.token_emb.weight[stoi['e'], 4] = 50.0  # LOC (where)
        model.token_emb.weight[stoi['e'], 5] = 40.0  # NUM (when)
        
        # 'i' at position 2: which (ENTY)
        model.token_emb.weight[stoi['i'], 2] = 50.0  # ENTY
        
        # 'y' at position 2: why (DESC)
        model.token_emb.weight[stoi['y'], 1] = 60.0  # DESC
        
        # Position 3: disambiguator
        # 'r' at position 3: where → boost LOC
        model.token_emb.weight[stoi['r'], 4] = 40.0  # LOC
        
        # 'n' at position 3: when → boost NUM
        model.token_emb.weight[stoi['n'], 5] = 40.0  # NUM
        
        # 't' at position 3: what → slight ENTY boost
        model.token_emb.weight[stoi['t'], 2] = 20.0  # ENTY
        
        # === ATTENTION (Block 0) ===
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Q and K project onto dim 8 for attention scoring
        # This means attention score depends on dim 8 (position signal)
        for i in range(d):
            block0.attn.W_q.weight[i, 8] = 1.0  # Query reads dim 8
            block0.attn.W_k.weight[i, 8] = 1.0  # Key reads dim 8
        
        # V passes through dims 0-7 (category signals) but NOT dim 8 (position)
        for i in range(8):
            block0.attn.W_v.weight[i, i] = 1.0  # Pass category signals
        
        # O passes through
        for i in range(d):
            block0.attn.W_o.weight[i, i] = 1.0
        
        # MLP: amplify category signals
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        d_ff = block0.mlp.fc1.weight.shape[0]
        
        # Amplify dims 0-5 (category signals)
        for i in range(6):
            block0.mlp.fc1.weight[i, i] = 5.0
            block0.mlp.fc2.weight[i, i] = 5.0
        
        # Other blocks: pass through
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
            model.head.weight[30 + cat, cat] = 50.0
    
    return


model_shorthand_name = "exp14_separate_dims"
model_description = "Separate position (dim 8) and category (dims 0-5) signals to avoid LayerNorm interference."


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
