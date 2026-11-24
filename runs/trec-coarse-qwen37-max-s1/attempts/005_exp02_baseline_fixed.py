"""
Interpretable transformer for TREC coarse question classification.
Task: classify question into category 0-5 based on text content.
Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
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


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """
    Strategy: Use token embeddings to encode 'category signals' for each character.
    Use attention to aggregate these signals to the output position.
    Use MLP and output head to classify.
    
    Key insight: The first word of the question strongly predicts the category.
    We encode bigram features in token embeddings, then use attention to pool
    these features to the last position, then classify.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        vocab = task.vocab
        d = model.d_model  # 64
        vocab_size = model.vocab_size  # 36
        
        # ---- Token embeddings ----
        # Each character gets a one-hot-like embedding in its own dimension
        # Plus category signals based on which characters appear in question words
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # Assign each letter its own dimension (26 letters in first 26 dims)
        for i in range(26):
            model.token_emb.weight[i, i] = 1.0
        
        # Space gets dim 26, apostrophe dim 27, underscore dim 28, = dim 29
        for i in range(26, 30):
            model.token_emb.weight[i, i] = 1.0
        
        # Answer tokens (0-5) at indices 30-35
        for i in range(30, 36):
            model.token_emb.weight[i, i] = 0.0
        
        # ---- Position embeddings ----
        # Make position 0 special (it's always the start of the question)
        # Position 80 is the = sign (prompt_len - 1)
        # Position 81 is where we predict (seq_len - 1)
        prompt_len = task.prompt_len  # 81
        answer_pos = prompt_len  # position 81 is where answer goes (but seq is 82 long, so positions 0-81)
        
        # Actually the prompt is 81 chars, answer is 1 char, seq_len is 82
        # So positions are 0..80 for prompt, 81 for answer
        # When generating, at position 81 (the 82nd token), we need to predict
        
        # Give each position a small positional encoding
        for p in range(model.max_seq_len):
            for i in range(d):
                if i % 2 == 0:
                    model.pos_emb.weight[p, i] = math.sin(p / (10000 ** (i / d)))
                else:
                    model.pos_emb.weight[p, i] = math.cos(p / (10000 ** ((i - 1) / d)))
        
        # Make the last prediction position special with a strong signal
        for i in range(d):
            model.pos_emb.weight[81, i] = 0.0  # Reset
        model.pos_emb.weight[81, 0:6] = 5.0  # Strong signal at prediction position
        
        # ---- Attention layers ----
        # Layer 1: Attend from last position to all positions (uniform attention)
        # This pools information from the entire sequence
        block0 = model.blocks[0]
        
        # Identity for Q, K, V, O so that attention passes through information
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Make each head do identity
        for i in range(d):
            block0.attn.W_q.weight[i, i] = 1.0
            block0.attn.W_k.weight[i, i] = 1.0
            block0.attn.W_v.weight[i, i] = 1.0
            block0.attn.W_o.weight[i, i] = 1.0 / model.n_heads
        
        # MLP for block 0: identity pass-through
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        d_ff = block0.mlp.fc1.weight.shape[0]
        for i in range(min(d, d_ff)):
            block0.mlp.fc1.weight[i, i] = 1.0
            block0.mlp.fc2.weight[i, i] = 1.0
        
        # Block 1 and 2: same identity-like
        for block in model.blocks[1:]:
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            for i in range(d):
                block.attn.W_q.weight[i, i] = 1.0
                block.attn.W_k.weight[i, i] = 1.0
                block.attn.W_v.weight[i, i] = 1.0
                block.attn.W_o.weight[i, i] = 1.0 / model.n_heads
            
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            for i in range(min(d, d_ff)):
                block.mlp.fc1.weight[i, i] = 1.0
                block.mlp.fc2.weight[i, i] = 1.0
        
        # LayerNorm: keep default (weight=1, bias=0)
        
        # ---- Output head ----
        # Map from hidden state at last position to category
        # After attention pooling, the last position should have averaged info
        model.head.weight.zero_()
        
        # For now, just output a fixed category to test
        # Map to category '1' (DESC) which is common
        model.head.weight[31, :] = 1.0  # Always predict '1'
    
    return


model_shorthand_name = "exp02_baseline_fixed"
model_description = "Baseline: identity transformer with fixed output prediction (always predict category 1)."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

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
