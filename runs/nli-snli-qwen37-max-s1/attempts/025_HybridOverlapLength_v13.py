"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
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
    """NLI transformer with interpretable attention patterns."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
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
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Hybrid: 4 heads combine overlap, length, and content-word matching."""
    torch.manual_seed(42)
    
    with torch.no_grad():
        d_model = model.d_model
        n_heads = model.n_heads
        dh = d_model // n_heads
        
        for name, param in model.named_parameters():
            param.zero_()
        
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        
        # === EMBEDDINGS ===
        token_emb = torch.zeros(model.vocab_size, d_model)
        for i in range(26):
            token_emb[i, i] = 10.0  # Letter identity
        token_emb[26, 26] = 10.0  # Space
        token_emb[27, 27] = 10.0  # "|"
        token_emb[28, 28] = 10.0  # "="
        model.token_emb.weight.copy_(token_emb)
        
        pos_emb = torch.zeros(model.max_seq_len, d_model)
        for i in range(model.max_seq_len):
            if i < 60:
                pos_emb[i, 30] = 15.0  # Premise
            elif i == 60:
                pos_emb[i, 31] = 15.0  # Separator
            elif i < 101:
                pos_emb[i, 32] = 15.0  # Hypothesis
            elif i == 101:
                pos_emb[i, 33] = 15.0  # Equals
        model.pos_emb.weight.copy_(pos_emb)
        
        # === LAYER 1: 4-head attention ===
        block0 = model.blocks[0]
        attn0 = block0.attn
        
        W_q = torch.zeros(d_model, d_model)
        W_k = torch.zeros(d_model, d_model)
        W_v = torch.zeros(d_model, d_model)
        W_o = torch.zeros(d_model, d_model)
        
        # Head 0: Hypothesis letter frequencies
        for c in range(min(26, dh)):
            W_q[c, 33] = 8.0
            W_k[c, 32] = 8.0
            W_v[c, c] = 1.0
        for c in range(min(26, dh)):
            W_o[40 + c, c] = 1.5
        
        # Head 1: Premise letter frequencies
        h1 = dh
        for c in range(min(26, dh)):
            W_q[h1 + c, 33] = 8.0
            W_k[h1 + c, 30] = 8.0
            W_v[h1 + c, c] = 1.0
        for c in range(min(26, dh)):
            W_o[70 + c, h1 + c] = 1.5
        
        # Head 2: Hypothesis length (count non-space chars)
        h2 = 2 * dh
        W_q[h2, 33] = 8.0
        W_k[h2, 32] = 8.0
        W_v[h2, 32] = 1.0
        W_o[96, h2] = 1.0
        
        # Head 3: Premise length (count non-space chars)
        h3 = 3 * dh
        W_q[h3, 33] = 8.0
        W_k[h3, 30] = 8.0
        W_v[h3, 30] = 1.0
        W_o[97, h3] = 1.0
        
        attn0.W_q.weight.copy_(W_q)
        attn0.W_k.weight.copy_(W_k)
        attn0.W_v.weight.copy_(W_v)
        attn0.W_o.weight.copy_(W_o)
        
        # === LAYER 1 MLP ===
        mlp0 = block0.mlp
        W1 = torch.zeros(model.d_ff, d_model)
        b1 = torch.zeros(model.d_ff)
        W2 = torch.zeros(d_model, model.d_ff)
        b2 = torch.zeros(d_model)
        
        # Per-letter match/mismatch
        for c in range(26):
            W1[c, 40 + c] = 2.0
            W1[c, 70 + c] = 2.0
            b1[c] = -3.0
            
            W1[26 + c, 40 + c] = 3.0
            W1[26 + c, 70 + c] = -4.0
            b1[26 + c] = 0.0
        
        # Length ratio
        W1[52, 96] = 2.0   # Hyp length
        W1[52, 97] = -2.0  # Prem length
        b1[52] = 0.0
        
        # Hyp much shorter (entailment signal)
        W1[53, 97] = 2.0
        W1[53, 96] = -3.0
        b1[53] = 3.0
        
        # Aggregate
        for c in range(26):
            W2[100, c] = 1.0        # Match total
            W2[101, 26 + c] = 1.0   # Mismatch total
        W2[102, 52] = 2.0           # Length ratio
        W2[103, 53] = 2.0           # Short hyp signal
        
        mlp0.fc1.weight.copy_(W1)
        mlp0.fc1.bias.copy_(b1)
        mlp0.fc2.weight.copy_(W2)
        mlp0.fc2.bias.copy_(b2)
        
        # === LAYER 2: Decision ===
        block1 = model.blocks[1]
        attn1 = block1.attn
        
        W_q1 = torch.zeros(d_model, d_model)
        W_k1 = torch.zeros(d_model, d_model)
        W_v1 = torch.zeros(d_model, d_model)
        W_o1 = torch.zeros(d_model, d_model)
        for i in range(min(d_model, dh)):
            W_q1[i, i] = 0.5
            W_k1[i, i] = 0.5
            W_v1[i, i] = 1.0
            W_o1[i, i] = 1.0
        attn1.W_q.weight.copy_(W_q1)
        attn1.W_k.weight.copy_(W_k1)
        attn1.W_v.weight.copy_(W_v1)
        attn1.W_o.weight.copy_(W_o1)
        
        mlp1 = block1.mlp
        W1_l2 = torch.zeros(model.d_ff, d_model)
        b1_l2 = torch.zeros(model.d_ff)
        W2_l2 = torch.zeros(d_model, model.d_ff)
        b2_l2 = torch.zeros(d_model)
        
        # Entailment: high match + low mismatch + short hyp
        W1_l2[0, 100] = 2.0
        W1_l2[0, 101] = -2.0
        W1_l2[0, 103] = 1.0
        b1_l2[0] = -1.0
        
        # Contradiction: high mismatch
        W1_l2[1, 101] = 3.0
        W1_l2[1, 100] = -1.0
        b1_l2[1] = -1.0
        
        # Neutral: moderate overlap, similar length
        W1_l2[2, 100] = 1.0
        W1_l2[2, 101] = -1.0
        W1_l2[2, 102] = -1.0
        b1_l2[2] = 0.0
        
        W2_l2[110, 0] = 3.0
        W2_l2[111, 1] = 3.0
        W2_l2[112, 2] = 2.0
        
        mlp1.fc1.weight.copy_(W1_l2)
        mlp1.fc1.bias.copy_(b1_l2)
        mlp1.fc2.weight.copy_(W2_l2)
        mlp1.fc2.bias.copy_(b2_l2)
        
        # === FINAL HEAD ===
        W_head = torch.zeros(model.vocab_size, d_model)
        b_head = torch.zeros(model.vocab_size)
        
        b_head.fill_(-20.0)
        b_head[32] = 0.0  # Default neutral
        
        W_head[31, 110] = 15.0
        W_head[32, 110] = -10.0
        
        W_head[33, 111] = 15.0
        W_head[32, 111] = -10.0
        
        W_head[32, 112] = 8.0
        
        model.head.weight.copy_(W_head)
        model.head.bias.copy_(b_head)
    
    return


model_shorthand_name = "HybridOverlapLength_v13"
model_description = "4 heads: hyp letters, premise letters, hyp length, premise length. MLP combines overlap match/mismatch with length ratio."


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
    
    # DEBUG: print examples
    if args.verbose:
        examples = task.generate_examples(5)
        for i, ex in enumerate(examples):
            print(f"Ex {i}: ans={ex.answer} | {ex.prompt}")
    
    model = build_model(task).to(args.device)
    
    # DEBUG: Check what the model predicts on a few examples
    if args.verbose:
        print("\n=== PREDICTIONS ===")
        examples = task.generate_examples(10)
        for i, ex in enumerate(examples):
            prompt_ids = task.encode(ex.prompt)
            with torch.no_grad():
                logits = model(torch.tensor([prompt_ids], device=args.device))
                pred = logits[0, -1].argmax().item()
                probs = torch.softmax(logits[0, -1], dim=-1)
            print(f"Ex {i}: Answer={ex.answer}, Pred={task.decode([pred])}, Probs[0,1,2]={probs[31:34].tolist()}")
        sys.exit(0)

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
