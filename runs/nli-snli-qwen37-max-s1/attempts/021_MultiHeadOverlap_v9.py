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
        n_heads: int = 2,
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
    """Better lexical overlap with word boundaries and position awareness."""
    torch.manual_seed(42)
    
    with torch.no_grad():
        d_model = model.d_model
        n_heads = model.n_heads
        dh = d_model // n_heads
        
        # Initialize all to zero (but LayerNorm weights to 1.0!)
        for name, param in model.named_parameters():
            param.zero_()
        
        # Fix LayerNorm weights
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        
        # === EMBEDDINGS ===
        # Token embeddings: letter identity + space marker
        token_emb = torch.zeros(model.vocab_size, d_model)
        for i in range(26):
            token_emb[i, i] = 10.0  # Strong letter signals
        # Space (token 26) gets special marker
        token_emb[26, 26] = 10.0
        # Separator "|" (token 27) and "=" (token 28)
        token_emb[27, 27] = 10.0
        token_emb[28, 28] = 10.0
        model.token_emb.weight.copy_(token_emb)
        
        # Position embeddings: mark regions
        pos_emb = torch.zeros(model.max_seq_len, d_model)
        for i in range(model.max_seq_len):
            if i < 60:
                pos_emb[i, 30] = 10.0  # Premise
            elif i == 60:
                pos_emb[i, 31] = 10.0  # Separator "|"
            elif i < 101:
                pos_emb[i, 32] = 10.0  # Hypothesis
            elif i == 101:
                pos_emb[i, 33] = 10.0  # Equals
        model.pos_emb.weight.copy_(pos_emb)
        
        # === LAYER 1: Multi-head attention ===
        block0 = model.blocks[0]
        attn0 = block0.attn
        
        W_q = torch.zeros(d_model, d_model)
        W_k = torch.zeros(d_model, d_model)
        W_v = torch.zeros(d_model, d_model)
        W_o = torch.zeros(d_model, d_model)
        
        # Head 0: Equals attends to hypothesis to collect word-start letters
        for c in range(min(26, dh)):
            W_q[c, 33] = 5.0   # Query from equals
            W_k[c, 32] = 5.0   # Key from hypothesis positions
            W_v[c, c] = 1.0    # Value: letter identity
        for c in range(min(26, dh)):
            W_o[40 + c, c] = 1.0  # Write to dims 40-65
        
        # Head 1: Equals attends to premise to collect word-start letters
        head_offset = dh
        for c in range(min(26, dh)):
            W_q[head_offset + c, 33] = 5.0
            W_k[head_offset + c, 30] = 5.0
            W_v[head_offset + c, c] = 1.0
        for c in range(min(26, dh)):
            W_o[70 + c, head_offset + c] = 1.0  # Write to dims 70-95
        
        attn0.W_q.weight.copy_(W_q)
        attn0.W_k.weight.copy_(W_k)
        attn0.W_v.weight.copy_(W_v)
        attn0.W_o.weight.copy_(W_o)
        
        # === LAYER 1 MLP: Compare hypothesis to premise ===
        mlp0 = block0.mlp
        W1 = torch.zeros(model.d_ff, d_model)
        b1 = torch.zeros(model.d_ff)
        W2 = torch.zeros(d_model, model.d_ff)
        b2 = torch.zeros(d_model)
        
        # For each letter, compute overlap score
        for c in range(min(26, model.d_ff // 4)):
            # Feature: letter c is in both hypothesis and premise (match)
            W1[c, 40 + c] = 1.0      # Hypothesis has letter
            W1[c, 70 + c] = 1.0      # Premise has letter
            b1[c] = -0.5             # Need both
            
            # Feature: letter c is in hypothesis but not premise (mismatch)
            W1[26 + c, 40 + c] = 1.0
            W1[26 + c, 70 + c] = -2.0
            b1[26 + c] = 0.5
        
        # Aggregate: high match score -> entailment, high mismatch -> contradiction
        for c in range(min(26, model.d_ff // 4)):
            W2[100, c] = 1.0        # Match score
            W2[101, 26 + c] = 1.0   # Mismatch score
        
        mlp0.fc1.weight.copy_(W1)
        mlp0.fc1.bias.copy_(b1)
        mlp0.fc2.weight.copy_(W2)
        mlp0.fc2.bias.copy_(b2)
        
        # === LAYER 2: Refine decision ===
        block1 = model.blocks[1]
        attn1 = block1.attn
        
        W_q1 = torch.zeros(d_model, d_model)
        W_k1 = torch.zeros(d_model, d_model)
        W_v1 = torch.zeros(d_model, d_model)
        W_o1 = torch.zeros(d_model, d_model)
        
        # Just pass through the scores
        for i in range(min(d_model, dh)):
            W_q1[i, 100] = 1.0
            W_k1[i, 100] = 1.0
            W_v1[i, i] = 1.0
            W_o1[i, i] = 1.0
        
        attn1.W_q.weight.copy_(W_q1)
        attn1.W_k.weight.copy_(W_k1)
        attn1.W_v.weight.copy_(W_v1)
        attn1.W_o.weight.copy_(W_o1)
        
        # === LAYER 2 MLP: Final decision ===
        mlp1 = block1.mlp
        W1_l2 = torch.zeros(model.d_ff, d_model)
        b1_l2 = torch.zeros(model.d_ff)
        W2_l2 = torch.zeros(d_model, model.d_ff)
        b2_l2 = torch.zeros(d_model)
        
        # Decision logic: high match -> entailment, high mismatch -> contradiction, else neutral
        W1_l2[0, 100] = 2.0   # Match score
        W1_l2[0, 101] = -1.0  # Minus mismatch
        b1_l2[0] = -0.5
        
        W1_l2[1, 101] = 2.0   # Mismatch score
        W1_l2[1, 100] = -1.0  # Minus match
        b1_l2[1] = -0.5
        
        # Output: adjust final representation
        W2_l2[100, 0] = 1.0   # Entailment signal
        W2_l2[101, 1] = 1.0   # Contradiction signal
        
        mlp1.fc1.weight.copy_(W1_l2)
        mlp1.fc1.bias.copy_(b1_l2)
        mlp1.fc2.weight.copy_(W2_l2)
        mlp1.fc2.bias.copy_(b2_l2)
        
        # === FINAL HEAD: Map to predictions ===
        W_head = torch.zeros(model.vocab_size, d_model)
        b_head = torch.zeros(model.vocab_size)
        
        # Default: neutral (class 1)
        b_head.fill_(-20.0)
        b_head[32] = 0.0  # '1' = neutral
        
        # High match score -> entailment (class 0)
        W_head[31, 100] = 10.0   # '0'
        W_head[32, 100] = -5.0
        
        # High mismatch score -> contradiction (class 2)
        W_head[33, 101] = 10.0   # '2'
        W_head[32, 101] = -5.0
        
        model.head.weight.copy_(W_head)
        model.head.bias.copy_(b_head)
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "MultiHeadOverlap_v9"
model_description = "Multi-head attention: hypothesis and premise word-start letters compared. Match=entailment, mismatch=contradiction."


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
