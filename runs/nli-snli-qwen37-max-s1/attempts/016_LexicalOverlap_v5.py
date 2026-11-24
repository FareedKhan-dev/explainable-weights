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
        n_heads: int = 8,
        n_layers: int = 4,
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
    """Lexical overlap: count how many hypothesis words appear in premise."""
    torch.manual_seed(42)
    
    with torch.no_grad():
        d_model = model.d_model
        n_heads = model.n_heads
        dh = d_model // n_heads
        
        # Initialize all to zero (but LayerNorm weights to 1.0!)
        for name, param in model.named_parameters():
            param.zero_()
        
        # Fix LayerNorm weights - they need to be 1.0 (identity), not 0.0
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        
        # === EMBEDDINGS ===
        # Token embeddings: character identity in first 26 dims
        token_emb = torch.zeros(model.vocab_size, d_model)
        for i in range(26):
            token_emb[i, i] = 5.0  # Large values to dominate LayerNorm
        token_emb[26, 26] = 5.0  # space - IMPORTANT for word boundaries
        token_emb[28, 28] = 5.0  # underscore (padding)
        model.token_emb.weight.copy_(token_emb)
        
        # Position embeddings: encode region
        pos_emb = torch.zeros(model.max_seq_len, d_model)
        for i in range(model.max_seq_len):
            if i < 60:
                pos_emb[i, 30] = 10.0  # Premise marker (strong signal)
            elif i == 60:
                pos_emb[i, 35] = 10.0  # Pipe separator
            elif i < 121:
                pos_emb[i, 32] = 10.0  # Hypothesis marker (strong signal)
            elif i == 121:
                pos_emb[i, 34] = 10.0  # Equals/prediction marker
        model.pos_emb.weight.copy_(pos_emb)
        
        # === LAYER 1: Match hypothesis words to premise ===
        # Strategy: Each hypothesis position attends to ALL premise positions
        # with matching character. This builds a "match score" for each position.
        
        block0 = model.blocks[0]
        attn0 = block0.attn
        
        W_q = torch.zeros(d_model, d_model)
        W_k = torch.zeros(d_model, d_model)
        W_v = torch.zeros(d_model, d_model)
        W_o = torch.zeros(d_model, d_model)
        
        # Head 0: Character-level matching
        # Query from hypothesis: "What character am I?"
        # Key from premise: "I have this character"
        # Attention weight = character similarity
        
        for c in range(min(26, dh)):
            # Query: activate on hypothesis character
            W_q[c, 32] = 3.0  # Must be in hypothesis region
            W_q[c, c] = 2.0   # Match on character identity
            
            # Key: activate on premise character
            W_k[c, 30] = 3.0  # Must be in premise region
            W_k[c, c] = 2.0   # Match on character identity
            
            # Value: pass through character identity
            W_v[c, c] = 1.0
        
        # Output: write match signals to dims 40-65
        for c in range(min(26, dh)):
            W_o[40 + c, c] = 1.0
        
        attn0.W_q.weight.copy_(W_q)
        attn0.W_k.weight.copy_(W_k)
        attn0.W_v.weight.copy_(W_v)
        attn0.W_o.weight.copy_(W_o)
        
        # === LAYER 1 MLP: Threshold matches ===
        mlp0 = block0.mlp
        W1 = torch.zeros(model.d_ff, d_model)
        b1 = torch.zeros(model.d_ff)
        W2 = torch.zeros(d_model, model.d_ff)
        b2 = torch.zeros(d_model)
        
        # For each character, check if it matched (signal > 0.1)
        for c in range(min(26, model.d_ff)):
            W1[c, 40 + c] = 20.0  # Amplify match signal
            b1[c] = -2.0  # Low threshold to detect any match
        
        # Sum up all character matches into dim 70
        for c in range(min(26, model.d_ff)):
            W2[70, c] = 1.0
        
        mlp0.fc1.weight.copy_(W1)
        mlp0.fc1.bias.copy_(b1)
        mlp0.fc2.weight.copy_(W2)
        mlp0.fc2.bias.copy_(b2)
        
        # === LAYER 2: Aggregate matches at equals position ===
        block1 = model.blocks[1]
        attn1 = block1.attn
        
        W_q1 = torch.zeros(d_model, d_model)
        W_k1 = torch.zeros(d_model, d_model)
        W_v1 = torch.zeros(d_model, d_model)
        W_o1 = torch.zeros(d_model, d_model)
        
        # Head 0: Equals position attends to hypothesis to collect match counts
        for d in range(dh):
            W_q1[d, 34] = 5.0  # Query from equals position
            W_k1[d, 32] = 5.0  # Key from hypothesis region
            W_v1[d, 70] = 1.0  # Value: match count from layer 1
        
        # Output: write total match count to dim 80
        for d in range(dh):
            W_o1[80, d] = 1.0 / dh  # Average across heads
        
        attn1.W_q.weight.copy_(W_q1)
        attn1.W_k.weight.copy_(W_k1)
        attn1.W_v.weight.copy_(W_v1)
        attn1.W_o.weight.copy_(W_o1)
        
        # === LAYER 2 MLP: Convert match count to prediction ===
        mlp1 = block1.mlp
        W1_1 = torch.zeros(model.d_ff, d_model)
        b1_1 = torch.zeros(model.d_ff)
        W2_1 = torch.zeros(d_model, model.d_ff)
        b2_1 = torch.zeros(d_model)
        
        # High overlap (>10 matches) -> entailment
        W1_1[0, 80] = 2.0
        b1_1[0] = -20.0  # Threshold at 10
        
        # Low overlap (<5 matches) -> contradiction
        W1_1[1, 80] = -2.0
        b1_1[1] = 10.0  # Threshold at 5
        
        # Output signals
        W2_1[90, 0] = 1.0  # Entailment signal
        W2_1[91, 1] = 1.0  # Contradiction signal
        
        mlp1.fc1.weight.copy_(W1_1)
        mlp1.fc1.bias.copy_(b1_1)
        mlp1.fc2.weight.copy_(W2_1)
        mlp1.fc2.bias.copy_(b2_1)
        
        # === FINAL HEAD: Map to predictions ===
        W_head = torch.zeros(model.vocab_size, d_model)
        b_head = torch.zeros(model.vocab_size)
        
        # Default: neutral (class 1)
        b_head.fill_(-20.0)
        b_head[32] = 5.0  # '1'
        
        # High entailment signal -> predict 0
        W_head[31, 90] = 15.0  # '0'
        W_head[32, 90] = -10.0
        
        # High contradiction signal -> predict 2
        W_head[33, 91] = 15.0  # '2'
        W_head[32, 91] = -10.0
        
        model.head.weight.copy_(W_head)
        model.head.bias.copy_(b_head)
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "LexicalOverlap_v5"
model_description = "Count matching characters between premise/hypothesis, high overlap=entailment, low=contradiction."


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
