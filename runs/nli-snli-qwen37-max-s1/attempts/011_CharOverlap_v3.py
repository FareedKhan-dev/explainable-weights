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
    """Compute lexical overlap between premise and hypothesis for NLI."""
    torch.manual_seed(42)
    
    with torch.no_grad():
        d_model = model.d_model
        n_heads = model.n_heads
        dh = d_model // n_heads
        n_layers = model.n_layers
        
        # Initialize all to zero
        for name, param in model.named_parameters():
            param.zero_()
        
        # === EMBEDDINGS ===
        # Token embeddings: first 26 dims = one-hot letter identity
        token_emb = torch.zeros(model.vocab_size, d_model)
        for i in range(26):
            token_emb[i, i] = 1.0
        # space at dim 26, underscore at dim 27
        token_emb[26, 26] = 1.0  # space
        token_emb[27, 27] = 1.0  # apostrophe
        token_emb[28, 28] = 1.0  # underscore
        token_emb[29, 29] = 1.0  # pipe
        token_emb[30, 30] = 1.0  # equals
        # Dims 31-33 reserved for answer tokens '0', '1', '2'
        token_emb[31, 31] = 1.0  # '0'
        token_emb[32, 32] = 1.0  # '1'
        token_emb[33, 33] = 1.0  # '2'
        model.token_emb.weight.copy_(token_emb)
        
        # Position embeddings: encode sentence region
        pos_emb = torch.zeros(model.max_seq_len, d_model)
        # Dim 34: premise flag (1 for positions 0-59)
        # Dim 35: hypothesis flag (1 for positions 61-120)
        # Dim 36: prediction flag (1 for position 121)
        for i in range(min(60, model.max_seq_len)):
            pos_emb[i, 34] = 1.0  # Premise
        if model.max_seq_len > 60:
            pos_emb[60, 34 + 1] = 1.0  # Pipe/separator - use different dim
        for i in range(61, min(121, model.max_seq_len)):
            pos_emb[i, 35] = 1.0  # Hypothesis
        if model.max_seq_len > 121:
            pos_emb[121, 36] = 1.0  # Equals/prediction position
        model.pos_emb.weight.copy_(pos_emb)
        
        # === LAYER 1: Word-level matching ===
        # Head 0-3: Each hypothesis position attends to matching premise characters
        # At hypothesis positions, the query matches premise tokens by character identity
        
        block0 = model.blocks[0]
        attn0 = block0.attn
        
        W_q = torch.zeros(d_model, d_model)
        W_k = torch.zeros(d_model, d_model)
        W_v = torch.zeros(d_model, d_model)
        W_o = torch.zeros(d_model, d_model)
        
        # Head 0: Character matching - hypothesis chars attend to matching premise chars
        # Query: "I'm a hypothesis character, find my match in premise"
        # Use dims 0-15 for this head
        head_start = 0
        head_end = dh
        
        for d in range(min(26, dh)):
            # Query activates on hypothesis letter dims (boosted by pos_emb dim 35)
            W_q[head_start + d, d] = 1.0  # Match on character identity
            W_q[head_start + d, 35] = 2.0  # Only when I'm in hypothesis region
            
            # Key activates on premise letter dims (boosted by pos_emb dim 34)
            W_k[head_start + d, d] = 1.0  # Match on character identity  
            W_k[head_start + d, 34] = 2.0  # Only when I'm in premise region
        
        # Value: when a match is found, pass the character identity
        for d in range(min(26, dh)):
            W_v[head_start + d, d] = 1.0
        
        # Output: write match result to dim 40-65 (match count region)
        for d in range(min(26, dh)):
            W_o[40 + d, head_start + d] = 1.0
        
        attn0.W_q.weight.copy_(W_q)
        attn0.W_k.weight.copy_(W_k)
        attn0.W_v.weight.copy_(W_v)
        attn0.W_o.weight.copy_(W_o)
        
        # === LAYER 2: Aggregate match scores ===
        # At position 121 (equals), sum up all match signals from hypothesis positions
        block1 = model.blocks[1]
        attn1 = block1.attn
        
        W_q1 = torch.zeros(d_model, d_model)
        W_k1 = torch.zeros(d_model, d_model)
        W_v1 = torch.zeros(d_model, d_model)
        W_o1 = torch.zeros(d_model, d_model)
        
        # Head 0: Equals position attends to hypothesis positions to collect match counts
        for d in range(min(26, dh)):
            # Query: activate on prediction position
            W_q1[d, 36] = 5.0  # Strong signal for equals position
            
            # Key: activate on hypothesis region
            W_k1[d, 35] = 5.0  # Only hypothesis positions
            
            # Value: pass match information
            W_v1[d, 40 + d] = 1.0  # The match signals from layer 1
        
        # Output: write aggregated match score to dim 70
        for d in range(min(26, dh)):
            W_o1[70, d] = 1.0 / dh  # Average to get overlap score
        
        attn1.W_q.weight.copy_(W_q1)
        attn1.W_k.weight.copy_(W_k1)
        attn1.W_v.weight.copy_(W_v1)
        attn1.W_o.weight.copy_(W_o1)
        
        # === LAYER 2 MLP: Convert overlap score to class ===
        mlp1 = block1.mlp
        W1 = torch.zeros(model.d_ff, d_model)
        b1 = torch.zeros(model.d_ff)
        W2 = torch.zeros(d_model, model.d_ff)
        b2 = torch.zeros(d_model)
        
        # Feature 0: Is overlap high? (entailment signal)
        W1[0, 70] = 20.0
        b1[0] = -8.0  # High threshold
        
        # Feature 1: Is overlap low? (contradiction signal)
        W1[1, 70] = -20.0
        b1[1] = 5.0  # Low threshold
        
        # Output to dim 80 (entailment score) and dim 81 (contradiction score)
        W2[80, 0] = 1.0
        W2[81, 1] = 1.0
        
        mlp1.fc1.weight.copy_(W1)
        mlp1.fc1.bias.copy_(b1)
        mlp1.fc2.weight.copy_(W2)
        mlp1.fc2.bias.copy_(b2)
        
        # === FINAL HEAD: Map to class predictions ===
        W_head = torch.zeros(model.vocab_size, d_model)
        b_head = torch.zeros(model.vocab_size)
        
        # Default: neutral (class 1, token 32)
        b_head.fill_(-20.0)
        b_head[32] = 10.0  # '1' (neutral)
        
        # High entailment score -> predict 0
        W_head[31, 80] = 20.0   # '0'
        W_head[32, 80] = -10.0  # reduce neutral
        
        # High contradiction score -> predict 2
        W_head[33, 81] = 20.0   # '2'
        W_head[32, 81] = -10.0  # reduce neutral
        
        model.head.weight.copy_(W_head)
        model.head.bias.copy_(b_head)
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
model_shorthand_name = "CharOverlap_v3"
model_description = "Character-level overlap: hypothesis chars attend to matching premise chars, aggregated at equals position."


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
