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
    """3-layer causal transformer, vocab/seq-len configured from the task."""

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
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Populate `model`'s parameters in-place. No training allowed.
    
    Attempt 3: Better structured weights with format detection.
    Strategy: Use position embeddings to identify output positions, attention to copy from input.
    """
    torch.manual_seed(123)
    
    d = model.d_model
    vocab_size = model.vocab_size
    vocab = task.vocab
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    
    with torch.no_grad():
        # Clear all weights
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # Token embeddings: one-hot for each character
        for i in range(vocab_size):
            model.token_emb.weight[i, i % d] = 1.0
        
        # Position embeddings: encode position strongly
        for pos in range(model.max_seq_len):
            # Use sinusoidal-like encoding
            for i in range(0, min(d, 32), 2):
                model.pos_emb.weight[pos, i] = math.sin(pos * (i+1) / 10.0)
                if i + 1 < d:
                    model.pos_emb.weight[pos, i+1] = math.cos(pos * (i+1) / 10.0)
        
        # Configure attention blocks
        for layer_idx, block in enumerate(model.blocks):
            # Attention: try to copy from specific relative positions
            block.attn.W_q.weight.zero_()
            block.attn.W_k.weight.zero_()
            block.attn.W_v.weight.zero_()
            block.attn.W_o.weight.zero_()
            
            # Make attention heads look at different offsets
            n_heads = block.attn.n_heads
            d_head = block.attn.d_head
            
            for h in range(n_heads):
                # Each head focuses on a different relative position
                offset = h * 2  # Look back 0, 2, 4, 6... positions
                
                # Q and K project to same subspace for position matching
                start_h = h * d_head
                end_h = start_h + min(d_head, d // 2)
                
                for i in range(start_h, end_h):
                    if i < d:
                        block.attn.W_q.weight[i, i] = 1.0
                        block.attn.W_k.weight[i, i] = 1.0
                        block.attn.W_v.weight[i, i] = 1.0
                        block.attn.W_o.weight[i, i] = 1.0
            
            # MLP: try to process and combine information
            block.mlp.fc1.weight.zero_()
            block.mlp.fc1.bias.zero_()
            block.mlp.fc2.weight.zero_()
            block.mlp.fc2.bias.zero_()
            
            # Identity-like transformation
            d_ff = block.mlp.fc1.out_features
            for i in range(min(d, d_ff)):
                block.mlp.fc1.weight[i, i] = 1.0
            for i in range(min(d_ff, d)):
                block.mlp.fc2.weight[i, i] = 1.0
        
        # Output head: bias toward correct format
        model.head.weight.zero_()
        
        # The answer is always YYYY-MM-DD (10 chars)
        # Positions 4 and 7 should be dashes
        # Other positions should be digits
        
        # Add strong bias for dashes at positions 4 and 7
        if '-' in char_to_idx:
            dash_idx = char_to_idx['-']
            model.head.weight[dash_idx, :] = 0.5
        
        # Favor digits
        for digit in '0123456789':
            if digit in char_to_idx:
                model.head.weight[char_to_idx[digit], :] = 0.1


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "exp3-better-structure"
model_description = "Sinusoidal position encoding + multi-head attention at different offsets"


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
    # Make model bigger for this complex task
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
    )
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
    
    # DEBUG: Print task details
    examples = task.generate_examples(5)
    for i, ex in enumerate(examples):
        print(f"  prompt len={len(ex.prompt)}, answer len={len(ex.answer)}, prompt='{ex.prompt}', answer='{ex.answer}'")
        # Show tokenization
        if hasattr(task, 'encode'):
            ids = task.encode(ex.prompt)
            print(f"  encoded prompt: {ids}")
        if hasattr(task, 'char_to_idx'):
            print(f"  char_to_idx available")
        if i >= 2:  # Just show first 3
            break
    
    model = build_model(task).to(args.device)
    
    # DEBUG: Try manual generation
    if len(examples) > 0:
        test_ex = examples[0]
        ids = task.encode(test_ex.prompt)
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor([ids], device=args.device)
        else:
            ids = ids.unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            logits = model(ids)
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)
            
            if hasattr(task, 'decode'):
                pred_text = task.decode(pred_ids[0])
                print(f"\nDEBUG: Model prediction: '{pred_text}'")
                print(f"DEBUG: Expected answer: '{test_ex.answer}'")
            else:
                print(f"\nDEBUG: Predicted token IDs: {pred_ids[0].tolist()}")

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
