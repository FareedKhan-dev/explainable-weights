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

    Default implementation: leave the random initialization untouched.
    The agent should replace this with hand-designed weights that implement
    the task (e.g. 5-digit addition).
    """
    # Keep deterministic so reruns of the same code produce the same model.
    torch.manual_seed(0)
    # No weight assignments yet — the model will keep its default random init.
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "RandomInitInterpretable"
model_description = "Default architecture, no weight assignment (random init, expected ~0% accuracy)."


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
    
    # DEBUG: analyze character overlap statistics
    print(f"=== ANALYSIS ===")
    examples = task.generate_examples(n=200, seed=0)
    
    from collections import Counter
    
    def char_overlap(s1, s2):
        """Bag-of-characters overlap coefficient"""
        c1, c2 = Counter(s1), Counter(s2)
        overlap = sum((c1 & c2).values())
        total = sum(c1.values()) + sum(c2.values())
        return overlap / total if total > 0 else 0
    
    def char_jaccard(s1, s2):
        c1, c2 = Counter(s1), Counter(s2)
        intersection = sum((c1 & c2).values())
        union = sum((c1 | c2).values())
        return intersection / union if union > 0 else 0
    
    def exact_match_ratio(s1, s2):
        matches = sum(1 for a, b in zip(s1, s2) if a == b)
        return matches / min(len(s1), len(s2))
    
    # Collect statistics for each threshold
    overlaps_0 = []
    overlaps_1 = []
    exact_0 = []
    exact_1 = []
    
    for ex in examples:
        pipe_pos = ex.prompt.index('|')
        s1 = ex.prompt[:pipe_pos]
        s2 = ex.prompt[pipe_pos+1:-1]  # remove trailing =
        
        ov = char_overlap(s1, s2)
        ex_ratio = exact_match_ratio(s1, s2)
        
        if ex.answer == '0':
            overlaps_0.append(ov)
            exact_0.append(ex_ratio)
        else:
            overlaps_1.append(ov)
            exact_1.append(ex_ratio)
    
    print(f"Overlap (class 0): mean={sum(overlaps_0)/len(overlaps_0):.3f}, min={min(overlaps_0):.3f}, max={max(overlaps_0):.3f}")
    print(f"Overlap (class 1): mean={sum(overlaps_1)/len(overlaps_1):.3f}, min={min(overlaps_1):.3f}, max={max(overlaps_1):.3f}")
    print(f"Exact match (class 0): mean={sum(exact_0)/len(exact_0):.3f}, min={min(exact_0):.3f}, max={max(exact_0):.3f}")
    print(f"Exact match (class 1): mean={sum(exact_1)/len(exact_1):.3f}, min={min(exact_1):.3f}, max={max(exact_1):.3f}")
    
    # Try different thresholds for overlap
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        correct = 0
        for ex in examples:
            pipe_pos = ex.prompt.index('|')
            s1 = ex.prompt[:pipe_pos]
            s2 = ex.prompt[pipe_pos+1:-1]
            ov = char_overlap(s1, s2)
            pred = '1' if ov >= thresh else '0'
            if pred == ex.answer:
                correct += 1
        print(f"  Threshold {thresh:.2f}: accuracy = {correct/len(examples):.3f}")
    
    # Try exact match thresholds
    for thresh in [0.10, 0.20, 0.30, 0.40, 0.50]:
        correct = 0
        for ex in examples:
            pipe_pos = ex.prompt.index('|')
            s1 = ex.prompt[:pipe_pos]
            s2 = ex.prompt[pipe_pos+1:-1]
            er = exact_match_ratio(s1, s2)
            pred = '1' if er >= thresh else '0'
            if pred == ex.answer:
                correct += 1
        print(f"  Exact match thresh {thresh:.2f}: accuracy = {correct/len(examples):.3f}")
    
    # Try combined: exact match > 0.5 OR overlap > 0.80
    correct = 0
    for ex in examples:
        pipe_pos = ex.prompt.index('|')
        s1 = ex.prompt[:pipe_pos]
        s2 = ex.prompt[pipe_pos+1:-1]
        ov = char_overlap(s1, s2)
        er = exact_match_ratio(s1, s2)
        pred = '1' if (er >= 0.30 or ov >= 0.80) else '0'
        if pred == ex.answer:
            correct += 1
    print(f"  Combined (exact>=0.30 OR overlap>=0.80): accuracy = {correct/len(examples):.3f}")
    
    # Try word overlap
    def word_overlap(s1, s2):
        w1 = set(s1.split())
        w2 = set(s2.split())
        if not w1 or not w2:
            return 0
        intersection = len(w1 & w2)
        return intersection / min(len(w1), len(w2))
    
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        correct = 0
        for ex in examples:
            pipe_pos = ex.prompt.index('|')
            s1 = ex.prompt[:pipe_pos]
            s2 = ex.prompt[pipe_pos+1:-1]
            wo = word_overlap(s1, s2)
            pred = '1' if wo >= thresh else '0'
            if pred == ex.answer:
                correct += 1
        print(f"  Word overlap thresh {thresh:.2f}: accuracy = {correct/len(examples):.3f}")
    
    print(f"=== END ANALYSIS ===")
    
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
