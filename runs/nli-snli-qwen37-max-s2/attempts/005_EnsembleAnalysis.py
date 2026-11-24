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
    """Populate `model`'s parameters in-place. No training allowed."""
    torch.manual_seed(0)
    
    # Analyze the task to find good heuristics
    examples = task.generate_examples(n=200, seed=42)
    
    # Compute multiple features for each example
    feature_data = []
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        
        # Split at |
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        
        premise = prompt[:pipe_idx].strip()
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_')
        
        # Feature 1: Word overlap ratio (hypothesis words found in premise)
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        word_overlap = len(premise_words & hypothesis_words) / max(len(hypothesis_words), 1)
        
        # Feature 2: Premise word coverage (what fraction of premise words are in hypothesis)
        premise_coverage = len(premise_words & hypothesis_words) / max(len(premise_words), 1)
        
        # Feature 3: Length ratio (hypothesis / premise)
        len_ratio = len(hypothesis) / max(len(premise), 1)
        
        # Feature 4: Has "not"/"never" in hypothesis but not premise
        negation_words = {'not', 'no', 'never', 'nobody', 'nothing', 'nowhere', 'nor'}
        hyp_neg = any(neg in hypothesis.lower().split() for neg in negation_words)
        prem_neg = any(neg in premise.lower().split() for neg in negation_words)
        neg_diff = int(hyp_neg and not prem_neg)
        
        feature_data.append({
            'answer': answer,
            'word_overlap': word_overlap,
            'premise_coverage': premise_coverage,
            'len_ratio': len_ratio,
            'neg_diff': neg_diff,
        })
    
    # Best heuristic: word_overlap based with negation
    best_acc = 0
    best_params = None
    for t_contra in range(0, 6):
        for t_entail in range(t_contra, 11):
            tc = t_contra / 10.0
            te = t_entail / 10.0
            correct = 0
            for d in feature_data:
                wo = d['word_overlap']
                if wo >= te:
                    pred = '0'
                elif wo < tc:
                    pred = '1'
                else:
                    pred = '2'
                if pred == d['answer']:
                    correct += 1
            acc = correct / len(feature_data)
            if acc > best_acc:
                best_acc = acc
                best_params = (tc, te)
    
    print(f"Best word overlap heuristic: contra<{best_params[0]:.2f}, entail>={best_params[1]:.2f}, acc={best_acc:.4f}")
    
    # Now let's also try premise_coverage
    best_acc_cov = 0
    best_params_cov = None
    for t_contra in range(0, 6):
        for t_entail in range(t_contra, 11):
            tc = t_contra / 10.0
            te = t_entail / 10.0
            correct = 0
            for d in feature_data:
                wo = d['premise_coverage']
                if wo >= te:
                    pred = '0'
                elif wo < tc:
                    pred = '1'
                else:
                    pred = '2'
                if pred == d['answer']:
                    correct += 1
            acc = correct / len(feature_data)
            if acc > best_acc_cov:
                best_acc_cov = acc
                best_params_cov = (tc, te)
    
    print(f"Best premise_coverage heuristic: contra<{best_params_cov[0]:.2f}, entail>={best_params_cov[1]:.2f}, acc={best_acc_cov:.4f}")
    
    # Combined: word_overlap + premise_coverage + negation
    best_acc_comb = 0
    best_params_comb = None
    for tw_contra in range(0, 6):
        for tw_entail in range(tw_contra, 11):
            for ts_contra in range(0, 6):
                for ts_entail in range(ts_contra, 11):
                    twc = tw_contra / 10.0
                    twe = tw_entail / 10.0
                    tsc = ts_contra / 10.0
                    tse = ts_entail / 10.0
                    correct = 0
                    for d in feature_data:
                        wo = d['word_overlap']
                        cov = d['premise_coverage']
                        neg = d['neg_diff']
                        
                        # Entailment: high word overlap AND high coverage
                        # Contradiction: low word overlap OR low coverage
                        if wo >= twe and cov >= tse:
                            pred = '0'
                        elif wo < twc or cov < tsc:
                            pred = '1'
                        else:
                            pred = '2'
                        
                        # Negation flips entailment to contradiction
                        if pred == '0' and neg:
                            pred = '1'
                        
                        if pred == d['answer']:
                            correct += 1
                    acc = correct / len(feature_data)
                    if acc > best_acc_comb:
                        best_acc_comb = acc
                        best_params_comb = (twc, twe, tsc, tse)
    
    print(f"Best combined heuristic: w_contra<{best_params_comb[0]:.2f}, w_entail>={best_params_comb[1]:.2f}, cov_contra<{best_params_comb[2]:.2f}, cov_entail>={best_params_comb[3]:.2f}, acc={best_acc_comb:.4f}")
    
    return


model_shorthand_name = "EnsembleAnalysis"
model_description = "Analyzing ensemble of features for NLI."


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
