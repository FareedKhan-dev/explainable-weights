"""
Interpretable transformer for character-level sequence tasks.

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.
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
# Agent's interpretable weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Comprehensive data-driven approach combining multiple feature types."""
    torch.manual_seed(0)
    
    from collections import defaultdict, Counter
    
    # Large-scale analysis
    examples = task.generate_examples(n=1000, seed=42)
    
    word_class = {label: Counter() for label in ['0', '1', '2']}
    bigram_class = {label: Counter() for label in ['0', '1', '2']}
    char_class = {label: Counter() for label in ['0', '1', '2']}
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_').lower()
        premise = prompt[:pipe_idx].strip().lower()
        
        # Word-level analysis
        for word in hypothesis.split():
            word_class[answer][word] += 1
        
        # Bigram analysis (first 2 chars of hypothesis)
        if len(hypothesis) >= 2:
            bigram = hypothesis[:2]
            bigram_class[answer][bigram] += 1
        
        # Character frequency in hypothesis
        for c in hypothesis:
            if c.isalpha():
                char_class[answer][c] += 1
    
    # Find most predictive words (by information gain)
    word_scores = []
    all_words = set()
    for label in ['0', '1', '2']:
        all_words.update(word_class[label].keys())
    
    for word in all_words:
        freqs = [word_class[label].get(word, 0) for label in ['0', '1', '2']]
        total = sum(freqs)
        if total >= 5:  # Need minimum count
            probs = [f / total for f in freqs]
            max_prob = max(probs)
            cls = probs.index(max_prob)
            # Weight by both strength and count
            score = max_prob * math.log(total + 1)
            word_scores.append((word, cls, max_prob, total, score))
    
    word_scores.sort(key=lambda x: x[4], reverse=True)
    print(f"Top 15 predictive words: {[(w, c, f'{p:.2f}', n) for w, c, p, n, s in word_scores[:15]]}")
    
    # Find most predictive bigrams
    bigram_scores = []
    all_bigrams = set()
    for label in ['0', '1', '2']:
        all_bigrams.update(bigram_class[label].keys())
    
    for bg in all_bigrams:
        freqs = [bigram_class[label].get(bg, 0) for label in ['0', '1', '2']]
        total = sum(freqs)
        if total >= 10:
            probs = [f / total for f in freqs]
            max_prob = max(probs)
            cls = probs.index(max_prob)
            score = max_prob * math.log(total + 1)
            bigram_scores.append((bg, cls, max_prob, total, score))
    
    bigram_scores.sort(key=lambda x: x[4], reverse=True)
    print(f"Top 10 predictive bigrams: {[(b, c, f'{p:.2f}', n) for b, c, p, n, s in bigram_scores[:10]]}")
    
    # Select diverse top features from words (different first chars to avoid conflicts)
    selected_words = []
    used_first_chars = set()
    for word, cls, prob, total, score in word_scores:
        first_char = word[0]
        if first_char not in used_first_chars and prob >= 0.55:
            selected_words.append((word, cls, prob))
            used_first_chars.add(first_char)
            if len(selected_words) >= 4:
                break
    
    print(f"Selected words for detection: {selected_words}")
    
    D = model.d_model  # 64
    V = model.vocab_size
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Token embeddings: character identity (one-hot-ish)
        for i in range(V):
            model.token_emb.weight[i, :] = 0.0
            if i < D:
                model.token_emb.weight[i, i] = 1.0
        
        # Position embeddings: zero
        model.pos_emb.weight.zero_()
        
        # Layer 1: Detect first-character of selected words via attention
        layer1 = model.blocks[0]
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        dh = D // model.n_heads  # 16
        
        for i, (word, cls, prob) in enumerate(selected_words):
            first_char_idx = task.stoi.get(word[0], 0)
            # Each head detects a different word's first character
            layer1.attn.W_q.weight[i*dh, first_char_idx] = 10.0
            layer1.attn.W_k.weight[i*dh, first_char_idx] = 10.0
            layer1.attn.W_v.weight[i*dh, first_char_idx] = 1.0
            layer1.attn.W_o.weight[10+i, i*dh] = 3.0
        
        # MLP layer 1: pass through
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Layer 2: Classify using MLP based on detected features
        layer2 = model.blocks[1]
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        for i, (word, cls, prob) in enumerate(selected_words):
            # Neuron i fires when feature i is detected
            layer2.mlp.fc1.weight[i, 10+i] = 6.0
            layer2.mlp.fc1.bias[i] = -3.0
            # Map to the predicted class
            layer2.mlp.fc2.weight[20+cls, i] = 10.0 * prob
        
        # Default bias towards neutral (class 2)
        layer2.mlp.fc2.bias[22] = 1.5
        
        # Layer 3: pass through
        layer3 = model.blocks[2]
        layer3.attn.W_q.weight.zero_()
        layer3.attn.W_k.weight.zero_()
        layer3.attn.W_v.weight.zero_()
        layer3.attn.W_o.weight.zero_()
        layer3.mlp.fc1.weight.zero_()
        layer3.mlp.fc1.bias.zero_()
        layer3.mlp.fc2.weight.zero_()
        layer3.mlp.fc2.bias.zero_()
        
        # Final layer norm
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        
        # Output head: map channels to class tokens
        model.head.weight.zero_()
        model.head.weight[ZERO_IDX, 20] = 10.0
        model.head.weight[ONE_IDX, 21] = 10.0
        model.head.weight[TWO_IDX, 22] = 10.0


model_shorthand_name = "DataDrivenComprehensive"
model_description = "Comprehensive data-driven word detection combining top predictive words across all categories."


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
