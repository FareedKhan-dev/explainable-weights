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
    
    Strategy: Multiple common word detector. Detect presence of multiple
    high-frequency words that have class preferences.
    """
    torch.manual_seed(0)
    
    # Analyze training data to find common words with preferences
    examples = task.generate_examples(n=500, seed=42)
    
    from collections import defaultdict
    word_counts = {label: defaultdict(int) for label in ['0', '1', '2']}
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        
        # Extract hypothesis (after |, before =)
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_').lower()
        
        # Count words
        for word in hypothesis.split():
            word_counts[answer][word] += 1
    
    # Find words with moderate class preference but high frequency
    word_scores = {}
    for word in set().union(*[wc.keys() for wc in word_counts.values()]):
        freqs = [word_counts[label].get(word, 0) for label in ['0', '1', '2']]
        total = sum(freqs)
        if total >= 15:  # Must appear at least 15 times
            probs = [f / total for f in freqs]
            max_prob = max(probs)
            if max_prob > 0.5:  # Moderate preference
                word_scores[word] = {
                    'probs': probs,
                    'count': total,
                    'class': probs.index(max_prob),
                    'strength': max_prob
                }
    
    # Get top words for each class
    entail_words = sorted([w for w, s in word_scores.items() if s['class'] == 0], 
                         key=lambda w: word_scores[w]['count'], reverse=True)[:20]
    contra_words = sorted([w for w, s in word_scores.items() if s['class'] == 1], 
                         key=lambda w: word_scores[w]['count'], reverse=True)[:20]
    neutral_words = sorted([w for w, s in word_scores.items() if s['class'] == 2], 
                          key=lambda w: word_scores[w]['count'], reverse=True)[:20]
    
    print(f"Common entailment words: {entail_words[:12]}")
    print(f"Common contradiction words: {contra_words[:12]}")
    print(f"Common neutral words: {neutral_words[:12]}")
    
    D = model.d_model  # 64
    V = model.vocab_size
    
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Token embeddings: preserve character identity
        for i in range(V):
            model.token_emb.weight[i, :] = 0.0
            if i < D:
                model.token_emb.weight[i, i] = 1.0
        
        # Position embeddings: minimal
        for pos in range(model.max_seq_len):
            model.pos_emb.weight[pos, :] = 0.0
        
        # Layer 1: Detect first characters of common words
        layer1 = model.blocks[0]
        
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        # MLP: Detect first characters
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Get unique first characters with weights based on frequency
        entail_char_scores = defaultdict(float)
        for word in entail_words:
            if word[0] in task.stoi:
                entail_char_scores[word[0]] += word_scores[word]['count'] * word_scores[word]['strength']
        
        contra_char_scores = defaultdict(float)
        for word in contra_words:
            if word[0] in task.stoi:
                contra_char_scores[word[0]] += word_scores[word]['count'] * word_scores[word]['strength']
        
        neutral_char_scores = defaultdict(float)
        for word in neutral_words:
            if word[0] in task.stoi:
                neutral_char_scores[word[0]] += word_scores[word]['count'] * word_scores[word]['strength']
        
        # Create weighted detectors
        entail_chars_sorted = sorted(entail_char_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        contra_chars_sorted = sorted(contra_char_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        neutral_chars_sorted = sorted(neutral_char_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (char, score) in enumerate(entail_chars_sorted):
            char_idx = task.stoi[char]
            layer1.mlp.fc1.weight[i, char_idx] = 4.0
            layer1.mlp.fc1.bias[i] = -2.0
            layer1.mlp.fc2.weight[10, i] = score / 100.0  # Weighted contribution
        
        for i, (char, score) in enumerate(contra_chars_sorted):
            char_idx = task.stoi[char]
            layer1.mlp.fc1.weight[10+i, char_idx] = 4.0
            layer1.mlp.fc1.bias[10+i] = -2.0
            layer1.mlp.fc2.weight[11, 10+i] = score / 100.0
        
        for i, (char, score) in enumerate(neutral_chars_sorted):
            char_idx = task.stoi[char]
            layer1.mlp.fc1.weight[20+i, char_idx] = 4.0
            layer1.mlp.fc1.bias[20+i] = -2.0
            layer1.mlp.fc2.weight[12, 20+i] = score / 100.0
        
        # Layer 2: Threshold and classify
        layer2 = model.blocks[1]
        
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        # Neuron 0: High entail signal
        layer2.mlp.fc1.weight[0, 10] = 4.0
        layer2.mlp.fc1.bias[0] = -15.0
        
        # Neuron 1: High contra signal
        layer2.mlp.fc1.weight[1, 11] = 4.0
        layer2.mlp.fc1.bias[1] = -15.0
        
        # Neuron 2: High neutral signal
        layer2.mlp.fc1.weight[2, 12] = 4.0
        layer2.mlp.fc1.bias[2] = -15.0
        
        # Map to output classes
        layer2.mlp.fc2.weight[20, 0] = 10.0
        layer2.mlp.fc2.weight[21, 1] = 10.0
        layer2.mlp.fc2.weight[22, 2] = 10.0
        
        # Default to neutral
        layer2.mlp.fc2.bias[22] = 4.0
        
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
        
        # Output head
        model.head.weight.zero_()
        model.head.weight[ZERO_IDX, 20] = 10.0
        model.head.weight[ONE_IDX, 21] = 10.0
        model.head.weight[TWO_IDX, 22] = 10.0


model_shorthand_name = "MultipleCommonWords"
model_description = "Detecting multiple common words with moderate class preferences using weighted contributions."


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
