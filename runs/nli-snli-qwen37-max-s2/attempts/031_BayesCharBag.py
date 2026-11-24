"""
Interpretable transformer for character-level sequence tasks.
Bag-of-characters classifier with optimized linear classifier.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


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
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4, n_layers=3, d_ff=256):
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

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    """Bag-of-characters classifier with naive Bayes log-odds.
    
    Strategy: Use uniform attention to average character embeddings.
    Token embeddings encode naive Bayes log-odds for each class.
    The output head reads the averaged log-odds for classification.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    from collections import defaultdict, Counter
    
    # Analyze character frequencies per class
    examples = task.generate_examples(n=2000, seed=42)
    
    D = model.d_model  # 64
    V = model.vocab_size
    
    # Count character frequencies per class in hypothesis
    char_class_counts = {label: Counter() for label in ['0', '1', '2']}
    class_counts = Counter()
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_').lower()
        
        class_counts[answer] += 1
        for c in hypothesis:
            char_class_counts[answer][c] += 1
    
    total = sum(class_counts.values())
    class_prior = {k: class_counts[k] / total for k in ['0', '1', '2']}
    
    # Compute log P(c | class=k) with Laplace smoothing
    char_log_prob = {}  # char -> [log P(c|0), log P(c|1), log P(c|2)]
    all_chars = set()
    for k in ['0', '1', '2']:
        all_chars.update(char_class_counts[k].keys())
    
    for c in all_chars:
        log_probs = []
        for k in ['0', '1', '2']:
            count = char_class_counts[k].get(c, 0)
            total_k = sum(char_class_counts[k].values())
            # Laplace smoothing
            prob = (count + 1) / (total_k + len(all_chars))
            log_probs.append(math.log(prob))
        char_log_prob[c] = log_probs
    
    # Compute naive Bayes log-odds relative to uniform prior
    # log_odds(c, k) = log P(c|k) - log P(c)
    # where P(c) = avg over classes
    char_evidence = {}  # c -> [evidence_0, evidence_1, evidence_2]
    for c in all_chars:
        lps = char_log_prob[c]
        avg_lp = sum(lps) / 3
        char_evidence[c] = [lp - avg_lp for lp in lps]
    
    # Print top evidence characters
    for k in range(3):
        top_chars = sorted(char_evidence.items(), key=lambda x: x[1][k], reverse=True)[:10]
        print(f"Top evidence for class {k}: {[(c, f'{e[k]:.3f}') for c, e in top_chars]}")
    
    # Also compute per-word analysis for a separate feature
    word_class_counts = {label: Counter() for label in ['0', '1', '2']}
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_').lower()
        for word in hypothesis.split():
            word_class_counts[answer][word] += 1
    
    # Find words with strong class preference AND reasonable count
    word_evidence = {}
    all_words = set()
    for k in ['0', '1', '2']:
        all_words.update(word_class_counts[k].keys())
    
    for word in all_words:
        freqs = [word_class_counts[k].get(word, 0) for k in ['0', '1', '2']]
        total_w = sum(freqs)
        if total_w >= 10:  # Need decent count
            probs = [f / total_w for f in freqs]
            avg_p = 1/3
            evidence = [math.log(max(p, 0.01) / avg_p) for p in probs]
            max_ev = max(evidence)
            cls = evidence.index(max_ev)
            word_evidence[word] = {'evidence': evidence, 'class': cls, 'strength': max_ev, 'count': total_w}
    
    top_words = sorted(word_evidence.items(), key=lambda x: x[1]['strength'] * math.log(x[1]['count']), reverse=True)
    print(f"Top words by evidence*count: {[(w, d['class'], f'{d[\"strength\"]:.2f}', d['count']) for w, d in top_words[:20]]}")
    
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Create random base embeddings (zero-mean, unit-variance per token)
        base_embeddings = np.random.randn(V, D).astype(np.float32)
        for i in range(V):
            base_embeddings[i] -= base_embeddings[i].mean()
            base_embeddings[i] /= (base_embeddings[i].std() + 1e-8)
        
        # Add naive Bayes evidence to channels 0, 1, 2
        evidence_scale = 2.0  # Make evidence signal strong
        for token_idx in range(V):
            # Find the character for this token
            char = task.itos.get(token_idx, '')
            if char in char_evidence:
                ev = char_evidence[char]
                base_embeddings[token_idx, 0] += ev[0] * evidence_scale
                base_embeddings[token_idx, 1] += ev[1] * evidence_scale
                base_embeddings[token_idx, 2] += ev[2] * evidence_scale
            
            # Re-normalize
            base_embeddings[token_idx] -= base_embeddings[token_idx].mean()
            base_embeddings[token_idx] /= (base_embeddings[token_idx].std() + 1e-8)
        
        model.token_emb.weight.copy_(torch.tensor(base_embeddings))
        model.pos_emb.weight.zero_()
        
        # Layer 1: Uniform attention with identity V and O
        layer1 = model.blocks[0]
        layer1.attn.W_q.weight.zero_()  # Zero Q -> all queries are zero
        layer1.attn.W_k.weight.zero_()  # Zero K -> all keys are zero -> uniform attention
        layer1.attn.W_v.weight.copy_(torch.eye(D))  # Identity: pass through LN(x)
        layer1.attn.W_o.weight.copy_(torch.eye(D))  # Identity: pass through attention output
        
        # Layer 1 MLP: zero (pass-through via residual)
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Layers 2, 3: all zero (pass-through via residual)
        for layer in [model.blocks[1], model.blocks[2]]:
            layer.attn.W_q.weight.zero_()
            layer.attn.W_k.weight.zero_()
            layer.attn.W_v.weight.zero_()
            layer.attn.W_o.weight.zero_()
            layer.mlp.fc1.weight.zero_()
            layer.mlp.fc1.bias.zero_()
            layer.mlp.fc2.weight.zero_()
            layer.mlp.fc2.bias.zero_()
        
        # Final layer norm
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        
        # Output head: read evidence from channels 0, 1, 2
        model.head.weight.zero_()
        head_scale = 10.0
        model.head.weight[ZERO_IDX, 0] = head_scale
        model.head.weight[ONE_IDX, 1] = head_scale
        model.head.weight[TWO_IDX, 2] = head_scale


model_shorthand_name = "BayesCharBag"
model_description = "Naive Bayes bag-of-characters classifier with log-odds evidence in token embeddings and uniform attention averaging."


def upsert_overall_results(rows, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing = []
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


def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)
    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed, device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{"task": args.task, "accuracy": f"{accuracy:.4f}", "status": "", "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}", "description": model_description}], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
