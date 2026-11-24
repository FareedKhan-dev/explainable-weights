"""
Interpretable transformer for character-level sequence tasks.
NLI classifier for SNLI task.
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
    """Sentence length difference heuristic.
    
    Strategy: Compare premise vs hypothesis length - contradictions often have
    very different lengths, while entailments tend to have similar lengths.
    """
    torch.manual_seed(0)
    
    examples = task.generate_examples(n=3000, seed=42)
    
    D = model.d_model  # 64
    V = model.vocab_size
    
    # Analyze length differences per class
    length_diffs = {'0': [], '1': [], '2': []}
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        premise = prompt[:pipe_idx].strip()
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_')
        
        diff = abs(len(premise) - len(hypothesis))
        ratio = len(hypothesis) / (len(premise) + 1e-8)
        length_diffs[answer].append((diff, ratio))
    
    # Compute statistics
    stats = {}
    for label in ['0', '1', '2']:
        diffs = [d[0] for d in length_diffs[label]]
        ratios = [d[1] for d in length_diffs[label]]
        stats[label] = {
            'mean_diff': sum(diffs) / len(diffs),
            'mean_ratio': sum(ratios) / len(ratios),
            'high_diff_frac': sum(1 for d in diffs if d > 20) / len(diffs)
        }
    
    print(f"Length stats: {stats}")
    
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Token embeddings: encode character identity
        for i in range(V):
            model.token_emb.weight[i, :] = 0.0
        
        # Space character for counting words
        space_idx = task.stoi.get(' ', -1)
        if space_idx >= 0:
            model.token_emb.weight[space_idx, 0] = 1.0
        
        # Separator characters
        pipe_idx = task.stoi.get('|', -1)
        if pipe_idx >= 0:
            model.token_emb.weight[pipe_idx, 1] = 1.0
        
        model.pos_emb.weight.zero_()
        
        # Layer 1: Count spaces in premise and hypothesis using position
        layer1 = model.blocks[0]
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        # Head 0: detect spaces (word count proxy)
        dh = D // model.n_heads
        layer1.attn.W_q.weight[0, 0] = 0.1
        layer1.attn.W_k.weight[0, 0] = 0.1
        layer1.attn.W_v.weight[0, 0] = 1.0
        layer1.attn.W_o.weight[0, 0] = 1.0
        
        # Head 1: detect pipe separator
        layer1.attn.W_q.weight[dh, 1] = 0.1
        layer1.attn.W_k.weight[dh, 1] = 0.1
        layer1.attn.W_v.weight[dh, 1] = 1.0
        layer1.attn.W_o.weight[dh, 1] = 1.0
        
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Layer 2: Use length heuristic
        layer2 = model.blocks[1]
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        # Use MLP to compute class scores based on position-weighted space count
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        # Feature 0: space count signal (word count)
        layer2.mlp.fc1.weight[0, 0] = 5.0
        layer2.mlp.fc1.bias[0] = -2.0
        
        # Map to classes based on length heuristic
        # Class 0 (entailment): moderate length match
        layer2.mlp.fc2.weight[20, 0] = 2.0
        # Class 1 (neutral): slightly longer hypothesis
        layer2.mlp.fc2.weight[21, 0] = 1.5
        # Class 2 (contradiction): very different lengths
        layer2.mlp.fc2.weight[22, 0] = -2.0
        
        # Bias toward neutral
        layer2.mlp.fc2.bias[20] = 0.0
        layer2.mlp.fc2.bias[21] = 0.5
        layer2.mlp.fc2.bias[22] = -0.5
        
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
        
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        
        model.head.weight.zero_()
        model.head.weight[ZERO_IDX, 20] = 10.0
        model.head.weight[ONE_IDX, 21] = 10.0
        model.head.weight[TWO_IDX, 22] = 10.0


model_shorthand_name = "SentenceLengthDiff"
model_description = "Using premise-hypothesis length difference as a discriminative feature."


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
