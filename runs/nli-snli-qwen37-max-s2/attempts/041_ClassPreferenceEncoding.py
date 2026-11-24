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
    """Direct class-preference encoding in token embeddings.
    
    Strategy: Each character's embedding directly encodes its class preference
    (log-odds from training data). Layer 1 uses uniform attention to average
    character signals. Layer 2 MLP amplifies and classifies.
    """
    torch.manual_seed(0)
    
    from collections import Counter
    
    examples = task.generate_examples(n=3000, seed=42)
    
    D = model.d_model  # 64
    V = model.vocab_size
    
    # Compute character class preferences
    char_counts = {label: Counter() for label in ['0', '1', '2']}
    class_totals = {'0': 0, '1': 0, '2': 0}
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_').lower()
        
        class_totals[answer] += 1
        for ch in hypothesis:
            if ch.isalpha():
                char_counts[answer][ch] += 1
    
    # Compute log-odds for each character and class
    total_examples = sum(class_totals.values())
    class_priors = {label: class_totals[label] / total_examples for label in ['0', '1', '2']}
    
    # For each character, compute class preference vector
    char_prefs = {}
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        prefs = []
        for label in ['0', '1', '2']:
            count = char_counts[label].get(ch, 0) + 1  # Laplace smoothing
            total = class_totals[label] + 26  # smoothing
            p_ch_given_class = count / total
            p_class = class_priors[label]
            # Log-odds: log(P(class|ch) / P(class))  ∝  log(P(ch|class) / P(ch))
            log_odds = math.log(p_ch_given_class + 1e-10) - math.log(p_class + 1e-10)
            prefs.append(log_odds)
        # Normalize to zero mean
        mean_pref = sum(prefs) / 3
        prefs = [p - mean_pref for p in prefs]
        char_prefs[ch] = prefs
    
    # Print top discriminative characters
    char_strength = {ch: max(abs(p) for p in prefs) for ch, prefs in char_prefs.items()}
    top_chars = sorted(char_strength.items(), key=lambda x: x[1], reverse=True)
    print(f"Top discriminative characters:")
    for ch, s in top_chars[:10]:
        print(f"  {ch}: prefs={[f'{p:.3f}' for p in char_prefs[ch]]}, strength={s:.3f}")
    
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # ---- Token embeddings: encode class preference in dims 0,1,2 ----
        # Also encode character identity in dims 3-28 for discrimination
        for i in range(V):
            model.token_emb.weight[i, :] = 0.0
        
        # Encode class preferences for lowercase chars
        scale = 3.0
        for ch in 'abcdefghijklmnopqrstuvwxyz':
            idx = task.stoi.get(ch, -1)
            if idx >= 0:
                prefs = char_prefs[ch]
                model.token_emb.weight[idx, 0] = prefs[0] * scale  # class 0 pref
                model.token_emb.weight[idx, 1] = prefs[1] * scale  # class 1 pref
                model.token_emb.weight[idx, 2] = prefs[2] * scale  # class 2 pref
                # Also encode character identity
                ch_dim = 3 + (ord(ch) - ord('a'))
                if ch_dim < D:
                    model.token_emb.weight[idx, ch_dim] = 1.0
        
        # Uppercase versions (same preferences)
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            idx = task.stoi.get(ch, -1)
            if idx >= 0:
                lower = ch.lower()
                prefs = char_prefs.get(lower, [0, 0, 0])
                model.token_emb.weight[idx, 0] = prefs[0] * scale
                model.token_emb.weight[idx, 1] = prefs[1] * scale
                model.token_emb.weight[idx, 2] = prefs[2] * scale
                ch_dim = 3 + (ord(lower) - ord('a'))
                if ch_dim < D:
                    model.token_emb.weight[idx, ch_dim] = 1.0
        
        # Space and other common chars get neutral preference
        for ch in ' |_=.,!?':
            idx = task.stoi.get(ch, -1)
            if idx >= 0:
                model.token_emb.weight[idx, :] = 0.0
        
        model.pos_emb.weight.zero_()
        
        # ---- Layer 1: Uniform attention to average class preferences ----
        layer1 = model.blocks[0]
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        dh = D // model.n_heads  # 16
        
        # Make Q and K output a constant vector so attention is uniform
        # Use dimension 0 as the "uniform attention" dimension
        # After LayerNorm, all inputs are normalized, so we need Q and K
        # to project to the same direction regardless of input
        # Approach: make Q and K both project to a fixed direction
        layer1.attn.W_q.weight[0, :] = 1.0  # All inputs → same query
        layer1.attn.W_k.weight[0, :] = 1.0  # All inputs → same key
        # This gives uniform attention scores (all Q·K^T are the same)
        
        # V copies the class preference signal (dims 0,1,2)
        for d in range(3):
            layer1.attn.W_v.weight[d, d] = 1.0
        
        # W_o copies the averaged signal to the residual stream
        for d in range(3):
            layer1.attn.W_o.weight[d, d] = 2.0  # Amplify
        
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # ---- Layer 2: Amplify class signal ----
        layer2 = model.blocks[1]
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        # MLP amplifies the class preference dimensions
        for c in range(3):
            layer2.mlp.fc1.weight[c, c] = 8.0
            layer2.mlp.fc1.bias[c] = 0.0
            layer2.mlp.fc2.weight[c, c] = 8.0
        
        # ---- Layer 3: pass through ----
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
        
        # Head reads class preference from dims 0,1,2
        model.head.weight.zero_()
        model.head.weight[ZERO_IDX, 0] = 10.0
        model.head.weight[ONE_IDX, 1] = 10.0
        model.head.weight[TWO_IDX, 2] = 10.0


model_shorthand_name = "ClassPreferenceEncoding"
model_description = "Direct class-preference encoding in token embeddings with uniform attention averaging and MLP amplification."


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
