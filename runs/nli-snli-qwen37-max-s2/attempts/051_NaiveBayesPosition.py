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
    """Naive Bayes character classifier with position-weighted attention.
    
    Strategy: For each token in vocabulary, compute P(class | char appears in hypothesis).
    Use position embeddings to weight hypothesis characters more than premise characters.
    This creates a proper Naive Bayes classifier in the transformer.
    """
    torch.manual_seed(0)
    
    from collections import Counter
    import numpy as np
    
    examples = task.generate_examples(n=5000, seed=42)
    
    D = model.d_model  # 64
    V = model.vocab_size
    max_seq = model.max_seq_len
    
    # For each character, count how often it appears in hypothesis for each class
    hyp_char_counts = {label: Counter() for label in ['0', '1', '2']}
    prem_char_counts = {label: Counter() for label in ['0', '1', '2']}
    class_counts = Counter()
    
    for ex in examples:
        prompt = ex.prompt
        answer = ex.answer
        pipe_idx = prompt.index('|')
        eq_idx = prompt.index('=')
        premise = prompt[:pipe_idx].strip()
        hypothesis = prompt[pipe_idx+1:eq_idx].strip().rstrip('_')
        
        class_counts[answer] += 1
        
        for ch in hypothesis.lower():
            if ch.isalpha():
                hyp_char_counts[answer][ch] += 1
        for ch in premise.lower():
            if ch.isalpha():
                prem_char_counts[answer][ch] += 1
    
    total = sum(class_counts.values())
    class_priors = {label: class_counts[label] / total for label in ['0', '1', '2']}
    print(f"Class priors: {class_priors}")
    
    # For each character, compute log-likelihood ratio for hypothesis chars
    # log P(class | ch in hyp) ∝ log P(ch in hyp | class) * P(class)
    char_log_odds = {}
    total_hyp_chars = {label: sum(hyp_char_counts[label].values()) for label in ['0', '1', '2']}
    
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        log_odds = []
        for label in ['0', '1', '2']:
            # P(ch | class) with Laplace smoothing
            p_ch_given_class = (hyp_char_counts[label][ch] + 1) / (total_hyp_chars[label] + 26)
            log_odds.append(math.log(p_ch_given_class) + math.log(class_priors[label]))
        char_log_odds[ch] = log_odds
    
    # Also compute for premise chars (should have less influence)
    prem_log_odds = {}
    total_prem_chars = {label: sum(prem_char_counts[label].values()) for label in ['0', '1', '2']}
    
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        log_odds = []
        for label in ['0', '1', '2']:
            p_ch_given_class = (prem_char_counts[label][ch] + 1) / (total_prem_chars[label] + 26)
            log_odds.append(math.log(p_ch_given_class) + math.log(class_priors[label]))
        prem_log_odds[ch] = log_odds
    
    # Normalize log odds to be centered
    for ch in char_log_odds:
        mean = sum(char_log_odds[ch]) / 3
        char_log_odds[ch] = [v - mean for v in char_log_odds[ch]]
    for ch in prem_log_odds:
        mean = sum(prem_log_odds[ch]) / 3
        prem_log_odds[ch] = [v - mean for v in prem_log_odds[ch]]
    
    print("Top hypothesis chars by discrimination:")
    char_disc = [(ch, max(char_log_odds[ch]) - min(char_log_odds[ch])) for ch in 'abcdefghijklmnopqrstuvwxyz']
    char_disc.sort(key=lambda x: x[1], reverse=True)
    for ch, disc in char_disc[:10]:
        print(f"  {ch}: log_odds={[f'{v:.3f}' for v in char_log_odds[ch]]}, disc={disc:.3f}")
    
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Token embeddings: encode log-odds for each character
        # Dims 0-2: hypothesis log-odds for each class
        # Dims 3-5: premise log-odds for each class
        for i in range(V):
            model.token_emb.weight[i, :] = 0.0
        
        # Encode character log-odds in token embeddings
        scale = 20.0  # Scale factor to make signals strong
        for ch in 'abcdefghijklmnopqrstuvwxyz':
            idx = task.stoi.get(ch, -1)
            if idx >= 0:
                for cls in range(3):
                    model.token_emb.weight[idx, cls] = char_log_odds[ch][cls] * scale
                    model.token_emb.weight[idx, 3+cls] = prem_log_odds[ch][cls] * scale
        
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            idx = task.stoi.get(ch, -1)
            if idx >= 0:
                ch_lower = ch.lower()
                for cls in range(3):
                    model.token_emb.weight[idx, cls] = char_log_odds[ch_lower][cls] * scale
                    model.token_emb.weight[idx, 3+cls] = prem_log_odds[ch_lower][cls] * scale
        
        # Pipe character: neutral (no signal)
        pipe_idx_tok = task.stoi.get('|', -1)
        if pipe_idx_tok >= 0:
            model.token_emb.weight[pipe_idx_tok, 6] = 1.0  # Separator marker
        
        # Position embeddings: weight hypothesis positions more than premise
        # Use dim 10 to create a "hypothesis weight" that increases with position
        model.pos_emb.weight.zero_()
        for pos in range(max_seq):
            # Linearly increasing weight for later positions (hypothesis tends to be later)
            model.pos_emb.weight[pos, 10] = pos / max_seq * 3.0
        
        # Layer 1: Attention with position bias to focus on hypothesis
        layer1 = model.blocks[0]
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        dh = D // model.n_heads
        
        # Head 0: uniform attention for averaging log-odds
        # Low temperature for uniform attention
        for d in range(6):
            layer1.attn.W_q.weight[d, d] = 0.01
            layer1.attn.W_k.weight[d, d] = 0.01
            layer1.attn.W_v.weight[d, d] = 1.0
            layer1.attn.W_o.weight[d, d] = 1.0
        
        # Head 1: position-biased attention (focus on later positions = hypothesis)
        layer1.attn.W_q.weight[dh+0, 10] = 3.0  # Q queries position
        layer1.attn.W_k.weight[dh+0, 10] = 3.0  # K uses position
        # V copies the hypothesis log-odds (dims 0-2)
        for d in range(3):
            layer1.attn.W_v.weight[dh+1+d, d] = 1.0
            layer1.attn.W_o.weight[dh+1+d, dh+1+d] = 1.0
        
        # Head 2: position-biased attention for premise (focus on earlier positions)
        layer1.attn.W_q.weight[2*dh+0, 10] = -3.0  # Q queries early position
        layer1.attn.W_k.weight[2*dh+0, 10] = -3.0  # K uses position  
        # V copies the premise log-odds (dims 3-5)
        for d in range(3):
            layer1.attn.W_v.weight[2*dh+1+d, 3+d] = 1.0
            layer1.attn.W_o.weight[2*dh+1+d, 2*dh+1+d] = 1.0
        
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Layer 2: Combine hypothesis and premise signals
        layer2 = model.blocks[1]
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        # MLP: sum up the hypothesis log-odds (from head 1 output) 
        # and subtract premise log-odds (from head 2 output)
        # Hypothesis log-odds are in dims dh+1 to dh+3
        # Premise log-odds are in dims 2*dh+1 to 2*dh+3
        
        # Use fc1 to create features, fc2 to map to class scores
        # Feature 0: hypothesis class 0 score - avg
        layer2.mlp.fc1.weight[0, dh+1] = 1.0  # hyp class 0
        layer2.mlp.fc1.weight[0, 2*dh+1] = -0.3  # subtract premise class 0
        layer2.mlp.fc1.bias[0] = 0.0
        
        # Feature 1: hypothesis class 1 score
        layer2.mlp.fc1.weight[1, dh+2] = 1.0
        layer2.mlp.fc1.weight[1, 2*dh+2] = -0.3
        layer2.mlp.fc1.bias[1] = 0.0
        
        # Feature 2: hypothesis class 2 score
        layer2.mlp.fc1.weight[2, dh+3] = 1.0
        layer2.mlp.fc1.weight[2, 2*dh+3] = -0.3
        layer2.mlp.fc1.bias[2] = 0.0
        
        # Also use the uniform average (dims 0-2)
        layer2.mlp.fc1.weight[3, 0] = 0.5
        layer2.mlp.fc1.weight[4, 1] = 0.5
        layer2.mlp.fc1.weight[5, 2] = 0.5
        
        # Map to class output dimensions
        amp = 10.0
        layer2.mlp.fc2.weight[20, 0] = amp  # class 0 from hyp
        layer2.mlp.fc2.weight[21, 1] = amp  # class 1 from hyp
        layer2.mlp.fc2.weight[22, 2] = amp  # class 2 from hyp
        layer2.mlp.fc2.weight[20, 3] = amp * 0.5  # class 0 from uniform
        layer2.mlp.fc2.weight[21, 4] = amp * 0.5
        layer2.mlp.fc2.weight[22, 5] = amp * 0.5
        
        # Slight neutral bias
        layer2.mlp.fc2.bias[20] = 0.0
        layer2.mlp.fc2.bias[21] = 0.3
        layer2.mlp.fc2.bias[22] = -0.3
        
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


model_shorthand_name = "NaiveBayesPosition"
model_description = "Naive Bayes log-odds in token embeddings with position-weighted attention for premise/hypothesis separation."


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
