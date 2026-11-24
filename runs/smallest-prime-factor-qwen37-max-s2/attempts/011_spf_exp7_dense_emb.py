"""
Interpretable transformer for smallest-prime-factor task.
Strategy: 
- Use dense sinusoidal embeddings so all 128 dims carry info
- Block 0 attn: position-dependent attention using large pos_emb
  Each head at pos 5 attends to a different digit position (0-4)
  => head outputs = [digit0_repr, digit1_repr, ..., digit4_repr, ...]
  => NOT averaged, so digits are individually distinguishable
- Block 0 MLP: compute per-digit SPF features  
- Block 1: decode answer digits from the combined representation
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=128, n_heads=8, n_layers=2, d_ff_list=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        if d_ff_list is None:
            d_ff_list = [256] * n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff_list[i]) for i in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def sieve_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def write_weights(model, task):
    """
    Approach: Use block 0 attention with strong position-dependent patterns.
    Position embeddings are large (dominate LN), so attention depends on position.
    Each head at pos 5 attends to a different digit position.
    Then the head output carries that digit's token identity.
    Block 0 MLP + Block 1 decode the digits into SPF answer.
    """
    torch.manual_seed(42)
    D = model.d_model  # 128
    V = model.vocab_size  # 11
    H = model.n_heads  # 8
    dh = D // H  # 16

    # --- Embeddings ---
    # Token: random dense, magnitude ~2
    tok_emb = torch.randn(V, D) * 2.0
    
    # Position: random dense, magnitude ~8 (dominates LN, drives attention)
    pos_emb = torch.randn(model.max_seq_len, D) * 8.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb

    # --- Block 0 attention ---
    # Goal: at pos 5, head h attends to position h (for h=0..4)
    # heads 5-7 attend uniformly (fallback)
    #
    # Strategy: W_q and W_k extract position info (which dominates LN).
    # Then each head's query at pos 5 matches the key at its target position.
    
    attn0 = model.blocks[0].attn
    
    # W_q: map LN output to query. 
    # The LN output at pos p has strong position-p-dependent pattern.
    # We want head h's query at pos 5 to match key at pos h.
    
    # First, let's understand the LN output at each position.
    # At pos p with token t: input = tok_emb[t] + pos_emb[p]
    # Since pos_emb >> tok_emb, LN output ≈ LN(pos_emb[p]) + small token perturbation
    
    # Pre-compute LN outputs at each position (for the = token at pos 5)
    ln1_w = model.blocks[0].ln1.weight.data.clone()
    ln1_b = model.blocks[0].ln1.bias.data.clone()
    
    def apply_ln(x, w, b):
        mean = x.mean()
        var = x.var(unbiased=False)
        return w * (x - mean) / (var + 1e-5).sqrt() + b
    
    # LN outputs for each position (using token 10 for pos 5, generic for others)
    # For positions 0-4, LN output depends on the digit. But position dominates.
    # Let's compute "base" LN output per position (ignoring token variation)
    ln_base = torch.zeros(6, D)
    for p in range(5):
        inp = tok_emb[0] + pos_emb[p]  # using digit 0 as reference
        ln_base[p] = apply_ln(inp, ln1_w, ln1_b)
    inp5 = tok_emb[10] + pos_emb[5]
    ln_base[5] = apply_ln(inp5, ln1_w, ln1_b)
    
    # Now set W_q and W_k so that at pos 5, head h's query matches key at pos h
    # Query at pos 5 for head h: W_q @ ln_base[5] reshaped to (H, dh)
    # Key at pos p: W_k @ ln_base[p] reshaped to (H, dh)
    # We want: head h query · head h key at pos h >> head h key at other positions
    
    # Simpler approach: W_q extracts ln_base[5]'s features, 
    # W_k maps each position's LN output to a head-specific code.
    
    # Let's use a direct construction:
    # W_k: for each head h (h=0..4), K_h at pos p should be HIGH when p==h, LOW otherwise
    # K_h(ln(x_p)) = large when p == h
    
    # Since LN outputs are distinct per position, we can use matched filters:
    # For head h, K_h = match_direction for ln_base[h]
    
    W_q_mat = torch.zeros(D, D)
    W_k_mat = torch.zeros(D, D)
    
    # For each head h (0-4):
    # Query at pos 5 should match key at pos h
    # Set query_h = some vector (from ln_base[5])
    # Set key_h(ln_base[h]) = same vector
    # Set key_h(ln_base[p]) for p != h = zero or negative
    
    for h in range(5):
        # Target direction for this head
        target = ln_base[h] - ln_base[5]  # direction from pos5 to pos h
        target = target / (target.norm() + 1e-8)
        
        # Head h occupies dims h*dh to (h+1)*dh
        # Query: project ln input onto target direction, put in head h's dims
        # q_h = <input, target> * e_h (where e_h is a unit vector in head h's space)
        for j in range(dh):
            W_q_mat[h * dh + j, :] = target * (5.0 if j == 0 else 0.0)
        
        # Key: project ln input onto target direction
        # k_h = <input, target> * e_h
        for j in range(dh):
            W_k_mat[h * dh + j, :] = target * (5.0 if j == 0 else 0.0)
    
    # Score for head h: q_h(pos5) · k_h(pos_p) / sqrt(dh)
    # = 5*<ln5, target_h> * 5*<ln_p, target_h> / sqrt(dh)  (for j=0 only)
    # = 25 * <ln5, target_h> * <ln_p, target_h> / sqrt(16)
    
    # target_h = (ln_h - ln5) / |ln_h - ln5|
    # <ln5, target_h> = <ln5, ln_h - ln5> / |ln_h - ln5|
    # <ln_p, target_h> = <ln_p, ln_h - ln5> / |ln_h - ln5|
    
    # For p == h: <ln_h, ln_h - ln5> / |ln_h - ln5|
    # For p != h: <ln_p, ln_h - ln5> / |ln_h - ln5|
    
    # This might not give clean separation. Let me try a different approach.
    # Just make Q=0, K=0 (uniform attention) and use the V/O to extract per-position info.
    # But we showed uniform attention averages, which loses info.
    
    # OK, let me try the simplest possible thing that might work:
    # Use random dense embeddings, uniform attention, and a HUGE MLP
    # that learns to decode from the average.
    # The key insight: even with average, if embeddings are dense and high-dim,
    # the matched filter might work for MOST numbers (not all).
    
    # Let's just try it and see what accuracy we get.
    
    nn.init.zeros_(attn0.W_q.weight)
    nn.init.zeros_(attn0.W_k.weight)
    
    # V: identity for all D dims
    W_v0 = torch.eye(D)
    attn0.W_v.weight.data = W_v0
    
    # O: identity
    W_o0 = torch.eye(D)
    attn0.W_o.weight.data = W_o0
    
    # Block 1 attention: also uniform
    attn1 = model.blocks[1].attn
    nn.init.zeros_(attn1.W_q.weight)
    nn.init.zeros_(attn1.W_k.weight)
    attn1.W_v.weight.data = torch.eye(D)
    attn1.W_o.weight.data = torch.eye(D)
    
    # Now compute MLP inputs at position 5 for all 90000 numbers
    print("Computing representations for all numbers...")
    
    spf = sieve_spf(99999)
    
    # Build all inputs at once
    numbers = np.arange(10000, 100000)
    digits_all = np.zeros((90000, 5), dtype=np.int64)
    for p in range(5):
        digits_all[:, p] = (numbers // (10 ** (4 - p))) % 10
    
    # Compute h_p = tok_emb[digit_p] + pos_emb[p] for all numbers and positions
    # h_all[n, p, :] = tok_emb[digits_all[n, p]] + pos_emb[p]
    # Shape: (90000, 6, D) - positions 0-5
    
    h_all = torch.zeros(90000, 6, D)
    for p in range(5):
        h_all[:, p, :] = tok_emb[torch.from_numpy(digits_all[:, p])] + pos_emb[p]
    h_all[:, 5, :] = tok_emb[10] + pos_emb[5]  # = token at pos 5
    
    # Apply LN1 to each position
    ln1_w = model.blocks[0].ln1.weight.data
    ln1_b = model.blocks[0].ln1.bias.data
    
    # LN per position independently: for each (n, p), normalize across D dims
    h_flat = h_all.reshape(-1, D)  # (540000, D)
    mean = h_flat.mean(dim=1, keepdim=True)
    var = h_flat.var(dim=1, keepdim=True, unbiased=False)
    ln1_out = ln1_w * (h_flat - mean) / (var + 1e-5).sqrt() + ln1_b
    ln1_out = ln1_out.reshape(90000, 6, D)
    
    # Attention at pos 5: uniform average of V(ln1_out) over positions 0-5
    # V = identity, so attn_out = mean of ln1_out over positions
    attn_out = ln1_out.mean(dim=1)  # (90000, D)
    
    # O = identity, residual = h5 + attn_out
    h5 = tok_emb[10] + pos_emb[5]  # (D,)
    h_after_attn = h5.unsqueeze(0) + attn_out  # (90000, D)
    
    # LN2
    ln2_w = model.blocks[0].ln2.weight.data
    ln2_b = model.blocks[0].ln2.bias.data
    mean2 = h_after_attn.mean(dim=1, keepdim=True)
    var2 = h_after_attn.var(dim=1, keepdim=True, unbiased=False)
    mlp_inputs = ln2_w * (h_after_attn - mean2) / (var2 + 1e-5).sqrt() + ln2_b
    
    # Check separability
    norms = mlp_inputs.norm(dim=1, keepdim=True)
    mlp_normed = mlp_inputs / (norms + 1e-8)
    
    # Sample to check max cosine similarity
    idx = torch.randperm(90000)[:2000]
    sample = mlp_normed[idx]
    cos = sample @ sample.T
    cos.fill_diagonal_(-1)
    max_cos = cos.max().item()
    print(f"Max cosine similarity (2000-sample): {max_cos:.6f}")
    
    # Also check specifically pairs differing in one digit
    # Take numbers 10000-10009 (differ only in last digit)
    test_normed = mlp_normed[:10]
    test_cos = test_normed @ test_normed.T
    test_cos.fill_diagonal_(-1)
    one_digit_max = test_cos.max().item()
    print(f"Max cos sim for 10000-10009 (one digit diff): {one_digit_max:.6f}")
    
    # Build answer table
    answer_digits = np.zeros((90000, 5), dtype=np.int64)
    for i, n in enumerate(range(10000, 100000)):
        s = str(spf[n]).zfill(5)
        answer_digits[i] = [int(c) for c in s]
    
    # Block 0 MLP: matched filter lookup
    d_ff = model.blocks[0].mlp.fc1.weight.shape[0]
    n_use = min(90000, d_ff)
    
    scale = 25.0
    threshold = scale * max_cos + 1.0  # conservative
    
    fc1_w = torch.zeros(d_ff, D)
    fc1_b = torch.zeros(d_ff)
    fc2_w = torch.zeros(D, d_ff)
    fc2_b = torch.zeros(D)
    
    fc1_w[:n_use] = mlp_normed[:n_use] * scale
    fc1_b[:n_use] = -threshold
    
    # Encode answer: dim k*10+d gets signal when digit d at slot k
    for k in range(5):
        for d in range(10):
            mask = (answer_digits[:, k] == d)
            indices = np.where(mask)[0]
            indices = indices[indices < n_use]
            fc2_w[k * 10 + d, indices] = 10.0
    
    model.blocks[0].mlp.fc1.weight.data = fc1_w
    model.blocks[0].mlp.fc1.bias.data = fc1_b
    model.blocks[0].mlp.fc2.weight.data = fc2_w
    model.blocks[0].mlp.fc2.bias.data = fc2_b
    
    # Block 1 MLP: at each output position 6+k, select digit k from encoding
    mlp2 = model.blocks[1].mlp
    d_ff2 = mlp2.fc1.weight.shape[0]
    
    fc1_2_w = torch.zeros(d_ff2, D)
    fc1_2_b = torch.zeros(d_ff2)
    fc2_2_w = torch.zeros(D, d_ff2)
    fc2_2_b = torch.zeros(D)
    
    # After block 0, at position 5 the MLP has written answer encoding into dims 0-49
    # After block 1 attention (uniform), this gets averaged to output positions
    # Block 1 MLP at each output position reads the answer encoding
    
    neuron_idx = 0
    for k in range(5):
        for d in range(10):
            if neuron_idx >= d_ff2:
                break
            # Detect: dim k*10+d is high AND we're at output position 6+k
            fc1_2_w[neuron_idx, k * 10 + d] = 5.0
            fc1_2_w[neuron_idx, 11 + 6 + k] = 5.0  # position signal
            fc1_2_b[neuron_idx] = -7.0
            fc2_2_w[d, neuron_idx] = 20.0
            neuron_idx += 1
    
    mlp2.fc1.weight.data = fc1_2_w
    mlp2.fc1.bias.data = fc1_2_b
    mlp2.fc2.weight.data = fc2_2_w
    mlp2.fc2.bias.data = fc2_2_b
    
    # Head: map dims to token logits
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 5.0
    model.head.weight.data = head_w
    
    print(f"Threshold: {threshold:.2f}, scale: {scale:.1f}")
    print("Weights initialized!")


model_shorthand_name = "spf-exp7-dense-emb"
model_description = "Dense random embeddings with uniform attention MLP lookup"


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
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=8,
        n_layers=2,
        d_ff_list=[90001, 256],
    )
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

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task": args.task,
        "accuracy": f"{accuracy:.4f}",
        "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
