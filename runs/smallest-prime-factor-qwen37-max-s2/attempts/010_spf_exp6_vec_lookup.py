"""
Interpretable transformer for smallest-prime-factor task.
Uses MLP as a lookup table keyed by digit-position encoding.
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
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff_list: list = None,
    ):
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def sieve_spf(limit):
    """Compute smallest prime factor for all numbers up to limit using sieve."""
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


def write_weights(model: SimpleTransformer, task) -> None:
    """Build a direct lookup using large MLP as memory."""
    torch.manual_seed(42)
    
    D = model.d_model
    V = model.vocab_size
    
    # Compute SPF using sieve
    print("Computing SPF sieve...")
    spf = sieve_spf(99999)
    
    # Build answer table: for each number, the 5 output digits
    print("Building answer table...")
    answer_digits = np.zeros((90000, 5), dtype=np.int64)
    for i, n in enumerate(range(10000, 100000)):
        s = str(spf[n]).zfill(5)
        answer_digits[i] = [int(c) for c in s]
    
    # Token embeddings: digits 0-9, token 10 is '='
    tok_emb = torch.zeros(V, D)
    for d in range(V):
        tok_emb[d, d] = 5.0
    
    # Position embeddings: dims 11+ for position identity
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        pos_emb[p, 11 + p] = 5.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb
    
    # Block 0: zero Q/K => uniform attention
    for block in model.blocks:
        nn.init.zeros_(block.attn.W_q.weight)
        nn.init.zeros_(block.attn.W_k.weight)
        
        W_v = torch.zeros(D, D)
        for i in range(min(22, D)):
            W_v[i, i] = 1.0
        block.attn.W_v.weight.data = W_v
        
        W_o = torch.zeros(D, D)
        for i in range(D):
            W_o[i, i] = 1.0
        block.attn.W_o.weight.data = W_o
    
    # Vectorized computation of MLP inputs at position 5
    print("Vectorized MLP input computation...")
    
    ln1_w = model.blocks[0].ln1.weight.data.clone()
    ln1_b = model.blocks[0].ln1.bias.data.clone()
    ln2_w = model.blocks[0].ln2.weight.data.clone()
    ln2_b = model.blocks[0].ln2.bias.data.clone()
    
    W_v_mat = model.blocks[0].attn.W_v.weight.data.clone()
    W_o_mat = model.blocks[0].attn.W_o.weight.data.clone()
    
    def apply_ln_batch(x, w, b):
        """x: (N, D), w: (D,), b: (D,) -> (N, D)"""
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return w * (x - mean) / (var + 1e-5).sqrt() + b
    
    # All 90000 numbers as digit arrays
    numbers = np.arange(10000, 100000)
    digits_all = np.zeros((90000, 5), dtype=np.int64)
    for p in range(5):
        digits_all[:, p] = (numbers // (10 ** (4 - p))) % 10
    
    # For each position p (0-5), compute h_p = tok_emb[digit] + pos_emb[p]
    # Then LN1, then V projection, sum them up (uniform attention, divide by 6)
    
    # Position 5 is '=' token (id=10) for all numbers
    h5 = tok_emb[10] + pos_emb[5]  # (D,)
    
    # Compute LN1 of h5
    ln1_h5 = apply_ln_batch(h5.unsqueeze(0), ln1_w, ln1_b).squeeze(0)  # (D,)
    v_h5 = W_v_mat @ ln1_h5  # (D,)
    
    attn_out = v_h5.unsqueeze(0).expand(90000, -1).clone() / 6.0  # (N, D)
    
    # For positions 0-4, compute contribution for each number
    for p in range(5):
        digit_ids = digits_all[:, p]  # (N,)
        h_p = tok_emb[digit_ids] + pos_emb[p]  # (N, D)
        ln1_hp = apply_ln_batch(h_p, ln1_w, ln1_b)  # (N, D)
        v_hp = ln1_hp @ W_v_mat.T  # (N, D)
        attn_out += v_hp / 6.0
    
    # Residual
    h_after_attn = (tok_emb[10] + pos_emb[5]).unsqueeze(0) + attn_out @ W_o_mat.T  # (N, D)
    
    # LN2
    mlp_inputs = apply_ln_batch(h_after_attn, ln2_w, ln2_b)  # (N, D)
    
    # Normalize for matched filter
    norms = mlp_inputs.norm(dim=1, keepdim=True)
    mlp_normed = mlp_inputs / (norms + 1e-8)
    
    # Sample to check max cosine similarity (full matrix too large)
    idx = torch.randperm(90000)[:1000]
    sample_normed = mlp_normed[idx]
    cos_sims = sample_normed @ sample_normed.T
    cos_sims.fill_diagonal_(-1)
    max_cos = cos_sims.max().item()
    print(f"Max cosine similarity (sampled): {max_cos:.4f}")
    
    d_ff = model.blocks[0].mlp.fc1.weight.shape[0]
    
    scale = 20.0
    threshold = scale * (max_cos + 1.0) / 2.0 + 2.0
    
    fc1_w = torch.zeros(d_ff, D)
    fc1_b = torch.zeros(d_ff)
    fc2_w = torch.zeros(D, d_ff)
    fc2_b = torch.zeros(D)
    
    n_use = min(90000, d_ff)
    
    # Set matched filters in batch
    fc1_w[:n_use] = mlp_normed[:n_use] * scale
    fc1_b[:n_use] = -threshold
    
    # fc2: for each number i, output digit d at slot k => fc2_w[k*10+d, i] = 10
    for k in range(5):
        for d in range(10):
            mask = (answer_digits[:, k] == d)  # (N,) bool
            indices = np.where(mask)[0]
            indices = indices[indices < n_use]
            fc2_w[k * 10 + d, indices] = 10.0
    
    model.blocks[0].mlp.fc1.weight.data = fc1_w
    model.blocks[0].mlp.fc1.bias.data = fc1_b
    model.blocks[0].mlp.fc2.weight.data = fc2_w
    model.blocks[0].mlp.fc2.bias.data = fc2_b
    
    # Block 1: zero attention (uniform) + MLP to select digit per output position
    nn.init.zeros_(model.blocks[1].attn.W_q.weight)
    nn.init.zeros_(model.blocks[1].attn.W_k.weight)
    
    W_v2 = torch.zeros(D, D)
    for i in range(min(50, D)):
        W_v2[i, i] = 1.0
    model.blocks[1].attn.W_v.weight.data = W_v2
    
    W_o2 = torch.zeros(D, D)
    for i in range(D):
        W_o2[i, i] = 1.0
    model.blocks[1].attn.W_o.weight.data = W_o2
    
    # Block 1 MLP: select digit for output position
    mlp2 = model.blocks[1].mlp
    d_ff2 = mlp2.fc1.weight.shape[0]
    
    fc1_2_w = torch.zeros(d_ff2, D)
    fc1_2_b = torch.zeros(d_ff2)
    fc2_2_w = torch.zeros(D, d_ff2)
    fc2_2_b = torch.zeros(D)
    
    neuron_idx = 0
    for k in range(5):
        for d in range(10):
            if neuron_idx >= d_ff2:
                break
            fc1_2_w[neuron_idx, k * 10 + d] = 5.0
            fc1_2_w[neuron_idx, 11 + 6 + k] = 5.0
            fc1_2_b[neuron_idx] = -7.0
            fc2_2_w[d, neuron_idx] = 20.0
            neuron_idx += 1
    
    mlp2.fc1.weight.data = fc1_2_w
    mlp2.fc1.bias.data = fc1_2_b
    mlp2.fc2.weight.data = fc2_2_w
    mlp2.fc2.bias.data = fc2_2_b
    
    # Head
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 5.0
    model.head.weight.data = head_w
    
    print("Weights initialized!")
    return


model_shorthand_name = "spf-exp6-vec-lookup"
model_description = "Vectorized MLP lookup with matched-filter neurons"


# ---------------------------------------------------------------------------
# Evaluation harness
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
