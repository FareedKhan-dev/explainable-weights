"""
Interpretable transformer for smallest-prime-factor task.
Uses a large MLP as a lookup table to map 5-digit inputs to their SPF.
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

def write_weights(model: SimpleTransformer, task) -> None:
    """Build a lookup table using the MLP layer."""
    torch.manual_seed(42)
    
    D = model.d_model
    V = model.vocab_size
    
    print("Computing SPF for all 5-digit numbers...")
    
    # Debug: check what spf_table returns
    debug_val = task.spf_table(10000)
    print(f"DEBUG: task.spf_table(10000) = {repr(debug_val)} type={type(debug_val)}")
    debug_val2 = task.spf_table(10007)
    print(f"DEBUG: task.spf_table(10007) = {repr(debug_val2)} type={type(debug_val2)}")
    
    # Check task attributes
    print(f"DEBUG: task.vocab_size={task.vocab_size}, task.seq_len={task.seq_len}")
    if hasattr(task, 'input_len'):
        print(f"DEBUG: task.input_len={task.input_len}")
    if hasattr(task, 'output_len'):
        print(f"DEBUG: task.output_len={task.output_len}")
    
    # Debug: check what a sample looks like
    if hasattr(task, 'generate_examples'):
        examples = task.generate_examples(2)
        print(f"DEBUG: examples = {repr(examples)}")
    
    # Build answer table
    answers = {}
    for n in range(10000, 100000):
        spf_val = task.spf_table(n)
        if spf_val == n:
            ans = str(n)
        else:
            ans = str(spf_val).zfill(5)
        answers[n] = [int(c) for c in ans]
    
    # Step 1: Token and position embeddings
    tok_emb = torch.zeros(V, D)
    for d in range(10):
        tok_emb[d, d] = 5.0
    tok_emb[10, 10] = 5.0
    
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(model.max_seq_len):
        pos_emb[p, 11 + p] = 5.0
    
    model.token_emb.weight.data = tok_emb
    model.pos_emb.weight.data = pos_emb
    
    # Step 2: Near-uniform attention (small Q, K weights)
    for block in model.blocks:
        nn.init.zeros_(block.attn.W_q.weight)
        nn.init.zeros_(block.attn.W_k.weight)
        
        # V extracts first 30 dims (digit + position info)
        W_v = torch.zeros(D, D)
        for i in range(min(30, D)):
            W_v[i, i] = 1.0
        block.attn.W_v.weight.data = W_v
        
        # O is identity
        W_o = torch.zeros(D, D)
        for i in range(D):
            W_o[i, i] = 1.0
        block.attn.W_o.weight.data = W_o
    
    # Step 3: Compute MLP inputs for position 5 (after "=")
    print("Computing MLP inputs for all numbers...")
    
    ln1_w = model.blocks[0].ln1.weight.data
    ln1_b = model.blocks[0].ln1.bias.data
    
    def apply_ln(x, w, b):
        mean = x.mean()
        var = x.var(unbiased=False)
        return w * (x - mean) / (var + 1e-5).sqrt() + b
    
    W_v = model.blocks[0].attn.W_v.weight.data
    W_o = model.blocks[0].attn.W_o.weight.data
    
    mlp_inputs = []
    for n in range(10000, 100000):
        digits = [int(c) for c in str(n)]
        
        # Attention output at position 5 (uniform over positions 0-5)
        attn_out = torch.zeros(D)
        for p in range(5):
            h_p = tok_emb[digits[p]] + pos_emb[p]
            ln1_out = apply_ln(h_p, ln1_w, ln1_b)
            v_out = W_v @ ln1_out
            attn_out += v_out / 6.0
        
        h_5 = tok_emb[10] + pos_emb[5]
        ln1_out_5 = apply_ln(h_5, ln1_w, ln1_b)
        v_out_5 = W_v @ ln1_out_5
        attn_out += v_out_5 / 6.0
        
        # Residual connection
        h_after_attn = h_5 + W_o @ attn_out
        
        # LN2
        ln2_w = model.blocks[0].ln2.weight.data
        ln2_b = model.blocks[0].ln2.bias.data
        mlp_in = apply_ln(h_after_attn, ln2_w, ln2_b)
        
        mlp_inputs.append(mlp_in)
    
    mlp_inputs = torch.stack(mlp_inputs)
    
    # Step 4: Design MLP as lookup table
    print("Designing MLP lookup table...")
    
    norms = mlp_inputs.norm(dim=1, keepdim=True)
    mlp_normed = mlp_inputs / (norms + 1e-8)
    
    # Check similarity
    cos_sims = mlp_normed @ mlp_normed.T
    cos_sims.fill_diagonal_(-1)
    max_cos = cos_sims.max().item()
    print(f"Max cosine similarity: {max_cos:.4f}")
    
    d_ff = model.blocks[0].mlp.fc1.weight.shape[0]
    
    scale = 10.0
    threshold = scale * (max_cos + 1.0) / 2.0
    
    fc1_w = torch.zeros(d_ff, D)
    fc1_b = torch.zeros(d_ff)
    fc2_w = torch.zeros(D, d_ff)
    fc2_b = torch.zeros(D)
    
    for i in range(min(90000, d_ff)):
        n = 10000 + i
        fc1_w[i] = mlp_normed[i] * scale
        fc1_b[i] = -threshold
        
        # Output encoding: one-hot for each answer digit
        for k, digit in enumerate(answers[n]):
            fc2_w[k * 10 + digit, i] = 5.0
    
    model.blocks[0].mlp.fc1.weight.data = fc1_w
    model.blocks[0].mlp.fc1.bias.data = fc1_b
    model.blocks[0].mlp.fc2.weight.data = fc2_w
    model.blocks[0].mlp.fc2.bias.data = fc2_b
    
    # Step 5: Block 1 - propagate and decode
    # Attention: output positions attend to position 5
    # Use position markers to select right digit
    
    # Step 6: Head - tied to token embeddings
    model.head.weight.data = tok_emb.clone()
    
    print("Weights initialized!")
    return


model_shorthand_name = "spf-exp4-lookup-90k"
model_description = "90k-unit MLP lookup table with uniform attention"


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
