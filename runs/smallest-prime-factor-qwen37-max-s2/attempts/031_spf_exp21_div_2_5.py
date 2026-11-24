"""
Interpretable transformer for smallest-prime-factor task.
Simple approach: check last digit for divisibility by 2, 3, 5.
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


def write_weights(model, task):
    """
    Strategy: 
    - The input is 5 digits at positions 0-4, '=' at position 5, then 5 output digits at positions 6-10.
    - Vocab: tokens 0-9 are digits, token 10 is '=' separator. V=11.
    - Check divisibility by 2 (last digit even), 3, 5 (last digit 0 or 5).
    - Output the SPF as 5-digit zero-padded number.
    
    Architecture: 1 layer, 1 head. D=32 to keep it simple.
    - Position embeddings are large (for routing).
    - Token embeddings encode digit identity.
    - Attention at output positions reads from input position 4 (last digit).
    - MLP detects divisibility and writes answer.
    """
    D = model.d_model  # 160
    V = model.vocab_size  # 11
    
    # Token embeddings: encode digit identity in first 11 dims
    tok_emb = torch.zeros(V, D)
    for i in range(V):
        tok_emb[i, i] = 50.0  # Large value for digit identity
    model.token_emb.weight.data = tok_emb
    
    # Position embeddings: encode position in dims 20-30
    pos_emb = torch.zeros(model.max_seq_len, D)
    for p in range(min(11, model.max_seq_len)):
        pos_emb[p, 20 + p] = 50.0
    model.pos_emb.weight.data = pos_emb
    
    # Zero all block weights
    for block in model.blocks:
        nn.init.zeros_(block.attn.W_q.weight)
        nn.init.zeros_(block.attn.W_k.weight)
        nn.init.zeros_(block.attn.W_v.weight)
        nn.init.zeros_(block.attn.W_o.weight)
        nn.init.zeros_(block.mlp.fc1.weight)
        nn.init.zeros_(block.mlp.fc1.bias)
        nn.init.zeros_(block.mlp.fc2.weight)
        nn.init.zeros_(block.mlp.fc2.bias)
    
    # === Block 0 Attention ===
    # Goal: At each output position (6-10), gather info from ALL input positions (0-4).
    # Use multiple heads: head 0 reads pos 0, head 1 reads pos 1, ..., head 4 reads pos 4.
    # At output positions, query uses position info to select which input position to attend to.
    
    attn0 = model.blocks[0].attn
    W_q = torch.zeros(D, D)
    W_k = torch.zeros(D, D)
    W_v = torch.zeros(D, D)
    W_o = torch.zeros(D, D)
    
    H = model.n_heads  # 5
    dh = D // H  # 32
    
    # Each head h: at any position, Q reads position dim (20+target_pos), K matches position dim (20+h)
    # Head h copies token from input position h to current position.
    for h_idx in range(H):
        hs = h_idx * dh  # head start in output dim
        
        # For heads 0-4: copy input position h_idx to output
        # Q: read position dim 20+h_idx (want to attend to position h_idx)
        W_q[hs, 20 + h_idx] = 10.0
        # K: key at position h_idx has large value in dim 20+h_idx (from pos_emb)
        W_k[hs, 20 + h_idx] = 10.0
        # V: read token identity from dims 0-10
        for d in range(min(V, dh)):
            W_v[hs + 1 + d, d] = 1.0
        # O: write to dedicated output dims
        for d in range(min(V + 1, dh)):
            W_o[h_idx * 12 + d, hs + d] = 1.0
    
    attn0.W_q.weight.data = W_q
    attn0.W_k.weight.data = W_k
    attn0.W_v.weight.data = W_v
    attn0.W_o.weight.data = W_o
    
    # After block 0 attention + residual:
    # Each position has its own token+pos embedding PLUS copies of tokens from positions 0-4.
    # dims 0-10: token identity at this position (from residual)
    # dims 20-30: position identity (from residual)
    # dims h*12 to h*12+10: token identity from input position h (from attention)
    
    # === Block 0 MLP ===
    # Detect divisibility by 2, 3, 5 using the last digit (position 4 = head 4).
    # Head 4 output is in dims 48-59 (4*12=48), with token identity in dims 48+1=49 to 48+11=59.
    # Actually let me recalculate: head 4 starts at hs=4*32=128.
    # V dims 128+1 to 128+11 (129-139) have token from pos 4.
    # O maps: W_o[4*12+d, 128+d] = 1.0, so output dims 48 to 48+10.
    # Token dim d from pos 4 goes to output dim 48+d.
    # So token 0 at pos 4 → dim 48 has value, token 5 at pos 4 → dim 53 has value.
    
    mlp0 = model.blocks[0].mlp
    fc1_w = torch.zeros(512, D)
    fc1_b = torch.zeros(512)
    fc2_w = torch.zeros(D, 512)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # Flag: divisible by 2 (last digit is 0,2,4,6,8)
    FLAG_DIV2 = 80
    for digit in [0, 2, 4, 6, 8]:
        fc1_w[neuron, 48 + digit] = 10.0
        fc1_b[neuron] = -5.0
        fc2_w[FLAG_DIV2, neuron] = 50.0
        neuron += 1
    
    # Flag: divisible by 5 (last digit is 0 or 5)
    FLAG_DIV5 = 81
    for digit in [0, 5]:
        fc1_w[neuron, 48 + digit] = 10.0
        fc1_b[neuron] = -5.0
        fc2_w[FLAG_DIV5, neuron] = 50.0
        neuron += 1
    
    mlp0.fc1.weight.data = fc1_w
    mlp0.fc1.bias.data = fc1_b
    mlp0.fc2.weight.data = fc2_w
    mlp0.fc2.bias.data = fc2_b
    
    # === Block 1 Attention === (zero = identity)
    # === Block 1 MLP ===
    # Use flags to write the answer at output positions.
    # If div2: write "00002" → at positions 6-10, write digits 0,0,0,0,2
    # If div5 and not div2: write "00005"
    # Otherwise: write "00003" as a fallback (or copy input for primes)
    
    # The problem: MLP at position p doesn't know what position p is.
    # We need position info. Position p is encoded in dim 20+p (from pos_emb residual).
    
    mlp1 = model.blocks[1].mlp
    fc1_w = torch.zeros(512, D)
    fc1_b = torch.zeros(512)
    fc2_w = torch.zeros(D, 512)
    fc2_b = torch.zeros(D)
    
    neuron = 0
    
    # For output positions 5-9:
    # If div2 flag is set AND we're at position p:
    #   - positions 5,6,7,8: output digit 0
    #   - position 9: output digit 2
    
    # Div2 + position 5-8 → digit 0
    for out_pos in [5, 6, 7, 8]:
        fc1_w[neuron, FLAG_DIV2] = 5.0
        fc1_w[neuron, 20 + out_pos] = 5.0
        fc1_b[neuron] = -7.5
        fc2_w[0, neuron] = 50.0  # boost digit 0
        neuron += 1
    
    # Div2 + position 9 → digit 2
    fc1_w[neuron, FLAG_DIV2] = 5.0
    fc1_w[neuron, 20 + 9] = 5.0
    fc1_b[neuron] = -7.5
    fc2_w[2, neuron] = 50.0  # boost digit 2
    neuron += 1
    
    # Div5 (not div2) + position 5-8 → digit 0
    for out_pos in [5, 6, 7, 8]:
        fc1_w[neuron, FLAG_DIV5] = 5.0
        fc1_w[neuron, FLAG_DIV2] = -50.0  # inhibit if div2
        fc1_w[neuron, 20 + out_pos] = 5.0
        fc1_b[neuron] = -7.5
        fc2_w[0, neuron] = 50.0  # boost digit 0
        neuron += 1
    
    # Div5 (not div2) + position 9 → digit 5
    fc1_w[neuron, FLAG_DIV5] = 5.0
    fc1_w[neuron, FLAG_DIV2] = -50.0
    fc1_w[neuron, 20 + 9] = 5.0
    fc1_b[neuron] = -7.5
    fc2_w[5, neuron] = 50.0  # boost digit 5
    neuron += 1
    
    mlp1.fc1.weight.data = fc1_w
    mlp1.fc1.bias.data = fc1_b
    mlp1.fc2.weight.data = fc2_w
    mlp1.fc2.bias.data = fc2_b
    
    # Final head: read from dims 0-10 (token identity from residual)
    head_w = torch.zeros(V, D)
    for d in range(V):
        head_w[d, d] = 10.0
    model.head.weight.data = head_w
    
    print(f"Weights initialized with {neuron} neurons!")


model_shorthand_name = "spf-exp21-div-2-5"
model_description = "Check divisibility by 2 and 5 using last digit, output 00002 or 00005"


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


def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=160,
        n_heads=5,
        n_layers=2,
        d_ff_list=[512, 512],
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
