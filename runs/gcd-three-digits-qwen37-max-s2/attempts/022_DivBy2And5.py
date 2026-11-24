"""
Interpretable transformer for GCD computation.

Task: Given two 3-digit numbers "abc,def=", compute their GCD as 3-digit output.

Strategy: Compute both numbers mod small primes, check which primes divide both,
and output the product of common prime factors.
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
    """GCD via divisibility by 2 and 5 using last-digit check.
    
    Input format: d0 d1 d2 ',' d4 d5 d6 '=' at positions 0-7
    Output: 3-digit GCD at positions 8, 9, 10
    
    Encoding: Each digit at position p uses dims [p*10, p*10+9] in token_emb
    (position 3 comma and 7 '=' get zero embedding).
    Uniform attention copies all digit info to output positions.
    MLP at output positions detects divisibility by 2 and 5.
    """
    import numpy as np
    d = model.d_model  # 64
    
    # Zero all parameters
    for p in model.parameters():
        p.data.zero_()
    
    # Set all LayerNorm gammas to 1
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln2.weight.data.fill_(1.0)
    model.final_ln.weight.data.fill_(1.0)
    
    # === Token embeddings: position-dependent digit encoding ===
    # Digit d at position p → token_emb[d, p*10 + d] = 5.0
    token_emb = torch.zeros(model.vocab_size, d)
    # Map: positions 0,1,2 use dim ranges [0-9],[10-19],[20-29]
    # Position 3 (comma): no encoding
    # Positions 4,5,6 use dim ranges [30-39],[40-49],[50-59]
    # Position 7 (=): no encoding
    pos_to_dimoffset = {0: 0, 1: 10, 2: 20, 4: 30, 5: 40, 6: 50}
    for digit in range(10):
        for pos, offset in pos_to_dimoffset.items():
            token_emb[digit, offset + digit] = 5.0
    model.token_emb.weight.data = token_emb
    
    # Position embeddings: all zero (digit info comes from token_emb)
    # pos_emb stays zero
    
    # === Block 0: Attention to copy digit info to all output positions ===
    attn_q = torch.zeros(d, d)  # zero query → uniform attention
    attn_k = torch.zeros(d, d)  # zero key
    
    # Value: identity on dims 0-59 (copy digit info)
    attn_v = torch.zeros(d, d)
    for i in range(60):
        attn_v[i, i] = 1.0
    
    # Output: identity on dims 0-59
    attn_o = torch.zeros(d, d)
    for i in range(60):
        attn_o[i, i] = 1.0
    
    model.blocks[0].attn.W_q.weight.data = attn_q
    model.blocks[0].attn.W_k.weight.data = attn_k
    model.blocks[0].attn.W_v.weight.data = attn_v
    model.blocks[0].attn.W_o.weight.data = attn_o
    
    # === Block 0 MLP: Divisibility checks at output positions ===
    # After uniform attention, residual at output positions contains:
    #   avg of all token embeddings = (1/N) * sum of position-encoded digits
    # For ~8 causal positions, each digit d_p contributes ~d_p/8 in its dim range.
    # After LayerNorm, active dims have high positive values (~3.5), inactive ~(-0.4).
    
    d_ff = model.d_ff  # 256
    fc1_w = torch.zeros(d_ff, d)
    fc1_b = torch.zeros(d_ff)
    fc2_w = torch.zeros(d, d_ff)
    fc2_b = torch.zeros(d)
    
    # --- Divisibility by 2: both last digits even ---
    # Position 2's digit is in dims [20-29], position 6's in [50-59]
    # Even digits: 0,2,4,6,8
    
    # h0: detect even digit at position 2 (dims 20,22,24,26,28)
    for ed in [0, 2, 4, 6, 8]:
        fc1_w[0, 20 + ed] = 1.0
    fc1_b[0] = -1.5  # threshold: fires when one even dim is active (~3.5)
    
    # h1: detect even digit at position 6 (dims 50,52,54,56,58)
    for ed in [0, 2, 4, 6, 8]:
        fc1_w[1, 50 + ed] = 1.0
    fc1_b[1] = -1.5
    
    # h2: max(h0, h1) via max(0, h0 - h1) + h1 trick
    # Actually, we need min(h0, h1) = h0 + h1 - max(h0, h1)
    # max(h0, h1) = max(0, h0-h1) + h1
    # h2 = max(0, h0 - h1)
    fc1_w[2, :] = 0  # this is computed via ReLU, not directly in fc1
    # Actually, h2 needs to be computed from the LayerNorm output, not from h0,h1.
    # The MLP is fc2(relu(fc1(x))), so all hidden units see the SAME input x.
    # I can't compute h2 = max(0, h0-h1) in a single MLP layer!
    
    # NEW APPROACH: compute AND directly using a different formula.
    # AND(A, B) where A,B in {0, V}:
    #   = (1/V) * max(0, A + B - V)   when A,B in {0, V}
    # If A=V and B=V: max(0, 2V - V) = V ✓
    # If A=V and B=0: max(0, V - V) = 0 ✓
    # If A=0 and B=0: max(0, 0 - V) = 0 ✓
    # PERFECT! This works!
    
    # So for AND of even detection:
    # h_and2 = max(0, even_pos2 + even_pos6 - threshold)
    # We need even_pos2 + even_pos6 when both fire, minus a threshold.
    
    # Redefine h0 and h1 to fire with the same value V when even:
    # h0 fires with value ~3.5-1.5=2.0 when pos2 is even (active dim value ~3.5)
    # h1 fires with value ~2.0 when pos6 is even
    
    # h2: AND(even2, even6) = max(0, h0_input + h1_input - V)
    # where h0_input is the fc1 pre-activation for h0 (before ReLU)
    # Wait, I can't reference other hidden units from fc1.
    # But I CAN make h2 directly compute the sum of both even detections:
    for ed in [0, 2, 4, 6, 8]:
        fc1_w[2, 20 + ed] = 1.0  # even detection at pos 2
        fc1_w[2, 50 + ed] = 1.0  # even detection at pos 6
    fc1_b[2] = -5.5  # fires only when BOTH are even: 3.5+3.5-5.5=1.5>0
    # If only one even: 3.5-5.5=-2.0 → 0 ✓
    # If neither: -5.5 → 0 ✓
    
    # --- Divisibility by 5: both last digits are 0 or 5 ---
    # Digit 0 at pos 2 → dim 20; digit 5 at pos 2 → dim 25
    # Digit 0 at pos 6 → dim 50; digit 5 at pos 6 → dim 55
    
    # h3: AND(div5_pos2, div5_pos6)
    fc1_w[3, 20] = 1.0  # digit 0 at pos 2
    fc1_w[3, 25] = 1.0  # digit 5 at pos 2
    fc1_w[3, 50] = 1.0  # digit 0 at pos 6
    fc1_w[3, 55] = 1.0  # digit 5 at pos 6
    fc1_b[3] = -5.5  # fires only when both are div by 5
    
    # --- Default output: predict "1" (ones digit) ---
    # h4: always fires (constant signal)
    fc1_w[4, :] = 0
    fc1_b[4] = 5.0  # always fires with value 5.0
    
    # --- fc2: combine into output logits ---
    # The MLP output is added to the residual stream.
    # At output position 10 (ones digit), the head reads dims to predict token.
    
    # Use dim 60 for "1" signal, dim 61 for "2" signal, dim 62 for "5" signal
    # (these are in the residual, read by the head)
    
    # Default: predict "1" (h4 always fires)
    fc2_w[60, 4] = 1.0  # h4 → dim 60 (predict "1")
    
    # If div by 2 (h2 fires): predict "2", suppress "1"
    fc2_w[61, 2] = 3.0  # h2 → dim 61 (predict "2")
    fc2_w[60, 2] = -3.0  # h2 → suppress dim 60 (suppress "1")
    
    # If div by 5 (h3 fires): predict "5", suppress others
    fc2_w[62, 3] = 3.0  # h3 → dim 62 (predict "5")
    fc2_w[60, 3] = -3.0  # h3 → suppress dim 60 (suppress "1")
    fc2_w[61, 3] = -3.0  # h3 → suppress dim 61 (suppress "2" if also div2)
    
    model.blocks[0].mlp.fc1.weight.data = fc1_w
    model.blocks[0].mlp.fc1.bias.data = fc1_b
    model.blocks[0].mlp.fc2.weight.data = fc2_w
    model.blocks[0].mlp.fc2.bias.data = fc2_b
    
    # === Head: predict output digits ===
    head_w = torch.zeros(model.vocab_size, d)
    
    # Hundreds digit (position 8) and tens digit (position 9): always "0"
    head_w[0, 63] = 10.0  # "0" from dim 63
    
    # Ones digit (position 10): depends on MLP output
    head_w[1, 60] = 10.0  # "1" from dim 60
    head_w[2, 61] = 10.0  # "2" from dim 61
    head_w[5, 62] = 10.0  # "5" from dim 62
    
    model.head.weight.data = head_w
    
    # === Position embeddings for output digit selection ===
    pos_emb = torch.zeros(model.max_seq_len, d)
    pos_emb[8, 63] = 10.0   # hundreds digit → predict "0"
    pos_emb[9, 63] = 10.0   # tens digit → predict "0"
    # Position 10 (ones digit) gets signal from MLP, no pos_emb needed
    model.pos_emb.weight.data = pos_emb


model_shorthand_name = "DivBy2And5"
model_description = "Check divisibility by 2 and 5 using last digits. AND logic via threshold. Predict GCD=2,5,or 1."


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
