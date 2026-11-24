"""
Interpretable transformer for character-level NLI.
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
    def __init__(self, vocab_size, max_seq_len=128, d_model=128, n_heads=4, n_layers=2, d_ff=512):
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
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task) -> None:
    """
    2-layer NLI circuit:
    
    Layer 1: Letter frequency collection
    - Head 0: Equals attends uniformly to premise, collects letter frequencies → dims 40-65
    - Head 1: Equals attends uniformly to hypothesis, collects letter frequencies → dims 70-95
    - Head 2: Equals attends to hypothesis space count → dim 96
    - Head 3: Equals attends to premise space count → dim 97
    Layer 1 MLP: Compute match/mismatch + word count features → dims 100-105
    
    Layer 2: Classification (pass-through attention, MLP classifies)
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 128
        H = model.n_heads  # 4
        dh = d // H  # 32
        
        for name, param in model.named_parameters():
            param.zero_()
        
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln2.weight.fill_(1.0)
        model.final_ln.weight.fill_(1.0)
        
        # EMBEDDINGS
        token_emb = torch.zeros(model.vocab_size, d)
        for i in range(26):
            token_emb[i, i] = 3.0
        token_emb[26, 26] = 3.0  # space
        token_emb[27, 27] = 3.0  # |
        token_emb[28, 28] = 3.0  # =
        model.token_emb.weight.copy_(token_emb)
        
        pos_emb = torch.zeros(model.max_seq_len, d)
        for i in range(model.max_seq_len):
            if i < 60:
                pos_emb[i, 30] = 5.0  # premise
            elif i == 60:
                pos_emb[i, 31] = 5.0  # separator
            elif i < 101:
                pos_emb[i, 32] = 5.0  # hypothesis
            elif i == 101:
                pos_emb[i, 33] = 5.0  # equals
        model.pos_emb.weight.copy_(pos_emb)
        
        # === LAYER 1: Letter freq collection at equals position ===
        block0 = model.blocks[0]
        attn0 = block0.attn
        
        W_q = torch.zeros(d, d)
        W_k = torch.zeros(d, d)
        W_v = torch.zeros(d, d)
        W_o = torch.zeros(d, d)
        
        # Head 0: Equals→Premise letter frequencies
        # Q from equals marker, K from premise region (small weights for uniform attn)
        W_q[0, 33] = 0.1
        W_k[0, 30] = 0.1
        # V: letter identity
        for c in range(26):
            W_v[c, c] = 1.0
        # W_o: output to dims 40-65
        for c in range(26):
            W_o[40 + c, c] = 3.0
        
        # Head 1: Equals→Hypothesis letter frequencies
        W_q[dh, 33] = 0.1
        W_k[dh, 32] = 0.1
        for c in range(26):
            W_v[dh + c, c] = 1.0
        for c in range(26):
            W_o[70 + c, dh + c] = 3.0
        
        # Head 2: Equals→Hypothesis space count (word count)
        W_q[2*dh, 33] = 0.1
        W_k[2*dh, 32] = 0.1
        W_v[2*dh + 26, 26] = 1.0  # space token
        W_o[96, 2*dh + 26] = 5.0
        
        # Head 3: Equals→Premise space count
        W_q[3*dh, 33] = 0.1
        W_k[3*dh, 30] = 0.1
        W_v[3*dh + 26, 26] = 1.0
        W_o[97, 3*dh + 26] = 5.0
        
        attn0.W_q.weight.copy_(W_q)
        attn0.W_k.weight.copy_(W_k)
        attn0.W_v.weight.copy_(W_v)
        attn0.W_o.weight.copy_(W_o)
        
        # Layer 1 MLP: Match/mismatch + features
        mlp0 = block0.mlp
        W1 = torch.zeros(model.d_ff, d)
        b1 = torch.zeros(model.d_ff)
        W2 = torch.zeros(d, model.d_ff)
        b2 = torch.zeros(d)
        
        # Per-letter match: both hyp(c) and prem(c) present
        for c in range(26):
            W1[c, 70 + c] = 2.0   # hyp letter c freq
            W1[c, 40 + c] = 2.0   # prem letter c freq
            b1[c] = -3.0           # need both present
            
            # Mismatch: hyp has letter c but premise doesn't
            W1[26 + c, 70 + c] = 2.0   # hyp letter c
            W1[26 + c, 40 + c] = -2.0  # prem letter c (absent)
            b1[26 + c] = 0.0
        
        # Word count features
        # More hyp words than premise → possible contradiction/neutral
        W1[52, 96] = 1.0
        W1[52, 97] = -1.0
        b1[52] = 0.0
        # Fewer hyp words → possible entailment
        W1[53, 97] = 1.0
        W1[53, 96] = -1.0
        b1[53] = 0.0
        
        # Aggregate
        for c in range(26):
            W2[100, c] = 2.0       # total match
            W2[101, 26 + c] = 2.0  # total mismatch
        W2[102, 52] = 3.0  # more hyp words
        W2[103, 53] = 3.0  # fewer hyp words
        
        mlp0.fc1.weight.copy_(W1)
        mlp0.fc1.bias.copy_(b1)
        mlp0.fc2.weight.copy_(W2)
        mlp0.fc2.bias.copy_(b2)
        
        # === LAYER 2: Classification ===
        block1 = model.blocks[1]
        attn1 = block1.attn
        
        # Pass-through attention
        W_q1 = torch.zeros(d, d)
        W_k1 = torch.zeros(d, d)
        W_v1 = torch.zeros(d, d)
        W_o1 = torch.zeros(d, d)
        for i in range(min(d, dh)):
            W_q1[i, i] = 0.3
            W_k1[i, i] = 0.3
            W_v1[i, i] = 1.0
            W_o1[i, i] = 1.0
        attn1.W_q.weight.copy_(W_q1)
        attn1.W_k.weight.copy_(W_k1)
        attn1.W_v.weight.copy_(W_v1)
        attn1.W_o.weight.copy_(W_o1)
        
        mlp1 = block1.mlp
        W1_l2 = torch.zeros(model.d_ff, d)
        b1_l2 = torch.zeros(model.d_ff)
        W2_l2 = torch.zeros(d, model.d_ff)
        b2_l2 = torch.zeros(d)
        
        # Entailment: high match, low mismatch, fewer hyp words
        W1_l2[0, 100] = 1.5
        W1_l2[0, 101] = -2.0
        W1_l2[0, 103] = 0.5  # fewer hyp words
        b1_l2[0] = -1.0
        
        # Contradiction: high mismatch, more hyp words
        W1_l2[1, 101] = 2.0
        W1_l2[1, 100] = -1.5
        W1_l2[1, 102] = 0.5
        b1_l2[1] = -0.5
        
        # Neutral: moderate match and mismatch
        W1_l2[2, 100] = 0.5
        W1_l2[2, 101] = 0.5
        b1_l2[2] = -1.5
        
        W2_l2[110, 0] = 5.0  # entailment
        W2_l2[111, 1] = 5.0  # contradiction
        W2_l2[112, 2] = 3.0  # neutral
        
        mlp1.fc1.weight.copy_(W1_l2)
        mlp1.fc1.bias.copy_(b1_l2)
        mlp1.fc2.weight.copy_(W2_l2)
        mlp1.fc2.bias.copy_(b2_l2)
        
        # FINAL HEAD
        W_head = torch.zeros(model.vocab_size, d)
        b_head = torch.zeros(model.vocab_size)
        b_head.fill_(-20.0)
        b_head[32] = 0.0
        
        W_head[31, 110] = 15.0
        W_head[32, 110] = -10.0
        W_head[33, 111] = 15.0
        W_head[32, 111] = -10.0
        W_head[32, 112] = 8.0
        
        model.head.weight.copy_(W_head)
        model.head.bias.copy_(b_head)


model_shorthand_name = "CalibratedOverlap_v18"
model_description = "Very small Q/K (0.1) for uniform attention. Letter freq collection + word counts + calibrated MLP thresholds for classification."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    
    if args.verbose:
        examples = task.generate_examples(5)
        for i, ex in enumerate(examples):
            print(f"Ex {i}: ans={ex.answer} | {ex.prompt}")
    
    model = build_model(task).to(args.device)
    
    if args.verbose:
        print("\n=== PREDICTIONS ===")
        examples = task.generate_examples(10)
        for i, ex in enumerate(examples):
            prompt_ids = task.encode(ex.prompt)
            with torch.no_grad():
                logits = model(torch.tensor([prompt_ids], device=args.device))
                pred = logits[0, -1].argmax().item()
                probs = torch.softmax(logits[0, -1], dim=-1)
            print(f"Ex {i}: Answer={ex.answer}, Pred={task.decode([pred])}, Probs[0,1,2]={probs[31:34].tolist()}")
        sys.exit(0)

    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
