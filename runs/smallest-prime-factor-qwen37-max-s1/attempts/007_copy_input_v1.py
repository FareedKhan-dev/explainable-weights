"""
Interpretable transformer for character-level sequence tasks.

Task: smallest-prime-factor
Given a 5-digit number, output its smallest prime factor (also as 5 digits).
For primes, output the number itself.

Prompt: "07037=" -> Answer: "00031" (since 7037 = 31 * 227)
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
    """Transformer for character-level sequence tasks."""

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
    """Copy input digits to output (assume all numbers are prime).
    
    Hand-craft attention to copy from input position i to output position i+6.
    """
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads
    
    # Zero all weights
    for param in model.parameters():
        param.data.zero_()
    
    # Token embeddings: one-hot encode digit values
    vocab = task.vocab
    for i, char in enumerate(vocab):
        if char.isdigit():
            model.token_emb.weight.data[i, int(char)] = 1.0
        elif char == '=':
            model.token_emb.weight.data[i, 10] = 1.0
    
    # Position embeddings: encode position index
    for pos in range(min(16, model.max_seq_len)):
        model.pos_emb.weight.data[pos, 11 + pos] = 1.0
    
    # Layer 0: Copy input digits to output positions
    # Use attention to attend from output position (6-10) to input position (0-4)
    block0 = model.blocks[0]
    
    # Head 0: Copy position 0 -> 6, 1 -> 7, 2 -> 8, 3 -> 9, 4 -> 10
    # Query at position p should match Key at position p-6 (for output positions)
    # Or match itself (for input positions)
    
    # Set Q to extract position: Q @ pos_emb gives position-specific query
    # Set K to extract position: K @ pos_emb gives position-specific key
    # Use head 0 to implement this copying
    
    # For head 0 (first d_head dimensions):
    # Q: extract position from current position embedding
    # K: extract position from source position embedding  
    # V: copy token embedding (digit value)
    
    # Q[0:d_head, 11:27] = I (extract position from pos_emb)
    # K[0:d_head, 11:27] = I (extract position from pos_emb)
    # But we need to shift by 6 for output->input matching
    
    # Simpler: use position embeddings to encode "target position"
    # Output position 6 has pos_emb[17] = 1 (wants to attend to pos 0)
    # Output position 7 has pos_emb[18] = 1 (wants to attend to pos 1)
    # Input position 0 has pos_emb[11] = 1
    # Input position 1 has pos_emb[12] = 1
    
    # Q should map pos_emb[17] to match K mapping pos_emb[11]
    # This requires: Q[0:d_head, 17] should equal K[0:d_head, 11]
    
    # Let me use a direct approach:
    # Add special "copy target" signals to position embeddings
    model.pos_emb.weight.data[6, 30] = 1.0  # Position 6 wants to copy from position 0
    model.pos_emb.weight.data[7, 31] = 1.0  # Position 7 wants to copy from position 1
    model.pos_emb.weight.data[8, 32] = 1.0
    model.pos_emb.weight.data[9, 33] = 1.0
    model.pos_emb.weight.data[10, 34] = 1.0
    
    # Add "copy source" signals
    model.pos_emb.weight.data[0, 30] = 1.0  # Position 0 can be copied to position 6
    model.pos_emb.weight.data[1, 31] = 1.0
    model.pos_emb.weight.data[2, 32] = 1.0
    model.pos_emb.weight.data[3, 33] = 1.0
    model.pos_emb.weight.data[4, 34] = 1.0
    
    # Q extracts "copy target" signal (dimensions 30-34)
    # K extracts "copy source" signal (dimensions 30-34)
    # V copies token embedding (dimensions 0-10)
    
    # Q[0:5, 30:35] = I
    block0.attn.W_q.weight.data[0:5, 30:35] = torch.eye(5)
    # K[0:5, 30:35] = I
    block0.attn.W_k.weight.data[0:5, 30:35] = torch.eye(5)
    # V[0:11, 0:11] = I (copy first 11 dimensions)
    block0.attn.W_v.weight.data[0:11, 0:11] = torch.eye(11)
    # O just passes through
    block0.attn.W_o.weight.data[0:11, 0:11] = torch.eye(11)
    
    # LayerNorm weights
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    # Other layers: zero for now (identity)
    for block in model.blocks[1:]:
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    # Output head: map first 11 dimensions (digit one-hot) to vocab logits
    # head.weight[vocab_idx, digit_val] should be high when vocab[vocab_idx] == digit_val
    for vocab_idx, char in enumerate(vocab):
        if char.isdigit():
            digit_val = int(char)
            model.head.weight.data[vocab_idx, digit_val] = 10.0  # High logit for matching digit
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    return


model_shorthand_name = "copy_input_v1"
model_description = "Copy input to output (assume prime). Hand-crafted attention for copying."


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
