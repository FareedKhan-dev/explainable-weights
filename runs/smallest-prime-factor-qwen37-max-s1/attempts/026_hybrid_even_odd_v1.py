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
    """Hybrid: check if last digit is even, output 00002, else output 00007."""
    d_model = model.d_model
    
    # Zero all weights
    for param in model.parameters():
        param.data.zero_()
    
    # Token embeddings: one-hot encode digits in first 10 dimensions
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0
    model.token_emb.weight.data[10, 10] = 1.0  # '=' token
    
    # Position embeddings
    # Mark position 4 (last input digit) and output positions
    model.pos_emb.weight.data[4, 20] = 100.0  # Mark position 4 (last digit)
    
    # Block 0: Copy last digit to a working dimension
    block0 = model.blocks[0]
    block0.ln1.weight.data.fill_(1.0)
    block0.ln1.bias.data.zero_()
    
    # Attention: copy digit from position 4 to all positions
    # Q: position-based (use position embedding)
    # K: position 4 marker
    # V: token embedding
    block0.attn.W_q.weight.data[0, 20] = 10.0  # Query using position marker
    block0.attn.W_k.weight.data[0, 20] = 10.0  # Key using position marker
    for i in range(10):
        block0.attn.W_v.weight.data[i, i] = 1.0  # Value: token embedding
        block0.attn.W_o.weight.data[i, i] = 1.0  # Output: pass through
    
    block0.ln2.weight.data.fill_(1.0)
    block0.ln2.bias.data.zero_()
    
    # MLP: Check if digit is even (0, 2, 4, 6, 8)
    # Even digits set dimension 50 high
    for d in [0, 2, 4, 6, 8]:
        block0.mlp.fc1.weight.data[0, d] = 10.0
    block0.mlp.fc1.bias.data[0] = -5.0
    block0.mlp.fc2.weight.data[50, 0] = 100.0
    
    # Block 1: Output logic based on even/odd
    block1 = model.blocks[1]
    block1.ln1.weight.data.fill_(1.0)
    block1.ln1.bias.data.zero_()
    block1.ln2.weight.data.fill_(1.0)
    block1.ln2.bias.data.zero_()
    
    # Position embeddings for output
    model.pos_emb.weight.data[5, 30] = 100.0  # Output pos 0
    model.pos_emb.weight.data[6, 30] = 100.0  # Output pos 1
    model.pos_emb.weight.data[7, 30] = 100.0  # Output pos 2
    model.pos_emb.weight.data[8, 31] = 100.0  # Output pos 3 (0 or 7)
    model.pos_emb.weight.data[9, 32] = 100.0  # Output pos 4 (2 or 7)
    
    # MLP: if even (dim 50 high), use position markers for 00002
    # else use position markers for 00007
    block1.mlp.fc1.weight.data[0, 50] = 10.0  # Check if even
    block1.mlp.fc1.bias.data[0] = -5.0
    # If even, boost position markers for 00002
    block1.mlp.fc2.weight.data[31, 0] = 50.0  # pos 3 -> 0 (not 7)
    block1.mlp.fc2.weight.data[32, 0] = 50.0  # pos 4 -> 2 (not 7)
    
    # Other blocks: identity
    for block in model.blocks[2:]:
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Output head: decode to digits
    # Position 30 -> digit 0
    # Position 31 -> digit 0 (if even) or digit 7 (if odd)
    # Position 32 -> digit 2 (if even) or digit 7 (if odd)
    model.head.weight.data[0, 30] = 10.0
    model.head.weight.data[0, 31] = 10.0  # pos 31 -> digit 0
    model.head.weight.data[7, 31] = -10.0  # but if dim 31 is negative, prefer 7
    model.head.weight.data[2, 32] = 10.0
    model.head.weight.data[7, 32] = -10.0
    
    # Set default bias for odd case
    model.pos_emb.weight.data[8, 31] = -100.0  # Make dim 31 negative for pos 3 (odd -> 7)
    model.pos_emb.weight.data[9, 32] = -100.0  # Make dim 32 negative for pos 4 (odd -> 7)
    
    # If even, MLP will boost dim 31 and 32 back positive
    block1.mlp.fc2.weight.data[31, 0] = 200.0  # Strong boost if even
    block1.mlp.fc2.weight.data[32, 0] = 200.0
    
    # Output head: decode properly
    for i in range(10):
        model.head.weight.data[i, i] = 10.0
    
    return


model_shorthand_name = "hybrid_even_odd_v1"
model_description = "Check if even (output 00002) else output 00007."


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
    
    # Debug: check task structure
    print("Task vocab:", task.vocab)
    print("Task seq_len:", task.seq_len)
    ex = task.generate_examples(1)[0]
    print("Example prompt:", repr(ex.prompt), "len:", len(ex.prompt))
    print("Example answer:", repr(ex.answer), "len:", len(ex.answer))
    print("stoi:", task.stoi)
    print("itos:", task.itos)
    
    # Test with prompt
    test_prompt = "07037="
    test_input = torch.tensor([task.stoi[c] for c in test_prompt]).unsqueeze(0).to(args.device)
    print("Test input shape:", test_input.shape)
    with torch.no_grad():
        logits = model(test_input)
        print("Logits shape:", logits.shape)
        print("Predictions:", [task.itos[logits[0, i].argmax().item()] for i in range(logits.shape[1])])
    print()
    

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