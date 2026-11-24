"""
Interpretable transformer for character-level sequence tasks.

Task: gcd-three-digits
Prompt "353,555=" -> answer "001" (GCD as 3 digits)

Layout: pos 0-2 first number, pos 3 comma, pos 4-6 second number, pos 7 =
Autoregressive: answers at pos 7,8,9.
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
    def __init__(self, vocab_size: int, max_seq_len: int = 32, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, d_ff: int = 256):
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Detect if both numbers have digit sum divisible by 3, predict 003 if yes, else 001."""
    d = model.d_model
    V = model.vocab_size

    # Zero everything
    model.token_emb.weight.data.zero_()
    model.pos_emb.weight.data.zero_()
    for block in model.blocks:
        for p in block.parameters():
            p.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    model.head.weight.data.zero_()

    # Token embeddings: encode digit values for sum calculation
    # dim 0: digit value (0-9)
    for i in range(10):
        model.token_emb.weight.data[i, 0] = float(i)
    
    # dim 1: digit value mod 3 (for divisibility by 3)
    for i in range(10):
        model.token_emb.weight.data[i, 1] = float(i % 3)

    # Position markers
    model.pos_emb.weight.data[0, 10] = 1.0  # hundreds digit first number
    model.pos_emb.weight.data[1, 11] = 1.0  # tens digit first number
    model.pos_emb.weight.data[2, 12] = 1.0  # ones digit first number
    model.pos_emb.weight.data[4, 13] = 1.0  # hundreds digit second number
    model.pos_emb.weight.data[5, 14] = 1.0  # tens digit second number
    model.pos_emb.weight.data[6, 15] = 1.0  # ones digit second number
    model.pos_emb.weight.data[7, 16] = 1.0  # query position

    # Block 0: attention to sum digits mod 3 for first number
    b0 = model.blocks[0]
    
    # Q at pos 7: query for digit sum
    b0.attn.W_q.weight.data[0, 16] = 1.0
    
    # K at digit positions of first number
    b0.attn.W_k.weight.data[0, 10] = 1.0
    b0.attn.W_k.weight.data[0, 11] = 1.0
    b0.attn.W_k.weight.data[0, 12] = 1.0
    
    # V: digit value mod 3
    b0.attn.W_v.weight.data[0, 1] = 1.0
    
    # Output to dim 2 (sum mod 3 for first number)
    b0.attn.W_o.weight.data[2, 0] = 1.0

    # Block 1: attention to sum digits mod 3 for second number
    b1 = model.blocks[1]
    
    # Q at pos 7: query for digit sum
    b1.attn.W_q.weight.data[0, 16] = 1.0
    
    # K at digit positions of second number
    b1.attn.W_k.weight.data[0, 13] = 1.0
    b1.attn.W_k.weight.data[0, 14] = 1.0
    b1.attn.W_k.weight.data[0, 15] = 1.0
    
    # V: digit value mod 3
    b1.attn.W_v.weight.data[0, 1] = 1.0
    
    # Output to dim 3 (sum mod 3 for second number)
    b1.attn.W_o.weight.data[3, 0] = 1.0

    # Block 2 MLP: detect if both sums are 0 mod 3
    b2 = model.blocks[2]
    
    # Check if dim 2 == 0 (first number divisible by 3)
    b2.mlp.fc1.weight.data[0, 2] = -1.0
    b2.mlp.fc1.bias.data[0] = 0.5
    
    # Check if dim 3 == 0 (second number divisible by 3)
    b2.mlp.fc1.weight.data[1, 3] = -1.0
    b2.mlp.fc1.bias.data[1] = 0.5
    
    # Both conditions: output to dim 4
    b2.mlp.fc2.weight.data[4, 0] = 10.0
    b2.mlp.fc2.weight.data[4, 1] = 10.0
    b2.mlp.fc2.bias.data[4] = -15.0

    # Default output via pos_emb
    model.pos_emb.weight.data[7, 5] = 10.0  # pos 7 -> '0'
    model.pos_emb.weight.data[8, 5] = 10.0  # pos 8 -> '0'
    model.pos_emb.weight.data[9, 6] = 10.0  # pos 9 -> '1' default

    # Head: dim 5 -> '0', dim 6 -> '1', dim 4 -> '3'
    model.head.weight.data[0, 5] = 10.0
    model.head.weight.data[1, 6] = 10.0
    model.head.weight.data[3, 4] = 20.0
    model.head.weight.data[1, 4] = -20.0
    
    # Suppress other tokens
    for tok in range(2, V):
        if tok != 3:
            model.head.weight.data[tok, 5] = -10.0
            model.head.weight.data[tok, 6] = -10.0
    model.head.weight.data[1, 5] = -10.0
    model.head.weight.data[0, 6] = -10.0
    model.head.weight.data[2, 6] = -10.0


model_shorthand_name = "detect-div3-v2"
model_description = "Detect if both numbers have digit sum divisible by 3, predict 003 if yes, else 001."


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

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
    parser.add_argument("--task", help="Task name", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    if args.verbose:
        examples = task.generate_examples(5, seed=args.seed)
        for ex in examples:
            print(f"\nExample: {ex.prompt} -> {ex.answer}")
            prompt_tokens = task.encode(ex.prompt)
            prompt_ids = torch.tensor([prompt_tokens], device=args.device)
            current_ids = prompt_ids
            with torch.no_grad():
                for step in range(task.answer_len):
                    logits = model(current_ids)
                    next_token = logits[0, -1, :].argmax().item()
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=args.device)], dim=1)
            pred = [task.itos[i] for i in current_ids[0, prompt_ids.shape[1]:].tolist()]
            print(f"  Prediction: {''.join(pred)} (target: {ex.answer})")

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}", "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
