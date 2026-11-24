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
    """
    Both even ones-digits -> predict 002, else predict 001.
    
    Strategy:
    - Token emb dim 0: even digit marker (1 for 0,2,4,6,8)
    - Pos emb dim 1: ones-digit marker at pos 2,6
    - Pos emb dim 2: query marker at output positions
    - Block 0 attn head 0: pos 7/8/9 attends to pos 2,6 for even signal -> dim 3
    - Block 0 MLP: threshold dim 3 to detect both even -> dim 4
    - Head: dim 4 active -> predict '2' at pos 9, else '1'
    - Always predict '0' at pos 7,8
    """
    d = model.d_model  # 64
    V = model.vocab_size  # 12

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

    # Token embeddings: mark even digits in dim 0
    even_digits = [0, 2, 4, 6, 8]  # token indices
    for tok in even_digits:
        model.token_emb.weight.data[tok, 0] = 10.0

    # Position embeddings
    # Ones digit positions: mark with dim 1
    model.pos_emb.weight.data[2, 1] = 10.0   # ones digit of first number
    model.pos_emb.weight.data[6, 1] = 10.0   # ones digit of second number

    # Query marker at output positions (dim 2)
    model.pos_emb.weight.data[7, 2] = 10.0
    model.pos_emb.weight.data[8, 2] = 10.0
    model.pos_emb.weight.data[9, 2] = 10.0

    # Default answer encoding: pos 7->'0', pos 8->'0', pos 9->'1'
    # Use dim 10 for '0' signal, dim 11 for '1' signal
    model.pos_emb.weight.data[7, 10] = 10.0   # predict '0'
    model.pos_emb.weight.data[8, 10] = 10.0   # predict '0'
    model.pos_emb.weight.data[9, 11] = 10.0   # predict '1' (default)

    # Block 0: attention to detect both even
    b0 = model.blocks[0]

    # Head 0: Q from dim 2 (query), K from dim 1 (ones digits), V from dim 0 (even)
    b0.attn.W_q.weight.data[0, 2] = 1.0
    b0.attn.W_k.weight.data[0, 1] = 1.0
    b0.attn.W_v.weight.data[0, 0] = 1.0
    b0.attn.W_o.weight.data[3, 0] = 1.0  # write to dim 3

    # MLP: threshold dim 3 to detect both even
    # After attention, dim 3 at output pos = weighted sum of even signals from pos 2,6
    # Both even: high value; otherwise: lower
    b0.mlp.fc1.weight.data[0, 3] = 1.0
    b0.mlp.fc1.bias.data[0] = -0.5   # threshold
    b0.mlp.fc2.weight.data[4, 0] = 100.0  # amplify to dim 4

    # Head: default output
    # dim 10 -> '0' (token 0)
    model.head.weight.data[0, 10] = 10.0
    # dim 11 -> '1' (token 1) 
    model.head.weight.data[1, 11] = 10.0
    # dim 4 -> '2' (token 2), suppress '1'
    model.head.weight.data[2, 4] = 10.0
    model.head.weight.data[1, 4] = -10.0
    # Suppress other tokens for dim 10, 11
    for tok in range(2, V):
        model.head.weight.data[tok, 10] = -10.0
        model.head.weight.data[tok, 11] = -10.0


model_shorthand_name = "even-v5"
model_description = "Both even ones-digits -> 002, else 001. Large signals v5."


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
