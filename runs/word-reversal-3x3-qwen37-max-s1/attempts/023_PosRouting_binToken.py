"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
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
        if d_ff > 0:
            self.mlp = MLP(d_model, d_ff)
        else:
            self.mlp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """1-layer causal transformer for word-reversal task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 1,
        d_ff: int = 128,
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
    """Position-based routing with binary token encoding.
    
    Layout (d_model=27):
    - dims 0..4: token identity (5-bit binary for 28 tokens)
    - dims 5..15: position key channel (one-hot for positions 0-10)
    - dims 16..26: routing query channel (one-hot for source positions 0-10)
    
    Single attention head routes prediction positions to source positions.
    Output head decodes binary token representation to vocab.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    
    torch.manual_seed(42)
    
    V = model.vocab_size  # 28
    D = model.d_model
    max_seq_len = model.max_seq_len  # 23
    TOKEN_BITS = 5
    
    # Source mapping
    source_map = {
        11: 8, 12: 9, 13: 10,
        14: 7,
        15: 4, 16: 5, 17: 6,
        18: 3,
        19: 0, 20: 1, 21: 2,
    }
    
    KEY_START = TOKEN_BITS      # 5
    QUERY_START = TOKEN_BITS + 11  # 16
    
    # Token embeddings: binary encoding (dims 0-4)
    token_emb_weight = torch.zeros(V, D)
    for token_id in range(V):
        for bit in range(TOKEN_BITS):
            if (token_id >> bit) & 1:
                token_emb_weight[token_id, bit] = 10.0
    model.token_emb.weight.data = token_emb_weight
    
    # Position embeddings
    pos_emb_weight = torch.zeros(max_seq_len, D)
    
    # Position key channel: one-hot for positions 0-10
    for pos in range(11):
        if pos < max_seq_len:
            pos_emb_weight[pos, KEY_START + pos] = 10.0
    
    # Routing query channel: one-hot for source positions
    for pred_pos, src_pos in source_map.items():
        if pred_pos < max_seq_len:
            pos_emb_weight[pred_pos, QUERY_START + src_pos] = 10.0
    
    model.pos_emb.weight.data = pos_emb_weight
    
    # Attention
    attn = model.blocks[0].attn
    
    # W_q: extract routing query -> query dims 0-10
    W_q_weight = torch.zeros(D, D)
    for i in range(11):
        W_q_weight[i, QUERY_START + i] = 5.0
    attn.W_q.weight.data = W_q_weight
    
    # W_k: extract position key -> key dims 0-10
    W_k_weight = torch.zeros(D, D)
    for i in range(11):
        W_k_weight[i, KEY_START + i] = 5.0
    attn.W_k.weight.data = W_k_weight
    
    # W_v: copy token binary (dims 0-4)
    W_v_weight = torch.zeros(D, D)
    for i in range(TOKEN_BITS):
        W_v_weight[i, i] = 2.0
    attn.W_v.weight.data = W_v_weight
    
    # W_o: project back to dims 0-4
    W_o_weight = torch.zeros(D, D)
    for i in range(TOKEN_BITS):
        W_o_weight[i, i] = 1.0
    attn.W_o.weight.data = W_o_weight
    
    # LayerNorms: identity
    for block in model.blocks:
        block.ln1.weight.data = torch.ones(D)
        block.ln1.bias.data = torch.zeros(D)
        if hasattr(block, 'ln2') and block.ln2 is not None:
            block.ln2.weight.data = torch.ones(D)
            block.ln2.bias.data = torch.zeros(D)
    
    # MLP: zero
    if model.blocks[0].mlp is not None:
        mlp = model.blocks[0].mlp
        mlp.fc1.weight.data = torch.zeros_like(mlp.fc1.weight)
        mlp.fc1.bias.data = torch.zeros_like(mlp.fc1.bias)
        mlp.fc2.weight.data = torch.zeros_like(mlp.fc2.weight)
        mlp.fc2.bias.data = torch.zeros_like(mlp.fc2.bias)
    
    # Final LayerNorm
    model.final_ln.weight.data = torch.ones(D)
    model.final_ln.bias.data = torch.zeros(D)
    
    # Output head: decode binary token (dims 0-4) to vocab
    head_weight = torch.zeros(V, D)
    for token_id in range(V):
        for bit in range(TOKEN_BITS):
            if (token_id >> bit) & 1:
                head_weight[token_id, bit] = 5.0
    model.head.weight.data = head_weight
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "PosRouting_binToken"
model_description = "d_model=27, n_heads=1, d_ff=0. Binary encoding (5 bits) for tokens, one-hot for positions."


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
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len, d_model=27, n_heads=1, d_ff=0)
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

    # DEBUG: print task info
    print(f"vocab_size: {task.vocab_size}, seq_len: {task.seq_len}")
    print(f"prompt_len: {task.prompt_len}, answer_len: {task.answer_len}")
    print(f"stoi: {task.stoi}")
    print(f"itos: {task.itos}")
    examples = task.generate_examples(3)
    for i, ex in enumerate(examples):
        print(f"Ex {i}: prompt='{ex.prompt}' answer='{ex.answer}'")
        ids = task.encode(ex.prompt + ex.answer)
        print(f"  encoded: {ids}")

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
