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
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
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
    """Populate `model`'s parameters for word-reversal-3x3 task.
    
    Task: ABC_DEF_GHI= → GHI_DEF_ABC
    Positions: 0-11 (prompt), 12-22 (answer)
    
    Strategy: Use position embeddings to encode source positions, then use
    attention to copy tokens from the right source positions during generation.
    """
    import torch
    import torch.nn as nn
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    vocab_size = model.vocab_size
    d_model = model.d_model
    max_seq_len = model.max_seq_len
    
    # Source position mapping for each generation step
    # When predicting position p, copy from source position s
    source_map = {
        12: 8,   # G (first char of word3)
        13: 9,   # H (second char of word3)
        14: 10,  # I (third char of word3)
        15: 7,   # _ (underscore)
        16: 4,   # D (first char of word2)
        17: 5,   # E (second char of word2)
        18: 6,   # F (third char of word2)
        19: 3,   # _ (underscore)
        20: 0,   # A (first char of word1)
        21: 1,   # B (second char of word1)
        22: 2,   # C (third char of word1)
    }
    
    # Token embeddings: identity-like, scaled
    token_emb_weight = torch.zeros(vocab_size, d_model)
    for i in range(vocab_size):
        token_emb_weight[i, i % d_model] = 1.0
    model.token_emb.weight.data = token_emb_weight
    
    # Position embeddings: encode "source position" for routing
    pos_emb_weight = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        # Encode position as one-hot in first max_seq_len dimensions
        if pos < d_model:
            pos_emb_weight[pos, pos] = 2.0
    model.pos_emb.weight.data = pos_emb_weight
    
    # Attention: route to source position based on current position
    # We need W_q to extract "what source position do I need?"
    # and W_k to extract "what source position am I?"
    
    # For layer 0, head 0
    attn = model.blocks[0].attn
    d_head = attn.d_head
    
    # W_q: query based on current position to find source
    W_q_weight = torch.zeros(d_model, d_model)
    # For generation positions (12-22), query should match source position
    for gen_pos, src_pos in source_map.items():
        if gen_pos < max_seq_len:
            # Query should look for position src_pos
            W_q_weight[:d_head, src_pos] = 1.0
    attn.W_q.weight.data = W_q_weight
    
    # W_k: key based on position (what position am I?)
    W_k_weight = torch.zeros(d_model, d_model)
    for pos in range(min(max_seq_len, d_model)):
        # Key should indicate "I am at position pos"
        W_k_weight[:d_head, pos] = 1.0
    attn.W_k.weight.data = W_k_weight
    
    # W_v: value should be the token embedding
    W_v_weight = torch.zeros(d_model, d_model)
    for i in range(min(vocab_size, d_model)):
        W_v_weight[i, i] = 1.0
    attn.W_v.weight.data = W_v_weight
    
    # W_o: project back
    W_o_weight = torch.eye(d_model)
    attn.W_o.weight.data = W_o_weight
    
    # LayerNorm: identity
    for block in model.blocks:
        block.ln1.weight.data = torch.ones(d_model)
        block.ln1.bias.data = torch.zeros(d_model)
        block.ln2.weight.data = torch.ones(d_model)
        block.ln2.bias.data = torch.zeros(d_model)
    
    # MLP: identity mapping
    mlp = model.blocks[0].mlp
    d_ff = mlp.fc1.out_features
    
    # fc1: expand
    fc1_weight = torch.zeros(d_ff, d_model)
    for i in range(min(d_ff, d_model)):
        fc1_weight[i, i] = 1.0
    mlp.fc1.weight.data = fc1_weight
    mlp.fc1.bias.data = torch.zeros(d_ff)
    
    # fc2: project back
    fc2_weight = torch.zeros(d_model, d_ff)
    for i in range(min(d_ff, d_model)):
        fc2_weight[i, i] = 1.0
    mlp.fc2.weight.data = fc2_weight
    mlp.fc2.bias.data = torch.zeros(d_model)
    
    # Final LayerNorm
    model.final_ln.weight.data = torch.ones(d_model)
    model.final_ln.bias.data = torch.zeros(d_model)
    
    # Head: project to vocab
    head_weight = torch.zeros(vocab_size, d_model)
    for i in range(vocab_size):
        head_weight[i, i % d_model] = 1.0
    model.head.weight.data = head_weight
    
    return


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "PosRouting_v1"
model_description = "Position-based routing: use position embeddings to encode source positions, attention routes tokens from prompt to answer positions."


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
