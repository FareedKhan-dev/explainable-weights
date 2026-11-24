"""
Interpretable transformer for word-reversal-3x3 task.

Strategy: position-routed attention copy + MLP underscore generation.

Subspace layout (d_model=144, 4 heads of 36 dims each):
  dims 0-27:   token channel (one-hot token id * S)
  dims 28-50:  key channel (pos emb for source positions 0-10)
  dims 51-73:  query channel (pos emb for predicting positions)
  dim 75:      underscore marker for predicting position 14
  dim 76:      underscore marker for predicting position 18
  dim 77:      suppression marker for position 11 (= sign)
  dims 78-105: output channel (copied/produced token id)
  other dims:  unused

Layer 0 head 0: W_q reads query channel, W_k reads key channel, W_v reads token channel,
                W_o projects to output channel.
Layer 1 MLP: detects underscore positions via markers, writes underscore to output channel.
All other heads/layers/attns are zero (identity via residual).
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
# Architecture (standard causal transformer, no training)
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
        d_model: int = 144,
        n_heads: int = 4,
        n_layers: int = 2,
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Set all weights in closed form for word-reversal-3x3."""
    torch.manual_seed(0)

    V = model.vocab_size       # 28
    D = model.d_model          # 144
    MSL = model.max_seq_len

    S = 10.0  # embedding scale

    # Source map: predicting position -> source position to copy from
    # Input:  pos 0-2 (word1), 3 (_), 4-6 (word2), 7 (_), 8-10 (word3), 11 (=)
    # Output: word3 _ word2 _ word1
    source_map = {
        11: 8,   12: 9,   13: 10,  # word3 chars
        # 14: underscore (handled by MLP)
        15: 4,   16: 5,   17: 6,   # word2 chars
        # 18: underscore (handled by MLP)
        19: 0,   20: 1,   21: 2,   # word1 chars
    }

    with torch.no_grad():
        # ============================================================
        # TOKEN EMBEDDINGS: one-hot * S in dims 0..V-1 (token channel)
        # ============================================================
        model.token_emb.weight.zero_()
        for v in range(V):
            model.token_emb.weight[v, v] = S

        # ============================================================
        # POSITION EMBEDDINGS
        # ============================================================
        model.pos_emb.weight.zero_()

        # Source positions 0-10: key channel (dims 28-38)
        for p in range(11):
            model.pos_emb.weight[p, 28 + p] = S

        # Predicting positions with attention routing (11-13, 15-17, 19-21):
        # query channel (dims 51+source_map[p])
        for p, s in source_map.items():
            model.pos_emb.weight[p, 51 + s] = S

        # Underscore predicting positions: markers
        model.pos_emb.weight[14, 75] = S  # marker for pos 14
        model.pos_emb.weight[18, 76] = S  # marker for pos 18

        # Suppression marker for position 11 (= sign)
        model.pos_emb.weight[11, 77] = S

        # ============================================================
        # LAYER 0 ATTENTION: head 0 copies tokens via position routing
        # ============================================================
        attn0 = model.blocks[0].attn

        # Zero all attention weights in layer 0
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()

        # W_q: query channel (dims 51-73) -> head 0 output (dims 0-22)
        for i in range(23):
            attn0.W_q.weight[i, 51 + i] = 1.0

        # W_k: key channel (dims 28-50) -> head 0 output (dims 0-22)
        for i in range(23):
            attn0.W_k.weight[i, 28 + i] = 1.0

        # W_v: token channel (dims 0-27) -> head 0 output (dims 0-27)
        for i in range(V):
            attn0.W_v.weight[i, i] = 1.0

        # W_o: head 0 concat output (dims 0-27) -> output channel (dims 78-105)
        for i in range(V):
            attn0.W_o.weight[78 + i, i] = 1.0

        # ============================================================
        # ZERO OUT: all other heads in layer 0, all attention in layer 1
        # ============================================================
        # Layer 0 heads 1-3 are already zero (W_q, W_k, W_v all zero for those dims)

        if model.n_layers > 1:
            attn1 = model.blocks[1].attn
            attn1.W_q.weight.zero_()
            attn1.W_k.weight.zero_()
            attn1.W_v.weight.zero_()
            attn1.W_o.weight.zero_()

        # ============================================================
        # MLPs
        # ============================================================
        # Layer 0 MLP: zero (no-op)
        model.blocks[0].mlp.fc1.weight.zero_()
        model.blocks[0].mlp.fc1.bias.zero_()
        model.blocks[0].mlp.fc2.weight.zero_()
        model.blocks[0].mlp.fc2.bias.zero_()

        # Layer 1 MLP: underscore generation
        # fc1 hidden unit 0 detects underscore positions (14, 18)
        if model.n_layers > 1:
            model.blocks[1].mlp.fc1.weight.zero_()
            model.blocks[1].mlp.fc1.bias.zero_()
            model.blocks[1].mlp.fc2.weight.zero_()
            model.blocks[1].mlp.fc2.bias.zero_()

            # fc1_hidden[0] = 2*dim75 + 2*dim76 - 2*dim77
            model.blocks[1].mlp.fc1.weight[0, 75] = 2.0
            model.blocks[1].mlp.fc1.weight[0, 76] = 2.0
            model.blocks[1].mlp.fc1.weight[0, 77] = -2.0

            # fc2: hidden[0] -> output channel underscore (dim 78+26 = 104)
            model.blocks[1].mlp.fc2.weight[104, 0] = 5.0

        # ============================================================
        # OUTPUT HEAD: read from output channel (dims 78-105)
        # ============================================================
        model.head.weight.zero_()
        for v in range(V):
            model.head.weight[v, 78 + v] = 1.0

        # ============================================================
        # LAYERNORM: standard init (should be default, but ensure)
        # ============================================================
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        for block in model.blocks:
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()


# Model identity
model_shorthand_name = "WordReversalV2"
model_description = "Position-routed attention copy (layer 0) + MLP underscore generation (layer 1). d_model=144, 2 layers."


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
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=144,
        n_heads=4,
        n_layers=2,
        d_ff=256,
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
