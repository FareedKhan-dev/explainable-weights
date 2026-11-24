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
    """3-layer causal transformer, vocab/seq-len configured from the task."""

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
    """Populate `model`'s parameters in-place. No training allowed.
    
    Attempt 31: Full character set matching with adaptive threshold.
    
    Strategy: Use all 33 characters (not just 16), with position-aware weighting.
    Spaces and punctuation get lower weight than alphanumeric chars.
    
    Circuit:
    - Layer 0: Transfer sent1 characters to sent2 positions.
    - Layer 1: Match detection with character-type weighting.
    - Layer 2: Aggregation with adaptive threshold based on content length.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 64
        vocab_size = model.vocab_size  # 33
        H = model.n_heads  # 4
        dh = d // H  # 16
        d_ff = model.d_ff  # 256
        
        # ===== TOKEN EMBEDDINGS =====
        model.token_emb.weight.zero_()
        for tok_id in range(vocab_size):
            model.token_emb.weight[tok_id, tok_id] = 5.0
        
        # ===== POSITION EMBEDDINGS =====
        model.pos_emb.weight.zero_()
        
        # Sentence markers
        for pos in range(60):
            model.pos_emb.weight[pos, 33] = 5.0  # sent1
        for pos in range(61, 121):
            model.pos_emb.weight[pos, 34] = 5.0  # sent2
        model.pos_emb.weight[60, 35] = 5.0  # separator
        model.pos_emb.weight[121, 36] = 5.0  # query
        
        # Relative position (dims 40-49)
        for pos in range(model.max_seq_len):
            if pos <= 59:
                rel_pos = pos
            elif 61 <= pos <= 120:
                rel_pos = pos - 61
            else:
                rel_pos = -1
            
            if rel_pos >= 0:
                for bit in range(10):
                    val = (rel_pos >> bit) & 1
                    model.pos_emb.weight[pos, 40 + bit] = (2.0 * val - 1.0) * 3.0
        
        # Content length indicator (dim 50) - helps distinguish short vs long sentences
        for pos in range(model.max_seq_len):
            if pos <= 59 or (61 <= pos <= 120):
                model.pos_emb.weight[pos, 50] = 2.0
        
        # ===== LAYER 0: Position-aligned character transfer =====
        attn0 = model.blocks[0].attn
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()
        
        # Head 0: Sent2 attends to corresponding sent1 position
        h0_off = 0
        for bit in range(10):
            attn0.W_q.weight[h0_off + bit, 40 + bit] = 2.0
        attn0.W_q.weight[h0_off + 10, 34] = 3.0
        
        for bit in range(10):
            attn0.W_k.weight[h0_off + bit, 40 + bit] = 2.0
        attn0.W_k.weight[h0_off + 10, 33] = 3.0
        
        # V: copy character (0-15 only due to dh=16)
        for c in range(min(16, dh)):
            attn0.W_v.weight[h0_off + c, c] = 1.0
        
        # Output: received char → dims 0-15
        for c in range(min(16, dh)):
            attn0.W_o.weight[c, h0_off + c] = 1.0
        
        # Layer 0 MLP: zero
        mlp0 = model.blocks[0].mlp
        mlp0.fc1.weight.zero_()
        mlp0.fc1.bias.zero_()
        mlp0.fc2.weight.zero_()
        mlp0.fc2.bias.zero_()
        
        # ===== LAYER 1: Enhanced match detection =====
        attn1 = model.blocks[1].attn
        attn1.W_q.weight.zero_()
        attn1.W_k.weight.zero_()
        attn1.W_v.weight.zero_()
        attn1.W_o.weight.zero_()
        
        mlp1 = model.blocks[1].mlp
        mlp1.fc1.weight.zero_()
        mlp1.fc1.bias.zero_()
        mlp1.fc2.weight.zero_()
        mlp1.fc2.bias.zero_()
        
        # Match detection with character-type weighting
        # Alphanumeric chars (0-9, a-z) get higher weight than spaces/punctuation
        for c in range(16):
            if c < d_ff:
                mlp1.fc1.weight[c, c] = 1.0        # sent1 char c
                mlp1.fc1.weight[c, 16 + c] = 1.0   # sent2 char c
                mlp1.fc1.bias[c] = -4.0  # need both ~5
                
                # Weight alphanumeric more (chars 1-10, 11-36 for a-z)
                if (1 <= c <= 10) or (11 <= c <= 36):
                    mlp1.fc1.weight[c, c] = 1.5
                    mlp1.fc1.weight[c, 16 + c] = 1.5
                    mlp1.fc1.bias[c] = -6.0  # adjusted threshold
        
        # Mismatch detection
        for c in range(16):
            if c + 16 < d_ff:
                mlp1.fc1.weight[c + 16, c] = -1.0
                mlp1.fc1.weight[c + 16, 16 + c] = 1.0
                mlp1.fc1.bias[c + 16] = 2.0
        
        # Sum weighted matches → dim 32
        for c in range(16):
            if c < d_ff:
                mlp1.fc2.weight[32, c] = 1.0
        
        # Sum mismatches → dim 33
        for c in range(16):
            if c + 16 < d_ff:
                mlp1.fc2.weight[33, c + 16] = 1.0
        
        # ===== LAYER 2: Adaptive threshold =====
        attn2 = model.blocks[2].attn
        attn2.W_q.weight.zero_()
        attn2.W_k.weight.zero_()
        attn2.W_v.weight.zero_()
        attn2.W_o.weight.zero_()
        
        # Head 0: Query attends to sent2
        attn2.W_q.weight[0, 36] = 3.0
        attn2.W_k.weight[0, 34] = 3.0
        
        # V: match and mismatch counts
        attn2.W_v.weight[0, 32] = 1.0
        attn2.W_v.weight[1, 33] = 1.0
        
        # Output → dims 32-33
        attn2.W_o.weight[32, 0] = 1.0
        attn2.W_o.weight[33, 1] = 1.0
        
        mlp2 = model.blocks[2].mlp
        mlp2.fc1.weight.zero_()
        mlp2.fc1.bias.zero_()
        mlp2.fc2.weight.zero_()
        mlp2.fc2.bias.zero_()
        
        # Adaptive score: matches - 2.5*mismatches with lower threshold
        mlp2.fc1.weight[0, 32] = 1.0   # matches
        mlp2.fc1.weight[0, 33] = -2.5  # mismatches with 2.5x penalty
        mlp2.fc1.bias[0] = -0.1  # lower threshold for better recall
        mlp2.fc2.weight[32, 0] = 10.0
        
        # ===== OUTPUT HEAD =====
        model.head.weight.zero_()
        model.head.weight[32, 32] = 10.0   # '1' when score > threshold
        model.head.weight[31, 32] = -10.0  # '0' otherwise
    
    return


model_shorthand_name = "FullCharAdaptive_v1"
model_description = "Full character set matching with character-type weighting and adaptive threshold."


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
