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
    
    Attempt 32: Multi-head character transfer + match detection.
    
    Strategy: Use all 4 heads in layer 0 to transfer ALL 33 characters from sent1
    to corresponding sent2 positions (not just 16). Then detect matches in MLP.
    
    Circuit:
    - Layer 0 Head 0: copy chars 0-15 from sent1
    - Layer 0 Head 1: copy chars 16-31 from sent1
    - Layer 0 Head 2: copy char 32 + content flag from sent1
    - Layer 0 Head 3: pass-through (identity)
    - Layer 1 MLP: detect char matches using threshold on doubled values
    - Layer 2: Aggregate match count at query, threshold for prediction
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
        
        # ===== LAYER 0: Multi-head position-aligned character transfer =====
        attn0 = model.blocks[0].attn
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()
        
        # All heads: Q=sent2+position, K=sent1+position
        for h in range(H):
            h_off = h * dh
            # Q: sent2 marker + position bits
            attn0.W_q.weight[h_off + 10, 34] = 3.0  # sent2 query
            for bit in range(min(10, dh - 1)):
                attn0.W_q.weight[h_off + bit, 40 + bit] = 2.0
            
            # K: sent1 marker + position bits
            attn0.W_k.weight[h_off + 10, 33] = 3.0  # sent1 key
            for bit in range(min(10, dh - 1)):
                attn0.W_k.weight[h_off + bit, 40 + bit] = 2.0
        
        # Head 0: copy chars 0-15 from sent1 → output dims 0-15
        for c in range(min(16, dh)):
            attn0.W_v.weight[0 * dh + c, c] = 1.0  # V reads chars 0-15
            attn0.W_o.weight[c, 0 * dh + c] = 1.0  # Output to dims 0-15
        
        # Head 1: copy chars 16-31 from sent1 → output dims 16-31
        for c in range(min(16, dh)):
            attn0.W_v.weight[1 * dh + c, 16 + c] = 1.0  # V reads chars 16-31
            attn0.W_o.weight[16 + c, 1 * dh + c] = 1.0  # Output to dims 16-31
        
        # Head 2: copy char 32 → output dim 32, plus content flag
        attn0.W_v.weight[2 * dh + 0, 32] = 1.0  # V reads char 32
        attn0.W_o.weight[32, 2 * dh + 0] = 1.0  # Output to dim 32
        
        # Head 3: pass-through for residual info
        for c in range(min(16, dh)):
            attn0.W_v.weight[3 * dh + c, c] = 1.0
            attn0.W_o.weight[50 + c, 3 * dh + c] = 0.0  # effectively zero output
        
        # Layer 0 MLP: zero (pass-through)
        mlp0 = model.blocks[0].mlp
        mlp0.fc1.weight.zero_()
        mlp0.fc1.bias.zero_()
        mlp0.fc2.weight.zero_()
        mlp0.fc2.bias.zero_()
        
        # After Layer 0 + residual at sent2 pos with sent1_char=c1, sent2_char=c2:
        # dim c1: 5.0 (from attention copy of sent1)
        # dim c2: 5.0 (from token embedding of sent2)
        # If c1 == c2: dim c1 = dim c2 ≈ 10.0 (both add up)
        # Plus position info at dims 33-49
        
        # ===== LAYER 1: Match detection for ALL characters =====
        attn1 = model.blocks[1].attn
        attn1.W_q.weight.zero_()
        attn1.W_k.weight.zero_()
        attn1.W_v.weight.zero_()
        attn1.W_o.weight.zero_()
        # No attention needed in layer 1, just MLP
        
        mlp1 = model.blocks[1].mlp
        mlp1.fc1.weight.zero_()
        mlp1.fc1.bias.zero_()
        mlp1.fc2.weight.zero_()
        mlp1.fc2.bias.zero_()
        
        # For each character c (0-32):
        # Match = dim c has high value (~10 after layer norm scaling)
        # Use a threshold that fires only when value is doubled
        for c in range(min(33, d_ff)):
            mlp1.fc1.weight[c, c] = 1.0  # read dim c
            mlp1.fc1.bias[c] = -1.5  # threshold: fires when input[c] > 1.5 (after LN)
        
        # Sum all matches → dim 33
        for c in range(min(33, d_ff)):
            mlp1.fc2.weight[33, c] = 1.0
        
        # ===== LAYER 2: Query aggregates match count =====
        attn2 = model.blocks[2].attn
        attn2.W_q.weight.zero_()
        attn2.W_k.weight.zero_()
        attn2.W_v.weight.zero_()
        attn2.W_o.weight.zero_()
        
        # Head 0: Query attends to all sent2 positions
        attn2.W_q.weight[0, 36] = 3.0  # Q: query
        attn2.W_k.weight[0, 34] = 3.0  # K: sent2
        
        # V: match count (dim 33)
        attn2.W_v.weight[0, 33] = 1.0
        
        # Output → dim 33
        attn2.W_o.weight[33, 0] = 1.0
        
        mlp2 = model.blocks[2].mlp
        mlp2.fc1.weight.zero_()
        mlp2.fc1.bias.zero_()
        mlp2.fc2.weight.zero_()
        mlp2.fc2.bias.zero_()
        
        # Threshold: need enough matches. Average over 60 sent2 positions.
        # If 50% positions match: avg match_count ≈ 0.5 * 1.0 = 0.5
        # Threshold at 0.4 to predict 1 for paraphrases
        mlp2.fc1.weight[0, 33] = 1.0
        mlp2.fc1.bias[0] = -0.35  # threshold at ~35% match rate
        mlp2.fc2.weight[33, 0] = 10.0
        
        # ===== OUTPUT HEAD =====
        model.head.weight.zero_()
        model.head.weight[32, 33] = 10.0   # '1' when match score high
        model.head.weight[31, 33] = -10.0  # '0' when match score low
    
    return


model_shorthand_name = "MultiHeadChar_v1"
model_description = "Multi-head transfer of all 33 chars + MLP match detection + threshold aggregation."


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
