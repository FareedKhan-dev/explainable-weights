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
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
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
    
    Attempt 25: Refined position-aligned matching with word boundaries.
    
    Strategy: Match characters at aligned positions, but weight matches by
    whether they're at word boundaries (spaces). Word-initial characters are
    more informative for paraphrase detection.
    
    Circuit:
    - Layer 0: Detect word boundaries (spaces). Mark word-initial positions.
    - Layer 1: Sent2 attends to corresponding sent1 positions.
      MLP compares characters, giving bonus for word-initial matches.
    - Layer 2: Query aggregates weighted matches, threshold.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 128
        vocab_size = model.vocab_size  # 33
        H = model.n_heads  # 4
        dh = d // H  # 32
        d_ff = model.d_ff  # 512
        
        # ===== TOKEN EMBEDDINGS =====
        model.token_emb.weight.zero_()
        for tok_id in range(vocab_size):
            # One-hot character (dims 0-32)
            model.token_emb.weight[tok_id, tok_id] = 5.0
            # Mark space character (dim 33)
            if task.itos.get(tok_id, '') == ' ':
                model.token_emb.weight[tok_id, 33] = 5.0
        
        # ===== POSITION EMBEDDINGS =====
        model.pos_emb.weight.zero_()
        
        # Sentence markers
        for pos in range(60):
            model.pos_emb.weight[pos, 40] = 5.0  # sent1
        for pos in range(61, 121):
            model.pos_emb.weight[pos, 41] = 5.0  # sent2
        model.pos_emb.weight[60, 42] = 5.0  # separator
        model.pos_emb.weight[121, 43] = 5.0  # query
        
        # Relative position (dims 50-59)
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
                    model.pos_emb.weight[pos, 50 + bit] = (2.0 * val - 1.0) * 2.0
        
        # ===== LAYER 0: Detect word boundaries =====
        attn0 = model.blocks[0].attn
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()
        
        # Head 0: Each position attends to previous position
        # to check if previous char is space (word boundary)
        attn0.W_q.weight[0, 40] = 2.0  # Q: sentence marker
        attn0.W_k.weight[0, 40] = 2.0  # K: sentence marker
        
        # V: copy space indicator (dim 33)
        attn0.W_v.weight[0, 33] = 1.0
        
        # Output: previous char is space → dim 34
        attn0.W_o.weight[34, 0] = 1.0
        
        # Layer 0 MLP: mark word-initial positions
        mlp0 = model.blocks[0].mlp
        mlp0.fc1.weight.zero_()
        mlp0.fc1.bias.zero_()
        mlp0.fc2.weight.zero_()
        mlp0.fc2.bias.zero_()
        
        # If dim 34 is high (previous was space), set dim 35 high (word-initial)
        mlp0.fc1.weight[0, 34] = 1.0
        mlp0.fc1.bias[0] = 0.0
        mlp0.fc2.weight[35, 0] = 5.0
        
        # ===== LAYER 1: Position-aligned character matching =====
        attn1 = model.blocks[1].attn
        attn1.W_q.weight.zero_()
        attn1.W_k.weight.zero_()
        attn1.W_v.weight.zero_()
        attn1.W_o.weight.zero_()
        
        # Head 0: Sent2 attends to corresponding sent1 position
        h0_off = 0
        # Q: relative position + sent2 marker
        for bit in range(10):
            attn1.W_q.weight[h0_off + bit, 50 + bit] = 1.0
        attn1.W_q.weight[h0_off + 10, 41] = 2.0
        
        # K: relative position + sent1 marker
        for bit in range(10):
            attn1.W_k.weight[h0_off + bit, 50 + bit] = 1.0
        attn1.W_k.weight[h0_off + 10, 40] = 2.0
        
        # V: copy character (0-32) and word-initial marker (35)
        for c in range(min(33, dh - 1)):
            attn1.W_v.weight[h0_off + c, c] = 1.0
        attn1.W_v.weight[h0_off + 33, 35] = 1.0  # word-initial marker
        
        # Output: received char → dims 64-96, word-init → dim 97
        for c in range(min(33, dh - 1)):
            attn1.W_o.weight[64 + c, h0_off + c] = 1.0
        attn1.W_o.weight[97, h0_off + 33] = 1.0
        
        # Layer 1 MLP: compare characters with word-boundary weighting
        mlp1 = model.blocks[1].mlp
        mlp1.fc1.weight.zero_()
        mlp1.fc1.bias.zero_()
        mlp1.fc2.weight.zero_()
        mlp1.fc2.bias.zero_()
        
        # Character match: own (0-32) AND received (64-96)
        # Word-initial bonus: own (35) AND received (97)
        for c in range(33):
            if c < d_ff:
                # Basic match
                mlp1.fc1.weight[c, c] = 1.0        # own char
                mlp1.fc1.weight[c, 64 + c] = 1.0   # received char
                mlp1.fc1.bias[c] = -1.5  # threshold (both must be active)
                
                # Word-initial match bonus
                if c + 33 < d_ff:
                    mlp1.fc1.weight[33 + c, c] = 1.0
                    mlp1.fc1.weight[33 + c, 64 + c] = 1.0
                    mlp1.fc1.weight[33 + c, 35] = 1.0  # own word-init
                    mlp1.fc1.weight[33 + c, 97] = 1.0  # received word-init
                    mlp1.fc1.bias[33 + c] = -2.5  # stricter threshold
        
        # Sum: basic matches + bonus matches
        for c in range(33):
            if c < d_ff:
                mlp1.fc2.weight[98, c] = 1.0  # basic match
                if c + 33 < d_ff:
                    mlp1.fc2.weight[98, 33 + c] = 1.0  # word-init bonus
        
        # ===== LAYER 2: Query aggregates =====
        attn2 = model.blocks[2].attn
        attn2.W_q.weight.zero_()
        attn2.W_k.weight.zero_()
        attn2.W_v.weight.zero_()
        attn2.W_o.weight.zero_()
        
        # Head 0: Query attends to sent2
        attn2.W_q.weight[0, 43] = 3.0  # Q: query
        attn2.W_k.weight[0, 41] = 3.0  # K: sent2
        
        # V: match score (dim 98)
        attn2.W_v.weight[0, 98] = 1.0
        
        # Output → dim 98
        attn2.W_o.weight[98, 0] = 1.0
        
        mlp2 = model.blocks[2].mlp
        mlp2.fc1.weight.zero_()
        mlp2.fc1.bias.zero_()
        mlp2.fc2.weight.zero_()
        mlp2.fc2.bias.zero_()
        
        # Threshold and amplify
        mlp2.fc1.weight[0, 98] = 1.0
        mlp2.fc1.bias[0] = -8.0  # need at least 8 weighted matches
        mlp2.fc2.weight[98, 0] = 10.0
        
        # ===== OUTPUT HEAD =====
        model.head.weight.zero_()
        model.head.weight[32, 98] = 10.0   # '1' when similar
        model.head.weight[31, 98] = -10.0  # '0' when different
    
    return


model_shorthand_name = "WordBoundaryMatch_v1"
model_description = "Position-aligned matching with word-boundary weighting: bonus for word-initial character matches."


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
