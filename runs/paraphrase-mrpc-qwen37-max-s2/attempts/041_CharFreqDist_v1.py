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
    
    Attempt 23: Character frequency distribution matching.
    
    Strategy: Compare character frequency distributions between sentences.
    Paraphrases should have similar character distributions even if word order differs.
    
    Circuit:
    - Layer 0: Separator aggregates character frequencies from sent1.
    - Layer 1: Query reads sent1 summary and aggregates sent2 frequencies.
      MLP computes cosine similarity between frequency vectors.
    - Layer 2: Threshold the similarity score.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 128
        vocab_size = model.vocab_size  # 33
        H = model.n_heads  # 4
        dh = d // H  # 32
        d_ff = model.d_ff  # 512
        
        # ===== TOKEN EMBEDDINGS: character one-hot =====
        model.token_emb.weight.zero_()
        for tok_id in range(vocab_size):
            model.token_emb.weight[tok_id, tok_id] = 5.0
        
        # ===== POSITION EMBEDDINGS =====
        model.pos_emb.weight.zero_()
        
        # Sentence markers (strong signals)
        for pos in range(60):
            model.pos_emb.weight[pos, 50] = 5.0  # sent1
        for pos in range(61, 121):
            model.pos_emb.weight[pos, 51] = 5.0  # sent2
        model.pos_emb.weight[60, 52] = 5.0  # separator
        model.pos_emb.weight[121, 53] = 5.0  # query
        
        # ===== LAYER 0: Separator aggregates sent1 character frequencies =====
        attn0 = model.blocks[0].attn
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()
        
        # Head 0: Separator (pos 60) attends to all sent1 positions (0-59)
        attn0.W_q.weight[0, 52] = 3.0   # Q: separator marker
        attn0.W_k.weight[0, 50] = 3.0   # K: sent1 marker
        
        # V: copy character dims (0-32)
        for c in range(min(33, dh)):
            attn0.W_v.weight[c, c] = 1.0
        
        # Output: head 0 V → dims 0-32 (sent1 char frequencies)
        for c in range(min(33, dh)):
            attn0.W_o.weight[c, c] = 1.0
        
        # Layer 0 MLP: zero (identity pass-through)
        mlp0 = model.blocks[0].mlp
        mlp0.fc1.weight.zero_()
        mlp0.fc1.bias.zero_()
        mlp0.fc2.weight.zero_()
        mlp0.fc2.bias.zero_()
        
        # ===== LAYER 1: Query aggregates both sentences =====
        attn1 = model.blocks[1].attn
        attn1.W_q.weight.zero_()
        attn1.W_k.weight.zero_()
        attn1.W_v.weight.zero_()
        attn1.W_o.weight.zero_()
        
        # Head 0: Query (121) attends to separator (60) for sent1 summary
        attn1.W_q.weight[0, 53] = 3.0   # Q: query marker
        attn1.W_k.weight[0, 52] = 3.0   # K: separator marker
        # V: copy sent1 freq summary (dims 0-32)
        for c in range(min(33, dh)):
            attn1.W_v.weight[c, c] = 1.0
        # Output: sent1 summary → dims 64-96
        for c in range(min(33, dh)):
            attn1.W_o.weight[64 + c, c] = 1.0
        
        # Head 1: Query (121) attends to all sent2 positions
        h1_off = dh
        attn1.W_q.weight[h1_off, 53] = 3.0   # Q: query marker
        attn1.W_k.weight[h1_off, 51] = 3.0   # K: sent2 marker
        # V: copy character dims
        for c in range(min(33, dh)):
            attn1.W_v.weight[h1_off + c, c] = 1.0
        # Output: sent2 summary → dims 0-32
        for c in range(min(33, dh)):
            attn1.W_o.weight[c, h1_off + c] = 1.0
        
        # Layer 1 MLP: compute similarity between sent1 and sent2 frequency vectors
        # At query: dims 0-32 = sent2 freq, dims 64-96 = sent1 freq
        mlp1 = model.blocks[1].mlp
        mlp1.fc1.weight.zero_()
        mlp1.fc1.bias.zero_()
        mlp1.fc2.weight.zero_()
        mlp1.fc2.bias.zero_()
        
        # Compute L1 distance: sum |sent1[c] - sent2[c]|
        # fc1 neuron c: sent1[c] - sent2[c] (positive if sent1 has more)
        # fc1 neuron 33+c: sent2[c] - sent1[c] (positive if sent2 has more)
        for c in range(33):
            if c < d_ff:
                # Neuron c: sent1[c] - sent2[c]
                mlp1.fc1.weight[c, 64 + c] = 1.0    # +sent1[c]
                mlp1.fc1.weight[c, c] = -1.0        # -sent2[c]
                mlp1.fc1.bias[c] = 0.0
                
                # Neuron 33+c: sent2[c] - sent1[c]
                if 33 + c < d_ff:
                    mlp1.fc1.weight[33 + c, c] = 1.0        # +sent2[c]
                    mlp1.fc1.weight[33 + c, 64 + c] = -1.0  # -sent1[c]
                    mlp1.fc1.bias[33 + c] = 0.0
        
        # fc2: sum all absolute differences → dim 97 (negative = similar)
        for c in range(33):
            if c < d_ff:
                mlp1.fc2.weight[97, c] = -1.0      # sent1 excess
                if 33 + c < d_ff:
                    mlp1.fc2.weight[97, 33 + c] = -1.0  # sent2 excess
        mlp1.fc2.bias[97] = 10.0  # positive bias (threshold)
        
        # ===== LAYER 2: Amplify and threshold =====
        attn2 = model.blocks[2].attn
        attn2.W_q.weight.zero_()
        attn2.W_k.weight.zero_()
        attn2.W_v.weight.zero_()
        attn2.W_o.weight.zero_()
        
        mlp2 = model.blocks[2].mlp
        mlp2.fc1.weight.zero_()
        mlp2.fc1.bias.zero_()
        mlp2.fc2.weight.zero_()
        mlp2.fc2.bias.zero_()
        
        # Pass through and amplify dim 97
        mlp2.fc1.weight[0, 97] = 1.0
        mlp2.fc1.bias[0] = 0.0
        mlp2.fc2.weight[97, 0] = 10.0  # amplify
        
        # ===== OUTPUT HEAD =====
        model.head.weight.zero_()
        # Token 32 = '1', Token 31 = '0'
        # Dim 97: positive → predict '1'
        model.head.weight[32, 97] = 10.0   # '1' when similar
        model.head.weight[31, 97] = -10.0  # '0' when different
    
    return


model_shorthand_name = "CharFreqDist_v1"
model_description = "Character frequency distribution matching: compare L1 distance between char frequency vectors."


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
