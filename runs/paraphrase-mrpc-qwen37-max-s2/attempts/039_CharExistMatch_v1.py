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
    
    Attempt 22: Character existence matching with separator aggregation.
    Strategy:
    - Layer 0: Separator (pos 60) aggregates sent1 character frequencies.
    - Layer 1: Query (121) reads separator for sent1, and aggregates sent2.
      Each sent2 position checks if its character exists in sent1 via MLP.
    - Layer 2: Query aggregates match scores, MLP thresholds.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        d = model.d_model  # 128
        vocab_size = model.vocab_size  # 33
        H = model.n_heads  # 4
        dh = d // H  # 32
        
        # ===== TOKEN EMBEDDINGS =====
        # Dims 0-32: one-hot character encoding (strong signal)
        for token_id in range(vocab_size):
            model.token_emb.weight[token_id, :] = 0.0  # clear
            model.token_emb.weight[token_id, token_id] = 5.0  # one-hot
        
        # ===== POSITION EMBEDDINGS =====
        for pos in range(model.max_seq_len):
            model.pos_emb.weight[pos, :] = 0.0  # clear
        
        # Sentence markers (strong signals in specific dims)
        SENT1_DIM = 33   # positions 0-59
        SEP_DIM = 34     # position 60
        SENT2_DIM = 35   # positions 61-120
        QUERY_DIM = 36   # position 121
        
        for pos in range(60):
            model.pos_emb.weight[pos, SENT1_DIM] = 5.0
        model.pos_emb.weight[60, SEP_DIM] = 5.0
        for pos in range(61, 121):
            model.pos_emb.weight[pos, SENT2_DIM] = 5.0
        model.pos_emb.weight[121, QUERY_DIM] = 5.0
        
        # ===== LAYER 0: Separator aggregates sentence 1 =====
        attn0 = model.blocks[0].attn
        # Zero all attention weights first
        attn0.W_q.weight.zero_()
        attn0.W_k.weight.zero_()
        attn0.W_v.weight.zero_()
        attn0.W_o.weight.zero_()
        
        # Head 0: separator (pos 60) attends to sentence 1 (pos 0-59)
        # Q reads SEP_DIM, K reads SENT1_DIM
        h0_q_offset = 0  # head 0 uses dims 0..dh-1 of Q/K/V
        attn0.W_q.weight[h0_q_offset, SEP_DIM] = 3.0  # Q for head 0
        attn0.W_k.weight[h0_q_offset, SENT1_DIM] = 3.0  # K for head 0
        
        # V copies character dims (0-32) into head 0's V space
        for i in range(min(32, dh)):
            attn0.W_v.weight[h0_q_offset + i, i] = 1.0
        
        # Output: head 0 V dims → output dims 0-31 (character frequency summary)
        for i in range(min(32, dh)):
            attn0.W_o.weight[i, h0_q_offset + i] = 1.0
        
        # Zero layer 0 MLP (identity pass-through)
        mlp0 = model.blocks[0].mlp
        mlp0.fc1.weight.zero_()
        mlp0.fc1.bias.zero_()
        mlp0.fc2.weight.zero_()
        mlp0.fc2.bias.zero_()
        
        # Zero LayerNorm params (they default to 1/0 which is fine)
        # LayerNorm already initialized correctly by PyTorch (weight=1, bias=0)
        
        # ===== LAYER 1: Query gets sent1 summary + sent2 characters =====
        attn1 = model.blocks[1].attn
        attn1.W_q.weight.zero_()
        attn1.W_k.weight.zero_()
        attn1.W_v.weight.zero_()
        attn1.W_o.weight.zero_()
        
        # Head 0: query (121) attends to separator (60) to get sent1 summary
        h0_off = 0
        attn1.W_q.weight[h0_off, QUERY_DIM] = 3.0
        attn1.W_k.weight[h0_off, SEP_DIM] = 3.0
        # V copies sent1 summary (dims 0-31 at separator after layer 0)
        for i in range(min(32, dh)):
            attn1.W_v.weight[h0_off + i, i] = 1.0
        # Output to dims 0-31 (sent1 summary)
        for i in range(min(32, dh)):
            attn1.W_o.weight[i, h0_off + i] = 1.0
        
        # Head 1: query (121) attends to sentence 2 (pos 61-120)
        h1_off = dh
        attn1.W_q.weight[h1_off, QUERY_DIM] = 3.0
        attn1.W_k.weight[h1_off, SENT2_DIM] = 3.0
        # V copies character dims from sent2 positions
        for i in range(min(32, dh)):
            attn1.W_v.weight[h1_off + i, i] = 1.0
        # Output to dims 32-63 (sent2 summary)
        for i in range(min(32, dh)):
            attn1.W_o.weight[32 + i, h1_off + i] = 1.0
        
        # Layer 1 MLP: compute character-level similarity
        # At query: dims 0-31 = sent1 summary, dims 32-63 = sent2 summary
        # Compute: for each character c, min(sent1[c], sent2[c]) → overlap
        # Then sum overlaps / sum(sent2) → Jaccard-like score
        mlp1 = model.blocks[1].mlp
        mlp1.fc1.weight.zero_()
        mlp1.fc1.bias.zero_()
        mlp1.fc2.weight.zero_()
        mlp1.fc2.bias.zero_()
        
        # For each character c (0-32):
        # fc1 neuron c: compute sent1[c] - sent2[c] (positive if sent1 has more)
        # After ReLU: max(0, sent1[c] - sent2[c]) = excess in sent1
        # fc1 neuron 33+c: compute sent2[c] - sent1[c] (positive if sent2 has more)
        # After ReLU: max(0, sent2[c] - sent1[c]) = excess in sent2
        
        for c in range(min(33, dh)):
            # Neuron c: sent1[c] - sent2[c]
            mlp1.fc1.weight[c, c] = 1.0       # +sent1[c]
            mlp1.fc1.weight[c, 32 + c] = -1.0  # -sent2[c]
            
            # Neuron 33+c: sent2[c] - sent1[c]
            if 33 + c < model.d_ff:
                mlp1.fc1.weight[33 + c, c] = -1.0       # -sent1[c]
                mlp1.fc1.weight[33 + c, 32 + c] = 1.0   # +sent2[c]
        
        # fc2: sum up all excesses → negative = more different
        # Total L1 distance = sum of all excesses
        for c in range(min(33, dh)):
            mlp1.fc2.weight[64, c] = -1.0      # negative contribution from sent1 excess
            if 33 + c < model.d_ff:
                mlp1.fc2.weight[64, 33 + c] = -1.0  # negative from sent2 excess
        mlp1.fc2.bias[64] = 5.0  # positive bias (threshold)
        
        # ===== LAYER 2: Threshold decision =====
        attn2 = model.blocks[2].attn
        attn2.W_q.weight.zero_()
        attn2.W_k.weight.zero_()
        attn2.W_v.weight.zero_()
        attn2.W_o.weight.zero_()
        # No attention needed - just use MLP
        
        # Layer 2 MLP: threshold the similarity score
        mlp2 = model.blocks[2].mlp
        mlp2.fc1.weight.zero_()
        mlp2.fc1.bias.zero_()
        mlp2.fc2.weight.zero_()
        mlp2.fc2.bias.zero_()
        
        # Dim 64 has the similarity score (high = similar)
        # Pass through ReLU to threshold at 0
        mlp2.fc1.weight[0, 64] = 1.0
        mlp2.fc1.bias[0] = 0.0
        # Output: positive score → predict 1
        mlp2.fc2.weight[64, 0] = 10.0  # amplify
        
        # ===== OUTPUT HEAD =====
        model.head.weight.zero_()
        # Token 32 = '1', Token 31 = '0'
        # Dim 64 has the decision signal: positive → predict 1
        model.head.weight[32, 64] = 10.0   # '1' gets high logit when score is positive
        model.head.weight[31, 64] = -10.0  # '0' gets low logit when score is positive
    
    return


model_shorthand_name = "CharExistMatch_v1"
model_description = "Character existence matching: separator aggregates sent1, query compares char frequencies."


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
