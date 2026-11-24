"""
Interpretable transformer for TREC coarse question classification.
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


def write_weights(model: SimpleTransformer, task) -> None:
    """
    Use attention to focus on first few positions where question words appear.
    Encode category predictions based on question word patterns.
    """
    torch.manual_seed(42)
    
    with torch.no_grad():
        stoi = task.stoi
        d = model.d_model  # 64
        n_heads = model.n_heads  # 4
        dh = d // n_heads  # 16
        
        model.token_emb.weight.zero_()
        model.pos_emb.weight.zero_()
        
        # === STRATEGY ===
        # Head 0 of layer 0: focus on positions 0-5 (question word)
        # The V values carry category signals
        # Layer 1 MLP: non-linear classification
        # Output head: read out category
        
        # --- Position embeddings ---
        # Use dim 0 for "is_early" signal (positions 0-7)
        for p in range(model.max_seq_len):
            if p <= 7:
                model.pos_emb.weight[p, 0] = 3.0  # "I am early"
            else:
                model.pos_emb.weight[p, 0] = -3.0  # "I am late"
        
        # Position 81 (answer position) gets a strong query signal
        model.pos_emb.weight[81, 1] = 10.0  # "I want early positions"
        
        # --- Token embeddings ---
        # Encode question word category signals
        # Categories: 0=ABBR, 1=DESC, 2=ENTY, 3=HUM, 4=LOC, 5=NUM
        
        # Character-based bigram features for question words
        # We need to distinguish: who, what, where, when, how, which
        
        # Use dimensions 2-7 for category signals
        # dim 2: HUM, dim 3: LOC, dim 4: NUM, dim 5: DESC, dim 6: ENTY, dim 7: ABBR
        
        # 'w' + next char determines the question word
        # 'wh' + third char: 'o'->who(HUM), 'a'->what(DESC/ENTY), 'e'->where/when(LOC/NUM), 'i'->which(ENTY)
        
        # Give each character in question words category signals
        char_signals = {}
        
        # who -> HUM (dim 2)
        for c in 'who':
            if c not in char_signals:
                char_signals[c] = {}
        
        # Instead of per-character, let me be more strategic
        # The third character after "wh" is most discriminative:
        # 'o' (who) -> HUM
        # 'a' (what) -> DESC or ENTY or HUM or LOC or NUM
        # 'e' (where/when) -> LOC or NUM
        # 'i' (which) -> ENTY or HUM
        
        # 'h' (how) -> NUM or DESC
        
        # Let me encode in token embeddings the character identity
        # and use attention to look at the 3rd character specifically
        
        # Use dims 8-33 for one-hot encoding of characters (26 chars)
        for i in range(26):
            model.token_emb.weight[i, 8 + i] = 1.0
        
        # Underscore (padding) gets a special signal
        model.token_emb.weight[stoi['_'], :] = 0.0  # No signal for padding
        model.token_emb.weight[stoi['='], :] = 0.0  # No signal for =
        
        # --- Layer 0 Attention ---
        block0 = model.blocks[0]
        block0.attn.W_q.weight.zero_()
        block0.attn.W_k.weight.zero_()
        block0.attn.W_v.weight.zero_()
        block0.attn.W_o.weight.zero_()
        
        # Head 0: Q looks for "is_early" (dim 0), K projects "is_early" (dim 0)
        # This makes position 81 attend strongly to positions 0-7
        # Q: dim 1 (query signal) -> head 0's query dim
        head0_q_start = 0
        head0_k_start = 0
        head0_v_start = 0
        
        # Q: project dim 1 (the "want early" signal) to head 0 query
        block0.attn.W_q.weight[head0_q_start, 1] = 1.0
        # K: project dim 0 (the "is early" signal) to head 0 key
        block0.attn.W_k.weight[head0_k_start, 0] = 1.0
        
        # V for head 0: pass through character identity (dims 8-33)
        for i in range(min(dh, 26)):
            block0.attn.W_v.weight[head0_v_start + i, 8 + i] = 1.0
        
        # O: pass head 0 output to dims 34-49
        for i in range(dh):
            block0.attn.W_o.weight[34 + i, head0_v_start + i] = 1.0
        
        # Heads 1-3: identity-like pass-through
        for h in range(1, n_heads):
            q_start = h * dh
            k_start = h * dh
            v_start = h * dh
            for i in range(dh):
                block0.attn.W_q.weight[q_start + i, q_start + i] = 0.01
                block0.attn.W_k.weight[k_start + i, k_start + i] = 0.01
                block0.attn.W_v.weight[v_start + i, v_start + i] = 0.01
                block0.attn.W_o.weight[q_start + i, v_start + i] = 0.01
        
        # Layer 0 MLP: detect 3rd character of question word
        # After attention, dims 34-49 should contain character info from early positions
        # Use MLP to detect specific characters and produce category signals
        block0.mlp.fc1.weight.zero_()
        block0.mlp.fc1.bias.zero_()
        block0.mlp.fc2.weight.zero_()
        block0.mlp.fc2.bias.zero_()
        
        d_ff = block0.mlp.fc1.weight.shape[0]
        
        # Use MLP neurons to detect character patterns
        # fc1 reads from dims 34+ (output of attention), fc2 writes to dims 2-7 (category signals)
        
        # Neuron 0: detect 'o' (from who) -> HUM
        # The character 'o' has index 14, so in dims 8+14=22, which maps to attn output dim 34+14=48
        # After attention, dim 48 should be high if 'o' is attended
        block0.mlp.fc1.weight[0, 48] = 1.0  # 'o' -> who -> HUM
        block0.mlp.fc1.bias[0] = -0.3
        block0.mlp.fc2.weight[2, 0] = 5.0  # HUM signal
        
        # Neuron 1: detect 'e' (from where/when) -> LOC/NUM
        block0.mlp.fc1.weight[1, 46] = 1.0  # 'e' -> where/when
        block0.mlp.fc1.bias[1] = -0.3
        block0.mlp.fc2.weight[3, 1] = 3.0  # LOC signal
        block0.mlp.fc2.weight[4, 1] = 3.0  # NUM signal
        
        # Neuron 2: detect 'a' (from what) -> DESC/ENTY
        block0.mlp.fc1.weight[2, 42] = 1.0  # 'a' -> what
        block0.mlp.fc1.bias[2] = -0.3
        block0.mlp.fc2.weight[5, 2] = 2.0  # DESC signal
        block0.mlp.fc2.weight[6, 2] = 2.0  # ENTY signal
        
        # Neuron 3: detect 'i' (from which) -> ENTY
        block0.mlp.fc1.weight[3, 50] = 1.0  # 'i' -> which
        block0.mlp.fc1.bias[3] = -0.3
        block0.mlp.fc2.weight[6, 3] = 3.0  # ENTY signal
        
        # Neuron 4: detect 'h' at position 0 (how starts with h, not wh) -> NUM
        block0.mlp.fc1.weight[4, 47] = 1.0  # 'h' -> how
        block0.mlp.fc1.bias[4] = -0.3
        block0.mlp.fc2.weight[4, 4] = 3.0  # NUM signal
        
        # --- Layer 1 ---
        block1 = model.blocks[1]
        # Identity attention to pass through
        block1.attn.W_q.weight.zero_()
        block1.attn.W_k.weight.zero_()
        block1.attn.W_v.weight.zero_()
        block1.attn.W_o.weight.zero_()
        for i in range(d):
            block1.attn.W_q.weight[i, i] = 0.01
            block1.attn.W_k.weight[i, i] = 0.01
            block1.attn.W_v.weight[i, i] = 0.01
            block1.attn.W_o.weight[i, i] = 0.01
        
        # MLP: aggregate category signals
        block1.mlp.fc1.weight.zero_()
        block1.mlp.fc1.bias.zero_()
        block1.mlp.fc2.weight.zero_()
        block1.mlp.fc2.bias.zero_()
        
        # Pass through category signals (dims 2-7)
        for i in range(6):
            block1.mlp.fc1.weight[i, 2 + i] = 1.0
            block1.mlp.fc2.weight[2 + i, i] = 1.0
        
        # --- Layer 2 ---
        block2 = model.blocks[2]
        block2.attn.W_q.weight.zero_()
        block2.attn.W_k.weight.zero_()
        block2.attn.W_v.weight.zero_()
        block2.attn.W_o.weight.zero_()
        for i in range(d):
            block2.attn.W_q.weight[i, i] = 0.01
            block2.attn.W_k.weight[i, i] = 0.01
            block2.attn.W_v.weight[i, i] = 0.01
            block2.attn.W_o.weight[i, i] = 0.01
        
        block2.mlp.fc1.weight.zero_()
        block2.mlp.fc1.bias.zero_()
        block2.mlp.fc2.weight.zero_()
        block2.mlp.fc2.bias.zero_()
        
        # --- Output head ---
        # Map category signals (dims 2-7) to output tokens (30-35)
        model.head.weight.zero_()
        for cat in range(6):
            model.head.weight[30 + cat, 2 + cat] = 10.0
    
    return


model_shorthand_name = "exp04_focused_attention"
model_description = "Focus attention on positions 0-5 (question words) using position-based attention, detect 3rd char via MLP, classify."


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
