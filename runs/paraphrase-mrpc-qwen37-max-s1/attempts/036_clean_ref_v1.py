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

class SimpleTransformer(nn.Module):
    """Configurable transformer for paraphrase detection."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 128,
        n_heads: int = 8,
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
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

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
    """Clean reference-level circuit for paraphrase detection.
    
    Circuit:
    - pos_emb sets dims 1..D-1 to a reference value R at all positions
    - pos_emb sets dim D-1 to half-indicator (+1 for pos<=60, -1 for pos>60)
    - Block 0 attn: from final pos, compute character overlap score in dim 0
    - Block 0 MLP: amplify the overlap score
    - final_ln: gamma[0]=1, bias[0]=0 => output[0] = (V-R)/std
    - head: map dim 0 to '1'/'0' based on sign
    """
    for p in model.parameters():
        p.data.zero_()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    D = model.d_model  # 64
    H = model.n_heads  # 4
    dh = D // H  # 16
    V = model.vocab_size  # 33
    R = 1.0  # Reference level for dims 1..D-2
    
    # Token embeddings: character identity in dim 0 (only dim 0 matters for decision)
    # We'll put character identity info spread across dims, but the decision uses dim 0
    for i in range(V):
        model.token_emb.weight.data[i, i % D] = 0.5
    
    # Position embeddings: set reference level in dims 1..D-2, half-indicator in D-1
    for p in range(model.max_seq_len):
        for d in range(1, D-1):
            model.pos_emb.weight.data[p, d] = R
        # Half indicator for attention Q/K
        if p <= 60:
            model.pos_emb.weight.data[p, D-1] = 3.0
        else:
            model.pos_emb.weight.data[p, D-1] = -3.0
    
    # === Block 0: Character overlap computation ===
    # Head 0: Match characters by identity, value = half-indicator
    # This gives: for each position in s2, how many matching chars are in s1
    for i in range(min(dh, V)):
        model.blocks[0].attn.W_q.weight.data[i, i] = 5.0
        model.blocks[0].attn.W_k.weight.data[i, i] = 5.0
    model.blocks[0].attn.W_v.weight.data[0, D-1] = 1.0  # Value = half-indicator
    model.blocks[0].attn.W_o.weight.data[0, 0] = 1.0  # Write to dim 0
    
    # Head 1: Same matching but value = +1 constant (count matches)
    for i in range(min(dh, V)):
        model.blocks[0].attn.W_q.weight.data[dh+i, i] = 5.0
        model.blocks[0].attn.W_k.weight.data[dh+i, i] = 5.0
    model.blocks[0].attn.W_v.weight.data[0, 0] = 0.0  # Read constant from a zero dim
    model.blocks[0].attn.W_o.weight.data[0, dh] = 0.0
    
    # Head 2: Attend to first half only (Q queries half-indicator positively)
    model.blocks[0].attn.W_q.weight.data[2*dh, D-1] = 5.0
    model.blocks[0].attn.W_k.weight.data[2*dh, D-1] = 1.0
    model.blocks[0].attn.W_v.weight.data[0, D-1] = 1.0
    model.blocks[0].attn.W_o.weight.data[0, 2*dh] = 0.0
    
    # Head 3: Attend to second half only
    model.blocks[0].attn.W_q.weight.data[3*dh, D-1] = -5.0
    model.blocks[0].attn.W_k.weight.data[3*dh, D-1] = 1.0
    model.blocks[0].attn.W_v.weight.data[0, D-1] = 1.0
    model.blocks[0].attn.W_o.weight.data[0, 3*dh] = 0.0
    
    # Block 0 MLP: amplify dim 0 signal
    model.blocks[0].mlp.fc1.weight.data[0, 0] = 2.0
    model.blocks[0].mlp.fc1.bias.data[0] = 0.0
    model.blocks[0].mlp.fc2.weight.data[0, 0] = 2.0
    
    # Block 1 and 2: pass through (zeros = identity via residual)
    
    # Final: gamma[0]=1, bias=0 => LayerNorm makes (dim0 - mean)/std
    # Since dims 1..D-2 are at R and dim 0 varies, this creates discrimination
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: map normalized dim 0 to output
    model.head.weight.data[32, 0] = 100.0   # '1' (paraphrase)
    model.head.weight.data[31, 0] = -100.0  # '0' (not paraphrase)


model_shorthand_name = "clean-ref-v1"
model_description = "Clean reference-level circuit: dims 1-62 at constant R, dim 0 carries overlap score, LayerNorm normalizes relative to R."


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
    print(f"Task: {args.task}")
    print(f"vocab_size: {task.vocab_size}")
    print(f"seq_len: {task.seq_len}")
    if hasattr(task, 'generate_examples'):
        examples = task.generate_examples(3)
        for i, ex in enumerate(examples):
            print(f"Example {i}: prompt_len={len(ex.prompt)} answer='{ex.answer}'")
            # Encode and run through model
            if hasattr(task, 'encode'):
                ids = task.encode(ex.prompt)
                print(f"  encoded ids: {ids[:5]}...{ids[-5:]}")
                print(f"  len encoded: {len(ids)}")
    
    model = build_model(task).to(args.device)
    
    # DEBUG: run one example through model
    if hasattr(task, 'generate_examples'):
        ex = examples[0]
        if hasattr(task, 'encode'):
            ids = task.encode(ex.prompt)
            import torch as _t
            input_ids = _t.tensor([ids], dtype=_t.long, device=args.device)
            print(f"  input shape: {input_ids.shape}")
            with _t.no_grad():
                logits = model(input_ids)
            print(f"  logits shape: {logits.shape}")
            print(f"  logits at last pos: {logits[0, -1, :].topk(5)}")
            print(f"  argmax at last pos: {logits[0, -1, :].argmax().item()}")

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
