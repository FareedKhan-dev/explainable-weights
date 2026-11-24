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
    """Simpler even detection with direct signal path."""
    torch.manual_seed(0)
    
    d = model.d_model
    V = model.vocab_size  # 12
    
    # Step 1: Encode digit parity in token embeddings
    # Even digits: 0,2,4,6,8 -> dim 0 = 1
    model.token_emb.weight.data.zero_()
    even_digits = [0, 2, 4, 6, 8]
    for digit_idx in even_digits:
        model.token_emb.weight.data[digit_idx, 0] = 1.0
    
    # Step 2: Mark positions of ones digits (2 and 6) with position embeddings
    model.pos_emb.weight.data.zero_()
    model.pos_emb.weight.data[2, 1] = 1.0  # marker for "ones digit of first number"
    model.pos_emb.weight.data[6, 1] = 1.0  # marker for "ones digit of second number"
    model.pos_emb.weight.data[7, 2] = 1.0  # marker for "=" position
    
    # Step 3: Use attention to sum parity signals from positions 2 and 6
    # Position 7 (=) queries, positions 2 and 6 respond
    block0 = model.blocks[0]
    
    # Zero all attention weights
    block0.attn.W_q.weight.data.zero_()
    block0.attn.W_k.weight.data.zero_()
    block0.attn.W_v.weight.data.zero_()
    block0.attn.W_o.weight.data.zero_()
    
    # Query: position 7 (has marker in dim 2) -> head 0 query
    block0.attn.W_q.weight.data[0, 2] = 1.0
    
    # Key: positions 2,6 (have marker in dim 1) -> head 0 key
    block0.attn.W_k.weight.data[0, 1] = 1.0
    
    # Value: parity signal (dim 0) -> head 0 value dim 0
    block0.attn.W_v.weight.data[0, 0] = 1.0
    
    # Output: head 0 -> residual stream dim 5
    block0.attn.W_o.weight.data[5, 0] = 1.0
    
    # Now dim 5 in residual stream at pos 7 = sum of parity from pos 2 and 6
    # If both even: 2.0, if one even: 1.0, if neither: 0.0
    
    # Step 4: MLP threshold - if dim 5 > 1.5, set "predict 002" flag
    block0.mlp.fc1.weight.data.zero_()
    block0.mlp.fc1.bias.data.zero_()
    block0.mlp.fc2.weight.data.zero_()
    
    # fc1: relu(dim5 - 1.5) -> dim 0 of hidden
    block0.mlp.fc1.weight.data[0, 5] = 1.0
    block0.mlp.fc1.bias.data[0] = -1.5
    
    # fc2: if hidden[0] > 0, set dim 10 = "predict 002"
    block0.mlp.fc2.weight.data[10, 0] = 10.0
    
    # Step 5: Use position embeddings to output tokens
    # Positions 7,8,9 need to output the answer
    model.pos_emb.weight.data[7, 15] = 1.0  # first digit marker
    model.pos_emb.weight.data[8, 16] = 1.0  # second digit marker
    model.pos_emb.weight.data[9, 17] = 1.0  # third digit marker
    
    # Final layer norm
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # Head: map to token logits
    model.head.weight.data.zero_()
    
    # Position 7: always predict '0' (token 0)
    model.head.weight.data[0, 15] = 10.0
    
    # Position 8: always predict '0' (token 0)
    model.head.weight.data[0, 16] = 10.0
    
    # Position 9: if dim 10 active, predict '2' (token 2), else '1' (token 1)
    # Default: predict '1'
    model.head.weight.data[1, 17] = 10.0
    # If dim 10 active, boost '2' and suppress '1'
    model.head.weight.data[2, 10] = 10.0
    model.head.weight.data[1, 10] = -10.0


model_shorthand_name = "detect-even-v2"
model_description = "Simplified even detection: check if both last digits even, predict 002 if yes."


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
    
    # Debug: manual autoregressive generation
    if args.verbose:
        ex = task.generate_examples(1)[0]
        print(f"\n=== Debug: Manual autoregressive generation ===")
        print(f"Example: {ex.prompt} -> {ex.answer}")
        
        # Start with prompt
        current_ids = torch.tensor([task.stoi[c] for c in ex.prompt], device=args.device).unsqueeze(0)
        print(f"Step 0: prompt '{ex.prompt}', len={current_ids.shape[1]}")
        
        with torch.no_grad():
            for step in range(task.answer_len):
                logits = model(current_ids)
                next_token_logits = logits[0, -1, :]  # logits for last position
                next_token = next_token_logits.argmax().item()
                pred_char = task.itos[next_token]
                
                print(f"Step {step+1}: seq_len={current_ids.shape[1]}, pos={current_ids.shape[1]-1}")
                print(f"  Top 3 logits: {next_token_logits.topk(3).values.tolist()}")
                print(f"  Top 3 tokens: {[task.itos[i] for i in next_token_logits.topk(3).indices.tolist()]}")
                print(f"  Predicted: '{pred_char}' (idx {next_token})")
                
                # Append predicted token
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=args.device)], dim=1)
                print(f"  New seq: {[task.itos[i] for i in current_ids[0].tolist()]}")
        
        print(f"Final prediction: {[task.itos[i] for i in current_ids[0, task.prompt_len:].tolist()]}")
        print(f"Target: {ex.answer}\n")

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
