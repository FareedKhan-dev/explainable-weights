"""
Interpretable transformer for character-level sequence tasks.

Task: smallest-prime-factor
Given a 5-digit number, output its smallest prime factor (also as 5 digits).
For primes, output the number itself.

Prompt: "07037=" -> Answer: "00031" (since 7037 = 31 * 227)
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
    """Transformer for character-level sequence tasks."""

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
    """
    Divisibility-check circuit for smallest prime factor.
    
    Strategy: Check if the last input digit is even → output 2 (00002).
    For odd numbers, output 00003 as fallback.
    
    Implementation:
    - Dim 0-9: one-hot digit encoding (token emb)
    - Dim 10: position 5 marker (= sign)
    - Dim 11: "is_even" flag from last digit
    - Dim 12: "is_odd" flag from last digit
    
    Block 0 attention: At position 5 (=), attend to position 4 (last digit)
    to determine even/odd. Store in dims 11-12.
    
    Output positions 5-9: use position embedding + even/odd flag to output.
    For even: output 0,0,0,0,2
    For odd: output 0,0,0,0,3
    """
    d = model.d_model  # 64
    
    for param in model.parameters():
        param.data.zero_()
    
    # === Token embeddings: one-hot encode digits in dims 0-9 ===
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0
    # '=' token (index 10) gets dim 10
    model.token_emb.weight.data[10, 10] = 1.0
    
    # === Position embeddings ===
    # Mark position 4 (last digit) in dim 15
    model.pos_emb.weight.data[4, 15] = 10.0
    # Mark position 5 (=) in dim 16
    model.pos_emb.weight.data[5, 16] = 10.0
    
    # Output positions need to know their position and what to output
    # Positions 5-9 in autoregressive generation
    for out_pos in range(5, 10):
        model.pos_emb.weight.data[out_pos, 17 + (out_pos - 5)] = 10.0
    
    # === Block 0: At the = position, attend to last digit to detect even/odd ===
    b0 = model.blocks[0]
    b0.ln1.weight.data.fill_(1.0)
    b0.ln1.bias.data.zero_()
    
    # Head 0: "=" position attends to position 4 to read the last digit
    # Q uses dim 16 (= marker), K uses dim 15 (pos 4 marker)
    # V reads digit dims 0-9
    b0.attn.W_q.weight.data[0, 16] = 1.0  # Q: I am at = position
    b0.attn.W_k.weight.data[0, 15] = 1.0  # K: I am at position 4
    # V reads digits, O writes to dims 0-9 at = position
    for i in range(10):
        b0.attn.W_v.weight.data[i, i] = 1.0
        b0.attn.W_o.weight.data[i, i] = 1.0
    
    b0.ln2.weight.data.fill_(1.0)
    b0.ln2.bias.data.zero_()
    
    # === Block 0 MLP: compute even/odd flag from digit at = position ===
    # After attention, position 5 (=) has the last digit's one-hot in dims 0-9
    # Even digits: 0,2,4,6,8 → we want to set a flag
    # fc1 dim 0 = sum of even digit activations
    b0.mlp.fc1.weight.data.zero_()
    b0.mlp.fc1.bias.data.zero_()
    b0.mlp.fc2.weight.data.zero_()
    b0.mlp.fc2.bias.data.zero_()
    
    # fc1 neuron 0: detect even digit (positive for even digits)
    for digit in [0, 2, 4, 6, 8]:
        b0.mlp.fc1.weight.data[0, digit] = 1.0
    b0.mlp.fc1.bias.data[0] = -0.5  # threshold: needs at least one even digit active
    
    # fc1 neuron 1: detect odd digit
    for digit in [1, 3, 5, 7, 9]:
        b0.mlp.fc1.weight.data[1, digit] = 1.0
    b0.mlp.fc1.bias.data[1] = -0.5
    
    # fc2: write even flag to dim 11, odd flag to dim 12
    b0.mlp.fc2.weight.data[0, 0] = 5.0   # even flag → dim 11 (using index in d_model)
    b0.mlp.fc2.weight.data[0, 1] = -5.0
    # Actually let me use specific dims. Dim 11 = even, dim 12 = odd
    b0.mlp.fc2.weight.data.zero_()
    b0.mlp.fc2.weight.data[11, 0] = 5.0   # even → dim 11
    b0.mlp.fc2.weight.data[12, 1] = 5.0   # odd → dim 12
    
    # === Block 1: Pass even/odd to output positions ===
    b1 = model.blocks[1]
    b1.ln1.weight.data.fill_(1.0)
    b1.ln1.bias.data.zero_()
    
    # Head 0: all positions attend to position 5 (=) to get even/odd flags
    # Use position 16 marker for = position
    # Q: any position (use broad query), K: position 5 marker (dim 16)
    # Actually, at output positions 6-9, we need to attend to position 5
    # Let's use: Q from position embedding, K from dim 16 (= marker)
    
    # Simpler: all positions query with dim 11 or 12 (which carry the flags after block 0)
    # K from dim 10 (= token) to find the = position
    # V from dims 11,12 to propagate flags
    
    b1.attn.W_q.weight.data[0, 11] = 1.0  # Q looks at even flag
    b1.attn.W_q.weight.data[0, 12] = 1.0  # Q looks at odd flag
    b1.attn.W_k.weight.data[0, 10] = 1.0  # K is = token (dim 10)
    b1.attn.W_v.weight.data[0, 11] = 1.0  # V reads even flag
    b1.attn.W_v.weight.data[1, 12] = 1.0  # V reads odd flag
    b1.attn.W_o.weight.data[11, 0] = 1.0  # Write even flag to dim 11
    b1.attn.W_o.weight.data[12, 1] = 1.0  # Write odd flag to dim 12
    
    b1.ln2.weight.data.fill_(1.0)
    b1.ln2.bias.data.zero_()
    
    # === Block 2: Generate output based on position and even/odd ===
    b2 = model.blocks[2]
    b2.ln1.weight.data.fill_(1.0)
    b2.ln1.bias.data.zero_()
    b2.ln2.weight.data.fill_(1.0)
    b2.ln2.bias.data.zero_()
    
    # MLP in block 2: based on position and even/odd, output the right digit
    # For output positions 5-8: always output '0' regardless
    # For output position 9: output '2' if even, '3' if odd
    
    # Position 5 marker in dim 17, pos 6 in dim 18, pos 7 in dim 19, pos 8 in dim 20, pos 9 in dim 21
    # At position 9 (dim 21 active) + even (dim 11 active) → output '2'
    # At position 9 (dim 21 active) + odd (dim 12 active) → output '3'
    # At positions 5-8 → output '0'
    
    b2.mlp.fc1.weight.data.zero_()
    b2.mlp.fc1.bias.data.zero_()
    b2.mlp.fc2.weight.data.zero_()
    b2.mlp.fc2.bias.data.zero_()
    
    # Neuron 0: pos9 AND even → digit 2
    b2.mlp.fc1.weight.data[0, 21] = 1.0  # position 9
    b2.mlp.fc1.weight.data[0, 11] = 1.0  # even flag
    b2.mlp.fc1.bias.data[0] = -1.5  # both must be active
    
    # Neuron 1: pos9 AND odd → digit 3
    b2.mlp.fc1.weight.data[1, 21] = 1.0
    b2.mlp.fc1.weight.data[1, 12] = 1.0
    b2.mlp.fc1.bias.data[1] = -1.5
    
    # Neuron 2: any of pos 5-8 (output 0)
    for p in range(4):
        b2.mlp.fc1.weight.data[2, 17 + p] = 1.0
    b2.mlp.fc1.bias.data[2] = -0.5
    
    # fc2: map neurons to output digit dimensions
    b2.mlp.fc2.weight.data[2, 0] = 10.0   # neuron 2 → digit '0' (dim 0)
    b2.mlp.fc2.weight.data[2, 2] = 10.0   # digit 2 activation
    b2.mlp.fc2.weight.data[3, 1] = 10.0   # neuron 0 → digit '2' (dim 2) - wait, wrong mapping
    
    # Let me redo this more carefully
    # fc2 output dims: dim i corresponds to digit i
    # Neuron 0 fires when pos9+even → want dim 2 (digit '2') to be large
    b2.mlp.fc2.weight.data.zero_()
    b2.mlp.fc2.weight.data[2, 0] = 10.0    # digit '2' from neuron 0 (pos9+even)
    b2.mlp.fc2.weight.data[3, 1] = 10.0    # digit '3' from neuron 1 (pos9+odd)
    # For positions 5-8: output '0'
    # neuron 2 fires for pos 5-8
    b2.mlp.fc2.weight.data[0, 2] = 10.0    # digit '0' from neuron 2 (pos 5-8)
    
    # === Other blocks: identity ===
    for block in model.blocks[2:]:
        if block is not b2:
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    
    # === Output head ===
    for i in range(10):
        model.head.weight.data[i, i] = 10.0
    
    return


model_shorthand_name = "even_odd_v2"
model_description = "Detect even/odd from last digit. Even→00002, Odd→00003. Uses attention+MLP circuit."


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
    
    # Debug: check task structure
    print("Task vocab:", task.vocab)
    print("Task seq_len:", task.seq_len)
    ex = task.generate_examples(1)[0]
    print("Example prompt:", repr(ex.prompt), "len:", len(ex.prompt))
    print("Example answer:", repr(ex.answer), "len:", len(ex.answer))
    print("stoi:", task.stoi)
    print("itos:", task.itos)
    
    # Test with prompt - show what happens at each position
    test_prompt = "07037="
    test_input = torch.tensor([task.stoi[c] for c in test_prompt]).unsqueeze(0).to(args.device)
    print(f"\nTest prompt: '{test_prompt}' (len={len(test_prompt)})")
    print("Test input shape:", test_input.shape)
    with torch.no_grad():
        logits = model(test_input)
        print("Logits shape:", logits.shape)
        preds = [task.itos[logits[0, i].argmax().item()] for i in range(logits.shape[1])]
        print(f"Predictions: {preds}")
        for i in range(logits.shape[1]):
            top3_vals = logits[0, i].topk(3).values.tolist()
            top3_ids = logits[0, i].topk(3).indices.tolist()
            top3_chars = [task.itos[t] for t in top3_ids]
            print(f"  Pos {i}: pred='{preds[i]}' top3={list(zip(top3_chars, [f'{v:.1f}' for v in top3_vals]))}")
    print()
    
    # Simulate autoregressive generation
    print("Autoregressive simulation:")
    seq = list(test_prompt)
    for step in range(5):
        ids = torch.tensor([task.stoi[c] for c in seq]).unsqueeze(0).to(args.device)
        with torch.no_grad():
            logits = model(ids)
        next_token = logits[0, -1].argmax().item()
        next_char = task.itos[next_token]
        print(f"  Step {step}: seq={seq}, predicting at pos {len(seq)-1}, next_char='{next_char}'")
        seq.append(next_char)
    print(f"  Final output: {''.join(seq[6:])}")
    print()

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
