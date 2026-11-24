"""
Interpretable transformer for character-level sequence tasks.

Task: gcd-three-digits
Prompt "353,555=" -> answer "001" (GCD as 3 digits)

Layout:
  pos 0-2: first number digits (hundreds, tens, ones)
  pos 3:   ','
  pos 4-6: second number digits (hundreds, tens, ones)
  pos 7:   '='
  pos 8-10: answer digits (autoregressively generated)

But generation is autoregressive: model sees prompt (8 tokens), predicts at pos 7,
then feeds that back, predicts at pos 8, etc. So answers are at pos 7,8,9.
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
    """Transformer with configurable architecture."""

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
    Combine heuristics: if both even -> 002, else if both end in 0/5 -> 005, else -> 001.
    
    Uses two attention heads to detect both conditions, then MLP to combine.
    """
    torch.manual_seed(0)
    
    d = model.d_model
    V = model.vocab_size
    
    # Token embeddings: mark even digits and 0/5 digits
    model.token_emb.weight.data.zero_()
    even_digits = [0, 2, 4, 6, 8]
    div5_digits = [0, 5]
    
    for digit_idx in even_digits:
        model.token_emb.weight.data[digit_idx, 0] = 5.0  # is even
    for digit_idx in div5_digits:
        model.token_emb.weight.data[digit_idx, 1] = 5.0  # ends in 0/5
    
    # Position embeddings
    model.pos_emb.weight.data.zero_()
    
    # Mark ones digit positions (2 and 6)
    model.pos_emb.weight.data[2, 10] = 5.0
    model.pos_emb.weight.data[6, 10] = 5.0
    
    # Mark query position
    model.pos_emb.weight.data[7, 11] = 5.0
    
    # Default answer: 001
    model.pos_emb.weight.data[7, 30] = 10.0
    model.pos_emb.weight.data[8, 30] = 10.0
    model.pos_emb.weight.data[9, 31] = 10.0
    
    # Zero blocks, fix LayerNorm
    for block in model.blocks:
        for p in block.parameters():
            p.data.zero_()
        block.ln1.weight.data.fill_(1.0)
        block.ln1.bias.data.zero_()
        block.ln2.weight.data.fill_(1.0)
        block.ln2.bias.data.zero_()
    
    block0 = model.blocks[0]
    
    # Head 0: detect both even
    # Q from dim 11 (query pos), K from dim 10 (ones digits), V from dim 0 (even signal)
    block0.attn.W_q.weight.data[0, 11] = 1.0
    block0.attn.W_k.weight.data[0, 10] = 1.0
    block0.attn.W_v.weight.data[0, 0] = 1.0
    block0.attn.W_o.weight.data[20, 0] = 1.0
    
    # Head 1: detect both div by 5
    # Q from dim 11, K from dim 10, V from dim 1 (div5 signal)
    block0.attn.W_q.weight.data[16, 11] = 1.0
    block0.attn.W_k.weight.data[16, 10] = 1.0
    block0.attn.W_v.weight.data[16, 1] = 1.0
    block0.attn.W_o.weight.data[21, 16] = 1.0
    
    # After attention: dim 20 = both even signal, dim 21 = both div5 signal
    
    # MLP: implement priority logic
    # relu(dim20 - 3.0) -> hidden[0] (both even flag)
    block0.mlp.fc1.weight.data[0, 20] = 1.0
    block0.mlp.fc1.bias.data[0] = -3.0
    
    # relu(dim21 - 3.0) -> hidden[1] (both div5 flag)
    block0.mlp.fc1.weight.data[1, 21] = 1.0
    block0.mlp.fc1.bias.data[1] = -3.0
    
    # relu(3.0 - dim20) -> hidden[2] (NOT both even)
    block0.mlp.fc1.weight.data[2, 20] = -1.0
    block0.mlp.fc1.bias.data[2] = 3.0
    
    # fc2: if both even -> dim 40, if (NOT both even) AND both div5 -> dim 41
    block0.mlp.fc2.weight.data[40, 0] = 10.0  # both even flag
    block0.mlp.fc2.weight.data[41, 1] = 10.0  # both div5 flag
    block0.mlp.fc2.weight.data[41, 2] = 10.0  # requires NOT both even
    
    # Output head
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.zero_()
    model.head.weight.data.zero_()
    
    model.head.weight.data[0, 30] = 10.0
    model.head.weight.data[1, 31] = 10.0
    for tok in range(2, V):
        model.head.weight.data[tok, 30] = -10.0
        model.head.weight.data[tok, 31] = -10.0
    
    # If both even (dim 40), predict 002
    model.head.weight.data[2, 40] = 20.0
    model.head.weight.data[1, 40] = -20.0
    
    # If both div5 and NOT both even (dim 41), predict 005
    model.head.weight.data[5, 41] = 20.0
    model.head.weight.data[1, 41] = -20.0


model_shorthand_name = "combined-even-div5"
model_description = "Combined: both even -> 002, else both div5 -> 005, else 001."


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

    if args.verbose:
        # Test with a few examples to debug
        examples = task.generate_examples(5, seed=args.seed)
        for ex in examples:
            print(f"\nExample: {ex.prompt} -> {ex.answer}")
            # Manual forward to inspect
            prompt_tokens = task.encode(ex.prompt)
            prompt_ids = torch.tensor([prompt_tokens], device=args.device)
            
            with torch.no_grad():
                # Embedding
                h = model.token_emb(prompt_ids) + model.pos_emb(torch.arange(prompt_ids.shape[1], device=args.device))
                print(f"  After embedding: pos 2 dim 0 = {h[0,2,0]:.3f}, pos 6 dim 0 = {h[0,6,0]:.3f}")
                
                # Block 0 attention
                block0 = model.blocks[0]
                ln_out = block0.ln1(h)
                print(f"  After LN1: pos 7 dim 2 = {ln_out[0,7,2]:.3f}, pos 2 dim 1 = {ln_out[0,2,1]:.3f}")
                print(f"  After LN1: pos 2 dim 0 = {ln_out[0,2,0]:.3f}, pos 6 dim 0 = {ln_out[0,6,0]:.3f}")
                
                # Attention forward
                B, T, D = ln_out.shape
                Hh, dh = block0.attn.n_heads, block0.attn.d_head
                q = block0.attn.W_q(ln_out).view(B, T, Hh, dh).transpose(1, 2)
                k = block0.attn.W_k(ln_out).view(B, T, Hh, dh).transpose(1, 2)
                v = block0.attn.W_v(ln_out).view(B, T, Hh, dh).transpose(1, 2)
                
                print(f"  Q at pos 7, head 0, dim 0: {q[0,0,7,0]:.3f}")
                print(f"  K at pos 2, head 0, dim 0: {k[0,0,2,0]:.3f}")
                print(f"  K at pos 6, head 0, dim 0: {k[0,0,6,0]:.3f}")
                print(f"  V at pos 2, head 0, dim 0: {v[0,0,2,0]:.3f}")
                print(f"  V at pos 6, head 0, dim 0: {v[0,0,6,0]:.3f}")
                
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=args.device), diagonal=1)
                scores = scores.masked_fill(mask, float("-inf"))
                attn = scores.softmax(dim=-1)
                
                print(f"  Attention weights at pos 7, head 0: {attn[0,0,7,:]}")
                
                attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
                attn_out = block0.attn.W_o(attn_out)
                
                print(f"  Attn output at pos 7, dim 10: {attn_out[0,7,10]:.3f}")
                
                h = h + attn_out
                print(f"  After residual: pos 7 dim 10 = {h[0,7,10]:.3f}")
                
                # MLP
                ln2_out = block0.ln2(h)
                print(f"  After LN2: pos 7 dim 10 = {ln2_out[0,7,10]:.3f}")
                
                mlp_hidden = F.relu(block0.mlp.fc1(ln2_out))
                print(f"  MLP hidden[0] at pos 7: {mlp_hidden[0,7,0]:.3f}")
                
                mlp_out = block0.mlp.fc2(mlp_hidden)
                print(f"  MLP output dim 20 at pos 7: {mlp_out[0,7,20]:.3f}")
                
                h = h + mlp_out
                print(f"  After MLP residual: pos 7 dim 20 = {h[0,7,20]:.3f}")
                
                # Final output
                final = model.head(model.final_ln(h))
                print(f"  Logits at pos 7: {final[0,7,:]}")
                pred_tok = final[0,7,:].argmax().item()
                print(f"  Predicted token at pos 7: {pred_tok} ('{task.itos[pred_tok]}')")
            
            # Generate
            current_ids = prompt_ids
            with torch.no_grad():
                for step in range(task.answer_len):
                    logits = model(current_ids)
                    next_token = logits[0, -1, :].argmax().item()
                    current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=args.device)], dim=1)
            pred = [task.itos[i] for i in current_ids[0, prompt_ids.shape[1]:].tolist()]
            print(f"  Full prediction: {''.join(pred)} (target: {ex.answer})")

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
