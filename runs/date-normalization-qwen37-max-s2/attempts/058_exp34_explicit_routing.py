"""
Interpretable transformer for character-level sequence tasks.

Task: date-normalization — convert dates to YYYY-MM-DD format.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
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
# Architecture
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
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 48,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
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
# Weight assignment
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Attempt 34: Explicit position routing with correct offsets.
    
    Prompt is always 32 chars ending with '='.
    Output generation starts at position 32.
    
    Strategy: Use one-hot position embeddings + attention to copy tokens from 
    specific input positions to output positions.
    
    For now, focus on YYYY.MM.DD format where input starts at pos 0:
    - Input: 2 0 1 6 . 0 4 . 1 0 (positions 0-9)
    - Output: 2 0 1 6 - 0 4 - 1 0 (positions 32-41)
    """
    torch.manual_seed(3434)
    
    d = model.d_model
    V = model.vocab_size
    vocab = task.vocab
    c2i = {c: i for i, c in enumerate(vocab)}
    H = model.n_heads
    dh = d // H
    
    with torch.no_grad():
        # === Token embeddings: one-hot in first V=43 dimensions ===
        model.token_emb.weight.zero_()
        for i in range(V):
            model.token_emb.weight[i, i] = 1.0
        
        # === Position embeddings: one-hot position encoding in dims 43..85 (43 positions) ===
        model.pos_emb.weight.zero_()
        MAX_POS = min(model.max_seq_len, d - V)  # 128 - 43 = 85
        for pos in range(MAX_POS):
            model.pos_emb.weight[pos, V + pos] = 10.0  # very strong position signal
        
        # === Layer 1: Attention copies by position matching ===
        layer = model.blocks[0]
        
        layer.attn.W_q.weight.zero_()
        layer.attn.W_k.weight.zero_()
        layer.attn.W_v.weight.zero_()
        layer.attn.W_o.weight.zero_()
        
        # === Head 0: Route from input positions to output positions ===
        # For output pos 32, we want it to attend to input pos 0
        # Q at output pos 32 reads pos_emb[32] which has activation at dim 43+32=75
        # K at input pos 0 reads pos_emb[0] which has activation at dim 43+0=43
        # For these to match, we need to cross-wire Q and K!
        
        # Q for head 0: read from position dimensions
        # At output pos p, Q picks up activation at dim V+p
        # We want this to match K at input pos q where q is the routing target
        
        # Key insight: We need Q at pos p to activate on the SAME dimension as K at pos q.
        # pos_emb at pos p has activation at dim (V+p).
        # pos_emb at pos q has activation at dim (V+q).
        # So Q_head0 at pos p should project dim (V+p) -> head_dim (V+q),
        # and K_head0 at pos q should project dim (V+q) -> head_dim (V+q).
        
        # This means Q needs to MOVE from dim (V+p) to some shared dimension.
        # But W_q is a fixed matrix - it can't depend on the position!
        
        # Solution: Q projects all position dims to a single routing dimension,
        # and the position information is used to select which input to attend to.
        
        # Actually, the SIMPLEST solution: 
        # - Q reads position dim (V+p) and maps to head dim 0 with weight proportional to p
        # - K reads position dim (V+q) and maps to head dim 0 with weight proportional to q
        # - High dot product when p ≈ q (same position = self-attention)
        
        # For ROUTING (different input/output positions), we need a different approach.
        # Let's use: Q at output pos p maps pos_dim (V+p) -> head_dim j,
        #            K at input pos q maps pos_dim (V+q) -> head_dim j,
        #            so Q·K is large when output_pos p's Q projection matches input_pos q's K projection.
        
        # For YYYY.MM.DD, the routing is:
        # out_pos 32 -> in_pos 0 (year digit 1)
        # out_pos 33 -> in_pos 1 (year digit 2)
        # out_pos 34 -> in_pos 2 (year digit 3)
        # out_pos 35 -> in_pos 3 (year digit 4)
        # out_pos 36 -> constant '-'
        # out_pos 37 -> in_pos 5 (month digit 1)
        # out_pos 38 -> in_pos 6 (month digit 2)
        # out_pos 39 -> constant '-'
        # out_pos 40 -> in_pos 8 (day digit 1)
        # out_pos 41 -> in_pos 9 (day digit 2)
        
        # Use 10 routing channels (one per output char), one per pair of dimensions in head 0
        # Head 0 dims: 0..dh-1
        
        # For each (out_pos, in_pos) pair, create a routing dimension
        # Q at out_pos activates on routing_dim[out_pos]
        # K at in_pos activates on routing_dim[out_pos]
        # This means multiple in_pos would activate on same routing dim... that's bad.
        
        # Better: For each routing pair (out_pos, in_pos):
        # Q at out_pos: W_q[routing_dim, V+out_pos] = big
        # K at in_pos:  W_k[routing_dim, V+in_pos] = big
        # This gives high attention from out_pos to in_pos.
        # routing_dim = out_pos - 32 (0..9) for the 10 output positions
        
        ROUTE_DIMS = 10  # dims 0..9 of head 0
        
        # Q: for output positions 32-41, create routing query
        for out_offset in range(10):
            out_pos = 32 + out_offset
            route_dim = out_offset  # dim within head 0
            q_dim = V + out_pos   # position dimension in input
            if q_dim < d:
                layer.attn.W_q.weight[route_dim, q_dim] = 20.0
        
        # K: for input positions, create routing keys
        # For YYYY.MM.DD: 
        routing = {0: 0, 1: 1, 2: 2, 3: 3, 5: 5, 6: 6, 8: 8, 9: 9}
        for in_pos, route_dim in routing.items():
            k_dim = V + in_pos
            if k_dim < d:
                layer.attn.W_k.weight[route_dim, k_dim] = 20.0
        
        # V: copy token identity (first V dims)
        for i in range(min(V, dh)):
            layer.attn.W_v.weight[i, i] = 5.0
        
        # O: project head output back to token dims
        for i in range(min(V, dh)):
            layer.attn.W_o.weight[i, i] = 3.0
        
        # MLP: pass through
        layer.mlp.fc1.weight.zero_()
        layer.mlp.fc1.bias.zero_()
        layer.mlp.fc2.weight.zero_()
        layer.mlp.fc2.bias.zero_()
        for i in range(min(d, layer.mlp.fc1.out_features)):
            layer.mlp.fc1.weight[i, i] = 0.5
        for i in range(min(layer.mlp.fc2.out_features, d)):
            layer.mlp.fc2.weight[i, i] = 0.5
        
        # Output head: strong diagonal
        model.head.weight.zero_()
        for v in range(V):
            model.head.weight[v, v] = 15.0


model_shorthand_name = "exp34-explicit-routing"
model_description = "Explicit position routing: Q/K wired for YYYY.MM.DD copying"


# ---------------------------------------------------------------------------
# Evaluation harness
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
    max_seq_len = max(task.seq_len, 48)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=1,
        d_ff=512,
    )
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
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    if args.debug:
        # Print vocab and task details
        print(f"Vocab ({task.vocab_size}): {task.vocab}")
        print(f"seq_len: {task.seq_len}")
        
        c2i = {c: i for i, c in enumerate(task.vocab)}
        
        # Generate a few examples and show what model generates
        examples = task.generate_examples(3)
        for i, ex in enumerate(examples):
            print(f"\n=== Example {i+1} ===")
            print(f"  prompt: '{ex.prompt}'")
            print(f"  answer: '{ex.answer}'")
            print(f"  prompt len: {len(ex.prompt)}")
            
            # Encode and generate
            prompt_ids = [c2i[c] for c in ex.prompt]
            input_tensor = torch.tensor([prompt_ids], device=args.device)
            
            # Generate 10 tokens
            for step in range(10):
                with torch.no_grad():
                    logits = model(input_tensor)
                    next_token = torch.argmax(logits[0, -1, :]).item()
                    input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=args.device)], dim=1)
            
            # Decode generated tokens (after prompt)
            generated_ids = input_tensor[0, len(prompt_ids):].tolist()
            generated_text = ''.join([task.vocab[id] for id in generated_ids])
            print(f"  generated: '{generated_text}'")
            
            if i >= 2:
                break

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
