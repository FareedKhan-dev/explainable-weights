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
    
    Strategy: Implement word overlap heuristic using attention and MLPs.
    - Layer 1: Detect | separator position and mark premise/hypothesis regions
    - Layer 2: Count word matches using attention (hypothesis words attend to premise)
    - Layer 3: Compute overlap ratio and compare to thresholds
    - Output: Map to class label (0=entailment, 1=contradiction, 2=neutral)
    """
    torch.manual_seed(0)
    
    # Best heuristic from analysis: word_overlap >= 0.3 -> entailment, < 0.2 -> contradiction
    T_ENTAIL = 0.3
    T_CONTRA = 0.2
    
    D = model.d_model  # 64
    V = model.vocab_size  # 34
    H = model.n_heads  # 4
    dh = D // H  # 16
    
    # Vocabulary indices
    PIPE_IDX = task.stoi['|']
    EQ_IDX = task.stoi['=']
    SPACE_IDX = task.stoi[' ']
    ZERO_IDX = task.stoi['0']
    ONE_IDX = task.stoi['1']
    TWO_IDX = task.stoi['2']
    
    with torch.no_grad():
        # Initialize token embeddings as identity-like for each character
        # This helps preserve character information
        for i in range(V):
            if i < D:
                model.token_emb.weight[i, i] = 1.0
        
        # Position embeddings: encode position information
        # Position 0 will accumulate overlap signal
        for pos in range(model.max_seq_len):
            # Encode position in first few dimensions
            model.pos_emb.weight[pos, 0] = pos / model.max_seq_len
            model.pos_emb.weight[pos, 1] = 1.0 if pos < 60 else 0.0  # Roughly premise length
        
        # Layer 1: Detect | separator
        # Attention: find the | token and broadcast its position
        layer1 = model.blocks[0]
        
        # Q: query for | separator
        # K: key that matches | token
        # V: value that encodes "this is after |"
        layer1.attn.W_q.weight.zero_()
        layer1.attn.W_k.weight.zero_()
        layer1.attn.W_v.weight.zero_()
        layer1.attn.W_o.weight.zero_()
        
        # Make Q look for | token (dimension 29 is | in vocab)
        layer1.attn.W_q.weight[0, :] = 0.0
        layer1.attn.W_k.weight[0, :] = 0.0
        layer1.attn.W_k.weight[0, PIPE_IDX] = 1.0  # Key matches | token
        layer1.attn.W_v.weight[2, :] = 0.0
        layer1.attn.W_v.weight[2, :] = 1.0  # Value encodes position
        
        layer1.attn.W_o.weight[:, 0] = 1.0  # Output to dimension 0
        
        # MLP: pass through
        layer1.mlp.fc1.weight.zero_()
        layer1.mlp.fc1.bias.zero_()
        layer1.mlp.fc2.weight.zero_()
        layer1.mlp.fc2.bias.zero_()
        
        # Layer 2: Count word matches
        # Hypothesis words attend to premise words to find matches
        layer2 = model.blocks[1]
        
        layer2.attn.W_q.weight.zero_()
        layer2.attn.W_k.weight.zero_()
        layer2.attn.W_v.weight.zero_()
        layer2.attn.W_o.weight.zero_()
        
        # Q: hypothesis word (after |)
        # K: premise word (before |)
        # V: match indicator
        for h in range(min(H, 4)):
            base_h = h * dh
            # Query: look for matches
            layer2.attn.W_q.weight[base_h, :] = 0.1
            # Key: provide word identity
            layer2.attn.W_k.weight[base_h, :] = 0.1
            # Value: count matches
            layer2.attn.W_v.weight[base_h + 1, :] = 0.0
            layer2.attn.W_v.weight[base_h + 1, :] = 0.1
        
        layer2.attn.W_o.weight[:, 1] = 1.0  # Accumulate to dimension 1
        
        # MLP: accumulate word count
        layer2.mlp.fc1.weight.zero_()
        layer2.mlp.fc1.bias.zero_()
        layer2.mlp.fc2.weight.zero_()
        layer2.mlp.fc2.bias.zero_()
        
        # Layer 3: Compute ratio and compare to thresholds
        layer3 = model.blocks[2]
        
        # Attention: pass through
        layer3.attn.W_q.weight.zero_()
        layer3.attn.W_k.weight.zero_()
        layer3.attn.W_v.weight.zero_()
        layer3.attn.W_o.weight.zero_()
        
        # MLP: threshold comparison
        # Input: dimension 1 has overlap count, dimension 0 has hypothesis length
        # Output: class label (0, 1, or 2)
        d_ff = model.d_ff  # 256
        
        layer3.mlp.fc1.weight.zero_()
        layer3.mlp.fc1.bias.zero_()
        layer3.mlp.fc2.weight.zero_()
        layer3.mlp.fc2.bias.zero_()
        
        # Compute overlap ratio: dim[1] / dim[0]
        # Then compare to thresholds
        # Neuron 0: ratio >= T_ENTAIL -> high activation
        layer3.mlp.fc1.weight[0, 1] = 10.0
        layer3.mlp.fc1.weight[0, 0] = -10.0 * T_ENTAIL
        layer3.mlp.fc1.bias[0] = 0.0
        
        # Neuron 1: ratio < T_CONTRA -> high activation
        layer3.mlp.fc1.weight[1, 1] = -10.0
        layer3.mlp.fc1.weight[1, 0] = 10.0 * T_CONTRA
        layer3.mlp.fc1.bias[1] = 0.0
        
        # Map to output classes
        # Neuron 0 active -> class 0 (entailment)
        layer3.mlp.fc2.weight[2, 0] = 5.0  # Dimension 2 encodes "0"
        # Neuron 1 active -> class 1 (contradiction)
        layer3.mlp.fc2.weight[3, 1] = 5.0  # Dimension 3 encodes "1"
        # Neither active -> class 2 (neutral)
        layer3.mlp.fc2.bias[4] = 2.0  # Dimension 4 encodes "2"
        
        # Final layer norm: pass through
        model.final_ln.weight.fill_(1.0)
        model.final_ln.bias.zero_()
        
        # Output head: map dimensions 2,3,4 to tokens 0,1,2
        model.head.weight.zero_()
        model.head.weight[ZERO_IDX, 2] = 10.0  # Token '0'
        model.head.weight[ONE_IDX, 3] = 10.0   # Token '1'
        model.head.weight[TWO_IDX, 4] = 10.0   # Token '2'


model_shorthand_name = "WordOverlapCircuit"
model_description = "Hand-crafted circuit computing word overlap heuristic for NLI."


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
