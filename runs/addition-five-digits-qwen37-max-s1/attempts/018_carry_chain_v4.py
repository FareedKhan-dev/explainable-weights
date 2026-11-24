"""
Interpretable transformer for 5-digit addition. No LayerNorm.
Simplified with stronger signals and clearer dimension allocation.

Dims (D=128):
  0-9:   HEAD OUTPUT (zero initially, written by output layer)
  10-19: token digit one-hot (S=10)
  20:    is-digit flag (S=10)
  21:    is-first-num (S=10)
  22:    is-second-num (S=10)
  23:    is-output (S=10)
  30-47: position one-hot (P=20)
  50-55: column one-hot (S=10)
  60-69: d1 one-hot [layer 0 attn]
  70-79: d2 one-hot [token_emb]
  80-89: sum%10 one-hot [layer 0 MLP]
  90:    carry flag (0 or S)
  91:    incoming carry (0 or S)
  100-109: final result one-hot [layer 0 MLP, updated by carry layers]
  110:   final carry flag (0 or S)
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
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

S = 10.0  # signal strength (increased)
P = 20.0  # position signal (increased)
W = 20.0  # attention weight (increased)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):
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
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=128, n_heads=4, n_layers=8, d_ff=512):
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
        self.head = nn.Linear(d_model, vocab_size, bias=True)
    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(h)

POS_TO_COL = {0:4,1:3,2:2,3:1,4:0, 6:4,7:3,8:2,9:1,10:0, 12:5,13:4,14:3,15:2,16:1,17:0}
COL_TO_SNDPOS = {0:10, 1:9, 2:8, 3:7, 4:6}
COL_TO_FSTPOS = {0:4, 1:3, 2:2, 3:1, 4:0}
COL_TO_OUTPOS = {0:17, 1:16, 2:15, 3:14, 4:13, 5:12}

def write_weights(model, task):
    D = model.d_model; V = model.vocab_size; H = model.n_heads; dh = D // H
    with torch.no_grad():
        # TOKEN EMB
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, 10+d] = S
            model.token_emb.weight[d, 20] = S
            model.token_emb.weight[d, 70+d] = S

        # POS EMB
        model.pos_emb.weight.zero_()
        for p in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[p, 30+p] = P
        for pos, col in POS_TO_COL.items():
            if pos < model.max_seq_len:
                model.pos_emb.weight[pos, 50+col] = S
        for p in range(5): model.pos_emb.weight[p, 21] = S
        for p in range(6, 11): model.pos_emb.weight[p, 22] = S
        for p in range(12, 18): model.pos_emb.weight[p, 23] = S

        _layer0(model, D, H, dh)
        for li in range(1, 6):
            _carry_layer(model, li, D, H, dh)
        _output_layer(model, 6, D, H, dh)
        _identity(model, 7)

        # HEAD: read dims 0-9 with strong weight, small bias
        model.head.weight.zero_(); model.head.bias.zero_()
        for d in range(min(10, V)):
            model.head.weight[d, d] = 20.0
            model.head.bias[d] = 0.0
        if V > 10: model.head.bias[10] = -200.0
        if V > 11: model.head.bias[11] = -200.0

def _layer0(model, D, H, dh):
    blk = model.blocks[0]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Head 0: column + type matching
        for c in range(6):
            blk.attn.W_q.weight[c, 50+c] = W
            blk.attn.W_k.weight[c, 50+c] = W
        blk.attn.W_q.weight[6, 22] = W
        blk.attn.W_k.weight[6, 21] = W

        # V: 10-19 -> 60-69
        for d in range(10):
            blk.attn.W_v.weight[60+d, 10+d] = 1.0
            blk.attn.W_o.weight[60+d, d] = 1.0

        # MLP: sum
        u = 0
        for d1 in range(10):
            for d2 in range(10):
                if u >= model.d_ff: break
                s = d1 + d2; r = s % 10; c = 1 if s >= 10 else 0
                blk.mlp.fc1.weight[u, 60+d1] = 20.0
                blk.mlp.fc1.weight[u, 70+d2] = 20.0
                blk.mlp.fc1.bias[u] = -160.0
                blk.mlp.fc2.weight[80+r, u] = S
                blk.mlp.fc2.weight[100+r, u] = S
                if c:
                    blk.mlp.fc2.weight[90, u] = S
                    blk.mlp.fc2.weight[110, u] = S
                u += 1

def _carry_layer(model, li, D, H, dh):
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        if li <= 4:
            qpos = COL_TO_SNDPOS[li]
            kpos = COL_TO_SNDPOS[li-1]
            qtype, ktype = 22, 22
        else:
            qpos, kpos = 12, 6
            qtype, ktype = 23, 22

        # Position-specific targeting
        blk.attn.W_q.weight[0, 30+qpos] = W * 2
        blk.attn.W_k.weight[0, 30+kpos] = W * 2
        blk.attn.W_q.weight[1, qtype] = W * 2
        blk.attn.W_k.weight[1, ktype] = W * 2

        # V: carry (90), O: incoming (91)
        blk.attn.W_v.weight[0, 90] = 1.0
        blk.attn.W_o.weight[91, 0] = 1.0

        # MLP: add carry
        u = 0
        for d in range(10):
            nd = (d + 1) % 10
            nc = 1 if (d + 1) >= 10 else 0

            # With carry
            if u < model.d_ff:
                blk.mlp.fc1.weight[u, 91] = 20.0
                blk.mlp.fc1.weight[u, 100+d] = 20.0
                blk.mlp.fc1.bias[u] = -160.0
                blk.mlp.fc2.weight[100+d, u] = -S
                blk.mlp.fc2.weight[100+nd, u] = S
                if nc:
                    blk.mlp.fc2.weight[90, u] = S
                    blk.mlp.fc2.weight[110, u] = S
                else:
                    blk.mlp.fc2.weight[90, u] = -S
                    blk.mlp.fc2.weight[110, u] = -S
                blk.mlp.fc2.weight[91, u] = -S
                u += 1

def _output_layer(model, li, D, H, dh):
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Head 0: column + type
        for c in range(6):
            blk.attn.W_q.weight[c, 50+c] = W
            blk.attn.W_k.weight[c, 50+c] = W
        blk.attn.W_q.weight[6, 23] = W
        blk.attn.W_k.weight[6, 22] = W

        # V: 100-109 and 110, O: 0-9 and 10
        for d in range(10):
            blk.attn.W_v.weight[d, 100+d] = 1.0
            blk.attn.W_o.weight[d, d] = 1.0
        blk.attn.W_v.weight[10, 110] = 1.0
        blk.attn.W_o.weight[10, 10] = 1.0

        # MLP: handle col5 carry
        # If carry (dim 10 ~ S) and col5 (dim 55 ~ S) and output (dim 23 ~ S):
        blk.mlp.fc1.weight[0, 10] = 20.0
        blk.mlp.fc1.weight[0, 55] = 20.0
        blk.mlp.fc1.weight[0, 23] = 20.0
        blk.mlp.fc1.bias[0] = -240.0
        for d in range(10):
            blk.mlp.fc2.weight[d, 0] = -S
        blk.mlp.fc2.weight[1, 0] = S

def _identity(model, li):
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

model_shorthand_name = "carry_chain_v4"
model_description = "Stronger signals (S=10, P=20, W=20), cleaner logic. D=128,L=8,H=4,FF=512."

def upsert_overall_results(rows, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("model_shorthand_name"), row.get("task")) not in new_keys:
                    existing.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OVERALL_CSV_COLS)
        writer.writeheader()
        writer.writerows(existing + [{k: r.get(k, "") for k in OVERALL_CSV_COLS} for r in rows])

def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    task = src.task.get_task(args.task)
    print(f"DEBUG: vocab_size={task.vocab_size}, seq_len={task.seq_len}")
    model = build_model(task).to(args.device)
    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed, device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{"task": args.task, "accuracy": f"{accuracy:.4f}", "status": "", "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}", "description": model_description}], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print(); print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")