"""
Interpretable transformer for 5-digit addition. No LayerNorm.
Carry chain: col0->col1->col2->col3->col4->col5
Second-number positions: pos10(col0), pos9(col1), pos8(col2), pos7(col3), pos6(col4)
Each carry layer propagates carry ONE step using column-specific attention.
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

# Dim layout (D=128):
# 0-9: digit one-hot (S=5)
# 10: is-digit
# 11: is-first-num, 12: is-second-num, 13: is-output
# 20-37: position one-hot (P=10)
# 40-45: column one-hot (S)
# 50-59: d1 (first digit) one-hot [written by layer 0 attn]
# 60-69: d2 (second digit) one-hot [copy of digit, from token_emb]
# 70-79: sum%10 one-hot [written by layer 0 MLP]
# 80: carry flag [written by layer 0 MLP]
# 81: incoming carry [written by carry layer attn]
# 90-99: final result one-hot [initialized by layer 0 MLP, updated by carry layers]
# 100: final carry flag

S = 5.0; P = 10.0; W = 10.0

def write_weights(model, task):
    D = model.d_model; V = model.vocab_size; H = model.n_heads; dh = D // H

    with torch.no_grad():
        # TOKEN EMB
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, d] = S
            model.token_emb.weight[d, 10] = S
            model.token_emb.weight[d, 60 + d] = S  # d2 copy

        # POS EMB
        model.pos_emb.weight.zero_()
        for p in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[p, 20 + p] = P

        pos_to_col = {0:4,1:3,2:2,3:1,4:0, 6:4,7:3,8:2,9:1,10:0, 12:5,13:4,14:3,15:2,16:1,17:0}
        for pos, col in pos_to_col.items():
            if pos < model.max_seq_len:
                model.pos_emb.weight[pos, 40 + col] = S

        for p in range(5): model.pos_emb.weight[p, 11] = S
        for p in range(6,11): model.pos_emb.weight[p, 12] = S
        for p in range(12,18): model.pos_emb.weight[p, 13] = S

        # LAYER 0: gather d1 at second-number positions
        _layer0(model, D, H, dh)
        # LAYERS 1-5: carry propagation (one column pair per layer)
        for li in range(1, 6):
            _carry_layer(model, li, D, H, dh)
        # LAYER 6: output positions read results
        _output_layer(model, 6, D, H, dh)
        # LAYER 7: identity
        _identity(model, 7)

        # HEAD
        model.head.weight.zero_(); model.head.bias.zero_()
        for d in range(min(10, V)):
            model.head.weight[d, d] = 10.0
            model.head.bias[d] = -5.0
        if V > 10: model.head.bias[10] = -100.0
        if V > 11: model.head.bias[11] = -100.0

def _layer0(model, D, H, dh):
    """Second-number pos attends to first-number pos with same column.
    Gathers d1 into dims 50-59. MLP computes sum."""
    blk = model.blocks[0]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()

        # Head 0: col match + type match
        for c in range(6):
            blk.attn.W_q.weight[c, 40+c] = W  # Q reads column
            blk.attn.W_k.weight[c, 40+c] = W  # K reads column
        blk.attn.W_q.weight[6, 12] = W  # Q: I am second number
        blk.attn.W_k.weight[6, 11] = W  # K: I am first number

        # V: read digit (0-9), O: write to 50-59
        for d in range(10):
            blk.attn.W_v.weight[d, d] = 1.0
            blk.attn.W_o.weight[50+d, d] = 1.0

        # MLP: sum lookup
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        u = 0
        for d1 in range(10):
            for d2 in range(10):
                if u >= model.d_ff: break
                s = d1 + d2; r = s % 10; c = 1 if s >= 10 else 0
                blk.mlp.fc1.weight[u, 50+d1] = 10.0
                blk.mlp.fc1.weight[u, 60+d2] = 10.0
                blk.mlp.fc1.bias[u] = -80.0
                blk.mlp.fc2.weight[70+r, u] = S   # sum one-hot
                blk.mlp.fc2.weight[r, u] = S       # also to dims 0-9
                blk.mlp.fc2.weight[90+r, u] = S    # final result
                if c:
                    blk.mlp.fc2.weight[80, u] = S  # carry
                    blk.mlp.fc2.weight[100, u] = S # final carry
                u += 1

def _carry_layer(model, li, D, H, dh):
    """Carry layer li: col (li) queries col (li-1) for carry.
    li=1: col1(pos9)->col0(pos10), li=2: col2(pos8)->col1(pos9), etc."""
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        qcol = li      # querying column (40+li)
        kcol = li - 1   # key column (40+li-1)

        # Head 0: col li queries col li-1
        blk.attn.W_q.weight[0, 40+qcol] = W * 3  # strong Q
        blk.attn.W_k.weight[0, 40+kcol] = W * 3  # strong K
        # Also require is-second-number to avoid first-number interference
        blk.attn.W_q.weight[1, 12] = W * 3
        blk.attn.W_k.weight[1, 12] = W * 3

        # V reads: carry (dim 80)
        blk.attn.W_v.weight[0, 80] = 1.0
        # O writes: incoming carry (dim 81)
        blk.attn.W_o.weight[81, 0] = 1.0

        # MLP: add incoming carry to result digit
        u = 0
        for d in range(10):
            if u >= model.d_ff: break
            nd = (d+1) % 10; nc = 1 if d+1 >= 10 else 0
            # Fire when incoming carry (dim 81 ~ S) AND result digit d (dim 90+d ~ S)
            blk.mlp.fc1.weight[u, 81] = 10.0
            blk.mlp.fc1.weight[u, 90+d] = 10.0
            blk.mlp.fc1.bias[u] = -80.0
            # Update result digit
            blk.mlp.fc2.weight[90+d, u] = -S    # remove old
            blk.mlp.fc2.weight[90+nd, u] = S    # add new
            blk.mlp.fc2.weight[d, u] = -S        # remove from dims 0-9
            blk.mlp.fc2.weight[nd, u] = S        # add to dims 0-9
            # Update carry
            if nc:
                blk.mlp.fc2.weight[80, u] = S
                blk.mlp.fc2.weight[100, u] = S
            else:
                blk.mlp.fc2.weight[80, u] = -S
                blk.mlp.fc2.weight[100, u] = -S
            # Clear incoming carry
            blk.mlp.fc2.weight[81, u] = -S
            u += 1

def _output_layer(model, li, D, H, dh):
    """Output positions attend to second-number positions by column."""
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()

        # Head 0: output pos queries second-number pos with same column
        for c in range(6):
            blk.attn.W_q.weight[c, 40+c] = W
            blk.attn.W_k.weight[c, 40+c] = W
        blk.attn.W_q.weight[6, 13] = W  # Q: I am output
        blk.attn.W_k.weight[6, 12] = W  # K: I am second number

        # V reads result digit (90-99) and carry (100)
        for d in range(10):
            blk.attn.W_v.weight[d, 90+d] = 1.0
        blk.attn.W_v.weight[10, 100] = 1.0
        # O writes to dims 0-9 and dim 10
        for d in range(10):
            blk.attn.W_o.weight[d, d] = 1.0
        blk.attn.W_o.weight[10, 10] = 1.0

        # MLP: for col5 (pos 12), output carry as digit
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # If carry (dim 10 ~ S) AND col5 (dim 45 ~ S) AND is-output (dim 13 ~ S):
        # Set digit to '1', clear others
        blk.mlp.fc1.weight[0, 10] = 10.0
        blk.mlp.fc1.weight[0, 45] = 10.0
        blk.mlp.fc1.weight[0, 13] = 10.0
        blk.mlp.fc1.bias[0] = -120.0  # need all three ~5: 150 > 120
        for d in range(10):
            blk.mlp.fc2.weight[d, 0] = -S
        blk.mlp.fc2.weight[1, 0] = S  # set digit '1'

def _identity(model, li):
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

model_shorthand_name = "carry_chain_v2"
model_description = "Fixed carry chain: per-layer column-specific attention. D=128,L=8,H=4,FF=512. No LayerNorm."

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