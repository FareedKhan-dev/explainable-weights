"""
Interpretable transformer for 5-digit addition. No LayerNorm.
Dimension allocation (D=128):
  0-9:   HEAD OUTPUT digit one-hot (zero in token_emb, written by output layer only)
  10-19: token digit one-hot (S=5)
  20:    is-digit flag
  21:    is-first-num, 22: is-second-num, 23: is-output
  30-47: position one-hot (P=10)
  50-55: column one-hot (S=5)
  60-69: d1 one-hot [layer 0 attn]
  70-79: d2 one-hot [token_emb copy]
  80-89: sum%10 one-hot [layer 0 MLP]
  90:    carry flag
  91:    incoming carry
  100-109: final result one-hot [layer 0 MLP, updated by carry layers]
  110:   final carry flag

Carry chain through second-number positions (pos10..pos6, cols 0..4).
Each carry layer propagates one step using position-specific attention.
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

S = 5.0; P = 10.0; W = 10.0

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

# Column mapping
POS_TO_COL = {0:4,1:3,2:2,3:1,4:0, 6:4,7:3,8:2,9:1,10:0, 12:5,13:4,14:3,15:2,16:1,17:0}
# Second-number pos for each column
COL_TO_SNDPOS = {0:10, 1:9, 2:8, 3:7, 4:6}
# First-number pos for each column
COL_TO_FSTPOS = {0:4, 1:3, 2:2, 3:1, 4:0}
# Output pos for each column
COL_TO_OUTPOS = {0:17, 1:16, 2:15, 3:14, 4:13, 5:12}

def write_weights(model, task):
    D = model.d_model; V = model.vocab_size; H = model.n_heads; dh = D // H  # 32
    with torch.no_grad():
        # === TOKEN EMBEDDINGS ===
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, 10+d] = S    # digit one-hot in 10-19
            model.token_emb.weight[d, 20] = S      # is-digit
            model.token_emb.weight[d, 70+d] = S    # d2 copy in 70-79

        # === POSITION EMBEDDINGS ===
        model.pos_emb.weight.zero_()
        for p in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[p, 30+p] = P      # position one-hot
        for pos, col in POS_TO_COL.items():
            if pos < model.max_seq_len:
                model.pos_emb.weight[pos, 50+col] = S  # column
        for p in range(5):
            model.pos_emb.weight[p, 21] = S    # first-num
        for p in range(6, 11):
            model.pos_emb.weight[p, 22] = S    # second-num
        for p in range(12, 18):
            model.pos_emb.weight[p, 23] = S    # output

        # === LAYER 0: Gather d1 at second-number positions ===
        _layer0(model, D, H, dh)
        # === LAYERS 1-5: Carry propagation ===
        for li in range(1, 6):
            _carry_layer(model, li, D, H, dh)
        # === LAYER 6: Output reads ===
        _output_layer(model, 6, D, H, dh)
        # === LAYER 7: Identity ===
        _identity(model, 7)

        # === HEAD ===
        model.head.weight.zero_(); model.head.bias.zero_()
        for d in range(min(10, V)):
            model.head.weight[d, d] = 10.0
            model.head.bias[d] = -5.0
        if V > 10: model.head.bias[10] = -100.0
        if V > 11: model.head.bias[11] = -100.0

def _layer0(model, D, H, dh):
    """Second-number positions attend to first-number positions (same column).
    Gather d1 into dims 60-69. MLP computes sum."""
    blk = model.blocks[0]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Head 0: column match + type match
        for c in range(6):
            blk.attn.W_q.weight[c, 50+c] = W   # Q reads column
            blk.attn.W_k.weight[c, 50+c] = W   # K advertises column
        blk.attn.W_q.weight[6, 22] = W  # Q: I am second-num
        blk.attn.W_k.weight[6, 21] = W  # K: I am first-num

        # V reads digit one-hot (dims 10-19), O writes to 60-69
        for d in range(10):
            blk.attn.W_v.weight[60+d, 10+d] = 1.0
            blk.attn.W_o.weight[60+d, d] = 1.0

        # MLP: d1(60-69) + d2(70-79) -> sum(80-89), result(100-109), carry(90), final_carry(110)
        u = 0
        for d1 in range(10):
            for d2 in range(10):
                if u >= model.d_ff: break
                s = d1 + d2; r = s % 10; c = 1 if s >= 10 else 0
                blk.mlp.fc1.weight[u, 60+d1] = 10.0
                blk.mlp.fc1.weight[u, 70+d2] = 10.0
                blk.mlp.fc1.bias[u] = -80.0
                blk.mlp.fc2.weight[80+r, u] = S
                blk.mlp.fc2.weight[100+r, u] = S
                if c:
                    blk.mlp.fc2.weight[90, u] = S
                    blk.mlp.fc2.weight[110, u] = S
                u += 1

def _carry_layer(model, li, D, H, dh):
    """Carry layer li (1-5):
    li=1: pos9(col1) reads carry from pos10(col0)
    li=2: pos8(col2) reads carry from pos9(col1)
    li=3: pos7(col3) reads carry from pos8(col2)
    li=4: pos6(col4) reads carry from pos7(col3)
    li=5: pos12(col5) reads carry from pos6(col4)
    Uses position-specific attention for precision."""
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Determine query and key positions
        if li <= 4:
            qpos = COL_TO_SNDPOS[li]      # second-num pos at col li
            kpos = COL_TO_SNDPOS[li-1]    # second-num pos at col li-1
            q_type_dim = 22  # is-second-num
            k_type_dim = 22  # is-second-num
        else:  # li == 5
            qpos = 12  # output pos for col5
            kpos = 6   # second-num pos for col4
            q_type_dim = 23  # is-output
            k_type_dim = 22  # is-second-num

        # Head 0: position-specific targeting
        # Dim 0: Q reads qpos's position, K reads kpos's position
        blk.attn.W_q.weight[0, 30+qpos] = W  # Q at qpos is strong
        blk.attn.W_k.weight[0, 30+kpos] = W  # K at kpos is strong
        # Dim 1: type match
        blk.attn.W_q.weight[1, q_type_dim] = W
        blk.attn.W_k.weight[1, k_type_dim] = W

        # V reads carry (dim 90)
        blk.attn.W_v.weight[0, 90] = 1.0
        # O writes to incoming carry (dim 91)
        blk.attn.W_o.weight[91, 0] = 1.0

        # MLP: add incoming carry to result digit
        # Need to handle 4 cases per digit (old carry present/absent x new carry present/absent)
        u = 0
        for d in range(10):
            nd = (d + 1) % 10
            nc = 1 if (d + 1) >= 10 else 0

            # Case A: incoming carry + digit d + old carry present
            if u < model.d_ff:
                blk.mlp.fc1.weight[u, 91] = 10.0     # incoming carry
                blk.mlp.fc1.weight[u, 100+d] = 10.0  # result digit d
                blk.mlp.fc1.weight[u, 90] = 10.0     # old carry present
                blk.mlp.fc1.bias[u] = -120.0          # need all three ~5: 150>120
                # Update digit
                blk.mlp.fc2.weight[100+d, u] = -S    # remove old
                blk.mlp.fc2.weight[100+nd, u] = S    # add new
                # Update carry
                if nc:
                    pass  # carry stays S, no change needed
                else:
                    blk.mlp.fc2.weight[90, u] = -S   # clear carry
                    blk.mlp.fc2.weight[110, u] = -S  # clear final carry
                blk.mlp.fc2.weight[91, u] = -S       # clear incoming
                u += 1

            # Case B: incoming carry + digit d + old carry absent
            if u < model.d_ff:
                blk.mlp.fc1.weight[u, 91] = 10.0
                blk.mlp.fc1.weight[u, 100+d] = 10.0
                blk.mlp.fc1.weight[u, 90] = -10.0    # old carry ABSENT
                blk.mlp.fc1.bias[u] = -70.0           # 50+50+0-70=30>0; 50+50-50-70=-20<0
                blk.mlp.fc2.weight[100+d, u] = -S
                blk.mlp.fc2.weight[100+nd, u] = S
                if nc:
                    blk.mlp.fc2.weight[90, u] = S    # set carry
                    blk.mlp.fc2.weight[110, u] = S   # set final carry
                # else: carry stays 0, no change
                blk.mlp.fc2.weight[91, u] = -S
                u += 1

def _output_layer(model, li, D, H, dh):
    """Output positions attend to second-number positions by column.
    pos12(col5) attends to pos6(col4) for carry."""
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

        # Head 0: column-based matching + type
        for c in range(6):
            blk.attn.W_q.weight[c, 50+c] = W
            blk.attn.W_k.weight[c, 50+c] = W
        blk.attn.W_q.weight[6, 23] = W  # Q: is-output
        blk.attn.W_k.weight[6, 22] = W  # K: is-second-num
        # Extra dim for col5->col4 targeting
        blk.attn.W_q.weight[7, 55] = W  # Q: col5
        blk.attn.W_k.weight[7, 54] = W  # K: col4

        # V reads result (100-109) and final carry (110)
        for d in range(10):
            blk.attn.W_v.weight[d, 100+d] = 1.0
        blk.attn.W_v.weight[10, 110] = 1.0
        # O writes to dims 0-9 and dim 10
        for d in range(10):
            blk.attn.W_o.weight[d, d] = 1.0
        blk.attn.W_o.weight[10, 10] = 1.0

        # MLP: for col5 (pos12), output carry flag as digit
        # When carry (dim 10 ~ S) and col5 (dim 55 ~ S) and is-output (dim 23 ~ S):
        blk.mlp.fc1.weight[0, 10] = 10.0
        blk.mlp.fc1.weight[0, 55] = 10.0
        blk.mlp.fc1.weight[0, 23] = 10.0
        blk.mlp.fc1.bias[0] = -120.0
        for d in range(10):
            blk.mlp.fc2.weight[d, 0] = -S   # clear all digits
        blk.mlp.fc2.weight[1, 0] = S        # set digit '1'

        # Also handle no-carry case for col5: output '0'
        # When no-carry (dim 10 ~ 0) and col5 (dim 55 ~ S) and is-output (dim 23 ~ S):
        blk.mlp.fc1.weight[1, 55] = 10.0
        blk.mlp.fc1.weight[1, 23] = 10.0
        blk.mlp.fc1.weight[1, 10] = -10.0   # no carry
        blk.mlp.fc1.bias[1] = -70.0
        for d in range(10):
            blk.mlp.fc2.weight[d, 1] = -S
        blk.mlp.fc2.weight[0, 1] = S        # set digit '0'

def _identity(model, li):
    blk = model.blocks[li]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()

model_shorthand_name = "carry_chain_v3"
model_description = "Clean dim separation, position-specific carry, no LayerNorm. D=128,L=8,H=4,FF=512."

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