"""
Interpretable transformer for 5-digit addition.
No LayerNorm. Carry chain through second-number positions.

Sequence: "ABCDE+FGHIJ=" -> "RRRRRR" (6 digits, MSB first)
  pos 0-4: A,B,C,D,E (first number, cols 4,3,2,1,0)
  pos 5: +
  pos 6-10: F,G,H,I,J (second number, cols 4,3,2,1,0)
  pos 11: =
  pos 12-17: output (cols 5,4,3,2,1,0)

Carry chain: col0->col1->col2->col3->col4->col5
In second-number positions: pos10->pos9->pos8->pos7->pos6
Causal mask allows pos9 to see pos10, pos8 to see pos9, etc.
So carry flows naturally through causal attention!

Architecture (8 layers, D=128, H=4, dh=32, FF=512):
  Layer 0: Second-number positions attend to matching first-number positions
           -> gather d1. MLP computes d1+d2, stores sum%10 and carry.
  Layers 1-5: Carry propagation. Each second-num pos attends to next
              second-num pos for incoming carry. MLP updates digit+carry.
  Layer 6: Output positions attend to corresponding second-num positions.
           MLP copies result digit to dims 0-9 for head.
  Layer 7: Identity (cleanup).
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


# ----- Dimension layout (D=128) -----
# Dims 0-9:   digit one-hot (value S=5.0)
# Dim 10:     is-digit flag (S)
# Dim 11:     is-first-number flag (S)
# Dim 12:     is-second-number flag (S)
# Dim 13:     is-output flag (S)
# Dim 14:     is-special (+ or =) flag (S)
# Dims 20-37: position one-hot (value P=10.0)
# Dims 40-45: column one-hot (value S)
# Dims 50-59: d1 (first digit) one-hot, written by layer 0 attention
# Dims 60-69: d2 (second digit) one-hot, from token_emb (same as 0-9 but stored separately)
# Dims 70-79: sum result one-hot (sum%10), written by layer 0 MLP
# Dim 80:     carry flag, written by MLP
# Dim 81:     incoming carry (from attention reading carry from previous pos)
# Dims 90-99: final result digit one-hot (after carry adjustments)
# Dim 100:    final carry flag

S = 5.0   # signal strength
P = 10.0  # position signal strength
W = 10.0  # attention weight strength


def write_weights(model, task):
    D = model.d_model
    V = model.vocab_size
    H = model.n_heads
    dh = D // H  # 32

    with torch.no_grad():
        # ========== TOKEN EMBEDDINGS ==========
        model.token_emb.weight.zero_()
        for d in range(min(10, V)):
            model.token_emb.weight[d, d] = S       # digit one-hot dims 0-9
            model.token_emb.weight[d, 10] = S      # is-digit
            model.token_emb.weight[d, 60 + d] = S  # copy to dims 60-69 (d2 one-hot)
        # token 10 (+): nothing special
        # token 11 (=): nothing special

        # ========== POSITION EMBEDDINGS ==========
        model.pos_emb.weight.zero_()
        # Position one-hot in dims 20-37
        for pos in range(min(18, model.max_seq_len)):
            model.pos_emb.weight[pos, 20 + pos] = P

        # Column one-hot in dims 40-45
        pos_to_col = {0:4,1:3,2:2,3:1,4:0, 6:4,7:3,8:2,9:1,10:0, 12:5,13:4,14:3,15:2,16:1,17:0}
        for pos, col in pos_to_col.items():
            if pos < model.max_seq_len:
                model.pos_emb.weight[pos, 40 + col] = S

        # Number-type flags
        for p in range(5):       # first number
            model.pos_emb.weight[p, 11] = S
        for p in range(6, 11):   # second number
            model.pos_emb.weight[p, 12] = S
        for p in range(12, 18):  # output
            model.pos_emb.weight[p, 13] = S
        model.pos_emb.weight[5, 14] = S   # +
        model.pos_emb.weight[11, 14] = S  # =

        # ========== LAYER 0: Gather d1 at second-number positions ==========
        _layer0_gather(model, D, H, dh)

        # ========== LAYERS 1-5: Carry propagation ==========
        for li in range(1, 6):
            _layer_carry(model, li, D, H, dh)

        # ========== LAYER 6: Output positions read from second-number positions ==========
        _layer_output_read(model, D, H, dh)

        # ========== LAYER 7: Identity ==========
        _layer_identity(model, 7, D, H, dh)

        # ========== HEAD ==========
        model.head.weight.zero_()
        model.head.bias.zero_()
        for d in range(min(10, V)):
            model.head.weight[d, d] = 10.0       # read dims 0-9
            model.head.bias[d] = -5.0
        if V > 10:
            model.head.bias[10] = -100.0  # suppress +
        if V > 11:
            model.head.bias[11] = -100.0  # suppress =


def _layer0_gather(model, D, H, dh):
    """
    Layer 0: Second-number positions (6-10) attend to matching first-number positions.
    Head 0: column+type matching, copies d1 one-hot to dims 50-59.
    Heads 1-3: near-zero (identity-ish).
    MLP: compute d1+d2 -> sum%10 in dims 70-79, carry in dim 80.
    Also copies sum%10 to dims 0-9 and carry to dims 90-99/dim 100 for later use.
    """
    blk = model.blocks[0]

    with torch.no_grad():
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_()
        blk.attn.W_o.weight.zero_()

        # Head 0: second-number -> first-number, same column
        # Q: read column (dims 40-45) and is-second-number (dim 12)
        # K: read column (dims 40-45) and is-first-number (dim 11)
        for col in range(6):
            blk.attn.W_q.weight[col, 40 + col] = W     # Q queries column
            blk.attn.W_k.weight[col, 40 + col] = W     # K advertises column
        blk.attn.W_q.weight[6, 12] = W    # Q: I'm second number
        blk.attn.W_k.weight[6, 11] = W    # K: I'm first number

        # V: read digit one-hot (dims 0-9) -> output dims 0-9 of head
        for d in range(10):
            blk.attn.W_v.weight[d, d] = 1.0
        # O: project to dims 50-59 (d1 storage)
        for d in range(10):
            blk.attn.W_o.weight[50 + d, d] = 1.0

        # Heads 1-3: zero weights (no interference)

        # --- MLP: compute sum of d1 (dims 50-59) + d2 (dims 60-69) ---
        blk.mlp.fc1.weight.zero_()
        blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()

        unit = 0
        for d1 in range(10):
            for d2 in range(10):
                if unit >= model.d_ff:
                    break
                s = d1 + d2
                result = s % 10
                carry = 1 if s >= 10 else 0

                # Fire when dim(50+d1) ~ S and dim(60+d2) ~ S
                blk.mlp.fc1.weight[unit, 50 + d1] = 10.0
                blk.mlp.fc1.weight[unit, 60 + d2] = 10.0
                blk.mlp.fc1.bias[unit] = -80.0  # need both ~50+50=100 > 80

                # Write result to dims 70-79 (sum one-hot)
                blk.mlp.fc2.weight[70 + result, unit] = S
                # Write carry to dim 80
                if carry:
                    blk.mlp.fc2.weight[80, unit] = S
                # Also write result to dims 0-9 (for final head reading later)
                blk.mlp.fc2.weight[result, unit] = S
                # Also write to dims 90-99 (final result) and dim 100 (final carry)
                blk.mlp.fc2.weight[90 + result, unit] = S
                if carry:
                    blk.mlp.fc2.weight[100, unit] = S

                unit += 1


def _layer_carry(model, layer_idx, D, H, dh):
    """
    Carry propagation layer. Second-number position at col c (pos 10-c for c=0..4)
    attends to position at col c-1 (pos 10-(c-1) = pos 11-c) for incoming carry.

    pos 9 (col1) attends to pos 10 (col0) for carry
    pos 8 (col2) attends to pos 9 (col1) for carry
    pos 7 (col3) attends to pos 8 (col2) for carry
    pos 6 (col4) attends to pos 7 (col3) for carry

    Each layer propagates carry one step. After 5 layers:
    - Layer 1: pos 9 gets carry from pos 10
    - Layer 2: pos 8 gets carry from pos 9 (which was updated in layer 1)
    - Layer 3: pos 7 gets carry from pos 8
    - Layer 4: pos 6 gets carry from pos 7
    - Layer 5: output pos 12 (col5) can get carry from pos 6

    Actually, in layer_idx=1: all carry pairs could fire, but only pos 9->10
    has correct carry (pos 10's carry was set in layer 0).
    In layer 2: pos 8->9 now works because pos 9 was updated in layer 1.
    Etc.

    Head 0: position p attends to position p+1
    V: reads carry (dim 80) and result (dims 70-79 or 90-99)
    O: writes to dim 81 (incoming carry) and dims 70-79/90-99 passthrough

    MLP: if incoming carry (dim 81), add 1 to result digit, update carry.
    """
    blk = model.blocks[layer_idx]

    with torch.no_grad():
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_()
        blk.attn.W_o.weight.zero_()

        # Head 0: each second-number pos p (6-9) attends to pos p+1 (7-10)
        # Also pos 12 (col5 output) attends to pos 6 (col4) for final carry
        # Use position one-hot for precise targeting

        # For pos 9->10, 8->9, 7->8, 6->7:
        for p in range(6, 10):
            target = p + 1
            # Q at pos p queries for target position
            blk.attn.W_q.weight[0, 20 + target] = W * 3
            # K at target pos advertises its position
            blk.attn.W_k.weight[0, 20 + target] = W * 3
        # pos 12 -> pos 6
        blk.attn.W_q.weight[1, 20 + 6] = W * 3
        blk.attn.W_k.weight[1, 20 + 6] = W * 3

        # V reads carry (dim 80) and result digit (dims 90-99) and carry (dim 100)
        blk.attn.W_v.weight[0, 80] = 1.0    # carry flag
        for d in range(10):
            blk.attn.W_v.weight[1 + d, 90 + d] = 1.0  # result digit
        blk.attn.W_v.weight[11, 100] = 1.0   # result carry

        # O: project carry to dim 81 (incoming carry), result to pass-through dims
        blk.attn.W_o.weight[81, 0] = 1.0     # incoming carry
        for d in range(10):
            blk.attn.W_o.weight[90 + d, 1 + d] = 1.0  # result passthrough (overwritten if carry)
        blk.attn.W_o.weight[100, 11] = 1.0   # carry passthrough

        # --- MLP: add incoming carry to result digit ---
        blk.mlp.fc1.weight.zero_()
        blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()

        u = 0
        # Case 1: incoming carry (dim 81 ~ S) and result digit d (dim 90+d ~ S)
        for d in range(10):
            if u >= model.d_ff:
                break
            new_d = (d + 1) % 10
            new_carry = 1 if (d + 1) >= 10 else 0

            blk.mlp.fc1.weight[u, 81] = 10.0      # incoming carry
            blk.mlp.fc1.weight[u, 90 + d] = 10.0  # current result digit
            blk.mlp.fc1.bias[u] = -80.0

            # Remove old result digit, add new
            blk.mlp.fc2.weight[90 + d, u] = -S      # remove old
            blk.mlp.fc2.weight[90 + new_d, u] = S    # add new
            blk.mlp.fc2.weight[0 + d, u] = -S        # remove from dims 0-9
            blk.mlp.fc2.weight[0 + new_d, u] = S     # add to dims 0-9

            # Update carry: set new carry, clear incoming carry
            if new_carry:
                blk.mlp.fc2.weight[80, u] = S      # set carry
                blk.mlp.fc2.weight[100, u] = S     # update final carry
            else:
                blk.mlp.fc2.weight[80, u] = -S     # clear carry
                blk.mlp.fc2.weight[100, u] = -S    # clear final carry
            blk.mlp.fc2.weight[81, u] = -S          # clear incoming carry

            u += 1

        # Case 2: no incoming carry, pass through (clear any spurious incoming)
        if u < model.d_ff:
            blk.mlp.fc1.weight[u, 81] = 10.0
            blk.mlp.fc1.bias[u] = -30.0  # fires when dim 81 ~ 5 -> 50 > 30, but also
            # Actually we want this to fire when there's NO incoming carry...
            # Use negative detection: fire when dim 81 is ~0
            # Can't do that directly with ReLU. Instead, just don't do anything.
            # The carry clearing is handled by Case 1 units.
            blk.mlp.fc1.weight[u, 81] = 0.0
            blk.mlp.fc1.bias[u] = 0.0


def _layer_output_read(model, D, H, dh):
    """
    Layer 6: Output positions (12-17) attend to second-number positions (6-10)
    to read the computed result digit.

    pos 17 (col0) -> pos 10 (col0)
    pos 16 (col1) -> pos 9 (col1)
    pos 15 (col2) -> pos 8 (col2)
    pos 14 (col3) -> pos 7 (col3)
    pos 13 (col4) -> pos 6 (col4)
    pos 12 (col5) -> pos 6 (col4, for carry)

    Head 0: column-based matching
    V: reads result digit (dims 90-99) and carry (dim 100)
    O: writes to dims 0-9 (for head) and special handling for col5

    MLP: For pos 12 (col5), output carry as the digit.
    """
    blk = model.blocks[6]

    with torch.no_grad():
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_()
        blk.attn.W_o.weight.zero_()

        # Head 0: output positions attend to second-number positions by column
        # Q: read column (dims 40-45) + is-output (dim 13)
        # K: read column (dims 40-45) + is-second-number (dim 12)
        for col in range(6):
            blk.attn.W_q.weight[col, 40 + col] = W
            blk.attn.W_k.weight[col, 40 + col] = W
        blk.attn.W_q.weight[6, 13] = W     # Q: I'm output
        blk.attn.W_k.weight[6, 12] = W     # K: I'm second number

        # V: read result digit (dims 90-99) and carry (dim 100)
        for d in range(10):
            blk.attn.W_v.weight[d, 90 + d] = 1.0
        blk.attn.W_v.weight[10, 100] = 1.0

        # O: project to dims 0-9 (digit) and dim 10 (carry)
        for d in range(10):
            blk.attn.W_o.weight[d, d] = 1.0
        blk.attn.W_o.weight[10, 10] = 1.0  # carry to dim 10

        # MLP: For col 5 (pos 12), the digit should be the carry (0 or 1)
        # For other positions, digits are already correct in dims 0-9
        blk.mlp.fc1.weight.zero_()
        blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()

        # If dim 10 has carry signal (~S) AND is-output (dim 13 ~S) AND col 5 (dim 45 ~S):
        # Output '1' instead of whatever digit
        if model.d_ff > 0:
            # Unit: fires when carry (dim 10) is high and col5 (dim 45) is high
            blk.mlp.fc1.weight[0, 10] = 10.0
            blk.mlp.fc1.weight[0, 45] = 10.0
            blk.mlp.fc1.bias[0] = -80.0

            # Clear all digit dims 0-9, set dim 1 (= digit '1')
            for d in range(10):
                blk.mlp.fc2.weight[d, 0] = -S
            blk.mlp.fc2.weight[1, 0] = S


def _layer_identity(model, layer_idx, D, H, dh):
    """Identity layer - zero out all weights so residual passes through."""
    blk = model.blocks[layer_idx]
    with torch.no_grad():
        blk.attn.W_q.weight.zero_()
        blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_()
        blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_()
        blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()


model_shorthand_name = "carry_chain_v1"
model_description = "Carry chain through second-number positions using causal attention. D=128,L=8,H=4,FF=512. No LayerNorm."


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
    print(f"Overall results saved → {path}")


def build_model(task):
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    print(f"DEBUG: vocab_size={task.vocab_size}, seq_len={task.seq_len}")
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}", "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")