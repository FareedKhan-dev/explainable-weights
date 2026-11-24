"""
Interpretable transformer for 5-digit addition.
Task: "12345+67890=" -> "080235"
Positions: 0-4 (A), 5 (+), 6-10 (B), 11 (=), 12-17 (answer, 6 digits)

Answer: ans[i] for i=0..5 where i=0 is leading carry, i=5 is ones digit.
  ans[i] = (A[i-1] + B[i-1] + carry(A[i]+B[i])) % 10
  (with A[-1]=B[-1]=0 for i=0, and carry(A[5]+B[5])=0 for i=5)

Strategy: 4 attention heads copy A[i-1], B[i-1], A[i], B[i] to answer pos 12+i.
MLP lookup table computes the result.
"""

from __future__ import annotations
import argparse, csv, math, os, sys, time
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4, n_layers=3, d_ff=256):
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

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


def write_weights(model, task):
    D = model.d_model   # 256
    H = model.n_heads   # 8
    dh = D // H          # 32

    for p in model.parameters():
        p.data.zero_()

    # D=256 dimension layout:
    # 0-11:   token identity (digits 0-9, +, =)
    # 20-37:  position identity
    # 40-45:  answer query for A[i-1] (answer pos 12+i, i=0..5)
    # 50-55:  answer query for B[i-1]
    # 60-65:  answer query for A[i]
    # 70-75:  answer query for B[i]
    # 80-84:  A-position key (pos 0-4)
    # 90-94:  B-position key (pos 6-10)
    # 100:    answer position flag
    # 140-149: head 0 output (A[i-1] digit)
    # 150-159: head 1 output (B[i-1] digit)
    # 160-169: head 2 output (A[i] digit)
    # 170-179: head 3 output (B[i] digit)
    # 180-189: MLP result digit output

    # Token embeddings: one-hot
    for i in range(10):
        model.token_emb.weight.data[i, i] = 1.0
    model.token_emb.weight.data[10, 10] = 1.0  # +
    model.token_emb.weight.data[11, 11] = 1.0  # =

    # Position embeddings
    for p in range(min(18, model.max_seq_len)):
        model.pos_emb.weight.data[p, 20 + p] = 1.0

    # Answer position queries
    for i in range(6):
        ap = 12 + i
        if ap >= model.max_seq_len:
            break
        model.pos_emb.weight.data[ap, 40 + i] = 1.0
        model.pos_emb.weight.data[ap, 50 + i] = 1.0
        model.pos_emb.weight.data[ap, 60 + i] = 1.0
        model.pos_emb.weight.data[ap, 70 + i] = 1.0
        model.pos_emb.weight.data[ap, 100] = 1.0

    # A-position keys
    for j in range(5):
        model.pos_emb.weight.data[j, 80 + j] = 1.0
    # B-position keys
    for j in range(5):
        model.pos_emb.weight.data[6 + j, 90 + j] = 1.0

    # LayerNorm identity
    for block in model.blocks:
        block.ln1.weight.data.fill_(1.0)
        block.ln2.weight.data.fill_(1.0)
    model.final_ln.weight.data.fill_(1.0)

    S = 20.0  # attention scale

    blk = model.blocks[0]

    # --- HEAD 0: answer pos 12+i -> A[i-1] at pos (i-1) ---
    # Query: pos 12+i has pos_emb[40+i]=1. Map to q-dim (i-1).
    # For i=1..5: q-dim i-1, for i=0: q-dim 5 (dummy -> position 5 '+')
    for i in range(1, 6):
        blk.attn.W_q.weight.data[0*dh + (i-1), 40 + i] = S
    blk.attn.W_q.weight.data[0*dh + 5, 40 + 0] = S  # i=0 dummy

    # Key: A pos j has pos_emb[80+j]=1. Map to k-dim j.
    for j in range(5):
        blk.attn.W_k.weight.data[0*dh + j, 80 + j] = S
    # Dummy key: position 5 ('+') has token_emb[10]=1. Map to k-dim 5.
    blk.attn.W_k.weight.data[0*dh + 5, 10] = S

    # Value: copy digit dims 0-9
    for d in range(10):
        blk.attn.W_v.weight.data[0*dh + d, d] = 1.0
    # Output: write to residual dims 140-149
    for d in range(10):
        blk.attn.W_o.weight.data[140 + d, 0*dh + d] = 1.0

    # --- HEAD 1: answer pos 12+i -> B[i-1] at pos 6+(i-1)=5+i ---
    for i in range(1, 6):
        blk.attn.W_q.weight.data[1*dh + (i-1), 50 + i] = S
    blk.attn.W_q.weight.data[1*dh + 5, 50 + 0] = S  # i=0 dummy

    for j in range(5):
        blk.attn.W_k.weight.data[1*dh + j, 90 + j] = S
    blk.attn.W_k.weight.data[1*dh + 5, 10] = S  # dummy -> position 5 '+'

    for d in range(10):
        blk.attn.W_v.weight.data[1*dh + d, d] = 1.0
    for d in range(10):
        blk.attn.W_o.weight.data[150 + d, 1*dh + d] = 1.0

    # --- HEAD 2: answer pos 12+i -> A[i] at pos i ---
    for i in range(5):
        blk.attn.W_q.weight.data[2*dh + i, 60 + i] = S
    blk.attn.W_q.weight.data[2*dh + 5, 60 + 5] = S  # i=5 dummy

    for j in range(5):
        blk.attn.W_k.weight.data[2*dh + j, 80 + j] = S
    blk.attn.W_k.weight.data[2*dh + 5, 11] = S  # dummy -> position 11 '='

    for d in range(10):
        blk.attn.W_v.weight.data[2*dh + d, d] = 1.0
    for d in range(10):
        blk.attn.W_o.weight.data[160 + d, 2*dh + d] = 1.0

    # --- HEAD 3: answer pos 12+i -> B[i] at pos 6+i ---
    for i in range(5):
        blk.attn.W_q.weight.data[3*dh + i, 70 + i] = S
    blk.attn.W_q.weight.data[3*dh + 5, 70 + 5] = S  # i=5 dummy

    for j in range(5):
        blk.attn.W_k.weight.data[3*dh + j, 90 + j] = S
    blk.attn.W_k.weight.data[3*dh + 5, 11] = S  # dummy -> position 11 '='

    for d in range(10):
        blk.attn.W_v.weight.data[3*dh + d, d] = 1.0
    for d in range(10):
        blk.attn.W_o.weight.data[170 + d, 3*dh + d] = 1.0

    # --- MLP: Lookup table for (a + b + carry(c + d)) % 10 ---
    # Reads: dims 140-149 (a=A[i-1]), 150-159 (b=B[i-1]), 160-169 (c=A[i]), 170-179 (d=B[i])
    # Also reads dim 100 (answer position flag)
    # Writes: dims 180-189 (result digit)
    # After LayerNorm2, active dims ~4.5, inactive ~-0.22
    # Threshold: 5 active features * 4.5 = 22.5, bias = -20

    n_idx = 0
    for a in range(10):
        for b in range(10):
            for c in range(10):
                for d_val in range(10):
                    if n_idx >= model.d_ff:
                        break
                    blk.mlp.fc1.weight.data[n_idx, 140 + a] = 1.0
                    blk.mlp.fc1.weight.data[n_idx, 150 + b] = 1.0
                    blk.mlp.fc1.weight.data[n_idx, 160 + c] = 1.0
                    blk.mlp.fc1.weight.data[n_idx, 170 + d_val] = 1.0
                    blk.mlp.fc1.weight.data[n_idx, 100] = 1.0
                    blk.mlp.fc1.bias.data[n_idx] = -20.0

                    carry = 1 if (c + d_val >= 10) else 0
                    result = (a + b + carry) % 10
                    blk.mlp.fc2.weight.data[180 + result, n_idx] += 1.0

                    n_idx += 1
                if n_idx >= model.d_ff:
                    break
            if n_idx >= model.d_ff:
                break
        if n_idx >= model.d_ff:
            break

    # --- Layers 2 & 3: Identity ---
    for bi in [1, 2]:
        b = model.blocks[bi]
        for j in range(min(dh, D)):
            b.attn.W_q.weight.data[j, j] = 1.0
            b.attn.W_k.weight.data[j, j] = 1.0
            b.attn.W_v.weight.data[j, j] = 1.0
            b.attn.W_o.weight.data[j, j] = 1.0

    # --- Output head ---
    for d in range(10):
        model.head.weight.data[d, 180 + d] = 10.0
    for d in range(10, model.vocab_size):
        model.head.weight.data[d, :] = -100.0


model_shorthand_name = "CarryLookup_v14"
model_description = "4 heads copy A[i-1],B[i-1],A[i],B[i] to answer pos. MLP lookup with carry. Dummy keys for edge cases."


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
    max_seq_len = max(task.seq_len, 18)
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=256,
        n_heads=8,
        n_layers=3,
        d_ff=10000,
    )
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
    model = build_model(task).to(args.device)

    accuracy, details = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    if hasattr(details, '__iter__') and len(details) > 0:
        print(f"First example: {details[0] if isinstance(details, list) else 'N/A'}")

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task": args.task,
        "accuracy": f"{accuracy:.4f}",
        "status": "",
        "model_shorthand_name": model_shorthand_name,
        "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
