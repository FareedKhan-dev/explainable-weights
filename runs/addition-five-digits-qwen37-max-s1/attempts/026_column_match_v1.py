"""
Simple column-matching addition. Output positions attend to input positions in same column.
D=128, L=8, H=1, FF=512
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

D_MODEL = 128; N_HEADS = 1; N_LAYERS = 8; D_FF = 512
W = 100.0

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
    def __init__(self, vocab_size, max_seq_len=32, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF):
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

# Sequence format: pos 0-4 (first), pos 5 (+), pos 6-10 (second), pos 11 (=), pos 12-17 (output)
# Columns (right to left): col 0 at pos 4,10,17; col 1 at pos 3,9,16; ... col 4 at pos 0,6,13; col 5 (overflow) at pos 12

def write_weights(model, task):
    with torch.no_grad():
        # Token embedding: digit value in dims 0-9
        model.token_emb.weight.zero_()
        for d in range(10):
            model.token_emb.weight[d, d] = 10.0
        
        # Position embedding: column ID in dims 10-15, type flags in 16-18
        model.pos_emb.weight.zero_()
        col_map = {4:0, 3:1, 2:2, 1:3, 0:4,  # first number
                   10:0, 9:1, 8:2, 7:3, 6:4,  # second number  
                   17:0, 16:1, 15:2, 14:3, 13:4, 12:5}  # output
        type_map = {0:16, 1:16, 2:16, 3:16, 4:16,  # first
                    6:17, 7:17, 8:17, 9:17, 10:17,  # second
                    12:18, 13:18, 14:18, 15:18, 16:18, 17:18}  # output
        
        for pos in range(min(18, model.max_seq_len)):
            if pos in col_map:
                model.pos_emb.weight[pos, 10 + col_map[pos]] = 20.0
            if pos in type_map:
                model.pos_emb.weight[pos, type_map[pos]] = 20.0
        
        # Layer 0: Output positions attend to input positions in same column
        blk = model.blocks[0]
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        
        # Q: output positions query their column
        for col in range(6):
            blk.attn.W_q.weight[col, 10 + col] = W
        blk.attn.W_q.weight[6, 18] = W  # output type
        
        # K: input positions key their column and type
        for col in range(6):
            blk.attn.W_k.weight[col, 10 + col] = W
        blk.attn.W_k.weight[6, 16] = W  # first number type
        blk.attn.W_k.weight[6, 17] = W  # second number type
        
        # V: extract digit value
        for d in range(10):
            blk.attn.W_v.weight[d, d] = 1.0
        for d in range(10):
            blk.attn.W_o.weight[d, d] = 1.0
        
        # MLP: compute sum mod 10 (no carry yet)
        u = 0
        for d1 in range(10):
            for d2 in range(10):
                if u >= model.d_ff: break
                s = (d1 + d2) % 10
                blk.mlp.fc1.weight[u, d1] = 30.0
                blk.mlp.fc1.weight[u, d2] = 30.0
                blk.mlp.fc1.bias[u] = -250.0
                blk.mlp.fc2.weight[s, u] = 10.0
                u += 1
        
        # Identity for other layers
        for li in range(1, N_LAYERS):
            blk = model.blocks[li]
            blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
            blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
            blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
            blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        
        # Output head
        model.head.weight.zero_(); model.head.bias.zero_()
        for d in range(10):
            model.head.weight[d, d] = 50.0
        if model.vocab_size > 10:
            model.head.bias[10] = -500.0
        if model.vocab_size > 11:
            model.head.bias[11] = -500.0

model_shorthand_name = "column_match_v1"
model_description = "Column matching: output attends to inputs in same column, computes sum mod 10. D=128,L=8,H=1,FF=512."

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
