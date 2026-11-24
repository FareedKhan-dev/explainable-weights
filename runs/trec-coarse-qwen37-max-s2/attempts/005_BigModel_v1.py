"""
Interpretable transformer for TREC coarse classification.
Strategy: Use d_model=128 to have enough space for clean character encoding
and word detection circuits.
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
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
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
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=128, n_heads=4, n_layers=3, d_ff=512):
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
    """
    First-word classifier with d_model=128.
    
    Dimension layout:
    - Dims 0-35: character one-hot (36 vocab chars)
    - Dims 36-41: position markers (pos 0-5)
    - Dim 42: '=' position marker (pos 80)
    - Dim 43: answer position marker (pos 81)
    - Dims 44-49: class score accumulators (classes 0-5)
    - Dims 50-63: reserved
    - Dims 64-99: char at pos 0 (copied by attention)
    - Dims 100-135: char at pos 1
    - Dims 136-171: char at pos 2 (but we only have 128 dims total)
    
    Actually, let's be simpler:
    - Dims 0-35: char identity at current position
    - Dims 36-41: position markers 0-5
    - Dim 42: pos 80 marker
    - Dim 43: pos 81 marker
    - Dims 44-79: char at pos 0 (copied from pos 0)
    - Dims 80-115: char at pos 1
    - Dims 116-127: char at pos 2 (12 dims)
    
    This is still messy. Let me use a cleaner approach:
    - Layer 1: detect character at each of first 4 positions
    - Layer 2: combine to detect words
    - Layer 3: map to classes
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model  # 128
    H = model.n_heads  # 4
    dh = d // H  # 32
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDING ----
    # One-hot for all 36 characters in dims 0-35
    for char, idx in stoi.items():
        model.token_emb.weight.data[idx, idx] = 3.0
    
    # ---- POSITION EMBEDDING ----
    # Dims 36-45: markers for positions 0-9
    for pos in range(10):
        model.pos_emb.weight.data[pos, 36 + pos] = 3.0
    # Dim 46: pos 80 marker
    if model.max_seq_len > 80:
        model.pos_emb.weight.data[80, 46] = 3.0
    # Dim 47: pos 81 marker
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 47] = 3.0
    
    # ---- LAYER 1: Copy chars from positions 0-3 ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    # Head 0: copy char from position 0 to dims 48-83
    blk0.attn.W_q.weight.data[0, 46] = 5.0  # Q: I'm at pos 80
    blk0.attn.W_k.weight.data[0, 36] = 5.0  # K: looking for pos 0
    for i in range(32):  # Copy first 32 char dims
        blk0.attn.W_v.weight.data[i, i] = 3.0
    for i in range(32):
        blk0.attn.W_o.weight.data[48 + i, i] = 3.0
    
    # Head 1: copy char from position 1 to dims 80-111
    blk0.attn.W_q.weight.data[dh, 46] = 5.0
    blk0.attn.W_k.weight.data[dh, 37] = 5.0
    for i in range(32):
        blk0.attn.W_v.weight.data[dh + i, i] = 3.0
    for i in range(32):
        blk0.attn.W_o.weight.data[80 + i, i] = 3.0
    
    # Head 2: copy char from position 2 to dims 112-127 (16 dims)
    blk0.attn.W_q.weight.data[2*dh, 46] = 5.0
    blk0.attn.W_k.weight.data[2*dh, 38] = 5.0
    for i in range(16):
        blk0.attn.W_v.weight.data[2*dh + i, i] = 3.0
    for i in range(16):
        blk0.attn.W_o.weight.data[112 + i, i] = 3.0
    
    # Head 3: copy char from position 3 (use remaining dims 0-15, will overlap)
    blk0.attn.W_q.weight.data[3*dh, 46] = 5.0
    blk0.attn.W_k.weight.data[3*dh, 39] = 5.0
    for i in range(16):
        blk0.attn.W_v.weight.data[3*dh + i, i] = 3.0
    for i in range(16):
        blk0.attn.W_o.weight.data[i, i] = 3.0
    
    # ---- LAYER 1 MLP: detect first words ----
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    # After attention at pos 80:
    # Dims 48-79: char at pos 0 (first 32 chars)
    # Dims 80-111: char at pos 1
    # Dims 112-127: char at pos 2 (first 16 chars)
    # Dims 0-15: char at pos 3 (first 16 chars)
    
    # Detect "who": w(22) at pos0, h(7) at pos1, o(14) at pos2
    # w at pos0: dim 48+22=70
    # h at pos1: dim 80+7=87
    # o at pos2: dim 112+14=126
    blk0.mlp.fc1.weight.data[0, 70] = 5.0
    blk0.mlp.fc1.weight.data[0, 87] = 5.0
    blk0.mlp.fc1.weight.data[0, 126] = 5.0
    blk0.mlp.fc1.bias.data[0] = -10.0
    blk0.mlp.fc2.weight.data[44+3, 0] = 10.0  # class 3
    
    # Detect "where": w at pos0, h at pos1, e(4) at pos2, r(17) at pos3
    # w: dim 70, h: dim 87, e: dim 112+4=116, r at pos3: dim 17
    blk0.mlp.fc1.weight.data[1, 70] = 5.0
    blk0.mlp.fc1.weight.data[1, 87] = 5.0
    blk0.mlp.fc1.weight.data[1, 116] = 5.0
    blk0.mlp.fc1.weight.data[1, 17] = 5.0
    blk0.mlp.fc1.bias.data[1] = -14.0
    blk0.mlp.fc2.weight.data[44+4, 1] = 10.0  # class 4
    
    # Detect "when": w at pos0, h at pos1, e at pos2, n(13) at pos3
    blk0.mlp.fc1.weight.data[2, 70] = 5.0
    blk0.mlp.fc1.weight.data[2, 87] = 5.0
    blk0.mlp.fc1.weight.data[2, 116] = 5.0
    blk0.mlp.fc1.weight.data[2, 13] = 5.0
    blk0.mlp.fc1.bias.data[2] = -14.0
    blk0.mlp.fc2.weight.data[44+5, 2] = 10.0  # class 5
    
    # Detect "how": h(7) at pos0, o(14) at pos1, w(22) at pos2
    # h: dim 48+7=55, o: dim 80+14=94, w: dim 112+22=134 -> out of bounds!
    # Use only first 16 chars for pos2, so w(22) won't fit
    # Let's skip "how" for now or use a different encoding
    
    # Detect "why": w at pos0, h at pos1, y(24) at pos2
    # w: dim 70, h: dim 87, y: dim 112+24=136 -> out of bounds!
    # Same problem
    
    # Detect "what": w at pos0, h at pos1, a(0) at pos2, t(19) at pos3
    # w: dim 70, h: dim 87, a: dim 112+0=112, t: dim 19
    blk0.mlp.fc1.weight.data[5, 70] = 5.0
    blk0.mlp.fc1.weight.data[5, 87] = 5.0
    blk0.mlp.fc1.weight.data[5, 112] = 5.0
    blk0.mlp.fc1.weight.data[5, 19] = 5.0
    blk0.mlp.fc1.bias.data[5] = -14.0
    blk0.mlp.fc2.weight.data[44+1, 5] = 10.0  # class 1
    
    # ---- LAYER 2: propagate class scores to final position ----
    blk1 = model.blocks[1]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    # Copy class scores from pos 80 to pos 81
    blk1.attn.W_q.weight.data[0, 47] = 5.0  # Q: I'm at pos 81
    blk1.attn.W_k.weight.data[0, 46] = 5.0  # K: looking for pos 80
    for i in range(6):
        blk1.attn.W_v.weight.data[i, 44+i] = 5.0
    for i in range(6):
        blk1.attn.W_o.weight.data[44+i, i] = 5.0
    
    blk1.ln2.weight.data.fill_(1.0)
    blk1.ln2.bias.data.fill_(0.0)
    # Identity MLP
    for i in range(min(64, model.d_ff)):
        blk1.mlp.fc1.weight.data[i, i] = 2.0
        blk1.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- LAYER 3: final aggregation ----
    blk2 = model.blocks[2]
    blk2.ln1.weight.data.fill_(1.0)
    blk2.ln1.bias.data.fill_(0.0)
    blk2.ln2.weight.data.fill_(1.0)
    blk2.ln2.bias.data.fill_(0.0)
    # Identity attention and MLP
    for i in range(min(64, model.d_ff)):
        blk2.mlp.fc1.weight.data[i, i] = 2.0
        blk2.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- OUTPUT HEAD ----
    # Map class scores (dims 44-49) to vocab logits for '0'-'5' (indices 30-35)
    for cls in range(6):
        model.head.weight.data[30 + cls, 44 + cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "BigModel-v1"
model_description = "d_model=128 first-word detector with clean character encoding"

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
    model = build_model(task).to(args.device)
    accuracy, _ = evaluate(model, task, n_samples=args.n_samples, seed=args.seed, device=args.device, verbose=args.verbose)
    n_params = sum(p.numel() for p in model.parameters())
    upsert_overall_results([{
        "task": args.task, "accuracy": f"{accuracy:.4f}", "status": "",
        "model_shorthand_name": model_shorthand_name, "n_params": f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)
    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
