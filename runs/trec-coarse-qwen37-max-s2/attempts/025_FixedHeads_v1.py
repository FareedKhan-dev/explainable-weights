"""
Interpretable transformer for TREC coarse classification.
Strategy: d_model=256, 8 heads (dh=32), first-word detection via character copy.

Key insight: each head h copies char from position h using its own dh-wide subspace.
Character info stored in dims 0-31 (shared), position markers replicated per head.

Dimension layout (d_model=256, 8 heads × 32 dims):
  Dims 0-31: character one-hot (26 letters + space + ' + _ + = + ? + !)
  For each head h (dims h*32 to h*32+31):
    h*32+0 to h*32+15: first copy of chars 0-15
    h*32+16 to h*32+21: position markers for pos 0-5
    h*32+22 to h*32+31: second copy of chars 16-25
  Dim 40 (shared): class_score marker (also set by '=' token for layer 1 attention)
  Dim 42 (shared): answer position marker (pos 81)
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=256, n_heads=8, n_layers=3, d_ff=1024):
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
    Correct multi-head character copy + MLP word detection.
    
    Each head h copies char from position h into its own dh=32 subspace.
    Within each head's subspace:
      dims 0-15: first copy of chars 0-15 (a-p)
      dims 16-21: position markers for pos 0-5
      dims 22-31: second copy of chars 16-25 (q-z) and specials
    
    MLP detects first words and writes class scores to dims 40-45.
    Layer 1 attention copies class scores from pos 80 to pos 81 (answer).
    Output head reads from dims 0-5 at answer position.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model   # 256
    H = model.n_heads   # 8
    dh = d // H          # 32
    d_ff = model.d_ff    # 1024
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # ---- CHARACTER MAPPING ----
    char_dim = {}
    for i, ch in enumerate('abcdefghijklmnopqrstuvwxyz'):
        char_dim[ch] = i
    char_dim[' '] = 26
    char_dim["'"] = 27
    char_dim['_'] = 28
    char_dim['='] = 29
    char_dim['?'] = 30
    char_dim['!'] = 31
    
    def char_to_dim(ch):
        return char_dim.get(ch, -1)
    
    def char_to_head_dim(ch):
        """Map char index to position within a head's 32-dim subspace"""
        cd = char_to_dim(ch)
        if cd < 0:
            return -1
        if cd < 16:
            return cd           # dims 0-15
        elif cd < 26:
            return cd + 6       # dims 22-31 (16->22, 17->23, ..., 25->31)
        else:
            return -1           # specials not handled in second copy
    
    # ---- TOKEN EMBEDDING ----
    for char, idx in stoi.items():
        cd = char_to_dim(char)
        if cd >= 0:
            model.token_emb.weight.data[idx, cd] = 3.0
    
    # '=' token: also set dim 40 (marker for layer 1 attention to find pos 80)
    if '=' in stoi:
        model.token_emb.weight.data[stoi['='], 40] = 5.0
    
    # ---- POSITION EMBEDDING ----
    # For each head h, position p (0-5): set dim h*32+16+p
    for pos in range(min(6, model.max_seq_len)):
        for h in range(H):
            model.pos_emb.weight.data[pos, h * dh + 16 + pos] = 5.0
    
    # Position 80 marker: set dim 32 (= 0*32+16+16, but that's beyond our marker range)
    # Actually, position 80 is identified by the '=' token (dim 40)
    # Position 81 marker: set dim 42 for layer 1 query
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 42] = 5.0
    
    # ---- LAYER 0: Copy chars from positions 0-5 to answer position ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    for h in range(min(6, H)):  # 6 heads for 6 positions (use 6 of 8 heads)
        hs = h * dh  # head start dim
        
        # Q: from position 80's ln1 output
        # At pos 80: token_emb['='] has dim 40 = 5.0, pos_emb[80] = 0 (unless < 6)
        # After LN1: dim 40 has high normalized value
        blk0.attn.W_q.weight.data[hs, 40] = 10.0
        
        # K: match position h (via position marker at dim h*32+16+h)
        blk0.attn.W_k.weight.data[hs, hs + 16 + h] = 10.0
        
        # V: copy ALL 32 shared character dims into head's subspace
        # First 16 chars (a-p): shared dims 0-15 → head dims 0-15
        for i in range(16):
            blk0.attn.W_v.weight.data[hs + i, i] = 3.0
        # Chars 16-25 (q-z): shared dims 16-25 → head dims 22-31
        for i in range(10):
            blk0.attn.W_v.weight.data[hs + 22 + i, 16 + i] = 3.0
        
        # O: project from head subspace back to residual stream
        for i in range(16):
            blk0.attn.W_o.weight.data[hs + i, hs + i] = 3.0
        for i in range(10):
            blk0.attn.W_o.weight.data[hs + 22 + i, hs + 22 + i] = 3.0
    
    # After attention at position 80:
    # Head h dims h*32+0 to h*32+15: chars 0-15 from position h
    # Head h dims h*32+22 to h*32+31: chars 16-25 from position h
    # (residual adds to original: token_emb['='] at dim 40, pos_emb[80] = ~0)
    
    # ---- LAYER 0 MLP: Word detection → class scores at dims 40-45 ----
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    # For a word char c at position p, the dim in the residual stream is:
    def pos_char_head_dim(pos, char):
        """Get the residual stream dim for char at position pos (within head pos's subspace)"""
        hd = char_to_head_dim(char)
        if hd < 0:
            return -1
        return pos * dh + hd
    
    # Word detection rules: (word, target_class)
    words = [
        # HUM (class 3)
        ("who", 3), ("whom", 3), ("whose", 3), ("name", 3),
        # LOC (class 4)
        ("where", 4),
        # NUM (class 5)
        ("when", 5), ("how", 5),
        # DESC (class 2)
        ("why", 2), ("define", 2),
        # ENTY (class 1)
        ("what", 1), ("which", 1),
        # ABBR (class 0)
        ("does", 0),
    ]
    
    extra_words = [
        ("in", 5),
        ("the", 1),
        ("for", 2),
    ]
    
    SCALE = 5.0
    neuron_idx = 0
    for word, cls in words + extra_words:
        if neuron_idx >= d_ff:
            break
        
        n_chars = len(word)
        if n_chars > 6:  # Max 6 positions
            continue
        
        # Check all chars can be mapped
        dims = []
        valid = True
        for i, ch in enumerate(word):
            d_out = pos_char_head_dim(i, ch)
            if d_out < 0 or d_out >= d:
                valid = False
                break
            dims.append(d_out)
        
        if not valid:
            continue
        
        # fc1: detect this word
        for d_out in dims:
            blk0.mlp.fc1.weight.data[neuron_idx, d_out] = SCALE
        
        blk0.mlp.fc1.bias.data[neuron_idx] = -SCALE * (n_chars - 0.5)
        
        # fc2: write to class score dim 40+cls
        blk0.mlp.fc2.weight.data[40 + cls, neuron_idx] = 10.0
        
        neuron_idx += 1
    
    # Default: weak bias toward class 1 (most common)
    if neuron_idx < d_ff:
        blk0.mlp.fc1.bias.data[neuron_idx] = 5.0  # Always active
        blk0.mlp.fc2.weight.data[40 + 1, neuron_idx] = 1.0
        neuron_idx += 1
    
    # ---- LAYER 1: Copy class scores from pos 80 to pos 81 ----
    blk1 = model.blocks[1]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    # Head 0: peaked attention from pos 81 to pos 80
    # Q at pos 81: from dim 42 (answer position marker set by pos_emb[81])
    blk1.attn.W_q.weight.data[0, 42] = 10.0
    # K at pos 80: from dim 40 ('=' token marker)
    blk1.attn.W_k.weight.data[0, 40] = 10.0
    # V: read class scores from dims 40-45
    for i in range(6):
        blk1.attn.W_v.weight.data[i, 40 + i] = 5.0
    # O: write to dims 0-5 (where output head reads from)
    for i in range(6):
        blk1.attn.W_o.weight.data[i, i] = 5.0
    
    # Heads 1-7: identity attention (preserve residual stream)
    for h in range(1, H):
        hs = h * dh
        for i in range(dh):
            blk1.attn.W_v.weight.data[hs + i, hs + i] = 1.0
            blk1.attn.W_o.weight.data[hs + i, hs + i] = 1.0
    
    blk1.ln2.weight.data.fill_(1.0)
    blk1.ln2.bias.data.fill_(0.0)
    # Identity MLP (with factor 2 to compensate for ReLU halving)
    for i in range(min(d, d_ff)):
        blk1.mlp.fc1.weight.data[i, i] = 2.0
        blk1.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- LAYER 2: identity pass-through ----
    blk2 = model.blocks[2]
    blk2.ln1.weight.data.fill_(1.0)
    blk2.ln1.bias.data.fill_(0.0)
    blk2.ln2.weight.data.fill_(1.0)
    blk2.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk2.mlp.fc1.weight.data[i, i] = 2.0
        blk2.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- OUTPUT HEAD ----
    # Classes 0-5 → token indices 30-35 (digits '0'-'5')
    # Read from dims 0-5 (where layer 1 attention copies class scores)
    for cls in range(6):
        model.head.weight.data[30 + cls, cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "FixedHeads-v1"
model_description = "Fixed multi-head char copy: each head reads chars from its own position using dual-range V weights"

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
    model = SimpleTransformer(
        vocab_size=task.vocab_size,
        max_seq_len=max_seq_len,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=3
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
