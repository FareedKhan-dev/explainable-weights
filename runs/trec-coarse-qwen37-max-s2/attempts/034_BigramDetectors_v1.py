"""
Two-layer architecture for TREC coarse classification.
Layer 1: Extract character patterns from positions 0-7
Layer 2: Aggregate and classify

Architecture: d_model=256, 8 heads (dh=32), d_ff=1024, n_layers=2
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=256, n_heads=8, n_layers=2, d_ff=1024):
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
    Two-layer architecture:
    Layer 1: Copy chars from positions 0-7 to answer position
    Layer 2: Detect words and classify
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
    
    # ---- TOKEN EMBEDDING ----
    for char, idx in stoi.items():
        cd = char_to_dim(char)
        if cd >= 0:
            model.token_emb.weight.data[idx, cd] = 3.0
    
    # Marker dim for '=' token (used for Q in layer 1)
    ANSWER_Q_DIM = 40
    if '=' in stoi:
        model.token_emb.weight.data[stoi['='], ANSWER_Q_DIM] = 5.0
    
    # ---- POSITION EMBEDDING ----
    POS_BASE = 48
    for pos in range(min(8, model.max_seq_len)):
        model.pos_emb.weight.data[pos, POS_BASE + pos] = 5.0
    
    # ---- LAYER 1: Copy chars from positions 0-7 to answer position ----
    blk1 = model.blocks[0]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    for h in range(min(8, H)):
        hs = h * dh
        blk1.attn.W_q.weight.data[hs, ANSWER_Q_DIM] = 10.0
        blk1.attn.W_k.weight.data[hs, POS_BASE + h] = 10.0
        for i in range(32):
            blk1.attn.W_v.weight.data[hs + i, i] = 3.0
        for i in range(32):
            blk1.attn.W_o.weight.data[hs + i, hs + i] = 3.0
    
    # Layer 1 MLP: Identity (pass through)
    blk1.ln2.weight.data.fill_(1.0)
    blk1.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk1.mlp.fc1.weight.data[i, i] = 1.0
        blk1.mlp.fc2.weight.data[i, i] = 1.0
    
    def pos_char_dim(pos, char):
        """Get the residual stream dim for char at position pos after layer 1"""
        cd = char_to_dim(char)
        if cd < 0:
            return -1
        return pos * dh + cd
    
    # ---- LAYER 2: Detect words and classify ----
    blk2 = model.blocks[1]
    blk2.ln1.weight.data.fill_(1.0)
    blk2.ln1.bias.data.fill_(0.0)
    
    # Layer 2 attention: identity (just pass through)
    for h in range(H):
        hs = h * dh
        for i in range(dh):
            blk2.attn.W_q.weight.data[hs + i, hs + i] = 1.0
            blk2.attn.W_k.weight.data[hs + i, hs + i] = 1.0
            blk2.attn.W_v.weight.data[hs + i, hs + i] = 1.0
            blk2.attn.W_o.weight.data[hs + i, hs + i] = 1.0
    
    # Layer 2 MLP: Word detection
    blk2.ln2.weight.data.fill_(1.0)
    blk2.ln2.bias.data.fill_(0.0)
    
    SCALE = 5.0
    
    # Word detection rules
    words = []
    
    # Character bigram detectors - more robust than full words
    extra_words = [
        # Bigrams for HUM (class 3)
        ("wh", 3), ("ho", 3), ("na", 3), ("am", 3),
        # Bigrams for LOC (class 4)
        ("wh", 4), ("he", 4), ("ci", 4), ("co", 4), ("st", 4), ("ri", 4),
        # Bigrams for NUM (class 5)
        ("wh", 5), ("he", 5), ("ho", 5), ("ow", 5), ("ye", 5), ("da", 5),
        # Bigrams for DESC (class 2)
        ("wh", 2), ("hy", 2), ("de", 2), ("fi", 2), ("ho", 2),
        # Bigrams for ENTY (class 1)
        ("wh", 1), ("ha", 1), ("wh", 1), ("ic", 1), ("ki", 1), ("ty", 1),
        # Bigrams for ABBR (class 0)
        ("wh", 0), ("do", 0), ("es", 0),
    ]
    
    # Also keep some key single words
    key_words = [
        ("who", 3), ("whom", 3), ("where", 4), ("when", 5), ("why", 2),
        ("what", 1), ("which", 1), ("how", 5), ("name", 3), ("define", 2),
    ]
    
    neuron_idx = 0
    for word, cls in words + extra_words:
        if neuron_idx >= d_ff:
            break
        
        chars_with_pos = []
        pos = 0
        for ch in word:
            if ch == ' ':
                pos += 1
                continue
            chars_with_pos.append((pos, ch))
            pos += 1
        
        n_chars = len(chars_with_pos)
        if max(p for p, _ in chars_with_pos) >= 8:
            continue
        
        dims = []
        valid = True
        for p, ch in chars_with_pos:
            d_out = pos_char_dim(p, ch)
            if d_out < 0 or d_out >= d:
                valid = False
                break
            dims.append(d_out)
        
        if not valid:
            continue
        
        for d_out in dims:
            blk2.mlp.fc1.weight.data[neuron_idx, d_out] = SCALE
        blk2.mlp.fc1.bias.data[neuron_idx] = -SCALE * (n_chars - 0.5)
        blk2.mlp.fc2.weight.data[cls, neuron_idx] = 10.0
        neuron_idx += 1
    
    # Default bias
    if neuron_idx < d_ff:
        blk2.mlp.fc1.bias.data[neuron_idx] = 5.0
        blk2.mlp.fc2.weight.data[1, neuron_idx] = 1.0
        neuron_idx += 1
    
    # ---- OUTPUT HEAD ----
    for cls in range(6):
        model.head.weight.data[30 + cls, cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "BigramDetectors-v1"
model_description = "Character bigram detectors: 2-char patterns for more robust classification"

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
        n_layers=2
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
