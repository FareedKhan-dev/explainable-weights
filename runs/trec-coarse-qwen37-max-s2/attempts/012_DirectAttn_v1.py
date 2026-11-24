"""
Interpretable transformer for TREC coarse classification.
Strategy: d_model=256, 8 heads, detect first word using character copies.
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
    Direct class computation via attention heads.
    
    Strategy: Use attention heads to directly detect key words and output class scores.
    Each head detects one specific word pattern and outputs the corresponding class score.
    
    Dimension layout (d_model=1024):
    - Dims 0-31: one-hot char encoding
    - Dims 32-47: position markers
    - Dim 48: pos 80 (= sign) marker
    - Dim 49: pos 81 (answer) marker
    - Dims 50-55: class scores (accumulated from multiple heads)
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model
    H = model.n_heads
    dh = d // H
    d_ff = model.d_ff
    
    # Zero everything
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDING ----
    char_dim = {}
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i, ch in enumerate(letters):
        char_dim[ch] = i
    char_dim[' '] = 26
    char_dim["'"] = 27
    char_dim['_'] = 28
    char_dim['='] = 29
    
    EMBED_SCALE = 5.0
    for char, idx in stoi.items():
        if char in char_dim:
            model.token_emb.weight.data[idx, char_dim[char]] = EMBED_SCALE
    
    def char_to_dim(ch):
        return char_dim.get(ch, -1)
    
    # ---- POSITION EMBEDDING ----
    POS_SCALE = 8.0
    for pos in range(16):
        model.pos_emb.weight.data[pos, 32 + pos] = POS_SCALE
    if model.max_seq_len > 80:
        model.pos_emb.weight.data[80, 48] = POS_SCALE
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 49] = POS_SCALE
    
    # ---- LAYER 1: Each head detects one word pattern ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    # Word patterns: (word, class, strength)
    # We'll use multiple heads to detect the same word with different strengths
    word_patterns = [
        # High-confidence patterns
        ("who", 3, 8.0),
        ("whom", 3, 8.0),
        ("whose", 3, 8.0),
        ("where", 4, 8.0),
        ("when", 5, 8.0),
        ("why", 2, 8.0),
        ("what year", 5, 10.0),
        ("what city", 4, 10.0),
        ("what country", 4, 10.0),
        ("how many", 5, 10.0),
        ("how much", 5, 10.0),
        ("how old", 5, 10.0),
        ("how long", 5, 10.0),
        ("what is", 2, 6.0),
        ("what are", 2, 6.0),
        ("what does", 0, 10.0),
        ("what do", 0, 10.0),
        ("define", 2, 8.0),
        ("name", 3, 6.0),
        ("which", 1, 4.0),
        ("what", 1, 3.0),  # Default fallback
        ("how", 2, 5.0),
        ("in what year", 5, 12.0),
        ("what kind", 1, 8.0),
        ("what type", 1, 8.0),
    ]
    
    # Each head will detect one pattern
    # Head h: query from pos 80, key matches positions for word[h], value outputs class score
    for head_idx, (word, cls, strength) in enumerate(word_patterns[:H]):
        h_start = head_idx * dh
        n_chars = len(word)
        
        # Query: from pos 80 (= sign position)
        blk0.attn.W_q.weight.data[h_start, 48] = 15.0
        
        # Key: match all characters in the word at their positions
        # We need to encode the pattern in the key weights
        for i, ch in enumerate(word):
            cd = char_to_dim(ch)
            if cd >= 0:
                # Key looks for this character at this position
                # After adding pos embedding, position i has marker at dim 32+i
                # and character at dim cd
                # We want to match both
                blk0.attn.W_k.weight.data[h_start, 32 + i] = 3.0  # Position marker
                blk0.attn.W_k.weight.data[h_start, cd] = 3.0  # Character
        
        # Value: output class score if pattern matches
        # The value should be high when all characters match
        # We'll encode the class in the value output
        blk0.attn.W_v.weight.data[h_start + cls, 0] = strength  # Use dim 0 as a constant
        
        # Output projection: route to class score dimensions
        blk0.attn.W_o.weight.data[50 + cls, h_start + cls] = 2.0
    
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    # MLP: identity pass-through
    for i in range(min(d, d_ff)):
        blk0.mlp.fc1.weight.data[i, i] = 3.0
        blk0.mlp.fc1.bias.data[i] = 0.0
        blk0.mlp.fc2.weight.data[i, i] = 3.0
    
    # ---- LAYER 2: propagate class scores to answer position ----
    blk1 = model.blocks[1]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    # Copy class scores from pos 80 to pos 81
    blk1.attn.W_q.weight.data[0, 49] = 15.0
    blk1.attn.W_k.weight.data[0, 48] = 15.0
    for i in range(6):
        blk1.attn.W_v.weight.data[i, 50+i] = 5.0
    for i in range(6):
        blk1.attn.W_o.weight.data[50+i, i] = 5.0
    
    blk1.ln2.weight.data.fill_(1.0)
    blk1.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk1.mlp.fc1.weight.data[i, i] = 3.0
        blk1.mlp.fc2.weight.data[i, i] = 3.0
    
    # ---- LAYER 3: identity pass-through ----
    blk2 = model.blocks[2]
    blk2.ln1.weight.data.fill_(1.0)
    blk2.ln1.bias.data.fill_(0.0)
    blk2.ln2.weight.data.fill_(1.0)
    blk2.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk2.mlp.fc1.weight.data[i, i] = 3.0
        blk2.mlp.fc2.weight.data[i, i] = 3.0
    
    # ---- OUTPUT HEAD ----
    for cls in range(6):
        model.head.weight.data[30 + cls, 50 + cls] = 15.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "DirectAttn-v1"
model_description = "Use attention heads to directly detect word patterns and output class scores, avoiding post-LayerNorm MLP"

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
        d_model=1024,
        n_heads=16,
        d_ff=4096,
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
