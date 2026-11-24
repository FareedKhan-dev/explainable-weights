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
    BigModel256 base + comprehensive patterns v3
    
    Building on the successful 42% approach with:
    - More word patterns (especially multi-word)
    - Better calibrated thresholds
    - Stronger priority weighting
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
    
    for char, idx in stoi.items():
        if char in char_dim:
            model.token_emb.weight.data[idx, char_dim[char]] = 3.0
    
    def char_to_dim(ch):
        return char_dim.get(ch, -1)
    
    # ---- POSITION EMBEDDING ----
    for pos in range(8):
        model.pos_emb.weight.data[pos, 30 + pos] = 5.0
    if model.max_seq_len > 80:
        model.pos_emb.weight.data[80, 40] = 5.0
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 41] = 5.0
    
    # ---- LAYER 1: Copy chars from positions 0-7 ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    for head_idx in range(8):
        h_start = head_idx * dh
        o_start = head_idx * dh
        
        blk0.attn.W_q.weight.data[h_start, 40] = 10.0
        blk0.attn.W_k.weight.data[h_start, 30 + head_idx] = 10.0
        for i in range(30):
            blk0.attn.W_v.weight.data[h_start + i, i] = 3.0
        for i in range(30):
            blk0.attn.W_o.weight.data[o_start + i, h_start + i] = 3.0
    
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    # ---- LAYER 1 MLP: detect words ----
    def pos_char_dim(pos, char):
        cd = char_to_dim(char)
        if cd < 0:
            return -1
        return pos * dh + cd
    
    # Comprehensive word patterns
    words = [
        # HUM (class 3)
        ("who", 3, 10),
        ("whom", 3, 10),
        ("whose", 3, 10),
        ("name", 3, 8),
        
        # LOC (class 4)
        ("where", 4, 10),
        ("what city", 4, 15),
        ("what country", 4, 15),
        ("what state", 4, 15),
        ("what river", 4, 15),
        ("what mountain", 4, 15),
        
        # NUM (class 5)
        ("when", 5, 10),
        ("how", 5, 6),
        ("in", 5, 3),
        ("what year", 5, 15),
        ("how many", 5, 15),
        ("how much", 5, 15),
        ("how old", 5, 15),
        ("how long", 5, 15),
        ("what time", 5, 12),
        ("what day", 5, 12),
        ("what date", 5, 12),
        ("what age", 5, 12),
        ("what number", 5, 12),
        ("in what year", 5, 18),
        
        # DESC (class 2)
        ("why", 2, 10),
        ("define", 2, 10),
        ("for", 2, 3),
        ("what is", 2, 12),
        ("what are", 2, 12),
        ("what causes", 2, 12),
        
        # ENTY (class 1)
        ("what", 1, 4),
        ("which", 1, 8),
        ("the", 1, 2),
        ("what kind", 1, 12),
        ("what type", 1, 12),
        ("what sort", 1, 12),
        ("what name", 1, 12),
        ("what color", 1, 12),
        ("what company", 1, 12),
        ("what team", 1, 12),
        ("what band", 1, 12),
        ("what movie", 1, 12),
        ("what book", 1, 12),
        ("what sport", 1, 12),
        ("what animal", 1, 12),
        ("what food", 1, 12),
        
        # ABBR (class 0)
        ("does", 0, 10),
        ("what does", 0, 15),
        ("what do", 0, 12),
    ]
    
    words.sort(key=lambda x: -x[2])
    
    neuron_idx = 0
    for word, cls, priority in words:
        if neuron_idx >= d_ff:
            break
        
        n_chars = len(word)
        if n_chars > 8:
            continue
        
        char_count = 0
        for i, ch in enumerate(word):
            d_out = pos_char_dim(i, ch)
            if d_out >= 0 and d_out < d:
                blk0.mlp.fc1.weight.data[neuron_idx, d_out] = 5.0
                char_count += 1
        
        if char_count == 0:
            continue
        
        blk0.mlp.fc1.bias.data[neuron_idx] = -5.0 * (char_count - 0.5)
        blk0.mlp.fc2.weight.data[42 + cls, neuron_idx] = priority
        
        neuron_idx += 1
    
    # Default class 1 bias
    if neuron_idx < d_ff:
        blk0.mlp.fc1.bias.data[neuron_idx] = 5.0
        blk0.mlp.fc2.weight.data[42 + 1, neuron_idx] = 2.0
        neuron_idx += 1
    
    # ---- LAYER 2: propagate ----
    blk1 = model.blocks[1]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    blk1.attn.W_q.weight.data[0, 41] = 10.0
    blk1.attn.W_k.weight.data[0, 40] = 10.0
    for i in range(6):
        blk1.attn.W_v.weight.data[i, 42+i] = 5.0
    for i in range(6):
        blk1.attn.W_o.weight.data[42+i, i] = 5.0
    
    blk1.ln2.weight.data.fill_(1.0)
    blk1.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk1.mlp.fc1.weight.data[i, i] = 2.0
        blk1.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- LAYER 3: identity ----
    blk2 = model.blocks[2]
    blk2.ln1.weight.data.fill_(1.0)
    blk2.ln1.bias.data.fill_(0.0)
    blk2.ln2.weight.data.fill_(1.0)
    blk2.ln2.bias.data.fill_(0.0)
    for i in range(min(d, d_ff)):
        blk2.mlp.fc1.weight.data[i, i] = 2.0
        blk2.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- OUTPUT HEAD ----
    for cls in range(6):
        model.head.weight.data[30 + cls, 42 + cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "BigModel256-v3"
model_description = "d=256, 8 heads with comprehensive multi-word patterns and priority weighting"

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
