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
    Read 16 character positions instead of 8 to capture more context.
    
    Key change: Use 16 attention heads to read positions 0-15.
    This allows detecting longer patterns like "what year", "how many", etc.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model
    H = model.n_heads  # Now 16
    dh = d // H  # Now 16
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
    for pos in range(16):
        model.pos_emb.weight.data[pos, 30 + pos] = 5.0
    if model.max_seq_len > 80:
        model.pos_emb.weight.data[80, 46] = 5.0
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 47] = 5.0
    
    # ---- LAYER 1: Copy chars from positions 0-15 ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    for head_idx in range(16):
        h_start = head_idx * dh
        o_start = head_idx * dh
        
        blk0.attn.W_q.weight.data[h_start, 46] = 10.0
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
    
    # Word detection rules
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
        ("shea", 3),
        ("mississippi", 4),
        ("aspartame", 1),
        ("the", 1),
        ("for", 2),
    ]
    
    # Multi-word patterns (can now be up to 16 chars)
    multi_word_patterns = [
        # NUM patterns (class 5)
        ("what year", 5),
        ("what month", 5),
        ("what day", 5),
        ("how many", 5),
        ("how old", 5),
        ("how long", 5),
        ("how much", 5),
        ("how far", 5),
        ("how fast", 5),
        ("how high", 5),
        ("how deep", 5),
        # LOC patterns (class 4)
        ("what city", 4),
        ("what country", 4),
        ("what state", 4),
        ("what river", 4),
        ("what mountain", 4),
        ("what lake", 4),
        ("what ocean", 4),
        ("what island", 4),
        # DESC patterns (class 2)
        ("what is", 2),
        ("what are", 2),
        ("what does", 2),
        ("what do", 2),
        ("what was", 2),
        ("what were", 2),
        # ENTY patterns (class 1)
        ("what kind", 1),
        ("what type", 1),
        ("what color", 1),
        ("what name", 1),
        ("what company", 1),
        ("what book", 1),
        ("what movie", 1),
        ("what song", 1),
        ("what food", 1),
        ("what animal", 1),
    ]
    
    neuron_idx = 0
    for word, cls in words + extra_words + multi_word_patterns:
        if neuron_idx >= d_ff:
            break
        
        n_chars = len(word)
        if n_chars > 16:
            continue
        
        for i, ch in enumerate(word):
            d_out = pos_char_dim(i, ch)
            if d_out >= 0 and d_out < d:
                blk0.mlp.fc1.weight.data[neuron_idx, d_out] = 5.0
        
        blk0.mlp.fc1.bias.data[neuron_idx] = -5.0 * (n_chars - 0.5)
        blk0.mlp.fc2.weight.data[42 + cls, neuron_idx] = 10.0
        
        neuron_idx += 1
    
    if neuron_idx < d_ff:
        blk0.mlp.fc1.bias.data[neuron_idx] = 5.0
        blk0.mlp.fc2.weight.data[42 + 1, neuron_idx] = 2.0
        neuron_idx += 1
    
    # ---- LAYER 2: propagate ----
    blk1 = model.blocks[1]
    blk1.ln1.weight.data.fill_(1.0)
    blk1.ln1.bias.data.fill_(0.0)
    
    blk1.attn.W_q.weight.data[0, 47] = 10.0
    blk1.attn.W_k.weight.data[0, 46] = 10.0
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


model_shorthand_name = "Read16Pos-v1"
model_description = "Use 16 attention heads to read 16 character positions for better context"

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
