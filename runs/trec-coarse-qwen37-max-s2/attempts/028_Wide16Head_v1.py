"""
Interpretable transformer for TREC coarse classification.
Strategy: d_model=512, 16 heads (dh=32), 16-position char copy + MLP word detection.
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=256, n_heads=8, n_layers=1, d_ff=1024):
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
    16-position char copy + MLP word detection.
    d_model=512, 16 heads, dh=32. Each head h copies char from pos h.
    MLP detects words and writes class scores to dims 0-5.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model   # 512
    H = model.n_heads   # 16
    dh = d // H          # 32
    d_ff = model.d_ff    # 2048
    
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
    
    # Marker dim for '=' token (used for Q in attention)
    ANSWER_Q_DIM = 40
    if '=' in stoi:
        model.token_emb.weight.data[stoi['='], ANSWER_Q_DIM] = 5.0
    
    # ---- POSITION EMBEDDING ----
    # Each position p (0-15) gets a marker at dim POS_BASE + p
    POS_BASE = 48
    for pos in range(min(16, model.max_seq_len)):
        model.pos_emb.weight.data[pos, POS_BASE + pos] = 5.0
    
    # ---- LAYER 0 ATTENTION ----
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    for h in range(min(16, H)):
        hs = h * dh
        
        # Q: from answer marker dim
        blk0.attn.W_q.weight.data[hs, ANSWER_Q_DIM] = 10.0
        
        # K: match position h via position marker
        blk0.attn.W_k.weight.data[hs, POS_BASE + h] = 10.0
        
        # V: copy ALL 32 char dims into head's subspace
        for i in range(32):
            blk0.attn.W_v.weight.data[hs + i, i] = 3.0
        
        # O: identity projection back
        for i in range(32):
            blk0.attn.W_o.weight.data[hs + i, hs + i] = 3.0
    
    # After attention at answer position:
    # Head h copies char from pos h to dims h*32 to h*32+31
    
    def pos_char_dim(pos, char):
        """Get the residual stream dim for char at position pos after attention"""
        cd = char_to_dim(char)
        if cd < 0:
            return -1
        return pos * dh + cd
    
    # ---- LAYER 0 MLP: Word detection → class scores at dims 0-5 ----
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    SCALE = 5.0
    
    # Comprehensive word/phrase list with priority weights
    # Format: (pattern, class, priority)
    # Higher priority = stronger signal (longer/more specific patterns)
    patterns = [
        # === HUM (class 3) ===
        ("who", 3, 1),
        ("whom", 3, 1),
        ("whose", 3, 1),
        ("name", 3, 1),
        ("who is", 3, 2),
        ("who was", 3, 2),
        ("who are", 3, 2),
        ("who did", 3, 2),
        ("who does", 3, 2),
        ("who has", 3, 2),
        ("who have", 3, 2),
        ("who won", 3, 2),
        ("who played", 3, 2),
        ("who invented", 3, 2),
        ("who created", 3, 2),
        ("who wrote", 3, 2),
        ("who said", 3, 2),
        ("who discovered", 3, 2),
        ("who founded", 3, 2),
        ("who first", 3, 2),
        ("who starred", 3, 2),
        ("who directed", 3, 2),
        ("who composed", 3, 2),
        ("who painted", 3, 2),
        ("who killed", 3, 2),
        ("who married", 3, 2),
        ("who became", 3, 2),
        ("who served", 3, 2),
        ("who sang", 3, 2),
        ("who portrayed", 3, 2),
        ("who performed", 3, 2),
        ("who appeared", 3, 2),
        ("name the", 3, 2),
        ("name a", 3, 2),
        
        # === LOC (class 4) ===
        ("where", 4, 1),
        ("where is", 4, 2),
        ("where was", 4, 2),
        ("where can", 4, 2),
        ("where did", 4, 2),
        ("where do", 4, 2),
        ("where does", 4, 2),
        ("where are", 4, 2),
        ("where were", 4, 2),
        
        # === NUM (class 5) ===
        ("when", 5, 1),
        ("when was", 5, 2),
        ("when did", 5, 2),
        ("when is", 5, 2),
        ("when do", 5, 2),
        ("when does", 5, 2),
        ("when were", 5, 2),
        ("how", 5, 1),
        ("how many", 5, 3),
        ("how much", 5, 3),
        ("how long", 5, 3),
        ("how old", 5, 3),
        ("how fast", 5, 3),
        ("how tall", 5, 3),
        ("how far", 5, 3),
        ("how high", 5, 3),
        ("how deep", 5, 3),
        ("how wide", 5, 3),
        ("how big", 5, 3),
        ("how often", 5, 3),
        ("in what year", 5, 3),
        ("in what decade", 5, 3),
        
        # === DESC (class 2) ===
        ("why", 2, 1),
        ("why is", 2, 2),
        ("why do", 2, 2),
        ("why does", 2, 2),
        ("why was", 2, 2),
        ("why did", 2, 2),
        ("why are", 2, 2),
        ("why were", 2, 2),
        ("why can", 2, 2),
        ("define", 2, 1),
        ("describe", 2, 1),
        ("explain", 2, 1),
        ("how did", 2, 3),
        ("how do", 2, 3),
        ("how does", 2, 3),
        ("how can", 2, 3),
        ("how could", 2, 3),
        ("how to", 2, 3),
        ("how is", 2, 3),
        ("how was", 2, 3),
        ("how come", 2, 3),
        
        # === ENTY (class 1) ===
        ("what", 1, 1),
        ("which", 1, 1),
        ("what is", 2, 2),  # "what is X" often DESC
        ("what are", 2, 2),
        ("what was", 1, 2),
        ("what were", 1, 2),
        ("what does", 0, 2),  # "what does X stand for" often ABBR
        ("what did", 1, 2),
        ("what do", 1, 2),
        ("what city", 4, 3),
        ("what country", 4, 3),
        ("what state", 4, 3),
        ("what year", 5, 3),
        ("what day", 5, 3),
        ("what date", 5, 3),
        ("what time", 5, 3),
        ("what kind", 1, 2),
        ("what type", 1, 2),
        ("what color", 1, 2),
        ("what animal", 1, 2),
        ("what sport", 1, 2),
        ("what movie", 1, 2),
        ("what book", 1, 2),
        ("what song", 1, 2),
        ("what food", 1, 2),
        ("what language", 1, 2),
        ("what river", 4, 3),
        ("what mountain", 4, 3),
        ("what ocean", 4, 3),
        ("what lake", 4, 3),
        ("what island", 4, 3),
        ("what continent", 4, 3),
        ("what name", 3, 3),
        ("what company", 3, 3),
        ("what president", 3, 3),
        ("what actor", 3, 3),
        ("what actress", 3, 3),
        ("what singer", 3, 3),
        ("what author", 3, 3),
        ("what writer", 3, 3),
        ("what team", 3, 3),
        ("what group", 3, 3),
        ("which of", 3, 2),
        ("which country", 4, 3),
        ("which city", 4, 3),
        ("which is", 1, 2),
        ("which was", 1, 2),
        ("which are", 1, 2),
        
        # === ABBR (class 0) ===
        ("what does", 0, 2),
        ("what do", 0, 2),
        ("what did", 0, 2),
    ]
    
    # Remove duplicates, keeping higher priority
    seen = {}
    for pattern, cls, priority in patterns:
        if pattern not in seen or priority > seen[pattern][1]:
            seen[pattern] = (cls, priority)
    
    patterns = [(p, c, pr) for p, (c, pr) in seen.items()]
    
    neuron_idx = 0
    for word, cls, priority in patterns:
        if neuron_idx >= d_ff:
            break
        
        # Split word into characters, handling spaces
        chars_with_pos = []
        pos = 0
        for ch in word:
            if ch == ' ':
                pos += 1
                continue
            chars_with_pos.append((pos, ch))
            pos += 1
        
        n_chars = len(chars_with_pos)
        if max(p for p, _ in chars_with_pos) >= 16:
            continue
        
        # Check all chars can be mapped
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
        
        # fc1: detect this word
        strength = SCALE * priority
        for d_out in dims:
            blk0.mlp.fc1.weight.data[neuron_idx, d_out] = strength
        
        blk0.mlp.fc1.bias.data[neuron_idx] = -strength * (n_chars - 0.5)
        
        # fc2: write to class score dim (dims 0-5 for classes 0-5)
        output_strength = 10.0 * priority
        blk0.mlp.fc2.weight.data[cls, neuron_idx] = output_strength
        
        neuron_idx += 1
    
    # Default: weak bias toward class 1 (most common)
    if neuron_idx < d_ff:
        blk0.mlp.fc1.bias.data[neuron_idx] = 5.0
        blk0.mlp.fc2.weight.data[1, neuron_idx] = 1.0
        neuron_idx += 1
    
    # ---- OUTPUT HEAD ----
    for cls in range(6):
        model.head.weight.data[30 + cls, cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "Wide16Head-v1"
model_description = "d_model=512, 16 heads copying pos 0-15, priority-weighted word patterns"

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
        d_model=512,
        n_heads=16,
        d_ff=2048,
        n_layers=1
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
