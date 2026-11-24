"""
Interpretable transformer for TREC coarse classification.
Strategy: Simple 1-layer approach. At answer position, attend to first word chars,
use MLP to classify, output head reads class scores.

Key insight: use a single transformer block. The attention copies chars from
positions 0-7 into the residual at the answer position. The MLP detects the
first word and writes class scores. The output head reads those scores.

Architecture: d_model=256, 8 heads (dh=32), d_ff=1024, n_layers=1
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
    Single-layer first-word classifier.
    
    Token embedding: one-hot in dims 0-31 (26 letters + space + ' + _ + = + ? + !)
    Position embedding: each pos p gets a marker at dim 32 + (p % 224)
    
    Layer 0 attention: At every position (including answer pos), each head h
    attends to position h and copies its char embedding.
    
    Layer 0 MLP: Detect first words and write class scores to dims 0-5.
    
    Output head: Map dims 0-5 to token indices 30-35 (digits '0'-'5').
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
    
    # ---- CHARACTER MAPPING (shared across all positions) ----
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
    
    # Special marker dim for '=' token (used for Q in attention)
    ANSWER_Q_DIM = 40
    if '=' in stoi:
        model.token_emb.weight.data[stoi['='], ANSWER_Q_DIM] = 5.0
    
    # ---- POSITION EMBEDDING ----
    # Each position p gets a marker at dim 48 + p (for positions 0-7 only)
    # This allows head h to key on position h
    POS_BASE = 48
    for pos in range(min(8, model.max_seq_len)):
        model.pos_emb.weight.data[pos, POS_BASE + pos] = 5.0
    
    # ---- LAYER 0 ATTENTION ----
    # Each head h: at any position, attend to position h, copy its char embedding
    blk0 = model.blocks[0]
    blk0.ln1.weight.data.fill_(1.0)
    blk0.ln1.bias.data.fill_(0.0)
    
    for h in range(min(8, H)):
        hs = h * dh
        
        # Q: universal query (same for all positions)
        # Use a constant signal: read from the char dims (which are always non-zero)
        # Actually, let's use a different approach:
        # Q should be high at the answer position to attend to first-word positions
        # Use the '=' marker dim for Q
        blk0.attn.W_q.weight.data[hs, ANSWER_Q_DIM] = 10.0
        
        # K: match position h via position marker
        blk0.attn.W_k.weight.data[hs, POS_BASE + h] = 10.0
        
        # V: copy ALL 32 char dims (shared dims 0-31) into head's subspace
        for i in range(32):
            blk0.attn.W_v.weight.data[hs + i, i] = 3.0
        
        # O: project back (identity within head subspace)
        for i in range(32):
            blk0.attn.W_o.weight.data[hs + i, hs + i] = 3.0
    
    # After attention at the answer position:
    # Head 0 copies chars from pos 0 to dims 0-31
    # Head 1 copies chars from pos 1 to dims 32-63
    # ...
    # Head 7 copies chars from pos 7 to dims 224-255
    #
    # The char at position p, character c, is stored at:
    # dim = p * 32 + char_to_dim(c)
    # (as long as char_to_dim(c) < 32, which is always true)
    
    def pos_char_dim(pos, char):
        """Get the residual stream dim for char at position pos after attention"""
        cd = char_to_dim(char)
        if cd < 0:
            return -1
        return pos * dh + cd
    
    # ---- LAYER 0 MLP: Word detection → class scores at dims 0-5 ----
    blk0.ln2.weight.data.fill_(1.0)
    blk0.ln2.bias.data.fill_(0.0)
    
    # IMPORTANT: After attention, the residual at answer pos contains:
    # - Original input (token_emb + pos_emb at answer pos)
    # - Attention output (chars from pos 0-7)
    # 
    # After LN2, the char signals are normalized.
    # The MLP needs to detect words based on these normalized values.
    #
    # Since LN normalizes, the relative ordering of dims matters more than absolute values.
    # For char 'w' at pos 0: dim 22 has value 3.0 * 3.0 = 9.0 (from attention),
    # plus original input at dim 22 (depends on answer token).
    
    SCALE = 5.0
    
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
        # Common two-word patterns
        ("in", 5),
        ("the", 1),
        ("for", 2),
        
        # "how" patterns
        ("how many", 5),
        ("how much", 5),
        ("how long", 5),
        ("how old", 5),
        ("how fast", 5),
        ("how tall", 5),
        ("how far", 5),
        ("how high", 5),
        ("how deep", 5),
        ("how wide", 5),
        ("how big", 5),
        ("how did", 2),
        ("how do", 2),
        ("how does", 2),
        ("how can", 2),
        ("how could", 2),
        ("how to", 2),
        ("how is", 2),
        ("how was", 2),
        ("how come", 2),
        
        # "what" patterns
        ("what is", 2),
        ("what are", 2),
        ("what was", 1),
        ("what were", 1),
        ("what does", 0),
        ("what did", 1),
        ("what do", 1),
        ("what city", 4),
        ("what country", 4),
        ("what state", 4),
        ("what year", 5),
        ("what day", 5),
        ("what date", 5),
        ("what time", 5),
        ("what kind", 1),
        ("what type", 1),
        ("what color", 1),
        ("what animal", 1),
        ("what sport", 1),
        ("what movie", 1),
        ("what book", 1),
        ("what song", 1),
        ("what food", 1),
        ("what language", 1),
        ("what river", 4),
        ("what mountain", 4),
        ("what ocean", 4),
        ("what lake", 4),
        ("what island", 4),
        ("what continent", 4),
        
        # "which" patterns
        ("which of", 3),
        ("which country", 4),
        ("which city", 4),
        ("which state", 4),
        ("which is", 1),
        ("which was", 1),
        ("which are", 1),
        
        # "who" patterns
        ("who is", 3),
        ("who was", 3),
        ("who are", 3),
        ("who did", 3),
        ("who does", 3),
        ("who has", 3),
        ("who have", 3),
        ("who won", 3),
        ("who played", 3),
        ("who invented", 3),
        ("who created", 3),
        ("who wrote", 3),
        ("who said", 3),
        ("who discovered", 3),
        ("who founded", 3),
        ("who first", 3),
        ("who last", 3),
        ("who starred", 3),
        ("who directed", 3),
        ("who composed", 3),
        ("who painted", 3),
        ("who killed", 3),
        ("who married", 3),
        ("who became", 3),
        ("who served", 3),
        ("who led", 3),
        ("who ran", 3),
        ("who sang", 3),
        ("who performed", 3),
        ("who appeared", 3),
        ("who portrayed", 3),
        ("who played the", 3),
        
        # "where" patterns
        ("where is", 4),
        ("where was", 4),
        ("where can", 4),
        ("where did", 4),
        ("where do", 4),
        ("where does", 4),
        ("where are", 4),
        ("where were", 4),
        ("where has", 4),
        ("where have", 4),
        
        # "when" patterns
        ("when was", 5),
        ("when did", 5),
        ("when is", 5),
        ("when do", 5),
        ("when does", 5),
        ("when were", 5),
        ("when has", 5),
        ("when have", 5),
        ("when will", 5),
        
        # "why" patterns
        ("why is", 2),
        ("why do", 2),
        ("why does", 2),
        ("why was", 2),
        ("why did", 2),
        ("why are", 2),
        ("why were", 2),
        ("why has", 2),
        ("why have", 2),
        ("why can", 2),
        ("why could", 2),
        
        # Single words
        ("define", 2),
        ("describe", 2),
        ("explain", 2),
        ("tell", 2),
        ("name", 3),
    ]
    
    neuron_idx = 0
    for word, cls in words + extra_words:
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
        if max(p for p, _ in chars_with_pos) >= 8:
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
        for d_out in dims:
            blk0.mlp.fc1.weight.data[neuron_idx, d_out] = SCALE
        
        blk0.mlp.fc1.bias.data[neuron_idx] = -SCALE * (n_chars - 0.5)
        
        # fc2: write to class score dim (use dims 0-5 for classes 0-5)
        blk0.mlp.fc2.weight.data[cls, neuron_idx] = 10.0
        
        neuron_idx += 1
    
    # Default: weak bias toward class 1 (most common) 
    if neuron_idx < d_ff:
        blk0.mlp.fc1.bias.data[neuron_idx] = 5.0  # Always active
        blk0.mlp.fc2.weight.data[1, neuron_idx] = 1.0
        neuron_idx += 1
    
    # ---- OUTPUT HEAD ----
    # Classes 0-5 → token indices 30-35 (digits '0'-'5')
    # Read from dims 0-5 (where MLP writes class scores)
    for cls in range(6):
        model.head.weight.data[30 + cls, cls] = 10.0
    
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "Simple1Layer-v2"
model_description = "Expanded word patterns: 150+ two-word combinations for better classification"

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
