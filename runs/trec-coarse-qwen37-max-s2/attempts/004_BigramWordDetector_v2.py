"""
Interpretable transformer for TREC coarse classification.
Strategy: Use position-specific embeddings to encode character identity at each
position, then use attention to detect the first word and map it to a class.
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
    """
    Bigram first-word classifier.
    
    TREC coarse classes:
      0=ABBR, 1=ENTY, 2=DESC, 3=HUM, 4=LOC, 5=NUM
    
    First-word -> class mapping (from data analysis):
      "who" -> 3, "whom" -> 3
      "where" -> 4
      "when" -> 5
      "how" -> 5 (majority) or 2
      "why" -> 2
      "what" -> 1 (most common) or 2 or 0
      "which" -> 1 or 3 or 4
      "name" -> 3
      "in" -> various
    
    Strategy: Use token embeddings to encode character identity in separate dimensions.
    Use position embeddings to mark position 0,1,2,3 distinctly.
    Layer 1 attention: each head detects a character at a specific position.
    Layer 1 MLP: combine character detections into word detection.
    Layer 2 attention: aggregate word detections to final position.
    Layer 2 MLP + head: map to class output.
    """
    torch.manual_seed(0)
    stoi = task.stoi
    d = model.d_model  # 64
    
    # Zero everything first
    for p in model.parameters():
        p.data.zero_()
    
    # ---- TOKEN EMBEDDING ----
    # Each character gets a unique pattern in dims 0-25 (one-hot for letters)
    # Plus dims 26=space, 27=apostrophe, 28=underscore, 29=equals
    for char, idx in stoi.items():
        if char in 'abcdefghijklmnopqrstuvwxyz':
            model.token_emb.weight.data[idx, ord(char) - ord('a')] = 3.0
        elif char == ' ':
            model.token_emb.weight.data[idx, 26] = 3.0
        elif char == "'":
            model.token_emb.weight.data[idx, 27] = 3.0
        elif char == '_':
            model.token_emb.weight.data[idx, 28] = 3.0
        elif char == '=':
            model.token_emb.weight.data[idx, 29] = 3.0
    
    # ---- POSITION EMBEDDING ----
    # Use dims 30-37 to encode position information for first 8 positions
    # Position 0: high signal in dim 30
    # Position 1: high signal in dim 31
    # ... etc
    for pos in range(8):
        model.pos_emb.weight.data[pos, 30 + pos] = 3.0
    
    # Position 80 (= sign) gets a special marker in dim 38
    if model.max_seq_len > 80:
        model.pos_emb.weight.data[80, 38] = 3.0
    # Position 81 (answer) gets marker in dim 39
    if model.max_seq_len > 81:
        model.pos_emb.weight.data[81, 39] = 3.0
    
    # ---- LAYER 1: Character detection at specific positions ----
    blk0 = model.blocks[0]
    H = model.n_heads  # 4
    dh = d // H  # 16
    
    # Head 0: detect character at position 0 (first letter)
    # Q at last position queries for pos 0, K matches pos 0, V copies char
    # The query dimension 0 should attend to positions with high dim 30 (pos 0)
    blk0.attn.W_q.weight.data[0, 38] = 5.0    # Q looks for pos=80 marker
    blk0.attn.W_k.weight.data[0, 30] = 5.0     # K matches pos=0
    blk0.attn.W_v.weight.data[0, 0] = 5.0      # V copies char dim 0 (a)
    blk0.attn.W_v.weight.data[1, 1] = 5.0      # V copies char dim 1 (b)
    blk0.attn.W_v.weight.data[2, 2] = 5.0      # etc
    blk0.attn.W_v.weight.data[3, 3] = 5.0
    blk0.attn.W_v.weight.data[4, 4] = 5.0
    blk0.attn.W_v.weight.data[5, 5] = 5.0
    blk0.attn.W_v.weight.data[6, 6] = 5.0
    blk0.attn.W_v.weight.data[7, 7] = 5.0
    blk0.attn.W_v.weight.data[8, 8] = 5.0
    blk0.attn.W_v.weight.data[9, 9] = 5.0
    blk0.attn.W_v.weight.data[10, 10] = 5.0
    blk0.attn.W_v.weight.data[11, 11] = 5.0
    blk0.attn.W_v.weight.data[12, 12] = 5.0
    blk0.attn.W_v.weight.data[13, 13] = 5.0
    blk0.attn.W_v.weight.data[14, 14] = 5.0
    blk0.attn.W_v.weight.data[15, 15] = 5.0
    blk0.attn.W_o.weight.data[0, 0] = 5.0  # Pass through dims 0-15
    
    # Head 1: detect character at position 1
    blk0.attn.W_q.weight.data[dh, 38] = 5.0
    blk0.attn.W_k.weight.data[dh, 31] = 5.0
    for i in range(16):
        blk0.attn.W_v.weight.data[dh + i, i] = 5.0
    for i in range(16):
        blk0.attn.W_o.weight.data[16 + i, i] = 5.0
    
    # Head 2: detect character at position 2
    blk0.attn.W_q.weight.data[2*dh, 38] = 5.0
    blk0.attn.W_k.weight.data[2*dh, 32] = 5.0
    for i in range(16):
        blk0.attn.W_v.weight.data[2*dh + i, i] = 5.0
    for i in range(16):
        blk0.attn.W_o.weight.data[32 + i, i] = 5.0
    
    # Head 3: detect character at position 3
    blk0.attn.W_q.weight.data[3*dh, 38] = 5.0
    blk0.attn.W_k.weight.data[3*dh, 33] = 5.0
    for i in range(16):
        blk0.attn.W_v.weight.data[3*dh + i, i] = 5.0
    for i in range(16):
        blk0.attn.W_o.weight.data[48 + i, i] = 5.0
    
    # After layer 1 attention, position 80 has:
    # dims 0-15: character at position 0
    # dims 16-31: character at position 1
    # dims 32-47: character at position 2
    # dims 48-63: character at position 3
    
    # Layer 1 MLP: detect specific first words
    # "who" = w(22) at pos0, h(7) at pos1, o(14) at pos2
    # "where" = w(22) at pos0, h(7) at pos1, e(4) at pos2, r(17) at pos3
    # "when" = w(22) at pos0, h(7) at pos1, e(4) at pos2, n(13) at pos3
    # "how" = h(7) at pos0, o(14) at pos1, w(22) at pos2
    # "why" = w(22) at pos0, h(7) at pos1, y(24) at pos2
    # "what" = w(22) at pos0, h(7) at pos1, a(0) at pos2, t(19) at pos3
    # "which" = w(22) at pos0, h(7) at pos1, i(8) at pos2, c(2) at pos3
    
    # MLP hidden dim = 256
    # Use first 10 hidden neurons to detect words
    # Neuron 0: "who" detector -> class 3
    blk0.mlp.fc1.weight.data[0, 22] = 5.0   # w at pos0 (dim 22)
    blk0.mlp.fc1.weight.data[0, 16+7] = 5.0  # h at pos1 (dim 16+7=23)
    blk0.mlp.fc1.weight.data[0, 32+14] = 5.0  # o at pos2 (dim 32+14=46)
    blk0.mlp.fc1.bias.data[0] = -10.0  # threshold: need all 3
    
    # Neuron 1: "where" detector -> class 4
    blk0.mlp.fc1.weight.data[1, 22] = 5.0   # w at pos0
    blk0.mlp.fc1.weight.data[1, 16+7] = 5.0  # h at pos1
    blk0.mlp.fc1.weight.data[1, 32+4] = 5.0  # e at pos2
    blk0.mlp.fc1.weight.data[1, 48+17] = 5.0  # r at pos3
    blk0.mlp.fc1.bias.data[1] = -14.0
    
    # Neuron 2: "when" detector -> class 5
    blk0.mlp.fc1.weight.data[2, 22] = 5.0
    blk0.mlp.fc1.weight.data[2, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[2, 32+4] = 5.0
    blk0.mlp.fc1.weight.data[2, 48+13] = 5.0
    blk0.mlp.fc1.bias.data[2] = -14.0
    
    # Neuron 3: "how" detector -> class 5
    blk0.mlp.fc1.weight.data[3, 7] = 5.0    # h at pos0
    blk0.mlp.fc1.weight.data[3, 16+14] = 5.0  # o at pos1
    blk0.mlp.fc1.weight.data[3, 32+22] = 5.0  # w at pos2
    blk0.mlp.fc1.bias.data[3] = -10.0
    
    # Neuron 4: "why" detector -> class 2
    blk0.mlp.fc1.weight.data[4, 22] = 5.0
    blk0.mlp.fc1.weight.data[4, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[4, 32+24] = 5.0  # y at pos2
    blk0.mlp.fc1.bias.data[4] = -10.0
    
    # Neuron 5: "what" detector -> class 1
    blk0.mlp.fc1.weight.data[5, 22] = 5.0
    blk0.mlp.fc1.weight.data[5, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[5, 32+0] = 5.0  # a at pos2
    blk0.mlp.fc1.weight.data[5, 48+19] = 5.0  # t at pos3
    blk0.mlp.fc1.bias.data[5] = -14.0
    
    # Neuron 6: "which" detector -> class 1 (or 3)
    blk0.mlp.fc1.weight.data[6, 22] = 5.0
    blk0.mlp.fc1.weight.data[6, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[6, 32+8] = 5.0  # i at pos2
    blk0.mlp.fc1.weight.data[6, 48+2] = 5.0  # c at pos3
    blk0.mlp.fc1.bias.data[6] = -14.0
    
    # Neuron 7: "name" detector -> class 3
    blk0.mlp.fc1.weight.data[7, 13] = 5.0   # n at pos0
    blk0.mlp.fc1.weight.data[7, 16+0] = 5.0  # a at pos1
    blk0.mlp.fc1.weight.data[7, 32+12] = 5.0  # m at pos2
    blk0.mlp.fc1.weight.data[7, 48+4] = 5.0  # e at pos3
    blk0.mlp.fc1.bias.data[7] = -14.0
    
    # Neuron 8: "whom" detector -> class 3
    blk0.mlp.fc1.weight.data[8, 22] = 5.0
    blk0.mlp.fc1.weight.data[8, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[8, 32+14] = 5.0  # o at pos2
    blk0.mlp.fc1.weight.data[8, 48+12] = 5.0  # m at pos3
    blk0.mlp.fc1.bias.data[8] = -14.0
    
    # Neuron 9: "whose" detector -> class 3
    blk0.mlp.fc1.weight.data[9, 22] = 5.0
    blk0.mlp.fc1.weight.data[9, 16+7] = 5.0
    blk0.mlp.fc1.weight.data[9, 32+14] = 5.0  # o at pos2
    blk0.mlp.fc1.weight.data[9, 48+18] = 5.0  # s at pos3
    blk0.mlp.fc1.bias.data[9] = -14.0
    
    # Map word detections (hidden neurons 0-9) to class signals in d_model dims
    # fc2 maps from hidden (256) back to d_model (64)
    # Use dims 40-45 for class scores
    
    # Neuron 0 (who) -> class 3 score (dim 43)
    blk0.mlp.fc2.weight.data[43, 0] = 10.0
    # Neuron 1 (where) -> class 4 score (dim 44)
    blk0.mlp.fc2.weight.data[44, 1] = 10.0
    # Neuron 2 (when) -> class 5 score (dim 45)
    blk0.mlp.fc2.weight.data[45, 2] = 10.0
    # Neuron 3 (how) -> class 5 score (dim 45)
    blk0.mlp.fc2.weight.data[45, 3] = 10.0
    # Neuron 4 (why) -> class 2 score (dim 42)
    blk0.mlp.fc2.weight.data[42, 4] = 10.0
    # Neuron 5 (what) -> class 1 score (dim 41)
    blk0.mlp.fc2.weight.data[41, 5] = 10.0
    # Neuron 6 (which) -> class 1 score (dim 41)
    blk0.mlp.fc2.weight.data[41, 6] = 5.0
    # Neuron 7 (name) -> class 3 score (dim 43)
    blk0.mlp.fc2.weight.data[43, 7] = 10.0
    # Neuron 8 (whom) -> class 3 score (dim 43)
    blk0.mlp.fc2.weight.data[43, 8] = 10.0
    # Neuron 9 (whose) -> class 3 score (dim 43)
    blk0.mlp.fc2.weight.data[43, 9] = 10.0
    
    # ---- LAYER 2: propagate class scores to final position ----
    blk1 = model.blocks[1]
    
    # Single head: copy class score dims from pos 80 to pos 81
    blk1.attn.W_q.weight.data[0, 39] = 5.0   # Q: I'm at pos 81
    blk1.attn.W_k.weight.data[0, 38] = 5.0    # K: looking for pos 80
    for i in range(6):
        blk1.attn.W_v.weight.data[i, 40+i] = 5.0  # V: copy class scores
    for i in range(6):
        blk1.attn.W_o.weight.data[i, i] = 5.0  # O: put in dims 0-5
    
    # Layer 2 MLP: identity (pass through class scores)
    # Just add a small boost for the winning class
    for i in range(6):
        blk1.mlp.fc1.weight.data[i, i] = 3.0
        blk1.mlp.fc1.bias.data[i] = 0.0
        blk1.mlp.fc2.weight.data[i, i] = 3.0
    
    # ---- LAYER 3: final aggregation ----
    blk2 = model.blocks[2]
    # Copy class scores from pos 80 to pos 81
    blk2.attn.W_q.weight.data[0, 39] = 5.0
    blk2.attn.W_k.weight.data[0, 38] = 5.0
    for i in range(6):
        blk2.attn.W_v.weight.data[i, i] = 5.0
    for i in range(6):
        blk2.attn.W_o.weight.data[i, i] = 5.0
    
    # Layer 3 MLP: identity pass-through
    for i in range(6):
        blk2.mlp.fc1.weight.data[i, i] = 2.0
        blk2.mlp.fc2.weight.data[i, i] = 2.0
    
    # ---- OUTPUT HEAD ----
    # Map class scores (dims 0-5) to vocab logits for '0'-'5' (indices 30-35)
    for cls in range(6):
        model.head.weight.data[30 + cls, cls] = 10.0
    
    # Add bias towards class 1 for "what" (most common default)
    # We can do this via the final_ln learnable bias/gamma
    model.final_ln.weight.data.fill_(1.0)
    model.final_ln.bias.data.fill_(0.0)


model_shorthand_name = "BigramWordDetector-v2"
model_description = "Position-specific character encoding + bigram word detection in first 4 chars. Maps who/where/when/how/why/what/which/name/whom/whose to classes."


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
