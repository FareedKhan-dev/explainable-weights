"""
6-head column-specific addition. Each head handles one output column.
D=128, L=2, H=6, FF=256. Head dim = 128//6 ≈ 21.
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

N_HEADS = 6; N_LAYERS = 2
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
    def __init__(self, vocab_size, max_seq_len=32, d_model=None, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=None):
        D = n_heads * 20  # d_model = 120 so d_head=20
        if d_model is None: d_model = D
        if d_ff is None: d_ff = 256
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

# Positions: 0-4 first(4=ones), 5 +, 6-10 second(10=ones), 11 =, 12-17 output(17=ones)
# Col 0: first_pos=4, second_pos=10, out_pos=17
# Col 1: first_pos=3, second_pos=9, out_pos=16
# Col 2: first_pos=2, second_pos=8, out_pos=15
# Col 3: first_pos=1, second_pos=7, out_pos=14
# Col 4: first_pos=0, second_pos=6, out_pos=13
# Col 5 (overflow): out_pos=12

FIRST_POS = [4,3,2,1,0]
SECOND_POS = [10,9,8,7,6]
OUT_POS = [17,16,15,14,13,12]

def write_weights(model, task):
    D = model.d_model
    dh = D // N_HEADS  # 20
    with torch.no_grad():
        # Token emb: dims 0-9 = digit one-hot
        model.token_emb.weight.zero_()
        for d in range(10):
            model.token_emb.weight[d, d] = 10.0
        
        # Pos emb: dims 10-15 = column one-hot (shared across first/second/output)
        # dims 16 = is-first, 17 = is-second, 18 = is-output
        model.pos_emb.weight.zero_()
        for col in range(5):
            model.pos_emb.weight[FIRST_POS[col], 10+col] = 20.0
            model.pos_emb.weight[FIRST_POS[col], 16] = 20.0
            model.pos_emb.weight[SECOND_POS[col], 10+col] = 20.0
            model.pos_emb.weight[SECOND_POS[col], 17] = 20.0
            model.pos_emb.weight[OUT_POS[col], 10+col] = 20.0
            model.pos_emb.weight[OUT_POS[col], 18] = 20.0
        # col 5 (overflow)
        model.pos_emb.weight[OUT_POS[5], 15] = 20.0
        model.pos_emb.weight[OUT_POS[5], 18] = 20.0
        
        # Layer 0: 6 heads, each handles one output column
        blk = model.blocks[0]
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        blk.mlp.fc1.weight.zero_(); blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_(); blk.mlp.fc2.bias.zero_()
        
        for h in range(min(5, N_HEADS)):
            base = h * dh  # head h uses dims base..base+dh-1
            col = h
            # Q: output pos for this col queries col ID + output type
            blk.attn.W_q.weight[base, 10+col] = W
            blk.attn.W_q.weight[base+1, 18] = W
            # K: input pos for this col keys col ID + (first or second) type
            blk.attn.W_k.weight[base, 10+col] = W
            blk.attn.W_k.weight[base+1, 16] = W
            blk.attn.W_k.weight[base+1, 17] = W
            # V: copy digit (dim 0-9) into head's slice starting at base+2
            for d in range(10):
                blk.attn.W_v.weight[base+2+d, d] = 1.0
            # O: write attended digits into result dims 20-29 (d1) and 30-39 (d2)
            # With softmax over 2 input positions, we get average. Need to separate.
            # Use 2 Q dimensions to attend separately to first and second.
            # Head h dim base+10: Q targets first-number type
            blk.attn.W_q.weight[base+10, 16] = W
            blk.attn.W_q.weight[base+11, 17] = W
            blk.attn.W_k.weight[base+10, 16] = W  
            blk.attn.W_k.weight[base+11, 17] = W
        
        # Actually this is getting complex. Let me use a simpler approach:
        # 2 separate attention heads per column would need 10 heads.
        # Instead: use position-based K to target specific positions.
        
        # RESET and redo: simpler approach with 6 heads
        blk.attn.W_q.weight.zero_(); blk.attn.W_k.weight.zero_()
        blk.attn.W_v.weight.zero_(); blk.attn.W_o.weight.zero_()
        
        for h in range(min(5, N_HEADS)):
            base = h * dh
            col = h
            # Q: I am output pos for column h
            blk.attn.W_q.weight[base, 10+col] = W
            blk.attn.W_q.weight[base+1, 18] = W
            # K: I am first OR second input for column h
            blk.attn.W_k.weight[base, 10+col] = W
            blk.attn.W_k.weight[base+1, 16] = W
            blk.attn.W_k.weight[base+1, 17] = W
            # V: copy digit value
            for d in range(10):
                blk.attn.W_v.weight[base+2+d, d] = 1.0
            # O: write to shared dims: first 10 of head slice = d1, next 10 = d2
            # But attention averages, so both d1 and d2 get averaged together
            # Solution: separate the two inputs with two V projections
            # Use base+2..base+11 for "first digit", base+12..base+19 for "second digit"  
            # But that's 18 dims > dh=20, tight but ok if we only use 18
            # Actually the softmax will mix first and second number digits.
            # 
            # KEY INSIGHT: Since Q matches (col, output_type) and K matches (col, first_OR_second),
            # both first and second numbers in that column will be attended to equally.
            # The V copies digits. So the attended value = 0.5 * d1_onehot + 0.5 * d2_onehot.
            # That's not useful for the MLP lookup.
            #
            # FIX: Use TWO sub-queries per head. Or better, use 2 layers:
            # Layer 0: each output pos copies first-number digit from its column
            # Layer 1: each output pos copies second-number digit from its column
            # Then MLP computes sum.
            pass
        
        # ABORT complex multi-head. Go back to 1 head, 2 layers approach.
        pass
        
        # Layer 0: gather first-number digits into dim 20-29
        # All output positions attend to first-number positions matching their column
        # Q: output pos queries its column
        # K: first-number pos keys its column  
        for col in range(5):
            # output pos col -> query dim col
            blk.attn.W_q.weight[col, 10+col] = W
            # first-number pos col -> key dim col
            blk.attn.W_k.weight[col, 10+col] = W
            blk.attn.W_k.weight[col, 16] = W  # only match first number
        # Q also needs to match "I'm output type" to avoid self-attention among inputs
        blk.attn.W_q.weight[5, 18] = W
        
        # V: copy digit to dim 20-29
        for d in range(10):
            blk.attn.W_v.weight[20+d, d] = 1.0
        # O: pass through dims 20-29
        for d in range(10):
            blk.attn.W_o.weight[20+d, 20+d] = 1.0
        
        # Layer 1: gather second-number digits into dim 30-39
        blk1 = model.blocks[1]
        blk1.attn.W_q.weight.zero_(); blk1.attn.W_k.weight.zero_()
        blk1.attn.W_v.weight.zero_(); blk1.attn.W_o.weight.zero_()
        blk1.mlp.fc1.weight.zero_(); blk1.mlp.fc1.bias.zero_()
        blk1.mlp.fc2.weight.zero_(); blk1.mlp.fc2.bias.zero_()
        
        for col in range(5):
            blk1.attn.W_q.weight[col, 10+col] = W
            blk1.attn.W_k.weight[col, 10+col] = W
            blk1.attn.W_k.weight[col, 17] = W  # only match second number
        blk1.attn.W_q.weight[5, 18] = W
        
        for d in range(10):
            blk1.attn.W_v.weight[30+d, d] = 1.0
        for d in range(10):
            blk1.attn.W_o.weight[30+d, 30+d] = 1.0
        
        # MLP in layer 1: compute (d1+d2) % 10, stored in dims 0-9
        u = 0
        for d1 in range(10):
            for d2 in range(10):
                if u >= model.d_ff: break
                s = (d1 + d2) % 10
                blk1.mlp.fc1.weight[u, 20+d1] = 30.0
                blk1.mlp.fc1.weight[u, 30+d2] = 30.0
                blk1.mlp.fc1.bias[u] = -250.0
                blk1.mlp.fc2.weight[s, u] = 10.0
                u += 1
        
        # Head
        model.head.weight.zero_(); model.head.bias.zero_()
        for d in range(10):
            model.head.weight[d, d] = 50.0
        if model.vocab_size > 10:
            model.head.bias[10] = -500.0
        if model.vocab_size > 11:
            model.head.bias[11] = -500.0

model_shorthand_name = "two_layer_gather_v1"
model_description = "2 layers: L0 gathers d1 from first-num, L1 gathers d2 from second-num + MLP sum mod 10. H=6,D=120."

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
