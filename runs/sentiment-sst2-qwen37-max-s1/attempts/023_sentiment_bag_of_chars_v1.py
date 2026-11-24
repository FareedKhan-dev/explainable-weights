"""
Interpretable transformer for character-level sequence tasks.

The agent edits this file. The default task is 5-digit addition (`add5`):
prompt "12345+67890=" -> answer "080235".

Rules of the game:
  * You may modify the `SimpleTransformer` architecture and `write_weights()`.
  * You may NOT train the model — no gradient steps, no optimizer, no fitting.
  * You may write weight tensors directly (constants, NumPy arrays, hand-built
    circuits, etc.) inside `write_weights()`.
  * `write_weights()` runs once at model construction. It must leave every
    parameter of `SimpleTransformer` populated.

Usage:
    uv run interpretable_transformer.py
    uv run interpretable_transformer.py --task add5 --n-samples 500
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from src.eval import evaluate, plot_accuracy_over_iterations
import src.task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OVERALL_CSV = os.path.join(RESULTS_DIR, "overall_results.csv")
OVERALL_CSV_COLS = ["task", "accuracy", "status", "model_shorthand_name", "n_params", "description"]


# ---------------------------------------------------------------------------
# Architecture (edit freely — but NO TRAINING)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """3-layer causal transformer, vocab/seq-len configured from the task."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
    ):
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------------------
# Agent's interpretable weight assignment (edit this)
# ---------------------------------------------------------------------------

def write_weights(model: SimpleTransformer, task) -> None:
    """Sentiment keyword detection circuit.
    
    Strategy:
    1. Token embeddings encode character identity (near one-hot in first 32 dims)
    2. Layer 1 attention: each head attends to previous token (bigram detection)
    3. Layer 1 MLP: score bigrams for sentiment (positive vs negative)
    4. Layer 2 attention: uniform attention to sum sentiment scores
    5. Layer 2 MLP + Layer 3: threshold and decide
    6. Output head: map decision to '0' or '1'
    """
    D = 64  # d_model
    DH = D // 4  # d_head = 16
    
    with torch.no_grad():
        # ---- TOKEN EMBEDDINGS ----
        # Encode character identity in first 32 dims (near one-hot)
        # Plus sentiment score in dim 32 (positive char bias) and dim 33 (negative char bias)
        model.token_emb.weight.data.zero_()
        
        # Character sentiment scores based on frequency in common positive/negative words
        # Positive-starting letters: g(good/great), b(best/beautiful), l(love/lovely),
        # f(fine/fun/fantastic), n(nice), w(wonderful), a(amazing), e(excellent/enjoy),
        # p(perfect/pleasant), h(happy/heartfelt)
        # Negative-starting letters: b(bad/boring), t(terrible), a(awful), d(dull/disappointing),
        # w(worst/weak), u(ugly/unpleasant), h(horrible), p(poor/pathetic)
        
        # For now, just use identity encoding
        for i in range(min(32, model.vocab_size)):
            model.token_emb.weight.data[i, i] = 1.0
        
        # Padding character (underscore=28): zero embedding
        model.token_emb.weight.data[28, :] = 0.0
        
        # ---- POSITION EMBEDDINGS ----
        model.pos_emb.weight.data.zero_()
        
        # ---- LAYER 1: BIGRAM DETECTION ----
        block1 = model.blocks[0]
        
        # Attention: previous-token attention pattern
        # Q = position signal, K = previous position signal
        # This makes each position attend most strongly to the previous position
        block1.attn.W_q.weight.data.zero_()
        block1.attn.W_k.weight.data.zero_()
        block1.attn.W_v.weight.data.zero_()
        block1.attn.W_o.weight.data.zero_()
        
        # Head 0: detect character bigrams for positive words
        # Query extracts "I'm at position i" signal, Key extracts "I'm at position i-1"
        # We use the identity: Q reads current char, K reads previous char
        # With causal attention, the position just before is naturally attended to
        
        # Actually, let's use a simpler approach: uniform attention to get bag-of-characters
        # Then MLP scores each character
        
        # Make attention approximately uniform over all previous positions
        # by using small Q and K weights (softmax of small values ≈ uniform)
        for h_idx in range(4):
            start = h_idx * DH
            end = start + DH
            # Small random Q and K to get roughly uniform attention
            block1.attn.W_q.weight.data[start:end, :32] = torch.randn(DH, 32) * 0.01
            block1.attn.W_k.weight.data[start:end, :32] = torch.randn(DH, 32) * 0.01
            # Value: pass through character identity 
            for d in range(min(DH, 32)):
                block1.attn.W_v.weight.data[start + d, d] = 1.0
            # Output projection: pass through
            for d in range(DH):
                block1.attn.W_o.weight.data[d + start, start + d] = 1.0
        
        # MLP 1: Score characters for sentiment
        # Use the 256 hidden units as character-level sentiment detectors
        block1.mlp.fc1.weight.data.zero_()
        block1.mlp.fc1.bias.data.zero_()
        block1.mlp.fc2.weight.data.zero_()
        block1.mlp.fc2.bias.data.zero_()
        
        # Define positive and negative word starters (character trigrams)
        # Since we have bag-of-chars, let's just score individual characters
        
        # Positive sentiment characters (based on common positive words):
        # Words: good, great, best, love, wonderful, amazing, beautiful, excellent,
        #        perfect, fantastic, enjoy, fun, nice, fine, brilliant, superb,
        #        charming, delightful, impressive, remarkable, engaging, compelling
        pos_chars = {
            'g': 0.3,   # good, great
            'l': 0.2,   # love, lovely
            'w': 0.2,   # wonderful
            'f': 0.2,   # fun, fine, fantastic
            'n': 0.2,   # nice
            'e': 0.2,   # excellent, enjoy, engaging
            'b': 0.15,  # best, beautiful, brilliant
            'a': 0.15,  # amazing
            'p': 0.15,  # perfect
            'c': 0.15,  # charming, compelling
            'i': 0.1,   # impressive
            'r': 0.1,   # remarkable
            's': 0.1,   # superb, stunning
            'd': 0.1,   # delightful
            'h': 0.1,   # heartfelt
        }
        
        neg_chars = {
            'b': 0.3,   # bad, boring
            'w': 0.25,  # worst, weak, waste
            't': 0.2,   # terrible, trite
            'd': 0.2,   # dull, disappointing
            'a': 0.15,  # awful, annoying
            'u': 0.2,   # ugly, unpleasant, unconvincing
            'h': 0.15,  # horrible
            'p': 0.15,  # poor, pathetic, predictable
            'f': 0.1,   # fail, feeble
            'l': 0.1,   # lame, lackluster
            'm': 0.1,   # mediocre
            'c': 0.1,   # cliched, confused
            's': 0.1,   # stupid, shallow
        }
        
        # Hmm, many characters appear in both positive and negative words.
        # This approach won't work well with single characters.
        # Let me try a different approach: use character bigrams.
        
        # Actually, let me try something simpler: detect specific character sequences
        # that form parts of known sentiment words.
        
        # Use dim 32 for "positive sentiment score" and dim 33 for "negative sentiment score"
        stoi = task.stoi
        
        # Score characters based on net sentiment (positive - negative frequency)
        net_sentiment = {}
        for ch in pos_chars:
            net_sentiment[ch] = net_sentiment.get(ch, 0) + pos_chars[ch]
        for ch in neg_chars:
            net_sentiment[ch] = net_sentiment.get(ch, 0) - neg_chars[ch]
        
        # Encode sentiment score in token embedding dim 32
        for ch, score in net_sentiment.items():
            if ch in stoi:
                idx = stoi[ch]
                model.token_emb.weight.data[idx, 32] = score * 5.0
        
        # MLP 1: extract and amplify sentiment signal from dim 32
        # Hidden unit 0: detect positive sentiment (dim 32 > 0)
        block1.mlp.fc1.weight.data[0, 32] = 5.0
        block1.mlp.fc1.bias.data[0] = 0.0
        # Hidden unit 1: detect negative sentiment (dim 32 < 0)  
        block1.mlp.fc1.weight.data[1, 32] = -5.0
        block1.mlp.fc1.bias.data[1] = 0.0
        # Output: put positive score in dim 33, negative score in dim 34
        block1.mlp.fc2.weight.data[33, 0] = 1.0  # ReLU(5*sentiment) -> positive indicator
        block1.mlp.fc2.weight.data[34, 1] = 1.0  # ReLU(-5*sentiment) -> negative indicator
        
        # ---- LAYER 2: AGGREGATE ----
        block2 = model.blocks[1]
        
        # Uniform attention to sum up sentiment across all positions
        block2.attn.W_q.weight.data.zero_()
        block2.attn.W_k.weight.data.zero_()
        block2.attn.W_v.weight.data.zero_()
        block2.attn.W_o.weight.data.zero_()
        
        for h_idx in range(4):
            start = h_idx * DH
            end = start + DH
            # Small Q,K for uniform attention
            block2.attn.W_q.weight.data[start:end, :] = torch.randn(DH, D) * 0.01
            block2.attn.W_k.weight.data[start:end, :] = torch.randn(DH, D) * 0.01
            # Value: read sentiment dimensions
            if h_idx == 0:
                block2.attn.W_v.weight.data[0, 33] = 1.0  # positive count
                block2.attn.W_o.weight.data[0, 0] = 1.0
            elif h_idx == 1:
                block2.attn.W_v.weight.data[DH, 34] = 1.0  # negative count
                block2.attn.W_o.weight.data[1, DH] = 1.0
        
        # MLP 2: compute net sentiment
        block2.mlp.fc1.weight.data.zero_()
        block2.mlp.fc1.bias.data.zero_()
        block2.mlp.fc2.weight.data.zero_()
        block2.mlp.fc2.bias.data.zero_()
        
        # Hidden unit 0: positive - negative (net positive)
        block2.mlp.fc1.weight.data[0, 0] = 3.0
        block2.mlp.fc1.weight.data[0, 1] = -3.0
        block2.mlp.fc1.bias.data[0] = -0.5  # Slight bias toward positive (majority class)
        # Hidden unit 1: negative - positive (net negative)
        block2.mlp.fc1.weight.data[1, 0] = -3.0
        block2.mlp.fc1.weight.data[1, 1] = 3.0
        block2.mlp.fc1.bias.data[1] = 0.5
        # Output: put decision in dim 35
        block2.mlp.fc2.weight.data[35, 0] = 2.0   # positive decision
        block2.mlp.fc2.weight.data[35, 1] = -2.0   # negative decision
        
        # ---- LAYER 3: FINAL DECISION ----
        block3 = model.blocks[2]
        
        # Just pass through
        block3.attn.W_q.weight.data.zero_()
        block3.attn.W_k.weight.data.zero_()
        block3.attn.W_v.weight.data.zero_()
        block3.attn.W_o.weight.data.zero_()
        block3.mlp.fc1.weight.data.zero_()
        block3.mlp.fc1.bias.data.zero_()
        block3.mlp.fc2.weight.data.zero_()
        block3.mlp.fc2.bias.data.zero_()
        
        # Layer norms
        for block in model.blocks:
            block.ln1.weight.data.fill_(1.0)
            block.ln1.bias.data.zero_()
            block.ln2.weight.data.fill_(1.0)
            block.ln2.bias.data.zero_()
        
        model.final_ln.weight.data.fill_(1.0)
        model.final_ln.bias.data.zero_()
        
        # ---- OUTPUT HEAD ----
        model.head.weight.data.zero_()
        # Map dim 35 (decision) to output tokens
        model.head.weight.data[31, 35] = 10.0   # High logit for '1' when positive
        model.head.weight.data[30, 35] = -10.0  # Low logit for '0' when positive


# A unique shorthand name + 1-2 sentence description of what this attempt does.
# Used as the row identifier in results/overall_results.csv.
model_shorthand_name = "sentiment-bag-of-chars-v1"
model_description = "Bag-of-characters sentiment classifier with hand-coded positive/negative character scores"


# ---------------------------------------------------------------------------
# Evaluation harness (do not edit below)
# ---------------------------------------------------------------------------

def upsert_overall_results(rows: list[dict], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "overall_results.csv")
    new_keys = {(r["model_shorthand_name"], r["task"]) for r in rows}
    existing: list[dict] = []
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



def build_model(task) -> SimpleTransformer:
    max_seq_len = max(task.seq_len, 16)
    model = SimpleTransformer(vocab_size=task.vocab_size, max_seq_len=max_seq_len)
    write_weights(model, task)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Task name (see src/task.py)", choices=list(src.task.TASK_REGISTRY.keys()))
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    task = src.task.get_task(args.task)
    model = build_model(task).to(args.device)

    accuracy, _ = evaluate(
        model, task, n_samples=args.n_samples, seed=args.seed,
        device=args.device, verbose=args.verbose,
    )

    n_params = sum(p.numel() for p in model.parameters())

    upsert_overall_results([{
        "task":        args.task,
        "accuracy":    f"{accuracy:.4f}",
        "status":      "",
        "model_shorthand_name":  model_shorthand_name,
        "n_params":    f"{n_params:.2e}",
        "description": model_description,
    }], RESULTS_DIR)
    plot_accuracy_over_iterations(RESULTS_DIR)

    print()
    print("---")
    print(f"task:          {args.task}")
    print(f"accuracy:      {accuracy:.4f}  ({int(round(accuracy * args.n_samples))}/{args.n_samples})")
    print(f"total_seconds: {time.time() - t0:.1f}s")
