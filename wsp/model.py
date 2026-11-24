"""wsp/model.py: the paper's starting network, a small causal transformer whose every weight can be set by hand.

Three blocks of attention plus a two-layer MLP, each with a residual connection and a LayerNorm in front, then a
linear head over the task's characters. This is the template every agent run starts from. The agent may change the
shapes and the wiring, but never trains a weight."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Four square weight tables, no bias
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape

        # Queries, keys and values, split into heads
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # How well each query matches each key
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # A token only sees itself and the past
        ones = torch.ones(T, T, dtype=torch.bool, device=x.device)
        mask = torch.triu(ones, diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)

        # Mix the values and join the heads
        mixed = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.W_o(mixed)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Read from the other positions
        x = x + self.attn(self.ln1(x))

        # Compute on what was read
        hidden = F.relu(self.fc1(self.ln2(x)))
        return x + self.fc2(hidden)


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4,
                 n_layers=3, d_ff=256):
        super().__init__()

        # The two embedding tables
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.blocks = nn.ModuleList(blocks)

        # A final LayerNorm and the output head
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        # Character vector plus position vector
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(positions)[None]

        # Run the stream through every block
        for block in self.blocks:
            h = block(h)

        # One score per character, at every position
        return self.head(self.final_ln(h))
